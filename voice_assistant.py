"""
Hands-free voice conversation with a self-hosted LLM.

    microphone -> Whisper (voicekit.speech_recognizer)
               -> LLM     (voicekit.llm_client)
               -> Kokoro  (voicekit.text_to_speech) -> speakers

Usage
-----
    python voice_assistant.py
    python voice_assistant.py --model qwen3:14b --voice am_michael
    python voice_assistant.py --interruptible        # with headphones
    python voice_assistant.py --api openai --host http://10.0.0.100:8000
    python voice_assistant.py --list-devices

Voice commands: "goodbye" quits, "new conversation" clears the history.
"""

import argparse
import logging
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

from voicekit import (
    ContinuousSpeechRecognizer,
    KokoroSpeaker,
    LLMClient,
    LLMConfig,
    LLMError,
    SpeechRecognizerConfig,
    TextToSpeechConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """
    Configuration for VoiceAssistant.

    interruptible
        Keep the microphone on while the assistant speaks, so you can
        talk over it to cut it off. Needs headphones (or a microphone
        with echo cancellation); otherwise the assistant hears itself and
        stops mid-sentence. When False the microphone is muted while the
        assistant is thinking and speaking.

    resume_delay_ms
        After speaking, wait this long before listening again so the end
        of the reply (and room echo) is not picked up.

    greeting
        Spoken at start-up. None for silence.

    exit_phrases, reset_phrases
        Spoken commands; a prompt must match one exactly (ignoring case
        and punctuation).
    """

    interruptible: bool = False
    resume_delay_ms: int = 300
    greeting: Optional[str] = "Hi, I'm listening."
    goodbye: str = "Goodbye!"
    exit_phrases: tuple = (
        "goodbye", "good bye", "bye bye", "exit", "quit", "stop listening",
    )
    reset_phrases: tuple = (
        "new conversation", "reset conversation", "start over",
        "forget everything", "clear history",
    )
    reset_reply: str = "Okay, starting a new conversation."
    error_reply: str = "Sorry, I couldn't reach the language model."


def _normalize_command(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def print_event(kind, text=""):
    """Default event handler: a simple console transcript."""

    if kind == "status":
        print(f"\n[{text}]", flush=True)
    elif kind == "user":
        print(f"\nYOU: {text}", flush=True)
    elif kind == "assistant_start":
        print("AI:  ", end="", flush=True)
    elif kind == "assistant_text":
        print(text, end="", flush=True)
    elif kind == "assistant_end":
        print(flush=True)
    elif kind == "interrupted":
        print(" [interrupted]", flush=True)
    elif kind == "error":
        print(f"\n[error] {text}", flush=True)


class VoiceAssistant:
    """
    Runs the listen -> think -> speak loop.

    The three components are passed in, so each can be configured,
    replaced or tested on its own.

    on_event(kind, text) receives: "status", "user", "assistant_start",
    "assistant_text" (streamed), "assistant_end", "interrupted", "error".
    """

    def __init__(
        self,
        recognizer: ContinuousSpeechRecognizer,
        speaker: KokoroSpeaker,
        llm: LLMClient,
        config: Optional[AssistantConfig] = None,
        on_event: Callable[[str, str], None] = print_event,
    ):
        self.recognizer = recognizer
        self.speaker = speaker
        self.llm = llm
        self.config = config or AssistantConfig()
        self.on_event = on_event

        self._exit_phrases = {_normalize_command(p) for p in self.config.exit_phrases}
        self._reset_phrases = {_normalize_command(p) for p in self.config.reset_phrases}

        # True from the moment a prompt is sent until the reply has been
        # spoken, so the user can interrupt while the LLM is still thinking.
        self._responding = False

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):
        """Check the LLM server and load the speech models."""

        self._emit("status", f"Connecting to {self.llm.config.base_url}")
        self.llm.check_connection()

        # Load the LLM into GPU memory on the server while we load ours.
        threading.Thread(target=self.llm.warm_up, daemon=True).start()

        self._emit("status", "Loading speech models")
        self.speaker.load()
        self.recognizer.load()

        if self.config.interruptible:
            self.recognizer.on_speech_start = self._on_user_speech

    def run(self):
        """Converse until a goodbye phrase or Ctrl+C."""

        self.start()

        try:
            if self.config.greeting:
                self._say(self.config.greeting)

            self._emit("status", "Listening - speak whenever you're ready")

            for prompt in self.recognizer.listen():
                if not self.handle(prompt["text"]):
                    break

        finally:
            self.recognizer.on_speech_start = None
            self.speaker.stop()

    def close(self):
        self.speaker.close()
        self.recognizer.close()
        self.llm.close()

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    def handle(self, text):
        """Respond to one transcribed prompt. Returns False to quit."""

        self._emit("user", text)
        command = _normalize_command(text)

        if command in self._exit_phrases:
            self._say(self.config.goodbye)
            return False

        if command in self._reset_phrases:
            self.llm.reset()
            self._say(self.config.reset_reply)
            return True

        self._respond(text)
        return True

    def _respond(self, text):

        self._mute()
        self._responding = True

        try:
            self._emit("assistant_start")

            completed = self.speaker.speak_stream(
                self._echo(self.llm.stream_reply(text))
            )

            self._emit("assistant_end")

            if not completed:
                self._emit("interrupted")

        except LLMError as exc:
            self._emit("assistant_end")
            self._emit("error", str(exc))
            logger.debug("LLM error", exc_info=True)
            self.speaker.speak(self.config.error_reply, wait=True)

        finally:
            self._responding = False
            self._unmute()

    def _say(self, text):
        """Speak a fixed phrase (greeting, confirmations)."""

        self._mute()

        try:
            self._emit("assistant_start")
            self._emit("assistant_text", text)
            self._emit("assistant_end")
            self.speaker.speak(text, wait=True)
        finally:
            self._unmute()

    def _echo(self, tokens):
        """Pass tokens through, showing them as they arrive."""

        try:
            for token in tokens:
                self._emit("assistant_text", token)
                yield token
        finally:
            tokens.close()

    # --------------------------------------------------
    # Microphone handling
    # --------------------------------------------------

    def _mute(self):
        if not self.config.interruptible:
            self.recognizer.pause()

    def _unmute(self):
        if not self.config.interruptible:
            self.speaker.wait()
            time.sleep(self.config.resume_delay_ms / 1000)
            self.recognizer.resume()

    def _on_user_speech(self):
        # Runs on the recognizer's capture thread. stop() also makes
        # speak_stream() abandon the LLM stream at its next token.
        if self._responding or self.speaker.is_speaking:
            self.speaker.stop()

    def _emit(self, kind, text=""):
        if self.on_event is None:
            return
        try:
            self.on_event(kind, text)
        except Exception:
            logger.exception("on_event handler failed")


# --------------------------------------------------
# Command line
# --------------------------------------------------

def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Talk to a self-hosted LLM with your voice.",
    )

    llm = parser.add_argument_group("LLM")
    llm.add_argument("--host", default=LLMConfig.base_url,
                     help="LLM server URL (default: %(default)s)")
    llm.add_argument("--api", choices=("ollama", "openai"), default="ollama",
                     help="server protocol (default: %(default)s)")
    llm.add_argument("--model", default=LLMConfig.model,
                     help="model name (default: %(default)s)")
    llm.add_argument("--api-key", default=None,
                     help="bearer token, if the server needs one")
    llm.add_argument("--think", action="store_true",
                     help="let reasoning models think first (slower)")
    llm.add_argument("--system-prompt-file", default=None,
                     help="file with a custom system prompt")

    speech = parser.add_argument_group("speech")
    speech.add_argument("--mic", default=None,
                        help="microphone ID or part of its name")
    speech.add_argument("--whisper-model", default="turbo",
                        help="Whisper model (default: %(default)s)")
    speech.add_argument("--language", default="en",
                        help="spoken language code (default: %(default)s)")
    speech.add_argument("--vocabulary", default=None,
                        help="names/jargon to help recognition, "
                             "e.g. 'Kubernetes, Ollama, Qwen'")
    speech.add_argument("--pause", type=float, default=1.5,
                        help="seconds of silence that end your turn "
                             "(default: %(default)s)")

    voice = parser.add_argument_group("voice")
    voice.add_argument("--voice", default="af_heart",
                       help="Kokoro voice (default: %(default)s)")
    voice.add_argument("--speed", type=float, default=1.0,
                       help="speaking rate (default: %(default)s)")
    voice.add_argument("--output", default=None,
                       help="speaker ID or part of its name")

    parser.add_argument("--interruptible", action="store_true",
                        help="talk over the assistant to interrupt it "
                             "(use headphones)")
    parser.add_argument("--list-devices", action="store_true",
                        help="list microphones and speakers, then exit")
    parser.add_argument("--debug", action="store_true",
                        help="verbose logging")

    return parser.parse_args(argv)


def _device(value):
    return int(value) if value is not None and value.isdigit() else value


def main(argv=None):

    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.debug:
        warnings.filterwarnings("ignore")

    if args.list_devices:
        ContinuousSpeechRecognizer.print_microphones()
        print("\nSpeakers:\n")
        for device in KokoroSpeaker.list_output_devices():
            print(f'{device["id"]:>3}  {device["name"]}')
        return 0

    system_prompt = LLMConfig.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()

    llm = LLMClient(LLMConfig(
        base_url=args.host,
        api=args.api,
        model=args.model,
        api_key=args.api_key,
        think=args.think,
        system_prompt=system_prompt,
    ))

    recognizer = ContinuousSpeechRecognizer(SpeechRecognizerConfig(
        mic_device=_device(args.mic),
        model_name=args.whisper_model,
        language=args.language,
        hotwords=args.vocabulary,
        initial_prompt=args.vocabulary,
        prompt_end_ms=int(args.pause * 1000),
    ))

    speaker = KokoroSpeaker(TextToSpeechConfig(
        voice=args.voice,
        lang_code=args.voice[0],
        speed=args.speed,
        output_device=_device(args.output),
    ))

    assistant = VoiceAssistant(
        recognizer,
        speaker,
        llm,
        AssistantConfig(interruptible=args.interruptible),
    )

    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n[stopped]")
    except LLMError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1
    finally:
        assistant.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
