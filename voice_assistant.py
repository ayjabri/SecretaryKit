"""
Hands-free voice conversation with a self-hosted LLM.

    microphone -> Whisper (voicekit.speech_recognizer)
               -> LLM     (voicekit.llm_client)
               -> Kokoro  (voicekit.text_to_speech) -> speakers

Usage
-----
    python voice_assistant.py
    python voice_assistant.py --model qwen3.6:27b --voice am_michael
    python voice_assistant.py --interruptible        # with headphones
    python voice_assistant.py --api openai --host http://10.0.0.100:8000
    python voice_assistant.py --list-devices

Voice commands (the whole utterance must be the command):
    "set think" / "thinking on"         reasoning models think before replying
    "set no think" / "thinking off"     answer straight away (default)
    "switch model to Qwen 3.5"          change model; history is kept
    "list models"                       models available on the server
    "status"                            current model, thinking and speed
    "speak faster" / "speak slower"     change the speaking rate
    "new conversation"                  clear the history
    "help"                              list the commands
    "goodbye"                           quit
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
from voicekit.commands import NOT_HANDLED, CommandRouter


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

    command_prefix
        Optional wake word for commands, e.g. "computer": then only
        "Computer, set think" is a command, never a plain "set think".

    thinking_cue
        Spoken when a question is sent in thinking mode, since the model
        may then be silent for a while. None to disable.

    show_thinking
        Print the model's reasoning (dimmed) while it thinks.

    speed_step
        How much "speak faster" / "speak slower" changes the rate.

    model_names_as_vocabulary
        Teach Whisper the server's model names ("Qwen", "Llama"...) so
        "switch model to ..." is transcribed correctly. Only applies when
        the recognizer has no hotwords configured.
    """

    interruptible: bool = False
    resume_delay_ms: int = 300
    greeting: Optional[str] = "Hi, I'm listening."
    goodbye: str = "Goodbye!"
    exit_phrases: tuple = (
        "dismissed", "goodbye", "good bye", "bye bye", "exit", "quit", "stop listening",
    )
    reset_phrases: tuple = (
        "new conversation", "reset conversation", "start over",
        "forget everything", "clear history",
    )
    reset_reply: str = "Okay, starting a new conversation."
    error_reply: str = "Sorry, I couldn't reach the language model."
    command_prefix: Optional[str] = None
    thinking_cue: Optional[str] = "Let me think about that."
    show_thinking: bool = True
    speed_step: float = 0.1
    model_names_as_vocabulary: bool = True


# Spoken command patterns. They are matched against the whole utterance
# after normalisation (lower case, punctuation replaced by spaces), so
# "Set no-think." arrives as "set no think".
_THINK = r"think(?:ing)?(?: mode)?"

THINK_ON_PATTERNS = (
    rf"(?:set |turn on |enable |start |use )?(?:the )?{_THINK}",
    rf"(?:set |turn |switch )?{_THINK} on",
)
THINK_OFF_PATTERNS = (
    rf"(?:set )?no ?{_THINK}",
    rf"(?:turn off|switch off|disable|stop)(?: the)? {_THINK}",
    rf"(?:set |turn |switch )?{_THINK} off",
    r"don t think",
)
MODEL_PATTERNS = (
    r"(?:switch|change|set)(?: the)? model to (?P<model>.+)",
    r"(?:set|use|load)(?: the)? model (?P<model>.+)",
)
# Looser forms; they fall through to the LLM unless the name is a model.
MODEL_LOOSE_PATTERNS = (
    r"(?:switch to|use|load) (?P<model>.+)",
)
LIST_MODELS_PATTERNS = (
    r"(?:list|show)(?: the)?(?: available)? models",
    r"(?:what|which) models (?:are available|are there|do you have|can i use)",
)
STATUS_PATTERNS = (
    r"status",
    r"what s your status",
    r"(?:current )?settings",
    r"(?:what|which) model are you(?: using)?",
    r"what mode are you in",
)
FASTER_PATTERNS = (r"(?:speak|talk)(?: a bit| a little)? faster",)
SLOWER_PATTERNS = (r"(?:speak|talk)(?: a bit| a little)? slower",)
NORMAL_SPEED_PATTERNS = (
    r"(?:speak |talk )?(?:at )?normal speed",
    r"reset (?:the )?speed",
)
HELP_PATTERNS = (
    r"help",
    r"what (?:commands|can i say)(?: are there)?",
    r"(?:list|voice) commands",
)


def spoken_model_name(model):
    """'qwen3.6:27b' -> 'Qwen 3.6 27B', for reading aloud."""

    name = re.sub(r":latest$", "", model)
    family, _, size = name.partition(":")
    family = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", family).replace("-", " ")
    family = family[:1].upper() + family[1:]

    return f"{family} {size.upper()}".strip()


def _join_spoken(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_DIM = "\033[2m"
_RESET = "\033[0m"


def print_event(kind, text=""):
    """Default event handler: a simple console transcript."""

    if kind == "status":
        print(f"\n[{text}]", flush=True)
    elif kind == "user":
        print(f"\nYOU: {text}", flush=True)
    elif kind == "thinking_start":
        print(f"{_DIM}(thinking) ", end="", flush=True)
    elif kind == "thinking":
        print(text, end="", flush=True)
    elif kind == "thinking_end":
        print(_RESET, flush=True)
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

    on_event(kind, text) receives: "status", "user", "thinking_start",
    "thinking" (streamed reasoning), "thinking_end", "assistant_start",
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

        self.commands = self._build_commands()
        self._quit = False

        # True from the moment a prompt is sent until the reply has been
        # spoken, so the user can interrupt while the LLM is still thinking.
        self._responding = False
        self._thinking_shown = False
        self._reply_started = False

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):
        """Check the LLM server and load the speech models."""

        self._emit("status", f"Connecting to {self.llm.config.base_url}")
        self.llm.check_connection()
        self._teach_model_names()

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

        reply = self.commands.dispatch(text)

        if reply is NOT_HANDLED:
            self._respond(text)
        elif reply:
            self._say(reply)

        return not self._quit

    def _respond(self, text):

        self._mute()
        self._responding = True
        self._thinking_shown = False
        self._reply_started = False

        try:
            if self.llm.config.think and self.config.thinking_cue:
                # Plays while the model thinks; the reply queues after it.
                self.speaker.speak(self.config.thinking_cue)

            tokens = self.llm.stream_reply(text, on_thinking=self._on_thinking)
            completed = self.speaker.speak_stream(self._echo(tokens))

            self._end_reply()

            if not completed:
                self._emit("interrupted")

        except LLMError as exc:
            self._end_reply()
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
                self._start_reply()
                self._emit("assistant_text", token)
                yield token
        finally:
            tokens.close()

    def _on_thinking(self, text):

        if not self._thinking_shown:
            self._thinking_shown = True
            self._emit("thinking_start")

        if self.config.show_thinking:
            self._emit("thinking", text)

    def _start_reply(self):

        if self._reply_started:
            return

        if self._thinking_shown:
            self._emit("thinking_end")

        self._reply_started = True
        self._emit("assistant_start")

    def _end_reply(self):

        if self._reply_started:
            self._emit("assistant_end")
        elif self._thinking_shown:
            self._emit("thinking_end")

    # --------------------------------------------------
    # Voice commands
    # --------------------------------------------------

    def _build_commands(self):

        router = CommandRouter(prefix=self.config.command_prefix)

        router.add_phrases("exit", self.config.exit_phrases, self._cmd_exit,
                           "goodbye")
        router.add_phrases("reset", self.config.reset_phrases, self._cmd_reset,
                           "new conversation")
        router.add("think_off", THINK_OFF_PATTERNS,
                   lambda: self._cmd_think(False), "thinking off")
        router.add("think_on", THINK_ON_PATTERNS,
                   lambda: self._cmd_think(True), "thinking on")
        router.add("list_models", LIST_MODELS_PATTERNS, self._cmd_list_models,
                   "list models")
        router.add("switch_model", MODEL_PATTERNS,
                   lambda model: self._cmd_switch_model(model, strict=True),
                   "switch model to, followed by a name")
        router.add("switch_model_loose", MODEL_LOOSE_PATTERNS,
                   lambda model: self._cmd_switch_model(model, strict=False))
        router.add("status", STATUS_PATTERNS, self._cmd_status, "status")
        router.add("faster", FASTER_PATTERNS, lambda: self._cmd_speed(+1),
                   "speak faster")
        router.add("slower", SLOWER_PATTERNS, lambda: self._cmd_speed(-1),
                   "speak slower")
        router.add("normal_speed", NORMAL_SPEED_PATTERNS,
                   lambda: self._cmd_speed(0), "normal speed")
        router.add("help", HELP_PATTERNS, self._cmd_help, "help")

        return router

    def _cmd_exit(self):
        self._quit = True
        return self.config.goodbye

    def _cmd_reset(self):
        self.llm.reset()
        return self.config.reset_reply

    def _cmd_think(self, on):

        model = spoken_model_name(self.llm.config.model)

        if on == self.llm.config.think:
            return f"Thinking is already {'on' if on else 'off'}."

        if on:
            try:
                supported = self.llm.supports_thinking()
            except LLMError:
                supported = None  # unknown; let the server decide

            if supported is False:
                return f"{model} doesn't support thinking."

        self.llm.config.think = on
        self._emit("status", f"thinking {'on' if on else 'off'}")

        if on:
            return "Thinking on. Replies will take longer."
        return "Thinking off."

    def _cmd_switch_model(self, model, strict):

        try:
            models = self.llm.list_models()
        except LLMError:
            return self.config.error_reply

        resolved = self.llm.resolve_model(model, models)

        if resolved is None:
            if not strict:
                return NOT_HANDLED
            return (
                f"I couldn't find a model called {model}. Available models "
                f"are {_join_spoken(spoken_model_name(m) for m in models)}."
            )

        name = spoken_model_name(resolved)

        if resolved == self.llm.config.model:
            return f"I'm already using {name}."

        self.llm.config.model = resolved
        self._emit("status", f"model: {resolved}")
        threading.Thread(target=self.llm.warm_up, daemon=True).start()

        reply = f"Switched to {name}."

        if self.llm.config.think:
            try:
                supported = self.llm.supports_thinking()
            except LLMError:
                supported = None

            if supported is False:
                self.llm.config.think = False
                reply += " It doesn't support thinking, so thinking is off."

        return reply

    def _cmd_list_models(self):

        try:
            models = self.llm.list_models()
        except LLMError:
            return self.config.error_reply

        names = _join_spoken(spoken_model_name(m) for m in models)
        current = spoken_model_name(self.llm.config.model)

        return f"Available models are {names}. I'm using {current}."

    def _cmd_status(self):

        return (
            f"I'm using {spoken_model_name(self.llm.config.model)}, "
            f"thinking is {'on' if self.llm.config.think else 'off'}, "
            f"and my speaking speed is {self.speaker.config.speed:.1f}."
        )

    def _cmd_speed(self, direction):

        config = self.speaker.config

        if direction == 0:
            config.speed = 1.0
            return "Back to normal speed."

        speed = round(config.speed + direction * self.config.speed_step, 2)
        speed = min(2.0, max(0.5, speed))

        if speed == config.speed:
            return f"That's as {'fast' if direction > 0 else 'slow'} as I go."

        config.speed = speed
        return f"Okay, speaking {'faster' if direction > 0 else 'slower'}."

    def _cmd_help(self):

        prefix = (
            f"Start commands with {self.config.command_prefix}. "
            if self.config.command_prefix else ""
        )
        return (
            f"{prefix}You can say: "
            f"{_join_spoken(self.commands.descriptions())}."
        )

    def _teach_model_names(self):
        """Help Whisper spell model names, e.g. 'Qwen' rather than 'Quinn'."""

        recognizer_config = getattr(self.recognizer, "config", None)

        if (
            not self.config.model_names_as_vocabulary
            or recognizer_config is None
            or recognizer_config.hotwords
        ):
            return

        try:
            models = self.llm.list_models()
        except LLMError:
            return

        families = sorted({
            spoken_model_name(m).split(" ")[0] for m in models
        })

        if families:
            recognizer_config.hotwords = ", ".join(families)

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
    parser.add_argument("--command-prefix", default=None,
                        help="wake word required before voice commands, "
                             "e.g. 'computer'")
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
        AssistantConfig(
            interruptible=args.interruptible,
            command_prefix=args.command_prefix,
        ),
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
