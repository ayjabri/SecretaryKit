"""
The voice loop: listen -> agent -> speak, for any agent.

    from voicekit import VoiceSession
    from voicekit.agents import OllamaAgent

    VoiceSession(OllamaAgent()).run()

The session knows nothing about models or tools. It:

- turns your speech into text and hands it to the agent,
- speaks the agent's reply while it streams, plus any Status updates
  (progress reports from long-running or multi-agent work),
- shows reasoning and tool activity on screen,
- lets the agent talk back mid-task: ctx.say(), ctx.ask(), ctx.confirm(),
- handles muting, barge-in (interruptible=True) and a short spoken cue
  when the agent is slow to start answering,
- offers voice-control tools (speak faster, new conversation, end the
  session) to agents that let their model call tools,
- runs exact voice commands ("goodbye", "speak faster", "status").
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .agent import (
    Agent,
    AgentError,
    Status,
    TextDelta,
    Thinking,
    ToolFinished,
    ToolStarted,
)
from .commands import NOT_HANDLED, CommandRouter, normalize_command
from .tools import classify_confirmation, tool


logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """
    interruptible
        Let you cut the assistant off while it speaks. When False the
        microphone is muted while the agent works and speaks.

    barge_in
        How an interruptible session is interrupted:
          "phrase"  only by a stop phrase ("hang on", "stop there").
                    Works with speakers: the assistant's own voice and
                    background talk are ignored, and nothing said while
                    it speaks becomes a prompt. Takes about a second.
          "speech"  by any speech, instantly, and what you say becomes
                    the next prompt. Needs headphones, otherwise the
                    microphone hears the assistant and it stops itself.

    echo_cancellation
        Remove the assistant's own voice from the microphone (WebRTC
        AEC3, needs `pip install "voicekit[aec]"`). None: on for
        interruptible sessions when installed. True: required. False: off.
        Only matters when the microphone is on while the assistant
        speaks, i.e. interruptible sessions.

    stop_phrases
        Phrases that stop the assistant. Matched anywhere in what was
        heard, except phrases the assistant itself just said. On their
        own they are also exact commands that do nothing, so "hang on"
        is never sent to the agent.

    resume_delay_ms
        After speaking, wait this long before listening again so the end
        of the reply (and room echo) is not picked up.

    greeting, goodbye
        Spoken at start and when the session ends. greeting=None for
        silence.

    exit_phrases, reset_phrases
        Exact voice commands (whole utterance, ignoring case/punctuation).

    command_prefix
        Optional wake word required before exact commands ("computer").

    wait_cue, wait_cue_after_s
        Spoken once if the agent hasn't said anything after this many
        seconds, so a slow model or orchestrator doesn't sound dead.

    show_thinking
        Print the model's reasoning (dimmed) while it thinks.

    speak_status
        Speak Status updates from the agent, not just print them.

    control_tools
        Offer the session's own tools (speaking speed, new conversation,
        status, end session) to agents whose model can call tools.

    speed_step
        How much "speak faster" / "speak slower" changes the rate.
    """

    interruptible: bool = False
    barge_in: Literal["phrase", "speech"] = "phrase"
    echo_cancellation: Optional[bool] = None
    stop_phrases: tuple = (
        "stop", "stop there", "stop it", "stop talking", "okay stop",
        "hang on", "hold on", "wait", "wait a second", "wait a minute",
        "one second", "pause", "be quiet", "quiet", "shut up", "enough",
        "that's enough", "never mind",
    )
    resume_delay_ms: int = 300
    greeting: Optional[str] = "Hi, I'm listening."
    goodbye: str = "Goodbye!"
    exit_phrases: tuple = (
        "dismissed", "goodbye", "good bye", "bye bye", "exit", "quit",
        "stop listening",
    )
    reset_phrases: tuple = (
        "new conversation", "reset conversation", "start over",
        "forget everything", "clear history",
    )
    reset_reply: str = "Starting a new conversation."
    error_reply: str = "Sorry, something went wrong."
    command_prefix: Optional[str] = None
    wait_cue: Optional[str] = "Let me think about that."
    wait_cue_after_s: float = 2.5
    show_thinking: bool = True
    speak_status: bool = True
    control_tools: bool = True
    speed_step: float = 0.1


_DIM = "\033[2m"
_RESET = "\033[0m"


def print_event(kind, text=""):
    """Default event handler: a console transcript."""

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
    elif kind == "tool":
        print(f"\n{_DIM}[tool] {text}{_RESET}", flush=True)
    elif kind == "tool_result":
        print(f"{_DIM}[tool result] {text}{_RESET}", flush=True)
    elif kind == "progress":
        print(f"\n[progress] {text}", flush=True)
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


class VoiceSession:
    """
    Runs a spoken conversation with an Agent. Also serves as the agent's
    TurnContext (say / ask / confirm / emit / cancelled).

    recognizer, speaker
        Default to ContinuousSpeechRecognizer() and KokoroSpeaker();
        pass configured instances to choose devices, voices and models.

    on_event(kind, text)
        Receives "status", "user", "thinking_start", "thinking",
        "thinking_end", "tool", "tool_result", "progress",
        "assistant_start", "assistant_text", "assistant_end",
        "interrupted", "error". Default: print_event (console).
    """

    def __init__(
        self,
        agent: Agent,
        recognizer=None,
        speaker=None,
        config: Optional[SessionConfig] = None,
        on_event: Optional[Callable[[str, str], None]] = print_event,
    ):
        if recognizer is None:
            from .speech_recognizer import ContinuousSpeechRecognizer
            recognizer = ContinuousSpeechRecognizer()

        if speaker is None:
            from .text_to_speech import KokoroSpeaker
            speaker = KokoroSpeaker()

        self.agent = agent
        self.recognizer = recognizer
        self.speaker = speaker
        self.config = config or SessionConfig()
        self.on_event = on_event

        self.commands = CommandRouter(prefix=self.config.command_prefix)
        self._register_commands()
        agent.register_commands(self.commands)

        if self.config.control_tools and getattr(agent, "tools", None) is not None:
            agent.tools.add(*control_tools())

        if self.config.barge_in not in ("phrase", "speech"):
            raise ValueError('barge_in must be "phrase" or "speech"')

        self._stop_patterns = [
            re.compile(rf"\b{re.escape(normalize_command(p))}\b")
            for p in self.config.stop_phrases
        ]
        # Speech mode keeps the microphone live; phrase mode mutes it as
        # usual and only watches for stop phrases.
        self._mic_live = self.config.interruptible and self.config.barge_in == "speech"
        self._said = ""

        self._prompts = None
        self._quit = False
        self._responding = False
        self._asking = False
        self._cancelled = False
        self._thinking_shown = False
        self._reply_started = False
        self._spoken = False
        self._cue_timer = None
        self._echo_canceller = None

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def start(self):
        """Start the agent and load the speech models."""

        self._emit("status", "Starting the agent")
        self.agent.start()
        self._teach_vocabulary()

        self._emit("status", "Loading speech models")
        self.speaker.load()
        self.recognizer.load()
        self._setup_echo_cancellation()

        if not self.config.interruptible:
            return

        if self._mic_live:
            self.recognizer.on_speech_start = self._on_user_speech
        elif hasattr(self.recognizer, "watch"):
            self.recognizer.watch(self._on_overheard)
        else:
            logger.warning("This recognizer can't listen for stop phrases.")

    def run(self):
        """Converse until a goodbye (spoken or via a tool) or Ctrl+C."""

        self.start()
        self._quit = False

        try:
            if self.config.greeting:
                self._say_alone(self.config.greeting)

            self._emit("status", "Listening - speak whenever you're ready")
            self._prompts = self.recognizer.listen()

            for prompt in self._prompts:
                if not self.handle(prompt["text"]):
                    break

        finally:
            self._prompts = None
            self.recognizer.on_speech_start = None
            if hasattr(self.recognizer, "watch"):
                self.recognizer.watch(None)
            self.speaker.stop()

    def close(self):
        self.agent.close()
        self.speaker.close()
        self.recognizer.close()

        if self._echo_canceller is not None:
            self.speaker.on_playback = None
            self.recognizer.echo_canceller = None
            self._echo_canceller.close()
            self._echo_canceller = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # --------------------------------------------------
    # Turns
    # --------------------------------------------------

    def handle(self, text):
        """Handle one utterance. Returns False when the session should end."""

        self._emit("user", text)

        reply = self.commands.dispatch(text)

        if reply is NOT_HANDLED:
            self._respond(text)
        elif reply:
            self._say_alone(reply)

        return not self._quit

    def _respond(self, text):

        self._mute()
        self._responding = True
        self._cancelled = False
        self._thinking_shown = False
        self._reply_started = False
        self._spoken = False
        self._said = ""
        self._start_wait_cue()

        stream = None

        try:
            stream = self.agent.respond(text, self)
            completed = self.speaker.speak_stream(self._speech(stream))
            self._end_reply()

            if not completed:
                self._emit("interrupted")

        except AgentError as exc:
            self._fail(str(exc), exc)

        except Exception as exc:
            logger.exception("Agent failed")
            self._fail(f"{type(exc).__name__}: {exc}", exc)

        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

            self._stop_wait_cue()
            self._responding = False
            self._unmute()

    def _speech(self, stream):
        """Turn agent events into text for the speaker, showing the rest."""

        for item in stream:
            if isinstance(item, str):
                item = TextDelta(item)

            if isinstance(item, TextDelta):
                if not item.text:
                    continue
                self._stop_wait_cue()
                self._start_reply()
                self._emit("assistant_text", item.text)
                self._said += item.text
                yield item.text

            elif isinstance(item, Status) and item.speak and self.config.speak_status:
                self._stop_wait_cue()
                self._show(item)
                self._said += f" {item.text} "
                # Its own sentence, in order with the rest of the reply.
                yield f"\n{item.text}\n"

            elif item is not None:
                self._show(item)

    def _fail(self, message, exc):
        self._end_reply()
        self._emit("error", message)
        logger.debug("Agent error", exc_info=exc)
        self._said += f" {self.config.error_reply}"
        self.speaker.speak(self.config.error_reply, wait=True)

    # --------------------------------------------------
    # TurnContext (used by agents and tools)
    # --------------------------------------------------

    @property
    def session(self):
        """For tools that receive ctx: ctx.session is this session."""
        return self

    @property
    def cancelled(self):
        return self._cancelled

    def emit(self, event):
        """Report an event out of band (callbacks, worker threads)."""

        if isinstance(event, str):
            event = TextDelta(event)

        if isinstance(event, TextDelta):
            self.say(event.text)
        elif isinstance(event, Status) and event.speak and self.config.speak_status:
            self._show(event)
            self._stop_wait_cue()
            self._said += f" {event.text}"
            self.speaker.speak(event.text)
        else:
            self._show(event)

    def say(self, text):
        """Speak a sentence now, outside the reply stream."""

        if not text:
            return

        self._stop_wait_cue()
        separator = " " if self._reply_started else ""
        self._start_reply()
        self._emit("assistant_text", separator + text)
        self._said += f" {text}"
        self.speaker.speak(text)

    def ask(self, question):
        """
        Speak a question and return the user's answer (None if the
        session is not listening). Prefer LangGraph interrupts or tool
        confirmations, which call this for you.
        """

        if self._prompts is None:
            raise AgentError("ask() needs a running voice session")

        self.say(question)
        self.speaker.wait()
        self._end_reply()

        self._asking = True
        self._unmute()

        try:
            prompt = next(self._prompts, None)
        finally:
            self._asking = False
            self._mute()

        if prompt is None:
            return None

        answer = prompt["text"]
        self._emit("user", answer)
        self._reply_started = False
        self._thinking_shown = False
        return answer

    def confirm(self, question):
        return classify_confirmation(self.ask(question) or "") is True

    # --------------------------------------------------
    # Session controls (used by commands and control tools)
    # --------------------------------------------------

    def change_speed(self, direction):
        """direction: +1 faster, -1 slower, 0 normal. Returns the reply."""

        config = self.speaker.config

        if direction == 0:
            config.speed = 1.0
            return "Back to normal speed."

        speed = round(config.speed + direction * self.config.speed_step, 2)
        speed = min(2.0, max(0.5, speed))

        if speed == config.speed:
            return f"That's as {'fast' if direction > 0 else 'slow'} as I go."

        config.speed = speed
        return f"Speaking {'faster' if direction > 0 else 'more slowly'}."

    def reset_conversation(self):
        self.agent.reset()
        return self.config.reset_reply

    def request_stop(self):
        """End the session after the current turn."""
        self._quit = True
        return self.config.goodbye

    def status_text(self):
        description = self.agent.describe()
        speed = f"My speaking speed is {self.speaker.config.speed:.1f}."
        return f"{description} {speed}".strip()

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    def _show(self, event):

        if isinstance(event, (ToolStarted, ToolFinished, Status)) and self._reply_started:
            # Close the "AI:" line; later text starts a fresh one.
            self._emit("assistant_end")
            self._reply_started = False

        if isinstance(event, Thinking):
            if not self._thinking_shown:
                self._thinking_shown = True
                self._emit("thinking_start")
            if self.config.show_thinking:
                self._emit("thinking", event.text)

        elif isinstance(event, ToolStarted):
            arguments = ", ".join(f"{k}={v}" for k, v in event.arguments.items())
            self._emit("tool", f"{event.name}({arguments})")

        elif isinstance(event, ToolFinished):
            self._emit("tool_result", f"{event.name}: {event.result}")

        elif isinstance(event, Status):
            self._emit("progress", event.text)

    def _start_reply(self):

        if self._reply_started:
            return

        if self._thinking_shown:
            self._emit("thinking_end")
            self._thinking_shown = False

        self._reply_started = True
        self._emit("assistant_start")

    def _end_reply(self):

        if self._reply_started:
            self._emit("assistant_end")
        elif self._thinking_shown:
            self._emit("thinking_end")

        self._reply_started = False
        self._thinking_shown = False

    def _say_alone(self, text):
        """Speak a complete phrase (greeting, command reply) and wait."""

        self._mute()
        self._said = text

        try:
            self._emit("assistant_start")
            self._emit("assistant_text", text)
            self._emit("assistant_end")
            self.speaker.speak(text, wait=True)
        finally:
            self._unmute()

    def _emit(self, kind, text=""):
        if self.on_event is None:
            return
        try:
            self.on_event(kind, text)
        except Exception:
            logger.exception("on_event handler failed")

    # --------------------------------------------------
    # Wait cue
    # --------------------------------------------------

    def _start_wait_cue(self):

        if not self.config.wait_cue or self.config.wait_cue_after_s <= 0:
            return

        def fire():
            if self._responding and not self._spoken:
                self._spoken = True
                self._said += f" {self.config.wait_cue}"
                self.speaker.speak(self.config.wait_cue)

        self._cue_timer = threading.Timer(self.config.wait_cue_after_s, fire)
        self._cue_timer.daemon = True
        self._cue_timer.start()

    def _stop_wait_cue(self):
        self._spoken = True
        if self._cue_timer is not None:
            self._cue_timer.cancel()
            self._cue_timer = None

    # --------------------------------------------------
    # Microphone
    # --------------------------------------------------

    def _mute(self):
        if not self._mic_live:
            self.recognizer.pause()

    def _unmute(self):
        if not self._mic_live:
            self.speaker.wait()
            time.sleep(self.config.resume_delay_ms / 1000)
            self.recognizer.resume()

    def _on_user_speech(self):
        # barge_in="speech": runs on the recognizer's capture thread as
        # soon as anyone starts talking.
        if self._asking:
            return
        if self._responding or self.speaker.is_speaking:
            self._interrupt()

    def _on_overheard(self, text):
        # barge_in="phrase": runs on the recognizer's worker thread with
        # what it heard while muted, which may be our own voice.
        if self._asking or not (self._responding or self.speaker.is_speaking):
            return
        if self.is_stop_request(text):
            logger.info("Stopped by %r", text)
            self._interrupt()

    def is_stop_request(self, heard):
        """
        True if `heard` contains a stop phrase that the assistant didn't
        just say itself (the microphone picks up the speakers).
        """
        heard = normalize_command(heard)
        said = normalize_command(self._said)
        return any(
            p.search(heard) and not p.search(said) for p in self._stop_patterns
        )

    def _interrupt(self):
        self._cancelled = True
        self.speaker.stop()

    def _setup_echo_cancellation(self):

        if self._echo_canceller is not None:
            return  # already set up (run() after start())

        wanted = self.config.echo_cancellation
        if wanted is None:
            wanted = self.config.interruptible
        if not wanted:
            return

        if not (hasattr(self.speaker, "on_playback")
                and hasattr(self.recognizer, "echo_canceller")):
            logger.warning("This speaker or recognizer doesn't support echo cancellation.")
            return

        from .echo import EchoCanceller

        try:
            canceller = EchoCanceller()
        except ImportError:
            if self.config.echo_cancellation:
                raise
            logger.warning(
                'Echo cancellation is off: pip install "voicekit[aec]" so the '
                "microphone doesn't hear the assistant."
            )
            return

        frame_s = getattr(getattr(self.recognizer, "config", None), "frame_ms", 30) / 1000
        canceller.set_delay((getattr(self.speaker, "output_latency", 0.0) + frame_s) * 1000)

        self.speaker.on_playback = canceller.playback
        self.recognizer.echo_canceller = canceller
        self._echo_canceller = canceller
        self._emit("status", "Echo cancellation on")

    def _teach_vocabulary(self):
        """Pass agent-specific words (e.g. model names) to Whisper."""

        words = getattr(self.agent, "vocabulary", lambda: [])()
        recognizer_config = getattr(self.recognizer, "config", None)

        if words and recognizer_config is not None and not recognizer_config.hotwords:
            recognizer_config.hotwords = ", ".join(words)

    # --------------------------------------------------
    # Exact commands
    # --------------------------------------------------

    def _register_commands(self):

        c = self.commands
        c.add_phrases("exit", self.config.exit_phrases, self.request_stop,
                      "goodbye")
        c.add_phrases("reset", self.config.reset_phrases,
                      self.reset_conversation, "new conversation")
        # "Hang on" on its own, when nothing is being said: just wait.
        c.add_phrases("stop", self.config.stop_phrases, lambda: None)
        c.add("faster", r"(?:speak|talk)(?: a bit| a little)? faster",
              lambda: self.change_speed(+1), "speak faster")
        c.add("slower", r"(?:speak|talk)(?: a bit| a little)? slower",
              lambda: self.change_speed(-1), "speak slower")
        c.add("normal_speed",
              [r"(?:speak |talk )?(?:at )?normal speed", r"reset (?:the )?speed"],
              lambda: self.change_speed(0), "normal speed")
        c.add("status",
              [r"status", r"what s your status", r"(?:current )?settings"],
              self.status_text, "status")
        c.add("help",
              [r"help", r"what (?:commands|can i say)(?: are there)?",
               r"(?:list|voice) commands"],
              self._help, "help")

    def _help(self):

        names = self.commands.descriptions()
        listed = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)
        prefix = (
            f"Start commands with {self.config.command_prefix}. "
            if self.config.command_prefix else ""
        )
        return f"{prefix}You can say: {listed}. Or just ask me in your own words."


# --------------------------------------------------
# Tools the session offers to agents
# --------------------------------------------------

def control_tools():
    """
    Voice-session tools a model can call: change the speaking speed,
    start a new conversation, report status, end the session. They use
    ctx.session, so they work with any VoiceSession.
    """

    @tool(
        return_direct=True,
        confirm=lambda change: {
            "faster": "Speak faster?", "slower": "Speak more slowly?",
        }.get(change, "Go back to normal speed?"),
    )
    def set_speaking_speed(ctx, change: Literal["faster", "slower", "normal"]) -> str:
        """Make the assistant's voice speak faster, slower, or at normal speed."""
        return ctx.session.change_speed({"faster": 1, "slower": -1}.get(change, 0))

    @tool(
        return_direct=True,
        confirm=lambda: "Start a new conversation? I'll forget what we've said.",
    )
    def new_conversation(ctx) -> str:
        """Clear the conversation history and start fresh."""
        return ctx.session.reset_conversation()

    @tool
    def get_status(ctx) -> str:
        """Report the assistant's current setup, such as the model in use and its speaking speed."""
        return ctx.session.status_text()

    @tool(return_direct=True, confirm=lambda: "Stop listening and quit?")
    def end_session(ctx) -> str:
        """End the voice session and stop listening. Only when the user says they are done."""
        return ctx.session.request_stop()

    return [set_speaking_speed, new_conversation, get_status, end_session]
