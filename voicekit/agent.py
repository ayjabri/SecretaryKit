"""
The contract between the voice loop and whatever does the thinking.

An agent receives what the user said and returns a stream of events. The
voice session speaks the text, shows the rest, and offers the agent a
context object to talk back mid-task:

    class MyAgent(Agent):
        def respond(self, text, ctx):
            ctx.say("Checking the build servers.")       # spoken now
            if not ctx.confirm("Deploy to production?"):  # asks, waits for the answer
                yield "Okay, not deploying."
                return
            yield Status("Deploying...")                  # spoken progress update
            yield "Deployed."                             # the reply (str = TextDelta)

Anything can sit behind this interface: a local Ollama model, Claude, a
LangGraph orchestrator running several agents, or a plain function. See
voicekit.agents for ready-made adapters.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional, Protocol, Union


# --------------------------------------------------
# Events
# --------------------------------------------------

@dataclass
class TextDelta:
    """Part of the spoken reply. Agents may also yield plain strings."""
    text: str


@dataclass
class Thinking:
    """The model's reasoning: shown on screen, never spoken."""
    text: str


@dataclass
class Status:
    """
    A progress update from a long-running task ("Agent B finished the
    tests"). Spoken as its own sentence when speak=True, otherwise only
    shown.
    """
    text: str
    speak: bool = True


@dataclass
class ToolStarted:
    name: str
    arguments: dict = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class ToolFinished:
    name: str
    result: str
    id: Optional[str] = None
    error: bool = False


Event = Union[TextDelta, Thinking, Status, ToolStarted, ToolFinished]


class AgentError(RuntimeError):
    """An agent could not answer (server down, API error...)."""


# --------------------------------------------------
# What the session offers an agent during a turn
# --------------------------------------------------

class TurnContext(Protocol):
    """
    Passed to Agent.respond(). Implemented by VoiceSession; tests and
    non-voice front ends can supply their own.
    """

    @property
    def cancelled(self) -> bool:
        """True once the user interrupted; stop work at the next chance."""

    def emit(self, event: Event) -> None:
        """Report an event from a callback or worker thread."""

    def say(self, text: str) -> None:
        """Speak a sentence now, outside the reply stream."""

    def ask(self, question: str) -> Optional[str]:
        """Speak a question and return the user's spoken answer."""

    def confirm(self, question: str) -> bool:
        """Ask a yes/no question; True only for a clear yes."""


# --------------------------------------------------
# Agents
# --------------------------------------------------

class Agent:
    """
    Base class for anything the voice session can talk to. Only
    respond() is required.
    """

    #: Optional voicekit.tools.ToolSet. Agents that let the model call
    #: tools expose it so the session can add its own (e.g. "speak
    #: faster", "end the session").
    tools = None

    def start(self) -> None:
        """Connect / load / warm up. Raise AgentError if unusable."""

    def respond(self, text: str, ctx: TurnContext) -> Iterable[Union[Event, str]]:
        """Yield the reply for one user utterance."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget the conversation."""

    def describe(self) -> str:
        """One spoken sentence about the agent's current setup."""
        return ""

    def register_commands(self, router) -> None:
        """Add exact voice commands to the session's CommandRouter."""

    def close(self) -> None:
        """Release connections."""


class FunctionAgent(Agent):
    """
    Wrap any callable as an agent: the quickest way to hook up your own
    orchestrator.

        def handle(text, ctx):
            return f"You said {text}"

        session = VoiceSession(FunctionAgent(handle))

    The function receives (text, ctx) - or just (text) - and may return a
    string, an iterable of strings / events, or be a generator.
    """

    def __init__(self, function: Callable, reset: Optional[Callable] = None,
                 describe: str = ""):
        self.function = function
        self._reset = reset
        self._description = describe

        import inspect

        try:
            parameters = inspect.signature(function).parameters
            self._wants_context = len(parameters) >= 2
        except (TypeError, ValueError):
            self._wants_context = True

    def respond(self, text, ctx):

        result = self.function(text, ctx) if self._wants_context else self.function(text)

        if result is None:
            return
        if isinstance(result, str):
            yield result
            return

        yield from result

    def reset(self):
        if self._reset:
            self._reset()

    def describe(self):
        return self._description


def iter_events(items: Iterable[Union[Event, str]]) -> Iterator[Event]:
    """Normalise an agent's output: plain strings become TextDelta."""

    for item in items:
        if isinstance(item, str):
            if item:
                yield TextDelta(item)
        elif item is not None:
            yield item
