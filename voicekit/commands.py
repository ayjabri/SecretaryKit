"""
Spoken commands for voice interfaces: the voice equivalent of typing
"/set think" in `ollama run`.

A command only runs when the *whole* utterance matches one of its
patterns, so ordinary questions ("how do I stop thinking about work?")
are never mistaken for commands.

Example
-------
    router = CommandRouter()
    router.add(
        "think_on",
        [r"(?:set |turn on |enable )?think(?:ing)?(?: mode)?(?: on)?"],
        lambda: "Thinking on.",
        description="think on",
    )

    reply = router.dispatch("Set think.")    # -> "Thinking on."
    reply = router.dispatch("What's 2+2?")   # -> NOT_HANDLED
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


class _NotHandled:
    def __repr__(self):
        return "NOT_HANDLED"

    def __bool__(self):
        return False


# Returned by dispatch() when no command matched, and by a handler to
# decline a match so the utterance falls through (e.g. "switch to X"
# where X turns out not to be a model).
NOT_HANDLED = _NotHandled()


def normalize_command(text):
    """
    Lower-case, turn punctuation into spaces and collapse whitespace:
    "Set no-think." -> "set no think".
    """
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class VoiceCommand:
    name: str
    patterns: Sequence["re.Pattern"]
    handler: Callable[..., object]
    description: str = ""


class CommandRouter:
    """
    Matches transcribed speech against registered commands.

    prefix
        Optional wake word, e.g. "computer". When set, commands only run
        when the utterance starts with it ("Computer, set think"), which
        rules out accidental matches entirely.
    """

    def __init__(self, prefix: Optional[str] = None):
        self.prefix = normalize_command(prefix) if prefix else None
        self.commands = []

    def add(self, name, patterns, handler, description=""):
        """
        Register a command.

        patterns
            Regular expressions matched against the whole normalised
            utterance (lower case, no punctuation). Named groups are
            passed to the handler as keyword arguments.

        handler
            Returns the reply to speak (or None for no reply), or
            NOT_HANDLED to let the utterance fall through.
        """

        if isinstance(patterns, str):
            patterns = [patterns]

        self.commands.append(
            VoiceCommand(
                name=name,
                patterns=[re.compile(p) for p in patterns],
                handler=handler,
                description=description,
            )
        )

    def add_phrases(self, name, phrases, handler, description=""):
        """Register a command triggered by exact phrases."""

        self.add(
            name,
            [re.escape(normalize_command(p)) for p in phrases],
            handler,
            description,
        )

    def dispatch(self, text):
        """
        Run the command matching `text`.

        Returns the handler's reply (possibly None), or NOT_HANDLED if
        the text is not a command and should go to the LLM.
        """

        command_text = self._strip_prefix(normalize_command(text))

        if command_text is None:
            return NOT_HANDLED

        for command in self.commands:
            for pattern in command.patterns:
                match = pattern.fullmatch(command_text)

                if match is None:
                    continue

                arguments = {
                    key: value
                    for key, value in match.groupdict().items()
                    if value is not None
                }
                result = command.handler(**arguments)

                if result is not NOT_HANDLED:
                    return result

        return NOT_HANDLED

    def descriptions(self):
        return [c.description for c in self.commands if c.description]

    def _strip_prefix(self, text):

        if self.prefix is None:
            return text

        if text == self.prefix:
            return ""

        if text.startswith(self.prefix + " "):
            return text[len(self.prefix) + 1:]

        return None
