"""
Tools a model can decide to call, shared by every tool-calling agent
(Ollama, OpenAI-compatible servers, Claude) and convertible to LangChain
tools for LangGraph.

Define one with the decorator; the schema comes from the type hints and
docstring:

    @tool(return_direct=True)
    def set_thinking(enabled: bool) -> str:
        '''Turn step-by-step thinking on or off.

        Args:
            enabled: True for more careful but slower answers.
        '''
        ...
        return "Turning thinking on."

A parameter named `ctx` receives the voice session's TurnContext, so a
tool can speak progress (ctx.say) or ask the user (ctx.ask, ctx.confirm).
"""

import enum
import inspect
import logging
import re
import typing
from dataclasses import dataclass, field
from typing import Callable, Optional

from .commands import normalize_command


logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """
    name, description
        What the model sees. A clear description is what makes the model
        pick the tool from loosely worded requests.

    parameters, required
        JSON-schema "properties" for the arguments, and which are
        required. Use "enum" for fixed choices so the model can only pick
        valid values.

    run
        Called with the arguments (plus ctx=... if wants_context);
        returns a short result for the model, or for the user when
        return_direct is set.

    return_direct
        Speak the result to the user and end the model's turn, instead of
        sending it back for the model to rephrase. Good for actions that
        just need an announcement ("Switching to Qwen 3.6 27B.").

    confirm
        Optional function returning a yes/no question to ask before
        running ("Switch to Qwen 3.6 27B?"). Only used when the agent is
        set to confirm tools.
    """

    name: str
    description: str
    run: Callable[..., object]
    parameters: dict = field(default_factory=dict)
    required: tuple = ()
    return_direct: bool = False
    confirm: Optional[Callable[..., str]] = None
    wants_context: bool = False

    # --------------------------------------------------
    # Schemas for each provider
    # --------------------------------------------------

    def json_schema(self):
        return {
            "type": "object",
            "properties": self.parameters,
            "required": list(self.required),
        }

    def schema(self):
        """Ollama / OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def anthropic_schema(self):
        """Claude Messages API format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema(),
            # Stream tool inputs as they are generated; we validate them.
            "eager_input_streaming": True,
        }

    # --------------------------------------------------
    # Running
    # --------------------------------------------------

    def validate(self, arguments):
        """
        Check and lightly coerce arguments ("true" -> True, "5" -> 5).
        Returns (arguments, error); error is None when valid.
        """

        if not isinstance(arguments, dict):
            return {}, f"arguments for {self.name} must be an object"

        cleaned = dict(arguments)

        for name in self.required:
            if name not in cleaned:
                return cleaned, f"missing required argument '{name}'"

        for name, value in list(cleaned.items()):
            spec = self.parameters.get(name)

            if spec is None:
                continue

            value, ok = _coerce(value, spec.get("type"))

            if not ok:
                return cleaned, f"'{name}' must be of type {spec.get('type')}"

            if "enum" in spec and value not in spec["enum"]:
                return cleaned, (
                    f"'{name}' must be one of: "
                    + ", ".join(str(v) for v in spec["enum"])
                )

            cleaned[name] = value

        return cleaned, None

    def __call__(self, ctx=None, **arguments):
        if self.wants_context:
            return self.run(ctx=ctx, **arguments)
        return self.run(**arguments)


def _coerce(value, json_type):

    if json_type is None:
        return value, True

    if json_type == "boolean":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true", True
        return value, False

    if json_type == "integer":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, True
        try:
            return int(str(value)), True
        except ValueError:
            return value, False

    if json_type == "number":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return value, True
        try:
            return float(str(value)), True
        except ValueError:
            return value, False

    expected = {"string": str, "array": list, "object": dict}.get(json_type)

    if expected is None or isinstance(value, expected):
        return value, True

    if expected is str:
        return str(value), True

    return value, False


# --------------------------------------------------
# @tool decorator
# --------------------------------------------------

def tool(function=None, *, name=None, description=None, return_direct=False,
         confirm=None):
    """
    Turn a function into a Tool. Type hints become the JSON schema
    (str, int, float, bool, list, dict, Literal[...], Enum, Optional);
    the docstring's first part becomes the description and an "Args:"
    section documents each parameter.
    """

    def build(fn):

        signature = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        summary, parameter_docs = _parse_docstring(inspect.getdoc(fn) or "")

        parameters = {}
        required = []
        wants_context = False

        for parameter_name, parameter in signature.parameters.items():

            if parameter_name == "ctx":
                wants_context = True
                continue

            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue

            schema = _type_schema(hints.get(parameter_name, str))

            if parameter_name in parameter_docs:
                schema["description"] = parameter_docs[parameter_name]

            parameters[parameter_name] = schema

            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        return Tool(
            name=name or fn.__name__,
            description=description or summary or fn.__name__.replace("_", " "),
            run=fn,
            parameters=parameters,
            required=tuple(required),
            return_direct=return_direct,
            confirm=confirm,
            wants_context=wants_context,
        )

    return build(function) if function is not None else build


def _type_schema(annotation):

    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)

    if origin is typing.Union or (
        origin is not None and getattr(origin, "__name__", "") == "UnionType"
    ):
        # Optional[X] / X | None
        options = [a for a in arguments if a is not type(None)]
        return _type_schema(options[0]) if options else {}

    if origin is typing.Literal:
        values = list(arguments)
        schema = {"enum": values}
        if all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        elif all(isinstance(v, bool) for v in values):
            schema["type"] = "boolean"
        elif all(isinstance(v, int) for v in values):
            schema["type"] = "integer"
        return schema

    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return {"type": "string", "enum": [m.value for m in annotation]}

    if origin in (list, tuple, set) or annotation in (list, tuple, set):
        schema = {"type": "array"}
        if arguments:
            schema["items"] = _type_schema(arguments[0])
        return schema

    if origin is dict or annotation is dict:
        return {"type": "object"}

    simple = {str: "string", int: "integer", float: "number", bool: "boolean"}

    if annotation in simple:
        return {"type": simple[annotation]}

    return {"type": "string"}


def _parse_docstring(doc):
    """Split a Google-style docstring into (summary, {param: text})."""

    match = re.search(r"^\s*(Args|Arguments|Parameters):\s*$", doc, re.MULTILINE)

    if not match:
        return " ".join(doc.split()), {}

    summary = " ".join(doc[:match.start()].split())
    parameter_docs = {}
    current = None

    for line in doc[match.end():].splitlines():
        entry = re.match(r"^\s{2,}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)

        if entry:
            current = entry.group(1)
            parameter_docs[current] = entry.group(2).strip()
        elif current and line.strip() and line.startswith("        "):
            parameter_docs[current] += " " + line.strip()
        elif line.strip() and not line.startswith(" "):
            break

    return summary, parameter_docs


# --------------------------------------------------
# ToolSet
# --------------------------------------------------

class ToolSet:

    def __init__(self, tools=()):
        self.tools = {}
        for t in tools:
            self.add(t)

    def add(self, *tools):
        for t in tools:
            if not isinstance(t, Tool):
                t = tool(t)  # plain function
            self.tools[t.name] = t

    def remove(self, name):
        self.tools.pop(name, None)

    def get(self, name):
        return self.tools.get(name)

    def __contains__(self, name):
        return name in self.tools

    def __len__(self):
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools.values())

    def schemas(self):
        return [t.schema() for t in self.tools.values()]

    def anthropic_schemas(self):
        return [t.anthropic_schema() for t in self.tools.values()]

    # Kept for callers that decide confirmation themselves.

    def needs_confirmation(self, calls, always=False):
        """True if any call has a confirmation question (or always=True)."""

        if always:
            return bool(calls)

        return any(
            call.name not in self.tools or self.tools[call.name].confirm
            for call in calls
        )

    def confirmation_question(self, calls):
        return " ".join(
            _question(self.tools[call.name], call)
            for call in calls if call.name in self.tools
        )

    def run(self, calls, ctx=None):
        """Run calls in order; returns one result string per call."""
        return [self.run_one(call, ctx)[0] for call in calls]

    def run_one(self, call, ctx=None):
        """Returns (result, is_error)."""

        t = self.tools.get(call.name)

        if t is None:
            return f"Error: there is no tool called {call.name}.", True

        arguments, error = t.validate(call.arguments)

        if error:
            logger.warning("Bad arguments for %s: %s", call.name, error)
            return f"Error: invalid arguments for {call.name}: {error}.", True

        try:
            result = t(ctx=ctx, **arguments)
        except TypeError as exc:
            logger.warning("Bad arguments for %s: %s", call.name, exc)
            return f"Error: invalid arguments for {call.name}.", True
        except Exception as exc:
            logger.exception("Tool %s failed", call.name)
            return f"Error: {call.name} failed: {exc}", True

        return ("" if result is None else str(result)), False


def _question(t, call):
    if t.confirm is None:
        return f"Run {t.name.replace('_', ' ')}?"
    try:
        return t.confirm(**call.arguments)
    except TypeError:
        return f"Run {t.name.replace('_', ' ')}?"


@dataclass
class ToolRun:
    """Outcome of run_tool_calls()."""

    results: list
    errors: list
    # Results of return_direct tools, to be spoken as the final answer.
    direct: list

    @property
    def ends_turn(self):
        return bool(self.direct)


def run_tool_calls(toolset, calls, ctx=None, confirm=False):
    """
    Run the model's tool calls, reporting ToolStarted / ToolFinished to
    ctx. With confirm=True, tools that define a confirmation question ask
    the user first; a "no" is reported back to the model.
    """

    from .agent import ToolFinished, ToolStarted

    results, errors, direct = [], [], []

    for call in calls:
        t = toolset.get(call.name)

        if ctx is not None:
            ctx.emit(ToolStarted(call.name, dict(call.arguments), call.id))

        if confirm and t is not None and t.confirm and ctx is not None:
            if not ctx.confirm(_question(t, call)):
                result, is_error = "The user declined; it was not run.", False
                results.append(result)
                errors.append(is_error)
                ctx.emit(ToolFinished(call.name, result, call.id, is_error))
                continue

        result, is_error = toolset.run_one(call, ctx)
        results.append(result)
        errors.append(is_error)

        if ctx is not None:
            ctx.emit(ToolFinished(call.name, result, call.id, is_error))

        if t is not None and t.return_direct and not is_error:
            direct.append(result)

    return ToolRun(results, errors, direct)


# --------------------------------------------------
# Spoken yes / no
# --------------------------------------------------

_NO_RE = re.compile(
    r"(?:no|nope|nah|cancel|don t|do not|never ?mind|stop|negative|"
    r"not now|abort|forget it|leave it)\b"
)
_YES_RE = re.compile(
    r"(?:yes|yeah|yep|yup|ya|sure|ok|okay|go ahead|do it|please do|"
    r"confirm(?:ed)?|correct|affirmative|absolutely|of course|"
    r"sounds good|that s right|right|alright|all right|please)\b"
)


def classify_confirmation(text, max_words=6):
    """
    Interpret a spoken answer to a yes/no question.

    Returns True (yes), False (no), or None if it isn't a short yes/no
    answer, e.g. the user moved on to something else.
    """

    answer = normalize_command(text or "")

    if not answer or len(answer.split()) > max_words:
        return None

    if _NO_RE.match(answer):
        return False

    if _YES_RE.match(answer):
        return True

    return None
