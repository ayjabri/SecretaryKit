"""
Talk to any compiled LangGraph graph: a single ReAct agent, or an
orchestrator running many agents.

    graph = builder.compile(checkpointer=InMemorySaver())
    VoiceSession(LangGraphAgent(graph)).run()

How the graph's stream maps to the voice session:

    LLM tokens from the top-level graph  -> spoken reply
                                            (sub-agents stay quiet unless
                                            speak_subgraphs / speak_nodes)
    get_stream_writer()(...)             -> spoken progress update
                                            (any node, any depth)
    tool calls / tool results            -> shown on screen
    interrupt("Deploy to prod?")         -> asked aloud; the spoken answer
                                            resumes the graph via
                                            Command(resume=answer)

Inside nodes and tools, voice_context() gives the live session, e.g.
voice_context().say("Starting the test agent.").

The graph should be compiled with a checkpointer (conversation memory and
interrupts need one). Without one, this agent keeps the message history
itself and passes it in on every turn.
"""

import logging
import uuid
from typing import Callable, Iterable, Optional

from ..agent import Agent, AgentError, Status, TextDelta, ToolFinished, ToolStarted


logger = logging.getLogger(__name__)


# Key under config["configurable"] holding the voice TurnContext.
VOICE_CONTEXT_KEY = "voicekit_context"


class LangGraphAgent(Agent):
    """
    graph
        A compiled graph (StateGraph.compile(), create_react_agent(), ...).

    input_builder
        fn(text) -> graph input. Default: {"messages": [HumanMessage(text)]}
        for MessagesState-style graphs.

    speak_nodes
        Only speak LLM tokens produced in these nodes (by name). Default:
        every node of the top-level graph except "tools".

    speak_subgraphs
        Also speak tokens from nested graphs (sub-agents).

    resume_builder
        fn(answer_text, interrupt_value) -> value passed to
        Command(resume=...). Default: the spoken answer text.

    thread_id
        Conversation thread for the checkpointer. reset() starts a new one.

    config
        Extra RunnableConfig entries (e.g. {"recursion_limit": 50}).
    """

    def __init__(self, graph, *, input_builder: Optional[Callable] = None,
                 speak_nodes: Optional[Iterable[str]] = None,
                 speak_subgraphs: bool = False,
                 resume_builder: Optional[Callable] = None,
                 thread_id: Optional[str] = None,
                 config: Optional[dict] = None,
                 description: str = ""):

        self.graph = graph
        self.input_builder = input_builder
        self.speak_nodes = set(speak_nodes) if speak_nodes else None
        self.speak_subgraphs = speak_subgraphs
        self.resume_builder = resume_builder
        self.thread_id = thread_id or str(uuid.uuid4())
        self.extra_config = config or {}
        self.description = description
        self._history = []

    @property
    def _has_checkpointer(self):
        return getattr(self.graph, "checkpointer", None) not in (None, False)

    # --------------------------------------------------
    # Agent interface
    # --------------------------------------------------

    def respond(self, text, ctx):

        from langgraph.types import Command

        config = {
            **self.extra_config,
            "configurable": {
                **self.extra_config.get("configurable", {}),
                "thread_id": self.thread_id,
                VOICE_CONTEXT_KEY: ctx,
            },
        }

        payload = self._input(text)

        while True:
            interrupt_value = _NO_INTERRUPT

            try:
                for part in self.graph.stream(
                    payload,
                    config,
                    stream_mode=self._stream_modes(),
                    subgraphs=True,
                    version="v2",
                ):
                    if ctx.cancelled:
                        return

                    value = self._handle_part(part, ctx)

                    if isinstance(value, TextDelta):
                        yield value
                    elif value is not None:
                        interrupt_value = value

            except Exception as exc:
                if isinstance(exc, AgentError):
                    raise
                raise AgentError(f"The graph failed: {exc}") from exc

            if interrupt_value is _NO_INTERRUPT:
                return

            # Human in the loop: ask aloud, resume with the answer.
            answer = ctx.ask(_interrupt_text(interrupt_value))

            if answer is None:
                return

            resume = (
                self.resume_builder(answer, interrupt_value)
                if self.resume_builder else answer
            )
            payload = Command(resume=resume)

    def reset(self):
        self.thread_id = str(uuid.uuid4())
        self._history = []

    def describe(self):
        return self.description

    # --------------------------------------------------
    # Stream handling
    # --------------------------------------------------

    def _stream_modes(self):
        modes = ["messages", "custom", "updates"]
        if not self._has_checkpointer:
            modes.append("values")
        return modes

    def _input(self, text):

        if self.input_builder:
            return self.input_builder(text)

        from langchain_core.messages import HumanMessage

        message = HumanMessage(content=text)

        if self._has_checkpointer:
            return {"messages": [message]}

        return {"messages": self._history + [message]}

    def _handle_part(self, part, ctx):
        """
        Returns a TextDelta to speak, an interrupt value, or None. Other
        events are reported straight to ctx.
        """

        kind = part.get("type")
        namespace = part.get("ns") or ()
        data = part.get("data")

        if kind == "messages":
            chunk, metadata = data
            if self._should_speak(chunk, namespace, metadata or {}):
                text = message_text(chunk.content)
                if text:
                    return TextDelta(text)

        elif kind == "custom":
            ctx.emit(_status(data))

        elif kind == "updates" and isinstance(data, dict):
            if "__interrupt__" in data:
                interrupts = data["__interrupt__"]
                if interrupts:
                    return getattr(interrupts[0], "value", interrupts[0])
            for update in data.values():
                self._report_tools(update, ctx)

        elif kind == "values" and not namespace and isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                self._history = list(data["messages"])

        return None

    def _should_speak(self, chunk, namespace, metadata):

        from langchain_core.messages import AIMessage, AIMessageChunk

        if not isinstance(chunk, (AIMessage, AIMessageChunk)):
            return False

        if namespace and not self.speak_subgraphs:
            return False

        node = metadata.get("langgraph_node")

        if self.speak_nodes is not None:
            return node in self.speak_nodes

        return node != "tools"

    @staticmethod
    def _report_tools(update, ctx):

        from langchain_core.messages import AIMessage, ToolMessage

        if not isinstance(update, dict):
            return

        messages = update.get("messages")

        if messages is None:
            return

        if not isinstance(messages, (list, tuple)):
            messages = [messages]

        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    ctx.emit(ToolStarted(call.get("name", ""), call.get("args") or {}, call.get("id")))
            elif isinstance(message, ToolMessage):
                ctx.emit(ToolFinished(
                    message.name or "tool",
                    message_text(message.content),
                    message.tool_call_id,
                    getattr(message, "status", "success") == "error",
                ))


_NO_INTERRUPT = object()


def message_text(content):
    """Text from a LangChain message's content (str or content blocks)."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return ""


def _status(data):
    """A custom stream event -> Status. Accepts a string or a dict."""

    if isinstance(data, Status):
        return data

    if isinstance(data, dict):
        text = next(
            (str(data[k]) for k in ("status", "message", "text", "progress") if k in data),
            str(data),
        )
        return Status(text, speak=bool(data.get("speak", True)))

    return Status(str(data))


def _interrupt_text(value):
    """What to say aloud for an interrupt value."""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("question", "message", "prompt", "text"):
            if key in value:
                return str(value[key])

    return str(value)


# --------------------------------------------------
# Helpers for graph authors
# --------------------------------------------------

class _NoVoice:
    """Stand-in when a node runs outside a voice session."""

    cancelled = False

    def say(self, text):
        logger.info("say: %s", text)

    def emit(self, event):
        logger.info("event: %s", event)

    def ask(self, question):
        return None

    def confirm(self, question):
        return False


def voice_context():
    """
    The live voice session inside a node or tool:
    voice_context().say("Agent B is done."). Outside a voice session,
    a no-op stand-in.
    """

    try:
        from langgraph.config import get_config
        context = get_config().get("configurable", {}).get(VOICE_CONTEXT_KEY)
    except Exception:
        context = None

    return context if context is not None else _NoVoice()


def to_langchain_tools(tools):
    """
    Convert voicekit Tools (e.g. VoiceSession's control_tools()) to
    LangChain tools for a LangGraph agent. Tools that take ctx get the
    live voice session; return_direct tools speak their result and end
    the agent's turn.
    """

    from langchain_core.tools import StructuredTool

    converted = []

    for t in tools:

        def run(_tool=t, **arguments):
            context = voice_context()
            arguments, error = _tool.validate(arguments)

            if error:
                return f"Error: {error}"

            result = str(_tool(ctx=context, **arguments))

            if _tool.return_direct:
                context.say(result)

            return result

        converted.append(StructuredTool.from_function(
            func=run,
            name=t.name,
            description=t.description,
            args_schema=t.json_schema(),
            return_direct=t.return_direct,
        ))

    return converted
