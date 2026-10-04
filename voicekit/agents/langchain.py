"""
Talk to any LangChain Runnable: a chat model, a prompt | model chain, or
anything with .stream().

    from langchain_ollama import ChatOllama
    agent = LangChainAgent(ChatOllama(model="qwen3.5:4b",
                                      base_url="http://10.0.0.100:11434"))
    VoiceSession(agent).run()

The agent keeps the conversation and passes it as a list of messages. For
models that call tools, use LangGraphAgent with create_react_agent(), which
runs the tool loop.
"""

from typing import Callable, Optional

from ..agent import Agent, AgentError, TextDelta
from ..llm_client import VOICE_SYSTEM_PROMPT
from .langgraph import message_text


class LangChainAgent(Agent):
    """
    runnable
        Anything with .stream(input).

    system_prompt
        Sent first on every turn (None to omit).

    input_builder
        fn(messages) -> runnable input, for chains that expect a dict
        (e.g. lambda m: {"history": m[:-1], "question": m[-1].content}).
        Default: the list of messages.
    """

    def __init__(self, runnable, system_prompt: Optional[str] = VOICE_SYSTEM_PROMPT,
                 input_builder: Optional[Callable] = None, description: str = ""):
        self.runnable = runnable
        self.system_prompt = system_prompt
        self.input_builder = input_builder
        self.description = description
        self.history = []

    def respond(self, text, ctx):

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = list(self.history) + [HumanMessage(content=text)]

        if self.system_prompt:
            messages.insert(0, SystemMessage(content=self.system_prompt))

        payload = self.input_builder(messages) if self.input_builder else messages
        reply = []
        completed = False

        try:
            for chunk in self.runnable.stream(payload):
                if ctx.cancelled:
                    return

                piece = _chunk_text(chunk)

                if piece:
                    reply.append(piece)
                    yield TextDelta(piece)

            completed = True

        except Exception as exc:
            if isinstance(exc, AgentError):
                raise
            raise AgentError(f"The chain failed: {exc}") from exc

        finally:
            content = "".join(reply).strip()
            if content:
                self.history.append(HumanMessage(content=text))
                self.history.append(
                    AIMessage(content=content if completed else content + " ...")
                )

    def reset(self):
        self.history = []

    def describe(self):
        return self.description


def _chunk_text(chunk):

    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", None)

    if content is not None:
        return message_text(content)

    if isinstance(chunk, dict):
        for key in ("output", "answer", "text", "content"):
            if isinstance(chunk.get(key), str):
                return chunk[key]

    return ""
