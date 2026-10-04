"""
Ready-made agents for VoiceSession. Each lives in its own module and only
imports its framework when used:

    from voicekit.agents import OllamaAgent      # Ollama / OpenAI-compatible
    from voicekit.agents import ClaudeAgent      # Claude (pip install anthropic)
    from voicekit.agents import LangGraphAgent   # any compiled LangGraph graph
    from voicekit.agents import LangChainAgent   # any LangChain Runnable
    from voicekit.agent import FunctionAgent     # any Python function
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # real imports for editors and type checkers only
    from ..agent import FunctionAgent
    from .claude import ClaudeAgent
    from .langchain import LangChainAgent
    from .langgraph import LangGraphAgent, to_langchain_tools, voice_context
    from .ollama import OllamaAgent

_EXPORTS = {
    "OllamaAgent": ".ollama",
    "ClaudeAgent": ".claude",
    "LangGraphAgent": ".langgraph",
    "to_langchain_tools": ".langgraph",
    "voice_context": ".langgraph",
    "LangChainAgent": ".langchain",
}

__all__ = list(_EXPORTS) + ["FunctionAgent"]


def __getattr__(name):
    if name == "FunctionAgent":
        from ..agent import FunctionAgent
        return FunctionAgent

    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)

    raise AttributeError(name)
