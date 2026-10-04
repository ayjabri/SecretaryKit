"""
voicekit: talk to any AI agent by voice, on your own hardware.

    from voicekit import VoiceSession
    from voicekit.agents import OllamaAgent

    VoiceSession(OllamaAgent()).run()

Building blocks:

    VoiceSession                  the voice loop (listen -> agent -> speak)
    Agent, FunctionAgent          the interface any "brain" implements
    TextDelta, Status, Thinking,  events an agent yields
    ToolStarted, ToolFinished
    Tool, tool, ToolSet           tools a model can decide to call
    voicekit.agents               OllamaAgent, ClaudeAgent, LangGraphAgent,
                                  LangChainAgent
    ContinuousSpeechRecognizer    Whisper speech-to-text
    KokoroSpeaker                 Kokoro text-to-speech
    LLMClient                     streaming Ollama / OpenAI-compatible client

Heavy dependencies (Whisper, torch, Kokoro, LangGraph, Anthropic) load only
when first used, so importing the package is quick.
"""

from .agent import (
    Agent,
    AgentError,
    Event,
    FunctionAgent,
    Status,
    TextDelta,
    Thinking,
    ToolFinished,
    ToolStarted,
    TurnContext,
)
from .commands import NOT_HANDLED, CommandRouter
from .echo import EchoCanceller
from .llm_client import (
    VOICE_SYSTEM_PROMPT,
    LLMClient,
    LLMConfig,
    LLMError,
    ToolCall,
)
from .session import SessionConfig, VoiceSession, control_tools, print_event
from .speech_recognizer import (
    ContinuousSpeechRecognizer,
    SpeechRecognizerConfig,
    clean_text,
)
from .text_to_speech import (
    KOKORO_SAMPLE_RATE,
    KokoroSpeaker,
    SentenceSplitter,
    TextToSpeechConfig,
    prepare_for_speech,
)
from .tools import Tool, ToolSet, classify_confirmation, run_tool_calls, tool

__all__ = [
    "VoiceSession", "SessionConfig", "control_tools", "print_event",
    "Agent", "AgentError", "FunctionAgent", "TurnContext", "Event",
    "TextDelta", "Status", "Thinking", "ToolStarted", "ToolFinished",
    "Tool", "tool", "ToolSet", "run_tool_calls", "classify_confirmation",
    "CommandRouter", "NOT_HANDLED",
    "LLMClient", "LLMConfig", "LLMError", "ToolCall", "VOICE_SYSTEM_PROMPT",
    "ContinuousSpeechRecognizer", "SpeechRecognizerConfig", "clean_text",
    "KokoroSpeaker", "TextToSpeechConfig", "SentenceSplitter",
    "prepare_for_speech", "KOKORO_SAMPLE_RATE", "EchoCanceller",
]
