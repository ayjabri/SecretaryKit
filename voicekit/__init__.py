"""
voicekit: local speech-to-text, text-to-speech and LLM chat building
blocks for voice interfaces.

    from voicekit import (
        ContinuousSpeechRecognizer, SpeechRecognizerConfig,  # Whisper
        KokoroSpeaker, TextToSpeechConfig,                   # Kokoro
        LLMClient, LLMConfig, LLMError,                      # Ollama / OpenAI API
    )

Heavy dependencies (Whisper, torch, Kokoro) load only when a model is
first used, so importing the package is quick.
"""

from .llm_client import VOICE_SYSTEM_PROMPT, LLMClient, LLMConfig, LLMError
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

__all__ = [
    "ContinuousSpeechRecognizer",
    "SpeechRecognizerConfig",
    "clean_text",
    "KokoroSpeaker",
    "TextToSpeechConfig",
    "SentenceSplitter",
    "prepare_for_speech",
    "KOKORO_SAMPLE_RATE",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "VOICE_SYSTEM_PROMPT",
]
