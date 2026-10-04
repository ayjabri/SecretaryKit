# voicekit

Talk to a self-hosted LLM with your voice, entirely on your own hardware.

```
microphone → Whisper (speech-to-text) → Ollama / OpenAI-compatible LLM → Kokoro (text-to-speech) → speakers
```

- **`voicekit`**: the library
  - `ContinuousSpeechRecognizer`: listens to the mic and returns one cleaned-up prompt each time you finish speaking (faster-whisper with WebRTC voice detection).
  - `KokoroSpeaker`: speaks text, including LLM replies while they are still streaming in. Markdown is cleaned up first, and `stop()` cuts speech off instantly.
  - `LLMClient`: a streaming chat client for Ollama or any OpenAI-compatible server, with conversation history.
- **`voice_assistant.py`**: a hands-free assistant built from the three components above.

## Install

The library alone:

```bash
pip install "voicekit[cuda] @ git+https://github.com/ayjabri/voicekit.git"
```

For development, from a clone:

```bash
pip install -e ".[cuda]"
```

On a GPU, install a CUDA build of PyTorch first (Kokoro needs it), for example from
https://pytorch.org/get-started/locally/. Leave out the `[cuda]` extra for CPU-only use.

## Voice assistant

```bash
python voice_assistant.py                                   # Ollama at 10.0.0.100, qwen3:8b
python voice_assistant.py --model qwen3:14b --voice am_michael
python voice_assistant.py --interruptible --output 7         # headphones: talk over it to interrupt
python voice_assistant.py --api openai --host http://server:8000
python voice_assistant.py --list-devices
```

Say "goodbye" to quit, or "new conversation" to clear the history.

## Library usage

```python
from voicekit import ContinuousSpeechRecognizer, KokoroSpeaker, LLMClient, LLMConfig

llm = LLMClient(LLMConfig(base_url="http://10.0.0.100:11434", model="qwen3:8b"))

with ContinuousSpeechRecognizer() as ears, KokoroSpeaker() as mouth:
    prompt = ears.listen_once()
    mouth.speak_stream(llm.stream_reply(prompt["text"]))
```

Each class takes a config dataclass (`SpeechRecognizerConfig`, `TextToSpeechConfig`,
`LLMConfig`), and every option is documented in its docstring.

## Tests

```bash
python -m unittest                                   # fast: fake devices, models and server
python tests/test_speech_recognizer.py --gpu --mic   # real Whisper and live microphone
python tests/test_text_to_speech.py --gpu --audio    # real Kokoro, then plays speech
python tests/test_llm_client.py --server             # real LLM server
python tests/test_voice_assistant.py --e2e           # full loopback: speak, transcribe, answer, speak
```
