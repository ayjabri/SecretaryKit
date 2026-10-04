"""Shared test doubles: microphone, speaker, LLM server, turn context."""

import json
from types import SimpleNamespace

import httpx

from tests.test_llm_client import (  # noqa: F401  (re-exported)
    FakeServer,
    ollama_lines,
    ollama_tool_call_lines,
    openai_lines,
)


class FakeRecognizer:
    """Yields scripted utterances from listen()."""

    def __init__(self, prompts, log=None):
        self.prompts = list(prompts)
        self.log = log if log is not None else []
        self.on_speech_start = None
        self.on_heard = None
        self.config = SimpleNamespace(hotwords=None)

    def load(self):
        self.log.append("recognizer.load")

    def watch(self, on_heard):
        self.on_heard = on_heard

    def listen(self):
        for text in self.prompts:
            yield {"text": text}

    def pause(self):
        self.log.append("mute")

    def resume(self):
        self.log.append("unmute")

    def close(self):
        pass


class FakeSpeaker:
    """Records what would be spoken; speak_stream consumes the stream."""

    def __init__(self, log=None, on_token=None):
        self.log = log if log is not None else []
        self.on_token = on_token
        self.spoken = []
        self.generation = 0
        self.is_speaking = False
        self.config = SimpleNamespace(speed=1.0)

    def load(self):
        self.log.append("speaker.load")

    def speak(self, text, wait=False):
        self.log.append(f"say:{text}")
        self.spoken.append(text)
        return True

    def speak_stream(self, tokens, wait=True):
        generation = self.generation
        text = ""
        try:
            for token in tokens:
                if self.generation != generation:
                    return False
                text += token
                if self.on_token:
                    self.on_token(token)
        finally:
            close = getattr(tokens, "close", None)
            if close:
                close()
        if text.strip():
            self.log.append(f"stream:{text}")
            self.spoken.append(" ".join(text.split()))
        return self.generation == generation

    def stop(self):
        self.generation += 1
        self.log.append("stop")

    def wait(self, timeout=None):
        return True

    def close(self):
        pass


class RecordingContext:
    """A TurnContext for testing agents without a voice session."""

    def __init__(self, answers=(), cancel_after=None):
        self.events = []
        self.said = []
        self.asked = []
        self.answers = list(answers)
        self._cancelled = False
        self.session = SimpleNamespace(
            change_speed=lambda d: f"speed {d}",
            reset_conversation=lambda: "Starting a new conversation.",
            request_stop=lambda: "Goodbye!",
            status_text=lambda: "All good.",
        )

    @property
    def cancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True

    def emit(self, event):
        self.events.append(event)

    def say(self, text):
        self.said.append(text)

    def ask(self, question):
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else None

    def confirm(self, question):
        from voicekit.tools import classify_confirmation
        return classify_confirmation(self.ask(question) or "") is True

    def of_type(self, cls):
        return [e for e in self.events if isinstance(e, cls)]


class OllamaServer(FakeServer):
    """FakeServer that also answers /api/show with capabilities."""

    def __init__(self, replies=(), models=("qwen3.5:4b", "qwen3.6:27b", "llama3.1:8b"),
                 capabilities=None):
        super().__init__(replies, models)
        self.capabilities = capabilities or {
            "qwen3.5:4b": ["completion", "tools", "thinking"],
            "qwen3.6:27b": ["completion", "tools", "thinking"],
            "llama3.1:8b": ["completion"],
        }

    def __call__(self, request):
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            self.requests.append((request.url.path, {"model": name}))
            return httpx.Response(200, json={"capabilities": self.capabilities.get(name, [])})
        return super().__call__(request)


def collect_text(events):
    """Join the spoken text from an agent's output."""
    from voicekit.agent import TextDelta

    return "".join(
        e if isinstance(e, str) else e.text
        for e in events
        if isinstance(e, (str, TextDelta))
    )
