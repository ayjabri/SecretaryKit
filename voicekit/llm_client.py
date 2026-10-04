"""
Streaming chat client for a self-hosted LLM: Ollama, or any server with an
OpenAI-compatible /v1/chat/completions endpoint (vLLM, LM Studio,
llama.cpp server, LiteLLM, most agent frameworks).

Keeps the conversation history and streams the reply token by token so
speech can start before the model has finished.

Example
-------
    llm = LLMClient(LLMConfig(base_url="http://10.0.0.100:11434",
                              model="qwen3.5:4b"))
    for text in llm.stream_reply("What's the capital of France?"):
        print(text, end="", flush=True)
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import httpx


logger = logging.getLogger(__name__)


VOICE_SYSTEM_PROMPT = """\
You are a helpful voice assistant. The user speaks to you through a \
speech-to-text system and hears your replies through text-to-speech.

- Reply in natural, conversational spoken English.
- Keep replies short: one to three sentences unless the user asks for \
detail.
- Never use markdown, bullet points, numbered lists, tables, headings, \
emojis or special symbols. Write numbers, units and symbols the way they \
are spoken.
- If the user asks for code, describe it briefly in words; only include \
a code block if they explicitly ask to see the code.
- The user's words were transcribed automatically and may contain \
recognition errors. Interpret them sensibly and ask a short clarifying \
question if the meaning is unclear.\
"""


class LLMError(RuntimeError):
    """The LLM server could not be reached or returned an error."""


@dataclass
class LLMConfig:
    """
    Configuration for LLMClient.

    base_url
        Server address. Ollama listens on port 11434 by default.

    api
        "ollama" for Ollama's native API, or "openai" for any
        OpenAI-compatible server (base_url without the /v1 suffix).

    model
        Model name as the server knows it, e.g. "qwen3.5:4b".

    api_key
        Bearer token, if the server requires one.

    system_prompt
        Instructions sent at the start of every conversation.
        The default keeps replies short and free of markdown.

    temperature
        Sampling temperature.

    think
        Let reasoning models (qwen3, deepseek-r1) think before replying.
        More accurate on hard questions but adds seconds of silence.
        Ollama only; ignored for "openai".

    max_history_messages
        Older messages are dropped beyond this count (the system prompt
        is always kept).

    keep_alive
        How long Ollama keeps the model in GPU memory after a request.

    connect_timeout, read_timeout
        Seconds to wait for a connection, and between streamed tokens.

    options
        Extra model options, e.g. {"num_ctx": 8192} for Ollama.
    """

    base_url: str = "http://10.0.0.100:11434"
    api: str = "ollama"
    model: str = "qwen3.5:4b"
    api_key: Optional[str] = None
    system_prompt: Optional[str] = VOICE_SYSTEM_PROMPT
    temperature: float = 0.7
    think: bool = False
    max_history_messages: int = 40
    keep_alive: str = "30m"
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")

        if self.api not in ("ollama", "openai"):
            raise ValueError("api must be 'ollama' or 'openai'")

        if self.max_history_messages < 2:
            raise ValueError("max_history_messages must be >= 2")


# How speech recognition tends to spell model names.
_SPOKEN_ALIASES = {
    "quinn": "qwen", "quin": "qwen", "quen": "qwen", "kwen": "qwen",
    "queen": "qwen", "qwin": "qwen", "lama": "llama", "jemma": "gemma",
    "gemma's": "gemma", "mistrel": "mistral", "fi": "phi", "fie": "phi",
    "deep seek": "deepseek",
}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fourteen": "14",
    "sixteen": "16", "twenty": "20", "thirty": "30", "seventy": "70",
    "point": ".", "dot": ".",
}


def _spoken_to_name(spoken):
    """'Quinn three point five' -> 'qwen 3 . 5'."""

    text = spoken.lower()

    for heard, meant in _SPOKEN_ALIASES.items():
        text = re.sub(rf"\b{re.escape(heard)}\b", meant, text)

    return " ".join(_NUMBER_WORDS.get(word, word) for word in text.split())


def _compact_name(name):
    """
    Reduce a model name to letters and digits for loose comparison:
    "qwen3.6:27b" and the spoken "Qwen 3.6, 27B" both become "qwen3627b".
    ":latest" is ignored.
    """
    name = re.sub(r":latest$", "", name.lower().strip())
    return re.sub(r"[^a-z0-9]", "", name)


class _ThinkFilter:
    """
    Removes <think>...</think> blocks from a token stream, for servers
    that put reasoning inline in the reply.
    """

    def __init__(self):
        self._buffer = ""
        # None until we know whether the reply starts with <think>.
        self._thinking = None
        self._after_think = False

    def feed(self, text):

        self._buffer += text

        if self._thinking is None:
            stripped = self._buffer.lstrip()

            # Only a reply that *starts* with <think> is reasoning;
            # wait until there is enough text to tell.
            if len(stripped) < len("<think>") and "<think>".startswith(stripped):
                return ""

            self._thinking = stripped.startswith("<think>")

        if self._thinking:
            end = self._buffer.find("</think>")

            if end == -1:
                return ""

            self._buffer = self._buffer[end + len("</think>"):]
            self._thinking = False
            self._after_think = True

        if self._after_think:
            # Drop the blank lines between the reasoning and the reply,
            # which may arrive in later tokens.
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                return ""
            self._after_think = False

        out, self._buffer = self._buffer, ""
        return out

    def flush(self):
        out, self._buffer = self._buffer, ""
        # Unterminated reasoning is never spoken.
        return "" if self._thinking else out


class LLMClient:

    def __init__(self, config=None, transport=None):
        """
        transport
            Optional httpx transport, used by tests to fake the server.
        """

        self.config = config or LLMConfig()

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._http = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(
                self.config.read_timeout,
                connect=self.config.connect_timeout,
            ),
            transport=transport,
        )

        self.history = []

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def reset(self):
        """Forget the conversation."""
        self.history = []

    # --------------------------------------------------
    # Server checks
    # --------------------------------------------------

    def list_models(self):

        try:
            if self.config.api == "ollama":
                response = self._http.get("/api/tags")
                response.raise_for_status()
                return [m["name"] for m in response.json()["models"]]

            response = self._http.get("/v1/models")
            response.raise_for_status()
            return [m["id"] for m in response.json()["data"]]

        except httpx.HTTPError as exc:
            raise LLMError(
                f"Cannot reach the LLM server at {self.config.base_url}: {exc}"
            ) from exc

    def check_connection(self):
        """Raise LLMError unless the server is up and has the model."""

        models = self.list_models()

        if self.config.model not in models:
            raise LLMError(
                f"Model '{self.config.model}' is not available on "
                f"{self.config.base_url}. Available: {', '.join(models)}"
            )

    def model_capabilities(self, model=None):
        """
        Capabilities Ollama reports for a model, e.g. ["completion",
        "tools", "thinking"]. None if unknown (OpenAI-compatible servers
        don't report them).
        """

        if self.config.api != "ollama":
            return None

        try:
            response = self._http.post(
                "/api/show", json={"model": model or self.config.model}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot get model details: {exc}") from exc

        return response.json().get("capabilities")

    def supports_thinking(self, model=None):
        """True, False, or None if the server can't tell."""

        capabilities = self.model_capabilities(model)
        return None if capabilities is None else "thinking" in capabilities

    def resolve_model(self, spoken_name, models=None):
        """
        Find the model the user meant from a loosely transcribed name,
        e.g. "Qwen 3.6 27B" -> "qwen3.6:27b".

        Returns the model name, or None if there is no single good match.
        """

        models = self.list_models() if models is None else models
        target = _compact_name(_spoken_to_name(spoken_name))

        if not target:
            return None

        compact = {model: _compact_name(model) for model in models}

        exact = [m for m, c in compact.items() if c == target]
        if exact:
            return min(exact, key=len)

        # "qwen 3.5" -> "qwen3.5:4b" when it is the only qwen3.5 model.
        prefixed = [m for m, c in compact.items() if c.startswith(target)]
        if len(prefixed) == 1:
            return prefixed[0]

        close = difflib.get_close_matches(
            target, list(compact.values()), n=2, cutoff=0.75
        )
        if len(close) == 1 or (
            len(close) == 2
            and difflib.SequenceMatcher(None, target, close[0]).ratio()
            > difflib.SequenceMatcher(None, target, close[1]).ratio() + 0.05
        ):
            return next(m for m, c in compact.items() if c == close[0])

        return None

    def warm_up(self):
        """Ask Ollama to load the model now so the first reply is fast."""

        if self.config.api != "ollama":
            return

        try:
            self._http.post(
                "/api/generate",
                json={
                    "model": self.config.model,
                    "keep_alive": self.config.keep_alive,
                },
            ).raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Could not preload model: %s", exc)

    # --------------------------------------------------
    # Chat
    # --------------------------------------------------

    def stream_reply(self, user_text, on_thinking=None) -> Iterator[str]:
        """
        Send a user message and yield the reply as it streams in.

        on_thinking
            Optional callable receiving the model's reasoning text as it
            streams (when `think` is on). Reasoning is never yielded as
            part of the reply or kept in the history.

        The exchange is added to the history when the stream ends. If the
        caller stops early (e.g. the user interrupted), the partial reply
        is kept so the model knows what was actually said.

        Raises LLMError on connection or server errors; the user message
        is then removed from the history so it can be retried.
        """

        user_message = {"role": "user", "content": user_text}
        self.history.append(user_message)
        self._trim_history()

        messages = list(self.history)
        if self.config.system_prompt:
            messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

        def thinking(text):
            if on_thinking is None:
                return
            try:
                on_thinking(text)
            except Exception:
                logger.exception("on_thinking callback failed")

        stream = (
            self._stream_ollama(messages, thinking)
            if self.config.api == "ollama"
            else self._stream_openai(messages, thinking)
        )

        think_filter = _ThinkFilter()
        reply = []
        completed = False
        failed = False

        try:
            for token in stream:
                text = think_filter.feed(token)
                if text:
                    reply.append(text)
                    yield text

            text = think_filter.flush()
            if text:
                reply.append(text)
                yield text

            completed = True

        except httpx.HTTPError as exc:
            failed = True
            raise LLMError(f"LLM request failed: {exc}") from exc

        except Exception:
            failed = True
            raise

        finally:
            stream.close()

            content = "".join(reply).strip()

            if self.history and self.history[-1] is user_message:
                if failed or not content:
                    # Forget the question so it can simply be asked again.
                    self.history.pop()
                else:
                    if not completed:
                        content += " ..."
                    self.history.append(
                        {"role": "assistant", "content": content}
                    )

    def chat(self, user_text):
        """Send a message and return the complete reply."""
        return "".join(self.stream_reply(user_text)).strip()

    def _trim_history(self):

        excess = len(self.history) - self.config.max_history_messages

        if excess > 0:
            self.history = self.history[excess:]

        # Never start the history with an orphaned assistant message.
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

    # --------------------------------------------------
    # Protocols
    # --------------------------------------------------

    def _stream_ollama(self, messages, on_thinking):

        body = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "think": self.config.think,
            "keep_alive": self.config.keep_alive,
            "options": {"temperature": self.config.temperature,
                        **self.config.options},
        }

        with self._http.stream("POST", "/api/chat", json=body) as response:
            self._raise_for_status(response)

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)

                if "error" in data:
                    raise LLMError(f"Ollama error: {data['error']}")

                message = data.get("message", {})

                if message.get("thinking"):
                    on_thinking(message["thinking"])

                if message.get("content"):
                    yield message["content"]

                if data.get("done"):
                    return

    def _stream_openai(self, messages, on_thinking):

        body = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            **self.config.options,
        }

        with self._http.stream(
            "POST", "/v1/chat/completions", json=body
        ) as response:
            self._raise_for_status(response)

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()

                if payload == "[DONE]":
                    return

                data = json.loads(payload)

                if "error" in data:
                    raise LLMError(f"Server error: {data['error']}")

                for choice in data.get("choices", []):
                    delta = choice.get("delta") or {}

                    # vLLM / DeepSeek style reasoning fields.
                    reasoning = delta.get("reasoning_content") or delta.get(
                        "reasoning"
                    )
                    if reasoning:
                        on_thinking(reasoning)

                    if delta.get("content"):
                        yield delta["content"]

    @staticmethod
    def _raise_for_status(response):

        if response.is_success:
            return

        response.read()

        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text

        raise LLMError(f"HTTP {response.status_code}: {detail}")
