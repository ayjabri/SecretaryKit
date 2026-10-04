"""
Tests for llm_client.py.

    python tests/test_llm_client.py            # fast tests (fake server)
    python tests/test_llm_client.py --server   # + your real Ollama server

The real server and model can be changed with LLM_HOST and LLM_MODEL,
e.g. LLM_HOST=http://10.0.0.100:11434 LLM_MODEL=qwen3.5:4b.
RUN_SERVER_TESTS=1 is equivalent to --server.
"""

import json
import os
import sys
import time
import unittest

import httpx

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit.llm_client import LLMClient, LLMConfig, LLMError, _ThinkFilter


def _take_flag(flag):
    # Read before the test classes are defined, since the skip
    # decorators are evaluated at import time.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_SERVER_TESTS = (
    _take_flag("--server") or os.environ.get("RUN_SERVER_TESTS") == "1"
)


# --------------------------------------------------
# Fake servers
# --------------------------------------------------

def ollama_lines(*tokens, error=None):
    lines = [
        json.dumps({"message": {"role": "assistant", "content": t}, "done": False})
        for t in tokens
    ]
    if error:
        lines.append(json.dumps({"error": error}))
    else:
        lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return "\n".join(lines) + "\n"


def openai_lines(*tokens):
    events = [
        "data: " + json.dumps({"choices": [{"delta": {"content": t}}]})
        for t in tokens
    ]
    return "\n\n".join(events + ["data: [DONE]"]) + "\n\n"


class FakeServer:
    """httpx transport that records requests and replays canned replies."""

    def __init__(self, replies=(), models=("qwen3.5:4b",)):
        self.replies = list(replies)
        self.models = list(models)
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        path = request.url.path

        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})

        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})

        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})

        reply = self.replies.pop(0)

        if isinstance(reply, httpx.Response):
            return reply

        return httpx.Response(200, text=reply)

    def chat_bodies(self):
        return [body for path, body in self.requests if "chat" in path]


def make_client(server, **config):
    return LLMClient(LLMConfig(**config), transport=httpx.MockTransport(server))


# --------------------------------------------------
# Ollama protocol
# --------------------------------------------------

class OllamaTests(unittest.TestCase):

    def test_streams_tokens(self):
        server = FakeServer([ollama_lines("Hello", " there", "!")])
        client = make_client(server)

        self.assertEqual(list(client.stream_reply("Hi")), ["Hello", " there", "!"])

    def test_request_body(self):
        server = FakeServer([ollama_lines("ok")])
        client = make_client(server, model="qwen3.6:27b", temperature=0.2,
                             options={"num_ctx": 8192}, system_prompt="Be brief.")

        client.chat("Hi")

        [body] = server.chat_bodies()
        self.assertEqual(body["model"], "qwen3.6:27b")
        self.assertTrue(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["options"], {"temperature": 0.2, "num_ctx": 8192})
        self.assertEqual(body["messages"], [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ])

    def test_keeps_conversation_history(self):
        server = FakeServer([ollama_lines("Paris."), ollama_lines("About 2 million.")])
        client = make_client(server, system_prompt=None)

        client.chat("Capital of France?")
        client.chat("Population?")

        self.assertEqual(server.chat_bodies()[1]["messages"], [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "Population?"},
        ])

    def test_reset_clears_history(self):
        server = FakeServer([ollama_lines("A"), ollama_lines("B")])
        client = make_client(server, system_prompt=None)

        client.chat("one")
        client.reset()
        client.chat("two")

        self.assertEqual(len(server.chat_bodies()[1]["messages"]), 1)

    def test_interrupted_reply_kept_as_partial(self):
        server = FakeServer([ollama_lines("One.", " Two.", " Three.")])
        client = make_client(server)

        stream = client.stream_reply("Count")
        next(stream)
        stream.close()

        self.assertEqual(client.history[-1],
                         {"role": "assistant", "content": "One. ..."})

    def test_interrupted_before_any_text_forgets_question(self):
        server = FakeServer([ollama_lines("Hello")])
        client = make_client(server)

        stream = client.stream_reply("Hi")
        stream.close()  # never started

        self.assertEqual(client.history, [])

    def test_server_error_line_raises_and_forgets_question(self):
        server = FakeServer([ollama_lines("Par", error="model crashed")])
        client = make_client(server)

        with self.assertRaisesRegex(LLMError, "model crashed"):
            client.chat("Capital of France?")

        self.assertEqual(client.history, [])

    def test_http_error_raises(self):
        server = FakeServer([httpx.Response(404, json={"error": "model 'x' not found"})])
        client = make_client(server)

        with self.assertRaisesRegex(LLMError, "404.*not found"):
            client.chat("Hi")

        self.assertEqual(client.history, [])

    def test_connection_error_raises_llm_error(self):
        def refuse(request):
            raise httpx.ConnectError("connection refused")

        client = LLMClient(transport=httpx.MockTransport(refuse))

        with self.assertRaisesRegex(LLMError, "refused"):
            client.chat("Hi")

        with self.assertRaisesRegex(LLMError, "Cannot reach"):
            client.check_connection()

    def test_check_connection_reports_missing_model(self):
        client = make_client(FakeServer(models=["llama3:8b"]), model="qwen3.5:4b")

        with self.assertRaisesRegex(LLMError, "qwen3\.5:4b.*llama3:8b"):
            client.check_connection()

    def test_history_trimmed(self):
        server = FakeServer([ollama_lines(f"reply {i}") for i in range(5)])
        client = make_client(server, max_history_messages=4, system_prompt=None)

        for i in range(5):
            client.chat(f"question {i}")

        messages = server.chat_bodies()[-1]["messages"]
        self.assertLessEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[-1]["content"], "question 4")

    def test_inline_think_block_removed(self):
        server = FakeServer([ollama_lines("<thi", "nk>planning", "...</think>", "\n\nAnswer.")])
        client = make_client(server)

        self.assertEqual(client.chat("Q"), "Answer.")
        self.assertEqual(client.history[-1]["content"], "Answer.")


class ThinkingTests(unittest.TestCase):

    def ollama_thinking_reply(self):
        lines = [
            {"message": {"role": "assistant", "content": "", "thinking": "Let me "}},
            {"message": {"role": "assistant", "content": "", "thinking": "compute."}},
            {"message": {"role": "assistant", "content": "391."}},
            {"message": {"content": ""}, "done": True},
        ]
        return "\n".join(json.dumps(line) for line in lines) + "\n"

    def test_thinking_sent_to_callback_not_reply(self):
        server = FakeServer([self.ollama_thinking_reply()])
        client = make_client(server, think=True)
        thoughts = []

        reply = "".join(client.stream_reply("17*23?", on_thinking=thoughts.append))

        self.assertEqual(reply, "391.")
        self.assertEqual("".join(thoughts), "Let me compute.")
        self.assertTrue(server.chat_bodies()[0]["think"])
        self.assertEqual(client.history[-1]["content"], "391.")

    def test_thinking_ignored_without_callback(self):
        client = make_client(FakeServer([self.ollama_thinking_reply()]))

        self.assertEqual(client.chat("17*23?"), "391.")

    def test_failing_callback_does_not_break_reply(self):
        client = make_client(FakeServer([self.ollama_thinking_reply()]))

        def broken(text):
            raise ValueError("display failed")

        with self.assertLogs("voicekit.llm_client", "ERROR"):
            reply = "".join(client.stream_reply("Q", on_thinking=broken))

        self.assertEqual(reply, "391.")

    def test_openai_reasoning_content(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"content": "Yes."}}]},
        ]
        body = "\n\n".join("data: " + json.dumps(e) for e in events) + "\n\ndata: [DONE]\n\n"
        client = make_client(FakeServer([body]), api="openai")
        thoughts = []

        reply = "".join(client.stream_reply("Q", on_thinking=thoughts.append))

        self.assertEqual((reply, thoughts), ("Yes.", ["hmm"]))


class ModelInfoTests(unittest.TestCase):

    MODELS = ["qwen3.6:27b", "qwen3.5:4b", "qwen3:14b",
              "qwen2.5-coder:latest", "qwen2.5-coder:7b", "llama3.1:8b"]

    def resolve(self, spoken):
        return LLMClient().resolve_model(spoken, self.MODELS)

    def test_resolve_spoken_model_names(self):
        cases = {
            "Qwen 3.6 27B": "qwen3.6:27b",
            "qwen 3 6 27b": "qwen3.6:27b",       # after command normalisation
            "Quinn three point five": "qwen3.5:4b",
            "Qwen 3 fourteen b": "qwen3:14b",
            "Qwen 2.5 coder": "qwen2.5-coder:latest",
            "Llama 3.1": "llama3.1:8b",
            "llama": "llama3.1:8b",
        }
        for spoken, expected in cases.items():
            with self.subTest(spoken=spoken):
                self.assertEqual(self.resolve(spoken), expected)

    def test_unknown_names_do_not_match(self):
        for spoken in ("mistral", "Python to sort a list", "the big one", ""):
            with self.subTest(spoken=spoken):
                self.assertIsNone(self.resolve(spoken))

    def test_capabilities(self):
        def server(request):
            name = json.loads(request.content)["model"]
            caps = ["completion", "thinking"] if "qwen3" in name else ["completion"]
            return httpx.Response(200, json={"capabilities": caps})

        client = LLMClient(transport=httpx.MockTransport(server))

        self.assertTrue(client.supports_thinking("qwen3.5:4b"))
        self.assertFalse(client.supports_thinking("llama3.1:8b"))

    def test_capabilities_unknown_for_openai_servers(self):
        client = LLMClient(LLMConfig(api="openai"))

        self.assertIsNone(client.supports_thinking())


# --------------------------------------------------
# OpenAI-compatible protocol
# --------------------------------------------------

class OpenAITests(unittest.TestCase):

    def test_streams_tokens_and_sends_auth(self):
        seen = {}
        server = FakeServer([openai_lines("Hi", " there")])

        def transport(request):
            seen["auth"] = request.headers.get("authorization")
            return server(request)

        client = LLMClient(
            LLMConfig(api="openai", base_url="http://agent:8000/", api_key="secret",
                      model="my-agent"),
            transport=httpx.MockTransport(transport),
        )

        self.assertEqual(client.chat("Hello"), "Hi there")
        self.assertEqual(seen["auth"], "Bearer secret")

        path, body = server.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "my-agent")
        self.assertNotIn("think", body)

    def test_check_connection_uses_models_endpoint(self):
        client = make_client(FakeServer(models=["my-agent"]), api="openai",
                             model="my-agent")

        client.check_connection()


# --------------------------------------------------
# Think filter
# --------------------------------------------------

class ThinkFilterTests(unittest.TestCase):

    def run_filter(self, tokens):
        f = _ThinkFilter()
        return "".join(f.feed(t) for t in tokens) + f.flush()

    def test_passes_normal_text(self):
        self.assertEqual(self.run_filter(["Hello", " world"]), "Hello world")

    def test_removes_leading_think_block(self):
        self.assertEqual(self.run_filter(["<think>", "x", "</think>", " Hi"]), "Hi")

    def test_keeps_think_tag_mentioned_later(self):
        text = "Use the <think> tag."
        self.assertEqual(self.run_filter([text]), text)

    def test_drops_unterminated_think(self):
        self.assertEqual(self.run_filter(["<think>never ends"]), "")

    def test_short_reply_resembling_tag_start(self):
        self.assertEqual(self.run_filter(["<"]), "<")


class ConfigTests(unittest.TestCase):

    def test_validation(self):
        with self.assertRaises(ValueError):
            LLMConfig(api="grpc")
        with self.assertRaises(ValueError):
            LLMConfig(max_history_messages=1)

    def test_trailing_slash_removed(self):
        self.assertEqual(LLMConfig(base_url="http://h:1/").base_url, "http://h:1")


# --------------------------------------------------
# Real server (--server)
# --------------------------------------------------

@unittest.skipUnless(RUN_SERVER_TESTS, "pass --server to run")
class RealServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = LLMConfig()
        cls.client = LLMClient(LLMConfig(
            base_url=os.environ.get("LLM_HOST", config.base_url),
            model=os.environ.get("LLM_MODEL", config.model),
        ))

        # Fall back to the smallest installed model if the default is
        # not on this server (and no LLM_MODEL was given).
        models = cls.client.list_models()
        if "LLM_MODEL" not in os.environ and cls.client.config.model not in models:
            cls.client.config.model = min(models, key=len)

        cls.client.check_connection()
        cls.client.warm_up()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_streams_a_reply_and_remembers_context(self):
        start = time.perf_counter()
        first_token = None
        reply = ""

        for token in self.client.stream_reply("My name is Sam. Just say hello."):
            first_token = first_token or time.perf_counter() - start
            reply += token

        print(f"\n  {self.client.config.model}: first token {first_token:.2f}s, "
              f"reply {reply.strip()!r}")
        self.assertTrue(reply.strip())

        answer = self.client.chat("What is my name? One word.")
        print(f"  memory check: {answer!r}")
        self.assertIn("sam", answer.lower())


if __name__ == "__main__":
    unittest.main()
