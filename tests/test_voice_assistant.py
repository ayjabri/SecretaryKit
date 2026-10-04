"""
Tests for voice_assistant.py (the example app).

    python tests/test_voice_assistant.py           # fast tests
    python tests/test_voice_assistant.py --e2e     # + real models and LLM server

The --e2e test is a loopback: Kokoro speaks a question, Whisper
transcribes it, the agent answers and Kokoro voices the answer. Nothing
is played or recorded, so it runs unattended.
"""

import os
import sys
import time
import unittest

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _take_flag(flag):
    # Read before the test classes are defined, since the skip
    # decorators are evaluated at import time.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_E2E_TESTS = _take_flag("--e2e") or os.environ.get("RUN_E2E_TESTS") == "1"

from voice_assistant import build_agent, parse_args
from voicekit.agents import ClaudeAgent, OllamaAgent


class CommandLineTests(unittest.TestCase):

    def test_defaults(self):
        args = parse_args([])

        self.assertEqual(args.backend, "ollama")
        self.assertEqual(args.host, "http://10.0.0.100:11434")
        self.assertIsNone(args.model)
        self.assertFalse(args.interruptible)

    def test_options(self):
        args = parse_args(["--model", "qwen3.6:27b", "--voice", "bf_emma",
                           "--interruptible", "--pause", "2", "--no-tools"])

        self.assertEqual(args.model, "qwen3.6:27b")
        self.assertEqual(args.voice, "bf_emma")
        self.assertTrue(args.interruptible)
        self.assertTrue(args.no_tools)
        self.assertEqual(args.pause, 2.0)


class BuildAgentTests(unittest.TestCase):

    def test_ollama_default(self):
        agent = build_agent(parse_args([]))

        self.assertIsInstance(agent, OllamaAgent)
        self.assertEqual(agent.llm.config.model, "qwen3.5:4b")
        self.assertEqual(agent.llm.config.api, "ollama")
        self.assertIn("switch_model", agent.tools)

    def test_openai_compatible(self):
        agent = build_agent(parse_args(["--backend", "openai", "--host", "http://h:8000",
                                        "--model", "my-agent", "--api-key", "k"]))

        config = agent.llm.config
        self.assertEqual((config.api, config.base_url, config.model, config.api_key),
                         ("openai", "http://h:8000", "my-agent", "k"))

    def test_think_and_tool_flags(self):
        agent = build_agent(parse_args(["--think", "--no-tools", "--confirm-tools"]))

        self.assertTrue(agent.llm.config.think)
        self.assertFalse(agent.use_tools)
        self.assertTrue(agent.confirm_tools)

    def test_claude(self):
        agent = build_agent(parse_args(["--backend", "claude", "--effort", "medium"]))

        self.assertIsInstance(agent, ClaudeAgent)
        self.assertEqual(agent.model, "claude-opus-5-5")
        self.assertEqual(agent.effort, "medium")

    def test_system_prompt_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("You are a pirate.")

        try:
            agent = build_agent(parse_args(["--system-prompt-file", f.name]))
        finally:
            os.unlink(f.name)

        self.assertEqual(agent.llm.config.system_prompt, "You are a pirate.")


@unittest.skipUnless(RUN_E2E_TESTS, "pass --e2e to run")
class EndToEndTests(unittest.TestCase):

    def test_spoken_question_gets_spoken_answer(self):
        from voicekit import KOKORO_SAMPLE_RATE, ContinuousSpeechRecognizer, KokoroSpeaker
        from tests.fakes import RecordingContext, collect_text

        speaker = KokoroSpeaker()
        recognizer = ContinuousSpeechRecognizer()
        agent = OllamaAgent()

        try:
            start = time.perf_counter()
            agent.start()
            speaker.load()
            recognizer.load()
            print(f"\n  start-up: {time.perf_counter() - start:.1f}s")

            audio = speaker.synthesize("What is the capital city of France?")

            start = time.perf_counter()
            heard = recognizer.transcribe_audio(audio, KOKORO_SAMPLE_RATE)["text"]
            print(f"  heard:  {heard!r} ({time.perf_counter() - start:.2f}s)")
            self.assertIn("capital", heard.lower())

            start = time.perf_counter()
            reply = collect_text(agent.respond(heard, RecordingContext()))
            print(f"  reply:  {reply.strip()!r} ({time.perf_counter() - start:.2f}s)")
            self.assertIn("paris", reply.lower())

            start = time.perf_counter()
            spoken = speaker.synthesize(reply)
            print(f"  voiced: {spoken.size / KOKORO_SAMPLE_RATE:.1f}s of speech "
                  f"in {time.perf_counter() - start:.2f}s")
            self.assertGreater(spoken.size, 0)

        finally:
            agent.close()
            speaker.close()
            recognizer.close()


if __name__ == "__main__":
    unittest.main()
