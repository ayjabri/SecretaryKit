"""
Tests for voice_assistant.py.

    python tests/test_voice_assistant.py           # fast tests (all fakes)
    python tests/test_voice_assistant.py --e2e     # + real models and LLM server

The --e2e test is a loopback: Kokoro speaks a question, Whisper
transcribes it, the LLM answers and Kokoro voices the answer. Nothing is
played or recorded, so it runs unattended.
"""

import os
import sys
import threading
import time
import unittest

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit.llm_client import LLMError
from voice_assistant import AssistantConfig, VoiceAssistant, parse_args


def _take_flag(flag):
    # Read before the test classes are defined, since the skip
    # decorators are evaluated at import time.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_E2E_TESTS = _take_flag("--e2e") or os.environ.get("RUN_E2E_TESTS") == "1"


# --------------------------------------------------
# Fakes
# --------------------------------------------------

class FakeRecognizer:

    def __init__(self, prompts, log):
        self.prompts = list(prompts)
        self.log = log
        self.on_speech_start = None

    def load(self):
        self.log.append("recognizer.load")

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

    def __init__(self, log, on_token=None):
        self.log = log
        self.on_token = on_token
        self.spoken = []
        self.generation = 0
        self.is_speaking = False

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
            tokens.close()
        self.log.append(f"stream:{text}")
        self.spoken.append(text)
        return self.generation == generation

    def stop(self):
        self.generation += 1
        self.log.append("stop")

    def wait(self, timeout=None):
        return True

    def close(self):
        pass


class FakeLLM:

    def __init__(self, replies, log):
        self.replies = list(replies)
        self.log = log
        self.prompts = []
        self.resets = 0
        self.config = type("C", (), {"base_url": "http://fake"})()

    def check_connection(self):
        self.log.append("llm.check")

    def warm_up(self):
        pass

    def stream_reply(self, text):
        self.prompts.append(text)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for word in reply.split(" "):
            yield word + " "

    def reset(self):
        self.resets += 1

    def close(self):
        pass


def make_assistant(prompts, replies=(), **config):
    log = []
    events = []
    config.setdefault("greeting", None)
    config.setdefault("resume_delay_ms", 0)

    recognizer = FakeRecognizer(prompts, log)
    speaker = FakeSpeaker(log)
    llm = FakeLLM(replies, log)

    assistant = VoiceAssistant(
        recognizer, speaker, llm,
        AssistantConfig(**config),
        on_event=lambda kind, text: events.append((kind, text)),
    )
    return assistant, recognizer, speaker, llm, log, events


# --------------------------------------------------
# Tests
# --------------------------------------------------

class ConversationTests(unittest.TestCase):

    def test_prompt_is_answered_aloud(self):
        assistant, _, speaker, llm, _, events = make_assistant(
            ["What is two plus two?"], ["It is four."]
        )

        assistant.run()

        self.assertEqual(llm.prompts, ["What is two plus two?"])
        self.assertEqual([s.strip() for s in speaker.spoken], ["It is four."])
        self.assertIn(("user", "What is two plus two?"), events)
        streamed = "".join(t for k, t in events if k == "assistant_text")
        self.assertEqual(streamed.strip(), "It is four.")

    def test_starts_by_checking_server_and_loading_models(self):
        assistant, _, _, _, log, _ = make_assistant([])

        assistant.run()

        self.assertEqual(log[:3], ["llm.check", "speaker.load", "recognizer.load"])

    def test_greeting_spoken_first(self):
        assistant, _, speaker, _, _, _ = make_assistant([], greeting="Hello!")

        assistant.run()

        self.assertEqual(speaker.spoken, ["Hello!"])

    def test_mic_muted_while_responding(self):
        assistant, _, _, _, log, _ = make_assistant(["Hi"], ["Hello there."])

        assistant.run()

        i = log.index("stream:Hello there. ")
        self.assertEqual(log[i - 1], "mute")
        self.assertEqual(log[i + 1], "unmute")

    def test_mic_stays_on_when_interruptible(self):
        assistant, recognizer, _, _, log, _ = make_assistant(
            ["Hi"], ["Hello."], interruptible=True
        )

        assistant.run()

        self.assertNotIn("mute", log)
        self.assertIsNone(recognizer.on_speech_start, "callback not cleared")


class CommandTests(unittest.TestCase):

    def test_goodbye_ends_conversation(self):
        assistant, _, speaker, llm, _, _ = make_assistant(
            ["Goodbye.", "This is never heard"], ["unused"]
        )

        assistant.run()

        self.assertEqual(llm.prompts, [])
        self.assertEqual(speaker.spoken, ["Goodbye!"])

    def test_goodbye_inside_sentence_is_not_a_command(self):
        assistant, _, _, llm, _, _ = make_assistant(
            ["How do you say goodbye in French?"], ["Au revoir."]
        )

        assistant.run()

        self.assertEqual(llm.prompts, ["How do you say goodbye in French?"])

    def test_new_conversation_resets_history(self):
        assistant, _, speaker, llm, _, _ = make_assistant(["New conversation!"])

        assistant.run()

        self.assertEqual(llm.resets, 1)
        self.assertEqual(speaker.spoken, ["Okay, starting a new conversation."])


class ErrorTests(unittest.TestCase):

    def test_llm_error_is_spoken_and_conversation_continues(self):
        assistant, _, speaker, llm, _, events = make_assistant(
            ["First", "Second"], [LLMError("server down"), "Back online."]
        )

        assistant.run()

        self.assertEqual(speaker.spoken[0], "Sorry, I couldn't reach the language model.")
        self.assertEqual(speaker.spoken[1].strip(), "Back online.")
        self.assertIn(("error", "server down"), events)

    def test_unreachable_server_fails_at_start(self):
        assistant, _, _, llm, _, _ = make_assistant(["Hi"])

        def down():
            raise LLMError("Cannot reach the LLM server")

        llm.check_connection = down

        with self.assertRaises(LLMError):
            assistant.run()


class InterruptTests(unittest.TestCase):

    def test_user_speech_stops_reply_and_llm_stream(self):
        assistant, recognizer, speaker, llm, _, events = make_assistant(
            ["Tell me a long story"], ["word " * 50], interruptible=True
        )

        closed = threading.Event()
        original = llm.stream_reply

        def tracked(text):
            try:
                yield from original(text)
            finally:
                closed.set()

        llm.stream_reply = tracked

        # The user starts talking after a few words.
        count = [0]

        def on_token(token):
            count[0] += 1
            if count[0] == 3:
                recognizer.on_speech_start()

        speaker.on_token = on_token

        assistant.run()

        self.assertIn(("interrupted", ""), events)
        self.assertTrue(closed.is_set(), "LLM stream was not closed")

    def test_speech_while_idle_does_not_stop_anything(self):
        assistant, _, speaker, _, log, _ = make_assistant([], interruptible=True)
        assistant.start()

        assistant._on_user_speech()

        self.assertNotIn("stop", log)


class CommandLineTests(unittest.TestCase):

    def test_defaults(self):
        args = parse_args([])

        self.assertEqual(args.host, "http://10.0.0.100:11434")
        self.assertEqual(args.model, "qwen3:8b")
        self.assertFalse(args.interruptible)

    def test_options(self):
        args = parse_args(["--model", "qwen3:14b", "--voice", "bf_emma",
                           "--interruptible", "--pause", "2"])

        self.assertEqual(args.model, "qwen3:14b")
        self.assertEqual(args.voice, "bf_emma")
        self.assertTrue(args.interruptible)
        self.assertEqual(args.pause, 2.0)


# --------------------------------------------------
# End to end with real models (--e2e)
# --------------------------------------------------

@unittest.skipUnless(RUN_E2E_TESTS, "pass --e2e to run")
class EndToEndTests(unittest.TestCase):

    def test_spoken_question_gets_spoken_answer(self):
        from voicekit.llm_client import LLMClient
        from voicekit.speech_recognizer import ContinuousSpeechRecognizer
        from voicekit.text_to_speech import KOKORO_SAMPLE_RATE, KokoroSpeaker

        speaker = KokoroSpeaker()
        recognizer = ContinuousSpeechRecognizer()
        llm = LLMClient()
        assistant = VoiceAssistant(recognizer, speaker, llm, on_event=None)

        try:
            start = time.perf_counter()
            assistant.start()
            print(f"\n  start-up: {time.perf_counter() - start:.1f}s")

            question = "What is the capital city of France?"
            audio = speaker.synthesize(question)

            start = time.perf_counter()
            heard = recognizer.transcribe_audio(audio, KOKORO_SAMPLE_RATE)["text"]
            stt = time.perf_counter() - start
            print(f"  heard:  {heard!r} ({stt:.2f}s)")
            self.assertIn("capital", heard.lower())

            start = time.perf_counter()
            first_token = None
            reply = ""
            for token in llm.stream_reply(heard):
                first_token = first_token or time.perf_counter() - start
                reply += token
            print(f"  reply:  {reply.strip()!r} (first token {first_token:.2f}s)")
            self.assertIn("paris", reply.lower())

            start = time.perf_counter()
            spoken = speaker.synthesize(reply)
            print(f"  voiced: {spoken.size / KOKORO_SAMPLE_RATE:.1f}s of speech "
                  f"in {time.perf_counter() - start:.2f}s")
            self.assertGreater(spoken.size, 0)

        finally:
            assistant.close()


if __name__ == "__main__":
    unittest.main()
