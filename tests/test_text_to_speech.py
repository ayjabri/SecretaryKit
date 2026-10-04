"""
Tests for text_to_speech.py.

    python tests/test_text_to_speech.py            # fast tests (no GPU, no sound)
    python tests/test_text_to_speech.py --gpu      # + real Kokoro synthesis
    python tests/test_text_to_speech.py --audio    # + plays speech on your speakers

Environment variables RUN_GPU_TESTS=1 / RUN_AUDIO_TESTS=1 work too.
"""

import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit import text_to_speech as tts
from voicekit.text_to_speech import (
    KOKORO_SAMPLE_RATE,
    KokoroSpeaker,
    SentenceSplitter,
    TextToSpeechConfig,
    prepare_for_speech,
)


def _take_flag(flag):
    # Read before the test classes are defined, since the skip
    # decorators are evaluated at import time.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_GPU_TESTS = _take_flag("--gpu") or os.environ.get("RUN_GPU_TESTS") == "1"
RUN_AUDIO_TESTS = _take_flag("--audio") or os.environ.get("RUN_AUDIO_TESTS") == "1"


# --------------------------------------------------
# Text preparation
# --------------------------------------------------

class PrepareForSpeechTests(unittest.TestCase):

    def check(self, raw, expected, **kwargs):
        self.assertEqual(prepare_for_speech(raw, **kwargs), expected)

    def test_plain_text_unchanged(self):
        self.check("Hello there. How are you?", "Hello there. How are you?")

    def test_strips_markdown_emphasis_and_headings(self):
        self.check("## Summary\nThis is **very** _important_.",
                   "Summary. This is very important.")

    def test_bullets_become_sentences(self):
        self.check("Options:\n- Fast\n- Cheap\n1. First\n2) Second",
                   "Options: Fast. Cheap. First. Second.")

    def test_links_and_urls(self):
        self.check("See [the docs](https://x.io/a) or https://y.com/b now.",
                   "See the docs or a link now.")

    def test_code_blocks_replaced(self):
        text = "Try this:\n```python\nprint('hi')\n```\nDone."

        self.check(text, "Try this: Code below. Done.",
                   code_block_text="Code below.")
        self.check(text, "Try this: Done.")

    def test_unclosed_code_block_dropped(self):
        self.check("Here:\n```py\nx = 1", "Here:")

    def test_inline_code_kept_as_words(self):
        self.check("Call `my_function` now.", "Call my function now.")

    def test_think_blocks_removed(self):
        self.check("<think>hmm, let me see</think>The answer is 4.",
                   "The answer is 4.")

    def test_emojis_tables_and_rules_removed(self):
        self.check("Great 🎉\n---\n| a | b |\n|---|---|\n| 1 | 2 |",
                   "Great. a, b. 1, 2.")

    def test_keeps_csharp(self):
        self.check("I like C# a lot.", "I like C# a lot.")


# --------------------------------------------------
# Sentence splitting
# --------------------------------------------------

def stream(splitter, text, step=3):
    """Feed text a few characters at a time, like LLM tokens."""
    chunks = []
    for i in range(0, len(text), step):
        chunks += splitter.feed(text[i:i + step])
    return chunks + splitter.flush()


class SentenceSplitterTests(unittest.TestCase):

    def splitter(self, **kwargs):
        params = dict(first_chunk_min_chars=10, min_chunk_chars=40,
                      max_chunk_chars=120)
        params.update(kwargs)
        return SentenceSplitter(**params)

    def test_first_sentence_released_early(self):
        splitter = self.splitter()

        self.assertEqual(splitter.feed("Sure thing. "), ["Sure thing."])

    def test_waits_for_whitespace_after_period(self):
        splitter = self.splitter()

        # "3." could be "3.5"; nothing is released yet.
        self.assertEqual(splitter.feed("It costs 3."), [])
        self.assertEqual(splitter.feed("5 dollars today. "),
                         ["It costs 3.5 dollars today."])

    def test_later_sentences_merged_to_min_length(self):
        text = "Hello there. Yes. No. Maybe so. That is a longer sentence here. End."
        chunks = stream(self.splitter(), text)

        self.assertEqual(chunks[0], "Hello there.")
        self.assertTrue(all(len(c.strip()) >= 40 for c in chunks[1:-1]))
        self.assertEqual("".join(chunks), text)

    def test_abbreviations_do_not_split(self):
        chunks = stream(self.splitter(first_chunk_min_chars=5),
                        "Use e.g. pandas here. Then stop.")

        self.assertEqual(chunks[0], "Use e.g. pandas here.")

    def test_code_block_held_then_replaced(self):
        splitter = self.splitter(code_block_text="CODE.")
        text = "Here you go. ```python\nx = 1. y = 2.\n``` Hope that helps you out."

        chunks = stream(splitter, text)
        joined = "".join(chunks)

        self.assertNotIn("x = 1", joined)
        self.assertIn("CODE.", joined)

    def test_think_block_never_released(self):
        chunks = stream(self.splitter(),
                        "<think>Let me think. Okay. Sure.</think>The answer is 42.")

        self.assertEqual("".join(chunks).strip(), "The answer is 42.")

    def test_long_text_without_punctuation_is_cut(self):
        chunks = stream(self.splitter(), "word " * 100)

        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertEqual("".join(chunks), "word " * 100)

    def test_newlines_split_list_items(self):
        chunks = stream(self.splitter(first_chunk_min_chars=5, min_chunk_chars=5),
                        "Steps\n- one thing\n- another thing\n")

        self.assertEqual([c.strip() for c in chunks],
                         ["Steps", "- one thing", "- another thing"])


# --------------------------------------------------
# Speaker with fake Kokoro and fake sound card
# --------------------------------------------------

FAKE_SPEAKER = {"index": 0, "name": "Fake speaker", "max_output_channels": 2}


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakePipeline:
    """Returns 10 ms of constant audio per character; optionally slow."""

    def __init__(self, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on
        self.texts = []

    def __call__(self, text, voice=None, speed=1.0, split_pattern=None):
        self.texts.append(text)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("synthesis failed")
        time.sleep(self.delay)
        samples = len(text) * KOKORO_SAMPLE_RATE // 100
        yield SimpleNamespace(audio=FakeTensor(np.full(samples, 0.25, np.float32)))


class FakeOutputStream:
    """Pulls audio from the callback 4x faster than real time."""

    def __init__(self, samplerate, callback, blocksize=1200, **kwargs):
        self.callback = callback
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.played = []
        self._running = threading.Event()

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        self._thread.join()

    def close(self):
        pass

    def _run(self):
        status = SimpleNamespace(output_underflow=False)
        while self._running.is_set():
            out = np.zeros((self.blocksize, 1), np.float32)
            self.callback(out, self.blocksize, None, status)
            self.played.append(out[:, 0].copy())
            time.sleep(self.blocksize / self.samplerate / 4)

    def nonzero_samples(self):
        return int(sum(np.count_nonzero(block) for block in self.played))


def make_speaker(pipeline=None, **config):
    """A KokoroSpeaker wired to FakePipeline and FakeOutputStream."""

    config.setdefault("sentence_gap_ms", 0)
    pipeline = pipeline or FakePipeline()
    streams = []

    def stream_factory(**kwargs):
        stream = FakeOutputStream(**kwargs)
        streams.append(stream)
        return stream

    patches = [
        mock.patch.object(tts.sd, "query_devices", return_value=FAKE_SPEAKER),
        mock.patch.object(tts.sd, "OutputStream", stream_factory),
        mock.patch.object(KokoroSpeaker, "_create_pipeline", lambda self: pipeline),
        mock.patch.object(KokoroSpeaker, "_resolve_voice", lambda self, v: v),
    ]
    for patch in patches:
        patch.start()

    speaker = KokoroSpeaker(TextToSpeechConfig(**config))
    speaker.load()

    def cleanup():
        speaker.close()
        for patch in patches:
            patch.stop()

    return speaker, pipeline, streams[0], cleanup


def slow_tokens(text, delay=0.01, closed=None):
    try:
        for word in text.split(" "):
            time.sleep(delay)
            yield word + " "
    finally:
        if closed is not None:
            closed.set()


class SpeakerTests(unittest.TestCase):

    def setUp(self):
        self.pipeline = FakePipeline()
        self.speaker, _, self.stream, cleanup = make_speaker(self.pipeline)
        self.addCleanup(cleanup)

    def expected_samples(self, texts):
        return sum(len(t) * KOKORO_SAMPLE_RATE // 100 for t in texts)

    def test_speak_plays_all_audio(self):
        self.pipeline.texts.clear()  # ignore the warm-up

        finished = self.speaker.speak("Hello there. This is a test.", wait=True)

        self.assertTrue(finished)
        self.assertFalse(self.speaker.is_speaking)
        self.assertEqual(self.stream.nonzero_samples(),
                         self.expected_samples(self.pipeline.texts))

    def test_speak_cleans_markdown(self):
        self.pipeline.texts.clear()

        self.speaker.speak("**Bold** move. ```x```", wait=True)

        self.assertEqual(" ".join(self.pipeline.texts),
                         "Bold move. I've put the code on screen.")

    def test_speak_returns_immediately_by_default(self):
        self.speaker.speak("A fairly long sentence to keep it busy for a while.")

        self.assertTrue(self.speaker.is_speaking)
        self.assertTrue(self.speaker.wait(timeout=5))

    def test_speak_stream_starts_before_stream_ends(self):
        started_at = []
        text = "First sentence here. " + "more words " * 30

        def tokens():
            for token in slow_tokens(text, delay=0.005):
                if self.stream.nonzero_samples() and not started_at:
                    started_at.append(time.monotonic())
                yield token
            tokens.ended = time.monotonic()

        self.assertTrue(self.speaker.speak_stream(tokens()))
        self.assertTrue(started_at, "audio did not start until the stream ended")
        self.assertLess(started_at[0], tokens.ended)

    def test_stop_interrupts_speak_stream_and_closes_source(self):
        closed = threading.Event()
        result = []

        thread = threading.Thread(target=lambda: result.append(
            self.speaker.speak_stream(slow_tokens("word. " * 200, closed=closed))
        ))
        thread.start()

        time.sleep(0.2)
        self.speaker.stop()
        thread.join(timeout=5)

        self.assertEqual(result, [False])
        self.assertTrue(closed.is_set(), "token source was not closed")
        self.assertTrue(self.speaker.wait(timeout=2))

    def test_stop_silences_playback_immediately(self):
        self.speaker.speak("A long sentence that will take a while to play. " * 5)
        time.sleep(0.1)

        self.speaker.stop()
        time.sleep(0.05)
        played_after_stop = self.stream.nonzero_samples()
        time.sleep(0.2)

        self.assertEqual(self.stream.nonzero_samples(), played_after_stop)
        self.assertTrue(self.speaker.wait(timeout=2))

    def test_wait_returns_false_when_stopped(self):
        self.speaker.speak("Some speech that is long enough to interrupt. " * 3)

        threading.Timer(0.1, self.speaker.stop).start()

        self.assertFalse(self.speaker.wait(timeout=5))

    def test_can_speak_again_after_stop(self):
        self.speaker.speak("First reply. " * 5)
        self.speaker.stop()

        self.assertTrue(self.speaker.speak("Second reply.", wait=True))

    def test_synthesize_returns_audio_without_playing(self):
        audio = self.speaker.synthesize("Hello.")

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.size, self.expected_samples(["Hello."]))
        self.assertEqual(self.stream.nonzero_samples(), 0)


class SpeakerFailureTests(unittest.TestCase):

    def test_synthesis_error_does_not_hang(self):
        speaker, pipeline, stream, cleanup = make_speaker(
            FakePipeline(fail_on="explode")
        )
        self.addCleanup(cleanup)

        with self.assertLogs("voicekit.text_to_speech", "ERROR"):
            finished = speaker.speak("This will explode. Next is fine.", wait=True)

        self.assertTrue(finished)
        self.assertFalse(speaker.is_speaking)

    def test_volume_applied(self):
        speaker, _, _, cleanup = make_speaker(volume=2.0)
        self.addCleanup(cleanup)

        self.assertAlmostEqual(float(speaker.synthesize("Hi.").max()), 0.5)


class ConfigTests(unittest.TestCase):

    def test_rejects_invalid_values(self):
        for kwargs in (dict(speed=3), dict(volume=-1), dict(voice=""),
                       dict(min_chunk_chars=500, max_chunk_chars=400)):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    TextToSpeechConfig(**kwargs)


# --------------------------------------------------
# Real Kokoro (--gpu) and real playback (--audio)
# --------------------------------------------------

@unittest.skipUnless(RUN_GPU_TESTS or RUN_AUDIO_TESTS, "pass --gpu to run")
class KokoroGpuTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.speaker = KokoroSpeaker()
        cls.speaker.load()

    @classmethod
    def tearDownClass(cls):
        cls.speaker.close()

    def test_synthesis_is_faster_than_real_time(self):
        start = time.perf_counter()
        audio = self.speaker.synthesize(
            "This sentence checks that Kokoro runs quickly on the GPU."
        )
        elapsed = time.perf_counter() - start
        seconds = audio.size / KOKORO_SAMPLE_RATE

        print(f"\n  {seconds:.2f}s of speech in {elapsed:.3f}s "
              f"({seconds / elapsed:.0f}x real time)")

        self.assertGreater(seconds, 1.0)
        self.assertLess(elapsed, seconds)
        self.assertLessEqual(float(np.abs(audio).max()), 1.0)

    @unittest.skipUnless(RUN_AUDIO_TESTS, "pass --audio to run")
    def test_plays_streamed_speech(self):
        print("\n  You should hear two sentences.", flush=True)

        tokens = slow_tokens(
            "This is Kokoro, speaking a reply while it streams in. "
            "If you can hear this clearly, playback works.",
            delay=0.03,
        )

        self.assertTrue(self.speaker.speak_stream(tokens))


if __name__ == "__main__":
    unittest.main()
