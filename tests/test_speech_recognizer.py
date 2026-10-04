"""
Tests for speech_recognizer.py.

    python tests/test_speech_recognizer.py             # fast tests (no GPU, no mic)
    python tests/test_speech_recognizer.py --gpu       # + real Whisper on output_0.wav
    python tests/test_speech_recognizer.py --mic       # + live microphone (you speak)
    python tests/test_speech_recognizer.py --gpu --mic -v

The flags can also be set with environment variables RUN_GPU_TESTS=1 and
RUN_MIC_TESTS=1 (e.g. when running `python -m unittest`).
"""

import os
import pathlib
import sys
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import numpy as np

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit import speech_recognizer as sr
from voicekit.speech_recognizer import (
    _END_OF_PROMPT,
    _SPEECH_STARTED,
    ContinuousSpeechRecognizer,
    SpeechRecognizerConfig,
    _Segmenter,
    _to_whisper_audio,
    _Utterance,
    clean_text,
)


HERE = pathlib.Path(__file__).parent
SAMPLE_WAV = HERE / "data" / "output_0.wav"

def _take_flag(flag):
    # Read before the test classes are defined, since the skip
    # decorators are evaluated at import time.
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


RUN_GPU_TESTS = _take_flag("--gpu") or os.environ.get("RUN_GPU_TESTS") == "1"
RUN_MIC_TESTS = _take_flag("--mic") or os.environ.get("RUN_MIC_TESTS") == "1"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def run_segmenter(pattern, include_speech_started=False, **overrides):
    """
    Feed a pattern of 'v' (voice) and '.' (silence) frames, one character
    per 30 ms frame. Each frame's payload is its index, so tests can check
    exactly which frames ended up in an utterance.

    Returns a list of (frame_index, event).
    """

    params = dict(
        frame_ms=30,
        start_frames=5,
        end_frames=17,
        min_speech_frames=10,
        pre_roll_frames=10,
        max_frames=1000,
        prompt_end_frames=50,
    )
    params.update(overrides)

    segmenter = _Segmenter(**params)
    events = []

    for index, char in enumerate(pattern):
        for event in segmenter.feed(index.to_bytes(4, "little"), char == "v"):
            if event is _SPEECH_STARTED and not include_speech_started:
                continue
            events.append((index, event))

    return events


def frame_indices(utterance):
    pcm = utterance.pcm
    return [
        int.from_bytes(pcm[i:i + 4], "little")
        for i in range(0, len(pcm), 4)
    ]


def utterances(events):
    return [event for _, event in events if isinstance(event, _Utterance)]


def kinds(events):
    return ["U" if isinstance(e, _Utterance) else "E" for _, e in events]


FAKE_MIC = {
    "index": 0,
    "name": "Fake microphone",
    "max_input_channels": 1,
    "default_samplerate": 16000.0,
}


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, texts):
        # One entry per transcribe() call; each is a list of segment texts.
        self.texts = list(texts)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(SimpleNamespace(audio=audio, kwargs=kwargs))
        texts = self.texts.pop(0) if self.texts else []
        segments = (SimpleNamespace(text=text) for text in texts)
        info = SimpleNamespace(language="en", language_probability=1.0)
        return segments, info


def make_recognizer(texts=(), **config):
    """A recognizer with a fake microphone and a fake Whisper model."""

    config.setdefault("noise_floor_dbfs", -60.0)

    with mock.patch.object(sr.sd, "query_devices", return_value=FAKE_MIC):
        recognizer = ContinuousSpeechRecognizer(SpeechRecognizerConfig(**config))

    recognizer._model = FakeWhisperModel(texts)
    return recognizer


class FakeInputStream:
    """
    Replaces sounddevice.RawInputStream. Plays a script of (seconds, kind)
    sections, where kind is "voice" (loud tone) or "silence", then keeps
    sending silence until closed.
    """

    def __init__(self, script, samplerate, blocksize, callback, **kwargs):
        self.script = script
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self._closed = threading.Event()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._closed.set()
        self._thread.join()

    def _frames(self):
        t = np.arange(self.blocksize) / self.samplerate
        voice = (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()
        silence = bytes(self.blocksize * 2)

        for seconds, kind in self.script:
            count = round(seconds * self.samplerate / self.blocksize)
            yield from [voice if kind == "voice" else silence] * count

        while True:
            yield silence

    def _run(self):
        for frame in self._frames():
            if self._closed.is_set():
                return
            self.callback(frame, self.blocksize, None, None)
            # Faster than real time, but slow enough not to flood the queue.
            time.sleep(0.0005)


def loud_frames_are_voice(frame):
    samples = np.frombuffer(frame, dtype=np.int16)
    return bool(np.abs(samples).max() > 1000)


# --------------------------------------------------
# Segmenter
# --------------------------------------------------

class SegmenterTests(unittest.TestCase):

    def test_pre_roll_includes_full_lead_in(self):
        events = run_segmenter("." * 20 + "v" * 20 + "." * 100)

        [utterance] = utterances(events)

        # Speech starts at frame 20; 10 frames (300 ms) of pre-roll.
        self.assertEqual(frame_indices(utterance)[0], 10)

    def test_end_of_prompt_after_prompt_end_silence(self):
        events = run_segmenter("v" * 20 + "." * 100)

        self.assertEqual(kinds(events), ["U", "E"])
        # Last voice frame is 19; prompt ends after 50 quiet frames.
        self.assertEqual(events[-1][0], 19 + 50)

    def test_short_pause_keeps_one_prompt(self):
        events = run_segmenter("v" * 20 + "." * 25 + "v" * 20 + "." * 100)

        self.assertEqual(kinds(events), ["U", "U", "E"])

    def test_long_pause_starts_new_prompt(self):
        events = run_segmenter("v" * 20 + "." * 60 + "v" * 20 + "." * 100)

        self.assertEqual(kinds(events), ["U", "E", "U", "E"])

    def test_short_noise_is_rejected(self):
        self.assertEqual(run_segmenter("v" * 6 + "." * 100), [])

    def test_max_length_split_loses_no_audio(self):
        events = run_segmenter("v" * 250 + "." * 100, max_frames=100)

        chunks = utterances(events)
        frames = sum((frame_indices(chunk) for chunk in chunks), [])

        self.assertEqual([chunk.split for chunk in chunks], [True, True, False])
        self.assertEqual(frames, list(range(frames[0], frames[0] + len(frames))))
        self.assertGreaterEqual(frames[-1], 249)

    def test_rejected_blip_does_not_reset_prompt_timer(self):
        events = run_segmenter("v" * 20 + "." * 20 + "v" * 6 + "." * 100)

        end_frames = [i for i, e in events if e is _END_OF_PROMPT]
        self.assertEqual(end_frames, [19 + 50])

    def test_speech_started_once_per_utterance_after_min_speech(self):
        events = run_segmenter(
            "v" * 250 + "." * 25 + "v" * 20 + "." * 25 + "v" * 6 + "." * 100,
            include_speech_started=True,
            max_frames=100,
        )

        started = [i for i, e in events if e is _SPEECH_STARTED]

        # Two real utterances (the long one split into chunks counts once);
        # the 6-frame blip never reaches min_speech_frames.
        self.assertEqual(started, [9, 275 + 9])

    def test_reset_discards_speech_in_progress(self):
        segmenter = _Segmenter(30, 5, 17, 10, 10, 1000, 50)

        for _ in range(20):
            segmenter.feed(b"\0" * 4, True)

        self.assertTrue(segmenter.active)
        segmenter.reset()
        self.assertFalse(segmenter.active)


# --------------------------------------------------
# Text cleanup
# --------------------------------------------------

class CleanTextTests(unittest.TestCase):

    def check(self, raw, expected, **kwargs):
        self.assertEqual(clean_text(raw, **kwargs), expected)

    def test_removes_fillers(self):
        self.check(" Um, I want to, uh, write a function.", "I want to write a function.")
        self.check("So, um, the plan is, uh.", "So the plan is.")

    def test_keeps_fillers_when_disabled(self):
        self.check("um, hello", "Um, hello", remove_fillers=False)

    def test_does_not_touch_words_containing_fillers(self):
        self.check("umbrella and drum", "Umbrella and drum")
        self.check("Measure it in mm, then hmm, ok", "Measure it in mm, then ok")

    def test_capitalises_sentence_after_removed_filler(self):
        self.check("Hello. Um, let's go.", "Hello. Let's go.")

    def test_keeps_abbreviations_and_ellipses(self):
        self.check("Use e.g. pandas... maybe", "Use e.g. pandas... maybe")

    def test_tidies_whitespace_and_punctuation(self):
        self.check("  hello   world ,  ok .", "Hello world, ok.")

    def test_empty(self):
        self.check("   ", "")


# --------------------------------------------------
# Audio conversion
# --------------------------------------------------

class AudioConversionTests(unittest.TestCase):

    def tone(self, sample_rate, seconds=1.0, freq=440, amplitude=0.5):
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_resamples_to_16k_preserving_pitch_and_level(self):
        for rate in (8000, 24000, 32000, 48000):
            with self.subTest(rate=rate):
                out = _to_whisper_audio(self.tone(rate), rate)

                self.assertEqual(out.size, 16000)
                self.assertEqual(np.argmax(np.abs(np.fft.rfft(out))), 440)
                self.assertAlmostEqual(float(np.abs(out).max()), 0.5, delta=0.01)

    def test_int16_bytes_are_scaled(self):
        pcm = np.array([0, 16384, -32768], dtype=np.int16).tobytes()

        out = _to_whisper_audio(pcm, 16000)

        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, [0.0, 0.5, -1.0])

    def test_stereo_is_mixed_to_mono(self):
        stereo = np.stack([self.tone(16000), self.tone(16000)], axis=1)

        self.assertEqual(_to_whisper_audio(stereo, 16000).ndim, 1)


# --------------------------------------------------
# Config
# --------------------------------------------------

class ConfigTests(unittest.TestCase):

    def test_defaults_are_valid(self):
        SpeechRecognizerConfig()

    def test_rejects_invalid_values(self):
        invalid = [
            dict(sample_rate=44100),
            dict(frame_ms=25),
            dict(vad_mode=4),
            dict(speech_start_ms=10),
            dict(prompt_end_ms=100, speech_end_ms=500),
            dict(max_speech_seconds=0),
            dict(beam_size=0),
        ]

        for kwargs in invalid:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SpeechRecognizerConfig(**kwargs)


# --------------------------------------------------
# Transcription (fake Whisper model)
# --------------------------------------------------

class TranscriptionTests(unittest.TestCase):

    def test_cleans_and_joins_segments(self):
        recognizer = make_recognizer([[" Um, write a", " Python function."]])

        result = recognizer.transcribe_audio(np.zeros(16000, np.float32))

        self.assertEqual(result["text"], "Write a Python function.")
        self.assertEqual(result["raw_text"], "Um, write a Python function.")
        self.assertAlmostEqual(result["audio_duration"], 1.0)

    def test_discards_whole_transcript_hallucination(self):
        recognizer = make_recognizer([[" Thank you."], [" Thank you for the help."]])
        silence = np.zeros(16000, np.float32)

        self.assertEqual(recognizer.transcribe_audio(silence)["text"], "")
        self.assertEqual(
            recognizer.transcribe_audio(silence)["text"],
            "Thank you for the help.",
        )

    def test_hallucination_filter_can_be_disabled(self):
        recognizer = make_recognizer([[" Thank you."]], hallucination_phrases=())

        result = recognizer.transcribe_audio(np.zeros(16000, np.float32))

        self.assertEqual(result["text"], "Thank you.")

    def test_drops_segments_without_words(self):
        recognizer = make_recognizer([[" ...", " Hello there.", " ♪"]])

        result = recognizer.transcribe_audio(np.zeros(16000, np.float32))

        self.assertEqual(result["text"], "Hello there.")

    def test_passes_config_and_context_to_whisper(self):
        recognizer = make_recognizer(
            [["x"]],
            initial_prompt="Claude, CUDA.",
            hotwords="faster-whisper",
            beam_size=5,
        )

        recognizer.transcribe_audio(np.zeros(16000, np.float32), context="Earlier text")

        kwargs = recognizer._model.calls[0].kwargs
        self.assertEqual(kwargs["initial_prompt"], "Claude, CUDA. Earlier text")
        self.assertEqual(kwargs["hotwords"], "faster-whisper")
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertFalse(kwargs["vad_filter"])

    def test_resamples_before_transcribing(self):
        recognizer = make_recognizer([["x"]])

        recognizer.transcribe_audio(np.zeros(48000, np.float32), 48000)

        self.assertEqual(recognizer._model.calls[0].audio.size, 16000)


# --------------------------------------------------
# Full listening pipeline (fake microphone stream)
# --------------------------------------------------

class ListenPipelineTests(unittest.TestCase):

    def listen_with_script(self, script, texts, **config):
        recognizer = make_recognizer(texts, **config)

        def stream_factory(**kwargs):
            return FakeInputStream(script, **kwargs)

        patches = [
            mock.patch.object(sr.sd, "RawInputStream", stream_factory),
            mock.patch.object(recognizer, "_is_voice", loud_frames_are_voice),
        ]

        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        return recognizer

    def test_listen_once_returns_one_prompt(self):
        recognizer = self.listen_with_script(
            [(0.5, "silence"), (1.0, "voice"), (2.0, "silence")],
            [[" Write a sorting function."]],
        )

        result = recognizer.listen_once(timeout=10)

        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Write a sorting function.")
        self.assertEqual(result["noise_floor_dbfs"], -60.0)
        # 1 s of voice + pre-roll + trailing silence, transcribed in one call.
        self.assertEqual(len(recognizer._model.calls), 1)
        self.assertGreater(result["audio_duration"], 1.0)
        self.assertFalse(recognizer._listening)

    def test_short_pauses_are_one_prompt_transcribed_once(self):
        recognizer = self.listen_with_script(
            [
                (1.0, "voice"), (0.7, "silence"),
                (1.0, "voice"), (0.7, "silence"),
                (1.0, "voice"), (2.0, "silence"),
            ],
            [[" One long prompt."]],
        )

        result = recognizer.listen_once(timeout=10)

        self.assertEqual(result["text"], "One long prompt.")
        self.assertEqual(len(recognizer._model.calls), 1)
        self.assertGreater(result["audio_duration"], 3.0)

    def test_listen_yields_separate_prompts(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence"), (1.0, "voice"), (2.0, "silence")],
            [[" First."], [" Second."]],
        )

        prompts = []
        for prompt in recognizer.listen(timeout=5):
            prompts.append(prompt["text"])
            if len(prompts) == 2:
                break

        self.assertEqual(prompts, ["First.", "Second."])
        self.assertFalse(recognizer._listening)

    def test_listen_utterances_yields_each_utterance_with_context(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (0.7, "silence"), (1.0, "voice"), (2.0, "silence")],
            [[" Part one."], [" part two."]],
        )

        texts = []
        for utterance in recognizer.listen_utterances(timeout=5):
            texts.append(utterance["text"])
            if len(texts) == 2:
                break

        self.assertEqual(texts, ["Part one.", "Part two."])
        second_call = recognizer._model.calls[1].kwargs
        self.assertIn("Part one.", second_call["initial_prompt"])

    def test_text_postprocessor_applied_to_prompt(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence")],
            [[" hello"]],
            text_postprocessor=str.upper,
        )

        self.assertEqual(recognizer.listen_once(timeout=10)["text"], "HELLO")

    def test_timeout_returns_none_when_nobody_speaks(self):
        recognizer = self.listen_with_script([], [])

        start = time.monotonic()
        result = recognizer.listen_once(timeout=0.5)

        self.assertIsNone(result)
        self.assertLess(time.monotonic() - start, 5)

    def test_paused_audio_is_ignored(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence")],
            [[" Should not appear."]],
        )

        recognizer.pause()
        result = recognizer.listen_once(timeout=1)

        self.assertIsNone(result)
        self.assertEqual(recognizer._model.calls, [])

    def test_watch_hears_snippets_while_paused(self):
        recognizer = self.listen_with_script(
            # 1 s of speech: cut into snippets of at most 1.5 s.
            [(0.5, "silence"), (1.0, "voice"), (2.0, "silence")],
            [[" Hang on."]],
        )
        heard = []
        recognizer.watch(heard.append)
        recognizer.pause()

        result = recognizer.listen_once(timeout=1.5)

        self.assertIsNone(result)  # nothing became a prompt
        self.assertEqual(heard, ["Hang on."])
        self.assertEqual(recognizer._model.calls[0].kwargs["beam_size"], 1)

    def test_long_speech_while_paused_is_cut_into_short_snippets(self):
        recognizer = self.listen_with_script(
            [(4.0, "voice"), (2.0, "silence")],
            [[" one"], [" two"], [" three"]],
        )
        heard = []
        recognizer.watch(heard.append)
        recognizer.pause()

        recognizer.listen_once(timeout=1.5)

        self.assertEqual(heard, ["One", "Two", "Three"])

    def test_nothing_watched_when_not_paused_or_not_watching(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence")], [[" Hello."]],
        )
        recognizer.pause()

        recognizer.listen_once(timeout=1)

        self.assertEqual(recognizer._model.calls, [])

    def test_echo_canceller_cleans_every_frame(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence")], [[" Should not appear."]],
        )

        class Silencer:
            """Pretends all the 'voice' was the speakers' echo."""
            frames = 0

            def process(self, pcm, sample_rate):
                self.frames += 1
                return bytes(len(pcm))

        recognizer.echo_canceller = Silencer()
        result = recognizer.listen_once(timeout=1)

        self.assertIsNone(result)
        self.assertGreater(recognizer.echo_canceller.frames, 30)

    def test_on_speech_start_called_before_transcription(self):
        recognizer = self.listen_with_script(
            [(1.0, "voice"), (2.0, "silence")],
            [[" Hello."]],
        )

        calls = []
        recognizer.on_speech_start = lambda: calls.append(
            len(recognizer._model.calls)
        )

        recognizer.listen_once(timeout=10)

        # Called once, while nothing had been transcribed yet.
        self.assertEqual(calls, [0])

    def test_calibration_measures_noise_floor(self):
        recognizer = self.listen_with_script([], [], noise_floor_dbfs=None)

        recognizer.listen_once(timeout=0.5)

        # The fake stream sends digital silence.
        self.assertEqual(recognizer.noise_floor_dbfs, -100.0)
        self.assertEqual(recognizer.noise_threshold_dbfs, -50.0)

    def test_calibration_ignores_speech_with_pauses(self):
        # Talking for half the calibration second: the pauses are the room.
        recognizer = self.listen_with_script(
            [(0.25, "voice"), (0.25, "silence")] * 4, [], noise_floor_dbfs=None,
        )

        recognizer.listen_once(timeout=0.5)

        self.assertEqual(recognizer.noise_floor_dbfs, -100.0)

    def test_calibration_during_continuous_sound_is_capped(self):
        recognizer = self.listen_with_script(
            [(3.0, "voice")], [], noise_floor_dbfs=None,
        )

        with self.assertLogs(sr.logger, "WARNING"):
            recognizer.listen_once(timeout=0.5)

        self.assertEqual(recognizer.noise_floor_dbfs, -40.0)
        self.assertEqual(recognizer.noise_threshold_dbfs, -30.0)

    def test_dead_microphone_raises(self):
        recognizer = make_recognizer()

        class SilentStream(FakeInputStream):
            def _run(self):
                pass  # never calls the callback

        with mock.patch.object(
            sr.sd,
            "RawInputStream",
            lambda **kwargs: SilentStream([], **kwargs),
        ):
            with self.assertRaisesRegex(RuntimeError, "Audio capture failed"):
                recognizer.listen_once()


# --------------------------------------------------
# Real Whisper on the GPU (--gpu)
# --------------------------------------------------

@unittest.skipUnless(RUN_GPU_TESTS, "pass --gpu to run")
class WhisperGpuTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recognizer = ContinuousSpeechRecognizer()
        cls.recognizer.load()

    @classmethod
    def tearDownClass(cls):
        cls.recognizer.close()

    def test_torch_not_imported(self):
        self.assertNotIn("torch", sys.modules)

    @unittest.skipUnless(SAMPLE_WAV.exists(), "output_0.wav not found")
    def test_transcribes_sample_wav(self):
        with wave.open(str(SAMPLE_WAV)) as wav:
            rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())

        result = self.recognizer.transcribe_audio(pcm, rate)
        print(f"\n  {SAMPLE_WAV.name} ({rate} Hz): {result['text']!r} "
              f"in {result['latency']:.2f}s")

        self.assertIn("kokoro", result["text"].lower())
        self.assertLess(result["latency"], 5)

    def test_noise_gives_no_text(self):
        noise = np.random.default_rng(0).normal(0, 0.01, 32000).astype(np.float32)

        self.assertEqual(self.recognizer.transcribe_audio(noise)["text"], "")


# --------------------------------------------------
# Live microphone (--mic)
# --------------------------------------------------

@unittest.skipUnless(RUN_MIC_TESTS, "pass --mic to run")
class LiveMicrophoneTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recognizer = ContinuousSpeechRecognizer()

    @classmethod
    def tearDownClass(cls):
        cls.recognizer.close()

    def test_1_silence_times_out(self):
        print("\n  Stay quiet for a few seconds...", flush=True)

        result = self.recognizer.listen_once(timeout=3)

        self.assertIsNone(result, f"Heard something in silence: {result}")

    def test_2_hears_a_prompt(self):
        print(
            "\n  Now say a sentence, e.g. 'Write a Python function that "
            "reverses a string', then pause...",
            flush=True,
        )

        result = self.recognizer.listen_once(timeout=15)

        self.assertIsNotNone(result, "Nothing heard within 15 seconds.")
        print(f"  Heard: {result['text']!r} "
              f"({result['audio_duration']:.1f}s audio, "
              f"{result['latency']:.2f}s transcription)")
        self.assertTrue(result["text"])


if __name__ == "__main__":
    unittest.main()
