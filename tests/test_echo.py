"""
Tests for voicekit/echo.py: WebRTC echo cancellation via LiveKit.

    python tests/test_echo.py

Skipped when livekit is not installed (pip install "voicekit[aec]").
"""

import os
import sys
import unittest

import numpy as np

# Make the project root importable when run as "python tests/<file>.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voicekit.echo import EchoCanceller

try:
    import livekit.rtc  # noqa: F401
    HAVE_LIVEKIT = True
except ImportError:
    HAVE_LIVEKIT = False


def speech_like(seconds, rate, seed=0):
    """Noise shaped into syllable-like bursts, deterministic."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    return (rng.standard_normal(t.size) * envelope * 0.2).astype(np.float32)


def level_db(samples):
    rms = np.sqrt(np.mean(np.square(samples.astype(np.float64))))
    return 20 * np.log10(max(rms, 1e-9) / 32768)


@unittest.skipUnless(HAVE_LIVEKIT, "pip install livekit")
class EchoCancellerTests(unittest.TestCase):

    def run_echo(self, seconds=6.0, delay_ms=60, gain=0.4):
        """
        Simulate speakers -> room -> mic: the mic hears the played audio
        delayed and quieter. Returns (mic, cleaned) int16 arrays.
        """
        far_rate, mic_rate, block_ms = 24000, 16000, 30

        played = speech_like(seconds, far_rate)
        # The same sound at the mic's rate, delayed and attenuated.
        heard = np.interp(
            np.arange(int(seconds * mic_rate)) / mic_rate,
            np.arange(played.size) / far_rate,
            played,
        )
        delay = mic_rate * delay_ms // 1000
        heard = np.concatenate([np.zeros(delay), heard[:-delay]]) * gain
        mic = (heard * 32767).astype(np.int16)

        canceller = EchoCanceller(delay_ms=delay_ms)
        far_block = far_rate * block_ms // 1000
        mic_block = mic_rate * block_ms // 1000
        cleaned = []

        for i in range(mic.size // mic_block):
            canceller.playback(played[i * far_block:(i + 1) * far_block], far_rate)
            frame = mic[i * mic_block:(i + 1) * mic_block].tobytes()
            cleaned.append(np.frombuffer(canceller.process(frame, mic_rate), np.int16))

        return mic, np.concatenate(cleaned)

    def test_removes_echo_of_played_audio(self):
        mic, cleaned = self.run_echo()

        # After a couple of seconds to adapt, the echo is mostly gone.
        tail = slice(16000 * 3, None)
        reduction = level_db(mic[tail]) - level_db(cleaned[tail])
        self.assertGreater(reduction, 20, f"only {reduction:.1f} dB")

    def test_keeps_sound_that_was_not_played(self):
        canceller = EchoCanceller()
        voice = (speech_like(3.0, 16000, seed=1) * 32767).astype(np.int16)
        cleaned = []

        for i in range(voice.size // 480):
            canceller.playback(np.zeros(720, np.float32), 24000)  # speakers silent
            frame = voice[i * 480:(i + 1) * 480].tobytes()
            cleaned.append(np.frombuffer(canceller.process(frame, 16000), np.int16))

        cleaned = np.concatenate(cleaned)
        tail = slice(16000, None)
        self.assertLess(abs(level_db(voice[tail]) - level_db(cleaned[tail])), 6)

    def test_frame_must_be_whole_10ms(self):
        with self.assertRaises(ValueError):
            EchoCanceller().process(bytes(2 * 100), 16000)

    def test_closed_canceller_passes_audio_through(self):
        canceller = EchoCanceller()
        canceller.close()
        canceller.close()  # idempotent

        frame = np.arange(480, dtype=np.int16).tobytes()
        self.assertEqual(canceller.process(frame, 16000), frame)

    def test_exit_without_close_is_silent(self):
        # LiveKit's own atexit hook shuts its library down; an unreleased
        # handle used to print "Exception ignored ... AssertionError".
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import numpy as np\n"
            "from voicekit.echo import EchoCanceller\n"
            "aec = EchoCanceller()\n"
            "aec.playback(np.zeros(240, np.float32), 24000)\n"
            "aec.process(bytes(960), 16000)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=root,
            capture_output=True, text=True, timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Exception ignored", result.stderr)
        self.assertNotIn("AssertionError", result.stderr)

    def test_played_audio_is_bounded_when_nobody_listens(self):
        canceller = EchoCanceller()
        for _ in range(2000):
            canceller.playback(np.zeros(240, np.float32), 24000)

        self.assertLessEqual(len(canceller._played), 500)


class MissingLivekitTests(unittest.TestCase):

    def test_helpful_error(self):
        from unittest import mock

        with mock.patch.dict(sys.modules, {"livekit": None}):
            with self.assertRaisesRegex(ImportError, "voicekit\\[aec\\]"):
                EchoCanceller()


if __name__ == "__main__":
    unittest.main(verbosity=2)
