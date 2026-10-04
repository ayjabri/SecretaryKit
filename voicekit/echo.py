"""
Acoustic echo cancellation: remove the assistant's own voice from the
microphone, so it hears you and not its speakers.

Uses WebRTC's echo canceller (AEC3, the one in Chrome and Google Meet)
through LiveKit's SDK:

    pip install "voicekit[aec]"        # or: pip install livekit

VoiceSession sets it up automatically for interruptible sessions. To use
it on its own:

    aec = EchoCanceller()
    speaker.on_playback = aec.playback        # what the speakers play
    recognizer.echo_canceller = aec           # cleans every mic frame

It learns the echo path (speaker -> room -> mic) by comparing what was
played with what the microphone heard, so it adapts to your speakers,
volume and room within a second or two of speech.
"""

import atexit
import collections
import logging
import weakref

import numpy as np


logger = logging.getLogger(__name__)


class EchoCanceller:
    """
    delay_ms
        Rough time from audio being handed to the speakers until the
        microphone delivers its echo: output latency + input latency.
        Only a starting point; AEC3 measures the real delay itself.

    noise_suppression
        Also apply WebRTC noise suppression. Leave off if the microphone
        already does it (e.g. NVIDIA Broadcast, Krisp).
    """

    def __init__(self, delay_ms=200, noise_suppression=False):

        try:
            from livekit import rtc
        except ImportError as exc:
            raise ImportError(
                'Echo cancellation needs LiveKit: pip install "voicekit[aec]"'
            ) from exc

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
        )
        self._apm.set_stream_delay_ms(int(delay_ms))
        self._closed = False

        # LiveKit shuts its native library down in an atexit hook; a
        # handle released after that fails noisily. atexit runs hooks in
        # reverse order, so this one (registered later) runs first.
        atexit.register(_close_at_exit, weakref.ref(self))

        # Played audio, appended by the speaker's audio thread and drained
        # by the microphone thread. Bounded so it can't grow while nobody
        # is listening (about 5 seconds of typical 10 ms blocks).
        self._played = collections.deque(maxlen=500)
        self._far = np.zeros(0, np.float32)
        self._far_rate = None

    def close(self):
        """Release the native echo canceller. process() then passes audio through."""
        if not self._closed:
            self._closed = True
            # LiveKit's APM has no public close(); dispose its handle.
            handle = getattr(self._apm, "_ffi_handle", None)
            if handle is not None:
                handle.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def set_delay(self, delay_ms):
        self._apm.set_stream_delay_ms(int(delay_ms))

    def playback(self, samples, sample_rate):
        """
        Report audio just handed to the speakers (float32 mono, silence
        included). Safe to call from an audio callback: it only copies.
        """
        self._played.append((np.array(samples, np.float32), sample_rate))

    def process(self, pcm, sample_rate):
        """
        Remove echo from a microphone frame (int16 mono bytes, a multiple
        of 10 ms) and return the cleaned frame.
        """

        if self._closed:
            return pcm

        self._feed_played()

        step = sample_rate // 100
        audio = np.frombuffer(pcm, np.int16)

        if audio.size % step:
            raise ValueError("Microphone frames must be a multiple of 10 ms.")

        out = []

        for start in range(0, audio.size, step):
            frame = self._frame(audio[start:start + step], sample_rate)
            self._apm.process_stream(frame)
            out.append(bytes(frame.data))

        return b"".join(out)

    def _feed_played(self):

        while self._played:
            samples, rate = self._played.popleft()

            if rate != self._far_rate:
                self._far = np.zeros(0, np.float32)
                self._far_rate = rate

            self._far = np.concatenate([self._far, samples])

        if self._far_rate is None:
            return

        step = self._far_rate // 100
        usable = self._far.size // step * step

        for start in range(0, usable, step):
            chunk = self._far[start:start + step]
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            self._apm.process_reverse_stream(self._frame(pcm, self._far_rate))

        self._far = self._far[usable:]

    def _frame(self, samples, sample_rate):
        return self._rtc.AudioFrame(
            samples.tobytes(), sample_rate, 1, len(samples)
        )


def _close_at_exit(ref):
    canceller = ref()
    if canceller is not None:
        canceller.close()
