"""
Continuous microphone speech-to-text, tuned for producing clean prompts
for an LLM.

Pipeline
--------
microphone -> capture thread   (WebRTC VAD + energy gate; splits speech
                                into utterances and whole prompts)
           -> listening thread (faster-whisper transcription, text cleanup)

Audio capture never stops while Whisper is busy, so nothing you say is
lost; finished speech simply queues until it has been transcribed.

Example
-------
    with ContinuousSpeechRecognizer() as recognizer:
        for prompt in recognizer.listen():
            send_to_llm(prompt["text"])
"""

import contextlib
import importlib.util
import itertools
import logging
import math
import os
import pathlib
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import sounddevice as sd
import webrtcvad


logger = logging.getLogger(__name__)


WHISPER_SAMPLE_RATE = 16000

# Raise an error if the microphone delivers nothing for this long.
NO_AUDIO_TIMEOUT_SECONDS = 3.0

# Phrases Whisper commonly invents from breaths, clicks and other
# non-speech. They are only discarded when they are the *entire*
# transcript, so they are still kept inside a longer sentence.
DEFAULT_HALLUCINATION_PHRASES = (
    "thank you",
    "thanks",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "you",
    "bye",
    "subtitles by the amaraorg community",
)


@dataclass
class SpeechRecognizerConfig:
    """
    Configuration for ContinuousSpeechRecognizer.

    Microphone
    ----------
    mic_device
        SoundDevice input device ID or partial device name.
        None uses the system default input device.

    sample_rate
        Microphone sample rate. WebRTC VAD supports
        8000, 16000, 32000 and 48000 Hz. Audio is resampled
        to 16 kHz before it reaches Whisper.

    frame_ms
        VAD frame duration. Must be 10, 20 or 30 ms.

    Speech detection
    ----------------
    vad_mode
        WebRTC VAD aggressiveness:
            0 = least aggressive
            3 = most aggressive (may clip quiet word endings)

    noise_floor_dbfs
        Explicit ambient noise level.
        None enables automatic calibration on first use.

    noise_margin_db
        Required signal level above ambient noise.

    absolute_min_dbfs
        Absolute minimum audio level considered speech.

    calibration_seconds
        Ambient-noise calibration duration.

    max_noise_floor_dbfs
        A calibrated noise floor above this logs a warning
        (usually means someone spoke during calibration).

    Segmentation
    ------------
    speech_start_ms
        Continuous detected speech required to begin an utterance.

    speech_end_ms
        Silence that ends an utterance (a short pause).

    prompt_end_ms
        Silence that ends a whole prompt. listen() yields once
        you have been quiet for this long.

    min_speech_ms
        Minimum detected speech required to keep an utterance.

    pre_roll_ms
        Audio retained from before speech was detected.

    max_speech_seconds
        Continuous speech longer than this is split into
        chunks. Nothing is dropped.

    Transcription
    -------------
    model_name
        Faster-Whisper model.

    device
        Faster-Whisper compute device ("cuda", "cpu" or "auto").

    compute_type
        CTranslate2 computation type.

    language
        Whisper language code. None auto-detects.

    beam_size
        Whisper beam size. 1 is fastest; 5 is more accurate.

    initial_prompt
        Text that primes Whisper's spelling and punctuation style,
        e.g. "Claude, Python, faster-whisper, CUDA, LLM."

    hotwords
        Words Whisper should favour, e.g. names and jargon.

    no_speech_threshold, log_prob_threshold, compression_ratio_threshold
        Whisper's own filters for silent, low-confidence or
        repetitive (hallucinated) segments.

    hallucination_phrases
        Transcripts consisting only of one of these phrases are
        discarded. Use () to disable.

    remove_fillers
        Strip filler words such as "um" and "uh".

    text_postprocessor
        Optional callable applied to each finished prompt, e.g. a
        cheap LLM pass that fixes punctuation.
    """

    mic_device: Optional[Union[int, str]] = None
    sample_rate: int = 16000
    frame_ms: int = 30

    vad_mode: int = 2
    noise_floor_dbfs: Optional[float] = None
    noise_margin_db: float = 10.0
    absolute_min_dbfs: float = -50.0
    calibration_seconds: float = 1.0
    max_noise_floor_dbfs: float = -40.0

    speech_start_ms: int = 150
    speech_end_ms: int = 500
    prompt_end_ms: int = 1500
    min_speech_ms: int = 300
    pre_roll_ms: int = 300
    max_speech_seconds: float = 30

    model_name: str = "turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: Optional[str] = "en"
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    hallucination_phrases: tuple = DEFAULT_HALLUCINATION_PHRASES
    remove_fillers: bool = True
    text_postprocessor: Optional[Callable[[str], str]] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                "sample_rate must be 8000, 16000, 32000 or 48000"
            )

        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20 or 30")

        if self.vad_mode not in (0, 1, 2, 3):
            raise ValueError("vad_mode must be 0, 1, 2 or 3")

        if self.speech_start_ms < self.frame_ms:
            raise ValueError("speech_start_ms must be >= frame_ms")

        if self.speech_end_ms < self.frame_ms:
            raise ValueError("speech_end_ms must be >= frame_ms")

        if self.prompt_end_ms < self.speech_end_ms:
            raise ValueError("prompt_end_ms must be >= speech_end_ms")

        if self.max_speech_seconds <= 0:
            raise ValueError("max_speech_seconds must be > 0")

        if self.noise_floor_dbfs is None and self.calibration_seconds <= 0:
            raise ValueError(
                "calibration_seconds must be > 0 "
                "when noise_floor_dbfs is None"
            )

        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")


# --------------------------------------------------
# Text cleanup
# --------------------------------------------------

_FILLER_RE = re.compile(
    r",?\s*\b(?:u+m+|u+h+m*|e+r+m+|h+m+|a+h+)\b,?",
    re.IGNORECASE,
)

# (?<!\.) leaves text after an ellipsis alone.
_SENTENCE_START_RE = re.compile(r"(^|(?<!\.)[.!?]\s+)([a-z])")

_ABBREVIATION_END_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs)\.\s+$")


def clean_text(text, remove_fillers=True):
    """Tidy a raw Whisper transcript for use as an LLM prompt."""

    if remove_fillers:
        text = _FILLER_RE.sub(" ", text)

    text = re.sub(r"\s+", " ", text)
    # "word ," -> "word,"
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # ", ." -> "."   and   ". ," -> "."
    text = re.sub(r"[,;:]\s*([.!?])", r"\1", text)
    text = re.sub(r"([.!?])\s*[,;:]", r"\1", text)
    # ". ." -> "." (an ellipsis has no spaces, so it is kept)
    text = re.sub(r"([.!?])(?:\s+[.!?])+", r"\1", text)
    text = re.sub(r"^[\s,;:.!?]+", "", text)
    text = re.sub(r"[\s,;:]+$", "", text)

    def capitalize(match):
        if _ABBREVIATION_END_RE.search(text[: match.end(1)]):
            return match.group(0)
        return match.group(1) + match.group(2).upper()

    return _SENTENCE_START_RE.sub(capitalize, text)


def _normalize_phrase(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()


# --------------------------------------------------
# Audio helpers
# --------------------------------------------------

def _to_whisper_audio(audio, sample_rate):
    """Convert int16 PCM bytes or a numpy array to 16 kHz mono float32."""

    if isinstance(audio, (bytes, bytearray, memoryview)):
        audio = np.frombuffer(audio, dtype=np.int16)

    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    if sample_rate != WHISPER_SAMPLE_RATE and audio.size:
        # FFT resampling: irfft truncates (or zero-pads) the spectrum,
        # which band-limits the signal, so downsampling does not alias.
        n_in = audio.size
        n_out = round(n_in * WHISPER_SAMPLE_RATE / sample_rate)
        audio = np.fft.irfft(np.fft.rfft(audio), n_out) * (n_out / n_in)
        audio = audio.astype(np.float32)

    return audio


def _dbfs(frame):

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)

    if samples.size == 0:
        return -100.0

    rms = np.sqrt(np.mean(samples * samples))

    if rms <= 1.0:
        return -100.0

    return 20.0 * math.log10(rms / 32768.0)


# --------------------------------------------------
# CUDA (Windows)
# --------------------------------------------------

_dll_handles = []


def _add_cuda_dll_directories():
    """
    Make cuBLAS/cuDNN DLLs visible to CTranslate2 on Windows.

    Uses the nvidia-*-cu12 pip packages when installed, falling back
    to PyTorch's bundled copies. Nothing is imported; the package
    folders are only located, so torch is never loaded.
    """

    if os.name != "nt" or _dll_handles:
        return

    directories = []

    for package in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            continue

        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                directories.append(pathlib.Path(location) / "bin")

    torch_spec = importlib.util.find_spec("torch")

    if torch_spec and torch_spec.origin:
        directories.append(pathlib.Path(torch_spec.origin).parent / "lib")

    for directory in directories:
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = (
                str(directory) + os.pathsep + os.environ.get("PATH", "")
            )


# --------------------------------------------------
# Segmentation
# --------------------------------------------------

@dataclass
class _Utterance:
    pcm: bytes
    speech_ms: int
    # True when cut at max_speech_seconds; speech continues in the
    # next utterance.
    split: bool


@dataclass
class _CaptureError:
    error: BaseException


_END_OF_PROMPT = object()

# Emitted once per utterance, as soon as it has min_speech_ms of speech
# (so coughs and clicks don't trigger it).
_SPEECH_STARTED = object()


class _Segmenter:
    """
    Turns a stream of (frame, is_voice) pairs into utterances and
    end-of-prompt markers.

    Pure logic with no audio I/O, so it can be tested with synthetic
    frames.
    """

    def __init__(
        self,
        frame_ms,
        start_frames,
        end_frames,
        min_speech_frames,
        pre_roll_frames,
        max_frames,
        prompt_end_frames,
    ):
        self.frame_ms = frame_ms
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.min_speech_frames = min_speech_frames
        self.max_frames = max_frames
        self.prompt_end_frames = prompt_end_frames

        # The frames that trigger speech detection also live in this
        # buffer, so it must hold them on top of the requested pre-roll.
        self._pre_roll = deque(maxlen=pre_roll_frames + start_frames)

        self.reset()

    def reset(self):
        self._pre_roll.clear()
        self._recording = None
        self._uncounted_start = 0
        self._consecutive_voice = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_confirmed = False
        self._continuation = False
        self._prompt_open = False
        self._quiet_frames = 0

    @property
    def active(self):
        """True while speech is being recorded or a prompt is open."""
        return self._recording is not None or self._prompt_open

    def feed(self, frame, is_voice):

        if self._recording is None:
            return self._feed_idle(frame, is_voice)

        self._recording.append(frame)

        if is_voice:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        events = self._confirm_speech()

        if self._silence_frames >= self.end_frames:
            return events + self._finish(split=False)

        if len(self._recording) >= self.max_frames:
            logger.debug("Maximum speech duration reached; splitting.")
            return events + self._finish(split=True)

        return events

    def _confirm_speech(self):

        if (
            not self._speech_confirmed
            and self._speech_frames >= self.min_speech_frames
        ):
            self._speech_confirmed = True
            return [_SPEECH_STARTED]

        return []

    def _feed_idle(self, frame, is_voice):

        self._pre_roll.append(frame)

        if is_voice:
            self._consecutive_voice += 1
        else:
            self._consecutive_voice = 0

        if self._consecutive_voice >= self.start_frames:
            logger.debug("Speech started.")
            self._recording = list(self._pre_roll)
            # Pre-roll frames, except this one, were already counted as
            # quiet while idle.
            self._uncounted_start = len(self._recording) - 1
            self._pre_roll.clear()
            self._speech_frames = self._consecutive_voice
            self._silence_frames = 0
            self._consecutive_voice = 0
            return self._confirm_speech()

        return self._add_quiet(1)

    def _finish(self, split):

        events = []

        keep = self._speech_frames >= self.min_speech_frames or (
            self._continuation and self._speech_frames > 0
        )

        if keep:
            events.append(
                _Utterance(
                    pcm=b"".join(self._recording),
                    speech_ms=self._speech_frames * self.frame_ms,
                    split=split,
                )
            )
            self._prompt_open = True
        else:
            logger.debug("Rejected short audio event.")

        if split:
            self._recording = []
            self._uncounted_start = 0
            self._speech_frames = 0
            self._continuation = True
            return events

        recorded_frames = len(self._recording) - self._uncounted_start
        self._recording = None
        self._continuation = False
        self._speech_confirmed = False

        if keep:
            self._quiet_frames = 0
            events.extend(self._add_quiet(self._silence_frames))
        else:
            # A rejected noise does not interrupt the prompt's silence.
            events.extend(self._add_quiet(recorded_frames))

        return events

    def _add_quiet(self, frames):

        if not self._prompt_open:
            return []

        self._quiet_frames += frames

        if self._quiet_frames >= self.prompt_end_frames:
            self._prompt_open = False
            self._quiet_frames = 0
            return [_END_OF_PROMPT]

        return []


# --------------------------------------------------
# Recognizer
# --------------------------------------------------

class ContinuousSpeechRecognizer:

    def __init__(self, config=None):

        self.config = config or SpeechRecognizerConfig()
        config = self.config

        self.mic_device = self._resolve_microphone(config.mic_device)

        logger.info(
            "Microphone: [%s] %s",
            self.mic_device,
            sd.query_devices(self.mic_device)["name"],
        )

        def to_frames(ms):
            return max(1, math.ceil(ms / config.frame_ms))

        self.frame_samples = config.sample_rate * config.frame_ms // 1000

        self._segmenter = _Segmenter(
            frame_ms=config.frame_ms,
            start_frames=to_frames(config.speech_start_ms),
            end_frames=to_frames(config.speech_end_ms),
            min_speech_frames=to_frames(config.min_speech_ms),
            pre_roll_frames=to_frames(config.pre_roll_ms),
            max_frames=to_frames(config.max_speech_seconds * 1000),
            prompt_end_frames=to_frames(config.prompt_end_ms),
        )

        self._vad = webrtcvad.Vad(config.vad_mode)

        self.noise_floor_dbfs = None
        self.noise_threshold_dbfs = None

        if config.noise_floor_dbfs is not None:
            self._set_noise_floor(config.noise_floor_dbfs)

        self._hallucinations = {
            _normalize_phrase(phrase)
            for phrase in config.hallucination_phrases
        }

        # 60 seconds of audio; the capture thread is light, so this only
        # fills if it has died.
        self._audio_queue = queue.Queue(
            maxsize=60 * 1000 // config.frame_ms
        )
        self._events = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._ready = threading.Event()
        self._listening = False
        self._model = None

        # Called (on the capture thread) as soon as the user has clearly
        # started speaking, before the utterance is transcribed. Use it
        # to interrupt text-to-speech. Must return quickly.
        self.on_speech_start = None

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def load(self):
        """Load the Whisper model (otherwise done on first use)."""

        if self._model is not None:
            return self._model

        if self.config.device in ("cuda", "auto"):
            _add_cuda_dll_directories()

        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model '%s'...", self.config.model_name)

        start = time.perf_counter()

        def create(local_files_only):
            return WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                local_files_only=local_files_only,
            )

        # Use the cached model without asking Hugging Face for updates;
        # only go online if it has never been downloaded.
        try:
            self._model = create(local_files_only=True)
        except Exception:
            logger.info("Model not cached; downloading from Hugging Face...")
            self._model = create(local_files_only=False)

        logger.info(
            "Whisper loaded in %.2f seconds",
            time.perf_counter() - start,
        )

        return self._model

    def close(self):
        """Stop listening and release the Whisper model."""
        self._stop.set()
        self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def pause(self):
        """
        Ignore the microphone until resume() is called, e.g. while a
        text-to-speech reply is playing. Speech in progress is discarded.
        """
        self._paused.set()

    def resume(self):
        self._paused.clear()

    # --------------------------------------------------
    # Public utilities
    # --------------------------------------------------

    @staticmethod
    def list_microphones():

        microphones = []

        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                microphones.append(
                    {
                        "id": index,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "default_sample_rate": device["default_samplerate"],
                    }
                )

        return microphones

    @staticmethod
    def print_microphones():

        print("\nAvailable microphones:\n")

        for mic in ContinuousSpeechRecognizer.list_microphones():
            print(f'{mic["id"]:>3}  {mic["name"]}')

    # --------------------------------------------------
    # Microphone resolution
    # --------------------------------------------------

    def _resolve_microphone(self, device):

        if device is None:
            try:
                return sd.query_devices(kind="input")["index"]
            except sd.PortAudioError as exc:
                raise RuntimeError("No default microphone found.") from exc

        if isinstance(device, int):
            # Raises ValueError if the device has no input channels.
            sd.query_devices(device, "input")
            return device

        search = str(device).lower()

        matches = [
            mic
            for mic in self.list_microphones()
            if search in mic["name"].lower()
        ]

        if not matches:
            raise ValueError(f"No microphone matching '{device}' found.")

        if len(matches) > 1:
            names = "\n".join(
                f'  {m["id"]}: {m["name"]}' for m in matches
            )
            raise ValueError(
                f"Multiple microphones match '{device}':\n"
                f"{names}\n"
                "Use the device ID instead."
            )

        return matches[0]["id"]

    # --------------------------------------------------
    # Audio capture (runs on the capture thread)
    # --------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):

        if status:
            logger.warning("Audio stream status: %s", status)

        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            logger.warning("Audio queue full; dropping audio.")

    def _frames(self):
        """Yield fixed-size VAD frames until stop() is requested."""

        frame_bytes = self.frame_samples * 2
        buffer = b""
        last_audio = time.monotonic()

        while not self._stop.is_set():

            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - last_audio > NO_AUDIO_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        "Microphone stopped delivering audio."
                    )
                continue

            last_audio = time.monotonic()

            # The driver may not honour blocksize exactly; re-chunk.
            buffer += chunk

            while len(buffer) >= frame_bytes:
                yield buffer[:frame_bytes]
                buffer = buffer[frame_bytes:]

    def _capture_loop(self):

        try:
            frames = self._frames()

            if self.noise_threshold_dbfs is None:
                self._calibrate_noise(frames)

            self._ready.set()
            logger.info("Listening...")

            was_paused = False

            for frame in frames:

                if self._paused.is_set():
                    if not was_paused:
                        self._segmenter.reset()
                        was_paused = True
                    continue

                was_paused = False

                for event in self._segmenter.feed(frame, self._is_voice(frame)):
                    if event is _SPEECH_STARTED:
                        self._notify_speech_started()
                    else:
                        self._events.put(event)

        except Exception as exc:
            self._events.put(_CaptureError(exc))

    def _notify_speech_started(self):

        logger.debug("Speech confirmed.")

        if self.on_speech_start is None:
            return

        try:
            self.on_speech_start()
        except Exception:
            logger.exception("on_speech_start callback failed")

    def _is_voice(self, frame):

        if _dbfs(frame) < self.noise_threshold_dbfs:
            return False

        return self._vad.is_speech(frame, self.config.sample_rate)

    # --------------------------------------------------
    # Noise calibration
    # --------------------------------------------------

    def _set_noise_floor(self, noise_floor):

        self.noise_floor_dbfs = noise_floor
        self.noise_threshold_dbfs = max(
            self.config.absolute_min_dbfs,
            noise_floor + self.config.noise_margin_db,
        )

    def _calibrate_noise(self, frames):

        logger.info(
            "Calibrating microphone for %.1f seconds. Please remain quiet.",
            self.config.calibration_seconds,
        )

        frames_needed = math.ceil(
            self.config.calibration_seconds * 1000 / self.config.frame_ms
        )

        levels = [
            _dbfs(frame) for frame in itertools.islice(frames, frames_needed)
        ]

        if not levels:
            return

        # Percentile is more resistant to brief clicks than the maximum.
        self._set_noise_floor(float(np.percentile(levels, 70)))

        logger.info("Ambient noise: %.1f dBFS", self.noise_floor_dbfs)
        logger.info(
            "Speech energy threshold: %.1f dBFS",
            self.noise_threshold_dbfs,
        )

        if self.noise_floor_dbfs > self.config.max_noise_floor_dbfs:
            logger.warning(
                "Ambient noise is high (%.1f dBFS). If anyone spoke "
                "during calibration, quiet speech will be ignored; set "
                "noise_floor_dbfs explicitly or recalibrate.",
                self.noise_floor_dbfs,
            )

    def recalibrate(self):
        """Measure ambient noise again at the start of the next listen."""
        self.noise_floor_dbfs = None
        self.noise_threshold_dbfs = None

    # --------------------------------------------------
    # Listening session
    # --------------------------------------------------

    @contextlib.contextmanager
    def _session(self):

        if self._listening:
            raise RuntimeError("Already listening.")

        # Load the model first so its start-up time is not spent
        # buffering microphone audio.
        self.load()

        self._listening = True
        self._stop.clear()
        self._ready.clear()
        self._segmenter.reset()

        for q in (self._audio_queue, self._events):
            while not q.empty():
                q.get_nowait()

        thread = threading.Thread(
            target=self._capture_loop,
            name="speech-capture",
            daemon=True,
        )

        try:
            with sd.RawInputStream(
                device=self.mic_device,
                samplerate=self.config.sample_rate,
                blocksize=self.frame_samples,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                thread.start()

                try:
                    yield
                finally:
                    self._stop.set()
                    thread.join()
        finally:
            self._listening = False

    def _event_stream(self, timeout, is_idle):
        """
        Yield segmenter events. Returns when nobody has started speaking
        for `timeout` seconds while idle.
        """

        with self._session():

            idle_since = time.monotonic()

            while not self._stop.is_set():

                try:
                    event = self._events.get(timeout=0.1)
                except queue.Empty:
                    busy = (
                        not self._ready.is_set()
                        or self._segmenter.active
                        or not is_idle()
                    )

                    if busy:
                        idle_since = time.monotonic()
                    elif (
                        timeout is not None
                        and time.monotonic() - idle_since >= timeout
                    ):
                        return

                    continue

                if isinstance(event, _CaptureError):
                    raise RuntimeError("Audio capture failed.") from event.error

                yield event

                idle_since = time.monotonic()

    # --------------------------------------------------
    # Public listening API
    # --------------------------------------------------

    def listen(self, timeout=None):
        """
        Continuously listen and yield one cleaned-up prompt each time you
        finish speaking.

        Speech is grouped into one prompt until you have been silent for
        `prompt_end_ms`; the whole prompt is then transcribed in a single
        pass so Whisper sees complete sentences.

        Parameters
        ----------
        timeout
            Stop if nobody starts speaking for this many seconds.
            None listens forever.

        Yields
        ------
        dict
            {
                "text": str,           # cleaned, ready for an LLM
                "raw_text": str,       # Whisper's output before cleanup
                "latency": float,      # transcription time in seconds
                "audio_duration": float,
                "language": str,
                "language_probability": float,
                "noise_floor_dbfs": float,
                "noise_threshold_dbfs": float,
            }
        """

        pending = []

        for event in self._event_stream(timeout, is_idle=lambda: not pending):

            if event is not _END_OF_PROMPT:
                pending.append(event)
                continue

            if not pending:
                continue

            result = self._transcribe_prompt(pending)
            pending = []

            if result["text"]:
                yield result

    def listen_once(self, timeout=None):
        """
        Listen for a single prompt and return it, or None if nobody
        spoke within `timeout` seconds.
        """

        prompts = self.listen(timeout=timeout)

        try:
            return next(prompts, None)
        finally:
            prompts.close()

    def listen_utterances(self, timeout=None):
        """
        Like listen(), but yields each utterance (speech between short
        pauses) as soon as it is transcribed. Lower latency, slightly
        lower accuracy. `text_postprocessor` is not applied.
        """

        context = ""

        for event in self._event_stream(timeout, is_idle=lambda: True):

            if event is _END_OF_PROMPT:
                context = ""
                continue

            result = self.transcribe_audio(
                event.pcm,
                self.config.sample_rate,
                context=context,
            )
            self._add_noise_info(result)

            if result["text"]:
                # Earlier utterances of the same prompt help Whisper with
                # spelling and capitalisation.
                context = (context + " " + result["text"])[-200:]
                yield result

    # --------------------------------------------------
    # Whisper
    # --------------------------------------------------

    def transcribe_audio(self, audio, sample_rate=WHISPER_SAMPLE_RATE, context=None):
        """
        Transcribe int16 PCM bytes or a numpy array.

        Usable on its own, e.g. for audio files or testing.
        """

        model = self.load()
        audio = _to_whisper_audio(audio, sample_rate)

        prompt = " ".join(
            part for part in (self.config.initial_prompt, context) if part
        )

        start = time.perf_counter()

        segments, info = model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            initial_prompt=prompt or None,
            hotwords=self.config.hotwords,
            condition_on_previous_text=False,
            no_speech_threshold=self.config.no_speech_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            vad_filter=False,
        )

        # Faster-Whisper runs lazily.
        segments = list(segments)

        latency = time.perf_counter() - start

        raw_text = "".join(
            segment.text
            for segment in segments
            if any(char.isalnum() for char in segment.text)
        ).strip()

        if _normalize_phrase(raw_text) in self._hallucinations:
            logger.debug("Discarded likely hallucination: %r", raw_text)
            raw_text = ""

        return {
            "text": clean_text(raw_text, self.config.remove_fillers),
            "raw_text": raw_text,
            "latency": latency,
            "audio_duration": audio.size / WHISPER_SAMPLE_RATE,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def _transcribe_prompt(self, utterances):

        pcm = b"".join(utterance.pcm for utterance in utterances)

        logger.debug(
            "Transcribing %.2f sec of audio.",
            len(pcm) / 2 / self.config.sample_rate,
        )

        result = self.transcribe_audio(pcm, self.config.sample_rate)

        if result["text"] and self.config.text_postprocessor:
            result["text"] = self.config.text_postprocessor(result["text"])

        self._add_noise_info(result)

        return result

    def _add_noise_info(self, result):
        result["noise_floor_dbfs"] = self.noise_floor_dbfs
        result["noise_threshold_dbfs"] = self.noise_threshold_dbfs


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with ContinuousSpeechRecognizer() as recognizer:
        for prompt in recognizer.listen():
            print(
                f'\n>>> {prompt["text"]}\n'
                f'    ({prompt["audio_duration"]:.1f}s audio, '
                f'{prompt["latency"]:.2f}s transcription)'
            )
