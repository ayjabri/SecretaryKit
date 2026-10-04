"""
Text-to-speech with Kokoro, built for speaking LLM replies aloud.

Pipeline
--------
text / LLM token stream
    -> SentenceSplitter     (speakable chunks as soon as a sentence ends)
    -> synthesis thread     (Kokoro on the GPU, one chunk at a time)
    -> audio output stream  (gapless playback from a buffer)

Speech starts after the first sentence instead of after the whole reply,
and stop() silences it immediately (e.g. when the user interrupts).

Example
-------
    with KokoroSpeaker() as speaker:
        speaker.speak("Hello! How can I help?", wait=True)

        # Speak an LLM reply while it is still being generated.
        speaker.speak_stream(llm_token_iterator)
"""

import logging
import queue
import re
import threading
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)


KOKORO_SAMPLE_RATE = 24000


@dataclass
class TextToSpeechConfig:
    """
    Configuration for KokoroSpeaker.

    Voice
    -----
    voice
        Kokoro voice name, e.g. "af_heart", "af_bella", "am_michael",
        "bf_emma". Blend voices with commas: "af_heart,af_bella".
        The first letter of the voice must match lang_code.

    lang_code
        "a" American English, "b" British English, "e" Spanish,
        "f" French, "h" Hindi, "i" Italian, "j" Japanese,
        "p" Brazilian Portuguese, "z" Mandarin.

    speed
        Speaking rate. 1.0 is normal.

    volume
        Output gain. 1.0 is unchanged.

    Model
    -----
    repo_id
        Hugging Face repository with the Kokoro weights and voices.

    device
        "cuda" or "cpu".

    Playback
    --------
    output_device
        SoundDevice output device ID or partial name.
        None uses the system default.

    sentence_gap_ms
        Silence inserted between spoken chunks.

    Chunking
    --------
    first_chunk_min_chars
        The first chunk is spoken as soon as a sentence of at least this
        length is complete, so speech starts quickly.

    min_chunk_chars
        Later sentences are merged until they reach this length, which
        gives Kokoro more context for natural intonation.

    max_chunk_chars
        Text without sentence breaks is cut at this length.

    code_block_text
        Spoken in place of fenced code blocks. "" skips them silently.
    """

    voice: str = "af_heart"
    lang_code: str = "a"
    speed: float = 1.0
    volume: float = 1.0

    repo_id: str = "hexgrad/Kokoro-82M"
    device: str = "cuda"

    output_device: Optional[Union[int, str]] = None
    sentence_gap_ms: int = 80

    first_chunk_min_chars: int = 20
    min_chunk_chars: int = 80
    max_chunk_chars: int = 400

    code_block_text: str = "I've put the code on screen."

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.voice:
            raise ValueError("voice is required")

        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")

        if self.volume < 0:
            raise ValueError("volume must be >= 0")

        if self.sentence_gap_ms < 0:
            raise ValueError("sentence_gap_ms must be >= 0")

        if not (
            0 < self.first_chunk_min_chars
            <= self.min_chunk_chars
            < self.max_chunk_chars
        ):
            raise ValueError(
                "require 0 < first_chunk_min_chars <= min_chunk_chars "
                "< max_chunk_chars"
            )


# --------------------------------------------------
# Text preparation
# --------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_RULE_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3}|~~)(\S.*?\S|\S)\1")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "️‍"
    "]+"
)


def prepare_for_speech(text, code_block_text=""):
    """
    Convert LLM output (often markdown) into plain text that sounds
    natural when read aloud.
    """

    text = _THINK_RE.sub(" ", text)
    text = _CODE_BLOCK_RE.sub(f"\n{code_block_text}\n", text)
    # An unclosed fence: drop everything after it.
    text = text.split("```", 1)[0]
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("a link", text)
    text = _TABLE_RULE_RE.sub("", text)
    text = _HORIZONTAL_RULE_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = text.replace("|", ", ")
    # Leftover markup; "#" is kept after a letter, as in "C#".
    text = re.sub(r"[*_]+|(?<!\w)#+", " ", text)

    # Each line (heading, bullet, table row) becomes its own sentence so
    # Kokoro pauses between them instead of running them together.
    lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line).strip(" ,")
        if not line:
            continue
        if not re.search(r"[.!?:;…]['\")\]]*$", line):
            line += "."
        lines.append(line)

    return " ".join(lines)


# Words whose trailing period does not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "approx", "fig", "inc", "ltd",
}

_SENTENCE_END_RE = re.compile(r"[.!?…]+['\")\]]*(?=\s)|\n+")


class SentenceSplitter:
    """
    Incrementally splits streamed text into chunks worth speaking.

    Fenced code blocks are held back until they close and then replaced
    as a whole, so their contents are never split or read out.
    """

    def __init__(
        self,
        first_chunk_min_chars=20,
        min_chunk_chars=80,
        max_chunk_chars=400,
        code_block_text="",
    ):
        self.first_chunk_min_chars = first_chunk_min_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.code_block_text = code_block_text
        self._buffer = ""
        self._emitted = False

    def feed(self, text):
        """Add streamed text; return any chunks that are ready."""

        self._buffer += text
        self._buffer = _CODE_BLOCK_RE.sub(
            f"\n{self.code_block_text}\n", self._buffer
        )
        self._buffer = _CLOSED_THINK_RE.sub(" ", self._buffer)

        chunks = []

        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            chunks.append(chunk)

        return chunks

    def flush(self):
        """Return whatever text remains at the end of the stream."""

        remainder = self._buffer
        self._buffer = ""

        if remainder.strip():
            self._emitted = True
            return [remainder]

        return []

    def _next_chunk(self):

        # Text inside an unclosed code fence or think block can't be
        # spoken yet.
        searchable = self._buffer
        for opener in ("```", "<think>"):
            index = searchable.lower().find(opener)
            if index != -1:
                searchable = searchable[:index]

        min_chars = (
            self.min_chunk_chars if self._emitted else self.first_chunk_min_chars
        )

        cut = None

        for match in _SENTENCE_END_RE.finditer(searchable):
            end = match.end()

            if match.group().startswith(".") and self._is_abbreviation(
                searchable, match.start()
            ):
                continue

            if len(searchable[:end].strip()) >= min_chars:
                cut = end
                break

        if cut is None and len(searchable) > self.max_chunk_chars:
            window = searchable[: self.max_chunk_chars]
            cut = max(window.rfind(", "), window.rfind("; "), window.rfind(" "))
            if cut <= 0:
                cut = self.max_chunk_chars

        if cut is None:
            return None

        chunk = self._buffer[:cut]
        self._buffer = self._buffer[cut:]

        if not chunk.strip():
            return None

        self._emitted = True
        return chunk

    @staticmethod
    def _is_abbreviation(text, period_index):
        word = re.search(r"([\w.]+)$", text[:period_index])
        return bool(word) and word.group(1).lower() in _ABBREVIATIONS


# --------------------------------------------------
# Speaker
# --------------------------------------------------

class KokoroSpeaker:
    """
    Speaks text through Kokoro with streaming, gapless playback.

    Thread-safe: speak(), speak_stream(), stop() and wait() may be called
    from any thread.
    """

    def __init__(self, config=None):

        self.config = config or TextToSpeechConfig()

        self.output_device = self._resolve_output_device(
            self.config.output_device
        )

        self._pipeline = None
        self._stream = None
        self._worker = None
        self._load_lock = threading.Lock()
        # Kokoro is not thread-safe; one inference at a time.
        self._model_lock = threading.Lock()

        # Guards everything below; held only briefly (the audio callback
        # takes it too).
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = 0
        self._text_queue = queue.Queue()
        self._audio = deque()
        self._audio_pos = 0
        self._idle = threading.Event()
        self._idle.set()

        self._gap = np.zeros(
            KOKORO_SAMPLE_RATE * self.config.sentence_gap_ms // 1000,
            dtype=np.float32,
        )

        # Called with every block handed to the sound card, silence
        # included: on_playback(samples, sample_rate). For echo
        # cancellation (voicekit.echo). Runs on the audio thread, so it
        # must be quick.
        self.on_playback = None

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def load(self):
        """Load Kokoro, warm it up and open the audio output."""

        with self._load_lock:
            if self._pipeline is not None:
                return

            self._pipeline = self._create_pipeline()
            self._voice = self._resolve_voice(self.config.voice)

            # The first inference is slow (CUDA kernels compile).
            for _ in self._pipeline("Warming up.", voice=self._voice):
                pass

            self._stream = sd.OutputStream(
                device=self.output_device,
                samplerate=KOKORO_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()

            self._worker = threading.Thread(
                target=self._synthesis_loop,
                name="kokoro-synthesis",
                daemon=True,
            )
            self._worker.start()

            logger.info(
                "Speaker ready: voice '%s' on [%s] %s",
                self.config.voice,
                self.output_device,
                sd.query_devices(self.output_device)["name"],
            )

    def close(self):
        """Stop speaking and release the audio device."""

        self.stop()

        if self._worker is not None:
            self._text_queue.put(None)
            self._worker.join(timeout=5)
            self._worker = None

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._pipeline = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc_info):
        self.close()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    @property
    def output_latency(self):
        """Seconds from audio leaving this class to the speakers (0 if closed)."""
        return self._stream.latency if self._stream is not None else 0.0

    @property
    def is_speaking(self):
        """True while text is queued, being synthesised or playing."""
        return not self._idle.is_set()

    def speak(self, text, wait=False):
        """
        Queue text to be spoken. Markdown is cleaned up first.

        Returns immediately unless wait=True.
        """

        self.load()

        with self._lock:
            generation = self._generation

        splitter = self._new_splitter()
        chunks = splitter.feed(text) + splitter.flush()

        for chunk in chunks:
            self._enqueue(generation, chunk)

        if wait:
            return self.wait()

        return True

    def speak_stream(self, text_chunks: Iterable[str], wait=True):
        """
        Speak text while it is still arriving, e.g. tokens streamed from
        an LLM. Speech starts as soon as the first sentence is complete.

        Stops consuming `text_chunks` (and closes it, if it is a
        generator) when stop() is called.

        Returns True if everything was spoken, False if interrupted.
        """

        self.load()

        with self._lock:
            generation = self._generation

        splitter = self._new_splitter()

        try:
            for text in text_chunks:
                if self._generation != generation:
                    return False

                for chunk in splitter.feed(text):
                    self._enqueue(generation, chunk)

            for chunk in splitter.flush():
                self._enqueue(generation, chunk)

        finally:
            close = getattr(text_chunks, "close", None)
            if close is not None:
                close()

        if self._generation != generation:
            return False

        if wait:
            return self.wait()

        return True

    def stop(self):
        """Silence speech immediately and discard anything queued."""

        with self._lock:
            self._generation += 1
            self._audio.clear()
            self._audio_pos = 0

            while True:
                try:
                    item = self._text_queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    self._pending -= 1
                else:
                    # Keep a shutdown request.
                    self._text_queue.put(None)
                    break

            self._update_idle()

    def wait(self, timeout=None):
        """
        Block until speech finishes or is stopped.

        Returns True if it finished, False if stopped or timed out.
        """

        with self._lock:
            generation = self._generation

        finished = self._idle.wait(timeout)
        return finished and self._generation == generation

    def synthesize(self, text):
        """Return speech audio (24 kHz float32) without playing it."""

        self.load()

        text = prepare_for_speech(text, self.config.code_block_text)

        with self._model_lock:
            parts = [
                result.audio.cpu().numpy()
                for result in self._pipeline(
                    text,
                    voice=self._voice,
                    speed=self.config.speed,
                    split_pattern=None,
                )
                if result.audio is not None
            ]

        if not parts:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(parts).astype(np.float32) * self.config.volume

    def save(self, text, path):
        """Write speech for `text` to a WAV file."""

        import soundfile

        soundfile.write(path, self.synthesize(text), KOKORO_SAMPLE_RATE)

    @staticmethod
    def list_output_devices():

        return [
            {"id": index, "name": device["name"]}
            for index, device in enumerate(sd.query_devices())
            if device["max_output_channels"] > 0
        ]

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------

    def _create_pipeline(self):

        # Imported lazily: torch and Kokoro take a few seconds to load.
        from huggingface_hub import hf_hub_download
        from kokoro import KModel, KPipeline

        def fetch(filename):
            # Use the local cache without contacting Hugging Face; only
            # download files that have never been fetched.
            try:
                return hf_hub_download(
                    self.config.repo_id, filename, local_files_only=True
                )
            except Exception:
                logger.info("Downloading %s from Hugging Face...", filename)
                return hf_hub_download(self.config.repo_id, filename)

        self._fetch = fetch

        model_file = KModel.MODEL_NAMES.get(
            self.config.repo_id, "kokoro-v1_0.pth"
        )

        logger.info("Loading Kokoro (%s)...", self.config.repo_id)

        with warnings.catch_warnings():
            # Kokoro triggers harmless PyTorch deprecation warnings.
            warnings.simplefilter("ignore")

            model = KModel(
                repo_id=self.config.repo_id,
                config=fetch("config.json"),
                model=fetch(model_file),
            )
            model = model.to(self.config.device).eval()

            return KPipeline(
                lang_code=self.config.lang_code,
                repo_id=self.config.repo_id,
                model=model,
            )

    def _resolve_voice(self, voice):
        """Map voice names to local .pt files (blends stay comma-joined)."""

        paths = []

        for name in voice.split(","):
            name = name.strip()

            if not name.endswith(".pt"):
                if name[0].lower() != self.config.lang_code.lower():
                    logger.warning(
                        "Voice '%s' does not match lang_code '%s'.",
                        name,
                        self.config.lang_code,
                    )
                name = self._fetch(f"voices/{name}.pt")

            paths.append(name)

        return ",".join(paths)

    def _resolve_output_device(self, device):

        if device is None:
            try:
                return sd.query_devices(kind="output")["index"]
            except sd.PortAudioError as exc:
                raise RuntimeError("No default audio output found.") from exc

        if isinstance(device, int):
            sd.query_devices(device, "output")
            return device

        matches = [
            d for d in self.list_output_devices()
            if str(device).lower() in d["name"].lower()
        ]

        if len(matches) != 1:
            names = "\n".join(f'  {d["id"]}: {d["name"]}' for d in matches)
            raise ValueError(
                f"{len(matches)} output devices match '{device}'"
                + (f":\n{names}\nUse the device ID instead." if names else ".")
            )

        return matches[0]["id"]

    def _new_splitter(self):
        return SentenceSplitter(
            first_chunk_min_chars=self.config.first_chunk_min_chars,
            min_chunk_chars=self.config.min_chunk_chars,
            max_chunk_chars=self.config.max_chunk_chars,
            code_block_text=self.config.code_block_text,
        )

    # --------------------------------------------------
    # Synthesis (worker thread)
    # --------------------------------------------------

    def _enqueue(self, generation, chunk):

        text = prepare_for_speech(chunk, self.config.code_block_text)

        if not re.search(r"\w", text):
            return

        with self._lock:
            if generation != self._generation:
                return
            self._pending += 1
            self._idle.clear()
            self._text_queue.put((generation, text))

    def _synthesis_loop(self):

        while True:
            item = self._text_queue.get()

            if item is None:
                return

            generation, text = item

            try:
                self._synthesize_chunk(generation, text)
            except Exception:
                logger.exception("Speech synthesis failed for %r", text)
            finally:
                # Counted in _pending even if stop() ran meanwhile; stop()
                # only uncounts items it removed from the queue.
                with self._lock:
                    self._pending -= 1
                    self._update_idle()

    def _synthesize_chunk(self, generation, text):

        logger.debug("Synthesising: %r", text)

        results = self._pipeline(
            text,
            voice=self._voice,
            speed=self.config.speed,
            split_pattern=None,
        )

        while True:
            if self._generation != generation:
                return

            # Kokoro does the work lazily as the generator advances.
            with self._model_lock:
                result = next(results, None)

            if result is None:
                return

            if result.audio is None:
                continue

            audio = result.audio.cpu().numpy().astype(np.float32)

            if self.config.volume != 1.0:
                audio = audio * self.config.volume

            with self._lock:
                if generation != self._generation:
                    return
                self._audio.append(audio)
                if self._gap.size:
                    self._audio.append(self._gap)

    # --------------------------------------------------
    # Playback (audio callback thread)
    # --------------------------------------------------

    def _audio_callback(self, outdata, frames, time_info, status):

        if status.output_underflow:
            logger.debug("Audio output underflow.")

        out = outdata[:, 0]
        filled = 0

        with self._lock:
            while filled < frames and self._audio:
                chunk = self._audio[0]
                take = min(frames - filled, chunk.size - self._audio_pos)
                out[filled:filled + take] = chunk[
                    self._audio_pos:self._audio_pos + take
                ]
                filled += take
                self._audio_pos += take

                if self._audio_pos >= chunk.size:
                    self._audio.popleft()
                    self._audio_pos = 0

            self._update_idle()

        out[filled:] = 0
        np.clip(out, -1.0, 1.0, out=out)

        on_playback = self.on_playback
        if on_playback is not None:
            try:
                on_playback(out, KOKORO_SAMPLE_RATE)
            except Exception:
                logger.exception("on_playback failed")

    def _update_idle(self):
        # Caller holds self._lock.
        if self._pending == 0 and not self._audio:
            self._idle.set()
        else:
            self._idle.clear()


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    text = " ".join(sys.argv[1:]) or (
        "Hello! This is Kokoro speaking. "
        "I can read replies from your language model as they stream in, "
        "so you don't have to wait for the whole answer."
    )

    with KokoroSpeaker() as speaker:
        speaker.speak(text, wait=True)
