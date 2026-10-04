"""
Keyboard and screen stand-ins for the microphone and speaker, so agents
can be developed and debugged without voice:

    VoiceSession(agent, recognizer=KeyboardInput(), speaker=SilentSpeaker()).run()

The session's console transcript shows everything that would be spoken.
"""

import sys
from types import SimpleNamespace


class KeyboardInput:
    """Reads one prompt per line from stdin (or a list of lines)."""

    def __init__(self, lines=None, prompt="> "):
        self.lines = lines
        self.prompt = prompt
        self.on_speech_start = None
        self.config = SimpleNamespace(hotwords=None)

    def load(self):
        pass

    def listen(self):
        if self.lines is not None:
            for line in self.lines:
                yield {"text": line}
            return

        while True:
            try:
                line = input(self.prompt if sys.stdin.isatty() else "")
            except EOFError:
                return
            if line.strip():
                yield {"text": line.strip()}

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        pass


class SilentSpeaker:
    """Accepts speech without playing it (the transcript shows it)."""

    def __init__(self):
        self.config = SimpleNamespace(speed=1.0)
        self.is_speaking = False
        self.spoken = []
        self._generation = 0

    def load(self):
        pass

    def speak(self, text, wait=False):
        self.spoken.append(text)
        return True

    def speak_stream(self, text_chunks, wait=True):
        generation = self._generation
        parts = []
        try:
            for chunk in text_chunks:
                if self._generation != generation:
                    return False
                parts.append(chunk)
        finally:
            close = getattr(text_chunks, "close", None)
            if close:
                close()
        if "".join(parts).strip():
            self.spoken.append("".join(parts))
        return True

    def stop(self):
        self._generation += 1

    def wait(self, timeout=None):
        return True

    def close(self):
        pass
