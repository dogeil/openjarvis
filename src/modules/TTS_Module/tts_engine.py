import sys
from typing import Optional

import pyttsx3


class TTSEngine:
    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        driver: Optional[str] = None,
        reuse_engine: bool = True,
        voice_index: Optional[int] = None,
    ):
        """
        Initialize a cross-platform TTS engine.

        - On Windows, default to 'sapi5' (if available).
        - On other platforms, let pyttsx3 pick the default driver.
        """
        self._driver = driver
        if self._driver is None and sys.platform.startswith("win"):
            self._driver = "sapi5"

        self._rate = rate
        self._volume = volume
        self._reuse_engine = reuse_engine
        self._voice_index = voice_index
        self.engine = None

        if self._reuse_engine:
            self._init_engine()

    def _init_engine(self) -> None:
        try:
            self.engine = pyttsx3.init(self._driver)
        except Exception as e:
            # In CI or unsupported environments, fall back to a no-op engine.
            print(f"[TTS Warning] pyttsx3 initialization failed ({e}). Falling back to silent mode.")
            self.engine = None
            return

        self.engine.setProperty("rate", self._rate)
        self.engine.setProperty("volume", self._volume)

        voices = self.engine.getProperty("voices")
        if not voices:
            print("[TTS Warning] No voices found on this system.")
            return

        # Choose voice based on explicit index when provided.
        if self._voice_index is not None and 0 <= self._voice_index < len(voices):
            self.engine.setProperty("voice", voices[self._voice_index].id)
        # Otherwise fall back to your previous preference of index 2, then 0.
        elif len(voices) > 2:
            self.engine.setProperty("voice", voices[2].id)
        else:
            self.engine.setProperty("voice", voices[0].id)

    def list_voices(self) -> list[str]:
        """Return a list of human-readable voice descriptions."""
        # Ensure we have an engine at least temporarily.
        if not self._reuse_engine and not self.engine:
            self._init_engine()
        if not self.engine:
            return []

        voices = self.engine.getProperty("voices")
        return [f"{idx}: {v.name} ({v.id})" for idx, v in enumerate(voices)]

    def set_voice_index(self, index: int) -> None:
        """Update the preferred voice index for subsequent speak() calls."""
        self._voice_index = index
        # Reapply immediately if an engine is active.
        if self.engine:
            voices = self.engine.getProperty("voices")
            if 0 <= index < len(voices):
                self.engine.setProperty("voice", voices[index].id)

    def speak(self, text: str) -> None:
        """
        Converts text to audible speech.
        In environments where the TTS engine could not be initialized,
        this degrades to a simple console print.
        """
        print(f"[JARVIS]: {text}")

        if not self._reuse_engine:
            # Most reliable for Windows/SAPI5: fresh engine per utterance.
            self._init_engine()

        if not getattr(self, "engine", None):
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] Failed to speak ({e}).")
        finally:
            # If we aren't reusing, fully tear down after each call.
            if not self._reuse_engine and getattr(self, "engine", None):
                try:
                    self.engine.stop()
                finally:
                    self.engine = None