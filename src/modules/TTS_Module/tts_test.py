import os
import sys


def _ensure_src_on_path() -> None:
    """
    Allow running this file directly:
      python src/modules/TTS_Module/tts_test.py
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main() -> int:
    _ensure_src_on_path()

    from modules.TTS_Module.tts_engine import TTSEngine

    # On Windows, reusing a single pyttsx3 engine can sometimes go silent after
    # the first utterance. Using a fresh engine per line is slower but reliable.
    current_voice_index = None
    tts = TTSEngine(reuse_engine=False, voice_index=current_voice_index)
    print("Jarvis TTS test. Type text and press Enter.")
    print("Commands: /quit, /exit, /voices, /voice <id>")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text.lower() in {"/quit", "/exit"}:
            break

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()

            if cmd == "/voices":
                voices = tts.list_voices()
                if not voices:
                    print("No voices available.")
                else:
                    print("Available voices:")
                    for line in voices:
                        print(" ", line)
                continue

            if cmd == "/voice" and len(parts) == 2:
                try:
                    idx = int(parts[1])
                except ValueError:
                    print("Usage: /voice <numeric_id>")
                    continue

                current_voice_index = idx
                tts.set_voice_index(idx)
                print(f"Voice index set to {idx}.")
                continue

            print("Unknown command. Use /voices, /voice <id>, /quit, /exit.")
            continue

        tts.speak(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

