import multiprocessing
import threading
import uvicorn
import time
import os
import sys

# Ensure the 'src' directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from modules.Hand_Module.hand_engine import HandEngine
from modules.STT_Module.stt_engine import STTEngine
from modules.TTS_Module.tts_engine import TTSEngine

def run_hand_module():
    print("[Launcher] Starting Hand Module...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models") 
    try:
        hand_module = HandEngine(model_dir=model_path)
        hand_module.start()
    except Exception as e:
        print(f"[Hand Module Error] {e}")

def run_stt_module():
    print("[Launcher] Starting STT Module (voice)...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "vosk-model") 

    def voice_loop() -> None:
        try:
            stt_engine = STTEngine(model_path=model_path)
            for text in stt_engine.listen():
                if text:
                    print(f"[STT User Said]: {text}")
        except Exception as e:
            print(f"[STT Error] {e}")

    voice_loop()


def run_server():
    print("[Launcher] Starting FastAPI Brain...")
    # FastAPI app is defined in src/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

def say_greeting():
    """Simple startup greeting"""
    try:
        tts = TTSEngine()
        tts.speak("All systems are online. JARVIS is ready.")
    except Exception as e:
        print(f"[TTS Greeting Error] {e}")


def console_loop():
    """Read commands from the main console."""
    print("[Console] Type commands here. Use /quit to stop Jarvis.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Console] Input closed.")
            break

        if not line:
            continue
        if line.lower() == "/quit":
            print("[Console] Stopping Jarvis by user request.")
            break

        print(f"[Console User Typed]: {line}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("--- INITIALIZING JARVIS CORE ---")

    # Define the processes
    hand_process = multiprocessing.Process(target=run_hand_module, name="Jarvis-Hand")
    stt_process = multiprocessing.Process(target=run_stt_module, name="Jarvis-STT")
    server_process = multiprocessing.Process(target=run_server, name="Jarvis-API")

    # Start all systems
    hand_process.start()
    stt_process.start()
    server_process.start()

    shutdown_event = threading.Event()

    def _console_worker() -> None:
        try:
            console_loop()
        finally:
            shutdown_event.set()

    console_thread = threading.Thread(
        target=_console_worker, name="Jarvis-Console", daemon=True
    )
    console_thread.start()

    # Give them a second to initialize before speaking
    time.sleep(2)
    say_greeting()

    try:
        while not shutdown_event.is_set():
            # If ESC is pressed in the Hand window, Hand module exits; we treat that
            # as a full Jarvis shutdown trigger.
            if not all(
                [hand_process.is_alive(), stt_process.is_alive(), server_process.is_alive()]
            ):
                print("[Launcher] A critical process died. Shutting down...")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[Launcher] Manual shutdown initiated.")
    finally:
        shutdown_event.set()
        hand_process.terminate()
        stt_process.terminate()
        server_process.terminate()
        print("[Launcher] All systems offline.")