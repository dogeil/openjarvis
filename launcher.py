import multiprocessing
import uvicorn
import time
import os
import sys

# Ensure the 'src' directory is in the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from modules.Hand_Module.hand_engine import HandEngine

def run_hand_module():
    print("[Launcher] Starting Hand Module...")
    # This creates a full path starting from your 'jarvis' root folder
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models") 
    
    print(f"[Launcher] Looking for models in: {model_path}")
    
    # Check if files actually exist before starting MediaPipe
    if not os.path.exists(os.path.join(model_path, "hand_landmarker.task")):
        print(f"[Hand Module Error] Critical: hand_landmarker.task is missing from {model_path}!")
        return

    try:
        hand_module = HandEngine(model_dir=model_path)
        hand_module.start()
    except Exception as e:
        print(f"[Hand Module Error] {e}")

def run_server():
    """Starts the FastAPI Backend"""
    print("[Launcher] Starting FastAPI Brain...")
    # 'main:app' assumes you have a file 'src/main.py' with 'app = FastAPI()'
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # On Windows, multiprocessing needs this to work correctly
    multiprocessing.freeze_support()

    print("--- INITIALIZING JARVIS CORE ---")

    # Create two separate processes
    hand_process = multiprocessing.Process(target=run_hand_module, name="Jarvis-Hand")
    server_process = multiprocessing.Process(target=run_server, name="Jarvis-API")

    # Start them both
    hand_process.start()
    server_process.start()

    try:
        # Keep the launcher alive while processes run
        while True:
            time.sleep(1)
            if not hand_process.is_alive() or not server_process.is_alive():
                print("[Launcher] Critical process died. Shutting down...")
                break
    except KeyboardInterrupt:
        print("[Launcher] Manual shutdown initiated.")
    finally:
        hand_process.terminate()
        server_process.terminate()
        print("[Launcher] All systems offline.")