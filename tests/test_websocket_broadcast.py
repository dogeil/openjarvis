import json
import time
import os
import sys

from fastapi.testclient import TestClient


# Allow running this file directly (python tests/test_websocket_broadcast.py)
# by ensuring `src` is on the import path. In CI we already set PYTHONPATH=src.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from main import app


def test_websocket_broadcast_to_other_clients():
    """
    This test runs in-process (no uvicorn / port 8000 needed).
    It verifies that messages are broadcast to other connected websockets,
    and that the broadcast payload is valid JSON.
    """
    client = TestClient(app)

    with client.websocket_connect("/ws/jarvis") as ws_sender, client.websocket_connect(
        "/ws/jarvis"
    ) as ws_receiver:
        payload = {
            "source": "test_client",
            "type": "hand_gesture",
            "payload": {"gesture": "thumbs_up", "confidence": 0.98},
            "timestamp": time.time(),
        }

        ws_sender.send_text(json.dumps(payload))
        received_raw = ws_receiver.receive_text()
        received = json.loads(received_raw)

        assert received["source"] == payload["source"]
        assert received["type"] == payload["type"]
        assert received["payload"] == payload["payload"]
        assert isinstance(received["timestamp"], (int, float))