from modules.STT_Module.stt_engine import STTEngine


def test_stt_engine_importable():
    # Keep CI non-interactive: just ensure the class is importable.
    assert STTEngine is not None