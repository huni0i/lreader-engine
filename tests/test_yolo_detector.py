from lreader_engine.yolo_detector import default_weights_path


def test_default_yolo_weights_point_at_det_text() -> None:
    path = default_weights_path()
    assert path.name == "best.pt"
    assert "det-text" in str(path)
