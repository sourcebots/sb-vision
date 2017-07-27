from pathlib import Path

from sb_vision import Vision, FileCamera, Token

TEST_DATA = Path(__file__).parent / 'test_data'


def test_recognises_basic_marker():
    camera = FileCamera(TEST_DATA / 'Photo 1.jpg', 1.2)
    vision = Vision(camera, 0.01)
    snaps = vision.snapshot()
    assert snaps == [Token(id=9)]
