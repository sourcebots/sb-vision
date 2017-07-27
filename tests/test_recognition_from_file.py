import pytest

from pathlib import Path

from sb_vision import Vision, FileCamera, Token

TEST_DATA = Path(__file__).parent / 'test_data'

TEST_IMAGES = [
    ("Photo 1.jpg", [Token(id=9)]),
]


@pytest.mark.parametrize("photo, expected_tokens", TEST_IMAGES)
def test_recognises_markers(photo, expected_tokens):
    camera = FileCamera(TEST_DATA / photo, 1.2)
    vision = Vision(camera, 0.01)
    snaps = vision.snapshot()
    assert snaps == expected_tokens
