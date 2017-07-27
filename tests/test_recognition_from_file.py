from pathlib import Path

import pytest

from sb_vision import Token, Vision, FileCamera

TEST_DATA = Path(__file__).parent / 'test_data'

TEST_IMAGES = [
    ("Photo 1.jpg", [Token(id=9)]),
    ("Photo 2.jpg", [Token(id=12)]),
    ("Photo 3.jpg", [Token(id=15)]),
    ("Photo 4.jpg", [Token(id=26)]),
    # ("Photo 5.jpg", [Token(id=78)]),
    ("Photo 6.jpg", [Token(id=95)]),
]


@pytest.mark.parametrize("photo, expected_tokens", TEST_IMAGES)
def test_recognises_markers(photo, expected_tokens):
    camera = FileCamera(TEST_DATA / photo, 1.2)
    vision = Vision(camera, 0.01)
    snaps = vision.snapshot()
    assert snaps == expected_tokens
