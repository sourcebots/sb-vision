"""Integration tests for coordinates."""

import math
from pathlib import Path

import pytest

from sb_vision import Cartesian, FileCamera, Polar, Vision

CALIBRATIONS = Path(__file__).parent.parent / 'calibrations' / 'c270'

TEST_IMAGES = [
    (
        '100cmL10cm.jpg',
        Cartesian(x=-0.1, y=0, z=1),
        Polar(
            rot_x=0,
            rot_y=math.atan(-0.1),  # or -5.710593137499643 degrees
            dist=(0.01 + 1)**0.5,  # or 1.004987562112089
        ),
    ),
    (
        '300cmL90cm.jpg',
        Cartesian(x=-0.9, y=0, z=3),
        Polar(
            rot_x=0,
            rot_y=math.atan(-0.9 / 3),  # or -16.69924423399362 degrees
            dist=(0.81 + 9)**0.5,  # or 3.132091952673165
        ),
    ),
]


@pytest.mark.parametrize("photo, expected_cartesian, expected_polar", TEST_IMAGES)
def test_image_coordinates(photo, expected_cartesian, expected_polar):
    """Make sure that this particular file gives these particular tokens."""
    camera = FileCamera(CALIBRATIONS / photo, distance_model='c270')
    vision = Vision(camera)
    token, = vision.snapshot()

    x, y, z = token.cartesian

    assert expected_cartesian.x == round(x, 3), "Wrong x-coordinate"
    assert expected_cartesian.y == round(y, 3), "Wrong y-coordinate"
    assert expected_cartesian.z == round(z, 3), "Wrong z-coordinate"

    rot_x, rot_y, dist = token.polar

    assert expected_polar.rot_x == round(rot_x, 3), "Wrong x rotation"
    assert round(expected_polar.rot_y, 3) == round(rot_y, 3), "Wrong y rotation"
    assert round(expected_polar.dist, 3) == round(dist, 3), "Wrong distance"
