"""Integration tests for coordinates."""

import math
from pathlib import Path

import pytest
from pytest import approx

from sb_vision import Cartesian, FileCamera, Spherical, Vision

# Ensure all distance values are within this tolerance
# (as a percentage of the total distance)
DIST_PERCENT_TOLERANCE = 0.15
# Rotation tolerance in radians
ROTATION_TOLERANCE = 0.075

CALIBRATIONS = Path(__file__).parent.parent / 'calibrations' / 'tecknet_25cm'

TEST_IMAGES = [
    (
        '1.0z-0.1x.jpg',
        Cartesian(x=-0.1, y=0, z=1),
        Spherical(
            rot_x=0,
            rot_y=math.atan(-0.1),  # or -5.710593137499643 degrees
            dist=(0.01 + 1)**0.5,  # or 1.004987562112089
        ),
    ),
    (
        '3.0z1.0x.jpg',
        Cartesian(x=1, y=0, z=3),
        Spherical(
            rot_x=0,
            rot_y=math.atan(1 / 3),  # or 18.43494882296774 degrees
            dist=(1**2 + 3**2) ** 0.5,  # or 3.1622776601683795
        ),
    ),
]


@pytest.mark.parametrize(
    "photo, expected_cartesian, expected_spherical",
    TEST_IMAGES,
)
def test_image_coordinates(photo, expected_cartesian, expected_spherical):
    """Make sure that this particular file gives these particular tokens."""
    camera = FileCamera(CALIBRATIONS / photo, camera_model='C016')
    vision = Vision(camera)
    token, = vision.snapshot()

    def approx_dist(expected):
        return approx(expected, abs=expected_spherical.dist * DIST_PERCENT_TOLERANCE)

    def approx_ang(expected):
        return approx(expected, abs=ROTATION_TOLERANCE)

    x, y, z = token.cartesian

    assert x == approx_dist(expected_cartesian.x), "Wrong x-coordinate"
    assert y == approx_dist(expected_cartesian.y), "Wrong y-coordinate"
    assert z == approx_dist(expected_cartesian.z), "Wrong z-coordinate"

    rot_x, rot_y, dist = token.spherical

    assert rot_x == approx_ang(expected_spherical.rot_x), "Wrong x-coordinate"
    assert rot_y == approx_ang(expected_spherical.rot_y), "Wrong y-coordinate"
    assert dist == approx_dist(expected_spherical.dist), "Wrong distance"
