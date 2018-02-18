"""Integration tests for coordinates."""

import math
from pathlib import Path

import pytest

from sb_vision import Cartesian, FileCamera, Spherical, Vision

# Ensure all distance values are within this tolerance
# (as a percentage of the total distance)
DIST_PERCENT_TOLERANCE = 0.15
# Rotation tolerance is independent of distance
ROTATION_TOLERANCE = 0.075

CALIBRATIONS = Path(__file__).parent.parent / 'calibrations' / 'tecknet_25cm'

TEST_IMAGES = [
    (
        '1.0z-0.1x.jpg',
        Cartesian(x=-0.1, y=0, z=1),
        Spherical(
            rot_x=0,
            rot_y=math.atan(-0.1),  # or -0.09966865249116204 radians
            dist=(0.1**2 + 1**2) ** 0.5,  # or 1.004987562112089
        ),
    ),
    (
        '3.0z1.0x.jpg',
        Cartesian(x=1, y=0, z=3),
        Spherical(
            rot_x=0,
            rot_y=math.atan(1 / 3), # or 0.3217505543966422 radians
            dist=(1**2 + 3**2) ** 0.5, # or 3.1622776601683795
        ),
    ),
]


def assertWithinTolerance(val, expected, tolerance, error_msg=""):
    """Assert that the value is within a certain tolerance value."""
    assert (expected - tolerance) < val < (expected + tolerance), error_msg


@pytest.mark.parametrize(
    "photo, expected_cartesian, expected_spherical",
    TEST_IMAGES,
)
def test_image_coordinates(photo, expected_cartesian, expected_spherical):
    """Make sure that this particular file gives these particular tokens."""
    camera = FileCamera(CALIBRATIONS / photo, camera_model='C016')
    vision = Vision(camera)
    token, = vision.snapshot()

    x, y, z = token.cartesian
    rot_x, rot_y, dist = token.spherical

    tolerance = dist * DIST_PERCENT_TOLERANCE

    assertWithinTolerance(x, expected_cartesian.x, tolerance,
                          "Wrong x-coordinate")
    assertWithinTolerance(y, expected_cartesian.y, tolerance,
                          "Wrong y-coordinate")
    assertWithinTolerance(z, expected_cartesian.z, tolerance,
                          "Wrong z-coordinate")

    assertWithinTolerance(rot_x, expected_spherical.rot_x, ROTATION_TOLERANCE,
                          "Wrong x rotation")
    assertWithinTolerance(rot_y, expected_spherical.rot_y, ROTATION_TOLERANCE,
                          "Wrong y rotation")
    assertWithinTolerance(dist, expected_spherical.dist, tolerance,
                          "Wrong distance")
