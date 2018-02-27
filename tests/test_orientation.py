"""Tests for marker orientations."""

import math
from pathlib import Path

import pytest
from pytest import approx

from sb_vision import FileCamera, Orientation, Vision

ROTATION_TOLERANCE_DEGREES = 6

CALIBRATIONS = Path(__file__).parent.parent / 'calibrations' / 'tecknet_rotations'

TEST_IMAGES = [
    (
        '0rot_x-45rot_y1.1z.png',
        Orientation(rot_x=0, rot_y=math.radians(-45), rot_z=0),
    ),
    (
        '-45rot_x0rot_y1.1z.png',
        Orientation(rot_x=math.radians(-45), rot_y=0, rot_z=0),
    ),
    (
        '-22.5rot_x0rot_y0.6z.png',
        Orientation(rot_x=math.radians(-22.5), rot_y=0, rot_z=0),
    ),
    (
        '0rot_x-22.5rot_y0.6z.png',
        Orientation(rot_x=0, rot_y=math.radians(-22.5), rot_z=0),
    ),
    (
        '0rot_x0rot_y0.55z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=0),
    ),
    (
        '-90rot_z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=math.radians(270)),
    ),
    (
        '90rot_z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=math.radians(90)),
    ),
    (
        '180rot_z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=math.radians(180)),
    ),
    (
        '135rot_z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=math.radians(135)),
    ),
    (
        '45rot_z.png',
        Orientation(rot_x=0, rot_y=0, rot_z=math.radians(45)),
    ),
]


@pytest.mark.parametrize(
    "photo, expected_orientation",
    TEST_IMAGES,
)
def test_image_coordinates(photo, expected_orientation):
    """Make sure that this particular file gives these particular tokens."""
    camera = FileCamera(CALIBRATIONS / photo, camera_model='C016')
    vision = Vision(camera)
    token, = vision.snapshot()

    def approx_ang(expected_degrees):
        return approx(expected_degrees, abs=ROTATION_TOLERANCE_DEGREES)

    def assert_angle(angle_radians, expected_degrees, message):
        expected_degrees = approx_ang(math.degrees(expected_degrees))
        # Check both +0 and +360 so approx can cover the jump between -180 and 180
        assert math.degrees(angle_radians) == expected_degrees or math.degrees(angle_radians) + 360 == expected_degrees, \
            message

    rot_x, rot_y, rot_z = token.orientation

    assert_angle(rot_x, expected_orientation.rot_x, "Wrong Orientation rot_x")
    assert_angle(rot_y, expected_orientation.rot_y, "Wrong Orientation rot_y")
    assert_angle(rot_z, expected_orientation.rot_z, "Wrong Orientation rot_z")
