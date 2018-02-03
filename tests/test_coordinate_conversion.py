"""Test that our coordinate conversions work."""

import math

import pytest

from sb_vision import Cartesian, Spherical, cartesian_to_spherical

TEST_DATA = [
    (
        Cartesian(x=0, y=0, z=1),
        Spherical(
            rot_x=0,
            rot_y=0,
            dist=1,
        ),
    ),
    (
        Cartesian(x=0, y=1, z=1),
        Spherical(
            rot_x=math.pi / 4,  # or 45 degrees
            rot_y=0,
            dist=2**0.5,  # or 1.4142135623730951
        ),
    ),
    (
        Cartesian(x=1, y=0, z=1),
        Spherical(
            rot_x=0,
            rot_y=math.pi / 4,  # or 45 degrees
            dist=2**0.5,  # or 1.4142135623730951
        ),
    ),
    (
        Cartesian(x=1, y=1, z=1),
        Spherical(
            rot_x=math.pi / 4,  # or 45 degrees
            rot_y=math.pi / 4,  # or 45 degrees
            dist=3**0.5,  # or 1.7320508075688772
        ),
    ),
    (
        Cartesian(x=-1, y=1, z=1),
        Spherical(
            rot_x=math.pi / 4,  # or -45 degrees
            rot_y=-math.pi / 4,  # or 45 degrees
            dist=3**0.5,  # or 1.7320508075688772
        ),
    ),
    (
        Cartesian(x=1, y=-1, z=1),
        Spherical(
            rot_x=-math.pi / 4,  # or -45 degrees
            rot_y=math.pi / 4,  # or 45 degrees
            dist=3**0.5,  # or 1.7320508075688772
        ),
    ),
    (
        Cartesian(x=-0.1, y=0, z=1),
        Spherical(
            rot_x=0,
            rot_y=math.atan(-0.1),  # or -5.710593137499643 degrees
            dist=(0.01 + 1)**0.5,  # or 1.004987562112089
        ),
    ),
    (
        Cartesian(x=-0.9, y=0, z=3),
        Spherical(
            rot_x=0,
            rot_y=math.atan(-0.9 / 3),  # or -16.69924423399362 degrees
            dist=(0.81 + 9)**0.5,  # or 3.132091952673165
        ),
    ),
    (
        Cartesian(x=0, y=-0.1, z=3),
        Spherical(
            rot_x=math.atan(-0.1 / 3),  # or -1.9091524329963763 degrees
            rot_y=0,
            dist=(0.01 + 9)**0.5,  # or 3.132091952673165
        ),
    ),
]


@pytest.mark.parametrize("cartesian, spherical", TEST_DATA)
def test_cartesian_to_spherical(cartesian, spherical):
    """Make sure that this conversion works."""
    actual = cartesian_to_spherical(cartesian)

    assert round(spherical.rot_x, 3) == round(actual.rot_x, 3), "Wrong x rotation"
    assert round(spherical.rot_y, 3) == round(actual.rot_y, 3), "Wrong y rotation"
    assert round(spherical.dist, 3) == round(actual.dist, 3), "Wrong distance"
