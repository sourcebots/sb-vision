"""
    Tests for marker sizing.
"""

import numpy as np

from sb_vision import Token

# Homography matrix of a real image.
homography_from_camera = np.array([
    [7.08277177e-01, -6.30171882e-03, 9.36194251e-01],
    [1.08272961e-02, 7.11294182e-01, 2.64415168e+00],
    [2.14327973e-05, 7.37441645e-06, 7.25640594e-03],
])
distance_model = 'c270'
image_resolution = (1280, 720)


def test_marker_size():
    """Ensure that marker distances are reported proportionally to their size."""

    big_size = (0.25, 0.25)
    small_size = (0.1, 0.1)

    ratio = big_size[0] / small_size[0]

    big_token = Token(0)
    big_token.infer_location_from_homography_matrix(
        homography_matrix=homography_from_camera,
        distance_model=distance_model,
        image_size=image_resolution,
        marker_size=big_size,
    )

    small_token = Token(0)
    small_token.infer_location_from_homography_matrix(
        homography_matrix=homography_from_camera,
        distance_model=distance_model,
        image_size=image_resolution,
        marker_size=small_size,
    )

    big_x, big_y, big_z = big_token.cartesian
    small_x, small_y, small_z = small_token.cartesian

    assert big_x == small_x * ratio, "Wrong x-coordinate"
    assert big_y == small_y * ratio, "Wrong y-coordinate"
    assert big_z == small_z * ratio, "Wrong z-coordinate"


