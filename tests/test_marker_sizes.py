"""Tests for marker sizing."""

from pathlib import Path
from typing import Dict, Tuple
from unittest import mock

import numpy as np

from sb_vision import FileCamera, Token, Vision
from sb_vision.camera_base import CameraBase

TEST_DATA = Path(__file__).parent / 'test_data'

# We're using TeckNet images with the c270 model, so these are expected to be a
# bit off; note that they still preserve the ratio of 1:2.5 (0.7 : 1.75 then rounded)
EXPECTED_LARGE_DISTANCE = 1.8
EXPECTED_SMALL_DISTANCE = 0.7

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

    assert round(big_x, 5) == round(small_x * ratio, 5), "Wrong x-coordinate"
    assert round(big_y, 5) == round(small_y * ratio, 5), "Wrong y-coordinate"
    assert round(big_z, 5) == round(small_z * ratio, 5), "Wrong z-coordinate"


def assertMarkerDistance(
    camera: CameraBase,
    *,
    marker_sizes: Dict[int, Tuple[float, float]],
    expected_distance: float
) -> None:
    """Assert that the processed distance is as expected for a marker size."""
    vision = Vision(camera)

    with mock.patch('sb_vision.tokens.MARKER_SIZES', marker_sizes):
        token, = vision.snapshot()

    dist = token.spherical.dist
    assert round(dist, 1) == expected_distance


def test_unknown_marker_size():
    """Test an unknown marker size defaults to the trained size."""
    # The c270 model is trained on 25cm markers; so it assume that all markers
    # are that size unless told otherwise.
    assertMarkerDistance(
        FileCamera(TEST_DATA / 'tecknet-10cm-at-1m.jpg', distance_model='c270'),
        marker_sizes={},
        expected_distance=EXPECTED_LARGE_DISTANCE,
    )


def test_large_marker_large_size():
    """Test a marker matching the trained size has the right distance."""
    assertMarkerDistance(
        FileCamera(TEST_DATA / 'tecknet-25cm-at-2.5m.jpg', distance_model='c270'),
        marker_sizes={23: (0.25, 0.25)},
        expected_distance=EXPECTED_LARGE_DISTANCE,
    )


def test_large_marker_small_size():
    """
    Test image with large marker gives small distance when configured for a small marker.
    """
    assertMarkerDistance(
        FileCamera(TEST_DATA / 'tecknet-25cm-at-2.5m.jpg', distance_model='c270'),
        marker_sizes={23: (0.1, 0.1)},
        expected_distance=EXPECTED_SMALL_DISTANCE,
    )


def test_small_marker_large_size():
    """
    Test image with small marker gives large distance when configured for a large marker.
    """
    assertMarkerDistance(
        FileCamera(TEST_DATA / 'tecknet-10cm-at-1m.jpg', distance_model='c270'),
        marker_sizes={44: (0.25, 0.25)},
        expected_distance=EXPECTED_LARGE_DISTANCE,
    )


def test_small_marker_small_size():
    """
    Test image with small marker gives small distance when configured for a small marker.
    """
    assertMarkerDistance(
        FileCamera(TEST_DATA / 'tecknet-10cm-at-1m.jpg', distance_model='c270'),
        marker_sizes={44: (0.1, 0.1)},
        expected_distance=EXPECTED_SMALL_DISTANCE,
    )
