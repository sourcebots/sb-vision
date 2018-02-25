"""Tests for loading calibration files."""

from pathlib import Path
from unittest import mock

import pytest

from sb_vision.find_3D_coords import (
    ResolutionMismatchError,
    load_camera_calibrations,
)

TEST_DATA = Path(__file__).parent / 'test_data'


def test_rejects_mismatching_resolutions():
    """Ensure that loading a mismatched calibration file produces an error."""
    with mock.patch(
        'pathlib.Path.open',
        return_value=(TEST_DATA / 'C016_calibration.xml').open(),
    ):
        with pytest.raises(ResolutionMismatchError):
            load_camera_calibrations('camera-model', (100, 100))


def test_accepts_matching_resolution():
    """Ensure that loading a matcing calibration file works."""
    with mock.patch(
        'pathlib.Path.open',
        return_value=(TEST_DATA / 'C016_calibration.xml').open(),
    ):
        camera_matrix, distance_coefficients = load_camera_calibrations(
            'camera-model',
            (1280, 720),
        )

        expected_camera_matrix = [
            [1.2522471126459800e+03, 0.0, 6.1724173580467107e+02],
            [0.0, 1.2522471126459800e+03, 3.0883837638358619e+02],
            [0.0, 0.0, 1.0],
        ]
        assert camera_matrix == expected_camera_matrix

        expected_distance_coefficients = [[
            -3.0075450734301268e-01,
            5.2034021230266936e-01,
            -2.7004158974004307e-03,
            1.0791605890294514e-02,
            -9.2571589159495127e-01,
        ]]
        assert distance_coefficients == expected_distance_coefficients
