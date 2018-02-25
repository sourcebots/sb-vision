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
    """Ensure that loading a mismatched calibration file produces an error."""
    with mock.patch(
        'pathlib.Path.open',
        return_value=(TEST_DATA / 'C016_calibration.xml').open(),
    ):
        load_camera_calibrations('camera-model', (1280, 720))
