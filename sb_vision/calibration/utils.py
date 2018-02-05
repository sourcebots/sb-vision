"""Utilities related to calibrating a camera."""

import pathlib
from typing import List, NamedTuple

import yaml

CalibrationReference = NamedTuple('CalibrationReference', (
    ('image_file', pathlib.Path),
    ('z_distance', float),
    ('x_offset_right', float),
))


def load_calibrations(directory: pathlib.Path) -> List[CalibrationReference]:
    """Load calibration data from the given directory."""
    with (directory / 'files.yaml').open('r') as f:
        config_data = yaml.load(f)
        config_version = config_data['version']
        if config_version > 1:
            raise SystemExit("Cannot handle config versions >1")
        files = config_data['files']

        return [
            CalibrationReference(
                image_file=directory / entry['image'],
                z_distance=entry['z'],
                x_offset_right=entry.get('x', 0.0),
            )
            for entry in files
        ]
