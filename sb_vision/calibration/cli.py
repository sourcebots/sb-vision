"""Command-line tool for camera calibration."""

import sys
import argparse
from pathlib import Path

import yaml

from .training_example import TrainingExample


def argument_parser():
    """Calibration tool argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute calibration from camera",
    )

    parser.add_argument(
        'directory',
        type=Path,
        help="calibration directory",
    )

    return parser


def main(args=sys.argv[1:]):
    """Main entry point."""
    options = argument_parser().parse_args(args)

    with (options.directory / 'files.yaml').open('r') as f:
        config_data = yaml.load(f)
        config_version = config_data['version']
        if config_version > 1:
            raise SystemExit("Cannot handle config versions >1")
        files = config_data['files']

    training_examples = [
        TrainingExample(
            image_file=options.directory / x['image'],
            z_distance=x['z'],
            x_offset_right=x.get('x', 0.0),
        )
        for x in files
    ]

    print(training_examples)
