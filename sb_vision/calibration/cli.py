"""Command-line tool for camera calibration."""

import sys
import lzma
import pickle
import argparse
from pathlib import Path

import yaml

from .fit import fit
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

    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help="file to save into",
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

    training_examples = []

    for entry in files:
        try:
            training_examples.append(
                TrainingExample(
                    image_file=options.directory / entry['image'],
                    z_distance=entry['z'],
                    x_offset_right=entry.get('x', 0.0),
                ),
            )
        except RuntimeError as e:
            print(str(e))

    calibration = fit(training_examples)

    print(calibration)

    if options.output:
        with lzma.open(str(options.output), 'wb') as f:
            pickle.dump(calibration, f)
