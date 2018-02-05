"""Command-line tool for camera calibration."""

import argparse
import lzma
import pickle
import sys
from pathlib import Path

from .fit import fit
from .training_example import TrainingExample
from .utils import load_calibrations


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


def main(args=None):
    """Main entry point."""
    if args is None:
        args = sys.argv[1:]

    options = argument_parser().parse_args(args)

    training_examples = []

    for calibration_reference in load_calibrations(options.directory):
        try:
            training_examples.append(
                TrainingExample(
                    image_file=calibration_reference.image_file,
                    z_distance=calibration_reference.z_distance,
                    x_offset_right=calibration_reference.x_offset_right,
                ),
            )
        except RuntimeError as e:
            print(str(e))

    calibration = fit(training_examples)

    print(calibration)

    if options.output:
        with lzma.open(str(options.output), 'wb') as f:
            pickle.dump(calibration, f)
