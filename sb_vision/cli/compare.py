"""Compare the output of a distance model for a set of images to the expected."""

import argparse
import contextlib
import pathlib
import sys
from typing import Sequence, TextIO

from ..camera import FileCamera
from ..vision import Vision


def mean_absolute_error(series: Sequence[float]) -> float:
    """Calculate the mean absolute error of a series."""
    return sum(abs(x) for x in series) / len(series)


def main(directory: pathlib.Path, distance_model: str, output: TextIO, verbose: bool):
    """Execute this command."""
    # ensure the transitive dependency on PyYAML remains optional via lazy import
    from ..calibration.utils import load_calibrations

    with contextlib.suppress(KeyboardInterrupt):
        x_errors = []
        z_errors = []
        for calibration_reference in load_calibrations(directory):
            camera = FileCamera(calibration_reference.image_file, distance_model)
            tokens = Vision(camera).snapshot()

            if len(tokens) != 1:
                print(
                    "Didn't see one token in '{}' (saw {:d})".format(
                        calibration_reference.image_file,
                        len(tokens),
                    ),
                    file=sys.stderr,
                )
                continue

            token, = tokens
            x_error = token.cartesian.x - calibration_reference.x_offset_right
            z_error = token.cartesian.z - calibration_reference.z_distance

            if verbose:
                print("image: {}".format(calibration_reference.image_file), file=output)
                print("x error: {:.3f}".format(x_error), file=output)
                print("z error: {:.3f}".format(z_error), file=output)
                print(file=output)

            x_errors.append(x_error)
            z_errors.append(z_error)

        print("Mean absolute X error: {:.3f}".format(mean_absolute_error(x_errors)))
        print("Mean absolute Z error: {:.3f}".format(mean_absolute_error(z_errors)))


def add_arguments(parser):
    """Add arguments for this command."""
    parser.add_argument(
        'directory',
        type=pathlib.Path,
        help="Directory of files .",
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help="Show the errors for each individual file.",
    )
    parser.add_argument(
        '-m',
        '--distance-model',
        type=pathlib.Path,
        help="Distance model to use.",
    )
    parser.add_argument(
        '-o',
        '--output',
        type=argparse.FileType(mode='w'),
        default=sys.stdout,
    )
