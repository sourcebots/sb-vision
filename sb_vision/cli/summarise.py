"""Summarise a collection of images using a distance model."""

import argparse
import contextlib
import json
import pathlib
import sys
from typing import Sequence, TextIO

from ..camera import FileCamera
from ..vision import Vision


def main(files: Sequence[pathlib.Path], distance_model: str, output: TextIO):
    """Execute this command."""
    with contextlib.suppress(KeyboardInterrupt):
        print('version: 1', file=output)
        print('files:', file=output)

        for image_file in sorted(files):
            camera = FileCamera(image_file, distance_model)
            tokens = Vision(camera).snapshot()

            if len(tokens) != 1:
                print(
                    "Didn't see one token in '{}' (saw {:d})".format(
                        image_file,
                        len(tokens),
                    ),
                    file=sys.stderr,
                )
                continue

            token, = tokens
            info = {
                'image': str(image_file),
                'x': round(token.cartesian[0], 4),
                'z': round(token.cartesian[2], 4),
            }
            print(' - {}'.format(json.dumps(info, sort_keys=True)), file=output)


def add_arguments(parser):
    """Add arguments for this command."""
    parser.add_argument(
        'files',
        metavar="IMAGE_FILE",
        nargs='+',
        type=pathlib.Path,
        help="Images to use as input.",
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
