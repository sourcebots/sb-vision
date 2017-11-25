#!/usr/bin/env python3

import argparse
import collections
import re

from PIL import Image

from sb_vision.cvcapture import CaptureDevice


class Resolution(collections.namedtuple('Resolution', ('width', 'height'))):
    __slots__ = ()

    @classmethod
    def from_argument(cls, res):
        try:
            return cls.from_string(res)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))

    @classmethod
    def from_string(cls, res):
        match = re.match(r'(?P<width>\d+)x(?P<height>\d+)', res)

        if match is None:
            raise ValueError(
                "Invalid resolution specified. Must be in the form 'WIDTHxHEIGHT'"
                " (got {})".format(repr(res)),
            )

        return cls(**{k: int(v) for k, v in match.groupdict().items()})

    def __str__(self):
        return '{0.width}x{0.height}'.format(self)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('device_id', type=int)
    parser.add_argument('image_template', type=str)
    parser.add_argument(
        '--resolution',
        type=Resolution.from_argument,
        default=Resolution(640, 480),
        help="The resolution to capture from the device, default: %(default)s",
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=2,
        help="The number of images to capture, default: %(default)s",
    )
    return parser.parse_args()


def main(args):
    with CaptureDevice(args.device_id) as capture_device:
        for num in range(args.num_images):
            print("Capturing image {}...".format(num), end='')

            image_bytes = capture_device.capture(*args.resolution)

            image = Image.frombytes('L', args.resolution, image_bytes)

            with open(args.image_template.format(num), mode='wb') as f:
                image.save(f)

            print("done")


if __name__ == '__main__':
    main(parse_args())
