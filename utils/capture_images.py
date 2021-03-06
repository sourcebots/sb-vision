#!/usr/bin/env python3

"""
Utility to capture images from the camera. Primarily used
for testing camera compatibility with the platform, though
potentially also useful for gathering calibration images.
"""

import argparse
import collections
import re
import time

from PIL import Image

from sb_vision.cvcapture import CaptureDevice


class Stopwatch:
    @staticmethod
    def print_duration(stopwatch):
        print("Took {} seconds".format(stopwatch.duration))

    def __init__(self, on_stop=lambda x: None):
        self._period = (None, None)
        self._on_stop = on_stop

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self._period = (time.time(), None)

    def stop(self):
        assert self._period[1] is None
        self._period = (self._period[0], time.time())

        self._on_stop(self)

    @property
    def duration(self):
        start, end = self._period
        assert end is not None
        return end - start


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'device_id',
        type=int,
        help="The id of the video device. For example: for the camera at "
             "'/dev/video0', this would be 0.",
    )
    parser.add_argument(
        'image_template',
        type=str,
        help="The file name and path to use for the images, optionally with a "
             "placeholder for the index of the image within the current "
             "sequence. For example: '/tmp/foo-{}.png' will result in images "
             "'/tmp/foo-0.png',' /tmp/foo-1.png' and so on.",
    )
    parser.add_argument(
        '--resolution',
        type=Resolution.from_argument,
        default=Resolution(640, 480),
        help="The resolution to capture from the device, default: %(default)s",
    )
    parser.add_argument(
        '--num-images',
        type=int,
        # Note: we default to 2 as we have seen issues with some cameras which
        # are only able to successfully campture a first image. By expecting
        # that all cameras will capture two images we avoid accidentally
        # believing that cameras which fail in this manner are fine when they
        # are in fact not fine.
        default=2,
        help="The number of images to capture, default: %(default)s",
    )
    parser.add_argument(
        '--timings',
        action='store_true',
        default=False,
        help="Print the time take to capture each image",
    )
    return parser.parse_args()


def main(args):
    stopwatch = Stopwatch(on_stop=Stopwatch.print_duration if args.timings else lambda x: None)

    with stopwatch:
        print("Initialising camera...")
        capture_device = CaptureDevice(args.device_id)

    with capture_device:
        for num in range(args.num_images):
            print("Capturing image {}...".format(num))

            with stopwatch:
                image_bytes = capture_device.capture(*args.resolution)

            image = Image.frombytes('L', args.resolution, image_bytes)

            file_name = args.image_template.format(num)
            with open(file_name, mode='wb') as f:
                image.save(f)

            print("done")


if __name__ == '__main__':
    main(parse_args())
