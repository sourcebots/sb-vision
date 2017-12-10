"""Main vision driver."""

import contextlib

from PIL import Image

from .camera import Camera, FileCamera
from .tokens import Token
from .cvcapture import clean_and_threshold
from .camera_base import CameraBase
from .token_display import display_tokens
from .native.apriltag import AprilTagDetector


class Vision:
    """Class that handles the vision library and the camera."""

    def __init__(self, camera: CameraBase):
        """General initialiser."""
        self._camera = camera
        self._camera_ready = False

        self._detector = None

    @property
    def camera(self):
        """
        Property wrapping our 'camera' instance.

        We assume that when we're given the instance it has not yet been
        initialised, so we do that on first use.
        """
        if not self._camera_ready:
            self._camera.init()
            self._camera_ready = True

        return self._camera

    @property
    def apriltag_library(self):
        """Lazy property wrapping our instance of the apriltag detector."""
        if self._detector is None:
            size = self.camera.get_image_size()
            self._detector = AprilTagDetector(size)

        return self._detector

    def capture_image(self):
        """
        Capture an image from the camera.

        :return: single PIL image
        """

        # get the PIL image from the camera
        return self.camera.capture_image()

    def threshold_image(self, img):
        """Run thresholding and preprocessing on an image."""
        as_bytes = img.convert('L').tobytes()
        cleaned_bytes = clean_and_threshold(
            as_bytes,
            img.size[0],
            img.size[1],
        )

        return Image.frombytes(
            mode='L',
            size=img.size,
            data=cleaned_bytes,
        )

    def process_image(self, img):
        """
        Run the given image through the apriltags detection library.

        :param img: PIL Luminosity image to be processed
        :return: python list of Token objects.
        """
        img = self.threshold_image(img)

        distance_model = self.camera.distance_model

        tokens = [
            Token.from_apriltag_detection(x, img.size, distance_model)
            for x in self.apriltag_library.detect_tags(img)
        ]

        return tokens

    def snapshot(self):
        """
        Get a single list of tokens from one camera snapshot.

        Equivalent to calling `process_image` on the result of `capture_image`.
        """
        return self.process_image(self.capture_image())


if __name__ == "__main__":
    """
    Debug code, load the first video device seen and capture an image
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        nargs='?',
        default=False,
        help="Pass an image file to use, otherwise use webcam; if filename "
             "is blank, uses 'tagsampler.png'",
    )
    parser.add_argument(
        '-i',
        '--device-id',
        default=0,
        help="device id of the camera to use",
        type=int,
    )
    parser.add_argument(
        '-n',
        action='store_false',
        dest='show',
        help="do not show the captured frames",
    )
    parser.add_argument(
        '-t',
        '--after-thresholding',
        action='store_true',
        help="show image after thresholding",
    )
    parser.add_argument(
        '-d',
        '--distance-model',
        help="distance model to use",
    )

    args = parser.parse_args()
    # Change the below for quick debugging
    if args.f is False:
        CAM_IMAGE_SIZE = (1280, 720)
        camera = Camera(args.device_id, CAM_IMAGE_SIZE, args.distance_model)
    else:
        if args.f is None:
            f = "tagsampler.png"
        else:
            f = args.f
        camera = FileCamera(f, args.distance_model)
    v = Vision(camera)
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            img = v.capture_image()
            tokens = v.process_image(img)
            if args.after_thresholding:
                img = v.threshold_image(img)
            if args.show:
                img = display_tokens(tokens, img)
                img.show()
            print("Saw {} token(s)".format(len(tokens)))
            for token in tokens:
                print("- {}".format(token))
                try:
                    print(token.cartesian)
                except AttributeError:
                    pass
            if args.f is not False:
                break
