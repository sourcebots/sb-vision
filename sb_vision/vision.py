"""Main vision driver."""

from typing import List, Optional  # noqa: F401

from PIL import Image

from .camera_base import CameraBase
from .cvcapture import clean_and_threshold
from .native.apriltag import AprilTagDetector
from .tokens import Token


class Vision:
    """Class that handles the vision library and the camera."""

    def __init__(self, camera: CameraBase) -> None:
        """General initialiser."""
        self._camera = camera
        self._camera_ready = False

        self._detector = None  # type: Optional[AprilTagDetector]

    @property
    def camera(self) -> CameraBase:
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
    def apriltag_detector(self) -> AprilTagDetector:
        """Lazy property wrapping our instance of the apriltag detector."""
        if self._detector is None:
            size = self.camera.get_image_size()
            self._detector = AprilTagDetector(size)

        return self._detector

    def capture_image(self) -> Image.Image:
        """
        Capture an image from the camera.

        :return: single PIL image
        """
        # get the PIL image from the camera
        return self.camera.capture_image()

    def threshold_image(self, img: Image.Image) -> Image.Image:
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

    def process_image(self, img: Image.Image) -> List[Token]:
        """
        Run the given image through the apriltags detection library.

        :param img: PIL Luminosity image to be processed
        :return: python list of Token objects.
        """
        img = self.threshold_image(img)

        distance_model = self.camera.distance_model

        tokens = [
            Token.from_apriltag_detection(x, img.size, distance_model)
            for x in self.apriltag_detector.detect_tags(img)
        ]

        return tokens

    def snapshot(self) -> List[Token]:
        """
        Get a single list of tokens from one camera snapshot.

        Equivalent to calling `process_image` on the result of `capture_image`.
        """
        return self.process_image(self.capture_image())
