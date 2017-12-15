"""Main vision driver."""

from PIL import Image

from sb_vision.native.apriltag._apriltag import ffi, lib

from .tokens import Token
from .cvcapture import clean_and_threshold
from .camera_base import CameraBase


class Vision:
    """Class that handles the vision library and the camera."""

    def __init__(self, camera: CameraBase):
        """General initialiser."""
        self.camera = camera
        # apriltag detector object
        self._detector = None
        # image from camera
        self.image = None

        self.initialised = False

    def _init(self):
        """
        Run 'actual initialisation'.

        This involves 'initting' the camera (opening the device, for instance)
        and setting up the AprilTags library.
        """
        self.camera.init()
        self._init_library()
        self.initialised = True

    def _lazily_init(self):
        """
        Lazily initialise.

        Read as 'call `_init` if nobody else has done so'.
        """
        if not self.initialised:
            self._init()

    def __del__(self):
        """
        Drop any referred-to resources.

        Make sure that the library is deinitialised when the `Vision` falls
        out of scope.
        """
        self._deinit_library()

    def _init_library(self):
        """
        Initialise the AprilTag library.

        This means creating and configuring the detector, which populates a
        number of tables in memory.
        """
        # init detector
        self._detector = lib.apriltag_detector_create()
        """
        apriltag_detector_t* td,
        float decimate,
          default: 1.0, "Decimate input image by this factor"
        float sigma,
          default: 0.0, "Apply low-pass blur to input; negative sharpens"
        int refine_edges,
          default: 1, "Spend more time trying to align edges of tags"
        int refine_decode,
          default: 0, "Spend more time trying to decode tags"
        int refine_pose
          default: 0, "Spend more time trying to find the position of the tag"
        """
        lib.apriltag_init(self._detector, 1.0, 0.0, 1, 0, 0)
        size = self.camera.get_image_size()
        self.image = lib.image_u8_create_stride(size[0], size[1], size[0])

    def _deinit_library(self):
        """Deinitialise the library."""
        # Always destroy the detector
        if self._detector:
            lib.apriltag_detector_destroy(self._detector)
        if self.image:
            lib.image_u8_destroy(self.image)

    def _parse_results(self, results):
        """
        Parse the array of results.

        :param results: cffi array of results
        :return: python iterable of individual token objects
        """
        image_size = self.camera.get_image_size()

        for i in range(results.size):
            detection = lib.zarray_get_detection(results, i)
            yield Token.from_apriltag_detection(
                detection,
                image_size,
                self.camera.distance_model,
            )
            lib.destroy_detection(detection)

    def capture_image(self):
        """
        Capture an image from the camera.

        :return: single PIL image
        """
        self._lazily_init()

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

        self._lazily_init()
        total_length = img.size[0] * img.size[1]
        # Detect the markers
        ffi.memmove(self.image.buf, img.tobytes(), total_length)
        results = lib.apriltag_detector_detect(self._detector, self.image)
        tokens = list(self._parse_results(results))
        # Remove the array now we've got them
        lib.zarray_destroy(results)
        return tokens

    def snapshot(self):
        """
        Get a single list of tokens from one camera snapshot.

        Equivalent to calling `process_image` on the result of `capture_image`.
        """
        self._lazily_init()
        return self.process_image(self.capture_image())
