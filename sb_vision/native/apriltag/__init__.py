"""AprilTag native utilities."""

from PIL import Image

from ._apriltag import ffi, lib


__all__ = ('AprilTagDetector',)


class AprilTagDetector:
    """Wrapper for the AprilTag tag detector."""

    def __init__(self, image_size):
        """
        Initialise the AprilTag tag detector.

        This means creating and configuring the detector, which populates a
        number of tables in memory.
        """
        self._image_size = image_size

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

        # TODO: keeping this around for the lifetime of the library feels like
        # it's an optiisation rather than actually required. Given that it
        # makes the instance non-thread-safe, work out whether it's desirable
        # and/or there are other reasons the instance isn't thread-safe anyway.
        self._working_image = lib.image_u8_create_stride(
            image_size[0],
            image_size[1],
            image_size[0],
        )

    def __del__(self):
        """Deinitialise the detector."""

        if self._detector:
            lib.apriltag_detector_destroy(self._detector)

        if self._working_image:
            lib.image_u8_destroy(self._working_image)

    @property
    def image_size(self):
        """
        The image size, a tuple of (width, height), for which the detector
        instance is configured.
        """
        return self._image_size

    def detect_tags(self, img: Image):
        """
        Run the given image through the apriltags detection routines.

        :param img: PIL Luminosity image to be processed
        :yield: python iterable of apriltag detections; these must be processed
                and discarded before continuing iteration
        """

        if self.image_size != img.size:
            raise ValueError(
                "Cannot process images of an incompatible size. Detector is "
                "configured for {}, given image at {}".format(
                    self.image_size,
                    img.size,
                ),
            )

        total_length = img.size[0] * img.size[1]

        ffi.memmove(self._working_image.buf, img.tobytes(), total_length)

        results = lib.apriltag_detector_detect(
            self._detector,
            self._working_image,
        )
        try:
            for i in range(results.size):
                detection = lib.zarray_get_detection(results, i)
                try:
                    yield detection
                finally:
                    lib.destroy_detection(detection)
        finally:
            lib.zarray_destroy(results)
