"""Definitions for training examples."""

from ..camera import FileCamera
from ..vision import Vision


class TrainingExample(object):
    """A single example (image/pose pair) for model training."""

    def __init__(self, *, image_file, z_distance, x_offset_right=0.0):
        """Construct from image file and distance metrics."""
        self.z_distance = z_distance
        self.x_offset_right = x_offset_right
        self.image_file = image_file

        self._load()

    def _load(self):
        """Load and recognise the single marker from this image."""
        pseudo_camera = FileCamera(self.image_file, distance_model=None)
        vision = Vision(pseudo_camera)

        image = vision.capture_image()
        self.size = image.size
        try:
            (single_token,) = vision.process_image(image)
        except ValueError:
            raise RuntimeError(
                "Cannot detect token in {}".format(self.image_file.name),
            )

        # Explicitly clean up vision. The `Vision` instance has some baked-in
        # assumptions about being a singleton and controls overall
        # initialisation and de-initialisation of the apriltags library.
        # We delete it so that `__del__` de-initialises the library, since
        # `_load` is called multiple times.
        del vision

        self.homography_matrix = single_token.homography_matrix

    def __repr__(self):
        """Debug representation."""
        return '<TrainingExample file={file}>'.format(file=self.image_file)

    __str__ = __repr__
