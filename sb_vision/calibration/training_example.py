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
        pseudo_camera = FileCamera(self.image_file, focal_length=None)
        vision = Vision(pseudo_camera, (0.01, 0.01))

        (single_token,) = vision.snapshot()

        # Explicitly clean up vision
        del vision

        self.homography_matrix = single_token.homography_matrix

    def __repr__(self):
        """Debug representation."""
        return '<TrainingExample file={file}>'.format(file=self.image_file)

    __str__ = __repr__
