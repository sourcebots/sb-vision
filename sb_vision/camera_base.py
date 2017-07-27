"""
Base class and utilities for cameras.

A camera is an arbitrary source of images.
"""

import abc


class CameraBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.initialised = False
        self.cam_image_size = None
        self.focal_length = None

    def init(self):
        """ Initialise the camera"""
        self.initialised = True

    def get_image_size(self):
        """ Get the size of images captured by the camera"""
        if not self.initialised:
            raise RuntimeError(
                "Must Initialise camera before getting image size",
            )
        return self.cam_image_size

    @abc.abstractmethod
    def capture_image(self):
        """
        :return: PIL Image captured
        """
        raise NotImplementedError()
