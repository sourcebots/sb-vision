"""
Base class and utilities for cameras.

A camera is an arbitrary source of images.
"""

import abc
from typing import Tuple

import PIL


class CameraBase(metaclass=abc.ABCMeta):
    """Base class for all cameras."""

    def __init__(self, distance_model: str) -> None:
        """Basic, general initialisation."""
        self.distance_model = distance_model

    def init(self) -> None:
        """
        Initialise the camera.

        This is guaranteed to only be called once.
        """
        pass

    @abc.abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of images captured by the camera."""
        raise NotImplementedError()

    @abc.abstractmethod
    def capture_image(self) -> PIL.Image:
        """
        Capture a single image from this camera.

        :return: PIL Image captured
        """
        raise NotImplementedError()
