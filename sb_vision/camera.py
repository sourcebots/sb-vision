"""
Cameras: sources of images.

We use the name `Camera` here but it's infact a "generalise" camera: a
Camera only has to serve as a source of images for further processing. A
Camera class could, for instance, fetch images from a file.
"""

import pathlib
from typing import Optional, Tuple, Union  # noqa: F401

from PIL import Image

from sb_vision.cvcapture import CaptureDevice

from .camera_base import CameraBase

_PathLike = Union[str, pathlib.Path]


class Camera(CameraBase):
    """Actual camera, hooked up to a physical device."""

    def __init__(
        self,
        device_id: int,
        proposed_image_size: Tuple[int, int],
        camera_model: Optional[str],
    ) -> None:
        """Initialise camera with focal length and image size."""
        super().__init__(camera_model)
        self.cam_image_size = proposed_image_size
        self.device_id = device_id
        self.camera = None  # type: Optional[CaptureDevice]

    def init(self) -> None:
        """Open the actual device."""
        super().init()  # Call parent
        self._init_camera()

    def _init_camera(self) -> None:
        self.camera = CaptureDevice(self.device_id)

    def _deinit_camera(self) -> None:
        if self.camera:
            self.camera.close()
        self.camera = None

    def __del__(self) -> None:
        """Make sure that the camera is deinitialised on closing."""
        self._deinit_camera()

    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of images captured by the camera."""
        if self.camera is None:
            raise RuntimeError(
                "Must initialise camera before getting image size",
            )
        return self.cam_image_size

    def capture_image(self) -> Image:
        """
        Capture an image.

        :return: PIL image object of the captured image in Luminosity
                 color scale
        """
        if self.camera is None:
            raise RuntimeError("Capture device not available")

        image_bytes = self.camera.capture(*self.cam_image_size)
        return Image.frombytes('L', self.cam_image_size, image_bytes)


class FileCamera(CameraBase):
    """Pseudo-camera debug class, getting images from files."""

    def __init__(self, file_path: _PathLike, camera_model: Optional[str]) -> None:
        """Open from a given path, with a given pseudo-focal-length."""
        super().__init__(camera_model)
        self.file_name = file_path
        self.image = None  # type: Optional[Image]

    def init(self) -> None:
        """Open the file and read in the image."""
        super().init()
        self.image = Image.open(self.file_name).convert('L')

    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of images captured by the camera."""
        if self.image is None:
            raise RuntimeError(
                "Must initialise camera before getting image size",
            )
        return self.image.size

    def capture_image(self) -> Image:
        """
        Capture a single image.

        Actually this just means returning the single image we loaded
        from a file.
        """
        if self.image is None:
            raise RuntimeError("init() not called")
        return self.image


class VirtualFileCamera(FileCamera):
    def __init__(self, image: Image, camera_model: Optional[str]):
        super().__init__("", camera_model)
        self.image = image.convert('L')

    def init(self):
        pass
