"""
Cameras: sources of images.

We use the name `Camera` here but it's infact a "generalise" camera: a
Camera only has to serve as a source of images for further processing. A
Camera class could, for instance, fetch images from a file.
"""

from PIL import Image

from sb_vision.cvcapture import CaptureDevice

from .camera_base import CameraBase


class Camera(CameraBase):
    """Actual camera, hooked up to a physical device."""

    def __init__(self, device_id, proposed_image_size, distance_model):
        """Initialise camera with focal length and image size."""
        super().__init__()
        self.cam_image_size = proposed_image_size
        self.device_id = device_id
        self.camera = None
        self.distance_model = distance_model

    def init(self):
        """Open the actual device."""
        super().init()  # Call parent
        self._init_camera()

    def _init_camera(self):
        self.camera = CaptureDevice(self.device_id)

    def _deinit_camera(self):
        if self.camera:
            self.camera.close()
        self.camera = None

    def __del__(self):
        """Make sure that the camera is deinitialised on closing."""
        self._deinit_camera()

    def capture_image(self):
        """
        Capture an image.

        :return: PIL image object of the captured image in Luminosity
                 color scale
        """
        image_bytes = self.camera.capture(*self.cam_image_size)
        return Image.frombytes('L', self.cam_image_size, image_bytes)


class FileCamera(CameraBase):
    """Pseudo-camera debug class, getting images from files."""

    def __init__(self, file_path, distance_model):
        """Open from a given path, with a given pseudo-focal-length."""
        super().__init__()
        self.file_name = file_path
        self.image = None
        self.distance_model = distance_model

    def init(self):
        """Open the file and read in the image."""
        super().init()
        self.image = Image.open(self.file_name).convert('L')
        self.cam_image_size = self.image.size

    def capture_image(self):
        """
        Capture a single image.

        Actually this just means returning the single image we loaded
        from a file.
        """
        if self.image is None:
            raise RuntimeError("init() not called")
        return self.image
