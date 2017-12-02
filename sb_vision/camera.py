"""
Cameras: sources of images.

We use the name `Camera` here but it's infact a "generalise" camera: a
Camera only has to serve as a source of images for further processing. A
Camera class could, for instance, fetch images from a file.
"""

from PIL import Image
try:
    from picamera import PiCamera
    CAN_USE_PI_CAMERA = True
except:
    CAN_USE_PI_CAMERA = False

from io import BytesIO
from .camera_base import CameraBase


class Camera(CameraBase):
    """Actual camera, hooked up to a physical device."""

    def __init__(self, device_id, proposed_image_size, distance_model):
        """Initialise camera with focal length and image size."""
        super().__init__()
        self.cam_image_size = proposed_image_size
        self.device_id = device_id
        self.camera = None
        self.stream = None
        self.distance_model = distance_model

    def init(self):
        """Open the actual device."""
        super().init()  # Call parent
        self._init_camera()

    def _init_camera(self):
        self.stream = BytesIO()
        self.camera = PiCamera()

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
        self.camera.capture(self.stream, format='jpeg')
        # "Rewind" the stream to the beginning so we can read its content
        self.stream.seek(0)
        image = Image.open(self.stream)
        gray, _, _ = image.split()
        return gray


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
