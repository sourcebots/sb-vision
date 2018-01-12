"""
Low-level device capture utility.

This builds upon some functionality sneaked in here from OpenCV.
"""

import threading

from sb_vision.native import _cvcapture  # type: ignore


class CvCaptureError(RuntimeError):
    """A generic OpenCV error."""

    pass


class DeviceOpenError(CvCaptureError):
    """An error when OpenCV cannot open a device."""

    def __init__(self, device_id: int) -> None:
        """Initialise the exception."""
        super().__init__("Unable to open capture device {}".format(device_id))


class DeviceClosedError(CvCaptureError):
    """An error when OpenCV cannot close a device."""

    def __init__(self) -> None:
        """Initialise the exception."""
        super().__init__("capture device is closed")


class ImageCaptureError(CvCaptureError):
    """An error when OpenCV cannot capture an image."""

    def __init__(self) -> None:
        """Initialise the exception."""
        super().__init__("cvcapture() failed")


class CaptureDevice(object):
    """A single device for capturing images."""

    def __init__(self, device_id: int) -> None:
        """
        Initialise the capture device.

        The ``device_id`` is the udev 'MINOR' device number for the camera
        device.
        """
        self.lock = threading.Lock()
        self.instance = _cvcapture.lib.cvopen(device_id)
        if self.instance == _cvcapture.ffi.NULL:
            raise DeviceOpenError(device_id)

    def capture(self, width: int, height: int) -> bytes:
        """Capture a single image with the given width and height."""
        if self.instance is None:
            raise DeviceClosedError()

        capture_buffer = _cvcapture.ffi.new(
            'uint8_t[{}]'.format(width * height),
        )

        with self.lock:
            status = _cvcapture.lib.cvcapture(
                self.instance,
                capture_buffer,
                width,
                height,
            )

        if status == 0:
            raise ImageCaptureError()

        return bytes(_cvcapture.ffi.buffer(capture_buffer))

    def __enter__(self) -> 'CaptureDevice':
        """Context manager protocol. Automatically closes on exit."""
        return self

    def __exit__(self, exc_value, exc_type, exc_traceback):
        """Context manager protocol. Automatically closes on exit."""
        self.close()

    def close(self) -> None:
        """
        Close the device for further access.

        This enables, for instance, other processes to use this device.
        """
        if self.instance is not None:
            with self.lock:
                if self.instance is not None:
                    _cvcapture.lib.cvclose(self.instance)
                    self.instance = None

    __del__ = close


def clean_and_threshold(image: bytes, width: int, height: int) -> bytes:
    """Prepare an image (as bytes) with thresholding/filtering."""
    if len(image) != width * height:
        raise ValueError("image has the wrong length")

    source_buffer = _cvcapture.ffi.new(
        'uint8_t[{}]'.format(width * height),
    )
    dst_buffer = _cvcapture.ffi.new(
        'uint8_t[{}]'.format(width * height),
    )

    _cvcapture.ffi.memmove(
        source_buffer,
        image,
        len(image),
    )

    _cvcapture.lib.cvthreshold(
        source_buffer,
        dst_buffer,
        width,
        height,
    )

    return bytes(_cvcapture.ffi.buffer(dst_buffer))
