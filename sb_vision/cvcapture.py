"""
Low-level device capture utility.

This builds upon some functionality sneaked in here from OpenCV.
"""

import threading

from sb_vision.native import _cvcapture


class CaptureDevice(object):
    """A single device for capturing images."""

    def __init__(self, device_id):
        """
        General initialiser.

        Where a ``device_id`` is the udev 'MINOR' device number for the camera
        device.
        """
        self.lock = threading.Lock()
        self.instance = _cvcapture.lib.cvopen(device_id)
        if self.instance == _cvcapture.ffi.NULL:
            raise RuntimeError("Unable to open capture device {}".format(device_id))

    def capture(self, width, height):
        """Capture a single image with the given width and height."""
        if self.instance is None:
            raise RuntimeError("capture device is closed")

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
            raise RuntimeError("cvcapture() failed")

        return bytes(_cvcapture.ffi.buffer(capture_buffer))

    def __enter__(self):
        """Context manager protocol. Automatically closes on exit."""
        return self

    def __exit__(self, exc_value, exc_type, exc_traceback):
        """Context manager protocol. Automatically closes on exit."""
        self.close()

    def close(self):
        """
        Close the device for further access.

        This enables, for instance, other processes to use this device.
        """
        if self.instance is not None:
            with self.lock:
                _cvcapture.lib.cvclose(self.instance)
            self.instance = None

    __del__ = close


def clean_and_threshold(image, width, height):
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
