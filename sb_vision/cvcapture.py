"""
Low-level device capture utility.

This builds upon some functionality sneaked in here from OpenCV.
"""

import numpy
import threading

from sb_vision.native import _cvcapture


class CaptureDevice(object):
    """A single device for capturing images."""

    def __init__(self, path=None):
        """
        General initialiser.

        Where a device path is well-defined (for instance, v4l devices in
        `/dev` on Linux) this can be passed in; otherwise a default device
        is used).
        """
        self.lock = threading.Lock()
        if path is not None:
            argument_c = _cvcapture.ffi.new(
                'char[]',
                path.encode('utf-8'),
            )
        else:
            argument_c = _cvcapture.ffi.NULL
        self.instance = _cvcapture.lib.cvopen(argument_c)
        if self.instance == _cvcapture.ffi.NULL:
            raise RuntimeError("Unable to open capture device")

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


def decompose_homography_matrix(homography, intrinsics):
    homography = numpy.array(homography).astype('float64')
    intrinsics = numpy.array(intrinsics).astype('float64')

    if homography.shape != (3, 3):
        raise ValueError("Homography must be a 3x3 matrix")

    if intrinsics.shape != (3, 3):
        raise ValueError("Intrinsics must be a 3x3 matrix")

    homography_ptr = _cvcapture.ffi.cast('double*', homography.ctypes.data)
    intrinsics_ptr = _cvcapture.ffi.cast('double*', intrinsics.ctypes.data)

    max_outputs = 4
    outputs = numpy.zeros(max_outputs * 3, dtype='float64')

    outputs_ptr = _cvcapture.ffi.cast('double*', outputs.ctypes.data)

    actual_outputs = _cvcapture.lib.cv_decompose_homography(
        homography_ptr,
        intrinsics_ptr,
        outputs_ptr,
        max_outputs,
    )

    if actual_outputs <= 0:
        raise ValueError("cv_decompose_homography() failed")

    outputs = outputs[:3 * actual_outputs]

    return outputs.reshape((-1, 3))
