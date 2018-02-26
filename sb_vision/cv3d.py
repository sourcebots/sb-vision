"""
Python wrapping around cffi layer to interop to OpenCV 3D functions.
"""

from typing import Sequence, Tuple

from sb_vision.native import _cv3d  # type: ignore

from .coordinates import Cartesian, Orientation, PixelCoordinate


class Cv3dError(RuntimeError):
    """A 3D related OpenCV error."""

    pass


def _ffi_flattened_float_array(values: Sequence[Sequence[float]]):
    if not values:
        raise ValueError("Refusing to create ffi matrix for empty 'values'.")

    lengths = list({len(x) for x in values})
    if len(lengths) != 1:
        raise ValueError(
            "'values' is not rectangular (got row lengths of {} and {})".format(
                ", ".join(str(x) for x in lengths[:-1]),
                lengths[-1],
            ),
        )

    flattened_values = sum([tuple(x) for x in values], ())
    count = len(values) * lengths[0]

    return _cv3d.ffi.new('double[{}]'.format(count), flattened_values)


def solve_pnp(
    object_points: Sequence[Sequence[float]],
    pixel_corners: Sequence[PixelCoordinate],
    camera_matrix: Sequence[Sequence[float]],
    distance_coefficients: Sequence[Sequence[float]],
) -> Tuple[Cartesian, Orientation]:
    """
    Wrapper around OpenCV solvePnP.

    See the OpenCV docs for details.
    """
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d

    orientation_vector = _cv3d.ffi.new('double[3]')
    translation_vector = _cv3d.ffi.new('double[3]')

    return_value = _cv3d.lib.solve_pnp(
        _ffi_flattened_float_array(object_points),
        _ffi_flattened_float_array(pixel_corners),
        _ffi_flattened_float_array(camera_matrix),
        _ffi_flattened_float_array(distance_coefficients),
        orientation_vector,
        translation_vector,
    )
    if not return_value:
        raise Cv3dError("OpenCV solvePnP failed")

    return (
        Cartesian(*translation_vector),
        Orientation(*orientation_vector),
    )
