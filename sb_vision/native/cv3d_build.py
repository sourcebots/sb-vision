"""
CFFI build script for the cv3d native module.

This little bit of code is responsible for fondling OpenCV so that we can
extract 3D position information from the 2D information we get from AprilTags.
"""

from pathlib import Path

import cffi

base = Path(__file__).parent

ffibuilder = cffi.FFI()

CV3D_DECLS = """
    int solve_pnp(
        const void* object_points,
        const void* image_points,
        const void* camera_matrix,
        const void* dist_coeffs,
        void* rvec,
        void* tvec
    );
"""

ffibuilder.set_source(
    "sb_vision.native._cv3d",
    CV3D_DECLS,
    sources=[
        str(base / 'cv3d.cpp'),
    ],
    libraries=[
        'opencv_core',
        'opencv_calib3d',
    ],
)

ffibuilder.cdef(CV3D_DECLS)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
