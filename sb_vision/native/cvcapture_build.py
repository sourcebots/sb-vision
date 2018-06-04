"""
CFFI build script for the cvcapture native module.

This little bit of code is responsible for fondling OpenCV so that we can get
snapshots from webcams.
"""

from pathlib import Path

import cffi

base = Path(__file__).parent

ffibuilder = cffi.FFI()

CVCAPTURE_DECLS = """
    int cvcapture(void* context, void* buffer, size_t width, size_t height);
    void* cvopen(const int path);
    void cvclose(void* context);
"""

ffibuilder.set_source(
    "sb_vision.native._cvcapture",
    CVCAPTURE_DECLS, sources=[
        str(base / 'cvcapture.cpp'),
    ], libraries=[
        'opencv_core',
        'opencv_highgui',
        'opencv_imgproc',
        # Uncomment if you have an issue with _ZNK2cv12VideoCapture8isOpenedEv as you've got a newer 
        # version of opencv than the current build.
        # 'opencv_videoio', 
    ],
)

ffibuilder.cdef(CVCAPTURE_DECLS)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
