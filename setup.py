from setuptools import setup, find_packages


setup(
    name='sb-vision',
    version='1.0',
    description="Vision system for SourceBots",
    author="SourceBots",
    author_email='',
    packages=find_packages(),
    setup_requires=[
        'cffi>=1.4.0',
    ],
    cffi_modules=[
        'sb_vision/native/cvcapture_build.py:ffibuilder',
        'sb_vision/native/apriltag/apriltag_build.py:ffi',
    ],
    install_requires=[
        'Pillow',
        'numpy',
        "cffi>=1.4.0",
    ],
    zip_safe=False,
)
