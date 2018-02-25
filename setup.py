import sys

from setuptools import find_packages, setup

# make pytest-runner be required only when needed
# This means pytest definitely isn't a dependency if tests aren't run.
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


setup(
    name='sb-vision',
    version='1.0',
    description="Vision system for SourceBots",
    author="SourceBots",
    author_email='',
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[
        'cffi>=1.4.0',
    ] + pytest_runner,
    cffi_modules=[
        'sb_vision/native/cvcapture_build.py:ffibuilder',
        'sb_vision/native/cv3d_build.py:ffibuilder',
        'sb_vision/native/apriltag/apriltag_build.py:ffi',
    ],
    install_requires=[
        'Pillow',
        'numpy',
        'scipy',
        "cffi>=1.4.0",
    ],
    tests_require=[
        'pytest',
    ],
    zip_safe=False,
)
