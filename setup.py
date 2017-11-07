from setuptools import setup, find_packages


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
    ],
    cffi_modules=[
        'sb_vision/native/cvcapture_build.py:ffibuilder',
        'sb_vision/native/apriltag/apriltag_build.py:ffi',
    ],
    install_requires=[
        'Pillow',
        'numpy',
        'scipy',
        'cffi>=1.4.0',
        'pytest-runner',
    ],
    extras_require={
        'calibration': [
            'pyyaml',
            'scikit-learn',
        ],
    },
    tests_require=[
        'pytest',
        'pytest-flake8',
        'flake8',
        'flake8-docstrings',
        'flake8-isort',
        'flake8-mutable',
        'flake8-debugger',
        'flake8-comprehensions',
        'flake8-todo',
    ],
    zip_safe=False,
)
