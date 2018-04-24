# sb-vision

[![CircleCI](https://circleci.com/gh/sourcebots/sb-vision.svg?style=shield)](https://circleci.com/gh/sourcebots/sb-vision)

Vision subsystem for SourceBots, built on top of AprilTags and OpenCV.


## Developing

### Setting up

While sb-vision vendors in the apriltags library, it expects opencv to be
installed from the system. On Ubuntu this means installing the `python3-dev` and
`libopencv-dev` packages. You'll also need a C++ compiler to build the native
components.

Developers are encouraged to make use of a [virtual environment](https://docs.python.org/3/tutorial/venv.html)
to avoid conflicts with other projects or system packages.

This package can be installed for development with:
```
pip install -e .
```
This command can also be re-issued to rebuild the native components when they
are changed.

The Python dependencies you'll need when developing (linters and testing tools)
can be installed with:
```
pip install -r script/requirements-dev.txt
```

### Useful commands

The `sb_vision` module is runnable and provides commands useful for debugging
the library as well as for some simple use-cases. Full details are in the
`--help` output, though some useful commands are:

- `debug`: simple usage of the library for easy debugging. Supports both loading
  from a file and capturing from a camera.
- `summarise`: summarise the markers found in a directory of images. Output is
  in the same format as the YAML files needed for calibration (this can be
  useful to see more detail about how good a calibration is).

### Style: linters & type hints

This project uses `flake8` with various plugins for linting, `isort` for
ordering imports and type hints checked by `mypy`. All are configured via
`setup.cfg` and have wrapper scripts in the `script` directory. Developers are
encouraged to enable their IDE integrations for all of these tools.

The code style of this project differs from PEP 8 in that it tries to lay code
out such that diff-noise will be reduced if changes are needed. This often means
that code requires more vertical space than might be expected, though this is an
accepted trade-off.

### Testing

Tests are implemented using `pytest`.
Run them by running `python setup.py test` (or `pytest`).


## Co-ordinate spaces

sb-vision describes the locations of identified markers in either Cartesian or
spherical co-ordinate spaces. A "legacy polar" co-ordinate space exists for
backwards compatibility but is deprecated, undocumented and will be removed in
May 2018.

### Cartesian co-ordinates

Standard three-axis relative position co-ordinates:
 - `x`: translation left (negative) or right (positive)
 - `y`: translation down (negative) or up (positive)
 - `z`: translation away from the observer

Available as a typed named tuple `Cartesian` from `Token.cartesian`.

### Spherical co-ordinates

A non-standard two-angle co-ordinate system built on the assumption that most
markers are roughly level with the observer:
 - `rot_x`: rotation about the Cartesian x-axis, positive angles are "upwards"
 - `rot_y`: rotation about the Cartesian y-axis, positive angles are "rightwards"
 - `dist`: absolute distance from the observer to the marker

Available as a typed named tuple `Spherical` from `Token.spherical`.

### Orientation

Rotation of the marker around its centre (relative to the observer)
 - `rot_x`: rotation about the Cartesian x-axis, positive angles represent the front going downwards.
 - `rot_y`: rotation about the Cartesian y-axis, positive angles represent the front going leftwards.
 - `rot_z`: rotation about the Cartesian z-axis, positive angles represent the top going rightwards.

Available as a typed named tuple `Orientation` from `Token.orientation`.

## Camera support

Since each model of camera is different, sb-vision has support for calibrating for
different cameras through the standard [OpenCV calibration tool][calibration-tool],
which produces an xml calibration file. This file is placed in `sb_vision` directory.
Any extra calibrations can be shipped as part of the package via addition to the
`MANIFEST.in` file.

### Compatibility note

`sb-vision` currently only has support for the TeckNet C016 camera. Since the
only known client of this project ([`robotd`](https://github.com/sourcebots/robotd))
has been asking for the `'c270'` model for all connected cameras, this project
offers the C016 camera model under that name as well.

This behaviour is deprecated and will be removed in June 2018.


[calibration-tool]: https://docs.opencv.org/3.4.0/d7/d21/tutorial_interactive_calibration.html
