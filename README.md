# sb-vision

[![CircleCI](https://circleci.com/gh/sourcebots/sb-vision.svg?style=shield)](https://circleci.com/gh/sourcebots/sb-vision)

Vision subsystem for SourceBots, built on top of AprilTags and OpenCV.


## Developing

While sb-vision vendors in the apriltags library, it expects opencv to be
installed from the system. On Ubuntu this means installing the `python3-dev` and
`libopencv-dev` packages. You'll also need a C++ compiler to build the native
components.

### Useful commands

The `sb_vision` module is runnable and provides commands useful for
debugging the library as well as for some simple use-cases. Full details are in the `--help` output, though some useful commands are:

- `debug`: simple usage of the library for easy debugging. Supports both loading
  from a file and capturing from a camera.
- `summarise`: summarise the markers found in a directory of images. Output is
  in the same format as the YAML files needed for calibration (this can be
  useful to see more detail about how good a calibration is).


## Testing

Tests are implemented using `pytest`. Run them by running `pytest`.


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


## Camera support

Since each model of camera is different, sb-vision has support for calibrating for
different cameras through the standard [OpenCV calibration tool](calibration-tool),
which produces an xml calibration file. This file is placed in `sb_vision` directory.
Any extra calibrations can be shipped as part of the package via addition to the
`MANIFEST.in` file.
