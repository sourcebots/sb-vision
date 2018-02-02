# sb-vision

[![CircleCI](https://circleci.com/gh/sourcebots/sb-vision.svg?style=shield)](https://circleci.com/gh/sourcebots/sb-vision)

Vision subsystem for SourceBots, built on top of AprilTags and OpenCV.


## Developing

While sb-vision vendors in the apriltags library, it expects opencv to be
installed from the system. On Ubuntu this means installing the `python3-dev` and
`libopencv-dev` packages. You'll also need a C++ compiler to build the native
components.


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

Since each camera is different, sb-vision has support for self-calibrating based
on sample images. Calibrations are stored as LZMA compressed pickles of
calibration data which is learned using scikit-learn. The pickles are stored
within the `sb_vision` directory, and shipped as part of the package via
addition to the `MANIFEST.in` file.

To calibrate a new camera, take a number of accurately measured images of a
single AprilTag marker from a variety of distances and positions. All images
should feature the marker facing in the inverse parallel direction to that of
the camera. That is, they should face "towards" the camera, though should only
be translated between images (and not rotated).

The images should be added to a new directory under `calibrations` and a
suitable `files.yaml` created which describes their positions in terms of `x`
and `z` Cartesian co-ordinates. See `calibrations/exmaple/files.yaml` for an
example of the format.

To learn the calibration from the sample images, run:
```bash
python -m sb_vision.calibration -o sb_vision/<camera_model>.pkl.xz calibrations/<camera_model>
```
