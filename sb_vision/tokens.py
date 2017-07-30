"""Tokens detections, and the utilities to manipulate them."""

import math

import numpy as np
import scipy.linalg


def _row_mul(m, corner, col):
    return m[col, 0] * corner[0] + m[col, 1] * corner[1] + m[col, 2]


def _homography_transform(corner, homog):
    """
    Perform the equivalent of an OpenCV WarpPerspectiveTransform on the points.

    See http://bit.ly/2eQOTue for the equation.
    """
    z = _row_mul(homog, corner, 2)
    x = _row_mul(homog, corner, 0) / z
    y = _row_mul(homog, corner, 1) / z
    return x, y


def _decompose_homography(Homography, Calibration):
    """
    Python rewrite of the openCV function decomposeHomographyMat.

    :param Homography:
    :param Calibration:
    :return: Rotation list, Translation List, Normals List
    """
    # This would be the first step for properly calculating the relative
    # location of the marker
    pass


def _get_pixel_corners(homog):
    """
    Get the co-ordinate of the corners given the homography matrix.

    :param homog: Numpy array Homography matrix as returned from Apriltags
    :return: list of (x,y) pixel co-ordinates of the corners of the token
    """
    # Define the corners of the marker
    corners = np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)])

    transformed = []

    for corner in corners:
        x, y = _homography_transform(corner, homog)
        transformed.append((x, y))

    return transformed


def _get_pixel_centre(homography_matrix):
    """Get the centre of the transform (ie how much translation there is)."""
    return _homography_transform((0, 0), homography_matrix)


def bees(v):
    return v[:-1] / v[-1]


def _get_cartesian(
    homography_matrix,
    image_size,
    focal_length,
    marker_size,
):
    calibration_matrix = np.array([
        [focal_length * image_size[0], 0.0, 0.5 * image_size[0]],
        [0.0, focal_length * image_size[1], 0.5 * image_size[1]],
        [0.0, 0.0, 1.0],
    ])

    translations = decompose_homography_matrix(
        homography_matrix,
        calibration_matrix,
    )

    # Require only solutions in front of the camera
    translations = [
        x
        for x in translations
        if x[2] > 0
    ]

    print(translations)
    raise KeyError('?')






    homography_matrix_with_fourth_column = np.array([
        homography_matrix[:, 0],
        homography_matrix[:, 1],
        np.cross(
            homography_matrix[:, 0],
            homography_matrix[:, 1],
        ),
        homography_matrix[:, 2],
    ]).T

    homography_matrix_with_fourth_column /= \
      homography_matrix_with_fourth_column[2,3]

    pose_matrix = scipy.linalg.solve(
        calibration_matrix,
        homography_matrix_with_fourth_column,
    )

    import pdb; pdb.set_trace()

    return pose_matrix[:, 3] / np.mean(marker_size)



DEFAULT_TOKEN_SIZE = (0.25, 0.25)


class Token:
    """Representation of the detection of one token."""

    def __init__(self, id, size=DEFAULT_TOKEN_SIZE, certainty=0.0):
        """
        General initialiser.

        This covers the main token properties but notably does _not_ populate
        the coordinate information.
        """
        self.id = id
        self.size = size
        self.certainty = certainty

    @classmethod
    def from_apriltag_detection(
        cls,
        apriltag_detection,
        sizes,
        image_size,
        focal_length
    ):
        """Construct a Token from an April Tag detection."""
        # *************************************************************************
        # NOTE: IF YOU CHANGE THIS PLEASE ADD THEM IN THE ROBOT-API camera.py
        # *************************************************************************

        instance = cls(
            id=apriltag_detection.id,
            size=sizes.get(apriltag_detection.id, DEFAULT_TOKEN_SIZE),
            certainty=apriltag_detection.goodness,
        )

        arr = [apriltag_detection.H.data[x] for x in range(9)]
        homography = np.reshape(arr, (3, 3))

        instance.infer_location_from_homography_matrix(
            homography_matrix=homography,
            focal_length=focal_length,
            image_size=image_size,
        )
        return instance

    def infer_location_from_homography_matrix(
        self,
        *,
        homography_matrix,
        focal_length,
        image_size
    ):
        """Infer coordinate information from a homography matrix."""
        # pixel coordinates of the corners of the marker
        self.pixel_corners = _get_pixel_corners(homography_matrix)
        # pixel coordinates of the centre of the marker
        self.pixel_centre = _get_pixel_centre(homography_matrix)
        self.homography_matrix = homography_matrix

        # We don't set cartesian and polar coordinates in the absence of a
        # focal length.
        if focal_length is None:
            return

        # Cartesian Co-ordinates in the 3D World, relative to the camera
        # (as opposed to somehow being compass-aligned)
        self.cartesian = _get_cartesian(
            homography_matrix,
            image_size,
            focal_length,
            self.size,
        )
        # Polar Co-ordinates in the 3D World, relative to the front of the
        # camera
        #self.polar = _cart_to_polar(self.cartesian)

    def __repr__(self):
        """General debug representation."""
        return "Token: {}, certainty:{}".format(self.id, self.certainty)

    __str__ = __repr__

    def __eq__(self, other):
        """Equivalent relation partitioning by `id`."""
        if not isinstance(other, Token):
            return False

        return self.id == other.id
