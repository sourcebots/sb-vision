"""Tokens detections, and the utilities to manipulate them."""

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from .coordinates import (
    Cartesian,
    Orientation,
    cartesian_to_legacy_polar,
    cartesian_to_spherical,
)
from .find_3D_coords import (
    PixelCoordinate,
    calculate_transforms,
    load_camera_calibrations,
)
from .game_specific import MARKER_SIZE_DEFAULT, MARKER_SIZES

if TYPE_CHECKING:
    # Interface-only definitions
    from .native.apriltag.types import ApriltagDetection  # noqa: F401


class Token:
    """Representation of the detection of one token."""

    def __init__(
        self,
        id: int,
        certainty=0,
        pixel_corners: List[PixelCoordinate] = None,
        pixel_centre: PixelCoordinate = PixelCoordinate(0.0, 0.0),
    ) -> None:
        """
        General initialiser.

        This covers the main token properties but notably does _not_ populate
        the coordinate information.
        """
        self.id = id
        self.certainty = certainty
        self.pixel_corners = pixel_corners
        self.pixel_centre = pixel_centre

    @classmethod
    def from_apriltag_detection(
        cls,
        apriltag_detection: 'ApriltagDetection',
        camera_model: Optional[str],
    ) -> 'Token':
        """Construct a Token from an April Tag detection."""

        pixel_corners = [PixelCoordinate(*l) for l in apriltag_detection.p]

        # centre of marker: average the corners
        pixel_centre = PixelCoordinate(*np.average(pixel_corners, axis=0))

        marker_id = apriltag_detection.id

        # *************************************************************************
        # NOTE: IF YOU CHANGE THIS, THEN CHANGE ROBOT-API camera.py
        # *************************************************************************

        # The pixel_corners value we expose is in clockwise order starting with
        # the bottom left corner of the marker (if it weren't rotated).
        # AprilTags returns an array with the first being the top left. thus we need to
        # shift the ordering of the values by one to match our output.
        offset_pixel_corners = [pixel_corners[3]] + pixel_corners[:3]

        instance = cls(
            id=marker_id,
            certainty=apriltag_detection.goodness,
            pixel_corners=offset_pixel_corners,
            pixel_centre=pixel_centre,
        )

        # We don't set coordinates in the absence of a camera model.
        if camera_model:
            camera_matrix, distance_coefficients = load_camera_calibrations(
                camera_model,
            )

            translation, orientation = calculate_transforms(
                MARKER_SIZES.get(marker_id, MARKER_SIZE_DEFAULT),
                pixel_corners,
                camera_matrix,
                distance_coefficients,
            )

            instance.update_3D_transforms(
                translation=translation,
                orientation=orientation,
            )
        return instance

    # noinspection PyAttributeOutsideInit
    def update_3D_transforms(
        self,
        *,
        translation: Cartesian,
        orientation: Orientation
    ):
        """Calculate 3D coordinate information from the given transformations."""
        # Cartesian Co-ordinates in the 3D World, relative to the camera
        # (as opposed to somehow being compass-aligned)
        self.cartesian = translation

        self.orientation = orientation

        # Polar co-ordinates in the 3D world, relative to the camera
        self.polar = cartesian_to_legacy_polar(self.cartesian)
        self.legacy_polar = cartesian_to_legacy_polar(self.cartesian)

        self.spherical = cartesian_to_spherical(self.cartesian)

    def __repr__(self) -> str:
        """General debug representation."""
        return "Token: {}, certainty: {}".format(self.id, self.certainty)

    __str__ = __repr__

    def __eq__(self, other: Any) -> bool:
        """Equivalent relation partitioning by `id`."""
        if not isinstance(other, Token):
            return False

        return self.id == other.id
