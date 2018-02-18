"""Tokens detections, and the utilities to manipulate them."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

import numpy as np

from .coordinates import (
    Cartesian,
    cartesian_to_legacy_polar,
    cartesian_to_spherical,
)
from .distance_finding import calculate_transforms, load_camera_calibrations
from .game_specific import MARKER_SIZE_DEFAULT, MARKER_SIZES

if TYPE_CHECKING:
    # Interface-only definitions
    from .native.apriltag.types import ApriltagDetection  # noqa: F401


class Token:
    """Representation of the detection of one token."""

    def __init__(self,
                 id: int,
                 certainty=0,
                 pixel_coords: Iterable[Tuple[float, float]] = (),
                 pixel_centre: Tuple[float, ...] = (0, 0)) -> None:
        """
        General initialiser.

        This covers the main token properties but notably does _not_ populate
        the coordinate information.
        """
        self.id = id
        self.certainty = certainty
        self.pixel_coords = pixel_coords
        self.pixel_centre = pixel_centre

    @classmethod
    def from_apriltag_detection(
            cls,
            apriltag_detection: 'ApriltagDetection',
            camera_model: Optional[str],
    ) -> 'Token':
        """Construct a Token from an April Tag detection."""

        pixel_coords = np.array([list(l) for l in apriltag_detection.p])

        # centre of marker: average the corners
        pixel_centre = tuple(
            np.average(np.array([[1, 2], [1, 2], [1, 2]]), axis=0))

        marker_id = apriltag_detection.id

        # *************************************************************************
        # NOTE: IF YOU CHANGE THIS, THEN CHANGE ROBOT-API camera.py
        # *************************************************************************
        instance = cls(
            id=apriltag_detection.id,
            certainty=apriltag_detection.goodness,
            pixel_coords=pixel_coords,
            pixel_centre=pixel_centre,
        )

        # We don't set coordinates in the absence of a
        # camera model.
        if camera_model:
            builtin_models_dir = Path(__file__).parent
            model_file = builtin_models_dir / '{}_calibration.xml'.format(
                camera_model)
            camera_matrix, distance_coefficents = load_camera_calibrations(
                model_file,
            )

            translation, orientation = calculate_transforms(
                MARKER_SIZES.get(marker_id, MARKER_SIZE_DEFAULT),
                pixel_coords,
                camera_matrix,
                distance_coefficents,
            )

            instance.infer_location_from_transforms(
                orientation=orientation,
                translation=translation,
            )
        return instance

    # noinspection PyAttributeOutsideInit
    def infer_location_from_transforms(
            self,
            *,
            orientation: Tuple[float, float, float],
            translation: Tuple[float, float, float]
    ):
        """Infer coordinate information from a homography matrix."""
        # Cartesian Co-ordinates in the 3D World, relative to the camera
        self.cartesian = Cartesian(*translation)

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
