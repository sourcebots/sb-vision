"""Types and helpers for manipulating coordinates."""

from typing import NamedTuple, Union

import numpy as np
from numpy import arctan2, float64, linalg

AnyFloat = Union[float, float64]

_Cartesian = NamedTuple('Cartesian', (
    ('x', AnyFloat),
    ('y', AnyFloat),
    ('z', AnyFloat),
))


class Cartesian(_Cartesian):
    """Cartesian coordinates."""

    __slots__ = ()

    def tolist(self):
        """Placeholder helper to ease migration within robotd."""
        return list(self)


Spherical = NamedTuple('Spherical', (
    ('rot_x', AnyFloat),
    ('rot_y', AnyFloat),
    ('dist', AnyFloat),
))

LegacyPolar = NamedTuple('LegacyPolar', (
    ('polar_x', AnyFloat),
    ('polar_y', AnyFloat),
    ('dist', AnyFloat),
))


PixelCoordinate = NamedTuple('PixelCoordinate', [('x', float), ('y', float)])


def cartesian_to_spherical(cartesian: Cartesian) -> Spherical:
    """Convert a Cartesian coordinate into a spherical one."""
    x, y, z = cartesian
    return Spherical(
        rot_x=arctan2(y, z),
        rot_y=arctan2(x, z),
        dist=linalg.norm(cartesian),
    )


def cartesian_to_legacy_polar(cartesian: Cartesian) -> LegacyPolar:
    """
    Convert cartesian co-ordinate space to the legacy "polar" space.

    This is kept for compatibility only.
    """
    cart_x, cart_y, cart_z = tuple(cartesian)
    polar_dist = np.linalg.norm(cartesian)
    polar_x = np.arctan2(cart_z, cart_x)
    polar_y = np.arctan2(cart_z, cart_y)
    return LegacyPolar(polar_x, polar_y, polar_dist)
