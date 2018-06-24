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

PixelCoordinate = NamedTuple('PixelCoordinate', [('x', AnyFloat), ('y', AnyFloat)])


def cartesian_to_spherical(cartesian: Cartesian) -> Spherical:
    """Convert a Cartesian coordinate into a spherical one."""
    x, y, z = cartesian
    return Spherical(
        rot_x=arctan2(y, z),
        rot_y=arctan2(x, z),
        dist=linalg.norm(cartesian),
    )
