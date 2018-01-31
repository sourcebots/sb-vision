"""Types and helpers for manipulating coordinates."""

from typing import NamedTuple, Union

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


Polar = NamedTuple('Polar', (
    ('rot_x', AnyFloat),
    ('rot_y', AnyFloat),
    ('dist', AnyFloat),
))


def cartesian_to_polar(cartesian: Cartesian) -> Polar:
    """Convert a Cartesian coordinate into a polar one."""
    x, y, z = cartesian
    return Polar(
        rot_x=arctan2(y, z),
        rot_y=arctan2(x, z),
        dist=linalg.norm(cartesian),
    )