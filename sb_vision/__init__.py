"""
SourceBots vision system.

Configure a source of images (a `Camera`, or a `FileCamera`), and
process images into sets of `Token`s, representing the detection
of AprilTags markers.
"""

from .camera import Camera, FileCamera
from .tokens import Token
from .vision import Vision

__all__ = [
    'Vision',
    'Camera',
    'FileCamera',
    'Token',
]
