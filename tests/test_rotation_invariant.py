import tempfile

from PIL import Image
from pathlib import Path

from pytest import approx

from sb_vision import Vision
from sb_vision.camera import VirtualFileCamera

test_dir = Path(__file__).parent / 'test_data'


def gen_location_rotation(image_file, pos, rotation, width=1280, height=720):
    """
    Generate a marker in a specific position in an image with a specific rotation.
    """
    img = Image.open(test_dir / image_file)
    img = img.rotate(rotation, expand=True)
    backing = Image.new("RGB", (width, height))
    backing.paste(img, (round(width / 2 + pos[0] - img.width / 2),
                        round(height / 2 + pos[1] - img.height / 2)))
    return backing


PRECISION = 0.01


def test_marker_rotation_invariant():
    """
    Make sure that the positional co-ordinates do not vary on rotation

    Thus, the centre is in the middle of the marker.
    """
    pos = None
    for rot in range(0, 360, 45):
        image = gen_location_rotation('marker_cropped.png', (150, 150), rot)
        vision = Vision(VirtualFileCamera(image, camera_model='C016'))
        snap = vision.process_image(image)[0]
        if pos is not None:
            assert snap.cartesian.x == approx(pos.x, rel=PRECISION)
            assert snap.cartesian.y == approx(pos.y, rel=PRECISION)
            assert snap.cartesian.z == approx(pos.z, rel=PRECISION)
        else:
            pos = snap.cartesian
