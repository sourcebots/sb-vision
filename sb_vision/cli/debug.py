"""Debug code, load the first video device seen and capture an image."""

import contextlib
import pathlib

from ..camera import Camera, CameraBase, FileCamera  # noqa: F401
from ..token_display import display_tokens
from ..vision import Vision


def _capture_and_display_image(
    vision,
    after_thresholding,
    draw_tokens,
    show_image,
    output_file,
):
    img = vision.capture_image()
    tokens = vision.process_image(img)

    if after_thresholding:
        img = vision.threshold_image(img)

    if draw_tokens:
        img = display_tokens(tokens, img)

    if show_image:
        img.show()

    if output_file is not None:
        img.save(output_file)

    print("Saw {} token(s):".format(len(tokens)))

    for token in tokens:
        print("- {}".format(token))
        try:
            print(token.cartesian)
        except AttributeError:
            pass


def main(input_file, device_id, distance_model, **options):
    """Execute this command."""
    with contextlib.suppress(KeyboardInterrupt):
        if device_id is not None:
            CAM_IMAGE_SIZE = (1280, 720)
            camera = Camera(
                device_id,
                CAM_IMAGE_SIZE,
                distance_model,
            )  # type: CameraBase

            def should_continue():
                return input(
                    "Process next frame? [Y/n]: ",
                ).lower() in ('y', '')

        else:
            camera = FileCamera(input_file, distance_model)

            def should_continue():
                return False

        vision = Vision(camera)
        while True:
            _capture_and_display_image(vision, **options)

            if not should_continue():
                break


def add_arguments(parser):
    """Add arguments for this command."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f',
        '--input-file',
        type=pathlib.Path,
        help="An image file to use as input.",
    )
    group.add_argument(
        '-i',
        '--device-id',
        help="Device id of the camera to use.",
        type=int,
    )
    parser.add_argument(
        '-m',
        '--distance-model',
        type=pathlib.Path,
        help="Distance model to use.",
    )

    parser.add_argument(
        '--after-thresholding',
        action='store_true',
        help="Apply thresholding to the image before output.",
    )
    parser.add_argument(
        '--draw-tokens',
        action='store_true',
        help="Draw tokens on top of the iamge before output.",
    )
    parser.add_argument(
        '--no-show-image',
        dest='show_image',
        action='store_false',
        default=True,
        help="Don't show the processed image in a GUI window.",
    )
    parser.add_argument(
        '--save-image',
        dest='output_file',
        type=pathlib.Path,
        help="Save the processed image (useful with --after-thresholding and "
             "--draw-tokens). Beware using this with a camera as the same "
             "file name will be used for all frames.",
    )
