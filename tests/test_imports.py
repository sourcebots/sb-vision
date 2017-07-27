"""Smoke tests for whether modules import at all."""


def test_can_import_cvcapture():
    """
    Make sure we can actually get the cvcapture module.

    This in turns makes sure it can do its own imports—which basically means
    that we can import the cvcapture native library without incurring the wrath
    of the system linker for OpenCV.
    """
    import sb_vision.cvcapture  # noqa
