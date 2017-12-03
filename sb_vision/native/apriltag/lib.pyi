class ImageU8:
    width: int
    height: int
    stride: int

    buf: Tuple[int, ...]


def image_u8_create_stride(width: int, height: int, stride: int) -> ImageU8:
    pass
