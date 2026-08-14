"""Decoding, shared by the detector and the thumbnailer.

This is the other half of the speed story and close to half the total cost.
Museum scans reach tens of megapixels while the detector only wants ~1024px, so
decoding at full resolution and then throwing the pixels away is the single
most expensive mistake available here.

Pillow's `draft()` avoids it for JPEG. The JPEG format stores an image as 8x8
blocks of frequency coefficients, and the decoder can reconstruct those blocks
at 1/2, 1/4 or 1/8 scale for roughly proportional work — so asking for a 1024px
version of an 8000px scan decodes about a sixty-fourth of the data. It is
lossless in the sense that matters: the result is a correctly downscaled image,
not a cropped or degraded one. It applies only to JPEG (and MPO); everything
else decodes in full and is resized afterwards.

`draft()` only scales in powers of two and will not overshoot, so it lands on
the first power of two at or above the target and we finish the last factor
with a normal high-quality resize.
"""

from __future__ import annotations

from PIL import Image

#: Pillow refuses very large files by default as a decompression-bomb guard.
#: Museum scans legitimately reach that size, and this is a local tool pointed
#: at a known drive rather than anything parsing untrusted uploads.
Image.MAX_IMAGE_PIXELS = None

_HEIF_REGISTERED: bool | None = None


def register_heif() -> bool:
    """Teach Pillow about HEIC/HEIF if `pillow-heif` is installed.

    macOS decoded these natively through ImageIO. Pillow does not without the
    plugin, so on Linux an unregistered HEIC is an unreadable file rather than
    a wrong one — it lands in the error column instead of being silently
    skipped.
    """
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED is None:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            _HEIF_REGISTERED = True
        except ImportError:
            _HEIF_REGISTERED = False
    return _HEIF_REGISTERED


def open_downscaled(path: str, max_side: int, mode: str = "RGB") -> tuple[Image.Image, int, int]:
    """Open an image no larger than ``max_side`` on its longest edge.

    Returns the image alongside the *original* pixel dimensions, which the
    index needs for aspect-ratio maths even though the pixels themselves are
    downscaled.

    ``mode`` is the Pillow mode to deliver: "RGB" for the CPU inference path
    and the thumbnailer, "RGBA" for the GPU one, which requires four channels.
    """
    register_heif()
    with Image.open(path) as im:
        width, height = im.size

        # draft() must happen before the pixels are loaded; it is a no-op on
        # formats that cannot do it.
        try:
            im.draft("RGB", (max_side, max_side))
        except (AttributeError, ValueError):
            pass

        im.load()
        # EXIF orientation: a portrait photo stored as landscape-plus-a-rotate
        # flag would otherwise be detected sideways, and no pose survives that.
        try:
            from PIL import ImageOps

            im = ImageOps.exif_transpose(im) or im
        except Exception:  # noqa: BLE001 - a broken EXIF block is not fatal
            pass

        rgb = im.convert(mode)

    # A rotate-90 EXIF flag means the stored dimensions are transposed relative
    # to how the image is meant to be seen. Downstream aspect maths works in
    # display orientation, so report the dimensions that way.
    if (width > height) != (rgb.width > rgb.height):
        width, height = height, width

    longest = max(rgb.size)
    if longest > max_side:
        ratio = max_side / longest
        rgb = rgb.resize(
            (max(1, round(rgb.width * ratio)), max(1, round(rgb.height * ratio))),
            Image.Resampling.BILINEAR,
        )
    return rgb, width, height
