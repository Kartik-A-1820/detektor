"""Upload validation: lenient content types, strict content, decompression-bomb guard."""

from __future__ import annotations

import io
import struct
import unittest
import zlib

import numpy as np
from PIL import Image

from api.validation import (
    MAX_PIXELS,
    ImageValidationError,
    peek_image_size,
    validate_mime_type,
    validate_uploaded_image,
)


def _png(width: int = 64, height: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), 120, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_huge_png(width: int, height: int) -> bytes:
    """A syntactically valid PNG header declaring a huge canvas with almost no pixel data."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")


class MimeTypeTests(unittest.TestCase):
    def test_missing_and_generic_types_are_accepted(self) -> None:
        for value in (None, "", "application/octet-stream", "APPLICATION/OCTET-STREAM; x=y"):
            validate_mime_type(value)

    def test_known_types_with_parameters_are_accepted(self) -> None:
        validate_mime_type("image/png")
        validate_mime_type("image/jpeg; charset=binary")

    def test_explicit_wrong_type_is_rejected(self) -> None:
        for value in ("text/plain", "application/pdf", "image/svg+xml"):
            with self.assertRaises(ImageValidationError):
                validate_mime_type(value)


class UploadedImageTests(unittest.TestCase):
    def test_valid_png_without_content_type(self) -> None:
        self.assertEqual(validate_uploaded_image(_png(64, 48), None), (64, 48))

    def test_garbage_bytes_rejected_even_with_octet_stream(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_uploaded_image(b"definitely not an image", "application/octet-stream")

    def test_empty_and_oversized_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_uploaded_image(b"", "image/png")
        with self.assertRaises(ImageValidationError):
            validate_uploaded_image(_png(), "image/png", max_file_size=10)

    def test_too_small_rejected(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_uploaded_image(_png(8, 8), "image/png")

    def test_peek_reads_header_only(self) -> None:
        self.assertEqual(peek_image_size(_png(30, 20)), (30, 20))
        self.assertIsNone(peek_image_size(b"nope"))

    def test_decompression_bomb_is_rejected_before_decode(self) -> None:
        side = int(MAX_PIXELS**0.5) + 100
        bomb = _fake_huge_png(side, side)
        self.assertLess(len(bomb), 1024)
        with self.assertRaises(ImageValidationError) as ctx:
            validate_uploaded_image(bomb, "image/png")
        self.assertIn("pixel limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
