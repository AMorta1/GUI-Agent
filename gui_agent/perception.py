"""OCR text recognition, coordinate mapping, and bounding-box drawing."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from numbers import Real
from typing import Any, Protocol, Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .capture import Size, map_box


PixelBox: TypeAlias = tuple[int, int, int, int]


class OcrReader(Protocol):
    def readtext(self, image: NDArray[np.uint8], detail: int) -> list[Any]: ...


@dataclass(frozen=True)
class TextElement:
    """Recognized text and its axis-aligned image coordinates."""

    text: str
    confidence: float
    box: PixelBox

    def __post_init__(self) -> None:
        left, top, right, bottom = self.box
        if not self.text:
            raise ValueError("Text element text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Text confidence must be between 0 and 1")
        if left >= right or top >= bottom:
            raise ValueError("Text element box must have positive width and height")

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.box
        return (left + right) // 2, (top + bottom) // 2


class EasyOcrRecognizer:
    """Small EasyOCR adapter that returns project-native text elements."""

    def __init__(
        self,
        languages: Sequence[str] = ("ch_sim", "en"),
        *,
        gpu: bool = True,
        reader: OcrReader | None = None,
    ) -> None:
        if not languages:
            raise ValueError("At least one OCR language is required")
        if reader is None:
            import easyocr

            reader = easyocr.Reader(list(languages), gpu=gpu)
        self._reader = reader

    def recognize(
        self,
        image: NDArray[np.uint8],
        *,
        min_confidence: float = 0.0,
    ) -> list[TextElement]:
        width, height = _validate_image(image)
        raw_results = self._reader.readtext(image, detail=1)
        return normalize_ocr_results(
            raw_results,
            image_size=(width, height),
            min_confidence=min_confidence,
        )


def normalize_ocr_results(
    raw_results: Sequence[Any],
    *,
    image_size: Size,
    min_confidence: float = 0.0,
) -> list[TextElement]:
    """Convert EasyOCR output into clipped axis-aligned text elements."""

    width, height = _validate_size(image_size)
    if not isinstance(min_confidence, Real) or isinstance(min_confidence, bool):
        raise TypeError("Minimum confidence must be numeric")
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("Minimum confidence must be between 0 and 1")

    elements: list[TextElement] = []
    for result in raw_results:
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise ValueError("OCR result must contain points, text, and confidence")
        points, raw_text, raw_confidence = result
        text = str(raw_text).strip()
        confidence = float(raw_confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("OCR confidence must be between 0 and 1")
        if not text or confidence < float(min_confidence):
            continue

        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] != 2:
            raise ValueError("OCR points must have shape (n, 2)")
        if not np.isfinite(coordinates).all():
            raise ValueError("OCR points must be finite")

        left = max(0, min(width - 1, floor(float(coordinates[:, 0].min()))))
        top = max(0, min(height - 1, floor(float(coordinates[:, 1].min()))))
        right = max(0, min(width - 1, ceil(float(coordinates[:, 0].max()))))
        bottom = max(0, min(height - 1, ceil(float(coordinates[:, 1].max()))))
        if left >= right or top >= bottom:
            continue
        elements.append(TextElement(text=text, confidence=confidence, box=(left, top, right, bottom)))

    return elements


def map_text_elements(
    elements: Sequence[TextElement],
    source_size: Size,
    target_size: Size,
) -> list[TextElement]:
    """Map recognized elements between processing and original resolutions."""

    return [
        TextElement(
            text=element.text,
            confidence=element.confidence,
            box=map_box(element.box, source_size, target_size),
        )
        for element in elements
    ]


def draw_text_boxes(
    image: NDArray[np.uint8],
    elements: Sequence[TextElement],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Return a copy of a BGR image with text bounding boxes."""

    width, height = _validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Box drawing requires a BGR image")
    if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness <= 0:
        raise ValueError("Box thickness must be a positive integer")

    annotated = image.copy()
    for element in elements:
        left, top, right, bottom = element.box
        if not (0 <= left < right < width and 0 <= top < bottom < height):
            raise ValueError("Text element box must lie within the image")
        cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
    return annotated


def _validate_image(image: NDArray[np.uint8]) -> Size:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Image must be a non-empty 2D or 3D array")
    height, width = image.shape[:2]
    return width, height


def _validate_size(size: Size) -> Size:
    if len(size) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) for value in size
    ):
        raise TypeError("Image size must contain integer width and height")
    width, height = size
    if width <= 1 or height <= 1:
        raise ValueError("Image width and height must be greater than one")
    return width, height
