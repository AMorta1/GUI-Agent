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


def detect_ui_candidates(
    image: NDArray[np.uint8],
    *,
    min_width: int = 24,
    min_height: int = 16,
    min_area: int = 400,
    max_area_ratio: float = 0.4,
    min_rectangularity: float = 0.6,
) -> list[PixelBox]:
    """Return rectangular visual candidates without assigning UI semantics."""

    width, height = _validate_image(image)
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Candidate detection requires a grayscale or BGR image")
    for value, name in (
        (min_width, "Minimum width"),
        (min_height, "Minimum height"),
        (min_area, "Minimum area"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (max_area_ratio, "Maximum area ratio"),
        (min_rectangularity, "Minimum rectangularity"),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be between 0 and 1")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = width * height
    raw_boxes: list[PixelBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        if (
            box_width < min_width
            or box_height < min_height
            or box_area < min_area
            or box_area / image_area > float(max_area_ratio)
        ):
            continue
        contour_area = cv2.contourArea(contour)
        if contour_area / box_area < float(min_rectangularity):
            continue
        raw_boxes.append((x, y, x + box_width - 1, y + box_height - 1))

    boxes: list[PixelBox] = []
    for box in sorted(raw_boxes, key=_box_area, reverse=True):
        if not any(_box_iou(box, kept) >= 0.8 for kept in boxes):
            boxes.append(box)
    return sorted(boxes, key=lambda box: (box[1], box[0], -_box_area(box)))


def draw_candidate_boxes(
    image: NDArray[np.uint8],
    boxes: Sequence[PixelBox],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Return a copy of a BGR image with visual candidate boxes."""

    width, height = _validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Box drawing requires a BGR image")
    if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness <= 0:
        raise ValueError("Box thickness must be a positive integer")

    annotated = image.copy()
    for box in boxes:
        if len(box) != 4 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in box
        ):
            raise TypeError("Candidate box must contain four integer coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right < width and 0 <= top < bottom < height):
            raise ValueError("Candidate box must lie within the image")
        cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
    return annotated


def _box_area(box: PixelBox) -> int:
    left, top, right, bottom = box
    return (right - left + 1) * (bottom - top + 1)


def _box_iou(first: PixelBox, second: PixelBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if left > right or top > bottom:
        return 0.0
    intersection = (right - left + 1) * (bottom - top + 1)
    return intersection / (_box_area(first) + _box_area(second) - intersection)


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
