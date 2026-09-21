from __future__ import annotations

import numpy as np
import pytest

from gui_agent.perception import (
    EasyOcrRecognizer,
    TextElement,
    draw_text_boxes,
    map_text_elements,
    normalize_ocr_results,
)


class FakeReader:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.detail = None

    def readtext(self, image: np.ndarray, detail: int) -> list[object]:
        self.detail = detail
        return self.results


def test_recognizer_normalizes_and_filters_results() -> None:
    reader = FakeReader(
        [
            ([[-5, 5], [80, 5], [80, 40], [-5, 40]], " Hello ", 0.95),
            ([[10, 10], [20, 10], [20, 20], [10, 20]], "low", 0.1),
            ([[30, 10], [40, 10], [40, 20], [30, 20]], "   ", 0.9),
        ]
    )
    image = np.zeros((50, 100, 3), dtype=np.uint8)

    elements = EasyOcrRecognizer(reader=reader).recognize(image, min_confidence=0.5)

    assert reader.detail == 1
    assert elements == [TextElement(text="Hello", confidence=0.95, box=(0, 5, 80, 40))]
    assert elements[0].center == (40, 22)


def test_normalize_clips_box_to_image() -> None:
    elements = normalize_ocr_results(
        [([[-10, -5], [120, -5], [120, 60], [-10, 60]], "screen", 0.8)],
        image_size=(100, 50),
    )

    assert elements[0].box == (0, 0, 99, 49)


def test_empty_results_are_supported() -> None:
    reader = FakeReader([])
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert EasyOcrRecognizer(reader=reader).recognize(image) == []


def test_map_text_elements_to_original_resolution() -> None:
    element = TextElement("Target", 0.9, (100, 50, 500, 250))

    mapped = map_text_elements([element], (1280, 800), (2560, 1600))

    assert mapped == [TextElement("Target", 0.9, (200, 100, 1000, 500))]


def test_draw_text_boxes_returns_annotated_copy() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    element = TextElement("Target", 0.9, (20, 10, 100, 50))

    annotated = draw_text_boxes(image, [element])

    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    assert np.count_nonzero(image) == 0
    assert np.count_nonzero(annotated) > 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: normalize_ocr_results(
                [([[0, 0], [1, 1]], "x", 0.5)],
                image_size=(10, 10),
                min_confidence=1.1,
            ),
            "between 0 and 1",
        ),
        (
            lambda: normalize_ocr_results(
                [([[0, 0]], "x", 0.5)], image_size=(10, 10)
            ),
            "shape",
        ),
        (
            lambda: normalize_ocr_results(
                [([[0, 0], [5, 5]], "x", 1.2)], image_size=(10, 10)
            ),
            "between 0 and 1",
        ),
        (
            lambda: draw_text_boxes(
                np.zeros((10, 10, 3), dtype=np.uint8),
                [TextElement("x", 0.5, (1, 1, 10, 9))],
            ),
            "within",
        ),
    ],
)
def test_invalid_ocr_data_is_rejected(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
