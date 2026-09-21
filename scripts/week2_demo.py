"""Run interactive Week 2 desktop perception and control checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import traceback

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore, QtTest, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui_agent.capture import capture_screen, resize_image
from gui_agent.control import DesktopController
from gui_agent.perception import (
    EasyOcrRecognizer,
    detect_ui_candidates,
    draw_candidate_boxes,
    draw_text_boxes,
    map_text_elements,
)


OCR_TARGET = "GUI AGENT OCR TEST 7429"
CHINESE_TARGET = "屏幕文字识别"
CHINESE_INPUT_TARGET = "中文输入验证"


class ValidationWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GUI Agent Week 2 Validation")
        self.setFixedSize(1100, 850)
        self.setStyleSheet("background: white; color: black;")
        self.button_clicked = False

        title = QtWidgets.QLabel(OCR_TARGET)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 42px; font-weight: 700;")

        chinese = QtWidgets.QLabel(CHINESE_TARGET)
        chinese.setAlignment(QtCore.Qt.AlignCenter)
        chinese.setStyleSheet("font-size: 38px; font-weight: 600;")

        note = QtWidgets.QLabel("Known text rendered in a controlled local test window")
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setStyleSheet("font-size: 20px;")

        self.action_button = QtWidgets.QPushButton("CLICK")
        self.action_button.setFixedHeight(52)
        self.action_button.setStyleSheet("font-size: 30px; font-weight: 700;")
        self.action_button.clicked.connect(self._mark_button_clicked)

        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setText("ASCII INPUT")
        self.input_field.setFixedHeight(52)
        self.input_field.setStyleSheet("font-size: 28px; font-weight: 600;")

        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.action_button)
        action_row.addWidget(self.input_field, 1)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setFixedHeight(150)
        scroll_content = QtWidgets.QWidget()
        scroll_content.setMinimumHeight(700)
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        for index in range(12):
            item = QtWidgets.QLabel(f"Scroll item {index + 1}")
            item.setStyleSheet("font-size: 18px;")
            scroll_layout.addWidget(item)
        self.scroll_area.setWidget(scroll_content)
        self.scroll_area.setWidgetResizable(True)

        self.drag_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.drag_slider.setRange(0, 100)
        self.drag_slider.setValue(20)
        self.drag_slider.setFixedHeight(44)

        self.candidate_region = QtWidgets.QFrame()
        self.candidate_region.setFixedSize(320, 80)
        self.candidate_region.setStyleSheet(
            "background: #d9d9d9; border: 4px solid #202020;"
        )

        self.status_label = QtWidgets.QLabel("READY")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(22)
        layout.addWidget(title)
        layout.addWidget(chinese)
        layout.addWidget(note)
        layout.addLayout(action_row)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.drag_slider)
        layout.addWidget(self.candidate_region, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def _mark_button_clicked(self) -> None:
        self.button_clicked = True
        self.action_button.setText("CLICKED")
        self.status_label.setText("BUTTON OK")


def _capture_window_image(
    window: ValidationWindow,
) -> tuple[NDArray[np.uint8], tuple[int, int], tuple[int, int]]:
    frame = capture_screen()
    window_geometry = window.frameGeometry()
    screen_geometry = window.screen().geometry()
    scale_x = frame.region.width / screen_geometry.width()
    scale_y = frame.region.height / screen_geometry.height()
    crop_left = round((window_geometry.left() - screen_geometry.left()) * scale_x)
    crop_top = round((window_geometry.top() - screen_geometry.top()) * scale_y)
    crop_right = round(
        (window_geometry.left() + window_geometry.width() - screen_geometry.left())
        * scale_x
    )
    crop_bottom = round(
        (window_geometry.top() + window_geometry.height() - screen_geometry.top())
        * scale_y
    )
    crop_left = max(0, min(frame.region.width - 1, crop_left))
    crop_top = max(0, min(frame.region.height - 1, crop_top))
    crop_right = max(crop_left + 1, min(frame.region.width, crop_right))
    crop_bottom = max(crop_top + 1, min(frame.region.height, crop_bottom))
    image = frame.image[crop_top:crop_bottom, crop_left:crop_right]
    return image, (crop_left, crop_top), frame.region.size


def run_ocr_validation(
    app: QtWidgets.QApplication,
    window: ValidationWindow,
) -> None:
    try:
        window_image, window_origin, _ = _capture_window_image(window)

        source_size = (window_image.shape[1], window_image.shape[0])
        processing_size = (source_size[0] // 2, source_size[1] // 2)
        processing_image = resize_image(window_image, processing_size)

        recognizer = EasyOcrRecognizer()
        processed_elements = recognizer.recognize(processing_image, min_confidence=0.2)
        original_elements = map_text_elements(
            processed_elements,
            source_size=processing_size,
            target_size=source_size,
        )
        matched = next(
            (element for element in original_elements if "7429" in element.text),
            None,
        )
        if matched is None:
            raise RuntimeError(f"OCR did not find the known target: {OCR_TARGET}")

        annotated = draw_text_boxes(window_image, original_elements)
        left, top, right, bottom = matched.box
        margin = 50
        result_left = max(0, left - margin)
        result_top = max(0, top - margin)
        result_right = min(source_size[0], right + margin)
        result_bottom = min(source_size[1], bottom + margin)
        result_crop = annotated[result_top:result_bottom, result_left:result_right]
        if result_crop.size == 0:
            raise RuntimeError("OCR validation produced an empty result crop")

        output_path = Path(tempfile.gettempdir()) / "gui_agent_week2_ocr.png"
        if not cv2.imwrite(str(output_path), result_crop):
            raise RuntimeError(f"Failed to save OCR validation crop: {output_path}")

        chinese_found = any(CHINESE_TARGET in element.text for element in original_elements)
        print(f"source_size={source_size}")
        print(f"processing_size={processing_size}")
        print(f"window_origin={window_origin}")
        print(f"detections={len(original_elements)}")
        print(f"matched_text={matched.text!r}")
        print(f"matched_box={matched.box}")
        print(f"chinese_target_found={chinese_found}")
        print(f"annotated_crop={output_path}")
        print("ocr_desktop_validation=PASS")
        app.exit(0)
    except Exception:
        traceback.print_exc()
        app.exit(1)


def run_integration_validation(
    app: QtWidgets.QApplication,
    window: ValidationWindow,
) -> None:
    try:
        window_image, window_origin, _ = _capture_window_image(window)
        source_size = (window_image.shape[1], window_image.shape[0])
        processing_size = (source_size[0] * 3 // 4, source_size[1] * 3 // 4)
        processing_image = resize_image(window_image, processing_size)

        recognizer = EasyOcrRecognizer()
        processed_elements = recognizer.recognize(processing_image, min_confidence=0.2)
        elements = map_text_elements(processed_elements, processing_size, source_size)
        button_element = next(
            (element for element in elements if "CLICK" in element.text.upper()),
            None,
        )
        input_element = next(
            (element for element in elements if "ASCII" in element.text.upper()),
            None,
        )
        if button_element is None or input_element is None:
            recognized_texts = [element.text for element in elements]
            raise RuntimeError(
                "OCR did not locate both the button and input targets; "
                f"recognized={recognized_texts!r}"
            )

        annotated = draw_text_boxes(window_image, elements)
        output_path = Path(tempfile.gettempdir()) / "gui_agent_week2_integration.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to save integration annotation: {output_path}")

        controller = DesktopController()
        button_point = (
            window_origin[0] + button_element.center[0],
            window_origin[1] + button_element.center[1],
        )
        controller.click(button_point, duration=0.1)
        QtTest.QTest.qWait(250)
        app.processEvents()
        if not window.button_clicked:
            raise RuntimeError("OCR-based button click did not update the window state")

        input_point = (
            window_origin[0] + input_element.center[0],
            window_origin[1] + input_element.center[1],
        )
        expected_text = "Integrated GUI Agent 456"
        window.input_field.clear()
        controller.click(input_point, duration=0.1)
        controller.type_text(expected_text, interval=0.02)
        QtTest.QTest.qWait(300)
        app.processEvents()
        if window.input_field.text() != expected_text:
            raise RuntimeError(
                f"OCR-based text input mismatch: {window.input_field.text()!r}"
            )

        window.status_label.setText("INTEGRATION PASS")
        print(f"source_size={source_size}, processing_size={processing_size}")
        print(f"button_text={button_element.text!r}, screen_point={button_point}")
        print(f"input_text={input_element.text!r}, screen_point={input_point}")
        print(f"result_text={window.input_field.text()!r}")
        print(f"annotated_window={output_path}")
        print("integration_validation=PASS")
        app.exit(0)
    except Exception:
        traceback.print_exc()
        app.exit(1)


def _physical_point(
    window: ValidationWindow,
    widget: QtWidgets.QWidget,
    local_point: QtCore.QPoint,
    physical_size: tuple[int, int],
) -> tuple[int, int]:
    logical_point = widget.mapToGlobal(local_point)
    screen_geometry = window.screen().geometry()
    scale_x = physical_size[0] / screen_geometry.width()
    scale_y = physical_size[1] / screen_geometry.height()
    return (
        round((logical_point.x() - screen_geometry.left()) * scale_x),
        round((logical_point.y() - screen_geometry.top()) * scale_y),
    )


def run_control_validation(
    app: QtWidgets.QApplication,
    window: ValidationWindow,
) -> None:
    try:
        physical_size = capture_screen().region.size
        controller = DesktopController()

        button_point = _physical_point(
            window,
            window.action_button,
            window.action_button.rect().center(),
            physical_size,
        )
        controller.click(button_point, duration=0.1)
        QtTest.QTest.qWait(250)
        app.processEvents()
        if not window.button_clicked:
            raise RuntimeError("Click did not update the test button")

        input_point = _physical_point(
            window,
            window.input_field,
            window.input_field.rect().center(),
            physical_size,
        )
        expected_text = "GUI Agent Control 123"
        window.input_field.clear()
        controller.click(input_point, duration=0.1)
        controller.type_text(expected_text, interval=0.02)
        QtTest.QTest.qWait(250)
        app.processEvents()
        if window.input_field.text() != expected_text:
            raise RuntimeError(
                f"Text input mismatch: {window.input_field.text()!r}"
            )

        scroll_bar = window.scroll_area.verticalScrollBar()
        scroll_before = scroll_bar.value()
        scroll_point = _physical_point(
            window,
            window.scroll_area.viewport(),
            window.scroll_area.viewport().rect().center(),
            physical_size,
        )
        controller.scroll(-5, point=scroll_point)
        QtTest.QTest.qWait(300)
        app.processEvents()
        scroll_after = scroll_bar.value()
        if scroll_after == scroll_before:
            raise RuntimeError("Scroll did not change the test scroll area")

        option = QtWidgets.QStyleOptionSlider()
        window.drag_slider.initStyleOption(option)
        handle = window.drag_slider.style().subControlRect(
            QtWidgets.QStyle.CC_Slider,
            option,
            QtWidgets.QStyle.SC_SliderHandle,
            window.drag_slider,
        )
        start_point = _physical_point(
            window,
            window.drag_slider,
            handle.center(),
            physical_size,
        )
        end_point = _physical_point(
            window,
            window.drag_slider,
            QtCore.QPoint(window.drag_slider.width() - 24, handle.center().y()),
            physical_size,
        )
        slider_before = window.drag_slider.value()
        controller.drag(start_point, end_point, duration=0.6)
        QtTest.QTest.qWait(300)
        app.processEvents()
        slider_after = window.drag_slider.value()
        if slider_after <= slider_before:
            raise RuntimeError("Drag did not move the test slider")

        window.status_label.setText("CONTROL VALIDATION PASS")
        print(f"physical_screen={physical_size}")
        print(f"click_point={button_point}, clicked={window.button_clicked}")
        print(f"input_point={input_point}, input_text={window.input_field.text()!r}")
        print(f"scroll_point={scroll_point}, scroll={scroll_before}->{scroll_after}")
        print(f"drag={start_point}->{end_point}, slider={slider_before}->{slider_after}")
        print("control_desktop_validation=PASS")
        app.exit(0)
    except Exception:
        traceback.print_exc()
        app.exit(1)


def run_chinese_input_validation(
    app: QtWidgets.QApplication,
    window: ValidationWindow,
) -> None:
    try:
        physical_size = capture_screen().region.size
        input_point = _physical_point(
            window,
            window.input_field,
            window.input_field.rect().center(),
            physical_size,
        )

        window.input_field.clear()
        controller = DesktopController()
        controller.click(input_point, duration=0.1)
        controller.paste_text(CHINESE_INPUT_TARGET)
        QtTest.QTest.qWait(300)
        app.processEvents()
        if window.input_field.text() != CHINESE_INPUT_TARGET:
            raise RuntimeError(
                f"Chinese input mismatch: {window.input_field.text()!r}"
            )

        window.status_label.setText("CHINESE INPUT PASS")
        print(f"input_point={input_point}")
        print(f"result_text={window.input_field.text()!r}")
        print("chinese_input_desktop_validation=PASS")
        app.exit(0)
    except Exception:
        traceback.print_exc()
        app.exit(1)


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if left > right or top > bottom:
        return 0.0
    intersection = (right - left + 1) * (bottom - top + 1)
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    second_area = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    return intersection / (first_area + second_area - intersection)


def run_candidate_validation(
    app: QtWidgets.QApplication,
    window: ValidationWindow,
) -> None:
    try:
        window_image, window_origin, physical_size = _capture_window_image(window)
        boxes = detect_ui_candidates(window_image)

        target_top_left = _physical_point(
            window,
            window.candidate_region,
            QtCore.QPoint(0, 0),
            physical_size,
        )
        target_bottom_right = _physical_point(
            window,
            window.candidate_region,
            QtCore.QPoint(
                window.candidate_region.width() - 1,
                window.candidate_region.height() - 1,
            ),
            physical_size,
        )
        target_box = (
            target_top_left[0] - window_origin[0],
            target_top_left[1] - window_origin[1],
            target_bottom_right[0] - window_origin[0],
            target_bottom_right[1] - window_origin[1],
        )
        best_iou = max((_box_iou(box, target_box) for box in boxes), default=0.0)
        if best_iou < 0.5:
            raise RuntimeError(
                "Candidate detector did not locate the known rectangular region; "
                f"target={target_box!r}, boxes={boxes!r}, best_iou={best_iou:.3f}"
            )

        annotated = draw_candidate_boxes(window_image, boxes)
        output_path = Path(tempfile.gettempdir()) / "gui_agent_week2_candidates.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to save candidate annotation: {output_path}")

        print(f"target_box={target_box}")
        print(f"candidate_count={len(boxes)}")
        print(f"candidate_boxes={boxes}")
        print(f"best_target_iou={best_iou:.3f}")
        print(f"annotated_window={output_path}")
        print("candidate_desktop_validation=PASS")
        app.exit(0)
    except Exception:
        traceback.print_exc()
        app.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ocr-check",
        action="store_true",
        help="Run the OCR desktop functional validation and exit.",
    )
    parser.add_argument(
        "--control-check",
        action="store_true",
        help="Run click, input, scroll, and drag validation and exit.",
    )
    parser.add_argument(
        "--chinese-input-check",
        action="store_true",
        help="Run clipboard-based Chinese input validation and exit.",
    )
    parser.add_argument(
        "--integration-check",
        action="store_true",
        help="Run the screenshot-to-control integration validation and exit.",
    )
    parser.add_argument(
        "--candidate-check",
        action="store_true",
        help="Run rectangular UI candidate detection and bbox validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chinese_input_check:
        print(
            "WARNING: This validation will overwrite the current text clipboard, "
            "move the mouse, click the test input, and send the paste shortcut."
        )
        input("Press Enter to continue, or Ctrl+C to cancel: ")

    app = QtWidgets.QApplication([])
    window = ValidationWindow()
    window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
    window.show()
    window.raise_()
    window.activateWindow()

    if args.ocr_check:
        QtCore.QTimer.singleShot(1200, lambda: run_ocr_validation(app, window))
    elif args.control_check:
        QtCore.QTimer.singleShot(1200, lambda: run_control_validation(app, window))
    elif args.chinese_input_check:
        QtCore.QTimer.singleShot(
            1200,
            lambda: run_chinese_input_validation(app, window),
        )
    elif args.integration_check:
        QtCore.QTimer.singleShot(1200, lambda: run_integration_validation(app, window))
    elif args.candidate_check:
        QtCore.QTimer.singleShot(1200, lambda: run_candidate_validation(app, window))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
