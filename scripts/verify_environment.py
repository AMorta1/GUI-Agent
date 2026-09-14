"""Verify the GUI Agent development environment without moving the mouse."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time


MINIMUM_PYTHON = (3, 10)
PACKAGES = (
    "mss",
    "PyAutoGUI",
    "pynput",
    "opencv-python",
    "PyQt5",
    "PyYAML",
    "psutil",
    "pytest",
)


def package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def verify_packages() -> None:
    import cv2
    import mss
    import pyautogui
    import pynput
    from PyQt5 import QtCore

    print("\n[packages]")
    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")
    print(f"OpenCV import: {cv2.__version__}")
    print(f"Qt runtime: {QtCore.QT_VERSION_STR}")
    print(f"PyAutoGUI screen size: {pyautogui.size()}")
    print(f"pynput import: {package_version('pynput')}")
    print(f"mss import: {package_version('mss')}")


def verify_screen_capture() -> None:
    import cv2
    import mss
    import numpy as np

    with mss.MSS() as capture:
        monitors = capture.monitors
        if len(monitors) < 2:
            raise RuntimeError("No physical monitor was detected by mss")
        frame = np.asarray(capture.grab(monitors[1]))
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if bgr.size == 0:
        raise RuntimeError("mss returned an empty frame")
    print("\n[screen capture]")
    print(f"Detected physical monitors: {len(monitors) - 1}")
    print(f"Primary capture shape: {bgr.shape}")


def verify_torch(require_cuda: bool) -> None:
    import torch

    print("\n[pytorch]")
    print(f"PyTorch: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    left = torch.randn((1024, 1024), device=device)
    right = torch.randn((1024, 1024), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = left @ right
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("PyTorch matrix multiplication produced non-finite values")
    print(f"Tensor test device: {device.type}")
    print(f"Tensor test shape: {tuple(result.shape)}")
    print(f"Tensor test elapsed: {elapsed_ms:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when PyTorch cannot use CUDA.",
    )
    parser.add_argument(
        "--skip-screen",
        action="store_true",
        help="Skip the in-memory screenshot test in a headless environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info < MINIMUM_PYTHON:
        raise RuntimeError("Python 3.10 or newer is required")

    print("[system]")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    verify_packages()
    if not args.skip_screen:
        verify_screen_capture()
    verify_torch(args.require_cuda)
    print("\nEnvironment verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
