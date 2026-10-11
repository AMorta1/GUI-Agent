"""Offline CPU OCR and saved-observation grounding benchmark; no GUI or API calls."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import psutil

from gui_agent.capture import ScreenRegion
from gui_agent.grounding import build_observation, resolve_target
from gui_agent.perception import EasyOcrRecognizer, TextElement


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values, unit):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        left = math.floor(position)
        right = math.ceil(position)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"count": len(values), "unit": unit, "mean": statistics.fmean(values),
            "p50": percentile(0.5), "p95": percentile(0.95),
            "min": ordered[0], "max": ordered[-1],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_method": "linear interpolation at (n-1)*p"}


class Resources:
    """Process-only samples, including warmup and bookkeeping, not system peaks."""

    def __enter__(self):
        self.process = psutil.Process()
        self.start = time.perf_counter()
        self.cpu_start = self.process.cpu_times()
        self.rss = [self.process.memory_info().rss]
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.1):
            try:
                self.rss.append(self.process.memory_info().rss)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.rss.append(self.process.memory_info().rss)
        cpu_end = self.process.cpu_times()
        self.result = {
            "scope": "whole phase including warmup, logging and resource sampler",
            "wall_seconds": time.perf_counter() - self.start,
            "process_cpu_seconds": cpu_end.user + cpu_end.system
                                   - self.cpu_start.user - self.cpu_start.system,
            "rss_start_mib": self.rss[0] / 1024**2,
            "rss_end_mib": self.rss[-1] / 1024**2,
            "rss_sampled_peak_mib": max(self.rss) / 1024**2,
            "rss_samples": len(self.rss), "sampling_interval_seconds": 0.1,
            "sampling_errors": self.errors,
        }


def load_case(label, summary_path, observation_id, target_text):
    summary_path = Path(summary_path).resolve()
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = document.get("runtime", document)
    matches = [event for event in runtime["events"]
               if event.get("kind") == "observation"
               and event.get("observation_id") == observation_id]
    if len(matches) != 1:
        raise ValueError("Saved observation must be unique")
    event = matches[0]
    context = event["context"]
    if context["observation_id"] != observation_id:
        raise ValueError("Context identity does not match saved observation")
    image_path = Path(event["image_path"]).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or tuple(context["image_size"]) != (image.shape[1], image.shape[0]):
        raise ValueError("Saved screenshot must match the observation dimensions")
    elements = []
    for index, item in enumerate(context["elements"], 1):
        if item["target_id"] != f"{observation_id}:text-{index}":
            raise ValueError("Saved target IDs must be complete and ordered")
        box = item["bbox"]
        if isinstance(box, str):
            box = [int(part) for part in box.split()]
        elements.append(TextElement(item["text"], item["confidence"], tuple(box)))
    observation = build_observation(
        observation_id, image_path, elements,
        screen_region=ScreenRegion(**event["screen_region"]),
        screen_size=tuple(event["screen_size"]), crop_origin=tuple(event["crop_origin"]),
        image_size=tuple(context["image_size"]), allowed_box=tuple(context["allowed_box"]),
    )
    targets = [target for target in observation.targets if target.element.text == target_text]
    if len(targets) != 1:
        raise ValueError("Benchmark target text must match exactly one saved element")
    target_id = targets[0].target_id
    point = resolve_target(observation, target_id, target_text)
    metadata = {"label": label, "summary_path": str(summary_path),
                "summary_sha256": digest(summary_path), "observation_id": observation_id,
                "image_path": str(image_path), "image_sha256": digest(image_path),
                "image_size": list(context["image_size"]), "image_format": "BGR uint8",
                "saved_ocr_elements": len(elements), "target_id": target_id,
                "target_text": target_text, "grounded_point": list(point),
                "point_used_for_execution": False, "saved_ocr": context["elements"]}
    return {"image": image, "observation": observation, "metadata": metadata}


def create_reader(model_dir):
    model_dir = Path(model_dir).resolve()
    for filename in ("craft_mlt_25k.pth", "zh_sim_g2.pth"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Cached OCR weights missing: {filename}; downloads forbidden")
    import easyocr

    return easyocr.Reader(["ch_sim", "en"], gpu=False, download_enabled=False,
                          model_storage_directory=str(model_dir), verbose=False)


def measure(operation, *, warmup, repeats, batch_size, unit, emit, describe):
    if warmup < 0 or repeats <= 0 or batch_size <= 0:
        raise ValueError("Invalid benchmark counts")
    samples = []
    for phase, count in (("warmup", warmup), ("measurement", repeats)):
        for index in range(count):
            started = time.perf_counter_ns()
            try:
                for _ in range(batch_size):
                    result = operation()
            except Exception as exc:
                emit({"phase": phase, "index": index + 1, "status": "error",
                      "elapsed_ns": time.perf_counter_ns() - started,
                      "error": f"{type(exc).__name__}: {exc}"})
                raise
            elapsed = time.perf_counter_ns() - started
            record = {"phase": phase, "index": index + 1, "status": "ok",
                      "elapsed_ns": elapsed, "batch_size": batch_size}
            try:
                record["result"] = describe(result)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                emit(record)
                raise
            emit(record)
            if phase == "measurement":
                samples.append(elapsed / batch_size / (1_000_000 if unit == "ms" else 1000))
    return summarize(samples, unit)


def environment():
    cpu_name = platform.processor()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable, "os": platform.platform(),
            "cpu": cpu_name, "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(), "ram_gib": psutil.virtual_memory().total / 1024**3,
            "packages": {name: version(name) for name in
                         ("easyocr", "torch", "opencv-python", "numpy", "psutil")},
            "timer": vars(time.get_clock_info("perf_counter")),
            "gpu_used": False, "api_calls": 0, "desktop_actions": 0,
            "desktop_captures": 0, "downloads_enabled": False}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=["offline"])
    cli.add_argument("--case", action="append", nargs=4, required=True,
                     metavar=("LABEL", "SUMMARY", "OBSERVATION_ID", "TARGET_TEXT"))
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--ocr-warmup", type=int, default=2)
    cli.add_argument("--ocr-repeat", type=int, default=20)
    cli.add_argument("--grounding-warmup", type=int, default=3)
    cli.add_argument("--grounding-batches", type=int, default=30)
    cli.add_argument("--grounding-batch-size", type=int, default=1000)
    return cli


def run(args):
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError("Evidence directory must be outside the repository")
    if len(args.case) != 2 or len({item[0] for item in args.case}) != 2:
        raise ValueError("Exactly two distinctly named saved inputs are required")
    if (args.ocr_warmup, args.ocr_repeat, args.grounding_warmup,
            args.grounding_batches, args.grounding_batch_size) != (2, 20, 3, 30, 1000):
        raise ValueError("Only the approved 2/20 OCR and 3/30/1000 grounding protocol is supported")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "command": [sys.executable, *sys.argv],
              "protocol": {"ocr_warmup": 2, "ocr_repeat": 20,
              "grounding_warmup_batches": 3, "grounding_measurement_batches": 30,
              "grounding_batch_size": 1000, "ocr_languages": ["ch_sim", "en"],
              "ocr_gpu": False, "ocr_min_confidence": 0.0,
              "grounding_min_confidence": 0.5,
              "ocr_timed_scope": "recognize including readtext and normalization; no image I/O or reader load",
              "grounding_timed_scope": "resolve_target on saved observation; no JSON parsing or observation build",
              "measurement_order": "OCR qt/search, then grounding qt/search",
              "resource_scope": "each phase including warmup and logging; 100ms process RSS sampling"},
              "inputs": [], "results": [], "errors": [],
              "limitations": ["Two fixed images, not general OCR accuracy or throughput",
                               "Grounding percentiles describe batch-average per-call cost",
                               "RSS is sampled process memory, not an exact system peak",
                               "No runtime, API, Qwen or desktop capture latency measurement"]}
    try:
        with (output / "benchmark-console.txt").open("x", encoding="utf-8") as console, \
                redirect_stdout(console), redirect_stderr(console), \
                (output / "samples.jsonl").open("x", encoding="utf-8") as raw:
            report["environment"] = environment()
            report["source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in
                                      (Path(__file__), ROOT / "gui_agent/perception.py",
                                       ROOT / "gui_agent/grounding.py")}
            cases = [load_case(*item) for item in args.case]
            report["inputs"] = [case["metadata"] for case in cases]
            report["weights"] = [{"path": str(args.model_dir.resolve() / filename),
                                  "sha256": digest(args.model_dir / filename)} for filename in
                                 ("craft_mlt_25k.pth", "zh_sim_g2.pth")]
            with Resources() as resources:
                started = time.perf_counter_ns()
                reader = create_reader(args.model_dir)
                initialization_ns = time.perf_counter_ns() - started
            report["ocr_initialization"] = {"elapsed_ns": initialization_ns,
                                            "resources": resources.result}
            import torch

            report["environment"].update(torch_threads=torch.get_num_threads(),
                                         torch_interop_threads=torch.get_num_interop_threads(),
                                         opencv_threads=cv2.getNumThreads())
            recognizer = EasyOcrRecognizer(reader=reader)
            for module in ("ocr", "grounding"):
                for case in cases:
                    metadata = case["metadata"]

                    def emit(record):
                        raw.write(json.dumps({"module": module, "input": metadata["label"],
                                              **record}, ensure_ascii=True, allow_nan=False) + "\n")
                        raw.flush()

                    if module == "ocr":
                        signatures = set()

                        def describe(elements):
                            if not elements:
                                raise ValueError("Real OCR returned no elements")
                            result = [asdict(element) for element in elements]
                            signatures.add(json.dumps(result, sort_keys=True))
                            return result

                        operation = lambda: recognizer.recognize(case["image"], min_confidence=0.0)
                        warmup, repeats, batch_size, unit = 2, 20, 1, "ms"
                    else:
                        operation = lambda: resolve_target(case["observation"],
                                                           metadata["target_id"], metadata["target_text"])

                        def describe(point):
                            if list(point) != metadata["grounded_point"]:
                                raise ValueError("Grounding result changed")
                            return {"point": point, "used_for_execution": False}

                        warmup, repeats, batch_size, unit = 3, 30, 1000, "us"
                    with Resources() as resources:
                        stats = measure(operation, warmup=warmup, repeats=repeats,
                                        batch_size=batch_size, unit=unit, emit=emit, describe=describe)
                    result = {"module": module, "input": metadata["label"],
                              "statistics": stats, "resources": resources.result}
                    if module == "ocr":
                        result["distinct_output_signatures_including_warmup"] = len(signatures)
                    report["results"].append(result)
                    if resources.result["sampling_errors"]:
                        raise RuntimeError("Resource sampler failed; see recorded errors")
            for item in report["inputs"]:
                if digest(item["image_path"]) != item["image_sha256"] or \
                        digest(item["summary_path"]) != item["summary_sha256"]:
                    raise ValueError("Historical input changed during measurement")
            for item in report["weights"]:
                if digest(item["path"]) != item["sha256"]:
                    raise ValueError("Cached model weights changed during measurement")
            report["historical_inputs_and_weights_unchanged"] = True
            report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        (output / "performance-summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(f"performance_status={report['status']} results={len(report['results'])} errors={len(report['errors'])}")
    for result in report["results"]:
        stats = result["statistics"]
        print(f"{result['module']} {result['input']}: n={stats['count']} "
              f"mean={stats['mean']:.3f} p50={stats['p50']:.3f} p95={stats['p95']:.3f} {stats['unit']}")
    print(f"evidence={output}")
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
