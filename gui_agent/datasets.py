"""Canonical GUI-task records and dataset conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"
ActionType = Literal[
    "click",
    "type",
    "select",
    "scroll",
    "drag",
    "key",
    "navigate",
    "stop",
    "other",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceInfo(StrictModel):
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)


class TaskInfo(StrictModel):
    task_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    domain: str | None = None
    application: str | None = None


class Observation(StrictModel):
    image_path: str | None = None
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    text: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def validate_image_dimensions(self) -> Observation:
        dimensions = (self.image_width, self.image_height)
        if (self.image_path is None) != (dimensions == (None, None)):
            raise ValueError("image path and dimensions must be provided together")
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("image width and height must be provided together")
        return self


class CanonicalAction(StrictModel):
    type: ActionType
    point: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    element_id: str | None = None
    text: str | None = None
    raw_action: Any


class TrajectoryStep(StrictModel):
    index: int = Field(ge=0)
    observation: Observation
    action: CanonicalAction


class CanonicalRecord(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    record_kind: Literal["task", "trajectory_step"]
    source: SourceInfo
    task: TaskInfo
    step: TrajectoryStep | None = None
    training_eligible: bool

    @model_validator(mode="after")
    def validate_record_kind(self) -> CanonicalRecord:
        if self.record_kind == "task" and self.step is not None:
            raise ValueError("task records cannot contain a trajectory step")
        if self.record_kind == "trajectory_step" and self.step is None:
            raise ValueError("trajectory_step records require a step")
        return self


class ConversionManifest(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    dataset: str
    split: str
    source_url: str
    revision: str
    license: str
    source_acquired_at: str
    source_records_seen: int = Field(ge=0)
    records_written: int = Field(ge=0)
    records_skipped: int = Field(ge=0)
    limit: int | None
    records_file: str = "records.jsonl"
    errors_file: str = "errors.jsonl"


@dataclass(frozen=True)
class ConversionEvent:
    source_id: str
    record: CanonicalRecord | None = None
    error_type: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if (self.record is None) == (self.error_type is None):
            raise ValueError("Conversion event must contain one record or one error")


def convert_screenagent(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    revision: str,
    source_acquired_at: str,
    limit: int | None = None,
) -> ConversionManifest:
    """Convert ScreenAgent JSON annotations into one record per action."""

    source_root = Path(source_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(f"ScreenAgent source directory not found: {source_root}")
    events = _iter_screenagent_events(source_root, revision)
    return _write_conversion(
        events,
        output_dir,
        dataset="screenagent",
        split="train",
        source_url="https://github.com/niuzaisheng/ScreenAgent",
        revision=revision,
        license_name="MIT",
        source_acquired_at=source_acquired_at,
        limit=limit,
    )


def convert_webarena(
    source: str | Path,
    output_dir: str | Path,
    *,
    revision: str,
    source_acquired_at: str,
    limit: int | None = None,
) -> ConversionManifest:
    """Convert WebArena task configurations without inventing action labels."""

    source_path = Path(source)
    if source_path.is_dir():
        source_path = source_path / "test.raw.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"WebArena task file not found: {source_path}")
    events = _iter_webarena_events(source_path, revision)
    return _write_conversion(
        events,
        output_dir,
        dataset="webarena",
        split="test",
        source_url="https://github.com/web-arena-x/webarena",
        revision=revision,
        license_name="Apache-2.0",
        source_acquired_at=source_acquired_at,
        limit=limit,
    )


def convert_mind2web(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    revision: str,
    source_acquired_at: str,
    limit: int | None = None,
) -> ConversionManifest:
    """Convert public Multimodal-Mind2Web train rows and save their images."""

    destination = Path(output_dir)
    image_dir = destination / "images"
    events = _iter_mind2web_events(rows, image_dir, revision)
    return _write_conversion(
        events,
        destination,
        dataset="mind2web",
        split="train",
        source_url="https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web",
        revision=revision,
        license_name="OpenRAIL",
        source_acquired_at=source_acquired_at,
        limit=limit,
    )


def _iter_screenagent_events(
    source_root: Path,
    revision: str,
) -> Iterator[ConversionEvent]:
    step_indexes: dict[str, int] = {}
    annotation_files = sorted(source_root.rglob("*_translate.json"))
    for annotation_file in annotation_files:
        source_name = annotation_file.relative_to(source_root).as_posix()
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("annotation root must be an object")
            instruction = _first_nonempty(
                payload.get("task_prompt_en"),
                payload.get("task_prompt"),
                payload.get("task_prompt_zh"),
            )
            if instruction is None:
                raise ValueError("task instruction is empty")
            session_id = _required_text(payload.get("session_id"), "session_id")
            image_name = _required_text(payload.get("saved_image_name"), "saved_image_name")
            image_path = annotation_file.parent / "images" / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"referenced image not found: {image_name}")
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                image.verify()
            declared_size = (payload.get("video_width"), payload.get("video_height"))
            if declared_size != (image_width, image_height):
                raise ValueError(
                    f"declared image size {declared_size} does not match "
                    f"{(image_width, image_height)}"
                )
            actions = payload.get("actions")
            if not isinstance(actions, list) or not actions:
                raise ValueError("actions must be a non-empty list")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            yield ConversionEvent(
                source_id=source_name,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            continue

        image_relative = image_path.relative_to(source_root).as_posix()
        for action in actions:
            index = step_indexes.get(session_id, 0)
            action_source_id = f"{source_name}#action-{index}"
            try:
                if not isinstance(action, dict):
                    raise TypeError("action must be an object")
                record = CanonicalRecord(
                    record_kind="trajectory_step",
                    source=SourceInfo(
                        dataset="screenagent",
                        split="train",
                        record_id=action_source_id,
                        revision=revision,
                        license="MIT",
                    ),
                    task=TaskInfo(task_id=session_id, instruction=instruction),
                    step=TrajectoryStep(
                        index=index,
                        observation=Observation(
                            image_path=image_relative,
                            image_width=image_width,
                            image_height=image_height,
                        ),
                        action=_convert_screenagent_action(action),
                    ),
                    training_eligible=True,
                )
                yield ConversionEvent(source_id=action_source_id, record=record)
            except (KeyError, TypeError, ValueError) as exc:
                yield ConversionEvent(
                    source_id=action_source_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            finally:
                step_indexes[session_id] = index + 1


def _convert_screenagent_action(action: Mapping[str, Any]) -> CanonicalAction:
    action_type = action.get("action_type")
    if action_type == "MouseAction":
        subtype = action.get("mouse_action_type")
        if subtype == "click":
            position = action.get("mouse_position")
            if not isinstance(position, Mapping):
                raise ValueError("click action is missing mouse_position")
            return CanonicalAction(
                type="click",
                point=(float(position["width"]), float(position["height"])),
                raw_action=dict(action),
            )
        if isinstance(subtype, str) and subtype.startswith("scroll"):
            return CanonicalAction(type="scroll", raw_action=dict(action))
    if action_type == "KeyboardAction":
        subtype = action.get("keyboard_action_type")
        if subtype == "text":
            return CanonicalAction(
                type="type",
                text=str(action.get("keyboard_text", "")),
                raw_action=dict(action),
            )
        if subtype == "press":
            return CanonicalAction(
                type="key",
                text=str(action.get("keyboard_key", "")),
                raw_action=dict(action),
            )
    descriptive_text = _first_nonempty(
        action.get("element"), action.get("advice"), action.get("situation")
    )
    return CanonicalAction(
        type="other",
        text=descriptive_text,
        raw_action=dict(action),
    )


def _iter_webarena_events(
    source_file: Path,
    revision: str,
) -> Iterator[ConversionEvent]:
    try:
        tasks = json.loads(source_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        yield ConversionEvent(
            source_id=source_file.name,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return
    if not isinstance(tasks, list):
        yield ConversionEvent(
            source_id=source_file.name,
            error_type="ValueError",
            message="task file root must be a list",
        )
        return

    for position, task in enumerate(tasks):
        source_id = str(task.get("task_id", position)) if isinstance(task, dict) else str(position)
        try:
            if not isinstance(task, dict):
                raise TypeError("task must be an object")
            instruction = _required_text(task.get("intent"), "intent")
            sites = task.get("sites")
            if not isinstance(sites, list) or not sites:
                raise ValueError("sites must be a non-empty list")
            application = ",".join(str(site) for site in sites)
            record = CanonicalRecord(
                record_kind="task",
                source=SourceInfo(
                    dataset="webarena",
                    split="test",
                    record_id=source_id,
                    revision=revision,
                    license="Apache-2.0",
                ),
                task=TaskInfo(
                    task_id=source_id,
                    instruction=instruction,
                    domain=str(sites[0]),
                    application=application,
                ),
                step=None,
                training_eligible=False,
            )
            yield ConversionEvent(source_id=source_id, record=record)
        except (TypeError, ValueError) as exc:
            yield ConversionEvent(
                source_id=source_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )


def _iter_mind2web_events(
    rows: Iterable[Mapping[str, Any]],
    image_dir: Path,
    revision: str,
) -> Iterator[ConversionEvent]:
    for position, row in enumerate(rows):
        source_id = str(row.get("action_uid", position))
        try:
            annotation_id = _required_text(row.get("annotation_id"), "annotation_id")
            instruction = _required_text(row.get("confirmed_task"), "confirmed_task")
            operation = _json_object(row.get("operation"), "operation")
            step_index = int(_required_text(row.get("target_action_index"), "target_action_index"))
            image = row.get("screenshot")
            if not isinstance(image, Image.Image):
                raise TypeError("screenshot must be a PIL image")
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{source_id}.jpg"
            image.convert("RGB").save(image_path, format="JPEG", quality=90)
            image_width, image_height = image.size
            candidate = _mind2web_target_candidate(row.get("pos_candidates"))
            element_id, bbox = _mind2web_element(candidate)
            raw_action = {
                "operation": operation,
                "target_action_reprs": row.get("target_action_reprs"),
                "target_candidate": candidate,
            }
            record = CanonicalRecord(
                record_kind="trajectory_step",
                source=SourceInfo(
                    dataset="mind2web",
                    split="train",
                    record_id=source_id,
                    revision=revision,
                    license="OpenRAIL",
                ),
                task=TaskInfo(
                    task_id=annotation_id,
                    instruction=instruction,
                    domain=_optional_text(row.get("domain")),
                    application=_optional_text(row.get("website")),
                ),
                step=TrajectoryStep(
                    index=step_index,
                    observation=Observation(
                        image_path=f"images/{source_id}.jpg",
                        image_width=image_width,
                        image_height=image_height,
                        text=_optional_text(row.get("cleaned_html")),
                    ),
                    action=CanonicalAction(
                        type=_mind2web_action_type(operation.get("op")),
                        bbox=bbox,
                        element_id=element_id,
                        text=_optional_text(operation.get("value")),
                        raw_action=raw_action,
                    ),
                ),
                training_eligible=True,
            )
            yield ConversionEvent(source_id=source_id, record=record)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            yield ConversionEvent(
                source_id=source_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )


def _mind2web_target_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list) or not value:
        return None
    candidates = [_json_object(item, "positive candidate") for item in value]
    return next(
        (candidate for candidate in candidates if candidate.get("is_original_target")),
        candidates[0],
    )


def _mind2web_element(
    candidate: Mapping[str, Any] | None,
) -> tuple[str | None, tuple[float, float, float, float] | None]:
    if candidate is None:
        return None, None
    element_id = _optional_text(candidate.get("backend_node_id"))
    attributes_value = candidate.get("attributes")
    attributes = (
        _json_object(attributes_value, "candidate attributes")
        if attributes_value
        else {}
    )
    element_id = _optional_text(attributes.get("backend_node_id")) or element_id
    rect = attributes.get("bounding_box_rect")
    if not isinstance(rect, str) or not rect.strip():
        return element_id, None
    values = [float(value.strip()) for value in rect.split(",")]
    if len(values) != 4:
        raise ValueError("bounding_box_rect must contain x, y, width, height")
    left, top, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("bounding_box_rect width and height must be positive")
    return element_id, (left, top, left + width, top + height)


def _mind2web_action_type(value: Any) -> ActionType:
    normalized = str(value or "").strip().upper()
    return {
        "CLICK": "click",
        "TYPE": "type",
        "SELECT": "select",
        "SCROLL": "scroll",
        "PRESS_ENTER": "key",
    }.get(normalized, "other")


def _write_conversion(
    events: Iterable[ConversionEvent],
    output_dir: str | Path,
    *,
    dataset: str,
    split: str,
    source_url: str,
    revision: str,
    license_name: str,
    source_acquired_at: str,
    limit: int | None,
) -> ConversionManifest:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "records.jsonl"
    errors_path = destination / "errors.jsonl"
    seen = written = skipped = 0

    with records_path.open("w", encoding="utf-8", newline="\n") as records_file, (
        errors_path.open("w", encoding="utf-8", newline="\n")
    ) as errors_file:
        event_iterator = iter(events)
        while limit is None or written < limit:
            try:
                event = next(event_iterator)
            except StopIteration:
                break
            seen += 1
            if event.record is not None:
                records_file.write(event.record.model_dump_json() + "\n")
                written += 1
            else:
                error = {
                    "dataset": dataset,
                    "source_id": event.source_id,
                    "error_type": event.error_type,
                    "message": event.message,
                }
                errors_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                skipped += 1

    manifest = ConversionManifest(
        dataset=dataset,
        split=split,
        source_url=source_url,
        revision=revision,
        license=license_name,
        source_acquired_at=source_acquired_at,
        source_records_seen=seen,
        records_written=written,
        records_skipped=skipped,
        limit=limit,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return value


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_nonempty(*values: Any) -> str | None:
    return next((text for value in values if (text := _optional_text(value))), None)
