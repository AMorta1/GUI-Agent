from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pydantic import ValidationError
import pytest

from gui_agent.datasets import (
    CanonicalAction,
    CanonicalRecord,
    Observation,
    SourceInfo,
    TaskInfo,
    TrajectoryStep,
    convert_mind2web,
    convert_screenagent,
    convert_webarena,
)


REVISION = "fixed-revision"
ACQUIRED_AT = "2026-09-27"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _screenagent_record(actions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "task_prompt_en": "Open the settings page",
        "session_id": "session-1",
        "saved_image_name": "screen.jpg",
        "video_width": 10,
        "video_height": 8,
        "actions": actions,
    }


def _write_screenagent_fixture(
    root: Path,
    payload: dict[str, object],
    *,
    include_image: bool = True,
) -> Path:
    session = root / "session-1"
    (session / "images").mkdir(parents=True)
    if include_image:
        Image.new("RGB", (10, 8), "white").save(session / "images" / "screen.jpg")
    (session / "001_translate.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return root


def test_schema_requires_step_for_trajectory_record() -> None:
    with pytest.raises(ValidationError, match="require a step"):
        CanonicalRecord(
            record_kind="trajectory_step",
            source=SourceInfo(
                dataset="source",
                split="train",
                record_id="1",
                revision=REVISION,
                license="MIT",
            ),
            task=TaskInfo(task_id="task", instruction="Do something"),
            step=None,
            training_eligible=True,
        )


def test_observation_requires_complete_image_metadata() -> None:
    with pytest.raises(ValidationError, match="provided together"):
        Observation(image_path="screen.jpg", image_width=10)


def test_screenagent_converts_actions_and_original_pixel_point(tmp_path: Path) -> None:
    source = _write_screenagent_fixture(
        tmp_path / "source",
        _screenagent_record(
            [
                {
                    "action_type": "MouseAction",
                    "mouse_action_type": "click",
                    "mouse_position": {"width": 7, "height": 3},
                },
                {"action_type": "FutureAction", "value": "kept in raw action"},
            ]
        ),
    )

    manifest = convert_screenagent(
        source,
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )
    rows = _read_jsonl(tmp_path / "output" / "records.jsonl")

    assert manifest.records_written == 2
    assert manifest.records_skipped == 0
    assert rows[0]["step"]["action"]["point"] == [7.0, 3.0]
    assert rows[0]["step"]["observation"]["image_width"] == 10
    assert rows[1]["step"]["action"]["type"] == "other"
    assert rows[1]["step"]["action"]["raw_action"]["value"] == "kept in raw action"


def test_screenagent_records_missing_image_error(tmp_path: Path) -> None:
    source = _write_screenagent_fixture(
        tmp_path / "source",
        _screenagent_record([{"action_type": "PlanAction", "element": "Plan"}]),
        include_image=False,
    )

    manifest = convert_screenagent(
        source,
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )

    assert manifest.records_written == 0
    assert manifest.records_skipped == 1
    errors = _read_jsonl(tmp_path / "output" / "errors.jsonl")
    assert errors[0]["error_type"] == "FileNotFoundError"


def test_screenagent_records_corrupt_json_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "001_translate.json").write_text("{not json", encoding="utf-8")

    manifest = convert_screenagent(
        source,
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )

    assert manifest.records_skipped == 1
    assert _read_jsonl(tmp_path / "output" / "errors.jsonl")[0][
        "error_type"
    ] == "JSONDecodeError"


def test_webarena_creates_non_training_task_without_action(tmp_path: Path) -> None:
    source = tmp_path / "test.raw.json"
    source.write_text(
        json.dumps(
            [
                {
                    "task_id": 7,
                    "intent": "Find the latest order",
                    "sites": ["shopping_admin"],
                    "start_url": "http://example.test/admin",
                    "eval": {"eval_types": ["string_match"]},
                }
            ]
        ),
        encoding="utf-8",
    )

    manifest = convert_webarena(
        source,
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )
    row = _read_jsonl(tmp_path / "output" / "records.jsonl")[0]

    assert manifest.records_written == 1
    assert row["record_kind"] == "task"
    assert row["step"] is None
    assert row["training_eligible"] is False
    assert row["task"]["application"] == "shopping_admin"


def test_webarena_empty_instruction_is_not_silently_dropped(tmp_path: Path) -> None:
    source = tmp_path / "test.raw.json"
    source.write_text(
        json.dumps([{"task_id": 1, "intent": "", "sites": ["reddit"]}]),
        encoding="utf-8",
    )

    manifest = convert_webarena(
        source,
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )

    assert manifest.records_written == 0
    assert manifest.records_skipped == 1
    assert "intent" in _read_jsonl(tmp_path / "output" / "errors.jsonl")[0]["message"]


def _mind2web_row(action_uid: str = "action-1") -> dict[str, object]:
    candidate = {
        "backend_node_id": "node-12",
        "is_original_target": True,
        "attributes": json.dumps(
            {
                "backend_node_id": "node-12",
                "bounding_box_rect": "1.5,2,3,4",
            }
        ),
    }
    return {
        "action_uid": action_uid,
        "annotation_id": "task-1",
        "confirmed_task": "Book a train ticket",
        "target_action_index": "2",
        "operation": json.dumps({"op": "TYPE", "value": "Boston"}),
        "screenshot": Image.new("RGB", (12, 9), "white"),
        "pos_candidates": [json.dumps(candidate)],
        "domain": "travel",
        "website": "rail",
        "cleaned_html": "<button>Book</button>",
        "target_action_reprs": "[button] Book -> TYPE: Boston",
    }


def test_mind2web_converts_action_bbox_and_saves_image(tmp_path: Path) -> None:
    manifest = convert_mind2web(
        [_mind2web_row()],
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )
    row = _read_jsonl(tmp_path / "output" / "records.jsonl")[0]

    assert manifest.records_written == 1
    assert row["step"]["index"] == 2
    assert row["step"]["action"]["type"] == "type"
    assert row["step"]["action"]["bbox"] == [1.5, 2.0, 4.5, 6.0]
    assert row["step"]["action"]["element_id"] == "node-12"
    assert (tmp_path / "output" / row["step"]["observation"]["image_path"]).is_file()


def test_mind2web_unknown_action_is_preserved_as_other(tmp_path: Path) -> None:
    row = _mind2web_row()
    row["operation"] = json.dumps({"op": "HOVER", "value": ""})

    convert_mind2web(
        [row],
        tmp_path / "output",
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )
    result = _read_jsonl(tmp_path / "output" / "records.jsonl")[0]

    assert result["step"]["action"]["type"] == "other"
    assert result["step"]["action"]["raw_action"]["operation"]["op"] == "HOVER"


def test_limit_does_not_consume_or_save_an_extra_mind2web_row(tmp_path: Path) -> None:
    output = tmp_path / "output"
    manifest = convert_mind2web(
        [_mind2web_row("first"), _mind2web_row("second")],
        output,
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
        limit=1,
    )

    assert manifest.source_records_seen == 1
    assert manifest.records_written == 1
    assert sorted(path.name for path in (output / "images").iterdir()) == ["first.jpg"]


def test_repeated_conversion_has_stable_jsonl_and_manifest(tmp_path: Path) -> None:
    source = _write_screenagent_fixture(
        tmp_path / "source",
        _screenagent_record([{"action_type": "PlanAction", "element": "Plan"}]),
    )
    output = tmp_path / "output"

    convert_screenagent(
        source,
        output,
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )
    first_records = (output / "records.jsonl").read_bytes()
    first_manifest = (output / "manifest.json").read_bytes()
    convert_screenagent(
        source,
        output,
        revision=REVISION,
        source_acquired_at=ACQUIRED_AT,
    )

    assert (output / "records.jsonl").read_bytes() == first_records
    assert (output / "manifest.json").read_bytes() == first_manifest


def test_jsonl_round_trips_through_canonical_schema(tmp_path: Path) -> None:
    record = CanonicalRecord(
        record_kind="trajectory_step",
        source=SourceInfo(
            dataset="sample",
            split="train",
            record_id="record-1",
            revision=REVISION,
            license="MIT",
        ),
        task=TaskInfo(task_id="task-1", instruction="Click Save"),
        step=TrajectoryStep(
            index=0,
            observation=Observation(),
            action=CanonicalAction(type="click", raw_action={"type": "click"}),
        ),
        training_eligible=True,
    )

    restored = CanonicalRecord.model_validate_json(record.model_dump_json())

    assert restored == record
