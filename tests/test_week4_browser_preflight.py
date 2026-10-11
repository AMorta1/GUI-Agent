"""Pure offline checks; no dependency on external scripts or archived evidence."""

from dataclasses import replace

import pytest

from gui_agent.capture import ScreenRegion
from gui_agent.grounding import build_observation
from gui_agent.perception import TextElement
from scripts import week4_browser_preflight as checks


def lines():
    return [TextElement("Week4", 0.9, (30, 10, 70, 26)),
            TextElement("Browser", 0.9, (20, 28, 80, 44)),
            TextElement("Test", 0.9, (35, 46, 65, 62))]


def observation(elements):
    return build_observation("offline-lines", "offline.png", elements,
                             screen_region=ScreenRegion(0, 0, 300, 300), screen_size=(300, 300))


def test_three_lines_use_real_ids_and_grounding_only(monkeypatch):
    scene = observation(lines())
    calls = []
    original = checks.resolve_target

    def resolve(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(checks, "resolve_target", resolve)
    record = {}
    checks.check_target(scene, record)
    assert [c[1] for c in calls] == [target.target_id for target in scene.targets]
    assert record["shortcut_check"]["match_count"] == 1
    assert record["grounding_check"]["passed"] is True
    assert record["grounding_check"]["point_used_for_execution"] is False
    assert record["shortcut_check"]["agent_shortcut_selection_or_launch_verified"] is False
    assert "target_id" not in record["grounding_check"] and "point" not in record["grounding_check"]


@pytest.mark.parametrize("fault", ["misaligned", "large_gap", "overlap", "intervening", "missing", "wrong_order", "wrong_word"])
def test_invalid_multiline_geometry_or_text_is_rejected(fault):
    base = lines()
    if fault == "misaligned":
        base[1] = replace(base[1], box=(60, 28, 120, 44))
    elif fault == "large_gap":
        base[2] = replace(base[2], box=(35, 80, 65, 96))
    elif fault == "overlap":
        base[1] = replace(base[1], box=(20, 20, 80, 36))
    elif fault == "intervening":
        base.append(TextElement("unrelated", 0.9, (30, 26, 70, 28)))
    elif fault == "missing":
        base.pop()
    elif fault == "wrong_order":
        base[0], base[2] = replace(base[0], text="Test"), replace(base[2], text="Week4")
    else:
        base[2] = replace(base[2], text="Tes")
    with pytest.raises(ValueError, match="found 0"):
        checks.check_target(observation(base), {})


@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_line_requires_existing_confidence_threshold(index):
    base = lines()
    base[index] = replace(base[index], confidence=0.49)
    record = {}
    with pytest.raises(ValueError, match="confidence"):
        checks.check_target(observation(base), record)
    assert record["shortcut_check"]["confidence_passed"] is False
    assert record["grounding_check"]["passed"] is False


def test_existing_threshold_equality_and_single_line_supported():
    checks.check_target(observation([replace(e, confidence=0.5) for e in lines()]), {})
    record = {}
    checks.check_target(observation([TextElement("Week4 Browser Test", 0.9, (10, 10, 180, 25))]), record)
    assert record["grounding_check"]["passed"] is True


@pytest.mark.parametrize("confidence", [0.9, 0.1])
def test_duplicate_groups_are_rejected_even_at_low_confidence(confidence):
    base = lines()
    duplicates = [replace(e, confidence=confidence, box=(e.box[0] + 120, e.box[1], e.box[2] + 120, e.box[3])) for e in base]
    record = {}
    with pytest.raises(ValueError, match="found 2"):
        checks.check_target(observation(base + duplicates), record)
    assert record["grounding_check"]["checked"] is False


def test_existing_grounding_rejects_duplicate_word_outside_group():
    record = {}
    with pytest.raises(ValueError, match="target must be unique"):
        checks.check_target(observation(lines() + [TextElement("Week4", 0.9, (150, 150, 190, 166))]), record)
    assert record["shortcut_check"]["match_count"] == 1
    assert record["grounding_check"]["passed"] is False


def test_expected_label_is_explicit_and_not_task_special_cased():
    record = {}
    checks.check_target(observation([TextElement("Other", 0.9, (30, 10, 70, 26)),
                                     TextElement("Tool", 0.9, (30, 28, 70, 44))]), record, label="Other Tool")
    assert record["shortcut_check"]["required_text"] == "Other Tool"
    assert record["grounding_check"]["passed"] is True
