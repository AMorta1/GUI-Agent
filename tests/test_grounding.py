from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gui_agent.capture import ScreenRegion
from gui_agent.grounding import (
    ActionDecision, FeedbackCheck, GroundingError, build_observation,
    evaluate_text_check, resolve_action, resolve_target,
)
from gui_agent.perception import TextElement, map_text_elements


def _observation(elements=None, **kwargs):
    return build_observation(
        "obs-1", Path("screen.png"),
        elements if elements is not None else [TextElement("Send", 0.95, (100, 120, 180, 160))],
        screen_region=ScreenRegion(0, 0, 1000, 800), screen_size=(1000, 800), **kwargs,
    )


def _decision(action=..., **kwargs):
    return ActionDecision.model_validate({
        "status": "action", "step_id": "step-1", "observation_id": "obs-1",
        "action": {"type": "click", "target_id": "obs-1:text-1", "target_text": "Send"} if action is ... else action,
        **kwargs,
    })


@pytest.mark.parametrize("action", [
    {"type": "click", "target_id": "obs-1:text-1", "target_text": "Send"},
    {"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": "hello", "mode": "replace"},
    {"type": "key", "key": "ctrl+a", "purpose": "Select the old input"},
    {"type": "scroll", "target_id": "obs-1:text-1", "target_text": "Send", "clicks": -10},
])
def test_action_contract_accepts_four_atomic_types(action):
    assert _decision(action).action.type == action["type"]


@pytest.mark.parametrize("action", [
    {"type": "click", "target_id": "obs-1:text-1", "target_text": "Send", "x": 140, "y": 140},
    {"type": "click", "target_id": "obs-1:text-1", "target_text": " "},
    {"type": "drag", "target_id": "obs-1:text-1", "target_text": "Send"},
    {"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": "", "mode": "append"},
    {"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": " ", "mode": "append"},
    {"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": "hello", "mode": "overwrite"},
    {"type": "key", "key": "win+r", "purpose": "Launch"},
    {"type": "key", "key": "delete", "purpose": "Delete"},
    {"type": "key", "key": "enter", "purpose": " "},
    *[{"type": "scroll", "target_id": "obs-1:text-1", "target_text": "Send", "clicks": value}
      for value in (0, 11, -11, True, 1.5, "2", float("nan"), float("inf"))],
])
def test_action_contract_rejects_illegal_parameters(action):
    with pytest.raises(ValidationError):
        _decision(action)


@pytest.mark.parametrize("changes", [
    {"action": None},
    {"reason": "Claimed safe"},
    {"status": "step_complete"},
    {"status": "blocked", "reason": "Missing target"},
    {"status": "blocked", "action": None},
    {"status": "step_complete", "action": None, "checks": []},
    {"unexpected": True},
])
def test_decision_status_fields_are_mutually_exclusive(changes):
    with pytest.raises(ValidationError):
        _decision(**changes)


def test_blocked_and_step_complete_require_reason_or_check():
    blocked = _decision(status="blocked", action=None, reason="No unique target")
    complete = _decision(status="step_complete", action=None, checks=[{
        "type": "text_present", "text": "Send", "region": (0, 0, 300, 300),
    }])
    assert blocked.reason
    assert complete.checks
    with pytest.raises(GroundingError, match="only action"):
        resolve_action(complete, _observation(), step_id="step-1")


@pytest.mark.parametrize("check", [
    {"type": "text_present", "text": "Send"},
    {"type": "text_equals", "region": (0, 0, 100, 100)},
    {"type": "text_equals", "text": "Send", "region": (0, 0, 0, 100)},
    {"type": "text_equals", "text": "Send", "region": (0, 0, float("nan"), 100)},
    {"type": "text_present", "text": "Send", "region": (0, 0, 100, 100), "window": "bound_target"},
    {"type": "window_closed"},
    {"type": "window_closed", "window": "some_other_app"},
    {"type": "window_present", "window": "bound_target", "text": "Send"},
])
def test_feedback_check_rejects_unbounded_or_contradictory_claims(check):
    with pytest.raises(ValidationError):
        FeedbackCheck.model_validate(check)


def test_grounding_unique_target_and_normalized_exact_text():
    observation = _observation()
    assert resolve_target(observation, "obs-1:text-1", " SEND ") == (140, 140)
    assert resolve_action(_decision(), observation, step_id="step-1").point == (140, 140)


@pytest.mark.parametrize("target_id,text", [
    ("obs-0:text-1", "Send"), ("obs-1:text-2", "Send"),
    ("obs-1:text-1", "Missing"), ("obs-1:text-1", "Sen"),
])
def test_grounding_rejects_unknown_stale_or_mismatched_targets(target_id, text):
    with pytest.raises(GroundingError):
        resolve_target(_observation(), target_id, text)


def test_id_does_not_bypass_ambiguity_even_with_low_confidence_duplicate():
    observation = _observation([
        TextElement("Send", 0.99, (100, 120, 180, 160)),
        TextElement("send", 0.1, (400, 120, 480, 160)),
    ])
    with pytest.raises(GroundingError, match="found 2"):
        resolve_target(observation, "obs-1:text-1", "Send")


def test_region_excludes_background_duplicates_without_changing_ids():
    observation = _observation([
        TextElement("Send", 0.95, (100, 120, 180, 160)),
        TextElement("Send", 0.95, (400, 120, 480, 160)),
    ], allowed_box=(300, 0, 900, 500))
    assert [element["target_id"] for element in observation.context()["elements"]] == ["obs-1:text-2"]
    assert resolve_target(observation, "obs-1:text-2", "Send") == (440, 140)
    with pytest.raises(GroundingError):
        resolve_target(observation, "obs-1:text-1", "Send")


def test_threshold_is_configurable_and_inclusive():
    observation = _observation([TextElement("Send", 0.5, (100, 120, 180, 160))])
    assert resolve_target(observation, "obs-1:text-1", "Send", min_confidence=0.5) == (140, 140)
    with pytest.raises(GroundingError, match="confidence"):
        resolve_target(observation, "obs-1:text-1", "Send", min_confidence=0.51)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_grounding_rejects_invalid_threshold(value):
    with pytest.raises(GroundingError):
        resolve_target(_observation(), "obs-1:text-1", "Send", min_confidence=value)


def test_resized_ocr_and_crop_origin_map_once_to_primary_pixels():
    restored = map_text_elements([TextElement("Send", 0.95, (10, 20, 30, 40))], (200, 100), (400, 200))
    observation = _observation(restored, crop_origin=(300, 200), image_size=(400, 200))
    assert resolve_target(observation, "obs-1:text-1", "Send") == (340, 260)


def test_screen_edge_coordinates_stay_in_bounds():
    observation = _observation([TextElement("Send", 0.95, (998, 798, 1000, 800))])
    assert resolve_target(observation, "obs-1:text-1", "Send") == (999, 799)


@pytest.mark.parametrize("box", [(-1, 0, 20, 20), (0, 0, 1001, 20), (0.0, 0, 20, 20), (True, 0, 20, 20), (0, 0, float("inf"), 20)])
def test_observation_rejects_invalid_ocr_coordinates(box):
    with pytest.raises(GroundingError):
        _observation([TextElement("Send", 0.95, box)])


@pytest.mark.parametrize("kwargs", [
    {"crop_origin": (-1, 0)}, {"crop_origin": (900, 700)},
    {"image_size": (0, 100)}, {"allowed_box": (0, 0, 1001, 800)},
    {"allowed_box": (0, 0, 0, 100)},
])
def test_observation_rejects_invalid_capture_scope(kwargs):
    with pytest.raises(GroundingError):
        _observation(**kwargs)


@pytest.mark.parametrize("region", [ScreenRegion(-1000, 0, 1000, 800), ScreenRegion(1000, 0, 1000, 800), ScreenRegion(0, 0, 800, 600)])
def test_observation_refuses_nonprimary_capture(region):
    with pytest.raises(GroundingError, match="primary"):
        build_observation("obs-1", "screen.png", [], screen_region=region, screen_size=(1000, 800))


@pytest.mark.parametrize("changes", [{"observation_id": "old"}, {"step_id": "step-2"}])
def test_decision_is_bound_to_observation_and_current_step(changes):
    with pytest.raises(GroundingError):
        resolve_action(_decision(**changes), _observation(), step_id="step-1")


def test_typing_requires_independent_focus_and_replace_selection():
    decision = _decision({"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": "  Hello  ", "mode": "replace"})
    for kwargs in ({}, {"focus_verified": True, "focused_target_id": "other"},
                   {"focus_verified": True, "focused_target_id": "obs-1:text-1"}):
        with pytest.raises(GroundingError):
            resolve_action(decision, _observation(), step_id="step-1", **kwargs)
    resolved = resolve_action(decision, _observation(), step_id="step-1", focus_verified=True,
                              focused_target_id="obs-1:text-1", selection_verified=True)
    assert resolved.action.text == "  Hello  "
    assert resolved.action.mode == "replace"


def test_append_does_not_require_select_all_but_still_requires_focus():
    decision = _decision({"type": "type", "target_id": "obs-1:text-1", "target_text": "Send", "text": "Hello", "mode": "append"})
    resolved = resolve_action(decision, _observation(), step_id="step-1", focus_verified=True,
                              focused_target_id="obs-1:text-1")
    assert resolved.action.mode == "append"


def test_key_requires_verified_application_focus():
    decision = _decision({"type": "key", "key": "ctrl+l", "purpose": "Focus the address field"})
    with pytest.raises(GroundingError, match="focus"):
        resolve_action(decision, _observation(), step_id="step-1")
    assert resolve_action(decision, _observation(), step_id="step-1", focus_verified=True).point is None


def test_feedback_uses_actual_region_not_completion_claim_or_background():
    observation = _observation([
        TextElement("Old", 0.95, (100, 120, 180, 160)),
        TextElement("New", 0.95, (400, 120, 480, 160)),
    ])
    check = FeedbackCheck(type="text_present", text="New", region=(0, 0, 300, 300))
    assert not evaluate_text_check(check, observation)
    assert evaluate_text_check(FeedbackCheck(type="text_present", text="New", region=(300, 0, 900, 500)), observation)


def test_text_equals_rejects_appended_old_content_and_low_confidence():
    check = FeedbackCheck(type="text_equals", text="New", region=(0, 0, 300, 300))
    assert not evaluate_text_check(check, _observation([TextElement("Old New", 0.95, (100, 120, 180, 160))]))
    assert not evaluate_text_check(check, _observation([TextElement("New", 0.1, (100, 120, 180, 160))]))
    assert not evaluate_text_check(check, _observation([TextElement("new", 0.95, (100, 120, 180, 160))]))
    assert evaluate_text_check(check, _observation([TextElement("New", 0.95, (100, 120, 180, 160))]))


def test_window_check_never_passes_from_model_claim_alone():
    with pytest.raises(GroundingError, match="runtime evidence"):
        evaluate_text_check(FeedbackCheck(type="window_closed", window="bound_target"), _observation())


def test_feedback_rejects_partially_hidden_ocr_at_region_boundary():
    observation = _observation([
        TextElement("Old", 0.95, (80, 120, 110, 160)),
        TextElement("New", 0.95, (150, 120, 200, 160)),
    ])
    check = FeedbackCheck(type="text_equals", text="New", region=(100, 100, 250, 200))
    assert not evaluate_text_check(check, observation)


def test_feedback_rejects_out_of_scope_region():
    observation = _observation(allowed_box=(0, 0, 300, 300))
    with pytest.raises(GroundingError, match="outside"):
        evaluate_text_check(FeedbackCheck(type="text_present", text="Send", region=(0, 0, 1000, 800)), observation)


@pytest.mark.parametrize(("elements", "expected"), [
    ([TextElement("New", 0.95, (100, 120, 180, 160))], True),
    ([TextElement("New", 0.95, (100, 120, 180, 160)),
      TextElement("Unrelated", 0.1, (200, 200, 280, 240))], True),
    ([TextElement("New", 0.1, (100, 120, 180, 160)),
      TextElement("Unrelated", 0.95, (200, 200, 280, 240))], False),
    ([TextElement("Unrelated", 0.95, (100, 120, 180, 160))], False),
    ([], False),
    ([TextElement("New", 0.5, (100, 120, 180, 160))], True),
    ([TextElement("New", 0.49, (100, 120, 180, 160))], False),
    ([TextElement("New content", 0.95, (100, 120, 180, 160))], False),
], ids=[
    "target-alone", "target-with-unrelated-low-confidence", "low-confidence-target",
    "missing-target", "no-ocr", "at-default-threshold", "below-default-threshold",
    "no-substring-match",
])
def test_text_present_requires_confident_matching_text_not_confident_unrelated_text(
    elements: list[TextElement], expected: bool,
) -> None:
    check = FeedbackCheck(type="text_present", text="New", region=(0, 0, 300, 300))

    assert evaluate_text_check(check, _observation(elements)) is expected


@pytest.mark.parametrize(("threshold", "expected"), [(0.75, True), (0.76, False)])
def test_text_present_uses_the_supplied_confidence_threshold(
    threshold: float, expected: bool,
) -> None:
    check = FeedbackCheck(type="text_present", text="New", region=(0, 0, 300, 300))
    observation = _observation([TextElement("New", 0.75, (100, 120, 180, 160))])

    assert evaluate_text_check(check, observation, min_confidence=threshold) is expected


@pytest.mark.parametrize(("other_confidence", "expected"), [(0.1, False), (0.95, True)])
def test_text_equals_still_requires_every_region_element_to_be_confident(
    other_confidence: float, expected: bool,
) -> None:
    check = FeedbackCheck(type="text_equals", text="New Other", region=(0, 0, 300, 300))
    observation = _observation([
        TextElement("New", 0.95, (100, 120, 180, 160)),
        TextElement("Other", other_confidence, (200, 200, 280, 240)),
    ])

    assert evaluate_text_check(check, observation) is expected


def test_text_present_still_rejects_partially_intersecting_low_confidence_ocr():
    check = FeedbackCheck(type="text_present", text="New", region=(100, 100, 250, 200))
    observation = _observation([
        TextElement("New", 0.95, (150, 120, 200, 160)),
        TextElement("Unrelated", 0.1, (80, 120, 110, 160)),
    ])

    assert not evaluate_text_check(check, observation)


@pytest.mark.parametrize("region", [(-1, 0, 300, 300), (0, 0, 0, 300), (100, 0, 50, 300)])
def test_text_present_still_rejects_invalid_region_structure(region):
    with pytest.raises(ValidationError):
        FeedbackCheck(type="text_present", text="New", region=region)
