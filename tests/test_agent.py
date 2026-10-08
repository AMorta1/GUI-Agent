from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui_agent.agent import GuiActionAgent, GuiPlanningAgent, PlanningError, TaskPlan
from gui_agent.capture import ScreenRegion
from gui_agent.grounding import (
    ActionDecision, GroundingError, build_observation, evaluate_text_check,
    resolve_action, validate_decision_context,
)
from gui_agent.models import MultimodalRequest, MultimodalResponse
from gui_agent.perception import TextElement


class FakeModelClient:
    def __init__(self, output: dict[str, object] | str) -> None:
        self.output = json.dumps(output) if isinstance(output, dict) else output
        self.requests: list[MultimodalRequest] = []

    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        self.requests.append(request)
        return MultimodalResponse(
            text=self.output,
            provider="fake",
            model="fake-planner",
        )


def _ready_plan(goal: str = "Open settings", step_count: int = 2) -> dict[str, object]:
    return {
        "goal": goal,
        "status": "ready",
        "clarification_question": None,
        "steps": [
            {
                "step_id": f"step-{index}",
                "description": f"Complete planning step {index}",
                "depends_on": [] if index == 1 else [f"step-{index - 1}"],
                "success_criteria": f"Planning step {index} is complete",
            }
            for index in range(1, step_count + 1)
        ],
    }


def test_agent_formats_prompt_and_forwards_static_context(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"image")
    client = FakeModelClient(_ready_plan())
    agent = GuiPlanningAgent(client)

    plan = agent.plan(
        "Open settings",
        image_path=image,
        context="OCR: Settings, Account",
    )

    assert plan.status == "ready"
    assert [step.step_id for step in plan.steps] == ["step-1", "step-2"]
    request = client.requests[0]
    assert request.image_path == image
    assert "OCR: Settings, Account" in request.instruction
    assert "A static screenshot is attached" in request.instruction
    assert "do not execute" in request.system_prompt.lower()
    assert "coordinates" in request.system_prompt.lower()
    normalized_system_prompt = " ".join(request.system_prompt.lower().split())
    assert "steps must be an empty list" in normalized_system_prompt
    assert "2 to 8" in normalized_system_prompt
    assert "application version" in normalized_system_prompt
    assert "current screen" in normalized_system_prompt
    assert "navigation task" in normalized_system_prompt


def test_agent_supports_text_only_planning() -> None:
    client = FakeModelClient(_ready_plan("Find an article"))

    plan = GuiPlanningAgent(client).plan("Find an article")

    assert plan.goal == "Find an article"
    assert client.requests[0].image_path is None
    assert "Text-only planning is available" in client.requests[0].instruction


def test_agent_parses_json_code_fence_and_enforces_upper_step_limit() -> None:
    output = "```json\n" + json.dumps(_ready_plan(step_count=8)) + "\n```"

    plan = GuiPlanningAgent(FakeModelClient(output)).plan("Open settings")

    assert plan.status == "ready"
    assert len(plan.steps) == 8

    with pytest.raises(PlanningError, match="2 to 8"):
        GuiPlanningAgent(FakeModelClient(_ready_plan(step_count=9))).plan("Open settings")


def test_agent_accepts_clarification_without_steps() -> None:
    client = FakeModelClient(
        {
            "goal": "Send a message",
            "status": "needs_clarification",
            "clarification_question": "Which contact should receive the message?",
            "steps": [],
        }
    )

    plan = GuiPlanningAgent(client).plan("Send a message")

    assert plan.status == "needs_clarification"
    assert plan.clarification_question == "Which contact should receive the message?"


@pytest.mark.parametrize(
    "task",
    [
        "Open the browser settings",
        "Find the latest order in the admin page",
        "Enter Boston in the destination field",
        "Compare two visible search results",
    ],
)
def test_representative_tasks_produce_valid_ordered_plans(task: str) -> None:
    plan_data = _ready_plan(task)

    plan = GuiPlanningAgent(FakeModelClient(plan_data)).plan(task)

    assert isinstance(plan, TaskPlan)
    assert len(plan.steps) == 2
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["steps"].__setitem__(1, {**plan["steps"][1], "step_id": "step-1"}),
        lambda plan: plan["steps"][0].__setitem__("depends_on", ["step-2"]),
        lambda plan: plan.__setitem__("steps", plan["steps"][:1]),
    ],
)
def test_agent_rejects_invalid_step_structure(mutation: object) -> None:
    plan_data = _ready_plan()
    mutation(plan_data)

    with pytest.raises(PlanningError, match="invalid task plan"):
        GuiPlanningAgent(FakeModelClient(plan_data)).plan("Open settings")


def test_agent_rejects_blank_task_without_calling_model() -> None:
    client = FakeModelClient(_ready_plan())

    with pytest.raises(ValueError, match="blank"):
        GuiPlanningAgent(client).plan("   ")

    assert client.requests == []


def test_clarification_plan_cannot_contain_steps() -> None:
    plan_data = _ready_plan()
    plan_data["status"] = "needs_clarification"
    plan_data["clarification_question"] = "Which settings page?"

    with pytest.raises(PlanningError, match="cannot contain"):
        GuiPlanningAgent(FakeModelClient(plan_data)).plan("Open settings")


def _action_observation():
    return build_observation("obs-1", "screen.png", [TextElement("Settings", 0.95, (10, 10, 100, 50))],
                             screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600))


def _action_output(**kwargs):
    return {"status": "action", "step_id": "step-1", "observation_id": "obs-1",
            "action": {"type": "click", "target_id": "obs-1:text-1", "target_text": "Settings"}, **kwargs}


def test_action_agent_forwards_plan_latest_observation_and_proposes_without_execution():
    observation = _action_observation()
    client = FakeModelClient(_action_output())
    decision = GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                           step_id="step-1", observation=observation)
    assert resolve_action(decision, observation, step_id="step-1").point == (55, 30)
    request = client.requests[0]
    assert request.image_path == observation.image_path
    assert "Original user instruction: Open settings" in request.instruction
    assert "obs-1:text-1" in request.instruction
    assert '"confidence": 0.95' in request.instruction
    assert '"focus_verified": false' in request.instruction
    assert "do not execute" in request.system_prompt
    assert "Never invent coordinates" in request.system_prompt
    assert "independent verification" in request.system_prompt


@pytest.mark.parametrize("kwargs", [
    {"step_id": "unknown"}, {"step_id": "step-2"},
    {"step_id": "step-1", "completed_steps": ("unknown",)},
    {"step_id": "step-1", "completed_steps": ("step-1",)},
    {"step_id": "step-1", "completed_steps": ("step-2",)},
])
def test_action_agent_rejects_bad_step_or_dependencies_before_model(kwargs):
    client = FakeModelClient(_action_output())
    with pytest.raises(ValueError):
        GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                     observation=_action_observation(), **kwargs)
    assert client.requests == []


def test_action_agent_accepts_dependency_satisfied_step():
    client = FakeModelClient(_action_output(step_id="step-2"))
    decision = GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                           step_id="step-2", completed_steps=("step-1",),
                                           observation=_action_observation())
    assert decision.step_id == "step-2"
    assert 'Completed steps: ["step-1"]' in client.requests[0].instruction


def test_action_agent_rejects_clarification_and_mutated_invalid_plan_before_model():
    client = FakeModelClient(_action_output())
    plan = TaskPlan(goal="Send", status="needs_clarification", clarification_question="Who?")
    with pytest.raises(ValueError, match="clarification"):
        GuiActionAgent(client).decide("Send", plan, step_id="step-1", observation=_action_observation())
    plan = TaskPlan.model_validate(_ready_plan())
    plan.steps.pop()
    with pytest.raises(ValueError, match="2 to 8"):
        GuiActionAgent(client).decide("Open settings", plan, step_id="step-1", observation=_action_observation())
    assert client.requests == []


@pytest.mark.parametrize("changes", [
    {"observation_id": "old"}, {"step_id": "step-2"}, {"action": None},
    {"action": {"type": "click", "x": 42, "y": 42}},
    {"action": {"type": "key", "key": "win+r", "purpose": "Launch"}},
    {"status": "step_complete", "action": None, "checks": []},
    {"status": "step_complete", "action": None, "checks": [
        {"type": "text_present", "text": "Settings", "region": (0, 0, 1000, 800)}]},
])
def test_action_agent_rejects_invalid_or_stale_model_decision(changes):
    client = FakeModelClient(_action_output(**changes))
    with pytest.raises(PlanningError, match="invalid action decision"):
        GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                     step_id="step-1", observation=_action_observation())
    assert len(client.requests) == 1


def test_action_agent_blocked_is_not_an_executable_action():
    client = FakeModelClient(_action_output(status="blocked", action=None, reason="Ambiguous target"))
    observation = _action_observation()
    decision = GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                           step_id="step-1", observation=observation)
    assert decision.status == "blocked"
    with pytest.raises(GroundingError):
        resolve_action(decision, observation, step_id="step-1")


def test_action_agent_step_complete_only_proposes_checks_not_success():
    client = FakeModelClient(_action_output(status="step_complete", action=None, checks=[
        {"type": "text_present", "text": "Absent", "region": (0, 0, 800, 600)}]))
    observation = _action_observation()
    decision = GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                           step_id="step-1", observation=observation)
    assert decision.status == "step_complete"
    assert not evaluate_text_check(decision.checks[0], observation)


def test_action_agent_rejects_blank_task_before_model():
    client = FakeModelClient(_action_output())
    with pytest.raises(ValueError, match="blank"):
        GuiActionAgent(client).decide(" ", TaskPlan.model_validate(_ready_plan()),
                                     step_id="step-1", observation=_action_observation())
    assert client.requests == []


def test_action_prompt_explicitly_documents_status_field_contract() -> None:
    client = FakeModelClient(_action_output())
    GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )
    prompt = " ".join(client.requests[0].system_prompt.split())

    assert "The status fields are mutually exclusive:" in prompt
    assert (
        "- action: action must contain exactly one action; checks must be [] or omitted; "
        "reason must be null or omitted."
    ) in prompt
    assert "Do not add a reason or explanation to an action." in prompt
    assert (
        "- step_complete: action must be null or omitted; checks must be a non-empty list; "
        "reason must be null or omitted."
    ) in prompt
    assert (
        "- blocked: action must be null or omitted; checks must be [] or omitted; "
        "reason must be a non-blank string describing the blocker."
    ) in prompt


def test_action_agent_rejects_explanatory_reason_without_repair_or_retry() -> None:
    output = _action_output(reason="Locate the Settings field on the screen.")
    client = FakeModelClient(output)

    with pytest.raises(PlanningError, match="without checks/reason") as error:
        GuiActionAgent(client).decide(
            "Open settings", TaskPlan.model_validate(_ready_plan()),
            step_id="step-1", observation=_action_observation(),
        )

    assert len(client.requests) == 1
    assert json.loads(client.output) == output
    assert output["reason"] in str(error.value)


@pytest.mark.parametrize(("step_id", "observation_id"), [
    ("step-1", "obs-1"), ("step-2", "obs-2"),
    ('step-"{one}"', 'obs-"{one}"'),
])
def test_action_prompt_template_contains_current_ids_and_all_status_rules(
    step_id: str, observation_id: str,
) -> None:
    plan_data = _ready_plan()
    plan_data["steps"][0]["step_id"] = step_id
    plan_data["steps"][1]["step_id"] = "later-step"
    plan_data["steps"][1]["depends_on"] = [step_id]
    observation = build_observation(
        observation_id, "screen.png", [TextElement("Settings", 0.95, (10, 10, 100, 50))],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    client = FakeModelClient(_action_output(
        step_id=step_id, observation_id=observation_id,
        action={"type": "click", "target_id": f"{observation_id}:text-1", "target_text": "Settings"},
    ))

    decision = GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(plan_data),
        step_id=step_id, observation=observation,
    )

    instruction = client.requests[0].instruction
    template = instruction.split("Required response template with this request's exact IDs:\n", 1)[1]
    template = json.loads(template.split("\nThis is an outer-field template", 1)[0])
    assert template == {
        "status": "<action|step_complete|blocked>", "step_id": step_id,
        "observation_id": observation_id, "action": None, "checks": [], "reason": None,
    }
    normalized = " ".join(instruction.split())
    assert "- action: replace action=null with one schema-valid action object" in normalized
    assert "- step_complete: keep action=null and reason=null" in normalized
    assert "- blocked: keep action=null and checks=[]" in normalized
    assert "Keep both step_id and observation_id exactly as shown, for every status." in normalized
    assert decision.step_id == step_id
    assert decision.observation_id == observation_id
    assert len(client.requests) == 1


@pytest.mark.parametrize("missing", [("step_id",), ("observation_id",), ("step_id", "observation_id")])
def test_action_agent_does_not_fill_missing_ids_or_retry(missing: tuple[str, ...]) -> None:
    output = _action_output()
    for field in missing:
        output.pop(field)
    client = FakeModelClient(output)

    with pytest.raises(PlanningError, match="Field required") as error:
        GuiActionAgent(client).decide(
            "Open settings", TaskPlan.model_validate(_ready_plan()),
            step_id="step-1", observation=_action_observation(),
        )

    assert len(client.requests) == 1
    assert json.loads(client.output) == output
    assert all(field in str(error.value) for field in missing)


@pytest.mark.parametrize("explicit_defaults", [False, True], ids=["omitted", "explicit"])
@pytest.mark.parametrize("status", ["action", "step_complete", "blocked"])
def test_action_agent_accepts_each_status_with_compatible_fields(
    status: str, explicit_defaults: bool,
) -> None:
    output = _action_output(status=status)
    if status == "step_complete":
        output.pop("action")
        output["checks"] = [
            {"type": "text_present", "text": "Settings", "region": (0, 0, 800, 600)},
        ]
    elif status == "blocked":
        output.pop("action")
        output["reason"] = "The target cannot be uniquely located."
    if explicit_defaults:
        output.setdefault("action", None)
        output.setdefault("checks", [])
        output.setdefault("reason", None)
    client = FakeModelClient(output)

    decision = GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )

    assert decision.status == status
    assert (decision.action is not None) == (status == "action")
    assert bool(decision.checks) == (status == "step_complete")
    assert (decision.reason is not None) == (status == "blocked")
    assert len(client.requests) == 1


@pytest.mark.parametrize("changes", [
    {"checks": [{"type": "text_present", "text": "Settings", "region": (0, 0, 800, 600)}]},
    {"status": "step_complete", "action": None,
     "checks": [{"type": "window_present", "window": "bound_target"}], "reason": "Located"},
    {"status": "step_complete", "checks": [{"type": "window_present", "window": "bound_target"}]},
    {"status": "blocked", "reason": "Not safe"},
    {"status": "blocked", "action": None, "reason": "Not safe",
     "checks": [{"type": "window_present", "window": "bound_target"}]},
    {"status": "blocked", "action": None, "reason": " "},
], ids=[
    "action-with-checks", "complete-with-reason", "complete-with-action",
    "blocked-with-action", "blocked-with-checks", "blocked-with-blank-reason",
])
def test_action_agent_rejects_incompatible_status_fields(changes: dict[str, object]) -> None:
    client = FakeModelClient(_action_output(**changes))

    with pytest.raises(PlanningError, match="invalid action decision"):
        GuiActionAgent(client).decide(
            "Open settings", TaskPlan.model_validate(_ready_plan()),
            step_id="step-1", observation=_action_observation(),
        )

    assert len(client.requests) == 1


def test_action_prompt_explicitly_documents_feedback_fields_and_semantics() -> None:
    client = FakeModelClient(_action_output())
    GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )
    prompt = " ".join(client.requests[0].system_prompt.split())

    assert "These action identifiers are not feedback fields." in prompt
    assert "text_present and text_equals require type, text and region" in prompt
    assert "A text check without region is invalid" in prompt
    assert "JSON schema lists text and region as nullable" in prompt
    assert "[left, top, right, bottom], four integers" in prompt
    assert "crop-local coordinates, not desktop coordinates" in prompt
    assert "entirely inside allowed_box" in prompt
    assert "referencing a supplied bbox for a check is not inventing a click coordinate" in prompt
    assert "text_present checks for a normalized exact text match" in prompt
    assert "text_equals checks that all region OCR text" in prompt
    assert "including case" in prompt
    assert "Do not crop away old content to manufacture a match." in prompt
    assert 'window_present and window_closed require window="bound_target"' in prompt
    assert "Never put target_id or target_text in checks." in prompt
    assert "Runtime re-observes and verifies each check" in prompt


@pytest.mark.parametrize("check_type", ["text_present", "text_equals"])
def test_action_prompt_complete_feedback_examples_match_their_fictional_evidence(
    check_type: str,
) -> None:
    client = FakeModelClient(_action_output())
    GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )
    instruction = client.requests[0].instruction
    examples = instruction.split(
        "Feedback check format examples (fictional data, not current evidence):\n", 1,
    )[1]
    examples = json.loads(examples.split("\nThese examples explain structure only", 1)[0])
    source = examples["example_observation"]
    width, height = source["image_size"]
    observation = build_observation(
        source["observation_id"], "unused-example.png",
        [TextElement(item["text"], item["confidence"], tuple(item["bbox"]))
         for item in source["elements"]],
        screen_region=ScreenRegion(0, 0, width, height), screen_size=(width, height),
        allowed_box=tuple(source["allowed_box"]),
    )
    output = next(item for item in examples["step_complete_examples"]
                  if item["checks"][0]["type"] == check_type)
    decision = ActionDecision.model_validate(output)

    validate_decision_context(decision, observation, "example-step")
    assert decision.status == "step_complete"
    assert decision.action is None and decision.reason is None
    assert output["checks"][0]["region"] == source["elements"][0]["bbox"]
    assert set(output["checks"][0]) == {"type", "text", "region", "window"}
    assert evaluate_text_check(decision.checks[0], observation)
    assert decision.observation_id != _action_observation().observation_id
    assert "Do not copy" in instruction
    assert "current step and observation" in instruction
    assert "use blocked if no safe verifiable check is available" in instruction
    assert "Keep both step_id and observation_id exactly as shown" in instruction
    assert len(client.requests) == 1


@pytest.mark.parametrize(("check", "error_message"), [
    ({"type": "text_equals", "text": "Settings", "target_id": "obs-1:text-1"},
     "Extra inputs are not permitted"),
    ({"type": "text_equals", "text": "Settings", "region": [10, 10, 100, 50],
      "target_id": "obs-1:text-1"}, "Extra inputs are not permitted"),
    ({"type": "text_present", "text": "Settings", "region": [10, 10, 100, 50],
      "target_text": "Settings"}, "Extra inputs are not permitted"),
    ({"type": "text_equals", "text": "Settings"}, "text checks require text and region"),
    ({"type": "text_present", "text": "Settings"}, "text checks require text and region"),
    ({"type": "text_equals", "region": [10, 10, 100, 50]}, "text checks require text and region"),
    ({"type": "text_present", "region": [10, 10, 100, 50]}, "text checks require text and region"),
    ({"type": "window_present"}, "window checks require bound_target"),
], ids=[
    "stage3-action-id-in-check", "action-id-even-with-region", "action-text-in-check",
    "equals-missing-region", "present-missing-region", "equals-missing-text",
    "present-missing-text", "window-missing-binding",
])
def test_action_agent_rejects_invalid_feedback_without_repair_or_retry(
    check: dict[str, object], error_message: str,
) -> None:
    output = _action_output(status="step_complete", action=None, checks=[check], reason=None)
    client = FakeModelClient(output)

    with pytest.raises(PlanningError, match=error_message):
        GuiActionAgent(client).decide(
            "Open settings", TaskPlan.model_validate(_ready_plan()),
            step_id="step-1", observation=_action_observation(),
        )

    assert json.loads(client.output) == output
    assert len(client.requests) == 1


@pytest.mark.parametrize("check_type", ["window_present", "window_closed"])
def test_action_agent_accepts_window_feedback_structure_without_claiming_evidence(
    check_type: str,
) -> None:
    client = FakeModelClient(_action_output(
        status="step_complete", action=None,
        checks=[{"type": check_type, "window": "bound_target", "text": None, "region": None}],
        reason=None,
    ))
    decision = GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )

    assert decision.status == "step_complete"
    assert decision.checks[0].window == "bound_target"
    with pytest.raises(GroundingError, match="independent runtime evidence"):
        evaluate_text_check(decision.checks[0], _action_observation())
    assert len(client.requests) == 1


def test_action_prompt_requires_completion_assessment_before_proposing_actions() -> None:
    client = FakeModelClient(_action_output())
    GuiActionAgent(client).decide(
        "Open settings", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=_action_observation(),
    )
    prompt = " ".join(client.requests[0].system_prompt.split())

    assert "Completion assessment takes priority over action generation on every request." in prompt
    assert "First compare the latest observation with the current step's success criteria." in prompt
    assert "you must return step_complete with non-empty, schema-valid checks" in prompt
    assert "do not return action or repeat the operation" in prompt
    assert "A step description is not an instruction to repeat an operation" in prompt
    assert "that omission does not mean more action is needed" in prompt
    assert "If evidence is missing, ambiguous or insufficient, do not claim completion." in prompt
    assert "Only propose an action when the criteria are not yet satisfied" in prompt
    assert prompt.index("Completion assessment takes priority") < prompt.index("Return action for exactly one")


@pytest.mark.parametrize("step_index", [0, 1])
def test_action_request_highlights_description_and_criteria_of_the_current_step(
    step_index: int,
) -> None:
    plan_data = _ready_plan()
    for index, step in enumerate(plan_data["steps"]):
        step["description"] = f'Inspect section {{section-{index}}}: "Report"\nDo not repeat.'
        step["success_criteria"] = f'The heading "Report {index}" is visible.'
    plan = TaskPlan.model_validate(plan_data)
    current = plan.steps[step_index]
    client = FakeModelClient(_action_output(step_id=current.step_id))
    GuiActionAgent(client).decide(
        "Inspect the report", plan, step_id=current.step_id,
        completed_steps=tuple(step.step_id for step in plan.steps[:step_index]),
        observation=_action_observation(),
    )
    instruction = client.requests[0].instruction
    description = instruction.split("Current step description: ", 1)[1].split("\n", 1)[0]
    criteria = instruction.split("Current step success criteria: ", 1)[1].split("\n", 1)[0]

    assert json.loads(description) == current.description
    assert json.loads(criteria) == current.success_criteria
    assert json.loads(criteria) != plan.steps[1 - step_index].success_criteria
    assert instruction.count("Current step description: ") == 1
    assert instruction.count("Current step success criteria: ") == 1
    assert len(client.requests) == 1


@pytest.mark.parametrize("result_text", ["Report Ready", "Search complete", "Message sent"])
def test_action_agent_forwards_updated_observation_and_accepts_model_completion_checks(
    result_text: str,
) -> None:
    plan_data = _ready_plan("Produce the visible result")
    plan_data["steps"][0].update(
        description="Activate the start control once.",
        success_criteria=f'The visible result is "{result_text}".',
    )
    plan = TaskPlan.model_validate(plan_data)
    before = build_observation(
        "before", "before.png", [TextElement("Start", 0.95, (10, 10, 150, 50))],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    after = build_observation(
        "after", "after.png", [TextElement(result_text, 0.95, (10, 10, 150, 50))],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    client = FakeModelClient(_action_output(
        observation_id="before",
        action={"type": "click", "target_id": "before:text-1", "target_text": "Start"},
    ))
    agent = GuiActionAgent(client)
    first = agent.decide(plan.goal, plan, step_id="step-1", observation=before)
    completion = _action_output(
        status="step_complete", observation_id="after", action=None, reason=None,
        checks=[{"type": "text_equals", "text": result_text, "region": [10, 10, 150, 50]}],
    )
    client.output = json.dumps(completion)
    second = agent.decide(plan.goal, plan, step_id="step-1", observation=after)
    request = client.requests[1]
    forwarded = request.instruction.split("Current observation: ", 1)[1].split("\n", 1)[0]

    assert first.status == "action"
    assert second.status == "step_complete" and second.action is None
    assert request.image_path == after.image_path != before.image_path
    assert json.loads(forwarded) == json.loads(json.dumps(after.context()))
    assert 'Current step success criteria: ' + json.dumps(plan.steps[0].success_criteria) in request.instruction
    assert 'Completed steps: []' in request.instruction
    assert evaluate_text_check(second.checks[0], after)
    assert json.loads(client.output) == completion
    assert len(client.requests) == 2


def test_action_agent_does_not_rewrite_repeated_model_action_into_completion() -> None:
    plan_data = _ready_plan()
    plan_data["steps"][0]["success_criteria"] = "Settings is visible."
    output = _action_output()
    client = FakeModelClient(output)
    decision = GuiActionAgent(client).decide(
        "Open settings once", TaskPlan.model_validate(plan_data),
        step_id="step-1", observation=_action_observation(),
    )

    assert decision.status == "action"
    assert decision.model_dump() == ActionDecision.model_validate(output).model_dump()
    assert json.loads(client.output) == output
    assert len(client.requests) == 1


@pytest.mark.parametrize("element", [
    TextElement("Pending", 0.95, (10, 10, 150, 50)),
    TextElement("Result ready", 0.1, (10, 10, 150, 50)),
], ids=["result-missing", "result-low-confidence"])
def test_model_completion_claim_still_needs_independent_observation_evidence(
    element: TextElement,
) -> None:
    observation = build_observation(
        "obs-1", "unused.png", [element],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    client = FakeModelClient(_action_output(
        status="step_complete", action=None,
        checks=[{"type": "text_equals", "text": "Result ready", "region": [10, 10, 150, 50]}],
    ))
    decision = GuiActionAgent(client).decide(
        "Produce the result", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=observation,
    )

    assert decision.status == "step_complete"
    assert not evaluate_text_check(decision.checks[0], observation)
    assert len(client.requests) == 1


def test_action_agent_candidate_is_proposal_context_not_verified_focus() -> None:
    observation = _action_observation()
    output = _action_output(action={"type": "type", "target_id": "obs-1:text-1",
                                   "target_text": "Settings", "text": "user value", "mode": "append"})
    client = FakeModelClient(output)
    decision = GuiActionAgent(client).decide(
        "Enter user value", TaskPlan.model_validate(_ready_plan()),
        step_id="step-1", observation=observation, focus_verified=True,
        input_focus_candidate_target_id="obs-1:text-1",
    )
    request = client.requests[0]
    evidence = json.loads(request.instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    assert evidence == {"focus_verified": True, "focused_target_id": None, "selection_verified": False,
                        "input_focus_candidate_target_id": "obs-1:text-1"}
    prompt = " ".join(request.system_prompt.split())
    assert "A candidate is NOT verified input focus." in prompt
    assert "exactly that current target" in prompt
    assert "obtain human focus confirmation and revalidate the target before typing" in prompt
    assert "selection_verified=false still forbids replace" in prompt
    assert decision.model_dump() == ActionDecision.model_validate(output).model_dump()
    with pytest.raises(GroundingError, match="verified input focus"):
        resolve_action(decision, observation, step_id="step-1", focus_verified=True)
    assert len(client.requests) == 1


def test_action_agent_default_context_has_no_input_candidate() -> None:
    client = FakeModelClient(_action_output())
    GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                 step_id="step-1", observation=_action_observation())
    evidence = json.loads(client.requests[0].instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    assert evidence == {"focus_verified": False, "focused_target_id": None, "selection_verified": False}
    assert "Without verified input focus or this candidate, return blocked instead of typing" in " ".join(client.requests[0].system_prompt.split())


@pytest.mark.parametrize("candidate", ["old:text-1", "obs-1:text-99", "", True, 1])
def test_action_agent_rejects_unknown_or_invalid_candidate_before_model(candidate) -> None:
    client = FakeModelClient(_action_output())
    with pytest.raises(ValueError, match="current observation target ID"):
        GuiActionAgent(client).decide("Open settings", TaskPlan.model_validate(_ready_plan()),
                                     step_id="step-1", observation=_action_observation(),
                                     input_focus_candidate_target_id=candidate)
    assert client.requests == []
