from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui_agent.agent import GuiPlanningAgent, PlanningError, TaskPlan
from gui_agent.models import MultimodalRequest, MultimodalResponse


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
