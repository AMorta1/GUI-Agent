"""LangChain-based planning for GUI tasks without desktop execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import MultimodalModelClient, MultimodalRequest


SYSTEM_PROMPT = """You are a GUI task planning assistant.
Create a short ordered plan, but do not execute actions or claim that any action was run.
Use only the user's instruction, supplied context, and visible screenshot evidence.
Do not invent precise coordinates. If a critical target or intent is missing, ask one
clarifying question instead of guessing. Each ready plan must contain 2 to 8
individually verifiable steps, and each dependency may reference only an earlier step.
When the application and desired result are clear, make an interface-adaptive plan.
Application version, theme, exact layout, and screen coordinates are not reasons to ask
for clarification.
Do not ask the user to describe the current screen or current UI state. When no
screenshot is supplied, make inspecting or locating the relevant interface the first
plan step. Reserve clarification for missing goal-defining values such as the target
file, recipient, account, or content.
For a navigation task that names both an application and a destination page, always use
ready even when no screenshot or current URL is available; locating the relevant menu
or entry is a valid first step.
Example status decision: "In a named browser, navigate to a named settings page" is
ready, with one step to locate the menu and a later step to verify the destination.

The two statuses are mutually exclusive:
- ready: clarification_question must be null and steps must contain 2 to 8 items.
- needs_clarification: clarification_question must contain one question; steps must be
  an empty list. Do not include hypothetical steps with a clarification question.

Return only the requested JSON structure.
{format_instructions}
"""

USER_PROMPT = """User goal: {task}
Screenshot: {screenshot_status}
Available OCR/UI context: {context}
"""


class PlanningError(RuntimeError):
    """Raised when model output cannot be validated as a safe task plan."""


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: str = Field(min_length=1)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    status: Literal["ready", "needs_clarification"]
    clarification_question: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> TaskPlan:
        if self.status == "needs_clarification":
            if not self.clarification_question or not self.clarification_question.strip():
                raise ValueError("clarification plans require one question")
            if self.steps:
                raise ValueError("clarification plans cannot contain executable steps")
            return self
        if self.clarification_question is not None:
            raise ValueError("ready plans cannot contain a clarification question")
        if not 2 <= len(self.steps) <= 8:
            raise ValueError("ready plans must contain 2 to 8 steps")

        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            unknown = [dependency for dependency in step.depends_on if dependency not in seen]
            if unknown:
                raise ValueError(
                    f"step {step.step_id} depends on non-earlier steps: {unknown}"
                )
            seen.add(step.step_id)
        return self


class GuiPlanningAgent:
    """Turn a user goal and optional static perception into a validated plan."""

    def __init__(self, model_client: MultimodalModelClient) -> None:
        self.model_client = model_client
        self.output_parser = PydanticOutputParser(pydantic_object=TaskPlan)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        ).partial(format_instructions=self.output_parser.get_format_instructions())

    def plan(
        self,
        task: str,
        *,
        image_path: str | Path | None = None,
        context: str | None = None,
    ) -> TaskPlan:
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be blank")
        messages = self.prompt.format_messages(
            task=normalized_task,
            screenshot_status=(
                "A static screenshot is attached."
                if image_path is not None
                else (
                    "Text-only planning is available; begin by locating the relevant "
                    "interface and do not ask for the current UI state."
                )
            ),
            context=context.strip() if context and context.strip() else "None provided.",
        )
        system_prompt = str(messages[0].content)
        user_prompt = str(messages[1].content)
        response = self.model_client.generate(
            MultimodalRequest(
                instruction=user_prompt,
                image_path=Path(image_path) if image_path is not None else None,
                system_prompt=system_prompt,
                max_new_tokens=1024,
            )
        )
        try:
            return self.output_parser.parse(response.text)
        except (OutputParserException, ValueError) as exc:
            raise PlanningError(f"Model returned an invalid task plan: {exc}") from exc
