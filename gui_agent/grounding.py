"""Validated single-action contracts and deterministic OCR grounding; no execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import (
    BaseModel, ConfigDict, Field, StrictInt, StringConstraints,
    field_validator, model_validator,
)

from .capture import ScreenRegion, Size
from .perception import PixelBox, TextElement


# Project safety defaults, not OCR guarantees or provider limits.
DEFAULT_MIN_OCR_CONFIDENCE = 0.5
MAX_SCROLL_CLICKS = 10
NonBlank = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Box = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
AllowedKey = Literal["enter", "esc", "ctrl+a", "ctrl+l", "ctrl+o", "alt+f4"]


class GroundingError(ValueError):
    """An untrusted decision cannot be safely grounded in the current observation."""


class ClickAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["click"]
    target_id: NonBlank
    target_text: NonBlank


class TypeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["type"]
    target_id: NonBlank
    target_text: NonBlank
    text: Annotated[str, StringConstraints(strict=True, min_length=1)]
    mode: Literal["append", "replace"]

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input text must not be blank")
        return value


class KeyAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["key"]
    key: AllowedKey
    purpose: NonBlank


class ScrollAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["scroll"]
    target_id: NonBlank
    target_text: NonBlank
    clicks: StrictInt = Field(ge=-MAX_SCROLL_CLICKS, le=MAX_SCROLL_CLICKS)

    @field_validator("clicks")
    @classmethod
    def reject_zero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("scroll clicks must not be zero")
        return value


GuiAction = Annotated[ClickAction | TypeAction | KeyAction | ScrollAction, Field(discriminator="type")]


def _validate_box(box: tuple[int, ...]) -> None:
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise GroundingError("bbox must contain four integer coordinates")
    left, top, right, bottom = box
    if left < 0 or top < 0 or left >= right or top >= bottom:
        raise GroundingError("bbox must have non-negative origin and positive dimensions")


def _contains(outer: PixelBox, inner: PixelBox) -> bool:
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


class FeedbackCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text_present", "text_equals", "window_present", "window_closed"]
    text: NonBlank | None = None
    region: Box | None = None
    window: Literal["bound_target"] | None = None

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: Box | None) -> Box | None:
        if value is not None:
            _validate_box(value)
        return value

    @model_validator(mode="after")
    def validate_check(self) -> FeedbackCheck:
        if self.type.startswith("text_"):
            if self.text is None or self.region is None or self.window is not None:
                raise ValueError("text checks require text and region, without window")
        elif self.window != "bound_target" or self.text is not None or self.region is not None:
            raise ValueError("window checks require bound_target, without text or region")
        return self


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["action", "step_complete", "blocked"]
    step_id: NonBlank
    observation_id: NonBlank
    action: GuiAction | None = None
    checks: list[FeedbackCheck] = Field(default_factory=list)
    reason: NonBlank | None = None

    @model_validator(mode="after")
    def validate_status(self) -> ActionDecision:
        if self.status == "action":
            if self.action is None or self.checks or self.reason is not None:
                raise ValueError("action status requires exactly one action, without checks/reason")
        elif self.status == "step_complete":
            if self.action is not None or not self.checks or self.reason is not None:
                raise ValueError("step_complete requires checks, without action/reason")
        elif self.action is not None or self.checks or self.reason is None:
            raise ValueError("blocked requires a reason, without action/checks")
        return self


@dataclass(frozen=True)
class OcrTarget:
    target_id: str
    element: TextElement


@dataclass(frozen=True)
class OcrObservation:
    """OCR boxes are in original crop pixels, not model/processing-image pixels."""

    observation_id: str
    image_path: Path
    screen_region: ScreenRegion
    screen_size: Size
    image_size: Size
    crop_origin: tuple[int, int]
    allowed_box: PixelBox
    targets: tuple[OcrTarget, ...]

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise GroundingError("observation ID must not be blank")
        for size in (self.screen_size, self.image_size, self.screen_region.size):
            if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
                raise GroundingError("sizes must be positive integers")
        screen_origin = (self.screen_region.left, self.screen_region.top)
        if (
            any(type(value) is not int for value in screen_origin)
            or screen_origin != (0, 0)
            or self.screen_region.size != self.screen_size
        ):
            raise GroundingError("only an origin-zero primary screen is supported")
        if len(self.crop_origin) != 2 or any(
            type(value) is not int or value < 0 for value in self.crop_origin
        ):
            raise GroundingError("crop origin must contain non-negative integers")
        width, height = self.image_size
        if (
            self.crop_origin[0] + width > self.screen_size[0]
            or self.crop_origin[1] + height > self.screen_size[1]
        ):
            raise GroundingError("crop must lie inside the primary screen")
        _validate_box(self.allowed_box)
        if not _contains((0, 0, width, height), self.allowed_box):
            raise GroundingError("allowed region must lie inside the original crop")
        for index, target in enumerate(self.targets, start=1):
            if target.target_id != f"{self.observation_id}:text-{index}":
                raise GroundingError("target IDs must belong to this observation")
            element = target.element
            if not isinstance(element.text, str) or not element.text.strip():
                raise GroundingError("OCR text must not be blank")
            if (
                not isinstance(element.confidence, Real)
                or isinstance(element.confidence, bool)
                or not isfinite(element.confidence)
                or not 0 <= element.confidence <= 1
            ):
                raise GroundingError("OCR confidence must be finite and between 0 and 1")
            _validate_box(element.box)
            if not _contains((0, 0, width, height), element.box):
                raise GroundingError("OCR bbox must lie inside the original crop")

    def context(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "image_size": self.image_size,
            "allowed_box": self.allowed_box,
            "elements": [
                {
                    "target_id": target.target_id,
                    "text": target.element.text,
                    "confidence": target.element.confidence,
                    "bbox": target.element.box,
                }
                for target in self.targets
                if _contains(self.allowed_box, target.element.box)
            ],
        }


def build_observation(
    observation_id: str,
    image_path: str | Path,
    elements: Sequence[TextElement],
    *,
    screen_region: ScreenRegion,
    screen_size: Size,
    crop_origin: tuple[int, int] = (0, 0),
    image_size: Size | None = None,
    allowed_box: PixelBox | None = None,
) -> OcrObservation:
    size = screen_region.size if image_size is None else image_size
    return OcrObservation(
        observation_id=observation_id,
        image_path=Path(image_path),
        screen_region=screen_region,
        screen_size=screen_size,
        image_size=size,
        crop_origin=crop_origin,
        allowed_box=allowed_box if allowed_box is not None else (0, 0, *size),
        targets=tuple(
            OcrTarget(f"{observation_id}:text-{index}", element)
            for index, element in enumerate(elements, start=1)
        ),
    )


def _validate_confidence(value: float) -> None:
    if (
        not isinstance(value, Real) or isinstance(value, bool)
        or not isfinite(value) or not 0 <= value <= 1
    ):
        raise GroundingError("minimum confidence must be finite and between 0 and 1")


def resolve_target(
    observation: OcrObservation,
    target_id: str,
    target_text: str,
    *,
    min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
) -> tuple[int, int]:
    """Resolve a unique exact OCR match; an ID never overrides ambiguity."""
    _validate_confidence(min_confidence)
    matches = [
        target for target in observation.targets
        if _contains(observation.allowed_box, target.element.box)
        and _normalize(target.element.text) == _normalize(target_text)
    ]
    # Count even low-confidence duplicates, rather than silently guessing another ID.
    if len(matches) != 1:
        raise GroundingError(f"target must be unique in the allowed region; found {len(matches)}")
    target = matches[0]
    if target.target_id != target_id:
        raise GroundingError("unknown/stale target ID or target text mismatch")
    if target.element.confidence < min_confidence:
        raise GroundingError("target OCR confidence is below the safety threshold")
    x, y = target.element.center
    point = (
        x + observation.crop_origin[0] + observation.screen_region.left,
        y + observation.crop_origin[1] + observation.screen_region.top,
    )
    if not (
        0 <= point[0] < observation.screen_size[0]
        and 0 <= point[1] < observation.screen_size[1]
    ):
        raise GroundingError("resolved point is outside the primary screen")
    return point


@dataclass(frozen=True)
class GroundedAction:
    action: ClickAction | TypeAction | KeyAction | ScrollAction
    point: tuple[int, int] | None


def validate_decision_context(
    decision: ActionDecision, observation: OcrObservation, step_id: str,
) -> None:
    if decision.observation_id != observation.observation_id:
        raise GroundingError("decision belongs to a stale observation")
    if decision.step_id != step_id:
        raise GroundingError("decision belongs to a different plan step")
    for check in decision.checks:
        if check.region is not None and not _contains(observation.allowed_box, check.region):
            raise GroundingError("feedback region lies outside the allowed region")


def resolve_action(
    decision: ActionDecision,
    observation: OcrObservation,
    *,
    step_id: str,
    min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
    focus_verified: bool = False,
    focused_target_id: str | None = None,
    selection_verified: bool = False,
) -> GroundedAction:
    """Preview only. Window freshness, task authorization and execution belong to runtime."""
    decision = ActionDecision.model_validate(decision.model_dump())
    validate_decision_context(decision, observation, step_id)
    if decision.status != "action" or decision.action is None:
        raise GroundingError("only action decisions can be resolved")
    action = decision.action
    if isinstance(action, KeyAction):
        if focus_verified is not True:
            raise GroundingError("key action requires verified application focus")
        return GroundedAction(action, None)
    point = resolve_target(
        observation, action.target_id, action.target_text, min_confidence=min_confidence,
    )
    if isinstance(action, TypeAction):
        if focus_verified is not True or focused_target_id != action.target_id:
            raise GroundingError("type action requires independently verified input focus")
        if action.mode == "replace" and selection_verified is not True:
            raise GroundingError("replace requires a separately verified select-all micro-action")
    return GroundedAction(action, point)


def evaluate_text_check(
    check: FeedbackCheck,
    observation: OcrObservation,
    *,
    min_confidence: float = DEFAULT_MIN_OCR_CONFIDENCE,
) -> bool:
    """Check actual OCR in a bounded region, not the model's completion claim."""
    _validate_confidence(min_confidence)
    if check.region is None or check.text is None:
        raise GroundingError("window checks require independent runtime evidence")
    if not _contains(observation.allowed_box, check.region):
        raise GroundingError("feedback region lies outside the allowed region")
    left, top, right, bottom = check.region
    elements = [
        target.element for target in observation.targets
        if target.element.box[0] < right and target.element.box[2] > left
        and target.element.box[1] < bottom and target.element.box[3] > top
    ]
    # Partially intersecting OCR could hide old content at the check-region boundary.
    if any(not _contains(check.region, element.box) for element in elements):
        return False
    if check.type == "text_present":
        expected = _normalize(check.text)
        return any(
            element.confidence >= min_confidence and _normalize(element.text) == expected
            for element in elements
        )
    if not elements or any(element.confidence < min_confidence for element in elements):
        return False
    visible = " ".join(
        element.text
        for element in sorted(elements, key=lambda item: (item.box[1], item.box[0]))
    )
    return visible == check.text
