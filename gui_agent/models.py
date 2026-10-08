"""Unified local and HTTP multimodal model clients."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, runtime_checkable

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Project input safety defaults, not official model-provider limits.
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 20_000_000


class ModelClientError(RuntimeError):
    """Base error for model configuration and invocation failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "client_error",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


class ModelConfigurationError(ModelClientError):
    """Raised when required local or API configuration is unavailable."""


class ModelInvocationError(ModelClientError):
    """Raised when a configured provider cannot produce a valid response."""


class MultimodalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    image_path: Path | None = None
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @field_validator("instruction", "system_prompt")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text must not be blank")
        return value.strip() if value is not None else None


class MultimodalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MultimodalModelClient(Protocol):
    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        """Generate one response for a text-only or text-and-image request."""


@dataclass(frozen=True)
class _LocalComponents:
    model: Any
    processor: Any


class TransformersVisionClient:
    """Lazily load Qwen2.5-VL through Transformers for local inference."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str | Path,
        *,
        device_map: str | int | Mapping[str, Any] = "cuda",
        max_memory: Mapping[Any, str] | None = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("pixel limits must be positive and ordered")
        _validate_image_limits(max_image_bytes, max_image_pixels)
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.device_map = device_map
        self.max_memory = dict(max_memory) if max_memory is not None else None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_image_bytes = max_image_bytes
        self.max_image_pixels = max_image_pixels
        self._components: _LocalComponents | None = None

    @property
    def is_loaded(self) -> bool:
        return self._components is not None

    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        image_reference: str | None = None
        if request.image_path is not None:
            image_path = _validate_image(
                request.image_path,
                max_image_bytes=self.max_image_bytes,
                max_image_pixels=self.max_image_pixels,
            )
            image_reference = str(image_path)

        components = self._ensure_loaded()
        messages = self._messages(request, image_reference)
        try:
            from qwen_vl_utils import process_vision_info

            prompt = components.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = components.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(components.model.device)
            generation_options: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                generation_options["temperature"] = request.temperature
            started_at = time.perf_counter()
            generated_ids = components.model.generate(**inputs, **generation_options)
            elapsed = time.perf_counter() - started_at
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            text = components.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except ModelClientError:
            raise
        except Exception as exc:
            raise ModelInvocationError(
                f"Local model generation failed: {exc}", category="local_generation"
            ) from exc
        if not text:
            raise ModelInvocationError(
                "Local model returned empty text", category="invalid_response"
            )
        return MultimodalResponse(
            text=text,
            provider="transformers",
            model=self.model_id,
            usage={"input_tokens": int(inputs.input_ids.shape[-1])},
            metadata={
                "device": str(components.model.device),
                "device_map": str(self.device_map),
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    def _ensure_loaded(self) -> _LocalComponents:
        if self._components is not None:
            return self._components
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            model_options: dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "device_map": self.device_map,
                "dtype": "auto",
                "low_cpu_mem_usage": True,
            }
            if self.max_memory is not None:
                model_options["max_memory"] = self.max_memory
            if self.device_map == "auto":
                offload_dir = self.cache_dir.parent / "offload"
                offload_dir.mkdir(parents=True, exist_ok=True)
                model_options["offload_folder"] = offload_dir
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_options,
            )
        except Exception as exc:
            raise ModelConfigurationError(
                f"Local model loading failed: {exc}", category="configuration"
            ) from exc
        self._components = _LocalComponents(model=model, processor=processor)
        return self._components

    @staticmethod
    def _messages(
        request: MultimodalRequest,
        image_reference: str | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append(
                {"role": "system", "content": request.system_prompt}
            )
        content: list[dict[str, Any]] = []
        if image_reference is not None:
            content.append({"type": "image", "image": image_reference})
        content.append({"type": "text", "text": request.instruction})
        messages.append({"role": "user", "content": content})
        return messages


class OpenAICompatibleVisionClient:
    """Call a Chat Completions-compatible multimodal HTTP endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 0,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if not base_url.strip():
            raise ValueError("base_url must not be blank")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        _validate_image_limits(max_image_bytes, max_image_pixels)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.transport = transport
        self.max_retries = max_retries
        self.max_image_bytes = max_image_bytes
        self.max_image_pixels = max_image_pixels

    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ModelConfigurationError(
                f"API key environment variable is not set: {self.api_key_env}",
                category="configuration",
            )
        payload = self._payload(request)
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    body = response.json()
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    error = self._request_error(exc)
                    if not error.retryable or attempt == self.max_retries:
                        raise error from exc
                    time.sleep(0.5 * (attempt + 1))
        try:
            text = body["choices"][0]["message"]["content"].strip()
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ModelInvocationError(
                "API response did not contain assistant text", category="invalid_response"
            ) from exc
        if not text:
            raise ModelInvocationError(
                "API response contained empty assistant text", category="invalid_response"
            )
        usage_body = body.get("usage") or {}
        if not isinstance(usage_body, dict):
            raise ModelInvocationError(
                "API response contained invalid usage", category="invalid_response"
            )
        usage = {
            key: int(value)
            for key, value in usage_body.items()
            if isinstance(value, int)
        }
        metadata = {"base_url": self.base_url}
        if "finish_reason" in body["choices"][0]:
            metadata["finish_reason"] = body["choices"][0]["finish_reason"]
        return MultimodalResponse(
            text=text,
            provider="openai-compatible",
            model=self.model,
            usage=usage,
            metadata=metadata,
        )

    def _payload(self, request: MultimodalRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        content: list[dict[str, Any]] = [
            {"type": "text", "text": request.instruction}
        ]
        if request.image_path is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(
                            request.image_path,
                            max_image_bytes=self.max_image_bytes,
                            max_image_pixels=self.max_image_pixels,
                        )
                    },
                }
            )
        messages.append({"role": "user", "content": content})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_new_tokens,
        }
        if request.temperature > 0:
            payload["temperature"] = request.temperature
        return payload

    @staticmethod
    def _request_error(exc: Exception) -> ModelInvocationError:
        status: int | None = None
        retryable = False
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                category = "authentication"
            elif status == 429:
                category = "rate_limit"
            elif status >= 500:
                category = "provider_unavailable"
            else:
                category = "request_rejected"
            retryable = status in (429, 502, 503, 504)
            message = OpenAICompatibleVisionClient._http_error_message(exc.response)
        elif isinstance(exc, httpx.TimeoutException):
            category = "timeout"
            retryable = isinstance(exc, httpx.ConnectTimeout)
            message = f"API request timed out ({type(exc).__name__})"
        elif isinstance(exc, httpx.ConnectError):
            category, retryable = "connection", True
            message = "API connection failed"
        elif isinstance(exc, httpx.HTTPError):
            category = "connection"
            message = f"API transport failed ({type(exc).__name__})"
        else:
            category = "invalid_response"
            message = "API response was not valid JSON"
        return ModelInvocationError(
            message, category=category, retryable=retryable, status_code=status
        )

    @staticmethod
    def _http_error_message(response: httpx.Response) -> str:
        status = response.status_code
        if status in (401, 403):
            return f"API authentication failed (HTTP {status})"
        if status == 429:
            return "API rate limit exceeded (HTTP 429)"
        if status >= 500:
            return f"API provider unavailable (HTTP {status})"
        return f"API request rejected (HTTP {status})"


def _validate_image_limits(max_image_bytes: int, max_image_pixels: int) -> None:
    for name, value in (
        ("max_image_bytes", max_image_bytes),
        ("max_image_pixels", max_image_pixels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_image(
    path: str | Path, *, max_image_bytes: int, max_image_pixels: int
) -> Path:
    image_path = Path(path).expanduser().resolve()
    try:
        if not image_path.is_file():
            raise ValueError(f"Image file not found: {image_path}")
        size = image_path.stat().st_size
        if size == 0:
            raise ValueError("Image file is empty")
        if size > max_image_bytes:
            raise ValueError("Image file exceeds max_image_bytes")
        expected_format = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(
            image_path.suffix.lower()
        )
        if expected_format is None:
            raise ValueError("Unsupported image type; use PNG or JPEG")
        with Image.open(image_path) as image:
            if image.format != expected_format:
                raise ValueError("Image format does not match its file extension")
            if image.width * image.height > max_image_pixels:
                raise ValueError("Image exceeds max_image_pixels")
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ModelConfigurationError(
            f"Invalid image input: {exc}", category="input"
        ) from exc
    return image_path


def _image_data_url(
    path: str | Path, *, max_image_bytes: int, max_image_pixels: int
) -> str:
    image_path = _validate_image(
        path, max_image_bytes=max_image_bytes, max_image_pixels=max_image_pixels
    )
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
