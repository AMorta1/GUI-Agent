from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
from PIL import Image
from pydantic import ValidationError
import pytest

from gui_agent.models import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    ModelConfigurationError,
    ModelInvocationError,
    MultimodalModelClient,
    MultimodalRequest,
    OpenAICompatibleVisionClient,
    TransformersVisionClient,
)


def test_request_rejects_blank_instruction() -> None:
    with pytest.raises(ValidationError, match="blank"):
        MultimodalRequest(instruction="   ")


def test_local_client_does_not_load_model_during_construction(tmp_path: Path) -> None:
    client = TransformersVisionClient("model/name", tmp_path / "cache")

    assert client.is_loaded is False
    assert not (tmp_path / "cache").exists()
    assert isinstance(client, MultimodalModelClient)


def test_local_client_checks_image_before_loading(tmp_path: Path) -> None:
    client = TransformersVisionClient("model/name", tmp_path / "cache")

    with pytest.raises(ModelConfigurationError, match="Image file not found"):
        client.generate(
            MultimodalRequest(
                instruction="Describe this image",
                image_path=tmp_path / "missing.png",
            )
        )

    assert client.is_loaded is False


def test_local_messages_keep_windows_compatible_native_image_path(
    tmp_path: Path,
) -> None:
    image = tmp_path / "screen.png"
    Image.new("RGB", (4, 3)).save(image)
    request = MultimodalRequest(instruction="Describe", image_path=image)

    messages = TransformersVisionClient._messages(request, str(image.resolve()))

    assert messages[0]["content"][0] == {
        "type": "image",
        "image": str(image.resolve()),
    }


def test_api_text_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "  A valid plan  "}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1/",
        api_key_env="TEST_API_KEY",
        transport=httpx.MockTransport(handler),
    )

    response = client.generate(
        MultimodalRequest(
            instruction="Plan the task",
            system_prompt="Return JSON",
            max_new_tokens=80,
        )
    )

    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": "Return JSON",
    }
    assert captured["payload"]["max_tokens"] == 80
    assert "temperature" not in captured["payload"]
    assert response.text == "A valid plan"
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 4}


@pytest.mark.parametrize(("suffix", "mime"), [("png", "image/png"), ("jpg", "image/jpeg")])
def test_api_encodes_local_image_as_data_url(
    suffix: str,
    mime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / f"sample.{suffix}"
    Image.new("RGB", (4, 3), "white").save(image)
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1",
        api_key_env="TEST_API_KEY",
        transport=httpx.MockTransport(handler),
    )

    client.generate(MultimodalRequest(instruction="Describe", image_path=image))

    content = payloads[0]["messages"][0]["content"]
    data_url = content[1]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == f"data:{mime};base64"
    assert base64.b64decode(encoded) == image.read_bytes()


def test_api_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1",
        api_key_env="MISSING_API_KEY",
    )

    with pytest.raises(ModelConfigurationError, match="MISSING_API_KEY"):
        client.generate(MultimodalRequest(instruction="Plan"))


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "authentication"), (429, "rate limit"), (503, "unavailable")],
)
def test_api_reports_actionable_http_errors(
    status: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1",
        api_key_env="TEST_API_KEY",
        transport=httpx.MockTransport(lambda request: httpx.Response(status)),
    )

    with pytest.raises(ModelInvocationError, match=message):
        client.generate(MultimodalRequest(instruction="Plan"))


def test_api_rejects_response_without_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1",
        api_key_env="TEST_API_KEY",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": []})
        ),
    )

    with pytest.raises(ModelInvocationError, match="assistant text"):
        client.generate(MultimodalRequest(instruction="Plan"))


def test_api_rejects_missing_image_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model",
        "https://example.test/v1",
        api_key_env="TEST_API_KEY",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("HTTP request should not be sent")
        ),
    )

    with pytest.raises(ModelConfigurationError, match="Image file not found"):
        client.generate(
            MultimodalRequest(
                instruction="Describe",
                image_path=tmp_path / "missing.png",
            )
        )


@pytest.mark.parametrize("provider", ["local", "api"])
def test_clients_share_project_image_defaults(provider: str, tmp_path: Path) -> None:
    client = (
        TransformersVisionClient("model/name", tmp_path / "cache")
        if provider == "local"
        else OpenAICompatibleVisionClient("vision-model", "https://example.test/v1")
    )
    assert client.max_image_bytes == DEFAULT_MAX_IMAGE_BYTES
    assert client.max_image_pixels == DEFAULT_MAX_IMAGE_PIXELS


@pytest.mark.parametrize("provider", ["local", "api"])
@pytest.mark.parametrize("parameter", ["max_image_bytes", "max_image_pixels"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_clients_reject_invalid_image_limits(
    provider: str, parameter: str, value: object, tmp_path: Path
) -> None:
    options = {parameter: value}
    with pytest.raises(ValueError, match=parameter):
        if provider == "local":
            TransformersVisionClient("model/name", tmp_path / "cache", **options)
        else:
            OpenAICompatibleVisionClient("vision-model", "https://example.test/v1", **options)


@pytest.mark.parametrize("provider", ["local", "api"])
@pytest.mark.parametrize(
    "case", ["missing", "directory", "empty", "corrupt", "unknown", "mismatch"]
)
def test_invalid_images_fail_before_model_load_or_http(
    provider: str, case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "sample.png"
    if case == "directory":
        image.mkdir()
    elif case == "empty":
        image.touch()
    elif case == "corrupt":
        image.write_bytes(b"not a PNG")
    elif case == "unknown":
        image = tmp_path / "sample.bin"
        Image.new("RGB", (4, 3)).save(image, format="PNG")
    elif case == "mismatch":
        Image.new("RGB", (4, 3)).save(image, format="JPEG")

    monkeypatch.setenv("TEST_API_KEY", "secret")
    if provider == "local":
        client = TransformersVisionClient("model/name", tmp_path / "cache")
        monkeypatch.setattr(client, "_ensure_loaded", lambda: pytest.fail("Model must not load"))
    else:
        client = OpenAICompatibleVisionClient(
            "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
            max_retries=2,
            transport=httpx.MockTransport(lambda request: pytest.fail("HTTP must not run")),
        )
    with pytest.raises(ModelConfigurationError) as caught:
        client.generate(MultimodalRequest(instruction="Describe", image_path=image))
    assert caught.value.category == "input"
    assert caught.value.retryable is False


class _ModelLoadingReached(Exception):
    pass


@pytest.mark.parametrize("provider", ["local", "api"])
@pytest.mark.parametrize("parameter", ["max_image_bytes", "max_image_pixels"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_image_limit_is_inclusive_and_configurable(
    provider: str, parameter: str, over_limit: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (4, 3)).save(image)
    exact_limit = image.stat().st_size if parameter == "max_image_bytes" else 12
    options = {parameter: exact_limit - int(over_limit)}
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    def load_marker() -> None:
        raise _ModelLoadingReached

    monkeypatch.setenv("TEST_API_KEY", "secret")
    if provider == "local":
        client = TransformersVisionClient("model/name", tmp_path / "cache", **options)
        monkeypatch.setattr(client, "_ensure_loaded", load_marker)
    else:
        client = OpenAICompatibleVisionClient(
            "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
            transport=httpx.MockTransport(handler), **options,
        )
    request = MultimodalRequest(instruction="Describe", image_path=image)
    if over_limit:
        with pytest.raises(ModelConfigurationError, match=parameter) as caught:
            client.generate(request)
        assert caught.value.category == "input"
        assert not calls
    elif provider == "local":
        with pytest.raises(_ModelLoadingReached):
            client.generate(request)
    else:
        assert client.generate(request).text == "ok"
        assert len(calls) == 1


@pytest.mark.parametrize("failure", ["connection", "connect_timeout", 429, 502, 503, 504])
def test_api_retries_transient_failure_then_succeeds(
    failure: str | int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            if failure == "connection":
                raise httpx.ConnectError("offline", request=request)
            if failure == "connect_timeout":
                raise httpx.ConnectTimeout("late", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr("gui_agent.models.time.sleep", sleeps.append)
    client = OpenAICompatibleVisionClient(
        "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
        max_retries=2, transport=httpx.MockTransport(handler),
    )
    assert client.generate(MultimodalRequest(instruction="Plan")).text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]
    assert calls[0].content == calls[1].content


@pytest.mark.parametrize("max_retries", [0, 2])
def test_api_stops_at_retry_limit(max_retries: int, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr("gui_agent.models.time.sleep", sleeps.append)
    client = OpenAICompatibleVisionClient(
        "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
        max_retries=max_retries, transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelInvocationError) as caught:
        client.generate(MultimodalRequest(instruction="Plan"))
    assert len(calls) == max_retries + 1
    assert len(sleeps) == max_retries
    assert caught.value.category == "provider_unavailable"
    assert caught.value.status_code == 503
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    ("failure", "category", "retryable", "expected_calls"),
    [
        (401, "authentication", False, 1), (403, "authentication", False, 1),
        (400, "request_rejected", False, 1), (429, "rate_limit", True, 3),
        (500, "provider_unavailable", False, 1), (503, "provider_unavailable", True, 3),
        (httpx.ConnectError, "connection", True, 3),
        (httpx.ConnectTimeout, "timeout", True, 3),
        (httpx.ReadTimeout, "timeout", False, 1),
        (httpx.WriteTimeout, "timeout", False, 1),
        (httpx.PoolTimeout, "timeout", False, 1),
    ],
)
def test_api_classifies_failures_and_applies_retry_policy(
    failure: object, category: str, retryable: bool, expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.extensions["timeout"] == {
            "connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0,
        }
        if isinstance(failure, int):
            return httpx.Response(failure)
        raise failure("mock failure", request=request)

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr("gui_agent.models.time.sleep", lambda seconds: None)
    client = OpenAICompatibleVisionClient(
        "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
        timeout=7.0, max_retries=2, transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelInvocationError) as caught:
        client.generate(MultimodalRequest(instruction="Plan"))
    assert caught.value.category == category
    assert caught.value.retryable is retryable
    assert caught.value.status_code == (failure if isinstance(failure, int) else None)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "body", [None, [], {}, {"choices": []}, {"choices": [{"message": {"content": " "}}]},
             {"choices": [{"message": {"content": "ok"}}], "usage": "invalid"}],
)
def test_api_invalid_responses_are_not_retried(
    body: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if body is None:
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(200, json=body)

    monkeypatch.setenv("TEST_API_KEY", "secret")
    client = OpenAICompatibleVisionClient(
        "vision-model", "https://example.test/v1", api_key_env="TEST_API_KEY",
        max_retries=2, transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelInvocationError) as caught:
        client.generate(MultimodalRequest(instruction="Plan"))
    assert caught.value.category == "invalid_response"
    assert len(calls) == 1


@pytest.mark.parametrize("max_retries", [-1, True, 1.5])
def test_api_rejects_invalid_retry_count(max_retries: object) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        OpenAICompatibleVisionClient(
            "vision-model", "https://example.test/v1", max_retries=max_retries
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_api_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleVisionClient("vision-model", "https://example.test/v1", timeout=timeout)
