from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
from pydantic import ValidationError
import pytest

from gui_agent.models import (
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
    image.write_bytes(b"image")
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


def test_api_encodes_local_image_as_data_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"fake-png-content")
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
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == b"fake-png-content"


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
