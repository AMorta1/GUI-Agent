"""Week 4 CLI and localhost-only search/message test fixtures."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui_agent.agent import GuiActionAgent, GuiPlanningAgent
from gui_agent.control import DesktopController
from gui_agent.grounding import FeedbackCheck
from gui_agent.models import OpenAICompatibleVisionClient, TransformersVisionClient
from gui_agent.perception import EasyOcrRecognizer
from gui_agent.runtime import DesktopObserver, GuiRuntime, RunLimits, WindowsWindowProbe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Observe/plan; execute only with explicit authorization")
    run.add_argument("--provider", choices=("api", "local"), required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--window-title", required=True, help="Exact foreground title after the countdown")
    run.add_argument("--checks-json", type=Path, required=True, help="Predefined final FeedbackCheck JSON array")
    run.add_argument("--output-dir", type=Path, required=True, help="New evidence directory outside the repository")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--supervised-input-focus", action="store_true", help="Controlled E2E only: confirm a prior Agent-clicked target's focus on matching append input; no selection proof")
    run.add_argument("--allow-screenshot-upload", action="store_true")
    run.add_argument("--start-delay", type=_positive_int, default=5)
    run.add_argument("--max-actions", type=_positive_int, default=20)
    run.add_argument("--max-step-actions", type=_positive_int, default=6)
    run.add_argument("--max-decisions", type=_positive_int, default=40)
    run.add_argument("--base-url", default=os.environ.get("GUI_AGENT_API_BASE"))
    run.add_argument("--model")
    run.add_argument("--api-key-env", default="GUI_AGENT_API_KEY")
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--device-map", choices=("cuda", "auto", "cpu"), default="cuda")
    run.add_argument("--offline", action="store_true")
    serve = commands.add_parser("serve", help="Serve test pages on 127.0.0.1 only; never launch a browser")
    serve.add_argument("--port", type=_port, default=8765)
    prepare = commands.add_parser("prepare", help="Create one harmless read-only-task input txt; no app launch")
    prepare.add_argument("--output-dir", type=Path, required=True)
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _port(value: str) -> int:
    number = _positive_int(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return number


def _external_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError("Evidence/fixture directories must be outside the code repository")
    return resolved


def _client(args: argparse.Namespace):
    if args.provider == "api":
        if not args.allow_screenshot_upload:
            raise ValueError("API mode requires --allow-screenshot-upload, including dry-run")
        model = args.model or os.environ.get("GUI_AGENT_API_MODEL")
        if not args.base_url or not model or not os.environ.get(args.api_key_env):
            raise ValueError("API endpoint/model/key environment variable is not configured")
        return OpenAICompatibleVisionClient(model, args.base_url, api_key_env=args.api_key_env, max_retries=0)
    if args.cache_dir is None:
        raise ValueError("Local mode requires --cache-dir")
    cache = args.cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HF_XET_CACHE"] = str(cache.parent / "xet")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return TransformersVisionClient(args.model or "Qwen/Qwen2.5-VL-3B-Instruct", cache, device_map=args.device_map)


class ResponseRecorder:
    """Retain requests and unparsed responses, never credentials or HTTP payloads."""

    def __init__(self, client, output_dir: Path) -> None:
        self.client = client
        self.output_dir = output_dir
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        record = {"request": request.model_dump(mode="json")}
        try:
            response = self.client.generate(request)
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "category": getattr(exc, "category", None)}
            raise
        else:
            record["response_before_parsing"] = response.model_dump(mode="json")
            return response
        finally:
            with (self.output_dir / f"model-{self.calls:02d}.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)


def _authorize(title: str, text: str, handle: int) -> bool:
    # A terminal prompt would steal application focus during real execution.
    api = ctypes.WinDLL("user32", use_last_error=True)
    function = api.MessageBoxW
    function.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
    function.restype = ctypes.c_int
    answer = function(handle, text, title, 0x00000004 | 0x00000100 | 0x00000020)
    time.sleep(0.3)
    return answer == 6


def run_cli(args: argparse.Namespace) -> int:
    if not args.task.strip() or not args.window_title.strip():
        raise ValueError("Task and window title must not be blank")
    if args.supervised_input_focus and not args.execute:
        raise ValueError("--supervised-input-focus requires --execute")
    raw_checks = json.loads(args.checks_json.read_text(encoding="utf-8"))
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError("Final checks must be a nonempty JSON array")
    checks = tuple(FeedbackCheck.model_validate(check) for check in raw_checks)
    output = _external_path(args.output_dir)
    if output.exists():
        raise FileExistsError("Output exists; prior evidence must not be overwritten")
    client = _client(args)
    print("WARNING: Captures/saves the bound window. API mode uploads its crops, even in dry-run.")
    if args.execute:
        print("EXECUTE: May move/click the mouse or send keys. Every action requires approval.")
    else:
        print("DRY-RUN: No mouse, keyboard or clipboard actions; model calls and screenshots still occur.")
    if args.supervised_input_focus:
        print("CONTROLLED E2E ONLY: Clicks create candidates, not focus proof. Matching append input requests require human focus confirmation; No stops the run.")
        print("No focus question after ordinary clicks. Action authorization dialogs may still dismiss menus; stale targets remain blocked.")
        print("Text may be typed; Unicode paste OVERWRITES the clipboard without restoring it. Replace remains blocked.")
    else:
        print("Strict input mode: text input is blocked.")
    print(f"Activate the exact target window within {args.start_delay} seconds; do not move it during the run.")
    time.sleep(args.start_delay)
    probe = WindowsWindowProbe()
    bound = probe.foreground()
    if bound is None or bound.title != args.window_title:
        raise ValueError("Foreground title does not match; stopped before screenshot/model request")
    output.mkdir(parents=True, exist_ok=False)
    recorder = ResponseRecorder(client, output)
    observer = DesktopObserver(bound, output / "observations", recognizer=EasyOcrRecognizer(gpu=False), window_probe=probe)
    runtime = GuiRuntime(
        GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, DesktopController(),
        limits=RunLimits(max_actions=args.max_actions, max_step_actions=args.max_step_actions, max_decisions=args.max_decisions),
    )
    result = runtime.run(
        args.task, checks, execute=args.execute,
        confirm_plan=lambda plan, final: _authorize(
            "Authorize plan and final checks",
            args.task + "\n\n" + plan.model_dump_json(indent=2) + "\n\nFinal checks:\n" + json.dumps([c.model_dump() for c in final], indent=2), bound.handle,
        ),
        confirm_action=lambda decision, grounded: _authorize(
            "Authorize one GUI action",
            args.task + "\n\n" + decision.model_dump_json(indent=2) + "\n\nComputed point: " + repr(grounded.point), bound.handle,
        ),
        supervised_input_focus=args.supervised_input_focus,
        confirm_input_focus=lambda grounded, scene: _authorize(
            "Confirm input focus only",
            "Before the proposed text input: is keyboard focus in the input field associated with the target previously clicked by the Agent?\n\n"
            + "Clicked text: " + grounded.action.target_text + "\nObservation: " + scene.observation.observation_id
            + "\n\nThe prior click is only a candidate, not focus proof."
            + "\nYes only if that input field has focus. No stops the run; choose No if uncertain or this is not an input."
            + "\nDo not click another target, supply coordinates, type, or change the plan."
            + "\nThis does NOT confirm selected text or task success.", bound.handle,
        ),
    )
    payload = result.to_dict()
    payload["model_calls"] = recorder.calls
    payload["supervised_input_focus"] = args.supervised_input_focus
    with (output / "run-summary.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: payload[key] for key in ("status", "reason", "mode", "executed_actions", "attempted_actions", "model_decisions", "model_calls")}, ensure_ascii=False, indent=2))
    print("output_directory=", output)
    if result.status == "completed":
        return 0
    if not args.execute and result.reason.startswith("dry_run:"):
        print("Preview only; this is not task completion.")
        return 0
    return 1


class FixtureState:
    """Test-server oracle only; never passed to the planning/execution agents."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.searches: list[str] = []
        self.messages: list[dict[str, str]] = []


def _page(title: str, body: str) -> bytes:
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>" + title +
            "</title><style>body{font:20px Arial;margin:32px;max-width:900px}input,button{font:20px Arial;padding:8px}"
            "label{display:block;margin:16px 0}section{margin-top:24px}</style><h1>" + title + "</h1>" + body + "</html>").encode("utf-8")


def make_fixture_server(port: int = 8765) -> ThreadingHTTPServer:
    state = FixtureState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def respond(self, status: int, content: bytes, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/search":
                query = parse_qs(url.query).get("q", [""])[0]
                if len(query) > 512:
                    self.respond(400, b"Query too long")
                    return
                if query:
                    with state.lock:
                        state.searches.append(query)
                value = html.escape(query, quote=True)
                result = "<section><h2>Search results</h2><p>Query: " + value + "</p><p>Local result: " + value + "</p></section>" if query else ""
                self.respond(200, _page("Week4 Search", "<form action='/search'><label>Search query <input name='q' value='" + value + "'></label><button>Search</button></form>" + result))
            elif url.path == "/messages":
                with state.lock:
                    messages = list(state.messages)
                records = "".join("<li>" + html.escape(m["recipient"]) + ": " + html.escape(m["body"]) + "</li>" for m in messages)
                status = "<p>Message sent</p>" if messages else "<p>No messages sent</p>"
                self.respond(200, _page("Week4 Messages", "<form method='post' action='/messages'><p>Recipient: Test Receiver</p><label>Message <input name='body' autocomplete='off'></label><button>Send</button></form><section><h2>Received messages</h2>" + status + "<ul>" + records + "</ul></section><form method='post' action='/reset'><button>Reset test state</button></form>"))
            elif url.path == "/records":
                with state.lock:
                    payload = {"searches": list(state.searches), "messages": list(state.messages)}
                self.respond(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            else:
                self.respond(404, b"Not found")

        def do_POST(self) -> None:
            # Reject browser cross-origin writes; loopback binding alone is not sufficient.
            authority = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != authority or self.headers.get("Origin") not in (None, "http://" + authority):
                self.respond(403, b"Local same-origin requests only")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= 8192:
                self.respond(413, b"Invalid body size")
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
                self.respond(415, b"Form data required")
                return
            try:
                fields = parse_qs(self.rfile.read(length).decode("utf-8"))
            except UnicodeDecodeError:
                self.respond(400, b"Invalid encoding")
                return
            path = urlsplit(self.path).path
            if path == "/messages":
                body = fields.get("body", [""])[0]
                if not body.strip() or len(body) > 512:
                    self.respond(400, b"Nonblank message of at most 512 characters required")
                    return
                with state.lock:
                    state.messages.append({"recipient": "Test Receiver", "body": body})
                destination = "/messages"
            elif path == "/reset":
                with state.lock:
                    state.searches.clear()
                    state.messages.clear()
                destination = "/messages"
            else:
                self.respond(404, b"Not found")
                return
            self.send_response(303)
            self.send_header("Location", destination)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "serve":
            with make_fixture_server(args.port) as server:
                print(f"Local test pages: http://127.0.0.1:{server.server_port}/search and /messages", flush=True)
                print("Manual oracle: /records. Reset with the page button or stop/restart this server. Ctrl+C stops.", flush=True)
                server.serve_forever()
            return 0
        if args.command == "prepare":
            output = _external_path(args.output_dir)
            output.mkdir(parents=True, exist_ok=False)
            target = output / "week4-sample.txt"
            target.write_text("Week4 file validation\nFILE TOKEN W4-OPEN-001\nRead-only task input; do not save changes.\n", encoding="utf-8")
            print("test_file=", target)
            return 0
        return run_cli(args)
    except KeyboardInterrupt:
        print("Cancelled; no automatic retry.")
        return 130
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
