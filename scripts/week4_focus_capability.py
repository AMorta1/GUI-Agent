"""Read-only UIA capability diagnostic. Never grants or executes GUI input."""

from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREP = ROOT.parent / ".test-tmp" / "week4-stage5-search-safety-prep-20261008"
HISTORY = ROOT.parent / ".test-tmp" / "week4-stage5-search-first-20261008"
URL = "http://127.0.0.1:8765/search"
LABEL = "Search query"

from gui_agent.grounding import GroundingError, resolve_target
from gui_agent.runtime import DesktopObserver


def normalized_ocr_text(text):
    return " ".join(text.split()).casefold()


def require_window(bound) -> None:
    if bound is None or "week4 search" not in bound.title.casefold() or "edge" not in bound.title.casefold():
        raise ValueError("Expected the dedicated Week4 Search Edge window")


def validate_diagnostic_profile(process: dict, pid: int, arguments: list[str]) -> dict:
    # Preserve the diagnostic's original profile checks, not the search execution policy.
    from scripts import week4_search_e2e as search
    if process.get("pid") != pid or normalized_ocr_text(str(process.get("exe", ""))) != normalized_ocr_text(str(search.EDGE)):
        raise ValueError("Bound process is not the expected Edge executable")
    profiles = []
    for index, value in enumerate(arguments):
        if value.startswith("--user-data-dir="):
            profiles.append(value.split("=", 1)[1])
        elif value == "--user-data-dir" and index + 1 < len(arguments):
            profiles.append(arguments[index + 1])
    if len(profiles) != 1 or Path(profiles[0]).resolve() != search.PROFILE.resolve():
        raise ValueError("Dedicated search test profile is not bound; personal/history profiles are prohibited")
    return {"pid": pid, "exe": process["exe"], "test_profile": str(search.PROFILE),
            "command_line": process["command_line"], "verified": True,
            "scope": "Read-only entry safety evidence; never supplied to Agent"}


def read_diagnostic_profile(pid: int) -> dict:
    from scripts import week4_search_e2e as search
    process = search.read_profile(pid)
    if not process.get("command_line"):
        raise ValueError("Process profile identity unavailable; no fallback permitted")
    return validate_diagnostic_profile(process, pid, search.windows_arguments(process["command_line"]))


class EvidenceObserver(DesktopObserver):
    def observe(self):
        scene = super().observe()
        with (self.output_dir / (scene.observation.observation_id + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(asdict(scene), stream, ensure_ascii=False, indent=2, default=path_json)
        return scene


def path_json(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unexpected evidence value: " + type(value).__name__)


def require_initial_scene(scene) -> None:
    if not scene.application_focused:
        raise GroundingError("Search test window is not focused")
    targets = scene.observation.targets
    if not any(normalized_ocr_text(t.element.text) == "127.0.0.1:8765/search" and t.element.confidence >= 0.5 for t in targets):
        raise GroundingError("Fixed local search address not independently visible in OCR")
    matches = [t for t in targets if normalized_ocr_text(t.element.text) == normalized_ocr_text(LABEL)]
    if len(matches) != 1:
        raise GroundingError("Search label is missing or ambiguous")
    resolve_target(scene.observation, matches[0].target_id, matches[0].element.text)
    from scripts import week4_search_e2e as search
    if any(search.TOKEN.casefold() in t.element.text.casefold() or normalized_ocr_text(t.element.text) == "search results" for t in targets):
        raise GroundingError("Expected an empty initial search page, not existing results")

# This worker only reads UIA properties. It has no UI interaction methods.
UIA_WORKER = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$expectedValue = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__EXPECTED_VALUE__'))
$allowedUrls = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__ALLOWED_URLS__')) | ConvertFrom-Json
$bound = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]__HANDLE__)
$walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
function In-Bound($element) {
    for ($i=0; $null -ne $element -and $i -lt 128; $i++) {
        if ([System.Windows.Automation.Automation]::Compare($bound, $element)) { return $true }
        $element = $walker.GetParent($element)
    }
    return $false
}
function Read-Element($element) {
    $c = $element.Current
    $r = $c.BoundingRectangle
    $value = $null
    $hasValue = $element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$value)
    $label = $c.LabeledBy
    $labelData = $null
    if ($null -ne $label) {
        $lc = $label.Current
        $lr = $lc.BoundingRectangle
        $labelData = @{runtime_id=@($label.GetRuntimeId()); name=$lc.Name;
            control_type=$lc.ControlType.ProgrammaticName;
            box=@($lr.Left,$lr.Top,$lr.Right,$lr.Bottom); in_bound=(In-Bound $label)}
    }
    $parents = @()
    $parent = $walker.GetParent($element)
    for ($i=0; $null -ne $parent -and $i -lt 12; $i++) {
        $pc = $parent.Current
        $parents += @{runtime_id=@($parent.GetRuntimeId()); name=$pc.Name;
            control_type=$pc.ControlType.ProgrammaticName; native_handle=$pc.NativeWindowHandle}
        if ([System.Windows.Automation.Automation]::Compare($bound,$parent)) { break }
        $parent = $walker.GetParent($parent)
    }
    return @{runtime_id=@($element.GetRuntimeId()); name=$c.Name;
        control_type=$c.ControlType.ProgrammaticName; process_id=$c.ProcessId;
        native_handle=$c.NativeWindowHandle; enabled=$c.IsEnabled; offscreen=$c.IsOffscreen;
        keyboard_focusable=$c.IsKeyboardFocusable; has_keyboard_focus=$c.HasKeyboardFocus;
        box=@($r.Left,$r.Top,$r.Right,$r.Bottom); in_bound=(In-Bound $element);
        value_pattern=$hasValue; value_read_only=$(if($hasValue){$value.Current.IsReadOnly}else{$null});
        value_empty=$(if($hasValue){$value.Current.Value -ceq ''}else{$null});
        value_matches_expected=$(if($hasValue){$value.Current.Value -ceq $expectedValue}else{$null});
        labeled_by=$labelData; ancestors=$parents}
}
$focused = [System.Windows.Automation.AutomationElement]::FocusedElement
$inBound = $null -ne $focused -and (In-Bound $focused)
$focusData = $null
if ($inBound) { $focusData = Read-Element $focused }
$edits = @()
$buttons = @()
$documents = @()
$nodes = $bound.FindAll([System.Windows.Automation.TreeScope]::Descendants,
    [System.Windows.Automation.Condition]::TrueCondition)
if ($nodes.Count -gt 1500) { throw 'Bound UIA tree exceeds diagnostic limit' }
for ($i=0; $i -lt $nodes.Count; $i++) {
    $node = $nodes.Item($i)
    $c = $node.Current
    if ($c.ControlType -eq [System.Windows.Automation.ControlType]::Edit) {
        $edits += Read-Element $node
    }
    if ($c.ControlType -eq [System.Windows.Automation.ControlType]::Button) {
        $buttons += Read-Element $node
    }
    if ($c.ControlType -eq [System.Windows.Automation.ControlType]::Document) {
        $v = $null
        $hasValue = $node.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern,[ref]$v)
        $documents += @{runtime_id=@($node.GetRuntimeId()); name=$c.Name;
            offscreen=$c.IsOffscreen;
            url_exact=$(if($hasValue){$v.Current.Value -ceq 'http://127.0.0.1:8765/search'}else{$false});
            url_allowed=$(if($hasValue){$allowedUrls -ccontains $v.Current.Value}else{$false});
            value_pattern=$hasValue}
    }
}
$focusedAgain = [System.Windows.Automation.AutomationElement]::FocusedElement
$stable = $null -ne $focused -and $null -ne $focusedAgain -and
    [System.Windows.Automation.Automation]::Compare($focused,$focusedAgain)
@{focused_present=($null -ne $focused); focused_in_bound=$inBound;
    focus_stable_during_read=$stable; focused=$focusData; edits=$edits; buttons=$buttons;
    documents=$documents; node_count=$nodes.Count;
    recorded_at=[DateTimeOffset]::Now.ToString('o')} | ConvertTo-Json -Depth 12 -Compress
"""


def classify(data: dict, *, require_empty: bool = True) -> dict:
    """Describe capability only; even a positive result is not input permission."""
    reason = None
    focus = data.get("focused")
    if data.get("focused_present") is not True or not isinstance(focus, dict):
        reason = "Focus evidence missing or outside bound window"
    elif data.get("focused_in_bound") is not True or focus.get("in_bound") is not True:
        reason = "Focused control does not belong to bound window"
    elif data.get("focus_stable_during_read") is not True:
        reason = "Focus changed during evidence collection"
    elif not valid_runtime_id(focus.get("runtime_id")):
        reason = "Actual FocusedElement identity is missing"
    elif data.get("initial_ocr_scope_verified") is not True:
        reason = "Initial local page and unique OCR label are unproven"
    elif (focus.get("control_type") != "ControlType.Edit"
          or focus.get("has_keyboard_focus") is not True
          or focus.get("enabled") is not True or focus.get("offscreen") is not False
          or focus.get("value_pattern") is not True
          or focus.get("value_read_only") is not False):
        reason = "Focused element is not a verified editable focused control"
    else:
        documents = [d for d in data.get("documents", [])
                     if d.get("name") == "Week4 Search" and d.get("offscreen") is False
                     and valid_runtime_id(d.get("runtime_id"))]
        document_ids = [d.get("runtime_id") for d in documents]
        if len(document_ids) != 1:
            return assessment("Visible test document is missing or not unique")
        matches = []
        for edit in data.get("edits", []):
            label = edit.get("labeled_by") or {}
            in_document = any(a.get("runtime_id") in document_ids for a in edit.get("ancestors", []))
            if in_document and label_matches(label):
                matches.append(edit)
        if len(matches) != 1:
            reason = "Explicit input-label association missing or not unique"
        elif (not valid_runtime_id(matches[0].get("runtime_id"))
              or matches[0]["runtime_id"] != focus["runtime_id"]):
            reason = "Focus is on another control, not the uniquely associated search input"
        elif (not label_matches(focus.get("labeled_by"))
              or focus["labeled_by"]["runtime_id"] != matches[0]["labeled_by"]["runtime_id"]
              or focus["labeled_by"].get("box") != matches[0]["labeled_by"].get("box")):
            reason = "FocusedElement label relationship changed during evidence collection"
        elif not label_contains_ocr_point(matches[0]["labeled_by"], data.get("initial_ocr_label")):
            reason = "OCR target/grounding point is missing, invalid or outside the explicit UIA label"
        elif require_empty and focus.get("value_empty") is not True:
            reason = "Search input is not independently verified empty"
    return assessment(reason)


def assessment(reason: str | None) -> dict:
    return {
        "uia_capability": "unproven" if reason else "explicit_label_link_observed",
        "reason": reason,
        "input_permission": False,
        "agent_click_binding_verified": False,
        "binding_note": "Read-only OCR/UIA association only; no Agent click or input-execution safety is verified",
    }


def valid_runtime_id(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(type(item) is int for item in value)


def label_matches(label: object) -> bool:
    return (isinstance(label, dict) and label.get("in_bound") is True
            and isinstance(label.get("name"), str) and label["name"].strip() == LABEL
            and valid_runtime_id(label.get("runtime_id")))


def finite_coordinates(value: object, length: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(item, Real) and not isinstance(item, bool) and isfinite(item) for item in value))


def label_contains_ocr_point(label: dict, ocr: dict | None) -> bool:
    if not isinstance(ocr, dict) or ocr.get("text") != LABEL:
        return False
    confidence = ocr.get("confidence")
    if (not isinstance(confidence, Real) or isinstance(confidence, bool)
            or not isfinite(confidence) or not 0.5 <= confidence <= 1):
        return False
    if "actual_ocr_text" in ocr and ocr["actual_ocr_text"] != LABEL:
        return False
    a, b = label.get("box"), ocr.get("screen_box")
    point = ocr.get("grounding_point_read_only")
    if not finite_coordinates(a, 4) or not finite_coordinates(b, 4) or not finite_coordinates(point, 2):
        return False
    # Validate the existing OCR point; UIA never supplies an execution coordinate.
    x, y = point
    return (a[0] <= x < a[2] and a[1] <= y < a[3]
            and b[0] <= x < b[2] and b[1] <= y < b[3])


def classify_saved_report(report: dict) -> dict:
    """Validate recorded scope, not current desktop identity or observation freshness."""
    window = report.get("bound_window") or {}
    profile = report.get("bound_profile") or {}
    reason = None
    if (type(window.get("handle")) is not int or window["handle"] <= 0
            or type(window.get("pid")) is not int or window["pid"] <= 0
            or not isinstance(window.get("title"), str)
            or "week4 search" not in window["title"].casefold()
            or "edge" not in window["title"].casefold()
            or not finite_coordinates(window.get("box"), 4)
            or window["box"][0] >= window["box"][2] or window["box"][1] >= window["box"][3]):
        reason = "Recorded bound window identity/range is missing or invalid"
    elif (profile.get("verified") is not True or type(profile.get("pid")) is not int
          or profile["pid"] != window["pid"]
          or PureWindowsPath(str(profile.get("test_profile", ""))) != PureWindowsPath(PREP / "edge-search-test-profile")
          or PureWindowsPath(str(profile.get("exe", ""))) != PureWindowsPath(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")):
        reason = "Recorded isolated profile identity is missing or inconsistent"
    elif report.get("status") != "evidence_collected_not_input_authorized" or report.get("error") is not None:
        reason = "Original diagnostic did not finish collecting evidence"
    elif not report.get("snapshots"):
        reason = "Saved UIA snapshots are missing"
    results = []
    document_ids = None
    for snapshot in report.get("snapshots", []):
        data = snapshot.get("uia") or {}
        item_reason = reason
        if item_reason is None and data.get("initial_ocr_label") != report.get("initial_ocr_label"):
            item_reason = "Snapshot OCR target differs from the original bound OCR target"
        current_ids = [d.get("runtime_id") for d in data.get("documents", [])
                       if d.get("name") == "Week4 Search" and d.get("offscreen") is False]
        if document_ids is None:
            document_ids = current_ids
        elif current_ids != document_ids:
            item_reason = item_reason or "Recorded document identity changed between snapshots"
        results.append({"phase": snapshot.get("phase"), "original_assessment": snapshot.get("assessment"),
                        "assessment": assessment(item_reason) if item_reason else classify(data)})
    return {"scope": "Offline interpretation of saved UIA/OCR evidence; not a fresh execution check",
            "recorded_scope_validated": reason is None, "scope_error": reason,
            "snapshots": results, "input_permission": False,
            "agent_click_binding_verified": False, "real_input_safety_verified": False}


def analyze_offline(source: Path, output: Path) -> int:
    raw = source.read_bytes()
    report = classify_saved_report(json.loads(raw.decode("utf-8")))
    report.update(source_file=str(source.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                  api_calls=0, uia_sample_calls=0, capture_calls=0, ocr_calls=0,
                  automated_desktop_actions=0, preauthorization_mode_implemented=False)
    with (output / "offline-assessment.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for snapshot in report["snapshots"]:
        print(f"{snapshot['phase']}: capability={snapshot['assessment']['uia_capability']} input_permission=False")
    print("OFFLINE ONLY: no sample/API/actions; real Agent click/input safety remains unverified.")
    print("evidence=" + str(output / "offline-assessment.json"))
    return 0 if report["recorded_scope_validated"] else 1


def read_uia(handle: int, *, expected_value: str = "", allowed_urls: tuple[str, ...] = (URL,)) -> dict:
    if type(handle) is not int or handle <= 0 or not isinstance(expected_value, str):
        raise ValueError("A valid bound HWND and expected string value are required")
    source = UIA_WORKER.replace("__HANDLE__", str(handle))
    source = source.replace("__EXPECTED_VALUE__", base64.b64encode(expected_value.encode("utf-8")).decode("ascii"))
    source = source.replace("__ALLOWED_URLS__", base64.b64encode(json.dumps(allowed_urls).encode("utf-8")).decode("ascii"))
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    worker = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Mta", "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if worker.returncode:
        raise RuntimeError("Read-only UIA worker failed: " + worker.stderr.strip()[:2000])
    return json.loads(worker.stdout.lstrip("\ufeff"))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Local records redirect prohibited")


def read_records() -> dict:
    with build_opener(ProxyHandler({}), NoRedirect()).open("http://127.0.0.1:8765/records", timeout=3) as response:
        if response.geturl() != "http://127.0.0.1:8765/records":
            raise RuntimeError("Records request redirected")
        result = json.load(response)
    if (not isinstance(result, dict) or set(result) != {"searches", "messages"}
            or any(not isinstance(result[key], list) for key in result)):
        raise RuntimeError("Unexpected local records schema")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--saved-evidence", type=Path, help="Offline JSON interpretation only; never samples the desktop")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.saved_evidence is not None:
        return analyze_offline(args.saved_evidence, args.output_dir)
    report = {
        "scope": "Read-only UIA capability diagnostic; NOT Agent/task E2E or input permission",
        "started_at": datetime.now().astimezone().isoformat(),
        "api_calls": 0, "automated_desktop_actions": 0, "browser_launches": 0,
        "input_permission": False, "snapshots": [], "error": None,
        "capture_calls": 0, "ocr_calls": 0,
    }
    try:
        from gui_agent.runtime import WindowsWindowProbe
        from scripts import week4_search_e2e as config
        report["profile"] = str(config.PROFILE)
        history = json.loads((HISTORY / "run-summary.json").read_text(encoding="utf-8"))
        report["historical_search_executed_actions"] = history["executed_actions"]
        report["records_before"] = read_records()
        if report["records_before"] != {"searches": [], "messages": []}:
            raise RuntimeError("Local records are not empty; no reset permitted")
        print("READ-ONLY UIA ONLY: no API, launch, click, typing, submission or focus restoration.", flush=True)
        print("Use ONLY the already-open isolated search profile and empty localhost /search page.", flush=True)
        print("Manual focus preparation is NOT an Agent click. Do not type, navigate or submit.", flush=True)
        print("Loading existing CPU OCR; detailed logs stay in diagnostic-console.txt.", flush=True)
        with (args.output_dir / "diagnostic-console.txt").open("x", encoding="utf-8") as console:
            with redirect_stdout(console), redirect_stderr(console):
                import easyocr
                recognizer = config.demo.EasyOcrRecognizer(
                    reader=easyocr.Reader(["ch_sim", "en"], gpu=False, download_enabled=False))
        probe = WindowsWindowProbe()
        bound = None
        observer = None
        ocr_label = None
        document_ids = None
        phases = [
            ("search_input", "Click ONLY the empty search input. Do not type."),
            ("address_bar", "Click ONLY the address bar. Do not type or press Enter."),
            ("other_control", "From the empty input, use Tab to focus Search button. Do NOT click it or press Enter/Space."),
        ]
        for phase, instructions in phases:
            input(f"{phase}: Enter here, then within 8 seconds: {instructions} Stay until THREE tones: ")
            time.sleep(8)
            window = probe.foreground()
            require_window(window)
            identity = read_diagnostic_profile(window.pid)
            if bound is None:
                bound = window
                report["bound_window"] = asdict(bound)
                report["bound_profile"] = identity
                report["initial_uia_before_ocr"] = read_uia(bound.handle)
                if probe.foreground() != bound or probe.read(bound.handle) != bound:
                    raise RuntimeError("Bound window changed or lost foreground during initial UIA read")
                observer = EvidenceObserver(bound, args.output_dir / "observations",
                                            recognizer=recognizer, window_probe=probe)
                with (args.output_dir / "diagnostic-console.txt").open("a", encoding="utf-8") as console:
                    with redirect_stdout(console), redirect_stderr(console):
                        scene = observer.observe()
                report["capture_calls"] = report["ocr_calls"] = 1
                require_initial_scene(scene)
                target = next(t for t in scene.observation.targets if normalized_ocr_text(t.element.text) == normalized_ocr_text(LABEL))
                point = resolve_target(scene.observation, target.target_id, target.element.text)
                x, y = scene.observation.crop_origin
                a, b, c, d = target.element.box
                ocr_label = {"observation_id": scene.observation.observation_id,
                             "target_id": target.target_id, "text": LABEL,
                             "actual_ocr_text": target.element.text, "confidence": target.element.confidence,
                             "screen_box": [a + x, b + y, c + x, d + y],
                             "grounding_point_read_only": point, "agent_click": False}
                report["initial_ocr_label"] = ocr_label
            elif window != bound:
                raise RuntimeError("Bound window identity/title/rectangle changed")
            data = read_uia(bound.handle)
            if probe.foreground() != bound or probe.read(bound.handle) != bound:
                raise RuntimeError("Bound window changed or lost foreground during read")
            current_documents = [d["runtime_id"] for d in data.get("documents", [])
                                 if d.get("name") == "Week4 Search" and d.get("offscreen") is False]
            if document_ids is None:
                document_ids = current_documents
            elif current_documents != document_ids:
                raise RuntimeError("UIA document identity changed between diagnostic snapshots")
            data["initial_ocr_scope_verified"] = True
            data["initial_ocr_label"] = ocr_label
            verdict = classify(data)
            report["snapshots"].append({"phase": phase, "uia": data, "assessment": verdict})
            print(f"{phase}: capability={verdict['uia_capability']} input_permission=False", flush=True)
            if len(current_documents) != 1:
                raise RuntimeError("Visible test document is missing or not unique; stop further probing")
            if any(e.get("name") == LABEL and e.get("value_empty") is not True for e in data.get("edits", [])):
                raise RuntimeError("Search input empty state changed or unavailable")
            if phase != "other_control":
                signal_done()
        report["status"] = "evidence_collected_not_input_authorized"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "diagnostic_blocked"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            report["records_after"] = read_records()
            before = report.get("records_before")
            report["records_unchanged"] = None if before is None else before == report["records_after"]
            if report["records_unchanged"] is False:
                report["status"] = "diagnostic_blocked"
                report["error"] = "Independent service records changed"
        except Exception as error:
            report["records_unchanged"] = None
            report["records_check_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "diagnostic_blocked"
        with (args.output_dir / "uia-capability.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        signal_done()
    print(f"status={report['status']} input_permission=False", flush=True)
    if report["error"]:
        print("reason=" + report["error"][:240], flush=True)
    print("evidence=" + str(args.output_dir / "uia-capability.json"), flush=True)
    print("STOP: no E2E, no input grant, no automatic repeat.", flush=True)
    return 0 if report["status"] == "evidence_collected_not_input_authorized" else 1


def signal_done() -> None:
    import winsound
    for _ in range(3):
        winsound.Beep(880, 100)
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
