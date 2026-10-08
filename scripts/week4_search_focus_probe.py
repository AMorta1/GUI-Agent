"""One no-API desktop focus probe: one click, mandatory veto before real input."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui_agent.agent import GuiActionAgent, GuiPlanningAgent, TaskPlan
from gui_agent.grounding import ClickAction, GroundingError, TypeAction
from gui_agent.models import MultimodalResponse
from gui_agent.runtime import GuiRuntime, WindowsWindowProbe
from scripts import week4_search_e2e as search


OUTPUT = ROOT.parent / ".test-tmp" / "week4-search-focus-probe-once-20261008"
TASK = "Validate the local search focus chain with one OCR-grounded click and a mandatory write veto; never type or submit."
VETO_REASON = "PRESET_PROBE_WRITE_VETO: real keyboard input is not authorized"
EMPTY_RECORDS = {"searches": [], "messages": []}


def save(output: Path, name: str, data) -> None:
    with (output / name).open("x", encoding="utf-8") as stream:
        json.dump(search.log_value(data), stream, ensure_ascii=False, indent=2, allow_nan=False)


class PresetWriteVeto(GroundingError):
    pass


class VetoBackend:
    """Only size and one left click can reach PyAutoGUI; never forwards write."""

    def __init__(self, raw, log):
        self._raw, self.log = raw, log
        self.click_calls = 0
        self.successful_clicks = 0
        self.write_veto_calls = 0

    @property
    def FAILSAFE(self):
        return self._raw.FAILSAFE

    @FAILSAFE.setter
    def FAILSAFE(self, value):
        if value is not True:
            raise GroundingError("Probe must retain PyAutoGUI FAILSAFE")
        self._raw.FAILSAFE = value

    def size(self):
        return self._raw.size()

    def click(self, *, x, y, button, duration):
        if self.click_calls != 0 or button != "left":
            raise GroundingError("Probe permits at most one actual left-click call")
        self.click_calls += 1
        self.log.emit("probe_click_call", point=(x, y), actual_backend="pyautogui", call_number=self.click_calls)
        self._raw.click(x=x, y=y, button=button, duration=duration)
        self.successful_clicks += 1
        self.log.emit("probe_click_returned", successful_clicks=self.successful_clicks)

    def write(self, text, *, interval):
        # Deliberately no reference to _raw.write: this veto is independent of UIA.
        self.write_veto_calls += 1
        self.log.emit("probe_write_intercepted", source="preset_probe_backend_veto", human_answer=None,
                      real_write_calls=0, attempt=self.write_veto_calls,
                      text_matches_frozen_word=text == search.TOKEN, interval=interval)
        raise PresetWriteVeto(VETO_REASON)

    def forbidden(self, *args, **kwargs):
        raise GroundingError("Probe forbids keyboard, submission, navigation and other desktop actions")

    press = hotkey = scroll = moveTo = dragTo = forbidden


class ProbeSession(search.SearchSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus_results = {}

    def declaration(self, plan: TaskPlan) -> dict:
        value = super().declaration(plan)
        value.update(scope="NO API FUNCTIONAL PROBE; NOT search E2E", task=TASK,
                     model="scripted mock responses only; no GLM/API",
                     allowed_actions=["one real click of the OCR search label associated with the empty input",
                                      "one simulated append proposal through both focus gates, then mandatory backend veto"],
                     forbidden="ALL real keyboard input, Enter, submission, navigation, second click, other apps",
                     actual_click_limit=1, real_keyboard_authorized=False,
                     fixed_search_word_usage="simulated action payload ONLY; never written",
                     final_checks_usage="original search checks retained for reference; not expected to pass in this probe",
                     probe_acceptance="one click, both focus gates pass, mandatory veto, field and service remain empty")
        return value

    def confirm_action(self, decision, grounded) -> bool:
        action = decision.action
        if isinstance(action, ClickAction):
            if action.target_text != search.uia.LABEL or self.last_click is not None:
                self.log.emit("probe_action_scope_rejected", reason="second click or non-query click")
                raise GroundingError("Probe scope permits one query click only")
        elif not isinstance(action, TypeAction):
            self.log.emit("probe_action_scope_rejected", reason="non-click/non-simulated-input action")
            raise GroundingError("Probe scope forbids keys, submission and other actions")
        return super().confirm_action(decision, grounded)

    def focused_input(self, scene, expected: str, phase: str):
        try:
            binding = super().focused_input(scene, expected, phase)
        except Exception as error:
            self.focus_results[phase] = {"checked": True, "passed": False, "reason": str(error)}
            raise
        self.focus_results[phase] = {"checked": True, "passed": True, "binding": asdict(binding),
                                     "observation_id": scene.observation.observation_id}
        return binding


class MockClient:
    """Scripted decisions use only the normal current OCR request, never UIA."""

    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls == 1:
            plan = TaskPlan(goal=TASK, status="ready", steps=[{
                "step_id": "probe-focus", "description": "Click the visible search label once; validate focus and stop at the write veto",
                "success_criteria": "Both independent focus checks pass; no actual text is written",
            }, {
                "step_id": "review-stopped-probe", "depends_on": ["probe-focus"],
                "description": "Review independent no-input evidence after the mandatory stop; this runtime step is intentionally unreachable",
                "success_criteria": "Independent field and service checks are recorded by the test side, not the mock model",
            }])
            return MultimodalResponse(text=plan.model_dump_json(), provider="mock", model="scripted-focus-probe")
        if self.calls not in (2, 3):
            raise GroundingError("Probe does not request further model decisions or retry")
        fields = {}
        for line in request.instruction.splitlines():
            for name in ("Current observation", "Current step_id"):
                if line.startswith(name + ": "):
                    fields[name] = line[len(name) + 2:]
        observation = json.loads(fields["Current observation"])
        matches = [item for item in observation["elements"] if item["text"].strip() == search.uia.LABEL]
        if len(matches) != 1:
            raise GroundingError("Mock decision has no unique real OCR search target")
        action = {"type": "click" if self.calls == 2 else "type",
                  "target_id": matches[0]["target_id"], "target_text": search.uia.LABEL}
        if self.calls == 3:
            action.update(text=search.TOKEN, mode="append")
        return MultimodalResponse(text=json.dumps({"status": "action", "step_id": fields["Current step_id"],
                                                   "observation_id": observation["observation_id"], "action": action}),
                                  provider="mock", model="scripted-focus-probe")


def execute_probe(session: ProbeSession, raw, output: Path, *, wait=time.sleep, clock=time.monotonic):
    veto = VetoBackend(raw, session.log)
    controller = search.SearchController(veto, session)
    recorder = search.ScopedRecorder(MockClient(), output, session)
    runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), session.observer, controller,
                         limits=search.LIMITS, wait=wait, clock=clock)
    session.start()
    result = runtime.run(TASK, session.scope.checks, execute=True, confirm_plan=session.confirm_plan,
                         confirm_action=session.confirm_action, supervised_input_focus=True,
                         confirm_input_focus=session.confirm_input_focus)
    return result, recorder, veto


def independent_field_check(session: ProbeSession) -> dict:
    # This evidence is collected only after runtime has stopped, never sent to it.
    session.window_profile_check(session.current_scene())
    scene = session.observer.observe()
    binding = session.focused_input(scene, "", "post_stop_independent")
    return {"checked": True, "empty": True, "source": "independent_post_stop_uia_value",
            "binding": asdict(binding), "observation_id": scene.observation.observation_id}


def probe_passed(result, recorder, veto, session, before, after, field, error) -> bool:
    return bool(error is None and result is not None and recorder is not None and veto is not None
                and session is not None and result.status == "blocked" and result.reason == VETO_REASON
                and result.executed_actions == 1 and result.attempted_actions == 2
                and recorder.calls == 3 and veto.click_calls == 1 and veto.successful_clicks == 1
                and veto.write_veto_calls == 1 and not session.typed and not session.submitted
                and all(session.focus_results.get(p, {}).get("passed") is True for p in ("focus_callback", "write_entry"))
                and before == after == EMPTY_RECORDS and field.get("checked") is True and field.get("empty") is True)


def load_recognizer():
    import easyocr
    return search.demo.EasyOcrRecognizer(reader=easyocr.Reader(["ch_sim", "en"], gpu=False, download_enabled=False))


def run_once(*, enabled: bool = False) -> int:
    if enabled is not True:
        raise GroundingError("Explicit --run-one-probe is required; real input remains forbidden")
    output = search.demo._external_path(OUTPUT)
    output.mkdir(parents=True, exist_ok=False)
    log = search.EvidenceLog(output)
    stop_file = output / "STOP"
    session = result = recorder = veto = None
    before = after = None
    field = {"checked": False, "empty": None}
    error = None
    try:
        save(output, "attempt-claimed.json", {"scope": TASK, "api_calls": 0, "real_keyboard_authorized": False,
                                             "actual_click_limit": 1, "profile": str(search.PROFILE)})
        before = search.uia.read_records()
        save(output, "service-before.json", before)
        if before != EMPTY_RECORDS:
            raise GroundingError("Service records must be empty; no reset or retry permitted")
        print("NO API PROBE: one actual OCR-grounded click; all real keyboard input is vetoed.", flush=True)
        print("One plan/scope approval, no action/focus dialogs. Keep the isolated search browser foreground.", flush=True)
        print("Independent STOP file: " + str(stop_file), flush=True)
        if input("Confirm EMPTY isolated /search start and local screenshot storage; type READY: ") != "READY":
            raise GroundingError("Preparation consent declined")
        print("Loading cached CPU OCR; downloads disabled, detailed logs saved to probe-console.txt.", flush=True)
        with (output / "probe-console.txt").open("x", encoding="utf-8") as console:
            with redirect_stdout(console), redirect_stderr(console):
                recognizer = load_recognizer()
            print("Activate ONLY the prepared test browser within 8 seconds. Do not click input or type. Stay until THREE tones.", flush=True)
            time.sleep(8)
            probe = WindowsWindowProbe()
            bound = probe.foreground()
            search.validate_capture_scope(bound, probe, log)
            save(output, "bound-window.json", asdict(bound))
            observer = search.SearchObserver(bound, output / "observations", recognizer=recognizer,
                                             window_probe=probe, stop_file=stop_file)
            with redirect_stdout(console), redirect_stderr(console):
                initial = observer.observe()
                checks = search.full_window_checks(initial)
                if any(t.element.text.strip() in search.FINAL_TEXTS or search.TOKEN in t.element.text
                       for t in initial.observation.targets):
                    raise GroundingError("Original empty/no-results start is missing")
                save(output, "final-checks-reference.json", [c.model_dump(mode="json") for c in checks])
                session = ProbeSession(search.Scope(bound, checks), observer, probe, log, stop_file, enabled=True)
                data = session.inspect(initial, "", "initial_probe_scope")
                binding = session.query_binding(data, initial, search.ocr_target(initial, search.uia.LABEL)[1])
                save(output, "initial-empty-input.json", {"binding": asdict(binding), "empty": True})
                save(output, "source-hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), Path(search.__file__), Path(search.uia.__file__))})
                result, recorder, veto = execute_probe(session, search.demo.DesktopController()._backend, output)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.emit("probe_entry_stopped", error=error)
    finally:
        if session is not None:
            try:
                with (output / "probe-console.txt").open("a", encoding="utf-8") as console:
                    with redirect_stdout(console), redirect_stderr(console):
                        field = independent_field_check(session)
            except (Exception, KeyboardInterrupt) as exc:
                field = {"checked": False, "empty": None, "error": f"{type(exc).__name__}: {exc}"}
        save(output, "independent-field-after.json", field)
        try:
            after = search.uia.read_records()
            save(output, "service-after.json", after)
        except (Exception, KeyboardInterrupt) as exc:
            error = error or f"Independent service check failed: {type(exc).__name__}: {exc}"
        passed = probe_passed(result, recorder, veto, session, before, after, field, error)
        summary = {"scope": "No API desktop functional probe; NOT search E2E", "probe_passed": passed,
                   "status": "probe_passed" if passed else "probe_not_passed", "entry_error": error,
                   "runtime": result.to_dict() if result else None, "api_calls": 0,
                   "mock_calls": recorder.calls if recorder else 0,
                   "click_calls": veto.click_calls if veto else 0, "successful_clicks": veto.successful_clicks if veto else 0,
                   "write_veto_calls": veto.write_veto_calls if veto else 0, "real_write_calls": 0,
                   "focus_results": session.focus_results if session else {}, "field_after": field,
                   "records_before": before, "records_after": after,
                   "records_unchanged": None if before is None or after is None else before == after,
                   "accepted_search_e2e_success": False, "selection_verified": False,
                   "input_permission": False,
                   "runtime_exit_code": None if result is None else 0 if result.status == "completed" else 1,
                   "entry_exit_code": 0 if passed else 1, "os_level_zero_egress_guarantee": False,
                   "stop_file": str(stop_file)}
        save(output, "probe-summary.json", summary)
        print("ALL PROBE/DESKTOP CHECKS STOPPED; ending tones indicate completion only.", flush=True)
        try:
            search.uia.signal_done()
        except Exception as exc:
            log.emit("ending_tones_unavailable", error=f"{type(exc).__name__}: {exc}")
    print(f"probe_passed={passed} runtime_status={result.status if result else 'preparation_blocked'} actual_clicks={summary['successful_clicks']} real_write_calls=0")
    print("reason=" + (error or (result.reason if result else "Preparation did not complete")))
    print("evidence=" + str(output))
    print("STOP: no retry, no input authorization, no search E2E or next task.")
    return summary["entry_exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one-probe", action="store_true")
    args = parser.parse_args()
    try:
        return run_once(enabled=args.run_one_probe)
    except Exception as exc:
        print(f"Stopped before probe: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
