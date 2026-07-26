"""Harness tests for the idle-nudge behavioral eval (no LLM calls).

The LLM loop itself is exercised by real sweeps against a live
endpoint; these pin everything deterministic around it — cell fixture
validity, stimulus shape, scoring semantics, and the stub client's
network inertness — so a sweep failure means the model, not the
harness.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from turnstone.core.session import COORDINATOR_TOOLS
from turnstone.core.storage._registry import get_storage, init_storage, reset_storage
from turnstone.core.workstream import WorkstreamKind
from turnstone.eval.nudges import (
    ARM_BARE_CONTINUE,
    ARM_NO_PROVENANCE,
    ARM_NUDGE,
    ARM_PAIR_CF,
    ARM_PAIR_TF,
    _seed_tasks,
    _seed_transcript,
    _StubCoordinatorClient,
    build_stimulus,
    render_tasks_body,
    score_nudge_run,
)
from turnstone.eval.scenarios.nudges import NUDGE_CELLS

_LIVE_TOOL_NAMES = {t["function"]["name"] for t in COORDINATOR_TOOLS}


def _cell(cell_id: str) -> dict[str, Any]:
    return next(c for c in NUDGE_CELLS if c["id"] == cell_id)


def _envelope(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "tasks": list(rows)}


@pytest.fixture
def eval_storage(tmp_path):
    reset_storage()
    init_storage("sqlite", path=str(tmp_path / "eval.db"), run_migrations=False)
    storage = get_storage()
    storage.register_workstream(
        "coord-eval-1",
        user_id="eval-user",
        name="eval-coordinator",
        kind=WorkstreamKind.COORDINATOR,
        state="idle",
    )
    yield storage
    reset_storage()


class TestCellFixtures:
    def test_nudge_cells_use_live_tool_names(self):
        """A forbid/expect matcher with a misspelt tool name silently
        never matches — which reads as a false PASS.  Every name must be
        on the coordinator's real wire."""
        for cell in NUDGE_CELLS:
            named = [a["tool"] for a in cell.get("forbid_actions", [])]
            named += [a["tool"] for a in cell.get("expect_actions", {}).get("actions", [])]
            for name in named:
                assert name in _LIVE_TOOL_NAMES, f"{cell['id']}: unknown tool {name!r}"

    def test_cell_ids_unique_and_arms_known(self):
        ids = [c["id"] for c in NUDGE_CELLS]
        assert len(ids) == len(set(ids))
        known = {ARM_NUDGE, ARM_BARE_CONTINUE, ARM_NO_PROVENANCE, ARM_PAIR_TF, ARM_PAIR_CF}
        for cell in NUDGE_CELLS:
            assert set(cell["arms"]) <= known, cell["id"]

    def test_every_cell_seed_spec_passes_real_validation(self, eval_storage):
        """Fixture drift guard: every cell's tasks seed through the REAL
        tasks_add (status vocabulary, length caps, renderability) — a
        vocabulary change that orphans a fixture fails here, not mid-
        sweep."""
        for cell in NUDGE_CELLS:
            client = _StubCoordinatorClient(
                eval_storage, coord_ws_id="coord-eval-1", user_id="eval-user"
            )
            id_map = _seed_tasks(client, "coord-eval-1", cell)
            assert len(id_map) == len(cell.get("tasks", []))
            assert all(tid.startswith("tsk_") for tid in id_map.values())
            # Clean the envelope between cells (same coord ws).
            env = client.tasks_get("coord-eval-1")
            for row in env.get("tasks", []):
                client.tasks_remove("coord-eval-1", task_id=row["id"])

    def test_bad_seed_fixture_fails_loudly(self, eval_storage):
        client = _StubCoordinatorClient(
            eval_storage, coord_ws_id="coord-eval-1", user_id="eval-user"
        )
        bad = {"id": "X", "tasks": [{"title": "t", "status": "not-a-status"}]}
        with pytest.raises(ValueError, match="rejected"):
            _seed_tasks(client, "coord-eval-1", bad)


class TestStimulus:
    def test_arm_turn_shapes(self):
        env = _envelope({"id": "tsk_1", "title": "audit auth.py", "status": "pending"})
        kids = [{"ws_id": "ws-c1", "name": "auditor", "state": "running"}]

        nudge = build_stimulus(ARM_NUDGE, envelope=env, children=kids)
        assert [t["role"] for t in nudge] == ["user", "system"]
        assert nudge[0]["content"] == "" and nudge[0]["_source"] == "system_nudge"
        assert nudge[1]["_source"] == "idle_tasks"

        tf = build_stimulus(ARM_PAIR_TF, envelope=env, children=kids)
        assert [t.get("_source") for t in tf[1:]] == ["idle_tasks", "idle_children"]
        cf = build_stimulus(ARM_PAIR_CF, envelope=env, children=kids)
        assert [t.get("_source") for t in cf[1:]] == ["idle_children", "idle_tasks"]

        bare = build_stimulus(ARM_BARE_CONTINUE, envelope=env, children=[])
        assert bare == [{"role": "user", "content": "continue"}]

        with pytest.raises(ValueError):
            build_stimulus("nope", envelope=env, children=[])

    def test_bodies_are_production_rendered(self):
        """The eval carries no copy of the body: the real formatter over
        the really-seeded envelope, including the co-delivery honesty
        line and the escape branch."""
        env = _envelope({"id": "tsk_1", "title": "audit auth.py", "status": "pending"})
        body = render_tasks_body(env)
        assert "tsk_1" in body and "audit auth.py" in body
        assert "may still be running" in body
        assert "needs_operator" in body
        assert "not from the operator" in body

    def test_no_provenance_arm_drops_only_the_first_paragraph(self):
        env = _envelope({"id": "tsk_1", "title": "audit auth.py", "status": "pending"})
        body = render_tasks_body(env, strip_provenance=True)
        assert "not from the operator" not in body
        assert "may still be running" not in body  # rides the provenance paragraph
        assert "needs_operator" in body and "tsk_1" in body

    def test_ragged_envelope_rows_do_not_raise(self):
        env = _envelope(
            {"id": "tsk_1", "title": None, "status": "pending", "note": 42},
            "not-a-dict-row",
        )
        body = render_tasks_body(env)
        assert "tsk_1" in body

    def test_seed_transcript_pairs_calls_with_results(self):
        wires = _seed_transcript(_cell("C6_co_delivery"))
        roles = [w["role"] for w in wires]
        assert roles == ["user", "assistant", "tool"]
        call = wires[1]["tool_calls"][0]
        assert call["function"]["name"] == "spawn_workstream"
        assert json.loads(call["function"]["arguments"])["name"] == "auditor"
        assert wires[2]["tool_call_id"] == call["id"]


class TestScoring:
    def test_forbidden_action_flags_without_failing_state(self):
        case = {"forbid_actions": [{"tool": "spawn_workstream"}]}
        log = [{"tool": "spawn_workstream", "args": {"name": "x"}, "result": "", "turn": 0}]
        r = score_nudge_run(log, _envelope(), case, {})
        assert not r["pass"] and r["forbidden"] and not r["failures"]

    def test_expect_state_checks_status_note_and_link(self):
        case = {
            "expect_state": {
                0: {"status": "needs_operator", "note_nonempty": True},
                1: {"status": "in_progress", "child_ws_id": "ws-c1"},
            }
        }
        good = _envelope(
            {"id": "tsk_a", "status": "needs_operator", "note": "need the token"},
            {"id": "tsk_b", "status": "in_progress", "child_ws_id": "ws-c1"},
        )
        r = score_nudge_run([], good, case, {0: "tsk_a", 1: "tsk_b"})
        assert r["pass"], r["failures"]

        bad = _envelope(
            {"id": "tsk_a", "status": "needs_operator", "note": "   "},
            {"id": "tsk_b", "status": "in_progress", "child_ws_id": ""},
        )
        r2 = score_nudge_run([], bad, case, {0: "tsk_a", 1: "tsk_b"})
        assert not r2["pass"] and len(r2["failures"]) == 2

    def test_missing_task_is_a_state_failure(self):
        case = {"expect_state": {0: {"status": "done"}}}
        r = score_nudge_run([], _envelope(), case, {0: "tsk_gone"})
        assert not r["pass"] and "missing" in r["failures"][0]

    def test_require_stop_flags_work_after_bookkeeping(self):
        case = {"require_stop": True}
        log = [
            {"tool": "tasks", "args": {"action": "update"}, "result": "", "turn": 0},
            {"tool": "spawn_workstream", "args": {}, "result": "", "turn": 1},
        ]
        r = score_nudge_run(log, _envelope(), case, {})
        assert not r["pass"] and "stop" in r["failures"][0]

    def test_allow_after_bookkeeping_permits_escalation_surfacing(self):
        """``notify`` after the bookkeeping is the escalation reaching
        the operator's channel — the POINT of a legit stop, not a
        violation of it.  Anything outside the allowlist still fails."""
        case = {"require_stop": True, "allow_after_bookkeeping": ["notify"]}
        ok = [
            {"tool": "tasks", "args": {"action": "update"}, "result": "", "turn": 0},
            {"tool": "notify", "args": {"message": "need the token"}, "result": "", "turn": 1},
        ]
        assert score_nudge_run(ok, _envelope(), case, {})["pass"]
        stray = ok + [{"tool": "list_nodes", "args": {}, "result": "", "turn": 2}]
        assert not score_nudge_run(stray, _envelope(), case, {})["pass"]

    def test_expected_actions_contains_any(self):
        case = {
            "expect_actions": {
                "mode": "contains_any",
                "actions": [{"tool": "wait_for_workstream"}],
            }
        }
        hit = [{"tool": "wait_for_workstream", "args": {}, "result": "", "turn": 0}]
        assert score_nudge_run(hit, _envelope(), case, {})["pass"]
        miss = [{"tool": "tasks", "args": {}, "result": "", "turn": 0}]
        assert not score_nudge_run(miss, _envelope(), case, {})["pass"]


class TestSessionConstruction:
    def test_coordinator_headless_session_constructs_with_coord_wire(self, eval_storage):
        """The passthrough seam: ``kind=COORDINATOR`` + ``coord_client``
        must reach ChatSession, the wire must be the coordinator tool
        list (not the CLI TOOLS constant), and the natural coordinator
        prompt must compose (no override)."""
        from openai import OpenAI

        from turnstone.eval.nudges import CoordinatorHeadlessSession

        client = _StubCoordinatorClient(
            eval_storage, coord_ws_id="coord-eval-1", user_id="eval-user"
        )
        session = CoordinatorHeadlessSession(
            client=OpenAI(base_url="http://eval.invalid/v1", api_key="x"),
            model="eval-model",
            coord_client=client,
            ws_id="coord-eval-1",
            user_id="eval-user",
            temperature=0.7,
            max_tokens=1024,
            reasoning_effort="medium",
            context_window=32768,
        )
        wire = {t["function"]["name"] for t in session._eval_tools}
        assert "tasks" in wire and "wait_for_workstream" in wire
        assert "bash" not in wire  # the CLI tool set must NOT leak in
        assert session.auto_approve is True


class TestStubClient:
    def test_network_methods_are_inert(self, eval_storage):
        client = _StubCoordinatorClient(
            eval_storage, coord_ws_id="coord-eval-1", user_id="eval-user"
        )
        out = client._post_url("http://eval.invalid/v1/api/route/x", {})
        assert "error" in out and "eval environment" in out["error"]

    def test_wait_default_completes_with_stubbed_children(self, eval_storage):
        client = _StubCoordinatorClient(
            eval_storage,
            coord_ws_id="coord-eval-1",
            user_id="eval-user",
            children=[{"ws_id": "ws-c1", "name": "auditor", "state": "idle"}],
        )
        out = client.wait_for_workstream(["ws-c1"])
        assert out["complete"] is True
        assert out["results"]["ws-c1"]["name"] == "auditor"

    def test_scripted_stub_takes_precedence(self, eval_storage):
        client = _StubCoordinatorClient(
            eval_storage,
            coord_ws_id="coord-eval-1",
            user_id="eval-user",
            tool_stubs={"wait_for_workstream": [{"complete": False, "results": {}}]},
        )
        assert client.wait_for_workstream(["ws-x"])["complete"] is False

    def test_tasks_stay_fully_real(self, eval_storage):
        """The point of the harness: tasks execute against the temp DB
        with production validation."""
        client = _StubCoordinatorClient(
            eval_storage, coord_ws_id="coord-eval-1", user_id="eval-user"
        )
        row = client.tasks_add("coord-eval-1", title="real", status="pending")
        assert row["id"].startswith("tsk_")
        assert "error" in client.tasks_add("coord-eval-1", title="<>")
