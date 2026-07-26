"""Observer that nudges idle coordinators with unfinished work.

Subscribes to a coordinator-side :class:`SessionManager`'s state events
and enqueues one of two nudges on the coord's :class:`NudgeQueue` when a
coord transitions to :class:`WorkstreamState.IDLE`.  The
:class:`turnstone.core.idle_nudge_watcher.IdleNudgeWatcher` (registered
*after* this observer in the lifespan, so subscriber-order has the
observer fire first on the same IDLE event) then peeks the queue and
dispatches the wake send.

Two nudge types, in two different **classes**:

* ``idle_children`` — LIVENESS.  The coord went idle with active
  interactive children; the wake exists so their results are collected
  instead of abandoned.  Suggests ``wait_for_workstream``.
* ``idle_tasks`` — ADVICE.  The coord went idle holding open
  (``pending`` / ``in_progress``) entries on its own ``tasks`` list and
  *no* active children.  Suggests reconciling the list, with the
  operator-escalation branch first.

The class decides three behaviours, each ruled at its site:

* **Config gating.**  LIVENESS is never gated on ``memory.nudges`` —
  that switch's operator-facing help text promises memory-reminder
  control, and suppressing the wake would strand a coordinator whose
  children finish unobserved (see ``_maybe_enqueue_children``).  ADVICE
  gates through ``ChatSession._nudges_enabled``, which also suppresses
  the nudge when the persona envelope hides the ``tasks`` tool its body
  instructs (``NUDGE_REQUIRED_TOOL``).
* **Per-type caps** (:data:`_NUDGE_TYPE_CAPS`) — liveness 3, advice 2
  per idle bracket; worst-case ceiling 5.
* **Drain-predicate failure direction.**  An indeterminate children
  read at drain time DELIVERS a liveness entry (stale noise is cheap;
  a lost wake is a stall) and DROPS an advice entry (resuming over live
  children is the contradictory pair this module exists to prevent).

Mutual exclusion: the two conditions cannot both hold in one IDLE event
("block on your children" and "resume your task" are contradictory
instructions), enforced by a **single shared children snapshot** per
event and re-checked by the advice drain predicate across the entry's
whole queued lifetime.

Gate order, ``idle_children`` (cheap → expensive; matches the code):
coordinator-kind → cooldown peek → cap peek → wait-tool skip →
children snapshot → ``should_nudge`` → atomic charge → enqueue.

Gate order, ``idle_tasks``: coordinator-kind → ``_nudges_enabled``
(config + ``tasks``-tool visibility) → cooldown peek → cap peek →
asked-operator skip → children snapshot (must be a known-empty read) →
envelope read → ``should_nudge`` → atomic charge → enqueue.

Caps reset when the ws leaves IDLE for a non-wake reason (tracked by
``ChatSession._wake_source_tag``); cooldowns live per-type in the
session's ``_metacog_state``.  Both are per-process state — if one
coordinator were ever live in two console processes, caps and cooldowns
would disagree across them (pre-existing shape, shared with every other
per-process cache).
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

from turnstone.console.coordinator_client import TASK_OPEN_STATUSES, load_task_envelope
from turnstone.core.log import get_logger
from turnstone.core.metacognition import (
    NUDGE_IDLE_TASKS_DISPLAY_CAP,
    _cooldown_allows,
    _field_str,
    format_idle_children_nudge,
    format_idle_tasks_nudge,
    sanitize_name,
    should_nudge,
)
from turnstone.core.trajectory import Role
from turnstone.core.workstream import WorkstreamKind, WorkstreamState

if TYPE_CHECKING:
    from collections.abc import Callable

    from turnstone.core.session import ChatSession
    from turnstone.core.session_manager import SessionManager
    from turnstone.core.storage._protocol import StorageBackend
    from turnstone.core.trajectory import Turn
    from turnstone.core.workstream import Workstream

log = get_logger(__name__)

# Active = the model can act on the child (it's still working,
# streaming, or waiting on user attention).  Excludes "idle" (the
# child is now waiting and can't be unblocked by the coord), "closed"
# (gone), "deleted" (gone), and "error" (the model can't unblock an
# errored child without operator intervention; cooldown handles repeat
# fires for stuck-error children).
_ACTIVE_CHILD_STATES: frozenset[str] = frozenset(
    {
        WorkstreamState.THINKING.value,
        WorkstreamState.RUNNING.value,
        WorkstreamState.ATTENTION.value,
    }
)

# Per-type hard caps on idle-nudge fires per idle bracket, sized by
# class: LIVENESS (``idle_children``) keeps the shipped budget of 3;
# ADVICE (``idle_tasks``) gets 2.  Worst-case ceiling per bracket is 5.
#
# Deliberately NOT a shared total: a summed cap lets advice fires spend
# the liveness budget, so a coordinator that used its wakes on task
# reminders reaches the silent-stall state (live children, no wake
# left) strictly sooner than before ``idle_tasks`` existed.  The
# liveness budget must be starvation-proof against advice.  The cost is
# a higher combined ceiling (5, was 3) — acceptable because the two
# conditions are mutually exclusive per event, so alternating fires
# require the coordinator to actually progress (children finishing,
# tasks reconciling) between wakes, which is the system working, not
# hammering.
#
# Every nudge type this observer emits MUST be registered here — the
# lookup KeyErrors on an unregistered type (surfacing via _on_state's
# ``log.exception`` and any test that drives the new path) rather than
# inheriting either class's budget silently.  Fail-loud is the point:
# defaulting a future liveness type to the advice budget (or vice
# versa) is exactly the misclassification this table exists to force a
# decision on.
_NUDGE_TYPE_CAPS: dict[str, int] = {
    "idle_children": 3,
    "idle_tasks": 2,
}

# Soft cap on the snapshot query.  Higher than ``WAIT_MAX_WS_IDS`` so
# the SQL ``LIMIT`` (applied before the Python state filter) doesn't
# clip genuinely-active children whose ``updated`` timestamp is older
# than recently-closed siblings.  Realistic coord histories are far
# smaller than this; if a coord ever exceeds it, the formatter still
# truncates to ``WAIT_MAX_WS_IDS`` for the model-facing suggestion.
_ACTIVE_CHILDREN_QUERY_LIMIT = 200


class CoordinatorIdleObserver:
    """Subscribe to a coord SessionManager's IDLE events and enqueue
    ``idle_children`` / ``idle_tasks`` nudges when unfinished work
    remains.
    """

    def __init__(self, manager: SessionManager, storage: StorageBackend) -> None:
        self._manager = manager
        self._storage = storage
        self._callback: Callable[[str, WorkstreamState], None] | None = None
        # Per-ws fire counts keyed by ``ws_id`` → ``{nudge_type: count}``.
        # Two-level dict makes the "any caps for this ws?" check at
        # leave-IDLE an O(1) ``ws_id in self._fire_counts`` lookup
        # instead of an O(N_caps) scan over a flat tuple-keyed map.
        # Lock protects against race with the leave-IDLE reset path
        # running on a different thread (state events fire on the
        # calling thread of ``set_state`` — currently always the
        # worker thread that did the transition, but the lock keeps
        # the contract robust).
        self._fire_counts: dict[str, dict[str, int]] = {}
        self._fire_counts_lock = threading.Lock()

    def start(self) -> None:
        """Idempotent — registering twice is a no-op."""
        if self._callback is not None:
            return

        def _on_state(ws_id: str, state: WorkstreamState) -> None:
            if state is not WorkstreamState.IDLE:
                # Reset hard-cap when leaving IDLE for a *real* reason
                # (not a wake-driven exit).  Skip the manager-lock /
                # session-attribute walk entirely when no caps are
                # accumulated for this ws — the common case for the
                # vast majority of state transitions.
                with self._fire_counts_lock:
                    has_caps = ws_id in self._fire_counts
                if not has_caps:
                    return
                ws = self._manager.get(ws_id)
                if ws is None or ws.session is None:
                    return
                # ``_wake_source_tag`` is set on the session iff a
                # wake send is in flight; if set, leaving IDLE is the
                # wake's own IDLE→THINKING→RUNNING transition and the
                # cap should NOT reset.  If unset, the user / a real
                # producer drove the coord forward and the cap should
                # clear so the next genuine idle bracket is fresh.
                if not ws.session._wake_source_tag:
                    self._reset_caps_for(ws_id)
                return

            # state == IDLE branch.
            try:
                self._on_idle(ws_id)
            except Exception:
                log.exception("coord_idle_observer.maybe_enqueue_failed ws=%s", ws_id[:8])

        self._callback = _on_state
        self._manager.subscribe_to_state(_on_state)

    def shutdown(self) -> None:
        """Unsubscribe; idempotent."""
        cb = self._callback
        if cb is None:
            return
        with contextlib.suppress(Exception):
            self._manager.unsubscribe_from_state(cb)
        self._callback = None

    def _on_idle(self, ws_id: str) -> None:
        """Dispatch both nudge paths for one IDLE event.

        Both paths consume the same children snapshot, computed **at
        most once per event** and memoised behind :data:`_UNSET` (never
        ``None`` — an indeterminate read IS ``None`` and must memoise
        too, or a storage failure re-runs the query and can hand the
        two paths different answers).

        Laziness is not an optimisation detail here, it is the gate
        order: each path checks its cooldown and cap first (microsecond
        dict lookups) and most IDLE events short-circuit there.
        Computing the snapshot eagerly would put a ``list_workstreams``
        round-trip on every coord state transition in the cluster.

        Sharing is not an optimisation at all — it is correctness.  Two
        independent ``_active_children`` calls observe two different
        snapshots, and a child that finishes between them yields
        ``idle_children`` (children active at T0) *and* ``idle_tasks``
        (no children at T1) in the same drain: the model is told to
        block on its children and to resume its task, in one turn.
        """
        ws = self._manager.get(ws_id)
        if ws is None or ws.session is None:
            return
        if ws.kind is not WorkstreamKind.COORDINATOR:
            return
        # Bind the non-Optional session for the closures below — mypy's
        # narrowing from the check above does not reach into them.
        session = ws.session

        # Single-slot list as the memo: EMPTY means "not computed yet",
        # so the slot's value is free to be ``None`` (an indeterminate
        # read) without colliding with the sentinel.  Using ``None``
        # itself as the sentinel would re-run the query on every access
        # after a storage failure, and a first-raise/second-success pair
        # would hand the two nudge paths different snapshots — exactly
        # the contradictory-pair race this shared snapshot prevents,
        # arriving through the failure door.
        memo: list[list[dict[str, str]] | None] = []

        def active_children() -> list[dict[str, str]] | None:
            if not memo:
                memo.append(self._active_children(ws))
            return memo[0]

        self._maybe_enqueue_children(ws, session, active_children)
        self._maybe_enqueue_tasks(ws, session, active_children)

    # ------------------------------------------------------------------
    # shared gate head / enqueue tail
    # ------------------------------------------------------------------

    def _common_gates_allow(self, session: ChatSession, ws_id: str, nudge_type: str) -> int | None:
        """The gate head both paths run identically: cooldown peek +
        per-type cap peek, both microsecond dict lookups.

        Returns the cooldown seconds (``should_nudge`` needs the same
        number again) when the gates pass, ``None`` when blocked.  Kept
        in one place so a cross-cutting change to the cheap gates lands
        on both nudge types at once — this file has already paid once
        for editing the head twice.
        """
        cooldown_secs = getattr(session._mem_cfg, "nudge_cooldown", 300)
        if not _cooldown_allows(nudge_type, session._metacog_state, cooldown_secs=cooldown_secs):
            return None
        if self._cap_reached(ws_id, nudge_type):
            return None
        return cooldown_secs

    def _enqueue_nudge(
        self,
        session: ChatSession,
        ws_id: str,
        nudge_type: str,
        *,
        text: str,
        metadata: dict[str, Any] | None,
        valid_until: Callable[[], bool],
        cooldown_secs: int,
    ) -> bool:
        """The shared enqueue tail: ``should_nudge`` (the authoritative
        cooldown check that also RECORDS the timestamp), the formatter
        empty-body guard, the atomic cap charge, and the enqueue itself.

        One implementation for both types so the charge discipline —
        cheap peek early, authoritative :meth:`_try_charge` at the fire
        position — can never drift between paths.

        ``memory_count=0`` is deliberate, not a stub: ``should_nudge``
        reads ``memory_count`` only for the ``tool_error`` / ``resume``
        / ``start`` types, so paying 1-2 ``count_structured_memories``
        round-trips per IDLE event to compute the real number bought
        nothing.  If a future nudge type routed through here needs it,
        thread it explicitly.
        """
        if not should_nudge(
            nudge_type,
            session._metacog_state,
            message_count=len(session.messages),
            memory_count=0,
            cooldown_secs=cooldown_secs,
        ):
            return False
        if not text:  # belt-and-braces: formatter empty-input guard
            return False
        if not self._try_charge(ws_id, nudge_type):
            return False
        session._nudge_queue.enqueue(
            nudge_type,
            text,
            "any",
            valid_until=valid_until,
            metadata=metadata,
        )
        return True

    # ------------------------------------------------------------------
    # idle_children — LIVENESS
    # ------------------------------------------------------------------

    def _maybe_enqueue_children(
        self,
        ws: Workstream,
        session: ChatSession,
        active_children: Callable[[], list[dict[str, str]] | None],
    ) -> None:
        """Enqueue ``idle_children`` when the coord went idle with
        active interactive children.

        LIVENESS class — deliberately NOT gated on
        ``ChatSession._nudges_enabled`` / ``memory.nudges``.  That
        switch's operator-facing help text promises control of
        memory-save reminders; suppressing this wake under it stranded
        coordinators whose children finished unobserved (results never
        collected, operator hand-restarts every stalled coord).  It is
        the only nudge producer that ever made that mistake — every
        other liveness-class producer (``watch_triggered``,
        ``background_shell_exit`` via the external-event rail,
        ``compaction_pending``) already bypasses the gate by
        construction.  Do not "unify" this with the advice path's gate.

        The body suggests ``wait_for_workstream`` but the wake fires
        even for a persona that hides that tool: the wake itself is the
        point — the model can also continue the user's work, inspect or
        message the children — and the tool line is decoration.  This
        asymmetry with the advice path (which IS visibility-gated) is
        deliberate: an advice body is nothing but ``tasks(...)`` calls,
        while the liveness body has a non-tool branch.
        """
        ws_id = ws.id

        cooldown_secs = self._common_gates_allow(session, ws_id, "idle_children")
        if cooldown_secs is None:
            return

        # Gate: skip if the coord's last assistant turn already used
        # ``wait_for_workstream``.  Don't nudge toward a tool the
        # model is already using.
        if self._last_assistant_used_wait(session):
            return

        # Gate: the shared per-event children snapshot.  ``None`` means
        # the read was indeterminate (storage raised) — fail closed and
        # say so, rather than letting a falsy non-answer read as "no
        # children".  ``[]`` is a REAL empty answer and also skips.
        active = active_children()
        if active is None:
            log.debug("coord_idle_observer.children_indeterminate ws=%s (no nudge)", ws_id[:8])
            return
        if not active:
            return

        text = format_idle_children_nudge(active)

        # Structured ``_source_meta`` for the FE idle-children card — the same
        # child list ``format_idle_children_nudge`` rendered into ``text`` above,
        # so the card and the model-facing prose derive from one source.  Names
        # are sanitized identically (``sanitize_name``) so a steering-vector /
        # angle-bracket name can't reach the operator card any more than the
        # text; the FE additionally renders every field via ``textContent``.
        children_meta = [
            {
                "ws_id": c.get("ws_id", ""),
                "name": sanitize_name(c.get("name", "")),
                "state": c.get("state", ""),
            }
            for c in active
        ]

        # Bind ws.id + user_id by closure so the predicate captures the
        # workstream identity (not the live ``ws`` reference, which
        # could mutate).  The predicate runs at drain time outside the
        # queue lock, on whichever worker thread drains the entry.
        bound_ws_id = ws.id
        bound_user_id = ws.user_id

        def _still_has_active_children() -> bool:
            now = self._active_children_now(bound_ws_id, bound_user_id)
            if now is None:
                # LIVENESS fails OPEN at drain: the entry only exists
                # because children were active at enqueue, and dropping
                # it on a storage blip silently abandons the wake — the
                # stalled-coordinator outcome this class exists to
                # prevent, with no retry behind it.  Delivering on an
                # indeterminate read risks only a stale "children still
                # running" nudge; if they finished, the suggested wait
                # returns immediately with their results.  The advice
                # predicate makes the OPPOSITE choice — see
                # ``_maybe_enqueue_tasks`` — because its failure harm
                # points the other way.
                return True
            return now

        if self._enqueue_nudge(
            session,
            ws_id,
            "idle_children",
            text=text,
            metadata={"children": children_meta},
            valid_until=_still_has_active_children,
            cooldown_secs=cooldown_secs,
        ):
            log.info(
                "coord_idle_observer.enqueued ws=%s active_children=%d",
                ws_id[:8],
                len(active),
            )

    # ------------------------------------------------------------------
    # idle_tasks — ADVICE
    # ------------------------------------------------------------------

    def _maybe_enqueue_tasks(
        self,
        ws: Workstream,
        session: ChatSession,
        active_children: Callable[[], list[dict[str, str]] | None],
    ) -> None:
        """Enqueue ``idle_tasks`` when the coord went idle holding open
        tasks and (provably) no active children.

        ADVICE class — gated on ``ChatSession._nudges_enabled``, which
        for this type means the ``memory.nudges`` config switch AND the
        persona envelope exposing the ``tasks`` tool
        (``NUDGE_REQUIRED_TOOL``): every branch of the nudge body is a
        ``tasks(...)`` call, so firing it at a persona that hides the
        tool produces "I don't have access" apology loops.
        """
        ws_id = ws.id

        if not session._nudges_enabled("idle_tasks"):
            return

        cooldown_secs = self._common_gates_allow(session, ws_id, "idle_tasks")
        if cooldown_secs is None:
            return

        # Gate: the coord's last assistant turn looks like a question put
        # to the operator.  Deliberately false-negative-biased — see the
        # method docstring; suppressing a legitimate nudge costs the
        # operator one "continue", while nudging a coord that correctly
        # escalated pushes it to guess on a decision that was not its to
        # make.
        if self._last_assistant_asked_operator(session):
            return

        # Gate: yield to ``idle_children``.  Two rulings live here:
        #
        # 1. The yield is to the CONDITION (children are active), not to
        #    whether ``idle_children`` actually fired — that nudge can be
        #    blocked by its own cooldown or cap while children are still
        #    running, and in that window ``idle_tasks`` must stay silent
        #    too.  "Block on your children" and "resume your task" are
        #    contradictory instructions; the model must never receive
        #    both in one drain.  Do NOT "fix" this into
        #    ``if idle_children_fired``.
        # 2. ``None`` (the read was indeterminate) yields exactly like
        #    "children active": firing the resume-your-task nudge when
        #    the children state is UNKNOWN risks the same contradictory
        #    pair through the failure door.  ADVICE fails closed.
        active = active_children()
        if active is None or active:
            return

        # Gate: read the task envelope.  A corrupt or unreadable blob
        # decodes as an empty list (``load_task_envelope`` swallows both
        # and flags the former), so this gate refuses to fire on a list
        # it cannot read — correct, but indistinguishable from "no open
        # tasks" without the log line below.
        #
        # Cost note (accepted): this row read + JSON parse runs on every
        # IDLE event that clears the cheap gates with no active children
        # — the common terminal state — even for coords that never used
        # the tasks tool.  One indexed PK fetch on a worker thread,
        # comparable to the children query beside it; the structural fix
        # (a narrow ``load_workstream_config_key`` accessor) needs a
        # storage-protocol change across both backends and is not worth
        # it at this rate.
        envelope, corrupt = load_task_envelope(self._storage, ws_id)
        if corrupt:
            log.debug(
                "coord_idle_observer.tasks_envelope_corrupt ws=%s (no nudge)",
                ws_id[:8],
            )
            return
        open_tasks = self._open_tasks(envelope)
        if not open_tasks:
            return

        text = format_idle_tasks_nudge(open_tasks)

        # Structured ``_source_meta`` for the FE idle-tasks card, derived
        # from the same normalised ``open_tasks`` list the formatter
        # rendered into ``text`` — one source, so the card and the
        # model-facing prose cannot drift.  ``title`` / ``note`` are
        # sanitized identically to the formatter (``sanitize_name``) so a
        # crafted task can't reach the operator card by a route the nudge
        # body closes; the FE additionally renders every field via
        # ``textContent``.  Rows arrive from ``_open_tasks`` with every
        # field already a ``str`` (``None`` → ``""``, never ``"None"``).
        tasks_meta = [
            {
                "id": t["id"],
                "title": sanitize_name(t["title"]),
                "status": t["status"],
                "note": sanitize_name(t["note"]),
            }
            for t in open_tasks[:NUDGE_IDLE_TASKS_DISPLAY_CAP]
        ]

        # Bind identity + the enqueue-time OPEN SET by closure (never the
        # live ``ws``).  The id set makes the predicate reject a body
        # whose named tasks were all resolved while the entry waited —
        # "some other task is open" is not license to deliver a snapshot
        # naming only completed work; a fresh IDLE event will re-derive a
        # fresh body for the survivor set.
        bound_ws_id = ws_id
        bound_user_id = ws.user_id
        bound_open_ids = frozenset(t["id"] for t in open_tasks)

        def _still_valid() -> bool:
            # Half 1 — children.  Re-checked across the entry's whole
            # queued lifetime, not just at enqueue: the coord can spawn
            # children between enqueue and drain (worker owns the ws, a
            # cancel demoted the entry to the quiet channel), and
            # delivering "resume your task" beside live children is the
            # contradictory pair.  ``None`` (indeterminate) drops too —
            # ADVICE fails CLOSED, the exact opposite of the liveness
            # predicate above, because the harms point opposite ways.
            # NOTE the checked-first order: this is the correctness
            # half; the staleness half below is only about wording.
            now = self._active_children_now(bound_ws_id, bound_user_id)
            if now is None or now:
                return False
            # Half 2 — the task list.  Corrupt/unreadable → drop;
            # otherwise deliver only if at least one task the body NAMES
            # is still open.
            try:
                env, is_corrupt = load_task_envelope(self._storage, bound_ws_id)
            except Exception:
                log.debug(
                    "coord_idle_observer.predicate_tasks_failed ws=%s",
                    bound_ws_id[:8],
                    exc_info=True,
                )
                return False
            if is_corrupt:
                return False
            open_now = {t["id"] for t in self._open_tasks(env)}
            return bool(open_now & bound_open_ids)

        if self._enqueue_nudge(
            session,
            ws_id,
            "idle_tasks",
            text=text,
            metadata={"tasks": tasks_meta, "total": len(open_tasks)},
            valid_until=_still_valid,
            cooldown_secs=cooldown_secs,
        ):
            log.info(
                "coord_idle_observer.enqueued_tasks ws=%s open_tasks=%d",
                ws_id[:8],
                len(open_tasks),
            )

    # ------------------------------------------------------------------
    # storage reads
    # ------------------------------------------------------------------

    def _active_children(self, ws: Workstream) -> list[dict[str, str]] | None:
        """Query storage for the coord's interactive children whose state
        is in :data:`_ACTIVE_CHILD_STATES`.

        Returns a (possibly empty) list on a successful read, or
        ``None`` when the answer is INDETERMINATE — the query raised, or
        a row was too ragged to classify.  The distinction is
        load-bearing: ``[]`` licenses the advice path to fire ("provably
        no children"), so an error collapsing into ``[]`` would fire the
        resume-your-task nudge while children run.  The whole row walk
        sits inside the ``try`` for the same reason — a row missing
        ``state`` must yield "indeterminate", not escape as a
        ``KeyError`` that kills both nudge paths for the event.

        ``list_workstreams`` orders by ``updated DESC`` and applies its
        ``LIMIT`` in SQL before any state filter, so a coord with many
        recently-closed children could clip out genuinely-active rows
        whose ``updated`` timestamp is older.  We bump the limit well
        above ``NUDGE_IDLE_CHILDREN_WAIT_CAP`` to absorb that —
        realistic coord histories are far smaller than the bumped
        limit.  Pushing the state filter into SQL would be the
        structural fix, but that requires a storage-protocol change;
        flagged as a follow-up.
        """
        try:
            rows = self._storage.list_workstreams(
                limit=_ACTIVE_CHILDREN_QUERY_LIMIT,
                parent_ws_id=ws.id,
                kind=WorkstreamKind.INTERACTIVE,
                user_id=ws.user_id,
            )
            out: list[dict[str, str]] = []
            for row in rows:
                mapping = getattr(row, "_mapping", row)
                state = mapping["state"]
                if state not in _ACTIVE_CHILD_STATES:
                    continue
                out.append(
                    {
                        "ws_id": mapping["ws_id"],
                        "name": mapping["name"] or "",
                        "state": state,
                    }
                )
        except Exception:
            log.debug("coord_idle_observer.list_failed ws=%s", ws.id[:8], exc_info=True)
            return None
        return out

    def _active_children_now(self, ws_id: str, user_id: str) -> bool | None:
        """Drain-time answer to "does this coord have active children?"

        ``True`` / ``False`` on a successful read; ``None`` when the
        read was INDETERMINATE.  Callers map ``None`` to their own
        class's failure direction — liveness delivers, advice drops —
        which is why this returns a trichotomy instead of baking either
        answer in (a bool-returning helper here once inverted the
        fail-closed direction through a bare ``not``).

        Uses the ``count_workstreams_by_state`` aggregate rather than
        the row-fetching gate query: the children predicate arms on the
        chat-loop user-attach path for every coord, where a row fetch
        is real latency.  The aggregate has NO kind filter while the
        enqueue gate filters ``kind=INTERACTIVE`` — an asymmetry, but a
        near-theoretical one: only two kinds exist and nothing today
        creates a COORDINATOR row with a parent, so the two questions
        coincide.  If nested coordinators ever land, either push a kind
        param into the aggregate (protocol change) or accept that the
        divergence lands in each class's safe direction (liveness
        over-delivers, advice over-drops).
        """
        try:
            counts = self._storage.count_workstreams_by_state(
                parent_ws_id=ws_id,
                user_id=user_id,
            )
        except Exception:
            log.debug(
                "coord_idle_observer.predicate_count_failed ws=%s",
                ws_id[:8],
                exc_info=True,
            )
            return None
        return any(counts.get(s, 0) > 0 for s in _ACTIVE_CHILD_STATES)

    @staticmethod
    def _open_tasks(envelope: dict[str, Any]) -> list[dict[str, str]]:
        """Filter a decoded task envelope to the rows that count as
        unfinished work (``TASK_OPEN_STATUSES``), NORMALISED: every
        returned row carries exactly ``id`` / ``title`` / ``status`` /
        ``note``, all ``str``.

        This is the single coercion point for both downstream consumers
        (the nudge formatter and the FE card metadata), so the two can
        never disagree on a ragged row.  ``_field_str`` maps ``None`` →
        ``""`` — a bare ``str()`` here once rendered a JSON ``null``
        note as a literal ``None`` line in the operator card while the
        prose showed nothing.  The envelope is a JSON blob a hand-edited
        DB or an older writer can leave ragged, and
        ``load_task_envelope`` shape-checks only the envelope, not the
        rows.
        """
        rows = envelope.get("tasks") or []
        if not isinstance(rows, list):
            return []
        return [
            {
                "id": _field_str(row.get("id")),
                "title": _field_str(row.get("title")),
                "status": row["status"],
                "note": _field_str(row.get("note")),
            }
            for row in rows
            if isinstance(row, dict) and row.get("status") in TASK_OPEN_STATUSES
        ]

    # ------------------------------------------------------------------
    # cap accounting
    # ------------------------------------------------------------------

    def _cap_reached(self, ws_id: str, nudge_type: str) -> bool:
        """Cheap advisory peek at this type's per-bracket budget.

        May be stale by the time the fire happens — :meth:`_try_charge`
        at the enqueue position is the authority.  The peek exists so a
        capped-out coord short-circuits before the message walk and the
        storage reads, mirroring the ``_cooldown_allows`` /
        ``should_nudge`` peek-then-authoritative split.
        """
        with self._fire_counts_lock:
            ws_caps = self._fire_counts.get(ws_id, {})
            return ws_caps.get(nudge_type, 0) >= _NUDGE_TYPE_CAPS[nudge_type]

    def _try_charge(self, ws_id: str, nudge_type: str) -> bool:
        """Atomically check-and-charge one fire of *nudge_type*.

        The single lock hold closes the check-then-act window the
        separate peek + record pair left open (two concurrent IDLE
        events for one ws could both pass the peek and over-fire).
        Sits immediately before the enqueue — NOT at the peek position —
        so a refused fire (question heuristic, no children, cooldown)
        never burns budget: the cap counts nudges enqueued, not IDLE
        events observed.  The enqueue after a successful charge cannot
        fail (the queue is unbounded and the channel literal is valid),
        so no rollback path is needed.
        """
        cap = _NUDGE_TYPE_CAPS[nudge_type]
        with self._fire_counts_lock:
            ws_caps = self._fire_counts.setdefault(ws_id, {})
            if ws_caps.get(nudge_type, 0) >= cap:
                return False
            ws_caps[nudge_type] = ws_caps.get(nudge_type, 0) + 1
            return True

    def _reset_caps_for(self, ws_id: str) -> None:
        """Drop every nudge-type cap counter for ``ws_id`` on a real
        (non-wake) leave-IDLE event — the operator drove the coord
        forward, so the next idle bracket starts fresh for BOTH classes.
        Whole-ws pop keeps the O(1) ``ws_id in self._fire_counts`` fast
        path intact.
        """
        with self._fire_counts_lock:
            self._fire_counts.pop(ws_id, None)

    # ------------------------------------------------------------------
    # last-assistant-turn heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _last_assistant_turn(session: ChatSession) -> Turn | None:
        """The most recent ASSISTANT turn, or ``None`` for a fresh
        session.  Shared by both skip heuristics so the "first assistant
        turn walking back" rule lives in exactly one place — a future
        change to how that turn is located (skipping synthesized system
        turns, say) cannot fix one heuristic and silently leave the
        other on the old rule.
        """
        for msg in reversed(session.messages):
            if msg.role is Role.ASSISTANT:
                return msg
        return None

    def _last_assistant_used_wait(self, session: ChatSession) -> bool:
        """True when the coord's most recent assistant turn issued a
        ``wait_for_workstream`` tool call — it is already using the tool
        the liveness nudge would suggest.
        """
        msg = self._last_assistant_turn(session)
        if msg is None:
            return False
        return any(tc.name == "wait_for_workstream" for tc in msg.tool_calls)

    def _last_assistant_asked_operator(self, session: ChatSession) -> bool:
        """True when the coord's most recent assistant turn looks like a
        question put to the *operator* rather than a premature stop.

        A coord that ends its turn asking the operator something has
        stopped for the right reason, and nudging it to resume pushes it
        to answer its own question — the exact failure ``idle_tasks``
        exists to avoid.

        The heuristic is narrow on purpose.  A trailing ``?`` alone
        over-fires: a turn that called ``send_to_workstream`` and ended
        with a question was addressing a *child*, and a turn that ends
        "shall I check the config?" mid-work is rhetorical.  Requiring
        the turn to carry **no tool calls** removes the first class
        cheaply.  The second class survives, so this stays
        false-negative-biased by design: the cost of over-suppressing is
        the operator typing "continue", while the cost of
        under-suppressing is a coordinator guessing on a decision it
        correctly escalated.

        Reads ``Turn.text`` (which joins the turn's ``TextBlock``s)
        rather than ``Turn.content``, which is a tuple of content
        blocks.
        """
        msg = self._last_assistant_turn(session)
        if msg is None:
            return False
        if msg.tool_calls:
            return False
        return msg.text.rstrip().endswith("?")
