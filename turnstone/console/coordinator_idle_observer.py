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
  (``pending`` / ``in_progress``) entries on its own ``tasks`` list.
  Suggests reconciling the list, with the operator-escalation branch
  first.

The classes are INDEPENDENT: both conditions can hold in one IDLE
event, and both nudges then fire and CO-DELIVER in one drain, tasks
first — the grooming instruction is instant, the park instruction is
open-ended, so the batch ends on the wait.  Each nudge asserts only its
own domain: the tasks body never claims the children are gone (it
carries an explicit "children may still be running" line, because the
liveness nudge can be blocked by its own cap or wait gate while advice
fires alone), so a consistent pair is two true statements, not a
contradiction.  One ordering caveat, accepted: a cross-bracket pair (an
older queued ``idle_children`` surviving into a bracket that enqueues
``idle_tasks``) delivers children-first by seq; both entries are still
predicate-valid, so that is a tuning miss, not a correctness one.  If
evals show small models fumbling even the ordered pair, the named
upgrade path is a single combined checkpoint type selected at produce
time — do NOT reintroduce a cross-domain fire gate.

The class decides three behaviours, each ruled at its site:

* **Config gating.**  LIVENESS is never gated on ``memory.nudges`` —
  that switch's operator-facing help text promises memory-reminder
  control, and suppressing the wake would strand a coordinator whose
  children finish unobserved (see ``_maybe_enqueue_children``).  ADVICE
  gates through ``ChatSession._nudges_enabled``, which also suppresses
  the nudge when the persona envelope hides the ``tasks`` tool its body
  instructs (``NUDGE_REQUIRED_TOOL``).
* **Per-type caps** (:data:`_NUDGE_TYPE_CAPS`) — ONE fire per type per
  idle bracket, and no cooldown at all: an idle nudge has a single exit
  (enqueue → wake → delivered), so repeat fires would only re-prompt a
  model that already read the body, at one autonomous turn each.  The
  re-arm is operator progress, not the clock.  Both classes may fire in
  one bracket, so a drain carries at most 2 bodies.
* **Drain-predicate scope and failure direction.**  Each predicate
  re-validates ONLY its own assertion.  An indeterminate children read
  at drain DELIVERS a liveness entry (stale noise is cheap; a lost wake
  is a stall) — at ENQUEUE the same class fails closed.  The advice
  predicate never reads children at all: it drops iff no task the body
  names is still open, or the envelope cannot be read (ADVICE fails
  closed).

Gate order, ``idle_children`` (cheap → expensive; matches the code):
coordinator-kind → cap peek → wait-tool skip →
children query (``None``/``[]`` → no fire) → permission check
(``nudge_allowed``) → atomic charge → enqueue → record.

Gate order, ``idle_tasks``: coordinator-kind → operator-Stop gate
(``_generation_abandoned``) → ``_nudges_enabled`` (config +
``tasks``-tool visibility) → cap peek → asked-operator skip →
wait-tool skip → envelope read →
unidentifiable-set refusal (``bound_open_ids``) → permission check
(``nudge_allowed``) → atomic charge → enqueue → record.

``_on_idle`` runs the tasks path BEFORE the children path so a
same-event pair carries ascending seq in tasks-first order — every
drain path delivers in seq order, which is what makes the ordering
ruling hold with no queue changes.

Caps reset when the ws leaves IDLE for a non-wake reason (tracked by
``ChatSession._wake_source_tag``).  That state is per-process — if one
coordinator were ever live in two console processes the counters would
disagree across them (pre-existing shape, shared with every other
per-process cache).
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any

from turnstone.console.coordinator_client import TASK_OPEN_STATUSES, load_task_envelope
from turnstone.core.log import get_logger
from turnstone.core.metacognition import (
    NUDGE_IDLE_TASKS_DISPLAY_CAP,
    _field_str,
    format_idle_children_nudge,
    format_idle_tasks_nudge,
    nudge_allowed,
    record_nudge,
    sanitize_name,
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
# errored child without operator intervention; the per-bracket cap
# handles repeat fires for stuck-error children).
_ACTIVE_CHILD_STATES: frozenset[str] = frozenset(
    {
        WorkstreamState.THINKING.value,
        WorkstreamState.RUNNING.value,
        WorkstreamState.ATTENTION.value,
    }
)

# ONE fire per type per idle bracket.  This cap is the ONLY limiter on
# the idle nudges — they carry no cooldown, deliberately.
#
# An idle nudge has exactly ONE exit: enqueueing it makes the watcher
# dispatch a wake, and that wake delivers it.  Unlike the memory-class
# advisories — which are queued mid-turn and must wait for whichever
# seam arrives next, so repeat fires buy extra chances at DELIVERY —
# there is no second seam to wait for here.  Extra fires would only
# re-prompt a model that has already read the body once, and each one
# costs a full autonomous inference turn.  So the wall-clock cooldown
# those other types need has no job on this path, and removing it makes
# the cap the single, legible limiter: one reminder per stall.
#
# The re-arm is the operator, not the clock: ``_reset_caps_for`` clears
# the counters on a real (non-wake) leave-IDLE, i.e. when the operator
# actually moves the coordinator forward.  A coordinator that ignores
# its nudge stays quiet until then, which is the honest behaviour — it
# was told once.
#
# Still keyed per TYPE, not a shared total: advice must never be able to
# spend the liveness budget, or a coord that used its wake on a task
# reminder reaches the silent-stall state (live children, no wake left)
# sooner than before ``idle_tasks`` existed.  Both classes may fire in
# one bracket (co-delivery), so the per-bracket ceiling is 2 bodies.
#
# Every nudge type this observer emits MUST be registered here — the
# lookup KeyErrors on an unregistered type (surfacing via _on_state's
# ``log.exception`` and any test that drives the new path) rather than
# inheriting either class's budget silently.  Fail-loud is the point:
# defaulting a future liveness type to the advice budget (or vice
# versa) is exactly the misclassification this table exists to force a
# decision on.
_NUDGE_TYPE_CAPS: dict[str, int] = {
    "idle_children": 1,
    "idle_tasks": 1,
}

# Soft cap on the snapshot query.  Higher than ``WAIT_MAX_WS_IDS`` so
# the SQL ``LIMIT`` (applied before the Python state filter) doesn't
# clip genuinely-active children whose ``updated`` timestamp is older
# than recently-closed siblings.  Realistic coord histories are far
# smaller than this; if a coord ever exceeds it, the formatter still
# truncates to ``WAIT_MAX_WS_IDS`` for the model-facing suggestion.
_ACTIVE_CHILDREN_QUERY_LIMIT = 200

# TTL (seconds) on the drain-time children answer.
#
# The LIVENESS predicate reads the children state through one memoised
# call.  Up to the per-type cap of liveness entries can sit queued at
# once, and ``NudgeQueue.drain_entries`` evaluates each ``valid_until``
# independently — unmemoised, two reads microseconds apart can disagree
# (one raising → deliver, fail-open; the next succeeding → drop), which
# delivers one stale "children still running" body while silently
# dropping its same-class sibling.  One observation per drain pass keeps
# same-class entries coherent, and N queued entries cost one query, not
# N.
#
# One second is long enough to span a single ``drain_entries`` loop and
# short enough that no entry is judged on a meaningfully stale view.
_DRAIN_CHILDREN_TTL_SECONDS = 1.0


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
        # Drain-time children answers, ``ws_id -> (monotonic_stamp, answer)``.
        # Read by the liveness predicate so same-class entries in one
        # drain pass cannot disagree — see
        # :data:`_DRAIN_CHILDREN_TTL_SECONDS`.  Pruned on write, so it holds
        # at most the coords drained within the last TTL.
        self._drain_children: dict[str, tuple[float, bool | None]] = {}
        self._drain_children_lock = threading.Lock()

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

        ``idle_tasks`` runs FIRST so a same-event pair enqueues in
        tasks-then-children seq order — every drain path delivers by
        seq, and the co-delivered batch should END on the open-ended
        park instruction, not start with it (module docstring).

        The children query lives INSIDE the children path, after its
        cheap gates: most IDLE events short-circuit on the cap peek
        (microsecond dict lookups), and computing it eagerly here would
        put a ``list_workstreams`` round-trip on every coord state
        transition in the cluster.  The tasks path does not read
        children at all — each nudge asserts only its own domain.
        """
        ws = self._manager.get(ws_id)
        if ws is None or ws.session is None:
            return
        if ws.kind is not WorkstreamKind.COORDINATOR:
            return
        # Bind the non-Optional session — mypy's narrowing from the
        # check above does not reach into the paths' closures.
        session = ws.session

        # SEPARATE failure domains, deliberately two literal statements.
        #
        # The classes are independent by design — per-type caps exist so
        # the liveness budget is starvation-proof against advice — and a
        # shared ``try`` silently broke that: any raise inside the advice
        # path (an unregistered type KeyError from ``_NUDGE_TYPE_CAPS``, a
        # ragged envelope value, a non-mapping config row escaping
        # ``load_task_envelope``) suppressed the liveness WAKE for the
        # same event, stranding a coordinator whose children finish
        # unobserved.  That is the outcome the liveness class exists to
        # prevent, caused by the class it is supposed to be independent
        # of.
        #
        # Do NOT "tidy" these into a loop or a list of callables: the
        # tasks-then-children ORDER is the co-delivery ruling (tasks
        # enqueues first so it carries the lower seq and the batch ends
        # on the park instruction), and a collection makes that order an
        # accident of iteration rather than a statement.
        try:
            self._maybe_enqueue_tasks(ws, session)
        except Exception:
            log.exception("coord_idle_observer.advice_path_failed ws=%s", ws_id[:8])
        try:
            self._maybe_enqueue_children(ws, session)
        except Exception:
            log.exception("coord_idle_observer.liveness_path_failed ws=%s", ws_id[:8])

    # ------------------------------------------------------------------
    # shared gate head / enqueue tail
    # ------------------------------------------------------------------

    def _common_gates_allow(self, session: ChatSession, ws_id: str, nudge_type: str) -> bool:
        """The gate head both paths run identically: the per-type cap
        peek, a microsecond dict lookup.

        Returns ``True`` when the gates pass.  Kept in one place so a
        cross-cutting change to the cheap gates lands on both nudge
        types at once — this file has already paid once for editing the
        head twice.

        NO cooldown peek: the idle nudges are cap-only (see
        :data:`_NUDGE_TYPE_CAPS`).  ``memory.nudge_cooldown`` governs the
        memory-class advisories, which wait for a seam; these fire once
        per bracket and are re-armed by operator progress instead.
        """
        return not self._cap_reached(ws_id, nudge_type)

    def _enqueue_nudge(
        self,
        session: ChatSession,
        ws_id: str,
        nudge_type: str,
        *,
        text: str,
        metadata: dict[str, Any] | None,
        valid_until: Callable[[], bool],
    ) -> bool:
        """The shared enqueue tail: ``nudge_allowed`` (the authoritative
        permission check), the formatter empty-body guard, the atomic
        cap charge, the enqueue, and finally ``record_nudge``.

        One implementation for both types so the charge discipline —
        cheap peek early, authoritative :meth:`_try_charge` at the fire
        position — can never drift between paths.

        ``memory_count=0`` is deliberate, not a stub: ``nudge_allowed``
        reads ``memory_count`` only for the ``tool_error`` / ``resume``
        / ``start`` types, so paying 1-2 ``count_structured_memories``
        round-trips per IDLE event to compute the real number bought
        nothing.  If a future nudge type routed through here needs it,
        thread it explicitly.
        """
        # ``cooldown_secs=0`` — the idle nudges are cap-only.  This call
        # is still here for the gates a cap cannot express: unknown type,
        # and ``message_count <= 1`` (a rehydrated or freshly-truncated
        # session should not be nudged about work it cannot yet see).
        if not nudge_allowed(
            nudge_type,
            session._metacog_state,
            message_count=len(session.messages),
            memory_count=0,
            cooldown_secs=0,
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
        # Record LAST.  Inert for these two types (they run
        # ``cooldown_secs=0``), but the ORDER is the discipline: a
        # refused fire must never burn budget.  ``should_nudge`` stamps on success, so calling
        # it here would spend the 300s window at the permission check,
        # before ``_try_charge`` (the authoritative gate) has said yes:
        # the loser of a cap race then loses its next fire too, having
        # delivered nothing.  Reordering the CHARGE above the permission
        # check instead would be worse — ``nudge_allowed`` refuses for
        # reasons the cheap peek never sees (``message_count <= 1`` on a
        # rehydrated session, an unregistered type), so the cap would be
        # charged on every one of those refusals.  Permission, then
        # charge, then deliver, then record.
        # Stamp the fire.  With ``cooldown_secs=0`` this no longer GATES
        # anything for these two types — the cap does — but it keeps the
        # per-type timestamp other tooling reads, and it keeps the
        # permission/charge/deliver/record order intact so a future type
        # routed through here that DOES want a cooldown inherits the
        # correct discipline rather than the bug (#924).
        record_nudge(nudge_type, session._metacog_state)
        return True

    # ------------------------------------------------------------------
    # idle_children — LIVENESS
    # ------------------------------------------------------------------

    def _maybe_enqueue_children(self, ws: Workstream, session: ChatSession) -> None:
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
        point, and the body is a ROSTER — the child list is useful to a
        model that cannot call the suggested tool (inspect or message
        them, or simply carry on knowing work is in flight).  This
        asymmetry with the advice path (which IS visibility-gated) is
        deliberate: every ACTIONABLE line in the advice body is a
        ``tasks(...)`` call, so hiding ``tasks`` leaves it with nothing
        to say, while the liveness body still carries its facts.
        """
        ws_id = ws.id

        if not self._common_gates_allow(session, ws_id, "idle_children"):
            return

        # Gate: skip if the coord's last assistant turn already used
        # ``wait_for_workstream``.  Don't nudge toward a tool the
        # model is already using.
        if self._last_assistant_used_wait(session):
            return

        # Gate: the children query.  ``None`` means the read was
        # indeterminate (storage raised) — fail closed and say so,
        # rather than letting a falsy non-answer read as "no children".
        # ``[]`` is a REAL empty answer and also skips.
        active = self._active_children(ws)
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
            now = self._children_state_at_drain(bound_ws_id, bound_user_id)
            if now is None:
                # LIVENESS fails OPEN at drain: the entry only exists
                # because children were active at enqueue, and dropping
                # it on a storage blip silently abandons the wake — the
                # stalled-coordinator outcome this class exists to
                # prevent, with no retry behind it.  Delivering on an
                # indeterminate read risks only a stale "children still
                # running" nudge; if they finished, the suggested wait
                # returns immediately with their results.  The ENQUEUE
                # gate for this same class fails the other way (no fire
                # on an unknown read) — a queued entry represents an
                # already-charged fire, so the drop would waste it.
                return True
            return now

        if self._enqueue_nudge(
            session,
            ws_id,
            "idle_children",
            text=text,
            metadata={"children": children_meta},
            valid_until=_still_has_active_children,
        ):
            log.info(
                "coord_idle_observer.enqueued ws=%s active_children=%d",
                ws_id[:8],
                len(active),
            )

    # ------------------------------------------------------------------
    # idle_tasks — ADVICE
    # ------------------------------------------------------------------

    def _maybe_enqueue_tasks(self, ws: Workstream, session: ChatSession) -> None:
        """Enqueue ``idle_tasks`` when the coord went idle holding open
        tasks.

        Fires INDEPENDENTLY of the children state — beside a same-event
        ``idle_children`` (co-delivery, tasks first) or alone while
        children run and the liveness nudge is blocked by its own
        cooldown/cap/wait gate.  The body never presumes the children
        are done (its "children may still be running" line plus the
        blocked-on-child branch are what make advice-alone honest), so
        this path reads no children state at enqueue OR at drain.

        ADVICE class — gated on ``ChatSession._nudges_enabled``, which
        for this type means the ``memory.nudges`` config switch AND the
        persona envelope exposing the ``tasks`` tool
        (``NUDGE_REQUIRED_TOOL``): every branch of the nudge body is a
        ``tasks(...)`` call, so firing it at a persona that hides the
        tool produces "I don't have access" apology loops.
        """
        ws_id = ws.id

        # Gate: the operator stopped this generation.  An abandoned turn
        # ends in ``_emit_state("idle")``, and that IDLE reaches here —
        # so without this check pressing Stop on a coord holding open
        # tasks enqueues a wake-eligible entry AFTER the cancel path
        # demoted the queue to quiet, and the watcher resumes the
        # workstream seconds later.  The operator said stop; a task
        # reminder is not a reason to override that.
        #
        # ADVICE only.  Liveness deliberately still fires: a cancelled
        # coordinator can still have children running whose results
        # would otherwise be abandoned, which is the outcome that class
        # exists to prevent.
        if session._generation_abandoned:
            return

        if not session._nudges_enabled("idle_tasks"):
            return

        if not self._common_gates_allow(session, ws_id, "idle_tasks"):
            return

        # Gate: the coord's last assistant turn looks like a question put
        # to the operator.  Deliberately false-negative-biased — see the
        # method docstring; suppressing a legitimate nudge costs the
        # operator one "continue", while nudging a coord that correctly
        # escalated pushes it to guess on a decision that was not its to
        # make.
        if self._last_assistant_asked_operator(session):
            return

        # Gate: the coord's last assistant turn already used
        # ``wait_for_workstream`` — it parked on a known wake source,
        # and waking it to groom tasks defeats the park.  Park signals
        # gate BOTH nudge paths; domain conditions gate only their own.
        # Nearly inert at a natural end-of-turn IDLE (the send loop only
        # breaks when the last turn carried no tool calls), so the
        # non-redundant coverage is a session rehydrated mid-wait —
        # shipped for the shared park-gate shape, not to close a live
        # hole.
        if self._last_assistant_used_wait(session):
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

        # THE ASSERTED SET.  Everything downstream — the model-facing
        # body, the operator card's metadata, and the drain predicate's
        # identity set — reads ``shown`` and nothing else.  Three
        # independent derivations of "what this nudge is about" is what
        # let the body render a capped slice while the predicate guarded
        # the uncapped one, so a coord with more than the display cap of
        # tasks was woken with a body naming work it had finished.
        #
        # ``id`` stays RAW: it is the identity key the drain predicate
        # matches against a fresh read of storage, and sanitising one
        # side of that comparison makes the intersection empty forever.
        # ``id_display`` is the rendered form — the body interpolates it
        # into a bullet line, so an embedded newline there would forge a
        # sibling row exactly as an unsanitized title would.
        shown = [
            {
                "id": t["id"],
                "id_display": sanitize_name(t["id"]),
                "title": sanitize_name(t["title"]),
                "status": t["status"],
                "note": sanitize_name(t["note"]),
            }
            for t in open_tasks[:NUDGE_IDLE_TASKS_DISPLAY_CAP]
        ]
        total_open = len(open_tasks)

        text = format_idle_tasks_nudge(shown, total=total_open)

        # Structured ``_source_meta`` for the FE idle-tasks card, read
        # from the SAME ``shown`` rows the formatter rendered — the card
        # and the model-facing prose cannot drift because there is only
        # one list.  ``id`` is the display form here (the card renders
        # it; nothing on the FE matches identities), and the FE also
        # writes every field via ``textContent``.  The card carries the
        # id because ``title`` may legitimately be empty and the prose's
        # "(untitled)" fallback would otherwise leave the operator an
        # unidentifiable row — the card has no other identifying column.
        tasks_meta = [
            {
                "id": t["id_display"],
                "title": t["title"],
                "status": t["status"],
                "note": t["note"],
            }
            for t in shown
        ]

        # Bind identity + the ASSERTED id set by closure (never the live
        # ``ws``).  Scoped to ``shown``, not to every open task: the body
        # names only these, so resolving all of them makes the body stale
        # even if other work remains — a fresh IDLE event re-derives a
        # fresh body for the survivors.  This is a deliberate behaviour
        # change from binding the full open set, which let a coord with
        # more than the display cap of tasks be woken with a body listing
        # only finished ones.  Raw ids on both sides: ``_still_valid``
        # re-reads storage through the same ``_open_tasks`` normaliser,
        # so sanitising either side alone would empty the intersection
        # permanently and silently.
        bound_ws_id = ws_id
        bound_open_ids = frozenset(t["id"] for t in shown if t["id"])
        if not bound_open_ids:
            # No usable identity for anything the body names (every shown
            # row came from a ragged envelope with no ``id``), so the
            # drain predicate could never match and the entry would be
            # enqueued and then dropped at every drain — silently
            # spending a cap slot and stamping the 300s cooldown each
            # time, until the coordinator's whole advice budget is gone
            # and it is never reminded at all.  Refusing to make a claim
            # we cannot re-validate is the honest failure: it costs
            # nothing and leaves the budget for a well-formed list.
            log.debug(
                "coord_idle_observer.tasks_unidentifiable ws=%s shown=%d (no nudge)",
                ws_id[:8],
                len(shown),
            )
            return

        def _still_valid() -> bool:
            # The predicate re-validates ONLY what the body asserts:
            # that at least one task it NAMES is still open.  It reads
            # no children state — each nudge asserts its own domain
            # (module docstring), so an entry that outlives a bracket,
            # survives a Stop demotion, or is resurrected by a failed
            # wake and then delivers beside live children is two true
            # statements, not a contradiction; the body's "children may
            # still be running" line covers exactly that state.
            #
            # Corrupt/unreadable envelope → drop (ADVICE fails closed).
            # ``_open_tasks`` sits INSIDE the try: it walks raw envelope
            # rows, so a ragged one is a data-shape condition, and
            # letting it escape would surface as ``predicate_raised`` —
            # which ``nudge_queue`` documents as a wiring bug, sending
            # operators after a phantom code defect.
            try:
                env, is_corrupt = load_task_envelope(self._storage, bound_ws_id)
                if is_corrupt:
                    return False
                open_now = {t["id"] for t in self._open_tasks(env)}
            except Exception:
                log.debug(
                    "coord_idle_observer.predicate_tasks_failed ws=%s",
                    bound_ws_id[:8],
                    exc_info=True,
                )
                return False
            return bool(open_now & bound_open_ids)

        if self._enqueue_nudge(
            session,
            ws_id,
            "idle_tasks",
            text=text,
            metadata={"tasks": tasks_meta, "total": total_open},
            valid_until=_still_valid,
        ):
            log.info(
                "coord_idle_observer.enqueued_tasks ws=%s open_tasks=%d shown=%d",
                ws_id[:8],
                total_open,
                len(shown),
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

    def _children_state_at_drain(self, ws_id: str, user_id: str) -> bool | None:
        """Memoised drain-time answer to "does this coord have active
        children?", for the LIVENESS ``valid_until`` predicate.

        The memo earns its keep with one caller: up to the liveness
        per-type cap of entries can sit queued at once, and
        ``drain_entries`` evaluates each ``valid_until`` independently —
        unmemoised, entry 1 could deliver on a raise (fail-open) while
        entry 2's read succeeds and drops microseconds later, one stale
        body delivered beside one silently dropped.  One observation per
        drain pass keeps same-class entries coherent, and N queued
        entries cost one query, not N.
        """
        now = time.monotonic()
        with self._drain_children_lock:
            hit = self._drain_children.get(ws_id)
            if hit is not None and now - hit[0] < _DRAIN_CHILDREN_TTL_SECONDS:
                return hit[1]
        answer = self._active_children_now(ws_id, user_id)
        with self._drain_children_lock:
            # Prune while we hold the lock: entries live one TTL, so the
            # map cannot outgrow the set of coords drained in that window.
            stale = [
                k
                for k, (t, _) in self._drain_children.items()
                if now - t >= _DRAIN_CHILDREN_TTL_SECONDS
            ]
            for k in stale:
                del self._drain_children[k]
            self._drain_children[ws_id] = (now, answer)
        return answer

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
        divergence lands in the safe direction (liveness over-delivers;
        the advice predicate no longer reads children at all).
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
        out: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Coerce BEFORE the membership test.  ``TASK_OPEN_STATUSES``
            # is a frozenset, so a non-hashable value (``status: []`` in
            # a hand-edited blob) raises ``TypeError`` from the ``in``
            # itself — not caught as an unknown status but escaping the
            # whole nudge path, which the observer swallows, silencing
            # this coordinator's nudge permanently.  Coerced, it simply
            # fails to match and the row is skipped.
            status = _field_str(row.get("status"))
            if status not in TASK_OPEN_STATUSES:
                continue
            out.append(
                {
                    "id": _field_str(row.get("id")),
                    "title": _field_str(row.get("title")),
                    "status": status,
                    "note": _field_str(row.get("note")),
                }
            )
        return out

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
