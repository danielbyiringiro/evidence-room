"""
The Operator's Copilot -- an agent over the evidence layer.

The agent performs ONE reversible operational action: it drafts a *suggested
deduction* for a (question, rubric item, student proof), grounded in retrieved
guidance and precedent, for a human grader to review. Drafting is reversible (a
PENDING proposal); *applying* a proposal is the irreversible, grader-facing step
and is gated behind explicit human approval.

The design goal is the Days 2-3 theme: enough autonomy to create leverage, enough
structure to stay safe. Safety is not a prompt instruction here -- it is control
flow. The required controls map to concrete mechanisms:

  authorization boundary   -- only permitted roles may invoke the draft action;
                              the role is supplied by the system, never inferred
                              from the proof text (defeats role-escalation
                              injections). See ACTORS.
  dry-run mode (default)   -- dry_run=True computes and returns the draft and
                              writes an audit entry, but persists NO proposal;
                              nothing downstream can see a state change.
  human approval           -- apply() flips PENDING -> APPLIED, requires an
                              approver, and refuses in dry-run. The reversible
                              draft is automatable; the irreversible apply is not.
  timeout + retry          -- every tool call goes through _invoke_tool: a wall
                              clock timeout, bounded retries on TransientToolError,
                              and a STRUCTURED error on exhaustion (never a raw
                              exception bubbling into a half-done action).
  audit log                -- append-only JSONL: timestamp, user, role, intent,
                              tool, arguments (student text hashed, not copied),
                              result, approval.
  idempotency              -- a key over (user, question, item, proof, intent)
                              makes duplicate execution a no-op: the existing
                              proposal is returned instead of a second one, and
                              apply() on an already-applied proposal is a no-op.

Missing evidence, ambiguity, and malicious instructions are handled by REFUSING
to draft and escalating to a human -- the agent never invents a judgment, and the
retrieval scope/permission filters mean an injected instruction cannot widen what
is retrievable or exfiltrate out-of-scope data.

Drafts are composed deterministically from the approved rubric guidance + the
majority expert label among retrieved precedent. No student proof text is sent to
any external model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import concurrent.futures
import hashlib
import json
import threading
import time

from .retrieval import (
    HybridRetriever, RetrievalResult, RefusalPolicy,
    ANSWER, REFUSE, FACULTY,
)

# Which roles may invoke which agent action. The authorization boundary is here,
# independent of (and stricter than) retrieval's read permissions: even though a
# trainee's retrieval would refuse anyway, the agent blocks before any tool runs.
ACTORS: dict[str, set[str]] = {
    "draft_deduction": {FACULTY},
}

# Agent outcome statuses.
DRAFTED = "drafted"
REFUSED = "refused"            # evidence insufficient / not permitted -> escalate
ESCALATED = "escalated"        # ambiguous or conflicting -> human review
DUPLICATE = "duplicate"        # idempotent no-op
UNAUTHORIZED = "unauthorized"  # role may not perform this action
TOOL_ERROR = "tool_error"      # tool failed under timeout/retry -> escalate
NEEDS_APPROVAL = "needs_approval"
APPLIED = "applied"

PENDING = "pending"
APPLIED_STATUS = "applied"


class TransientToolError(Exception):
    """A tool failure that is worth retrying (e.g., a timeout, a flaky backend)."""


# --------------------------------------------------------------------------
# structured tool-call wrapper: timeout + bounded retry + structured error
# --------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok: bool
    tool: str
    attempts: int
    value: Any = None
    error: dict | None = None       # {"type": ..., "message": ...} when ok is False


def _invoke_tool(name: str, fn: Callable, *args,
                 timeout_s: float = 10.0, max_attempts: int = 3,
                 backoff_s: float = 0.05, **kwargs) -> ToolResult:
    """
    Run a tool with a wall-clock timeout and bounded retries.

    Retries only on TransientToolError or timeout; any other exception is a
    non-retryable structured error returned immediately. On exhaustion the caller
    gets a ToolResult with ok=False and a structured error -- never a raw traceback
    mid-action. (A timed-out worker thread cannot be force-killed; for the local,
    fast retrieval tool this is acceptable and the result is simply discarded.)
    """
    last: tuple[str, str] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn, *args, **kwargs)
                value = fut.result(timeout=timeout_s)
            return ToolResult(ok=True, tool=name, attempts=attempt, value=value)
        except concurrent.futures.TimeoutError:
            last = ("timeout", f"{name} exceeded {timeout_s}s")
        except TransientToolError as e:
            last = ("transient", str(e))
        except Exception as e:  # non-retryable
            return ToolResult(ok=False, tool=name, attempts=attempt,
                              error={"type": type(e).__name__, "message": str(e)})
        if attempt < max_attempts:
            time.sleep(backoff_s * attempt)
    return ToolResult(ok=False, tool=name, attempts=max_attempts,
                      error={"type": last[0], "message": last[1]})


# --------------------------------------------------------------------------
# data models
# --------------------------------------------------------------------------

@dataclass
class Proposal:
    proposal_id: str
    idempotency_key: str
    user: str
    role: str
    question_key: str
    rubric_item: str
    suggested_label: str
    rationale: str
    citations: list[dict]
    confidence: float
    status: str = PENDING
    created_at: str = ""
    applied_by: str | None = None
    applied_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentOutcome:
    status: str
    reason: str | None = None
    proposal: Proposal | None = None
    tool_error: dict | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "proposal": None if self.proposal is None else self.proposal.to_dict(),
            "tool_error": self.tool_error,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------

class OperatorCopilot:
    """Agent that drafts suggested deductions over the evidence layer."""

    def __init__(self, retriever: HybridRetriever,
                 log_dir: Path | str,
                 k: int = 5,
                 tool_timeout_s: float = 10.0,
                 max_attempts: int = 3,
                 majority_margin: int = 1):
        self.retriever = retriever
        self.k = k
        self.tool_timeout_s = tool_timeout_s
        self.max_attempts = max_attempts
        self.majority_margin = majority_margin

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.log_dir / "audit.jsonl"
        self.proposal_path = self.log_dir / "proposals.jsonl"
        self._lock = threading.Lock()

    # -- audit -------------------------------------------------------------

    def _audit(self, *, user: str, role: str, intent: str, tool: str,
               arguments: dict, result: dict, approval: str | None,
               dry_run: bool) -> None:
        entry = {
            "timestamp": _now(),
            "user": user,
            "role": role,
            "intent": intent,
            "tool": tool,
            "arguments": arguments,       # student text is hashed, never copied
            "result": result,
            "approval": approval,
            "dry_run": dry_run,
        }
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    @staticmethod
    def _sanitize_args(question_key: str, rubric_item: str, role: str,
                       proof_text: str, intent: str) -> dict:
        """Record WHAT was requested without copying student proof text into logs."""
        return {
            "question_key": question_key,
            "rubric_item": rubric_item,
            "role": role,
            "intent": intent,
            "proof_sha256": _sha(proof_text),
            "proof_chars": len(proof_text),
        }

    # -- proposal store ----------------------------------------------------

    def _load_proposals(self) -> list[dict]:
        if not self.proposal_path.exists():
            return []
        out = []
        with self.proposal_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _find_by_key(self, key: str) -> Proposal | None:
        # last write wins, so scan and keep the latest record for this key
        latest = None
        for row in self._load_proposals():
            if row["idempotency_key"] == key:
                latest = row
        return None if latest is None else Proposal(**latest)

    def _find_by_id(self, proposal_id: str) -> Proposal | None:
        latest = None
        for row in self._load_proposals():
            if row["proposal_id"] == proposal_id:
                latest = row
        return None if latest is None else Proposal(**latest)

    def _persist(self, proposal: Proposal) -> None:
        with self._lock:
            with self.proposal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(proposal.to_dict()) + "\n")

    # -- deterministic draft composer --------------------------------------

    def _compose(self, result: RetrievalResult) -> tuple[str | None, str, list[dict], float]:
        """
        Return (suggested_label, rationale, citations, confidence).

        suggested_label is the MAJORITY expert label among retrieved precedent. If
        there is no clear majority (a tie within margin), returns None so the agent
        escalates a conflicting case rather than guessing.
        """
        labels = [e.provenance.get("label") for e in result.exemplars]
        counts = Counter(labels)
        ranked = counts.most_common()
        top_label, top_n = ranked[0]
        second_n = ranked[1][1] if len(ranked) > 1 else 0
        suggested = top_label if (top_n - second_n) >= self.majority_margin else None

        citations = []
        if result.guidance is not None:
            citations.append({"chunk_id": result.guidance.chunk_id,
                              "kind": "rubric_guidance"})
        for e in result.exemplars:
            citations.append({
                "chunk_id": e.chunk_id, "kind": "precedent",
                "label": e.provenance.get("label"),
                "score": round(float(e.score), 6),
            })

        guide = result.guidance.text if result.guidance else ""
        if suggested is None:
            rationale = (f"Precedent on '{result.rubric_item}' is split "
                         f"({dict(counts)}); no confident majority. Escalating for "
                         f"human judgment.")
        else:
            rationale = (
                f"Suggested judgment: {suggested} on '{result.rubric_item}'. "
                f"Basis: {top_n} of {len(labels)} closely-matching prior proofs on "
                f"this item were graded '{suggested}'. Rubric guidance: "
                f"{guide[:220]}"
            )
        return suggested, rationale, citations, float(result.confidence)

    # -- main loop ---------------------------------------------------------

    def draft(self, *, user: str, role: str, question_key: str, rubric_item: str,
              proof_text: str, intent: str = "draft_deduction",
              dry_run: bool = True,
              policy: RefusalPolicy | None = None) -> AgentOutcome:
        """
        Plan -> authorize -> (idempotency) -> retrieve -> verify -> act -> audit.

        Reversible: produces a PENDING proposal (or none). Never applies it.
        dry_run defaults True: computes the draft, audits it, persists nothing.
        """
        args = self._sanitize_args(question_key, rubric_item, role, proof_text, intent)

        def done(outcome: AgentOutcome, tool: str, approval: str | None = None):
            self._audit(user=user, role=role, intent=intent, tool=tool,
                        arguments=args,
                        result={"status": outcome.status, "reason": outcome.reason,
                                "proposal_id": None if not outcome.proposal
                                else outcome.proposal.proposal_id,
                                "tool_error": outcome.tool_error},
                        approval=approval, dry_run=dry_run)
            return outcome

        # 1) authorization boundary -- role is trusted from the system, not the proof
        if role not in ACTORS.get(intent, set()):
            return done(AgentOutcome(UNAUTHORIZED,
                                     reason=f"role '{role}' may not perform '{intent}'"),
                        tool="authorize")

        # 2) ambiguity / missing inputs -> hold for a human, do not guess
        if not (question_key and rubric_item and proof_text.strip()):
            return done(AgentOutcome(ESCALATED,
                                     reason="ambiguous or incomplete request; "
                                            "missing question, item, or proof"),
                        tool="validate")

        # 3) idempotency -> duplicate execution is a no-op
        key = _sha("|".join([user, question_key, rubric_item, intent, proof_text]))
        existing = self._find_by_key(key)
        if existing is not None:
            return done(AgentOutcome(DUPLICATE,
                                     reason="request already processed; "
                                            "returning existing proposal",
                                     proposal=existing),
                        tool="idempotency")

        # 4) retrieve evidence through the guarded tool wrapper
        tr = _invoke_tool(
            "retrieve_evidence", self.retriever.retrieve,
            question_key, rubric_item, proof_text,
            timeout_s=self.tool_timeout_s, max_attempts=self.max_attempts,
            k=self.k, role=role, policy=policy,
        )
        if not tr.ok:
            return done(AgentOutcome(TOOL_ERROR,
                                     reason="evidence tool failed; escalating",
                                     tool_error=tr.error),
                        tool="retrieve_evidence")
        result: RetrievalResult = tr.value

        # 5) verify: missing evidence / not permitted -> refuse to draft, escalate
        if result.decision == REFUSE:
            return done(AgentOutcome(REFUSED,
                                     reason=f"insufficient evidence to draft "
                                            f"({result.refusal_reason})"),
                        tool="retrieve_evidence")

        suggested, rationale, citations, confidence = self._compose(result)
        # conflicting precedent -> escalate rather than assert a judgment
        if suggested is None:
            return done(AgentOutcome(ESCALATED,
                                     reason="conflicting precedent; human review needed"),
                        tool="compose")

        # 6) act: build the reversible proposal
        proposal = Proposal(
            proposal_id=key[:16],
            idempotency_key=key,
            user=user, role=role, question_key=question_key, rubric_item=rubric_item,
            suggested_label=suggested, rationale=rationale, citations=citations,
            confidence=confidence, status=PENDING, created_at=_now(),
        )
        if not dry_run:
            self._persist(proposal)
        return done(AgentOutcome(DRAFTED, reason=None, proposal=proposal),
                    tool="draft_deduction")

    # -- irreversible step: apply, gated by human approval -----------------

    def apply(self, *, proposal_id: str, approver: str,
              dry_run: bool = False) -> AgentOutcome:
        """
        Apply a proposal (the irreversible, grader-facing step). Requires an
        approver and refuses in dry-run. Idempotent: applying an already-applied
        proposal returns it unchanged.
        """
        proposal = self._find_by_id(proposal_id)
        args = {"proposal_id": proposal_id}
        if proposal is None:
            out = AgentOutcome(TOOL_ERROR, reason="no such proposal")
            self._audit(user=approver, role="approver", intent="apply_deduction",
                        tool="apply", arguments=args,
                        result={"status": out.status, "reason": out.reason},
                        approval=None, dry_run=dry_run)
            return out

        if not approver or dry_run:
            out = AgentOutcome(NEEDS_APPROVAL,
                               reason="apply is irreversible; requires an approver "
                                      "and dry_run=False", proposal=proposal)
            self._audit(user=approver or "?", role="approver",
                        intent="apply_deduction", tool="apply", arguments=args,
                        result={"status": out.status, "reason": out.reason,
                                "proposal_id": proposal_id},
                        approval=None, dry_run=dry_run)
            return out

        if proposal.status == APPLIED_STATUS:      # idempotent apply
            out = AgentOutcome(APPLIED, reason="already applied", proposal=proposal)
            self._audit(user=approver, role="approver", intent="apply_deduction",
                        tool="apply", arguments=args,
                        result={"status": out.status, "reason": out.reason,
                                "proposal_id": proposal_id},
                        approval=approver, dry_run=dry_run)
            return out

        proposal.status = APPLIED_STATUS
        proposal.applied_by = approver
        proposal.applied_at = _now()
        self._persist(proposal)                    # last-write-wins record
        out = AgentOutcome(APPLIED, reason=None, proposal=proposal)
        self._audit(user=approver, role="approver", intent="apply_deduction",
                    tool="apply", arguments=args,
                    result={"status": out.status, "proposal_id": proposal_id},
                    approval=approver, dry_run=dry_run)
        return out


if __name__ == "__main__":
    import argparse
    from .embeddings import DEFAULT_INDEX

    ap = argparse.ArgumentParser(description="Operator's Copilot: draft a suggested deduction")
    ap.add_argument("--chunks", default="chunks_real.jsonl")
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--user", required=True)
    ap.add_argument("--role", default=FACULTY)
    ap.add_argument("--question", required=True)
    ap.add_argument("--item", required=True)
    ap.add_argument("--proof", required=True)
    ap.add_argument("--commit", action="store_true", help="persist the proposal (disable dry-run)")
    args = ap.parse_args()

    retr = HybridRetriever(args.chunks, index_path=args.index)
    agent = OperatorCopilot(retr, log_dir=args.logs)
    outcome = agent.draft(user=args.user, role=args.role, question_key=args.question,
                          rubric_item=args.item, proof_text=args.proof,
                          dry_run=not args.commit)
    print(f"\nStatus: {outcome.status.upper()}"
          + (f"  ({outcome.reason})" if outcome.reason else ""))
    if outcome.proposal:
        p = outcome.proposal
        print(f"Proposal {p.proposal_id}: suggest '{p.suggested_label}' "
              f"on '{p.rubric_item}'  (confidence {p.confidence:.3f}, status {p.status})")
        print(f"Rationale: {p.rationale}")
        print(f"Citations: {[c['chunk_id'] for c in p.citations]}")
    if outcome.tool_error:
        print(f"Tool error: {outcome.tool_error}")
    print(f"\nAudit: {agent.audit_path}")
