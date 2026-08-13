"""
Offline evaluation harness for the evidence layer.

Runs the eval set (eval/eval_set.jsonl) through the retriever, logs a retrieval
trace per case, computes metrics, calibrates the refusal threshold, and writes a
before/after report with a failure taxonomy.

WHAT "BEFORE/AFTER" MEANS HERE
  before  = dense-only retrieval, no similarity refusal (the naive baseline)
  after   = hybrid (dense + BM25) retrieval, refusal calibrated from this eval
The gap between them is the measured value of the two design choices under test:
sparse fusion (retrieval quality) and the refusal threshold (refusal quality).

METRICS (adapted to a retrieval-only, grading-precedent setting)
  refusal quality   -- do we ANSWER the answerable and REFUSE the unanswerable?
                       reported as accuracy plus over-refusal / missed-refusal.
  label precision@k -- of the top-k precedent, what fraction carries the label a
                       correct deduction needs (a groundedness proxy: retrieved
                       precedent should support the right judgment).
  label recall@k    -- is at least one correctly-labelled precedent in the top-k?
  conflict surfaced -- on genuinely mixed items, does the top-k show >1 label
                       rather than hiding the disagreement?
  scope/permission leakage -- MUST be zero: nothing out of (question, item) scope
                       or visible to a role that may not see it. This is the hard
                       invariant, checked on every case including adversarial.

Because the eval proofs are hand-written (never real student text), eval_set.jsonl
is safe to commit; the traces log chunk ids/labels/scores but no proof text and no
grader names, and are gitignored regardless.

Run (from src/, with the corpus + embedding index built):
    python -m evidence_room.evaluate
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import statistics

from .retrieval import (
    HybridRetriever, RefusalPolicy, ANSWER, REFUSE,
    R_LOW_SIMILARITY, FACULTY, TRAINEE,
)

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
DEFAULT_EVAL_SET = EVAL_DIR / "eval_set.jsonl"
DEFAULT_TRACE_DIR = EVAL_DIR / "traces"
DEFAULT_REPORT = EVAL_DIR / "report.md"

# never let similarity trigger a refusal while collecting traces; decisions at any
# threshold are reconstructed analytically afterwards from the recorded confidence.
_ALWAYS_ANSWER = RefusalPolicy(min_similarity=-1e9)
_HARD_REASONS = {"unknown_rubric_item", "no_candidates_in_scope", "not_permitted_for_role"}
SWEEP_GRID = [round(0.05 * i, 2) for i in range(0, 20)]  # 0.00 .. 0.95


# --------------------------------------------------------------------------
# running the eval set -> traces
# --------------------------------------------------------------------------

def load_eval_set(path: Path | str = DEFAULT_EVAL_SET) -> list[dict]:
    cases = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_traces(retriever: HybridRetriever, cases: list[dict],
               signals: tuple[str, ...], k: int = 5) -> list[dict]:
    """One trace per case. Uses the always-answer policy so the recorded decision
    reflects only structural refusals (unknown item / no candidates / permission);
    similarity refusals are applied later at a chosen threshold."""
    traces = []
    for c in cases:
        res = retriever.retrieve(
            c["question_key"], c["rubric_item"], c["proof"],
            k=k, role=c.get("role", FACULTY), policy=_ALWAYS_ANSWER, signals=signals,
        )
        ex = [{
            "chunk_id": e.chunk_id,
            "label": e.provenance.get("label"),
            "question_key": e.provenance.get("question_key"),
            "rubric_item": e.provenance.get("rubric_item"),
            "score": round(float(e.score), 6),
        } for e in res.exemplars]

        hard = res.refusal_reason if (res.decision == REFUSE and
                                      res.refusal_reason in _HARD_REASONS) else None
        # leakage guard: anything the query scope / role should have excluded
        leaked = [e for e in ex
                  if e["question_key"] != c["question_key"]
                  or e["rubric_item"] != c["rubric_item"]]
        if c.get("role") == TRAINEE and ex:
            leaked = ex  # a trainee should never receive exemplars at all

        traces.append({
            "id": c["id"],
            "category": c["category"],
            "role": c.get("role", FACULTY),
            "question_key": c["question_key"],
            "rubric_item": c["rubric_item"],
            "signals": list(signals),
            "hard_refuse": hard,
            "confidence": round(float(res.confidence), 6),
            "n_candidates": res.n_candidates,
            "guidance_id": None if res.guidance is None else res.guidance.chunk_id,
            "exemplars": ex,
            "leaked": leaked,
        })
    return traces


def decision_at(trace: dict, threshold: float) -> tuple[str, str | None]:
    """Reconstruct the (decision, reason) a given threshold would have produced."""
    if trace["hard_refuse"]:
        return REFUSE, trace["hard_refuse"]
    if trace["confidence"] < threshold:
        return REFUSE, R_LOW_SIMILARITY
    return ANSWER, None


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _expected_labels(case: dict) -> set[str]:
    e = case.get("expect", {})
    if "label" in e:
        return {e["label"]}
    return set(e.get("labels", []))


def refusal_metrics(cases: list[dict], traces: dict[str, dict],
                    threshold: float) -> dict:
    """Answer-the-answerable / refuse-the-unanswerable, over the two decisive
    categories. Conflicting and adversarial have their own metrics."""
    tp = fp = tn = fn = 0
    over_refused, missed = [], []
    for c in cases:
        exp = c.get("expect", {}).get("decision")
        if c["category"] not in ("answerable", "unanswerable") or exp is None:
            continue
        dec, _ = decision_at(traces[c["id"]], threshold)
        should_answer = exp == ANSWER
        if should_answer and dec == ANSWER:
            tp += 1
        elif should_answer and dec == REFUSE:
            fn += 1; over_refused.append(c["id"])
        elif not should_answer and dec == REFUSE:
            tn += 1
        else:
            fp += 1; missed.append(c["id"])
    n = tp + fp + tn + fn
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "answered_answerable": tp,
        "over_refused": over_refused,          # answerable we wrongly refused
        "missed_refusals": missed,             # unanswerable we wrongly answered
        "refused_unanswerable": tn,
    }


def label_metrics(cases: list[dict], traces: dict[str, dict], k: int = 5) -> dict:
    """Label precision/recall@k over cases that carry an expected label."""
    precs, recs = [], []
    for c in cases:
        exp = _expected_labels(c)
        if not exp:
            continue
        ex = traces[c["id"]]["exemplars"][:k]
        if not ex:
            precs.append(0.0); recs.append(0.0); continue
        hits = [e for e in ex if e["label"] in exp]
        precs.append(len(hits) / len(ex))
        recs.append(1.0 if hits else 0.0)
    return {
        "n": len(precs),
        "precision_at_k": statistics.mean(precs) if precs else 0.0,
        "recall_at_k": statistics.mean(recs) if recs else 0.0,
    }


def conflict_metrics(cases: list[dict], traces: dict[str, dict], k: int = 5) -> dict:
    surfaced = 0
    total = 0
    for c in cases:
        if c["category"] != "conflicting":
            continue
        total += 1
        labels = {e["label"] for e in traces[c["id"]]["exemplars"][:k]}
        if len(labels) >= 2:
            surfaced += 1
    return {"n": total, "surfaced_rate": surfaced / total if total else 0.0}


def leakage_report(cases: list[dict], traces: dict[str, dict]) -> dict:
    total = 0
    offenders = []
    adversarial_total = 0
    for c in cases:
        leaked = traces[c["id"]]["leaked"]
        if leaked:
            total += len(leaked)
            offenders.append(c["id"])
        if c["category"] == "adversarial" and leaked:
            adversarial_total += len(leaked)
    return {"leaked_chunks": total, "offending_cases": offenders,
            "adversarial_leaks": adversarial_total}


def threshold_sweep(cases: list[dict], traces: dict[str, dict],
                    grid: list[float]) -> list[dict]:
    """Balanced accuracy across the threshold-sensitive decision:
      positives = should ANSWER (answerable + conflicting)
      negatives = should REFUSE on similarity (the off-topic unanswerables)
    Structural refusals are threshold-independent and excluded here."""
    positives = [c["id"] for c in cases
                 if c["category"] in ("answerable", "conflicting")]
    negatives = [c["id"] for c in cases
                 if c.get("expect", {}).get("reason") == R_LOW_SIMILARITY]
    rows = []
    for t in grid:
        ans_pos = sum(decision_at(traces[i], t)[0] == ANSWER for i in positives)
        ref_neg = sum(decision_at(traces[i], t)[0] == REFUSE for i in negatives)
        tpr = ans_pos / len(positives) if positives else 0.0
        tnr = ref_neg / len(negatives) if negatives else 0.0
        rows.append({
            "threshold": t,
            "answer_rate_positives": round(tpr, 3),
            "refuse_rate_negatives": round(tnr, 3),
            "balanced_accuracy": round(0.5 * (tpr + tnr), 3),
        })
    return rows


def best_threshold(sweep: list[dict]) -> float:
    # highest balanced accuracy; tie-break toward the lower threshold (less over-refusal)
    return max(sweep, key=lambda r: (r["balanced_accuracy"], -r["threshold"]))["threshold"]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass
class ConfigResult:
    name: str
    signals: tuple[str, ...]
    threshold: float
    traces: dict[str, dict]
    refusal: dict
    labels: dict
    conflict: dict
    leakage: dict


def evaluate_config(name: str, retriever: HybridRetriever, cases: list[dict],
                    signals: tuple[str, ...], threshold: float, k: int) -> ConfigResult:
    traces = {t["id"]: t for t in run_traces(retriever, cases, signals, k)}
    return ConfigResult(
        name=name, signals=signals, threshold=threshold, traces=traces,
        refusal=refusal_metrics(cases, traces, threshold),
        labels=label_metrics(cases, traces, k),
        conflict=conflict_metrics(cases, traces, k),
        leakage=leakage_report(cases, traces),
    )


def _failure_taxonomy(cases: list[dict], after: ConfigResult) -> list[tuple[str, str, list[str]]]:
    """Named failure modes with the case ids that tripped them under the after-config."""
    tr = after.traces
    by_id = {c["id"]: c for c in cases}

    over = after.refusal["over_refused"]
    missed = after.refusal["missed_refusals"]
    leak = after.leakage["offending_cases"]
    # label mismatch: answerable answered but top-1 label not in expected
    mism = []
    for c in cases:
        if c["category"] != "answerable":
            continue
        dec, _ = decision_at(tr[c["id"]], after.threshold)
        ex = tr[c["id"]]["exemplars"]
        if dec == ANSWER and ex and ex[0]["label"] not in _expected_labels(c):
            mism.append(c["id"])
    # duplicate crowding: same submission_id appears >1 in top-k
    dup = []
    for c in cases:
        subs = [e["chunk_id"].split("::")[0] for e in tr[c["id"]]["exemplars"]]
        if len(subs) != len(set(subs)):
            dup.append(c["id"])

    return [
        ("F1 over-refusal (threshold too high)",
         "answerable queries refused because no precedent cleared the similarity floor", over),
        ("F2 missed refusal (threshold too low / injection)",
         "unanswerable queries answered when they should have been declined", missed),
        ("F3 label mismatch (wrong-judgment precedent on top)",
         "top precedent carried a label the correct deduction does not need", mism),
        ("F4 duplicate-submission crowding",
         "the 7x per-submission duplication surfaced the same submission more than once in top-k", dup),
        ("F5 scope/permission leakage (must be 0)",
         "an exemplar out of scope or visible to a role that may not see it", leak),
    ]


def write_report(path: Path, cases: list[dict], before: ConfigResult,
                 after: ConfigResult, sweep: list[dict], k: int) -> None:
    cat = Counter(c["category"] for c in cases)
    L = []
    L.append("# Evidence Room -- retrieval evaluation report\n")
    L.append("_Generated by `python -m evidence_room.evaluate`. Reproducible: "
             "same eval set + index yields the same numbers._\n")

    L.append("## Eval set\n")
    L.append(f"{len(cases)} cases across four classes: "
             + ", ".join(f"{v} {kk}" for kk, v in sorted(cat.items())) + ".\n")
    L.append("Proofs are hand-written (no student text), so the set is committable; "
             "traces log ids/labels/scores only.\n")

    L.append("## Before / after\n")
    L.append(f"- **before** -- {before.name} (signals={list(before.signals)}, "
             f"no similarity refusal)\n")
    L.append(f"- **after** -- {after.name} (signals={list(after.signals)}, "
             f"refusal threshold = {after.threshold})\n")
    L.append("\n| metric | before | after |")
    L.append("|---|---|---|")
    L.append(f"| refusal accuracy (answerable+unanswerable) | "
             f"{before.refusal['accuracy']:.2f} | {after.refusal['accuracy']:.2f} |")
    L.append(f"| over-refusals (answerable wrongly refused) | "
             f"{len(before.refusal['over_refused'])} | {len(after.refusal['over_refused'])} |")
    L.append(f"| missed refusals (unanswerable wrongly answered) | "
             f"{len(before.refusal['missed_refusals'])} | {len(after.refusal['missed_refusals'])} |")
    L.append(f"| label precision@{k} | {before.labels['precision_at_k']:.2f} | "
             f"{after.labels['precision_at_k']:.2f} |")
    L.append(f"| label recall@{k} | {before.labels['recall_at_k']:.2f} | "
             f"{after.labels['recall_at_k']:.2f} |")
    L.append(f"| conflict surfaced rate | {before.conflict['surfaced_rate']:.2f} | "
             f"{after.conflict['surfaced_rate']:.2f} |")
    L.append(f"| scope/permission leaks | {before.leakage['leaked_chunks']} | "
             f"{after.leakage['leaked_chunks']} |")
    L.append("")

    L.append("## Refusal threshold calibration\n")
    L.append("Balanced accuracy of the similarity-refusal decision "
             "(positives = should answer; negatives = off-topic, should refuse). "
             f"Recommended `EVIDENCE_ROOM_MIN_SIM` = **{after.threshold}**.\n")
    L.append("| threshold | answer-rate (pos) | refuse-rate (neg) | balanced acc |")
    L.append("|---|---|---|---|")
    for r in sweep:
        star = "  <-- chosen" if r["threshold"] == after.threshold else ""
        L.append(f"| {r['threshold']:.2f} | {r['answer_rate_positives']:.2f} | "
                 f"{r['refuse_rate_negatives']:.2f} | {r['balanced_accuracy']:.2f}{star} |")
    L.append("")

    L.append("## Failure taxonomy\n")
    L.append("_Counts under the after-config. F5 is the hard invariant and must stay 0._\n")
    for name, desc, ids in _failure_taxonomy(cases, after):
        L.append(f"- **{name}** -- {desc}. Count: {len(ids)}"
                 + (f" ({', '.join(ids)})" if ids else "") + ".")
    L.append("")

    L.append("## Next highest-leverage improvement\n")
    L.append(
        "Whichever failure mode above carries the most cases is where to spend "
        "next. By construction the layout points at **partial-credit precedent**: "
        "the corpus is severely `correct`-skewed on the later rubric items "
        "(~75 partial examples in 3,600 submissions), so label recall on the "
        "conflicting/partial cases is the metric most starved of evidence. The "
        "highest-leverage move is label-stratified indexing or retrieval "
        "(guarantee partial-labelled precedent is reachable when the query looks "
        "borderline), rather than a better embedding model -- the ceiling here is "
        "set by how much minority-label evidence exists to retrieve, not by "
        "similarity quality.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def write_traces(trace_dir: Path, name: str, traces: dict[str, dict]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = trace_dir / f"{name}.jsonl"
    out.write_text("\n".join(json.dumps(traces[i]) for i in traces), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(retriever: HybridRetriever, cases: list[dict], k: int = 5,
        report_path: Path = DEFAULT_REPORT,
        trace_dir: Path = DEFAULT_TRACE_DIR) -> dict:
    """Full pipeline: calibrate on hybrid traces, evaluate before/after, write
    traces + report. Returns a small summary dict."""
    # collect hybrid traces once to calibrate (confidence is signal-independent)
    hybrid_traces = {t["id"]: t for t in run_traces(retriever, cases, ("dense", "sparse"), k)}
    sweep = threshold_sweep(cases, hybrid_traces, SWEEP_GRID)
    chosen = best_threshold(sweep)

    before = evaluate_config("dense-only, no refusal", retriever, cases,
                             ("dense",), threshold=-1e9, k=k)
    after = evaluate_config("hybrid + calibrated refusal", retriever, cases,
                            ("dense", "sparse"), threshold=chosen, k=k)

    write_traces(trace_dir, "before", before.traces)
    write_traces(trace_dir, "after", after.traces)
    write_report(report_path, cases, before, after, sweep, k)
    return {
        "chosen_threshold": chosen,
        "before_refusal_acc": before.refusal["accuracy"],
        "after_refusal_acc": after.refusal["accuracy"],
        "leaks": after.leakage["leaked_chunks"],
        "report": str(report_path),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate the evidence-layer retriever")
    ap.add_argument("--chunks", default="chunks_real.jsonl")
    ap.add_argument("--index", default=None, help="embedding index path (defaults to package default)")
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--traces", default=str(DEFAULT_TRACE_DIR))
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    from .embeddings import DEFAULT_INDEX
    retr = HybridRetriever(args.chunks, index_path=args.index or DEFAULT_INDEX)
    cases = load_eval_set(args.eval_set)
    summary = run(retr, cases, k=args.k,
                  report_path=Path(args.report), trace_dir=Path(args.traces))

    print(f"Eval set: {len(cases)} cases")
    print(f"Calibrated refusal threshold: {summary['chosen_threshold']}")
    print(f"Refusal accuracy  before={summary['before_refusal_acc']:.2f}  "
          f"after={summary['after_refusal_acc']:.2f}")
    print(f"Scope/permission leaks: {summary['leaks']} (must be 0)")
    print(f"Report: {summary['report']}")
    print(f"Traces: {args.traces}/before.jsonl, {args.traces}/after.jsonl")
