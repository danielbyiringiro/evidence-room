"""
Tests for the evaluation harness.

The metric functions are pure (cases + traces in, numbers out), so most of this
runs with hand-built traces and needs no model or corpus. One integration test
drives the whole harness through a synthetic retriever + stub embedder, proving
run() wires together and writes a report + traces without any download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_room import evaluate as E  # noqa: E402
from evidence_room.embeddings import EmbeddingIndex  # noqa: E402
from evidence_room.retrieval import HybridRetriever, ANSWER, REFUSE  # noqa: E402


# --- pure metric tests -----------------------------------------------------

def _trace(cid, category, role="faculty", q="divisibility", item="Hypothesis is stated",
           hard=None, conf=0.8, exemplars=None, leaked=None):
    return {
        "id": cid, "category": category, "role": role, "question_key": q,
        "rubric_item": item, "signals": ["dense", "sparse"], "hard_refuse": hard,
        "confidence": conf, "n_candidates": len(exemplars or []),
        "guidance_id": "RUBRIC::x", "exemplars": exemplars or [], "leaked": leaked or [],
    }


def test_decision_at_hard_refuse_ignores_threshold():
    t = _trace("x", "unanswerable", hard="unknown_rubric_item", conf=0.99)
    assert E.decision_at(t, 0.0) == (REFUSE, "unknown_rubric_item")


def test_decision_at_similarity_threshold():
    t = _trace("x", "answerable", conf=0.5)
    assert E.decision_at(t, 0.4)[0] == ANSWER
    assert E.decision_at(t, 0.6)[0] == REFUSE
    assert E.decision_at(t, 0.6)[1] == "insufficient_similarity"


def test_refusal_metrics_counts_over_and_missed():
    cases = [
        {"id": "a", "category": "answerable", "expect": {"decision": "answer"}},
        {"id": "b", "category": "unanswerable", "expect": {"decision": "refuse"}},
    ]
    traces = {
        "a": _trace("a", "answerable", conf=0.1),   # will refuse at thr 0.5 -> over-refusal
        "b": _trace("b", "unanswerable", conf=0.9),  # will answer at thr 0.5 -> missed refusal
    }
    m = E.refusal_metrics(cases, traces, threshold=0.5)
    assert m["over_refused"] == ["a"]
    assert m["missed_refusals"] == ["b"]
    assert m["accuracy"] == 0.0


def test_threshold_sweep_and_best_threshold():
    cases = [
        {"id": "p", "category": "answerable", "expect": {"decision": "answer"}},
        {"id": "n", "category": "unanswerable",
         "expect": {"decision": "refuse", "reason": "insufficient_similarity"}},
    ]
    traces = {
        "p": _trace("p", "answerable", conf=0.6),    # answer while thr <= 0.6
        "n": _trace("n", "unanswerable", conf=0.2),  # refuse once thr > 0.2
    }
    sweep = E.threshold_sweep(cases, traces, [0.1, 0.3, 0.7])
    row = {r["threshold"]: r for r in sweep}
    assert row[0.3]["balanced_accuracy"] == 1.0   # answers p, refuses n
    assert E.best_threshold(sweep) == 0.3


def test_leakage_report_flags_out_of_scope():
    cases = [{"id": "c", "category": "adversarial", "question_key": "divisibility",
              "rubric_item": "Hypothesis is stated"}]
    leaked = [{"chunk_id": "recurrence-1::x", "label": "correct",
               "question_key": "recurrence", "rubric_item": "Hypothesis is stated",
               "score": 0.1}]
    traces = {"c": _trace("c", "adversarial", leaked=leaked, exemplars=leaked)}
    rep = E.leakage_report(cases, traces)
    assert rep["leaked_chunks"] == 1 and rep["adversarial_leaks"] == 1


def test_label_metrics_precision_recall():
    cases = [{"id": "c", "category": "answerable", "question_key": "q",
              "rubric_item": "i", "expect": {"label": "correct"}}]
    ex = [{"chunk_id": f"q-{i}::i", "label": lbl, "question_key": "q",
           "rubric_item": "i", "score": 0.1}
          for i, lbl in enumerate(["correct", "correct", "partial", "not present"])]
    traces = {"c": _trace("c", "answerable", exemplars=ex)}
    m = E.label_metrics(cases, traces, k=4)
    assert m["precision_at_k"] == 0.5    # 2 of 4 correct
    assert m["recall_at_k"] == 1.0       # at least one correct present


def test_conflict_metrics_surfaced():
    cases = [{"id": "c", "category": "conflicting", "question_key": "q",
              "rubric_item": "i", "expect": {"labels": ["partial", "correct"]}}]
    ex = [{"chunk_id": "q-1::i", "label": "partial", "question_key": "q",
           "rubric_item": "i", "score": 0.1},
          {"chunk_id": "q-2::i", "label": "correct", "question_key": "q",
           "rubric_item": "i", "score": 0.1}]
    traces = {"c": _trace("c", "conflicting", exemplars=ex)}
    assert E.conflict_metrics(cases, traces, k=5)["surfaced_rate"] == 1.0


# --- integration: run() over a synthetic retriever -------------------------

class _StubEmbedder:
    def __init__(self, dim): self.model_id = "stub"; self._v = np.ones(dim, np.float32) / np.sqrt(dim)
    def encode_query(self, text): return self._v


def _mini_retriever(tmp_path):
    import json
    rows = [{
        "chunk_id": "RUBRIC::Hypothesis_is_stated", "doc_type": "rubric_guidance",
        "text": "Rubric item: Hypothesis is stated. General rules that also apply: x.",
        "question_key": None,
        "provenance": {"code_name": "Hypothesis is stated", "proof_section": "IH",
                       "source": "Grading Instructions.csv"},
    }]
    for s in range(4):
        rows.append({
            "chunk_id": f"divisibility-{s}::Hypothesis.is.stated",
            "doc_type": "graded_exemplar",
            "text": f"Rubric item: Hypothesis is stated. Expert judgment: correct. proof {s}",
            "question_key": "divisibility",
            "provenance": {"submission_id": f"divisibility-{s}", "question_key": "divisibility",
                           "rubric_item": "Hypothesis is stated", "score": 2, "label": "correct"},
        })
    cpath = tmp_path / "chunks.jsonl"
    cpath.write_text("\n".join(json.dumps(r) for r in rows))
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((4, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = EmbeddingIndex(
        vectors=vecs,
        chunk_ids=np.array([f"divisibility-{s}::Hypothesis.is.stated" for s in range(4)]),
        question_key=np.array(["divisibility"] * 4),
        rubric_item=np.array(["Hypothesis is stated"] * 4), model="stub")
    r = HybridRetriever(cpath, index=index)
    r._embedder = _StubEmbedder(8)
    return r


def test_run_writes_report_and_traces_with_zero_leaks(tmp_path):
    r = _mini_retriever(tmp_path)
    cases = [
        {"id": "a", "category": "answerable", "role": "faculty",
         "question_key": "divisibility", "rubric_item": "Hypothesis is stated",
         "proof": "assume it holds for k", "expect": {"decision": "answer", "label": "correct"}},
        {"id": "u", "category": "unanswerable", "role": "faculty",
         "question_key": "divisibility", "rubric_item": "No Such Item",
         "proof": "x", "expect": {"decision": "refuse", "reason": "unknown_rubric_item"}},
        {"id": "t", "category": "unanswerable", "role": "trainee",
         "question_key": "divisibility", "rubric_item": "Hypothesis is stated",
         "proof": "x", "expect": {"decision": "refuse", "reason": "not_permitted_for_role"}},
    ]
    report = tmp_path / "report.md"
    traces = tmp_path / "traces"
    summary = E.run(r, cases, k=5, report_path=report, trace_dir=traces)

    assert report.exists() and (traces / "before.jsonl").exists() and (traces / "after.jsonl").exists()
    assert summary["leaks"] == 0
    assert "chosen_threshold" in summary
    assert "# Evidence Room" in report.read_text()
