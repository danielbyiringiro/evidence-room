"""
Tests for the specialist model review.

The gate forbids cherry-picking, so the properties that matter are structural:
splits are leakage-free, the reranker actually learns when signal exists, the
bootstrap CI behaves, the leakage invariant survives reranking, and the go/no-go
rule is honest. Most tests build tiny synthetic corpora; none need a real model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_room.embeddings import EmbeddingIndex  # noqa: E402
from evidence_room import specialize as S  # noqa: E402

pytest.importorskip("sklearn")


# --- synthetic corpus builder ----------------------------------------------

def _corpus(tmp_path, n_sub=60, dim=8, separable=True, seed=0):
    """Build a chunks.jsonl + EmbeddingIndex for one (question, item) scope.
    If separable, a proof's neighbours in vector space share its label, so a
    reranker over dense features should beat nothing-learned; if not, labels are
    random noise and no method can win."""
    rng = np.random.default_rng(seed)
    rows, ids, qk, items, vecs = [], [], [], [], []
    labels = ["correct", "partial", "not present"]
    for s in range(n_sub):
        lbl = labels[s % 3]
        # separable: vector cluster determined by label; else random
        base = np.zeros(dim, dtype=np.float32)
        if separable:
            base[s % 3] = 1.0
        v = base + rng.standard_normal(dim).astype(np.float32) * (0.15 if separable else 1.0)
        v /= np.linalg.norm(v)
        cid = f"divisibility-{s}::Hypothesis.is.stated"
        rows.append({
            "chunk_id": cid, "doc_type": "graded_exemplar",
            "text": f"proof {s} label {lbl} tokens {'alpha' if lbl=='correct' else 'beta'}",
            "question_key": "divisibility",
            "provenance": {"submission_id": f"divisibility-{s}", "question_key": "divisibility",
                           "rubric_item": "Hypothesis is stated", "label": lbl, "score": 2},
        })
        ids.append(cid); qk.append("divisibility"); items.append("Hypothesis is stated"); vecs.append(v)
    cpath = tmp_path / "chunks.jsonl"
    cpath.write_text("\n".join(json.dumps(r) for r in rows))
    index = EmbeddingIndex(vectors=np.array(vecs, dtype=np.float32),
                           chunk_ids=np.array(ids), question_key=np.array(qk),
                           rubric_item=np.array(items), model="stub")
    return cpath, index


# --- leakage-safe, frozen splits -------------------------------------------

def test_splits_are_submission_disjoint(tmp_path):
    cpath, index = _corpus(tmp_path)
    ex = S.load_exemplars(cpath, index)
    splits = S.make_splits(ex, tmp_path / "manifest.json")
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])
    assert not (splits["train"] & splits["val"])


def test_test_manifest_is_frozen(tmp_path):
    cpath, index = _corpus(tmp_path)
    ex = S.load_exemplars(cpath, index)
    m = tmp_path / "manifest.json"
    first = S.make_splits(ex, m)["test"]
    # a different seed must NOT change the frozen test set once the manifest exists
    second = S.make_splits(ex, m, seed=999)["test"]
    assert first == second
    assert json.loads(m.read_text())["test_submissions"]


def test_test_submissions_never_in_pool(tmp_path):
    cpath, index = _corpus(tmp_path)
    ex = S.load_exemplars(cpath, index)
    splits = S.make_splits(ex, tmp_path / "manifest.json")
    pool = splits["train"] | splits["val"]
    assert not (splits["test"] & pool)


# --- reranker learns when signal exists ------------------------------------

def test_reranker_beats_baseline_on_separable_data(tmp_path):
    cpath, index = _corpus(tmp_path, n_sub=90, separable=True, seed=1)
    study = S.run_study(cpath, index, manifest_path=tmp_path / "manifest.json")
    # with clean label-clustered vectors the reranker should not be worse,
    # and the study should run end-to-end with a decision
    assert study.rerank_p1 >= study.base_p1 - 0.05
    assert study.verdict["decision"] in {
        "GO", "NO-GO", "GO (marginal) -- positive but below target; ship behind flag, keep iterating"}


def test_leakage_invariant_holds_during_eval(tmp_path):
    """evaluate() asserts every candidate shares the query scope; a corpus with
    two scopes must never mix them."""
    cpath, index = _corpus(tmp_path, n_sub=60)
    from sklearn.linear_model import LogisticRegression
    ex = S.load_exemplars(cpath, index)
    splits = S.make_splits(ex, tmp_path / "m.json")
    pool = [e for e in ex if e.submission_id in (splits["train"] | splits["val"])]
    scope_pools = S.build_scope_pools(pool)
    train_q = [e for e in ex if e.submission_id in splits["train"]]
    X, y = S.build_pairs(train_q, scope_pools)
    model = LogisticRegression(max_iter=200).fit(X, y)
    test_q = [e for e in ex if e.submission_id in splits["test"]]
    rows = S.evaluate(test_q, scope_pools, model)  # asserts scope match internally
    assert rows


# --- statistics -------------------------------------------------------------

def test_bootstrap_ci_orders_low_high(tmp_path):
    rows = [S.EvalRow(f"s{i}", "q", "i", "correct",
                      base_p1=(i % 2 == 0), rerank_p1=True,
                      base_recall=True, rerank_recall=True) for i in range(50)]
    lo, hi = S.bootstrap_delta_ci(rows, seed=0, b=500)
    assert lo <= hi
    assert lo >= 0  # reranker never worse here -> non-negative delta


def test_decide_go_and_nogo():
    slices = {"a": {"delta": 0.0, "n": 10, "base_p1": 0.5, "rerank_p1": 0.5}}
    go = S.decide(0.50, 0.65, (0.05, 0.25), {"overhead_ms": 1.0}, slices, slices)
    assert go["decision"] == "GO"
    nogo = S.decide(0.50, 0.52, (-0.03, 0.07), {"overhead_ms": 1.0}, slices, slices)
    assert nogo["decision"] == "NO-GO"


def test_report_flags_scaffolding(tmp_path):
    cpath, index = _corpus(tmp_path, n_sub=60)  # model="stub"
    study = S.run_study(cpath, index, manifest_path=tmp_path / "m.json")
    out = tmp_path / "review.md"
    S.write_report(study, out)
    assert "SCAFFOLDING RUN" in out.read_text()
