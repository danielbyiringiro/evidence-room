"""
Guards on the retrieval-layer invariants.

The scoping invariant established at ingestion must survive retrieval: a fused,
re-ranked result set is still not allowed to surface an exemplar from another
question or another rubric item. And the two-part shape (rubric guidance by
lookup, exemplars by hybrid search) must hold.

These tests inject a synthetic embedding index and a stub embedder, so the
retrieval logic is exercised WITHOUT downloading a model or needing the corpus.
A separate model-gated test covers the real fastembed path.

Run: pytest -q  (from repo root)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_room.embeddings import EmbeddingIndex
from evidence_room.retrieval import (
    HybridRetriever, reciprocal_rank_fusion, tokenize, RefusalPolicy,
    FACULTY, TRAINEE, REFUSE, ANSWER,
    R_UNKNOWN_ITEM, R_NO_CANDIDATES, R_NOT_PERMITTED, R_LOW_SIMILARITY,
)

# Answer regardless of similarity (stub vectors are random); use where the test
# is about scoping/permissions, not the refusal threshold.
_ALWAYS_ANSWER = RefusalPolicy(min_similarity=-1.0)


# --- pure-function tests (no fixtures) -------------------------------------

def test_tokenize_keeps_algebra_tokens():
    toks = tokenize("Assume 2^k > k+1 for n = k")
    assert "2" in toks and "k" in toks and "1" in toks and "n" in toks


def test_rrf_rewards_agreement_across_signals():
    # 'b' is ranked high by both signals; 'a' and 'c' only by one each.
    dense = ["a", "b", "c"]
    sparse = ["b", "c", "a"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    assert scores["b"] == max(scores.values()), "item ranked high by both should win"


def test_rrf_uses_rank_not_score_scale():
    # Even with wildly different implied magnitudes, only rank position matters.
    r1 = ["x", "y"]
    r2 = ["x", "y"]
    s = reciprocal_rank_fusion([r1, r2])
    assert s["x"] > s["y"]


# --- synthetic fixtures for the retriever ----------------------------------

class _StubEmbedder:
    """Returns a fixed unit vector; dense scores become uniform, so RRF/scoping
    behavior is what's under test, not embedding quality."""

    def __init__(self, dim: int):
        self.model_id = "stub"
        self._v = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    def encode_query(self, text: str) -> np.ndarray:
        return self._v


def _write_chunks(path: Path) -> None:
    import json

    rows = []
    # rubric guidance for two items (shared, question_key=None)
    for item in ("Hypothesis is stated", "Identify Base Case"):
        rows.append({
            "chunk_id": f"RUBRIC::{item.replace(' ', '_')}",
            "doc_type": "rubric_guidance",
            "text": f"Rubric item: {item}. Grading guidance blah. "
                    f"General rules that also apply: be consistent.",
            "question_key": None,
            "provenance": {"code_name": item, "proof_section": "x",
                           "source": "Grading Instructions.csv"},
        })
    # exemplars across two questions x two items x two submissions
    for q in ("divisibility", "recurrence"):
        for item, col in (("Hypothesis is stated", "Hypothesis.is.stated"),
                          ("Identify Base Case", "Identify.Base.Case")):
            for s in range(3):
                rows.append({
                    "chunk_id": f"{q}-{s}::{col}",
                    "doc_type": "graded_exemplar",
                    "text": f"Rubric item: {item}. Expert judgment: correct. "
                            f"Student proof for {q} number {s}: assume 2^k works.",
                    "question_key": q,
                    "provenance": {
                        "submission_id": f"{q}-{s}", "question_key": q,
                        "rubric_item": item, "rubric_column": col,
                        "proof_section": "x", "score": 2, "label": "correct",
                        "graded_by": "grader-A",
                    },
                })
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _synthetic_index(chunks_path: Path, dim: int = 8) -> EmbeddingIndex:
    import json

    ids, qk, items, vecs = [], [], [], []
    rng = np.random.default_rng(0)
    with chunks_path.open() as fh:
        for line in fh:
            c = json.loads(line)
            if c["doc_type"] != "graded_exemplar":
                continue
            ids.append(c["chunk_id"])
            qk.append(c["question_key"])
            items.append(c["provenance"]["rubric_item"])
            v = rng.standard_normal(dim).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
    return EmbeddingIndex(
        vectors=np.array(vecs, dtype=np.float32),
        chunk_ids=np.array(ids), question_key=np.array(qk),
        rubric_item=np.array(items), model="stub",
    )


@pytest.fixture
def retriever(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks)
    index = _synthetic_index(chunks)
    r = HybridRetriever(chunks, index=index)
    r._embedder = _StubEmbedder(index.vectors.shape[1])  # no model download
    return r


# --- invariant tests -------------------------------------------------------

def test_no_cross_question_or_item_leakage(retriever):
    """Every retrieved exemplar must match BOTH the queried question and item."""
    res = retriever.retrieve("divisibility", "Hypothesis is stated",
                             "assume 2^k works", k=10)
    assert res.exemplars, "expected some candidates"
    for e in res.exemplars:
        assert e.provenance["question_key"] == "divisibility"
        assert e.provenance["rubric_item"] == "Hypothesis is stated"


def test_guidance_comes_from_direct_lookup(retriever):
    res = retriever.retrieve("recurrence", "Identify Base Case", "base case n=1", k=3)
    assert res.guidance is not None
    assert res.guidance.doc_type == "rubric_guidance"
    assert res.guidance.provenance["code_name"] == "Identify Base Case"


def test_unknown_item_yields_no_guidance_and_no_exemplars(retriever):
    res = retriever.retrieve("divisibility", "No Such Item", "text", k=5)
    assert res.guidance is None
    assert res.exemplars == []


def test_exemplars_are_citable(retriever):
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof", k=5)
    required = {"submission_id", "question_key", "rubric_item", "score", "label"}
    for e in res.exemplars:
        assert required <= set(e.provenance)


def test_k_caps_result_count(retriever):
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof", k=2,
                             policy=_ALWAYS_ANSWER)
    assert len(res.exemplars) <= 2


# --- permission levels -----------------------------------------------------

def test_trainee_cannot_see_exemplars(retriever):
    """A trainee sees rubric guidance but never student proof text -- and because
    there is then nothing to ground a suggestion on, the query is refused."""
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof",
                             role=TRAINEE)
    assert res.role == TRAINEE
    assert res.guidance is not None            # guidance carries no student text
    assert res.exemplars == []                 # exemplars withheld pre-retrieval
    assert res.decision == REFUSE
    assert res.refusal_reason == R_NOT_PERMITTED


def test_faculty_sees_exemplars(retriever):
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof",
                             role=FACULTY, policy=_ALWAYS_ANSWER)
    assert res.role == FACULTY
    assert res.exemplars, "faculty should see precedent"
    assert res.decision == ANSWER


def test_unknown_role_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("divisibility", "Hypothesis is stated", "p", role="admin")


def test_permission_gate_holds_even_if_scope_matches(retriever):
    """Trainee refusal is about the family, not about whether candidates exist:
    the same query answers for faculty."""
    trainee = retriever.retrieve("divisibility", "Identify Base Case", "proof",
                                 role=TRAINEE)
    faculty = retriever.retrieve("divisibility", "Identify Base Case", "proof",
                                 role=FACULTY, policy=_ALWAYS_ANSWER)
    assert trainee.exemplars == [] and faculty.exemplars


# --- refusal when evidence is insufficient ---------------------------------

def test_refuse_unknown_rubric_item(retriever):
    res = retriever.retrieve("divisibility", "No Such Item", "proof")
    assert res.decision == REFUSE
    assert res.refusal_reason == R_UNKNOWN_ITEM
    assert res.guidance is None and res.exemplars == []


def test_refuse_when_no_candidates_in_scope(retriever):
    """Known item, but a question with no exemplars in the index."""
    res = retriever.retrieve("no-such-question", "Hypothesis is stated", "proof")
    assert res.decision == REFUSE
    assert res.refusal_reason == R_NO_CANDIDATES
    assert res.n_candidates == 0
    assert res.guidance is not None            # the item is real; guidance stands


def test_refuse_on_low_similarity_but_keep_weak_evidence(retriever):
    """An unmeetable threshold forces the similarity refusal path; the weak
    exemplars are still returned so the refusal is inspectable in a trace."""
    strict = RefusalPolicy(min_similarity=1.1)   # cosine can never reach 1.1
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof",
                             role=FACULTY, policy=strict)
    assert res.decision == REFUSE
    assert res.refusal_reason == R_LOW_SIMILARITY
    assert res.confidence < 1.1
    assert res.exemplars, "weak evidence should still be attached for the trace"


def test_answer_when_similarity_clears_threshold(retriever):
    res = retriever.retrieve("divisibility", "Hypothesis is stated", "proof",
                             role=FACULTY, policy=_ALWAYS_ANSWER)
    assert res.decision == ANSWER
    assert res.refusal_reason is None
    assert res.confidence >= -1.0


# --- model-gated integration test ------------------------------------------

_HAS_FASTEMBED = False
try:
    import fastembed  # noqa: F401
    _HAS_FASTEMBED = True
except Exception:
    pass

needs_model = pytest.mark.skipif(not _HAS_FASTEMBED, reason="fastembed not installed")


@needs_model
def test_real_embedder_encodes_query(tmp_path):
    """Smoke test: the real model loads and produces a normalized query vector.

    Requires the fastembed model to be downloadable (network); skipped if
    fastembed itself is absent."""
    from evidence_room.embeddings import Embedder

    emb = Embedder()
    try:
        v = emb.encode_query("Assume the statement holds for n = k.")
    except Exception as e:  # model download blocked / offline
        pytest.skip(f"fastembed model unavailable: {type(e).__name__}")
    assert v.ndim == 1 and v.shape[0] > 0
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
