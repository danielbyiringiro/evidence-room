"""
Hybrid retrieval for the evidence layer.

A grading query is always scoped: "for question Q, rubric item I, here is the
student's proof P -- what guidance and prior judgments justify a deduction?"
Retrieval answers that in two parts, because the two document families play
different roles (see README):

  1. WHY  -- the rubric_guidance chunk for item I. There are only 7 of these and
     they are the same approved rubric for every question, so they are addressed
     by DIRECT LOOKUP, not searched. Vector search over a 7-document set is
     strictly noisier than fetching by key.

  2. PRECEDENT -- graded_exemplar chunks: real proofs previously judged on item I.
     These are retrieved by HYBRID search (dense + sparse), because "find a proof
     that looks like this one and was judged on this item" is exactly a
     similarity problem, and the two signals fail differently:
       - dense (embeddings) captures paraphrase / structural similarity,
       - sparse (BM25) anchors on the exact algebra tokens (n+1, 2^k, mod) that
         dense models routinely blur.
     They are combined with Reciprocal Rank Fusion.

ACCESS SCOPING IS ENFORCED PRE-RETRIEVAL, at the candidate-set level -- the same
invariant ingestion establishes. Before any scoring, the exemplar pool is
restricted to (question_key == Q AND rubric_item == I). An exemplar from another
question or another item is never a candidate, so it cannot surface even under a
confused or injected query. Filtering the set (rather than instructing the model
to ignore out-of-scope hits) is what makes the guarantee hold.

PERMISSION LEVELS extend that same candidate-set filter to WHO is asking. Two
roles (see PERMISSIONS): a faculty grader sees rubric guidance and graded
exemplars; a trainee sees only rubric guidance -- graded exemplars carry verbatim
student proof text and a named grader's judgment, so they are removed from the
candidate set before scoring. The system cannot retrieve what the role cannot
see, so it cannot cite it either.

REFUSAL closes the loop: retrieval returns a decision, not just hits. It declines
to suggest a deduction when the rubric item is unknown, when nothing is in scope,
when the role may not see precedent, or when no visible exemplar actually
resembles the query proof (cosine below RefusalPolicy.min_similarity). A refusal
is preferable to citing a weak match as if it justified a deduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder, EmbeddingIndex, DEFAULT_INDEX

_TOKEN = re.compile(r"[A-Za-z0-9]+")

# --- permission model ------------------------------------------------------
# Two permission levels, enforced PRE-retrieval at the candidate-set level -- the
# same principle as question scoping. A document family the role cannot see is
# never a retrieval candidate, so it cannot surface or be cited even under a
# confused or injected query ("a system must not retrieve what the user cannot
# see"). The sensitive family is graded_exemplar: real, consented student proof
# text plus a named grader's judgment. Rubric guidance is the approved rubric and
# carries no student text, so it is visible to both levels.
FACULTY = "faculty"    # Faculty Intern / grader: full access
TRAINEE = "trainee"    # onboarding TA / auditor: rubric guidance only

PERMISSIONS: dict[str, set[str]] = {
    FACULTY: {"rubric_guidance", "graded_exemplar"},
    TRAINEE: {"rubric_guidance"},
}
DEFAULT_ROLE = FACULTY


# --- refusal policy --------------------------------------------------------
# Decisions, so a caller (and the eval harness) can score refusal quality.
ANSWER = "answer"
REFUSE = "refuse"

# Refusal reasons -- distinct causes, because "why did it refuse" is exactly what
# the offline eval's refusal-quality metric needs to separate good refusals
# (genuinely unanswerable / out of scope) from over-refusal.
R_UNKNOWN_ITEM = "unknown_rubric_item"
R_NO_CANDIDATES = "no_candidates_in_scope"
R_NOT_PERMITTED = "not_permitted_for_role"
R_LOW_SIMILARITY = "insufficient_similarity"


@dataclass(frozen=True)
class RefusalPolicy:
    """
    When to decline rather than suggest a deduction.

    ``min_similarity`` is a cosine floor on the best-matching precedent: if no
    visible exemplar actually resembles the query proof, any suggested deduction
    would be unsupported, so the system refuses instead of citing a weak match.

    The default (0.65) is the value CALIBRATED by the offline eval for the shipped
    model (bge-small-en-v1.5): balanced accuracy of the similarity-refusal
    decision peaks at 0.65-0.70 on eval_set.jsonl, with over-refusal setting in
    past 0.75. It is model- and corpus-specific -- if you change the embedding
    model, re-run `python -m evidence_room.evaluate` and reset it. Override with
    EVIDENCE_ROOM_MIN_SIM or by passing a RefusalPolicy.
    """
    min_candidates: int = 1
    min_similarity: float = float(os.environ.get("EVIDENCE_ROOM_MIN_SIM", "0.65"))


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Keeps algebra tokens like 'n', '2', 'k'."""
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> dict[str, float]:
    """
    Combine ranked id lists into one score per id: sum of 1/(k + rank).

    RRF fuses rankings without needing the two scorers to share a scale -- BM25
    scores and cosine similarities are not comparable, but their *ranks* are.
    k=60 is the value from the original Cormack et al. paper; larger k flattens
    the contribution of top ranks.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    score: float
    doc_type: str
    provenance: dict[str, Any]


@dataclass
class RetrievalResult:
    """
    The two halves of the answer, plus the decision to stand behind them.

    ``decision`` is ANSWER or REFUSE. On REFUSE, ``refusal_reason`` says why and a
    caller should NOT present a suggested deduction -- but guidance/exemplars are
    still populated where available, so the refusal stays inspectable in a trace
    rather than being an opaque dead end. ``confidence`` is the best precedent's
    cosine similarity; ``n_candidates`` is how many exemplars were visible after
    scoping and permission filtering.
    """
    question_key: str
    rubric_item: str
    role: str
    guidance: Retrieved | None          # the rubric chunk ("why"); None if unknown/not permitted
    exemplars: list[Retrieved]          # precedent, best first
    decision: str = ANSWER
    refusal_reason: str | None = None
    confidence: float = 0.0
    n_candidates: int = 0

    @property
    def refused(self) -> bool:
        return self.decision == REFUSE

    def as_dict(self) -> dict:
        return {
            "question_key": self.question_key,
            "rubric_item": self.rubric_item,
            "role": self.role,
            "decision": self.decision,
            "refusal_reason": self.refusal_reason,
            "confidence": self.confidence,
            "n_candidates": self.n_candidates,
            "guidance": None if self.guidance is None else vars(self.guidance),
            "exemplars": [vars(e) for e in self.exemplars],
        }


class HybridRetriever:
    """
    Loads the chunk store + dense index and answers scoped grading queries.

    The dense EmbeddingIndex covers only exemplars; the chunk store is read in
    full so rubric guidance (not embedded) is still available for direct lookup.
    BM25 is built lazily per (question_key, rubric_item) candidate pool, since a
    single global BM25 over 25k chunks would mix items and questions that access
    scoping forbids from co-occurring anyway.
    """

    def __init__(self, chunks_path: Path | str,
                 index: EmbeddingIndex | None = None,
                 index_path: Path | str = DEFAULT_INDEX,
                 policy: RefusalPolicy | None = None):
        self.policy = policy or RefusalPolicy()
        self.chunks: dict[str, dict] = {}
        self.rubric_by_item: dict[str, dict] = {}
        with Path(chunks_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                self.chunks[c["chunk_id"]] = c
                if c["doc_type"] == "rubric_guidance":
                    self.rubric_by_item[c["provenance"]["code_name"]] = c

        self.index = index if index is not None else EmbeddingIndex.load(index_path)
        self._embedder = Embedder(self.index.model)
        # Cache BM25 corpora per candidate pool so repeated queries are cheap.
        self._bm25_cache: dict[tuple[str, str], tuple[BM25Okapi, np.ndarray]] = {}

    # -- candidate scoping (pre-retrieval access filter) --------------------

    def _candidate_mask(self, question_key: str, rubric_item: str) -> np.ndarray:
        return (self.index.question_key == question_key) & (
            self.index.rubric_item == rubric_item
        )

    # -- the two retrieval signals ------------------------------------------

    def _dense_ranking(self, qvec: np.ndarray, cand_idx: np.ndarray,
                       depth: int) -> tuple[list[str], float]:
        """Return (ranked chunk ids, best cosine similarity). The top similarity
        is the confidence signal the refusal policy thresholds on."""
        sims = self.index.vectors[cand_idx] @ qvec        # cosine (all normalized)
        order = np.argsort(-sims)[:depth]
        ranking = [self.index.chunk_ids[cand_idx[i]] for i in order]
        top_sim = float(sims.max()) if len(sims) else 0.0
        return ranking, top_sim

    def _bm25_ranking(self, query: str, question_key: str, rubric_item: str,
                      cand_idx: np.ndarray, depth: int) -> list[str]:
        key = (question_key, rubric_item)
        if key not in self._bm25_cache:
            ids = self.index.chunk_ids[cand_idx]
            corpus = [tokenize(self.chunks[cid]["text"]) for cid in ids]
            self._bm25_cache[key] = (BM25Okapi(corpus), ids)
        bm25, ids = self._bm25_cache[key]
        scores = bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:depth]
        return [ids[i] for i in order]

    # -- public API ----------------------------------------------------------

    def retrieve(self, question_key: str, rubric_item: str, proof_text: str,
                 k: int = 5, fusion_depth: int = 50,
                 role: str = DEFAULT_ROLE,
                 policy: RefusalPolicy | None = None,
                 signals: tuple[str, ...] = ("dense", "sparse")) -> RetrievalResult:
        """
        Retrieve guidance + top-k precedent exemplars for a scoped query, gated by
        the caller's permission level and a refusal policy.

        ``role`` selects which document families are visible (see PERMISSIONS);
        ``policy`` decides when to refuse rather than suggest a deduction.
        ``signals`` selects which rankings feed the fusion -- ("dense",) is the
        embeddings-only baseline, ("dense", "sparse") is the full hybrid; used by
        the offline eval to measure hybrid's lift over dense-only. The confidence
        signal is always the top dense cosine, independent of ``signals``.
        fusion_depth caps how deep each signal contributes to the fusion; k is how
        many fused results to return.
        """
        policy = policy or self.policy
        if role not in PERMISSIONS:
            raise ValueError(f"unknown role {role!r}; known: {sorted(PERMISSIONS)}")
        visible = PERMISSIONS[role]

        def result(guidance, exemplars, decision, reason, confidence, n):
            return RetrievalResult(
                question_key=question_key, rubric_item=rubric_item, role=role,
                guidance=guidance, exemplars=exemplars, decision=decision,
                refusal_reason=reason, confidence=confidence, n_candidates=n,
            )

        # Guidance is permission-gated too, though it carries no student text.
        guidance = None
        guidance_chunk = self.rubric_by_item.get(rubric_item)
        if guidance_chunk is not None and "rubric_guidance" in visible:
            guidance = Retrieved(
                chunk_id=guidance_chunk["chunk_id"],
                text=guidance_chunk["text"],
                score=1.0,                       # direct lookup, not scored
                doc_type="rubric_guidance",
                provenance=guidance_chunk["provenance"],
            )

        # Refuse: the query names a rubric item that does not exist.
        if guidance_chunk is None:
            return result(guidance, [], REFUSE, R_UNKNOWN_ITEM, 0.0, 0)

        # Refuse: this role cannot see precedent, so there is nothing to ground a
        # suggested deduction on. Enforced before any candidate is scored.
        if "graded_exemplar" not in visible:
            return result(guidance, [], REFUSE, R_NOT_PERMITTED, 0.0, 0)

        cand_idx = np.nonzero(self._candidate_mask(question_key, rubric_item))[0]
        n = int(len(cand_idx))
        # Refuse: nothing in scope for this (question, item).
        if n < policy.min_candidates:
            return result(guidance, [], REFUSE, R_NO_CANDIDATES, 0.0, n)

        qvec = self._embedder.encode_query(proof_text)
        # Dense is always computed: its top cosine is the confidence signal the
        # refusal policy thresholds on, regardless of what feeds the fusion.
        dense, confidence = self._dense_ranking(qvec, cand_idx, fusion_depth)
        rankings = []
        if "dense" in signals:
            rankings.append(dense)
        if "sparse" in signals:
            rankings.append(self._bm25_ranking(proof_text, question_key,
                                               rubric_item, cand_idx, fusion_depth))
        if not rankings:
            raise ValueError(f"signals must include 'dense' and/or 'sparse'; got {signals}")

        fused = reciprocal_rank_fusion(rankings)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        exemplars = [
            Retrieved(
                chunk_id=cid,
                text=self.chunks[cid]["text"],
                score=score,
                doc_type="graded_exemplar",
                provenance=self.chunks[cid]["provenance"],
            )
            for cid, score in top
        ]

        # Refuse: candidates exist but none actually resembles the query proof, so
        # a suggested deduction would be unsupported. Exemplars are still returned
        # (weak evidence) so the refusal is inspectable in the trace.
        if confidence < policy.min_similarity:
            return result(guidance, exemplars, REFUSE, R_LOW_SIMILARITY, confidence, n)

        return result(guidance, exemplars, ANSWER, None, confidence, n)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Query the hybrid retriever")
    ap.add_argument("--chunks", default="chunks_real.jsonl")
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--question", required=True, help="question_key, e.g. divisibility")
    ap.add_argument("--item", required=True, help="rubric item, e.g. 'Hypothesis is stated'")
    ap.add_argument("--proof", required=True, help="student proof text to find precedent for")
    ap.add_argument("-k", type=int, default=5, help="number of exemplars to return")
    ap.add_argument("--role", default=DEFAULT_ROLE, choices=sorted(PERMISSIONS),
                    help="permission level (faculty=full, trainee=guidance only)")
    ap.add_argument("--min-sim", type=float, default=None,
                    help="override cosine refusal threshold for this query")
    args = ap.parse_args()

    policy = None if args.min_sim is None else RefusalPolicy(min_similarity=args.min_sim)
    r = HybridRetriever(args.chunks, index_path=args.index)
    result = r.retrieve(args.question, args.item, args.proof, k=args.k,
                        role=args.role, policy=policy)

    print(f"\nQuery: [{result.question_key}] {result.rubric_item}  (role: {result.role})")
    print(f"Decision: {result.decision.upper()}"
          + (f"  reason={result.refusal_reason}" if result.refused else "")
          + f"  | confidence={result.confidence:.3f}  candidates={result.n_candidates}\n")

    if result.refused:
        print("REFUSED -- evidence is insufficient to suggest a deduction.")
        if result.refusal_reason == R_NOT_PERMITTED:
            print("  This permission level cannot see graded exemplars.")
        elif result.refusal_reason == R_LOW_SIMILARITY:
            print("  No visible precedent resembles this proof closely enough.")
        elif result.refusal_reason == R_NO_CANDIDATES:
            print("  No graded exemplars in scope for this question/item.")
        elif result.refusal_reason == R_UNKNOWN_ITEM:
            print("  That rubric item is not part of the approved rubric.")

    if result.guidance:
        print("\nWHY (rubric guidance):")
        print(f"  {result.guidance.text[:300]}...")

    if result.exemplars:
        label = "PRECEDENT" if not result.refused else "PRECEDENT (weak; not cited)"
        print(f"\n{label} (top {len(result.exemplars)} exemplars):")
        for i, e in enumerate(result.exemplars, 1):
            p = e.provenance
            print(f"  {i}. {e.chunk_id}  score={e.score:.4f}")
            print(f"     label={p['label']}  graded_by={p.get('graded_by','')}")
            print(f"     {e.text[:180]}...")
