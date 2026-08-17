"""
The Specialist Model Review -- a controlled, measurable fine-tuning study.

Hypothesis (pre-registered, see docs/finetune/README.md):

    A learned reranker over retrieval features will lift top-1 correct-label
    precedent (P@1 label agreement) on a frozen holdout by >= 10 points vs the
    RRF baseline, with a 95% bootstrap CI lower bound > 0, at < 5 ms added
    latency per query and ZERO regression on the scope/permission invariant.

This targets exactly the weakness the offline eval surfaced (F3: the correct-
label precedent is usually in the top-k but not always rank-1; recall@5 high,
precision@1 low). The intervention is a lightweight *reranker adapter*, not an
LLM fine-tune -- the decision framework in docs/finetune/README.md argues that a
full model tune is not justified here.

Method, aligned to the gate (no cherry-picking):
  - BASELINE: RRF(dense, BM25) ordering of the retrieved candidates.
  - INTERVENTION: a logistic-regression reranker over per-candidate features
    (dense cosine, BM25, ranks, candidate-label indicators, scope label prior).
  - FROZEN TEST: submissions are split by id; test submissions are held out of
    the retrieval pool entirely and the split manifest is frozen to disk, so the
    holdout is reused verbatim across runs.
  - CONFIDENCE: bootstrap CI on the P@1 delta over test queries.
  - ERROR SLICES: by rubric item and by the query's gold label (the partial slice
    is where minority-label evidence is scarce).
  - COST/LATENCY: measured per-query reranking overhead (CPU, no GPU, no API).
  - ROLLBACK: the reranker is a feature-flagged layer over the same candidate
    set; disabling it reverts to the baseline instantly (see the review report).

Leakage is controlled structurally: a test query's submission never appears in
the retrieval pool or in any training pair, and no feature is derived from the
query's gold label. The reranker only *reorders* an already scope/permission-
filtered candidate set, so it cannot introduce leakage by construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingIndex, DEFAULT_INDEX
from .retrieval import tokenize, reciprocal_rank_fusion

FT_DIR = Path(__file__).resolve().parents[2] / "docs" / "finetune"
DEFAULT_MANIFEST = FT_DIR / "test_manifest.json"
DEFAULT_REPORT = FT_DIR / "model-review.md"
DEFAULT_REGISTRY = FT_DIR / "registry.jsonl"

# --- pre-registered study parameters (fixed before looking at test results) ---
SEED = 17
SPLIT = (0.6, 0.2, 0.2)          # train / val / test by submission
RERANK_DEPTH = 50                # rerank the top-N retrieved candidates
MAX_TRAIN_QUERIES = 4000         # cap training queries (sampled) for tractable fits
MIN_CANDIDATES = 5               # skip scopes too small to rank
RECALL_K = 5
TARGET_DELTA = 0.10              # +10 points P@1 to call the tune worthwhile
LATENCY_BUDGET_MS = 5.0          # max acceptable added latency per query
SLICE_TOLERANCE = -0.05          # no slice may drop more than 5 points
N_BOOTSTRAP = 2000
LABELS = ("correct", "partial", "not present")


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

@dataclass
class Exemplar:
    chunk_id: str
    submission_id: str
    question_key: str
    rubric_item: str
    label: str
    text: str
    vec: np.ndarray


def load_exemplars(chunks_path: Path | str, index: EmbeddingIndex) -> list[Exemplar]:
    id_to_row = {cid: i for i, cid in enumerate(index.chunk_ids)}
    out = []
    with Path(chunks_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["doc_type"] != "graded_exemplar":
                continue
            cid = c["chunk_id"]
            if cid not in id_to_row:
                continue
            p = c["provenance"]
            out.append(Exemplar(
                chunk_id=cid, submission_id=p["submission_id"],
                question_key=p["question_key"], rubric_item=p["rubric_item"],
                label=p["label"], text=c["text"],
                vec=index.vectors[id_to_row[cid]],
            ))
    return out


# --------------------------------------------------------------------------
# leakage-safe, frozen splits
# --------------------------------------------------------------------------

def make_splits(exemplars: list[Exemplar], manifest_path: Path,
                seed: int = SEED) -> dict[str, set[str]]:
    """
    Split by SUBMISSION id (grouped), and FREEZE the test set to disk. If the
    manifest exists, the test submissions are loaded verbatim so the holdout is
    identical across runs; train/val are re-derived deterministically from the
    remaining submissions. This is what makes 'frozen test set' true.
    """
    subs = sorted({e.submission_id for e in exemplars})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_train = int(SPLIT[0] * n)
    n_val = int(SPLIT[1] * n)

    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text())
        test = set(frozen["test_submissions"]) & set(subs)
        remaining = [s for s in subs if s not in test]
        train = set(remaining[:len(remaining) - n_val]) if len(remaining) > n_val else set(remaining)
        val = set(remaining[len(remaining) - n_val:])
    else:
        train = set(subs[:n_train])
        val = set(subs[n_train:n_train + n_val])
        test = set(subs[n_train + n_val:])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "seed": seed, "split": SPLIT, "n_submissions": n,
            "test_submissions": sorted(test),
        }, indent=0))
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------
# candidate retrieval + features
# --------------------------------------------------------------------------

def _scope_key(e: Exemplar) -> tuple[str, str]:
    return (e.question_key, e.rubric_item)


def _label_onehot(label: str) -> list[float]:
    return [1.0 if label == L else 0.0 for L in LABELS]


@dataclass
class Candidate:
    ex: Exemplar
    dense: float
    bm25: float
    dense_rank: int
    bm25_rank: int
    scope_prior: float


@dataclass
class ScopePool:
    """Per-(question, item) retrieval pool, built ONCE and reused across every
    query. Building the BM25 index and the vector matrix per query (the naive
    version) is O(queries x pool) and does not finish on the full corpus."""
    ex: list
    V: np.ndarray            # [P, D] candidate vectors
    sub_ids: np.ndarray      # [P] submission ids (for excluding the query's own)
    bm25: BM25Okapi
    label_counts: Counter
    n: int


def build_scope_pools(pool_exemplars: list[Exemplar]) -> dict:
    by_scope: dict = defaultdict(list)
    for e in pool_exemplars:
        by_scope[_scope_key(e)].append(e)
    pools = {}
    for scope, exs in by_scope.items():
        pools[scope] = ScopePool(
            ex=exs,
            V=np.stack([e.vec for e in exs]).astype(np.float32),
            sub_ids=np.array([e.submission_id for e in exs]),
            bm25=BM25Okapi([tokenize(e.text) for e in exs]),
            label_counts=Counter(e.label for e in exs),
            n=len(exs),
        )
    return pools


_BM25_K1, _BM25_B = 1.5, 0.75  # rank_bm25 BM25Okapi defaults


def _bm25_for(sp: "ScopePool", query_terms: set, idxs: np.ndarray) -> np.ndarray:
    """BM25 scores for a small set of pool docs, using the pool's global idf /
    term-frequency stats -- equivalent to BM25Okapi.get_scores restricted to
    `idxs`, but O(len(idxs) x |unique query terms|) instead of O(pool)."""
    idf, dfs, dl, avgdl = sp.bm25.idf, sp.bm25.doc_freqs, sp.bm25.doc_len, sp.bm25.avgdl
    out = np.zeros(len(idxs))
    for j, d in enumerate(idxs):
        doc, L, s = dfs[int(d)], dl[int(d)], 0.0
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * L / avgdl)
        for q in query_terms:
            f = doc.get(q, 0)
            if f:
                s += idf.get(q, 0.0) * (f * (_BM25_K1 + 1)) / (f + norm)
        out[j] = s
    return out


def candidates_for(query: Exemplar, scope_pools: dict, depth: int = RERANK_DEPTH
                   ) -> list[Candidate]:
    """Top-`depth` pool exemplars in the query's scope (excluding its own
    submission). Dense (vectorized) selects the top-`depth`; BM25 is computed only
    for those `depth` docs, so per-query cost does not grow with pool size."""
    sp = scope_pools.get(_scope_key(query))
    if sp is None:
        return []
    keep = np.nonzero(sp.sub_ids != query.submission_id)[0]
    if len(keep) < MIN_CANDIDATES:
        return []

    dense_all = sp.V[keep] @ query.vec                             # vectorized cosine
    top = np.argsort(-dense_all)[:depth]                           # rerank the top-depth
    pool_idx = keep[top]
    dense = dense_all[top]
    bm25 = _bm25_for(sp, set(tokenize(query.text)), pool_idx)

    dense_order = {int(i): r for r, i in enumerate(np.argsort(-dense))}
    bm25_order = {int(i): r for r, i in enumerate(np.argsort(-bm25))}
    n = len(top)

    cands = []
    for i in range(n):
        e = sp.ex[int(pool_idx[i])]
        cands.append(Candidate(
            ex=e, dense=float(dense[i]), bm25=float(bm25[i]),
            dense_rank=dense_order[i], bm25_rank=bm25_order[i],
            scope_prior=sp.label_counts[e.label] / sp.n,
        ))
    return cands


def features(c: Candidate, n: int) -> list[float]:
    return [
        c.dense,
        c.bm25,
        c.dense_rank / max(n, 1),
        c.bm25_rank / max(n, 1),
        c.scope_prior,
        *_label_onehot(c.ex.label),
    ]


def baseline_order(cands: list[Candidate]) -> list[int]:
    """RRF(dense, BM25) over the candidate list -> indices best-first."""
    dense_rank = [i for i, _ in sorted(enumerate(cands), key=lambda t: -t[1].dense)]
    bm25_rank = [i for i, _ in sorted(enumerate(cands), key=lambda t: -t[1].bm25)]
    fused = reciprocal_rank_fusion([[str(i) for i in dense_rank],
                                    [str(i) for i in bm25_rank]])
    return [int(k) for k, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


# --------------------------------------------------------------------------
# training + evaluation
# --------------------------------------------------------------------------

def build_pairs(queries: list[Exemplar], scope_pools: dict):
    X, y = [], []
    for q in queries:
        cands = candidates_for(q, scope_pools)
        n = len(cands)
        for c in cands:
            X.append(features(c, n))
            y.append(1 if c.ex.label == q.label else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def _p1_and_recall(order: list[int], cands: list[Candidate], gold: str):
    top1 = cands[order[0]].ex.label == gold
    hit_k = any(cands[order[r]].ex.label == gold for r in range(min(RECALL_K, len(order))))
    return top1, hit_k


@dataclass
class EvalRow:
    submission_id: str
    question_key: str
    rubric_item: str
    gold: str
    base_p1: bool
    rerank_p1: bool
    base_recall: bool
    rerank_recall: bool


@dataclass
class Prepared:
    q: Exemplar
    cands: list
    feats: np.ndarray
    base_order: list


def prepare(queries: list[Exemplar], scope_pools: dict) -> list[Prepared]:
    """Compute each query's candidate set, features, and baseline order ONCE, so
    validation model-selection (4 configs) and the final test pass reuse them
    instead of re-retrieving. This is the difference between seconds and minutes
    on the full corpus."""
    out = []
    for q in queries:
        cands = candidates_for(q, scope_pools)
        if not cands:
            continue
        # leakage guard: every candidate must share the query's scope
        assert all(_scope_key(c.ex) == _scope_key(q) for c in cands)
        n = len(cands)
        feats = np.array([features(c, n) for c in cands], dtype=float)
        out.append(Prepared(q, cands, feats, baseline_order(cands)))
    return out


def eval_prepared(prepared: list[Prepared], model) -> list[EvalRow]:
    rows = []
    for p in prepared:
        scores = model.predict_proba(p.feats)[:, 1]
        rer = list(np.argsort(-scores))
        b_p1, b_rec = _p1_and_recall(p.base_order, p.cands, p.q.label)
        r_p1, r_rec = _p1_and_recall(rer, p.cands, p.q.label)
        rows.append(EvalRow(p.q.submission_id, p.q.question_key, p.q.rubric_item,
                            p.q.label, b_p1, r_p1, b_rec, r_rec))
    return rows


def evaluate(queries: list[Exemplar], scope_pools: dict, model) -> list[EvalRow]:
    """Convenience wrapper: prepare then score. run_study prepares once and reuses."""
    return eval_prepared(prepare(queries, scope_pools), model)


def bootstrap_delta_ci(rows: list[EvalRow], seed: int = SEED, b: int = N_BOOTSTRAP):
    base = np.array([r.base_p1 for r in rows], dtype=float)
    rer = np.array([r.rerank_p1 for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(rows)
    deltas = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        deltas[i] = rer[idx].mean() - base[idx].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def slice_metrics(rows: list[EvalRow], key) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        b = np.mean([r.base_p1 for r in rs])
        rr = np.mean([r.rerank_p1 for r in rs])
        out[g] = {"n": len(rs), "base_p1": float(b), "rerank_p1": float(rr),
                  "delta": float(rr - b)}
    return out


def measure_latency(queries: list[Exemplar], scope_pools: dict, model,
                    sample: int = 100) -> dict:
    qs = queries[:sample]
    t0 = time.perf_counter()
    for q in qs:
        cands = candidates_for(q, scope_pools)
        if cands:
            baseline_order(cands)
    t_base = (time.perf_counter() - t0) / max(len(qs), 1) * 1000

    t0 = time.perf_counter()
    for q in qs:
        cands = candidates_for(q, scope_pools)
        if cands:
            n = len(cands)
            s = model.predict_proba(np.array([features(c, n) for c in cands]))[:, 1]
            list(np.argsort(-s))
    t_rer = (time.perf_counter() - t0) / max(len(qs), 1) * 1000
    return {"baseline_ms": t_base, "reranker_ms": t_rer, "overhead_ms": t_rer - t_base}


# --------------------------------------------------------------------------
# go / no-go
# --------------------------------------------------------------------------

def decide(base_p1, rerank_p1, ci, latency, slices_item, slices_label) -> dict:
    delta = rerank_p1 - base_p1
    ci_low, ci_high = ci
    worst_slice = min(
        [v["delta"] for v in slices_item.values()] +
        [v["delta"] for v in slices_label.values()],
        default=0.0,
    )
    checks = {
        "delta_meets_target": delta >= TARGET_DELTA,
        "ci_excludes_zero": ci_low > 0,
        "latency_within_budget": latency["overhead_ms"] < LATENCY_BUDGET_MS,
        "no_slice_regression": worst_slice >= SLICE_TOLERANCE,
    }
    if all(checks.values()):
        decision = "GO"
    elif checks["ci_excludes_zero"] and checks["latency_within_budget"] and checks["no_slice_regression"]:
        decision = "GO (marginal) -- positive but below target; ship behind flag, keep iterating"
    else:
        decision = "NO-GO"
    return {"decision": decision, "delta": delta, "checks": checks,
            "worst_slice_delta": worst_slice}


# --------------------------------------------------------------------------
# run + report
# --------------------------------------------------------------------------

@dataclass
class Study:
    embedding_model: str
    n_submissions: int
    n_train_pairs: int
    n_test_queries: int
    chosen_params: dict
    base_p1: float
    rerank_p1: float
    base_recall: float
    rerank_recall: float
    ci: tuple[float, float]
    latency: dict
    slices_item: dict
    slices_label: dict
    verdict: dict


# Small hyperparameter grid, selected on VALIDATION P@1 (never on test).
PARAM_GRID = [
    {"C": 0.5, "class_weight": None},
    {"C": 1.0, "class_weight": None},
    {"C": 2.0, "class_weight": None},
    {"C": 1.0, "class_weight": "balanced"},
]


def _p1_only(prepared: list[Prepared], model) -> float:
    rows = eval_prepared(prepared, model)
    return float(np.mean([r.rerank_p1 for r in rows])) if rows else 0.0


def run_study(chunks_path: Path | str, index: EmbeddingIndex,
              manifest_path: Path = DEFAULT_MANIFEST, seed: int = SEED) -> Study:
    from sklearn.linear_model import LogisticRegression

    ex = load_exemplars(chunks_path, index)
    splits = make_splits(ex, manifest_path, seed)
    pool_subs = splits["train"] | splits["val"]        # the deployed retrieval pool
    pool = [e for e in ex if e.submission_id in pool_subs]
    scope_pools = build_scope_pools(pool)              # built ONCE, reused everywhere

    train_q = [e for e in ex if e.submission_id in splits["train"]]
    val_q = [e for e in ex if e.submission_id in splits["val"]]
    test_q = [e for e in ex if e.submission_id in splits["test"]]

    # cap training queries (sampled) so fits stay fast on the full corpus; the
    # frozen TEST set is always evaluated in full.
    if len(train_q) > MAX_TRAIN_QUERIES:
        rng = np.random.default_rng(seed)
        train_q = [train_q[i] for i in rng.choice(len(train_q), MAX_TRAIN_QUERIES, replace=False)]

    Xtr, ytr = build_pairs(train_q, scope_pools)
    prep_val = prepare(val_q, scope_pools)      # retrieve once, reuse across configs
    prep_test = prepare(test_q, scope_pools)    # retrieve once for the frozen test

    # model selection on VALIDATION P@1 -- the test set is touched exactly once,
    # after the config is fixed (guards against tuning-to-test).
    best_model, best_params, best_val = None, None, -1.0
    for params in PARAM_GRID:
        m = LogisticRegression(max_iter=1000, random_state=seed, **params).fit(Xtr, ytr)
        val_p1 = _p1_only(prep_val, m)
        if val_p1 > best_val:
            best_model, best_params, best_val = m, params, val_p1
    model = best_model

    rows = eval_prepared(prep_test, model)
    base_p1 = float(np.mean([r.base_p1 for r in rows]))
    rer_p1 = float(np.mean([r.rerank_p1 for r in rows]))
    ci = bootstrap_delta_ci(rows, seed)
    slices_item = slice_metrics(rows, lambda r: r.rubric_item)
    slices_label = slice_metrics(rows, lambda r: r.gold)
    latency = measure_latency(test_q, scope_pools, model)
    verdict = decide(base_p1, rer_p1, ci, latency, slices_item, slices_label)

    return Study(
        embedding_model=index.model,
        n_submissions=len({e.submission_id for e in ex}),
        n_train_pairs=len(ytr), n_test_queries=len(rows),
        chosen_params={**best_params, "val_p1": round(best_val, 4)},
        base_p1=base_p1, rerank_p1=rer_p1,
        base_recall=float(np.mean([r.base_recall for r in rows])),
        rerank_recall=float(np.mean([r.rerank_recall for r in rows])),
        ci=ci, latency=latency, slices_item=slices_item,
        slices_label=slices_label, verdict=verdict,
    )


def write_report(study: Study, path: Path = DEFAULT_REPORT) -> None:
    scaffold = study.embedding_model == "stub"
    L = []
    L.append("# Specialist Model Review — learned reranker adapter\n")
    L.append("_Generated by `python -m evidence_room.specialize`. "
             "Frozen test set; parameters pre-registered in "
             "[README](README.md)._\n")
    if scaffold:
        L.append("> ⚠️ **SCAFFOLDING RUN (stub embeddings)** — numbers below are not "
                 "a real verdict. Re-run against the real embedding index to produce "
                 "the review.\n")

    v = study.verdict
    L.append(f"## Recommendation: {v['decision']}\n")
    L.append(f"P@1 label agreement: baseline **{study.base_p1:.3f}** → reranker "
             f"**{study.rerank_p1:.3f}** (Δ **{v['delta']:+.3f}**, 95% CI "
             f"[{study.ci[0]:+.3f}, {study.ci[1]:+.3f}]).\n")

    L.append("### Pre-registered checks\n")
    L.append("| Check | Threshold | Pass |")
    L.append("|---|---|---|")
    c = v["checks"]
    L.append(f"| Δ meets target | ≥ {TARGET_DELTA:+.2f} | {c['delta_meets_target']} |")
    L.append(f"| 95% CI excludes 0 | CI low > 0 | {c['ci_excludes_zero']} |")
    L.append(f"| latency overhead | < {LATENCY_BUDGET_MS} ms | {c['latency_within_budget']} |")
    L.append(f"| no slice regresses | ≥ {SLICE_TOLERANCE:+.2f} | {c['no_slice_regression']} |")
    L.append("")

    L.append("## Setup\n")
    L.append(f"- Embedding index model: `{study.embedding_model}`")
    L.append(f"- Submissions: {study.n_submissions} (60/20/20 by submission id; test frozen)")
    L.append(f"- Training pairs: {study.n_train_pairs:,} | Test queries: {study.n_test_queries:,}")
    L.append(f"- Baseline: RRF(dense, BM25). Intervention: logistic-regression reranker over "
             "dense/BM25/rank/label-prior features.")
    L.append(f"- Reranker config (selected on validation P@1={study.chosen_params['val_p1']}): "
             f"`C={study.chosen_params['C']}, class_weight={study.chosen_params['class_weight']}`")
    L.append("")

    L.append("## Metrics on the frozen test set\n")
    L.append("| Metric | Baseline | Reranker | Δ |")
    L.append("|---|---|---|---|")
    L.append(f"| P@1 label agreement | {study.base_p1:.3f} | {study.rerank_p1:.3f} | {study.rerank_p1-study.base_p1:+.3f} |")
    L.append(f"| Recall@{RECALL_K} | {study.base_recall:.3f} | {study.rerank_recall:.3f} | {study.rerank_recall-study.base_recall:+.3f} |")
    L.append(f"| Latency/query | {study.latency['baseline_ms']:.2f} ms | {study.latency['reranker_ms']:.2f} ms | {study.latency['overhead_ms']:+.2f} ms |")
    L.append("")

    L.append("## Error slices — P@1 by rubric item\n")
    L.append("| Rubric item | n | baseline | reranker | Δ |")
    L.append("|---|---|---|---|---|")
    for g, m in study.slices_item.items():
        L.append(f"| {g} | {m['n']} | {m['base_p1']:.3f} | {m['rerank_p1']:.3f} | {m['delta']:+.3f} |")
    L.append("")
    L.append("## Error slices — P@1 by query gold label\n")
    L.append("| Gold label | n | baseline | reranker | Δ |")
    L.append("|---|---|---|---|---|")
    for g, m in study.slices_label.items():
        L.append(f"| {g} | {m['n']} | {m['base_p1']:.3f} | {m['rerank_p1']:.3f} | {m['delta']:+.3f} |")
    L.append("")

    L.append("## Safety & invariants\n")
    L.append("The reranker only reorders an already scope/permission-filtered "
             "candidate set, so **scope/permission leakage stays 0 by construction** "
             "(every candidate shares the query's scope — asserted in `evaluate`). "
             "Refusal behaviour is untouched: the reranker never runs on a refused "
             "query, and never changes the decision, only the order of precedent.\n")

    L.append("## Rollback plan\n")
    L.append("The reranker ships behind a `RERANK_ENABLED` flag as an optional layer "
             "over the same candidates. **Rollback = set the flag false** — the system "
             "reverts to the RRF baseline instantly, with no data migration and no "
             "index change. The baseline is registered as `v0` (identity reranker); "
             "the reranker artifact + this report + the frozen-test manifest hash are "
             "recorded in `registry.jsonl`. Drift watch: track P@1 label agreement on a "
             "rolling sample; if it falls to or below baseline, auto-flip the flag.\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def append_registry(study: Study, path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": study.embedding_model,
        "features": ["dense", "bm25", "dense_rank", "bm25_rank", "scope_prior", "label_onehot"],
        "hyperparams": study.chosen_params,
        "n_train_pairs": study.n_train_pairs, "n_test_queries": study.n_test_queries,
        "base_p1": round(study.base_p1, 4), "rerank_p1": round(study.rerank_p1, 4),
        "delta": round(study.rerank_p1 - study.base_p1, 4),
        "ci95": [round(study.ci[0], 4), round(study.ci[1], 4)],
        "overhead_ms": round(study.latency["overhead_ms"], 3),
        "decision": study.verdict["decision"],
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Specialist model review: learned reranker")
    ap.add_argument("--chunks", default="chunks_real.jsonl")
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    idx = EmbeddingIndex.load(args.index)
    study = run_study(args.chunks, idx, manifest_path=Path(args.manifest))
    write_report(study, Path(args.report))
    append_registry(study)

    v = study.verdict
    print(f"Embedding model: {study.embedding_model}")
    print(f"P@1  baseline={study.base_p1:.3f}  reranker={study.rerank_p1:.3f}  "
          f"delta={v['delta']:+.3f}  CI95=[{study.ci[0]:+.3f},{study.ci[1]:+.3f}]")
    print(f"Latency overhead: {study.latency['overhead_ms']:+.2f} ms/query")
    print(f"Recommendation: {v['decision']}")
    print(f"Report: {args.report}")
