# The Specialist Model Review

Day 4 build lab. Fine-tuning is not a status symbol; it is a controlled
intervention with a measurable hypothesis — and often the right answer is *don't*.
This review starts from the baseline system, forms a pre-registered hypothesis,
builds a dataset, trains the intervention, evaluates against a **frozen holdout**,
and issues a go/no-go recommendation with a rollback plan.

- Decision framework + hypothesis: this file
- Dataset design (schema, provenance, splits, leakage controls): [dataset.md](dataset.md)
- Harness: [`src/evidence_room/specialize.py`](../../src/evidence_room/specialize.py)
- Generated verdict: `model-review.md` (run the harness), registry: `registry.jsonl`

## Should we tune at all? (decision framework)

Four interventions, cheapest first. The rule of thumb: **exhaust the cheaper,
more reversible options before touching model weights.**

| Intervention | Use when | Cost / reversibility | Our case |
|---|---|---|---|
| **Prompting** | behaviour is in the model, you just need to elicit it | trivial / instant | N/A — no LLM in the ranking loop (deterministic composer) |
| **Retrieval** | the model lacks *facts/precedent*, not skill | low / instant | already the core system; strong (eval: refusal 1.0, leakage 0) |
| **Workflow / adapter** | a cheap learned layer can fix a localized error | low / feature-flag | **the intervention under test** — a reranker over retrieval features |
| **Fine-tune (SFT/LoRA)** | the base model systematically fails a skill and cheaper options are exhausted | high / retrain + registry + rollback | **not justified** — see below |

**Why not fine-tune the drafter (an LLM SFT/LoRA).** Three reasons, and any one
is sufficient: (1) *data residency* — SFT on graded proofs would move consented
student coursework into a training pipeline, against the rule that has governed
the whole project; (2) *the bar is already met* — retrieval + the deterministic
composer pass the Days 1-2 gate, so there is no failure for a tune to fix; (3)
*cost/rollback* — a tuned generator adds GPU cost, a model registry, drift risk,
and a slow rollback, to buy fluency the system does not need for correctness.

**Why not fine-tune the embedder (LoRA) yet.** The eval says the correct-label
precedent is almost always retrieved (recall@5 = 0.94) but not always ranked
first (precision@1 ≈ 0.68). That is an *ordering* problem, not a *recall*
problem — so the evidence points at the cheapest thing that could fix ordering: a
reranker over the features we already compute. A heavier embedding LoRA is
deferred **unless** the reranker shows the ordering signal exists but a linear
model cannot capture it. That is what "specialize only when the evidence says to"
means in practice.

## Pre-registered hypothesis

> A learned reranker over retrieval features will lift **top-1 correct-label
> precedent (P@1 label agreement)** on the frozen holdout by **≥ 10 points** vs
> the RRF baseline, with a **95% bootstrap CI lower bound > 0**, at **< 5 ms**
> added latency per query, and **zero regression** on the scope/permission
> invariant.

Parameters are fixed in code before the test set is touched (`specialize.py`:
`TARGET_DELTA`, `LATENCY_BUDGET_MS`, `SLICE_TOLERANCE`, `SEED`). Go/no-go rule:

- **GO** — all four checks pass (Δ ≥ target, CI excludes 0, latency in budget, no
  slice regresses > 5 points).
- **GO (marginal)** — CI excludes 0 and latency/slices are fine but Δ < target:
  ship behind the flag, keep iterating (e.g., escalate to the deferred embedding
  LoRA).
- **NO-GO** — otherwise. A NO-GO is a real, publishable result: it says the
  cheap intervention does not clear the bar and the heavier one is not yet
  warranted either.

This structure exists so the recommendation cannot be cherry-picked: the metric,
the holdout, and the thresholds are committed up front, and the test set is
evaluated exactly once (model selection happens on validation).

## Evaluation beyond loss

The review reports, on the frozen test set (see `model-review.md`): task success
(P@1 label agreement, recall@k), **error slices** by rubric item and by query
gold label (the *partial* slice is where minority-label evidence is scarce and
where a win would matter most), a **bootstrap confidence interval** on the delta,
and a **latency/cost** comparison. Safety is invariant by construction: the
reranker only reorders an already scope/permission-filtered candidate set, so
leakage stays 0 and the refusal decision is untouched.

## Registry, rollback, drift

- **Registry** — every run appends to `registry.jsonl`: embedding model, feature
  list, chosen hyperparameters, metrics, CI, latency, and decision. The baseline
  is `v0` (identity reranker = RRF).
- **Rollback** — the reranker ships behind a `RERANK_ENABLED` flag as an optional
  layer over the same candidates. Rollback is flipping the flag: instant revert
  to RRF, no data migration, no index rebuild.
- **Drift** — monitor P@1 label agreement on a rolling sample of graded items; if
  it falls to or below the registered baseline, auto-flip the flag and re-open the
  review. The frozen test manifest (`test_manifest.json`) makes every re-run
  comparable to the original holdout.

## Reproduce

```bash
cd src
python -m evidence_room.specialize        # writes docs/finetune/model-review.md + registry.jsonl
```

Requires the built embedding index (`evidence_room.embeddings`) and
`scikit-learn`. The frozen test set is created on first run and reused thereafter.
