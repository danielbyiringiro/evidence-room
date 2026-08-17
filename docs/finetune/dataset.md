# Dataset design — reranker adapter

The intervention is a learned reranker, so the "dataset" is a set of
**(query, candidate) pairs** derived from the existing corpus. No new labeling is
required: the expert grades already in the corpus are the supervision.

## Task definition

Given a held-out student proof scoped to a `(question, rubric_item)`, rank the
retrieved candidate exemplars so that those sharing the query's **gold expert
label** appear first. Success is measured as **P@1 label agreement** (does the
top-ranked precedent carry the label the query was actually graded).

This is deliberately the *ordering* task the offline eval flagged (F3), not a new
classification task — the reranker reorders precedent, it does not grade.

## Schema

One training row per (query exemplar, candidate exemplar) pair within the query's
scope:

| Field | Type | Notes |
|---|---|---|
| `dense` | float | cosine(query vec, candidate vec) — both from the embedding index |
| `bm25` | float | BM25(query text, candidate text) within the scope pool |
| `dense_rank`, `bm25_rank` | float | candidate's rank under each signal, normalized |
| `scope_prior` | float | fraction of the scope pool carrying the candidate's label |
| `label_onehot` | 3× float | candidate label ∈ {correct, partial, not present} |
| **target** | 0/1 | 1 if candidate label == **query gold label** |

The query's gold label is the *target basis* and is **never a feature** — the
reranker must predict relevance from similarity + candidate-side signals that are
all available at inference time. Candidate labels are known at inference (they are
graded precedent), so using them is legitimate, not leakage.

## Diversity & edge cases

- **All four questions** (sum-formula, factorial-sum, recurrence, divisibility)
  and **all seven rubric items** contribute pairs; the reranker is not tuned to
  one scope.
- **Label imbalance is represented, not corrected away.** The corpus is heavily
  `correct`-skewed on the later items (~75 `partial` examples in 3,600
  submissions). Rather than resample, the review reports an **error slice by gold
  label**, so a method that only helps the majority `correct` slice cannot hide a
  regression on the scarce `partial` slice.
- **Small scopes** (fewer than `MIN_CANDIDATES` in-scope candidates) are skipped —
  ranking a set of 2 is not a meaningful test.
- Candidates are the **top-`RERANK_DEPTH`** by dense similarity, mirroring
  deployment (a reranker reorders the top of the retrieved list, not the whole
  corpus).

## Splits, leakage, contamination

Splits are by **submission id**, not by pair — the unit that must not straddle
the boundary is a student's submission.

- **60 / 20 / 20** train / validation / test, grouped by submission (seeded).
- **The retrieval pool = train + validation submissions** (the "deployed index").
  **Test submissions are held out of the pool entirely**, so a test query
  retrieves precedent only from indexed submissions — exactly the deployment
  condition (a new proof, not in the index, retrieving prior graded work).
- **No pair straddles the split:** a test query's submission never appears as a
  candidate in training, and never in the pool.
- **No target leakage:** no feature is derived from the query's gold label.
- **Model selection on validation only.** Hyperparameters are chosen by
  validation P@1; the **test set is evaluated exactly once**, after the config is
  frozen. This is the guard against tuning-to-test.

### Frozen holdout

The test submission ids are written to `test_manifest.json` on first run and
reused verbatim thereafter, so every subsequent run — and every future model
version in the registry — is scored against the **same** holdout. Changing the
seed does not move a frozen test set; deleting the manifest (deliberately)
re-draws it.

## Provenance & licensing

Derived from the **Student Proof by Induction** dataset — Poulsen, Seth (2024),
Harvard Dataverse, <https://doi.org/10.7910/DVN/OTRLXF> — licensed **CC BY-NC
4.0**. This review is non-commercial academic work under those terms.

The reranker dataset is **derived features + a binary target**, not redistributed
student text; the pairs are built at runtime from the local corpus and are not
committed. The frozen-test manifest stores **submission ids only** (no proof
text). As with every stage, verbatim student coursework never leaves the machine
and is never committed.
