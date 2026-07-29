# Evidence Room

A retrieval layer for rubric-based grading of induction proofs.

Built for the Pareto FDE Academy (Cohort 01), Ground stage. The wider project is
a grading companion that shifts a Faculty Intern's work from *authoring* a
deduction to *reviewing* a suggested one -- see [`docs/opportunity-brief.md`](docs/opportunity-brief.md).

## The problem this addresses

A Faculty Intern grading proof-based assignments builds a deduction bank live,
while grading. Two costs recur every assignment:

1. **Repeated authoring** -- the same judgment is written out again each time a
   recurring error appears.
2. **Inconsistent application** -- once a tag exists, whether it gets applied to
   the remaining scripts depends on the grader remembering it, not on the tag
   being checked against each one.

This repo builds the evidence layer for addressing (2): given a student proof and
a rubric item, retrieve the guidance and prior expert judgments that justify a
suggested deduction, with citations back to source.

## Status

| Stage | State |
|---|---|
| Type-aware ingestion + chunking | done |
| Access-scoped index | done |
| Embeddings + hybrid retrieval | not started |
| Stratified evaluation set | not started |
| Retrieval traces + eval report | not started |

## Design notes

**Chunking is type-aware, not size-aware.** A fixed 512-token window would be
actively wrong here. Rubric guidance is the unit a lecturer approved; splitting
it breaks the audit trail. A graded exemplar is evidence about exactly one
rubric item; merging seven items into one chunk makes it impossible to cite
which judgment an exemplar supports.

**Two document families, different roles.**

- `rubric_guidance` (7 chunks) -- the deduction bank. Each rubric item with its
  written edge-case rationale. Retrieved as *why* a deduction applies.
- `graded_exemplar` (~25k chunks) -- one per (submission x rubric item). Real
  student proofs with expert labels. Retrieved as *precedent*: a proof
  previously judged this way on this item.

**General grading rules are appended to every rubric chunk** rather than indexed
separately. A retrieved deduction is only defensible if the caveats governing it
travel with it -- otherwise you can cite "base case not identified" while missing
the rule that implicit identification earns credit.

**Access filtering happens pre-retrieval, at the index level.** When grading
question X, exemplars from question Y are never candidates -- their labels were
assigned against a different claim. Filtering the candidate set rather than
instructing the model means an out-of-scope chunk cannot be cited even under a
confused or injected prompt.

## Known weaknesses

- **Severe label imbalance, worsening down the proof.** `Identify Base Case`
  splits 48/40/12 (correct/absent/partial), but `Inductive Hypothesis is
  applied` is 28/70/2 -- only ~75 partial examples across 3,600 submissions.
  Partial credit is where grader judgment is hardest and where retrieval would
  add most value, so the eval set must be stratified rather than randomly
  sampled.
- **Proof text duplicates 7x** across the item-chunks of a submission. The
  embedded text genuinely differs (each is framed by its own rubric item and
  label), but near-duplicate co-retrieval is likely. `provenance.submission_id`
  supports de-duplication at display time.
- **Retrieval over 7 rubric chunks is probably unnecessary.** A direct lookup by
  rubric item is more reliable than vector search over a 7-document set.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Obtain the corpus (see [`data/README.md`](data/README.md)), then:

```bash
cd src
python -m evidence_room.ingest --verify     # check corpus is present
python -m evidence_room.ingest --limit 50   # fast iteration
python -m evidence_room.ingest              # full index
```

Set `EVIDENCE_ROOM_DATA` to point elsewhere if the corpus lives outside `data/`.

## Data handling

The corpus is real student coursework collected under institutional consent. It
is **not committed to this repo** and must not be. Derived artifacts (chunk
files, embedding indexes) contain verbatim student text and are gitignored for
the same reason. See [`data/README.md`](data/README.md).

## Prior art

Zhao, Silva & Poulsen, *Language Models are Few-Shot Graders* (2025) evaluated
LLM grading of induction proofs against this same 7-item rubric, and found that
retrieval-selected graded examples outperformed randomly selected ones, with the
rubric in-prompt improving accuracy further.

This project differs in what it retrieves *for*: not a final score, but a
suggested deduction the grader reviews, drawn from a bank that grows during the
session.

## Layout

```
src/evidence_room/
  ingest.py             type-aware chunking, access filter, CLI
  synthetic_corpus.py   hand-written corpus from the design phase, kept for tests
data/                   corpus goes here (untracked)
eval/                   evaluation sets and reports
docs/                   opportunity brief, process map, decision records
tests/
```
