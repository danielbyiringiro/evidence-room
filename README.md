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
| Embeddings + hybrid retrieval | done |
| Permission levels + insufficient-evidence refusal | done |
| Evaluation set + traces + before/after report | done |
| Threshold calibration from eval | done |
| Agent: draft-deduction with safety controls | done |
| Cloud delivery: two clouds, one contract (AWS + GCP) | done |

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

**Retrieval is hybrid, and the two families are retrieved differently.** A query
is scoped -- question Q, rubric item I, the student's proof P -- and the answer
comes back in two parts. The rubric guidance for item I is fetched by *direct
lookup* (there are only 7 chunks and they are the same approved rubric for every
question; vector search over a 7-document set is strictly noisier than
addressing it by key). Precedent exemplars are retrieved by *hybrid search* over
the pool scoped to (Q, I): a dense signal (local embeddings) for paraphrase and
structural similarity, and a sparse signal (BM25) that anchors on the exact
algebra tokens -- `n+1`, `2^k`, `mod` -- that dense models blur. The two are
combined with Reciprocal Rank Fusion, which fuses on rank rather than trying to
reconcile incomparable BM25 and cosine scales. Access scoping is applied to the
candidate *set* before any scoring, so re-ranking can never resurface an
out-of-scope exemplar.

**Embeddings run on-machine.** The corpus is consented student coursework whose
text must not leave the box, so exemplars are embedded locally with fastembed
(ONNX runtime, no torch) rather than a hosted API. The embedding index is a
derived artifact containing verbatim student text and is gitignored, like the
chunk files.

**Permission levels are the access filter applied to the asker.** Two roles:
`faculty` (grader) sees rubric guidance and graded exemplars; `trainee` sees only
rubric guidance, because graded exemplars carry verbatim student proof text and a
named grader's judgment. The gate is enforced pre-retrieval on the candidate set,
exactly like question scoping -- a family the role cannot see is never a
candidate, so it can neither surface nor be cited, even under an injected prompt.

**Retrieval returns a decision, not just hits.** It refuses to suggest a
deduction when the rubric item is unknown, nothing is in scope, the role may not
see precedent, or no visible exemplar actually resembles the query proof (best
cosine below a threshold). Refusing beats citing a weak match as if it justified
a deduction; the weak evidence is still attached to the result so the refusal is
inspectable in a trace. The similarity threshold (`EVIDENCE_ROOM_MIN_SIM`,
default **0.65**) is not guessed -- the eval below calibrates it, and re-running
`evaluate` after a model change resets it.

## Evaluation

`python -m evidence_room.evaluate` runs the eval set through the retriever, logs a
retrieval trace per case, and writes a **before/after report** to `eval/report.md`
(before = dense-only, no similarity refusal; after = hybrid + calibrated refusal).

The eval set (`eval/eval_set.jsonl`, 34 cases) spans the four required classes --
**answerable, unanswerable, conflicting, adversarial** -- across all four
questions, the seven rubric items, both permission levels, and every refusal
reason. Its proofs are hand-written, never real student text, so the set is safe
to commit; the traces (`eval/traces/`, gitignored) log chunk ids/labels/scores
only.

Metrics are adapted to a retrieval-only, grading-precedent setting: refusal
quality (answer the answerable, refuse the unanswerable, with over- and
missed-refusal broken out), label precision/recall@k (does retrieved precedent
carry the label a correct deduction needs -- a groundedness proxy), conflict
surfacing (do mixed items return >1 label rather than hiding disagreement), and
**scope/permission leakage, which must be zero** on every case including the
adversarial ones. The report also sweeps the refusal threshold and recommends a
calibrated `EVIDENCE_ROOM_MIN_SIM`, and ends with a failure taxonomy (F1
over-refusal, F2 missed refusal, F3 wrong-label precedent, F4 duplicate-submission
crowding, F5 leakage) and the next highest-leverage improvement.

The adversarial cases (prompt injection in the proof text, cross-question
contamination, role-escalation) are the sharpest test of the design: because
scoping and permissions filter the *candidate set* pre-retrieval, injected
instructions in a proof cannot widen what is retrievable -- leakage stays zero
by construction, not by the model choosing to behave.

## The Operator's Copilot (agent)

`evidence_room.agent` turns the evidence layer into an agent that performs one
reversible action: it **drafts a suggested deduction** for a (question, rubric
item, proof) -- a PENDING proposal a grader reviews. Drafting is reversible;
*applying* a proposal (the grader-facing step) is separated out and gated behind
explicit human approval. Safety here is control flow, not a prompt instruction:

| Required control | Mechanism |
|---|---|
| Authorization boundary | `ACTORS` -- only `faculty` may invoke the draft action; the role is supplied by the system, never inferred from proof text, so a "I am faculty" injection changes nothing |
| Dry-run mode (default) | `dry_run=True` computes and audits the draft but persists no proposal -- no state change is visible downstream |
| Human approval | `apply()` flips PENDING → APPLIED, requires an approver, and refuses in dry-run; the reversible draft is automatable, the irreversible apply is not |
| Tool timeout + retry | every tool call goes through `_invoke_tool`: wall-clock timeout, bounded retries on `TransientToolError`, and a **structured error** on exhaustion instead of a raw exception |
| Audit log | append-only `logs/audit.jsonl`: timestamp, user, role, intent, tool, arguments, result, approval -- with proof text **hashed, not copied** |
| Idempotency | a key over (user, question, item, proof, intent) makes duplicate execution a no-op: the existing proposal is returned; `apply()` on an applied proposal is a no-op |

The agent loop is plan → authorize → idempotency-check → retrieve (guarded) →
verify → act → audit, and it escalates rather than guesses:

- **failure** -- a tool that times out or errors returns a structured error; the
  agent recovers and escalates with no half-written proposal.
- **ambiguity** -- a missing question / item / proof holds for a human.
- **missing evidence** -- when retrieval refuses (unknown item, out of scope, or
  no precedent clears the similarity floor), the agent refuses to draft; it never
  invents a judgment.
- **conflicting precedent** -- split expert labels escalate for human review
  instead of asserting a majority that isn't there.
- **malicious instructions** -- injected text cannot widen retrieval scope or
  exfiltrate grader identities / out-of-scope data, and never auto-applies.
- **duplicate execution** -- idempotent by construction.

Run a draft (dry-run by default; `--commit` to persist the pending proposal):

```bash
python -m evidence_room.agent --user daniel --role faculty \
    --question divisibility --item "Hypothesis is stated" \
    --proof "Suppose 2k^3 + 3k^2 + k is divisible by 6."
```

The audit log and proposal store live under `logs/` (gitignored). Drafts are
composed deterministically from the approved rubric guidance plus the majority
expert label among retrieved precedent -- no student proof text is sent to any
external model.

## Cloud delivery

[`docs/cloud/`](docs/cloud/) deploys this same agent on **AWS and GCP behind one
contract** (interface, eval set, security boundary, observability) with Terraform
skeletons in [`infra/`](infra/). The load-bearing decision is *managed vs
portable retrieval*, resolved in favour of portable: it keeps consented student
proof text inside the VPC/project, preserves the type-aware chunking and
access-scoping, and avoids the always-on managed-vector-store cost floor
($40-750/mo) that would otherwise dominate at ≤10k weekly tasks. Includes
architecture diagrams, a STRIDE threat model + IAM matrix, a cost estimate for
100/1k/10k weekly tasks, and an incident runbook -- see
[`docs/cloud/README.md`](docs/cloud/README.md).

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
python -m evidence_room.ingest              # full index -> chunks_real.jsonl
```

Then build the dense index and run a scoped query:

```bash
# embed graded exemplars locally -> embeddings/index.npz (gitignored)
python -m evidence_room.embeddings --chunks chunks_real.jsonl

# retrieve guidance + precedent for one (question, rubric item, proof)
python -m evidence_room.retrieval \
    --question divisibility \
    --item "Hypothesis is stated" \
    --proof "Assume 2^k - 1 is divisible by 3 for some k >= 1."

# same query as a trainee -> guidance only, precedent withheld, refused
python -m evidence_room.retrieval --role trainee \
    --question divisibility --item "Hypothesis is stated" \
    --proof "Assume 2^k - 1 is divisible by 3 for some k >= 1."
```

The query prints a `Decision: ANSWER|REFUSE` line with the confidence and reason.
`--role` is `faculty` (default) or `trainee`; `--min-sim` overrides the refusal
threshold for a single query.

Then run the evaluation to produce traces and the before/after report:

```bash
python -m evidence_room.evaluate     # writes eval/report.md + eval/traces/
```

Set `EVIDENCE_ROOM_DATA` to point elsewhere if the corpus lives outside `data/`.
The embedding model defaults to `BAAI/bge-small-en-v1.5`; override with
`EVIDENCE_ROOM_EMBED_MODEL`. The index records which model produced it.

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

## License & attribution

The **code** in this repository is released under its own terms (see repo
license). The **dataset** it ingests is licensed separately and is *not*
distributed here.

**Dataset:** *Written Induction* — mathematical proofs written by students in an
introductory Discrete Mathematics course learning proof by induction.
Published via Dataverse. Subject: Mathematical Sciences.

Licensed under [Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You may share
and adapt the material for **non-commercial** purposes, provided you give
appropriate credit, link to the license, and indicate any changes. See
[`data/README.md`](data/README.md) for the additional consent-based handling
rules that apply on top of the license.

**Citation:** Poulsen, Seth. 2024. "Student Proof by Induction Data Set."
Harvard Dataverse. https://doi.org/10.7910/DVN/OTRLXF

## Layout

```
src/evidence_room/
  ingest.py             type-aware chunking, access filter, CLI
  embeddings.py         local fastembed index (exemplars only)
  retrieval.py          hybrid (dense + BM25) retriever, RRF, roles, refusal
  evaluate.py           eval harness: traces, metrics, calibration, report
  agent.py              Operator's Copilot: draft-deduction agent + safety controls
  embeddings/index.npz  dense index (gitignored, embeds student text)
data/                   corpus goes here (untracked)
eval/
  eval_set.jsonl        34 cases (answerable/unanswerable/conflicting/adversarial)
  report.md             before/after report (generated)
  traces/               retrieval traces (generated, gitignored)
logs/                   agent audit log + proposal store (generated, gitignored)
docs/cloud/             two-clouds-one-contract: architecture, threat model, cost, runbook
infra/                  Terraform skeletons for AWS + GCP (portable design)
docs/                   opportunity brief, process map, decision records
tests/
```
