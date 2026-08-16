# The application contract

*Two clouds, one contract.* The model and managed services may differ between AWS
and GCP. The four things below may **not** — they are the contract every
deployment has to honour, and the thing the evaluation gates on.

This is the same Evidence Room agent from the earlier stages (retrieval →
refusal → draft-deduction agent), lifted onto a cloud. Nothing about *what the
system does or is allowed to do* changes; only *where the boxes run*.

## 1. Interface (stable API)

Three operations, typed. Request/response shapes are identical on both clouds;
only the transport (API Gateway + Lambda vs Cloud Run) differs.

| Operation | Input | Output | Notes |
|---|---|---|---|
| `POST /retrieve` | `{question_key, rubric_item, proof_text, role}` | `RetrievalResult{decision, refusal_reason, confidence, n_candidates, guidance, exemplars[]}` | read-only; access-scoped |
| `POST /draft` | `{user, role, question_key, rubric_item, proof_text, intent, dry_run}` | `AgentOutcome{status, reason, proposal?}` | reversible; default `dry_run=true` |
| `POST /apply` | `{proposal_id, approver}` | `AgentOutcome{status, proposal}` | irreversible; requires approval |

These map 1:1 to `HybridRetriever.retrieve`, `OperatorCopilot.draft`, and
`OperatorCopilot.apply` in the codebase. The contract is the Python package's
public surface, wrapped in HTTP — so the *same* container image serves both
clouds (see [architecture.md](architecture.md)).

Typed errors are part of the contract: a tool failure returns a structured
`{status: "tool_error", tool_error:{type,message}}`, never a 500 with a stack
trace.

## 2. Evaluation set

`eval/eval_set.jsonl` (34 cases: answerable / unanswerable / conflicting /
adversarial) is **cloud-invariant**. The same set runs against each deployment
and the gate thresholds must hold on both:

- refusal accuracy ≥ the calibrated baseline (currently 1.00 after calibration),
- **scope / permission leakage = 0** on every case, including adversarial,
- refusal-threshold calibration reproducible from the deployed model.

A cloud swap that changes the embedding model (e.g. `bge-small` → Titan V2 →
Vertex `text-embedding`) **must re-run `evidence_room.evaluate`** and re-calibrate
`EVIDENCE_ROOM_MIN_SIM`; the number is model-specific and the eval is how it is
re-derived. Evaluation is a release gate on both clouds, not a one-off.

## 3. Security boundary

The invariant that has driven every design decision: **verbatim student proof
text must not leave the institutional trust boundary** (the VPC / project), and
graded exemplars are never exposed across question scope, rubric item, or
permission level.

Concretely, on both clouds:

- **Authorization** is server-side and role-based (`faculty` / `trainee`); the
  role is supplied by the platform's identity layer, never inferred from request
  content. Pre-retrieval scoping + permission filtering are enforced in the app,
  not delegated to a prompt.
- **Data residency.** Proof text is embedded and drafted **inside** the trust
  boundary. This is the decisive input to the managed-vs-portable choice: a
  managed embedding/LLM API (Bedrock, Vertex) transmits proof text to the
  provider's endpoint. That is only acceptable under an institution-approved
  zero-retention / no-training agreement; absent that, the **portable** embedding
  layer (self-hosted `fastembed` in-container) keeps proof text in-VPC. See the
  [architecture](architecture.md) recommendation.
- **Secrets** live in Secrets Manager (AWS) / Secret Manager (GCP), never in env
  files or images.
- **Audit** is append-only and immutable (object-lock / retention policy); proof
  text is stored **hashed**, never verbatim (matches the agent's audit design).
- **Least privilege** IAM: each component gets only the actions on only the
  resources it needs. See the [IAM matrix](threat-model.md#iam-matrix).

## 4. Observability

Same signals, same alarm conditions on both clouds; only the product names differ
(CloudWatch vs Cloud Logging/Monitoring).

| Signal | What | Alarm |
|---|---|---|
| Audit log | every `draft`/`apply`: user, role, intent, tool, args (hashed), result, approval | write failure |
| Retrieval traces | per request: decision, confidence, n_candidates, cited ids | — |
| **Leakage counter** | out-of-scope / cross-role citations | **> 0 → page** (hard invariant) |
| Refusal rate | share of requests refused, by reason | sudden spike or drop |
| Latency | p50/p95 for retrieve/draft | p95 breach |
| Cost per task | model + infra spend / tasks | budget threshold ([runbook](runbook.md)) |

The leakage counter firing is the one condition that pages a human immediately;
the [runbook](runbook.md) covers the rest.

---

Everything below the contract — which model, which vector store, which compute —
is a tradeoff, made per cloud in [architecture.md](architecture.md), priced in
[cost-estimate.md](cost-estimate.md), and defended in
[threat-model.md](threat-model.md).
