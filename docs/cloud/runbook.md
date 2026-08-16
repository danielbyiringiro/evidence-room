# Incident runbook

Four incident classes the lab requires, plus the one hard-invariant page. Each
entry: **detect → contain → diagnose → recover → prevent**. Signal names map to
the observability contract ([contract.md](contract.md#4-observability)); product
names are CloudWatch (AWS) / Cloud Monitoring (GCP).

On-call reaches the agent through its own controls: the system is designed to
**fail closed** (refuse / escalate), so most incidents degrade to "refuses more
than usual," not "grades wrongly."

---

## 0. PAGE — scope/permission leakage (hard invariant)

- **Detect:** `leakage_counter > 0` (any out-of-scope or cross-role citation).
  This should be impossible; if it fires, treat as a security incident.
- **Contain:** flip the `RETRIEVAL_ENABLED=false` flag (returns refusal for all
  `/draft`) — the agent keeps working as a safe no-op. Revoke the runtime role's
  data-store read if compromise is suspected.
- **Diagnose:** pull the offending trace (it names the leaked chunk id, scope,
  role). Re-run `evidence_room.evaluate` against the deployed index — the eval's
  leakage metric must be 0; a non-zero means the deployed filter diverged from the
  tested code (config drift, wrong index, tampered data).
- **Recover:** redeploy the known-good image + index; confirm eval leakage = 0.
- **Prevent:** leakage is a release gate; block deploys whose eval shows leakage
  > 0. Index writes restricted to the CI principal only.

---

## 1. Bad retrieval (irrelevant / low-quality precedent)

- **Detect:** refusal-rate spike or drop; `label precision@k` regression in the
  scheduled eval; graders flagging weak suggestions.
- **Contain:** raise `EVIDENCE_ROOM_MIN_SIM` a notch — the agent refuses the
  borderline cases instead of drafting weak ones (fails closed). Grading
  continues manually for refused items.
- **Diagnose:** likely causes — (a) index/model mismatch (index built with a
  different embedding model than the one serving queries; the index records its
  model, compare it), (b) stale index after a corpus change, (c) a bad
  re-embedding job.
- **Recover:** rebuild embeddings with the serving model, re-run the eval,
  re-calibrate `EVIDENCE_ROOM_MIN_SIM` from the sweep, redeploy.
- **Prevent:** pin the embedding model id in the index metadata; run the eval as a
  pre-deploy gate; alert on precision@k regression.

## 2. Model outage (Bedrock / Vertex unavailable)

*Only relevant when optional LLM drafting is enabled; the default deterministic
composer has no model dependency and is unaffected.*

- **Detect:** model invoke error rate / latency alarm; `tool_error` audit entries
  with `type: timeout|transient` on the draft tool.
- **Contain:** the `_invoke_tool` wrapper already times out + retries with backoff,
  then returns a structured error → the agent escalates, never hangs. Flip
  `DRAFT_ENGINE=deterministic` to fall back to the template composer with **zero
  downtime** (the whole reason it is the default).
- **Diagnose:** check the provider status page and the region; confirm it is the
  model endpoint, not IAM (a revoked `bedrock:InvokeModel` / `aiplatform.user`
  looks similar — the error `type` distinguishes them).
- **Recover:** restore the model call once the provider clears; or stay on
  deterministic drafting (no correctness loss, only fluency).
- **Prevent:** deterministic default; multi-region model endpoint or a
  second-provider fallback if LLM drafting becomes load-bearing.

## 3. Tool failure (retrieval/store errors, timeouts)

- **Detect:** `tool_error` rate alarm; elevated `/draft` 5xx; retry-exhaustion
  logs.
- **Contain:** the agent returns a structured `tool_error` and escalates — **no
  half-written proposal**, and idempotency means a client retry cannot double-act.
  If a specific store is down (proposals/audit), `/retrieve` still serves.
- **Diagnose:** identify the failing dependency from `tool_error.tool` +
  `.type` (timeout vs transient vs a non-retryable exception). Common: object
  store throttling, DynamoDB/Firestore hot partition, VPC/networking.
- **Recover:** restore the dependency; the append-only audit lets you replay what
  was attempted. Re-drive any escalated tasks (idempotent — no duplicates).
- **Prevent:** tune timeout/retry budget; provision store throughput; keep the
  audit write path independent of the proposal write path so one failing does not
  block the other.

## 4. Runaway cost

- **Detect:** AWS Budgets / Cloud Billing budget alert; cost-anomaly detection;
  `cost-per-task` metric breach.
- **Contain:** the usual culprit is LLM drafting under a request flood or a retry
  storm. Immediate: throttle `/draft` (API Gateway usage plan / Cloud Run max
  instances), and flip `DRAFT_ENGINE=deterministic` to zero out per-task model
  spend. Idempotency + refusal already suppress duplicate and unanswerable calls.
- **Diagnose:** attribute spend via cost allocation tags / labels
  (`app=evidence-room`, `component=…`). Distinguish (a) legitimate volume, (b)
  abuse/DoS, (c) a retry loop, (d) an always-on resource left running (a managed
  vector store accidentally provisioned — the Scenario C floor).
- **Recover:** scale limits, kill orphaned resources, re-enable drafting once the
  cause is fixed.
- **Prevent:** hard concurrency caps; per-caller rate limits; budget alarms wired
  before launch; prefer the portable/deterministic default whose cost is flat and
  bounded ([cost-estimate.md](cost-estimate.md)); tag everything for allocation.

---

### Fail-closed summary

| Incident | Degrades to | Correctness impact |
|---|---|---|
| Leakage | safe no-op (refuse all) | none (stops) |
| Bad retrieval | more refusals | none (humans grade refused items) |
| Model outage | deterministic drafts | none (fluency only) |
| Tool failure | escalate, no partial write | none |
| Runaway cost | throttle + deterministic | none |

Every failure path lands on "the human does the grading," never on "the agent
grades wrongly and silently." That is the safety property the design buys.
