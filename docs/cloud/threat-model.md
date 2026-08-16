# Threat model & IAM matrix

Scope: the deployed Evidence Room agent (the [contract](contract.md)) on AWS and
GCP. The trust boundary is the VPC/project; the asset under protection is
**consented student coursework** and the **integrity of grading judgments**.

## Trust boundaries & data classification

| Data | Class | Rule |
|---|---|---|
| Student proof text | Sensitive (consented) | never leaves the VPC/project; stored hashed in audit |
| Graded exemplars (proof + expert label + grader id) | Sensitive | scoped per question/item/role; never cross-scope |
| Rubric guidance | Internal | shared across scopes; no student text |
| Embedding index | Sensitive (derived) | encrypted at rest (SSE-KMS / CMEK); private buckets |
| Proposals | Internal | no verbatim proof text; role-attributed |
| Audit log | Internal, immutable | append-only (Object Lock / retention); hashed args |

## STRIDE

Each threat is paired with the control that already exists in the app plus the
cloud control that backs it.

| # | Threat (STRIDE) | Scenario | App control | Cloud control |
|---|---|---|---|---|
| S | **Spoofing** | Caller claims `faculty` to read exemplars | role comes from the verified identity token, never request body; `ACTORS` gate | Cognito / Identity Platform JWT authorizer; API Gateway/IAP rejects unauthenticated |
| S | Spoofing (injection) | Proof text says "I am faculty" | role never inferred from content (tested) | — |
| T | **Tampering** | Alter a proposal or audit entry after the fact | proposals last-write-wins with idempotency key; audit hashed | S3 Object Lock (WORM) / GCS retention lock; DynamoDB/Firestore IAM write-scoped |
| T | Tampering (index) | Poison the embedding index | index built from access-controlled corpus; eval leakage gate | bucket write restricted to CI principal; object versioning |
| R | **Repudiation** | "I never approved that deduction" | `apply()` records approver + timestamp | immutable audit log; CloudTrail / Cloud Audit Logs on control plane |
| I | **Information disclosure** | Cross-question / cross-role exemplar leak | pre-retrieval scope + permission filter; **leakage=0 eval gate** | KMS/CMEK at rest; TLS in transit; VPC-private data stores |
| I | Disclosure (exfil) | Injected "dump grader names" | deterministic composer omits grader ids; citations are ids only | egress restricted; no public data endpoints |
| I | Disclosure (residency) | Proof text sent to a managed embedder | **portable in-VPC embeddings** (default) | if managed used: provider zero-retention DPA required |
| D | **Denial of service** | Flood `/draft` to burn model/infra budget | idempotency dedups; refusal short-circuits cheap | API Gateway/Cloud Run throttling + WAF; Budgets alarm ([runbook](runbook.md)) |
| D | DoS (cost) | Runaway retries | bounded retry + structured error | per-call timeout; concurrency cap; anomaly detection |
| E | **Elevation of privilege** | Trainee triggers an apply | authorization boundary blocks non-faculty before any tool runs | least-privilege IAM (below); approver is a separate principal |
| E | EoP (compromised runtime) | App creds used to delete audit | runtime cannot delete audit (append-only) | IAM denies `Delete*` on audit store to the runtime role |

The two hard invariants — **leakage = 0** and **no auto-apply without approval**
— are enforced in code *and* re-checked every release by the eval gate, so a
cloud misconfiguration cannot silently relax them without the eval going red.

## IAM matrix (least privilege)

Principals get only the actions they need on only the resources they touch.
"—" means explicitly no access.

| Principal | Embedding index (obj store) | Proposals | Audit log | Secrets | Model (Bedrock/Vertex) | Logs/Metrics |
|---|---|---|---|---|---|---|
| **App runtime** (Lambda role / Cloud Run SA) | read | read + write | **append only** (no delete/overwrite) | read (one secret) | invoke (optional, if enabled) | put |
| **Approver** (human, faculty lead) | — | read + status-update via `/apply` | read | — | — | read |
| **Grader** (faculty) | — | read own (via API) | — | — | — | — |
| **Trainee** | — | — | — | — | — | — |
| **CI / deploy** | write (index refresh) | — | — | manage | — | — |
| **Analyst / auditor** | — | read | read | — | — | read |

### AWS bindings

| Principal | Policy sketch |
|---|---|
| App runtime (Lambda exec role) | `s3:GetObject` on `index/*`; `dynamodb:GetItem/PutItem/Query` on `proposals`; `s3:PutObject` on `audit/*` (bucket has Object Lock, no `s3:DeleteObject`); `secretsmanager:GetSecretValue` on one ARN; `kms:Decrypt` on the data key; `bedrock:InvokeModel` on the Haiku model ARN *only if drafting enabled*; `logs:PutLogEvents`, `cloudwatch:PutMetricData` |
| CI / deploy | `s3:PutObject` on `index/*`; Terraform state + resource management; no data-plane read of audit |
| Approver | API-only (Cognito group `faculty-lead`); no direct AWS data access |

### GCP bindings

| Principal | Roles sketch |
|---|---|
| App runtime (Cloud Run SA) | `roles/storage.objectViewer` on the index bucket; `roles/datastore.user` on `proposals`; custom role `audit.append` (`storage.objects.create` only) on the audit bucket (retention lock, no `storage.objects.delete`); `roles/secretmanager.secretAccessor` on one secret; `roles/aiplatform.user` scoped to the Claude model *only if drafting enabled*; `roles/logging.logWriter`, `roles/monitoring.metricWriter` |
| CI / deploy | `roles/storage.objectAdmin` on the index bucket; Terraform SA; no read on audit |
| Approver | IAP-gated `faculty-lead` group; no direct project data access |

Principle applied throughout: the runtime can **write but never delete** the
audit log, can **read but never write** the index, and can reach the model **only
when drafting is explicitly enabled** — so the blast radius of a compromised
runtime is bounded to what grading actually requires.
