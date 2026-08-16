# Two clouds, one contract

Days 3-4 build lab: deploy the Evidence Room agent on **AWS and GCP** behind a
single application contract. The model and managed services differ; the
interface, evaluation set, security boundary, and observability requirements do
not.

The thesis: cloud fluency is making sound production tradeoffs, not memorising
services. The sharp decision here is **managed vs portable retrieval**, and the
evidence points to portable - for data residency, cost, and preserving the
type-aware chunking and access-scoping that are the whole product.

## Deliverables

| Deliverable | Where |
|---|---|
| The application contract (interface, eval, security, observability) | [contract.md](contract.md) |
| Architecture diagrams - AWS + GCP (Mermaid) + managed-vs-portable decision | [architecture.md](architecture.md) |
| Infrastructure-as-code (Terraform, both clouds) + reproducible setup | [`../../infra/`](../../infra/) |
| Threat model (STRIDE) + IAM matrix, least privilege | [threat-model.md](threat-model.md) |
| Cost estimate - 100 / 1,000 / 10,000 weekly tasks | [cost-estimate.md](cost-estimate.md) |
| Incident runbook - bad retrieval, model outage, tool failure, runaway cost | [runbook.md](runbook.md) |

## The one-paragraph version

Both clouds run the same container (`evidence_room` behind `/retrieve`, `/draft`,
`/apply`). Retrieval stays **portable** - fastembed + numpy + BM25 in-VPC - so
consented student proof text never leaves the trust boundary and the always-on
managed-vector-store floor ($40–750/mo) is avoided; at ≤10k weekly tasks the
portable stack is ~$2–5/mo and the workload is trivial in token terms. Security
is enforced in code and re-checked every release by the eval's **leakage = 0**
gate, so a cloud misconfiguration cannot silently relax it. Every failure path
degrades to "a human grades it," never "the agent grades wrongly." Managed
services remain a documented swap behind the same interface for when the corpus
reaches millions of vectors - the portable layer is the seam that keeps the cloud
choice reversible.
