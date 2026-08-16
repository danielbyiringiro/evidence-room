# Cost estimate: 100 / 1,000 / 10,000 weekly tasks

A "task" is one `/draft` call (embed the query → retrieve → compose a suggested
deduction). Weekly volumes convert to monthly at ×4.33: **433 / 4,330 / 43,300**
tasks per month.

> Estimates in USD, us-east-1 / us-central1, list price. Figures are
> rounded and meant for **relative** decisions.

## Rates used

| Item | Rate |
|---|---|
| Bedrock Claude Haiku 4.5 | $1 / 1M input, $5 / 1M output tokens |
| Vertex AI Claude Haiku | ~parity with Anthropic list ($1 / $5) |
| Titan Text Embeddings V2 / Vertex `text-embedding` | $0.02 / 1M tokens |
| AWS Lambda (Arm) | $0.20 / 1M req + ~$0.0000133/GB-s; free 1M req + 400k GB-s/mo |
| Cloud Run | $0.000024/vCPU-s + $0.0000025/GiB-s + $0.40/1M req; free 180k vCPU-s + 360k GiB-s + 2M req/mo |
| ECS Fargate (small, warm) | ~$0.04048/vCPU-hr + $0.004445/GB-hr |
| OpenSearch Serverless | ~$0.24/OCU-hr; Classic min ~2 OCU; NextGen scales to zero |
| Vertex AI Vector Search | node-hour; ~$250/mo single small node, ~$750/mo 3-replica HA |
| S3 / GCS standard | ~$0.02–0.023/GB-mo |
| CloudWatch / Cloud Logging | ~$0.50/GB ingest (Cloud Logging first 50 GiB free) |
| Secrets Manager / Secret Manager | ~$0.40/secret/mo / ~$0.06/version/mo |

Per-task token footprint: query embedding ~200 tokens (≈ $0.000004, negligible);
optional Claude Haiku draft ≈ 2,000 input + 300 output tokens ≈ **$0.0035/task**.

## Scenario A — Portable, deterministic draft (recommended default)

Retrieval runs in-container (fastembed + numpy + BM25); no managed vector store,
no LLM. Serverless compute scales to ~zero; the bill is small fixed services.

| Component | 100/wk | 1,000/wk | 10,000/wk |
|---|---|---|---|
| Compute (Lambda / Cloud Run, scale-to-zero) | ~$0 (free tier) | ~$0 | ~$1 |
| Object storage (index + chunks, <1 GB) | ~$0.02 | ~$0.02 | ~$0.05 |
| Proposals + audit (DynamoDB/Firestore) | ~$0.05 | ~$0.10 | ~$0.50 |
| Secrets + KMS | ~$1.50 | ~$1.50 | ~$1.50 |
| Logging/metrics | ~$0.20 | ~$0.50 | ~$2 |
| **Total** | **~$2/mo** | **~$3/mo** | **~$5/mo** |

Essentially **flat** across the range — at these volumes the workload is trivial;
you are paying for a few fixed services, not for throughput. Add a warm instance
(ECS Fargate small / Cloud Run `min-instances=1`) if p95 cold-start latency
matters: **+$15–45/mo flat**.

## Scenario B — Portable + Claude Haiku draft rationale

Same as A, but each task also calls Claude Haiku for a written rationale (only if
the institution's zero-retention terms permit proof text to reach the model).

| Component | 100/wk | 1,000/wk | 10,000/wk |
|---|---|---|---|
| Scenario A base | ~$2 | ~$3 | ~$5 |
| Claude Haiku drafting (@ $0.0035/task) | ~$1.50 | ~$15 | ~$152 |
| **Total** | **~$4/mo** | **~$18/mo** | **~$157/mo** |

This is the one line that scales with volume. It is why deterministic drafting is
the default: the LLM buys fluency, not correctness, and correctness is already
carried by the retrieved precedent + rubric guidance.

## Scenario C — Managed vector store, deterministic draft

Bedrock Knowledge Bases + OpenSearch Serverless (AWS) / Vertex AI Vector Search
(GCP). The managed ANN index is always-on, so cost is a **floor** that barely
moves with task volume.

| Component | 100/wk | 1,000/wk | 10,000/wk |
|---|---|---|---|
| AWS OpenSearch Serverless (Classic ~2 OCU) | ~$350 | ~$350 | ~$355 |
| AWS OpenSearch Serverless (NextGen, scale-to-zero) | ~$40 | ~$60 | ~$110 |
| GCP Vertex Vector Search (single node) | ~$250 | ~$250 | ~$255 |
| GCP Vertex Vector Search (3-replica HA) | ~$750 | ~$750 | ~$760 |
| + compute + misc (as Scenario A) | ~$2 | ~$3 | ~$6 |

## Takeaways

- **At ≤10k weekly tasks, volume is not the cost driver — the vector tier is.**
  Portable is ~$2–5/mo; managed adds a $40–750/mo always-on floor for a 25k-chunk
  index that `numpy` searches in single-digit milliseconds anyway. This is the
  central production tradeoff the lab asks for, in one number.
- **The portable layer is ~50–150× cheaper here** and keeps proof text in-VPC.
  Managed's value (managed ANN at millions of vectors, high QPS, HA) does not
  bite until far above this regime.
- **The only volume-scaling cost is optional LLM drafting** (~$0.0035/task). Cap
  it with the refusal path (refused tasks skip the model) and a Budgets alarm
  ([runbook](runbook.md)).
- **Cross-cloud parity:** because compute is the same container, AWS and GCP land
  within a few dollars of each other in the portable design; the divergence only
  appears if you adopt each cloud's managed vector store, whose floors differ
  (OpenSearch NextGen vs Vertex Vector Search).

## Sources

- [Amazon Bedrock pricing (Claude, Titan embeddings)](https://aws.amazon.com/bedrock/pricing/) · [CloudZero: Bedrock pricing 2026](https://www.cloudzero.com/blog/amazon-bedrock-pricing/)
- [Amazon OpenSearch Serverless pricing](https://aws.amazon.com/opensearch-service/pricing/) · [Lucidity: OpenSearch pricing 2026](https://www.lucidity.cloud/blog/aws-opensearch-pricing)
- [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/) · [CloudZero: Lambda pricing 2026](https://www.cloudzero.com/blog/lambda-pricing/)
- [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) · [CloudZero: Vertex AI pricing 2026](https://www.cloudzero.com/blog/google-vertex-ai-pricing/)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
