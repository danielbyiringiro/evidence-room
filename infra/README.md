# Infrastructure

Terraform skeletons that provision the [cloud contract](../docs/cloud/contract.md)
on AWS (`aws/`) and GCP (`gcp/`) in the **portable** design
([architecture](../docs/cloud/architecture.md)): the `evidence_room` container
carries retrieval; the cloud supplies compute, encrypted storage, secrets,
identity, observability, and a cost guardrail.

> Both stacks intentionally mirror each other so the single contract is visible > in the IaC itself.

## What each stack creates

| | AWS (`aws/`) | GCP (`gcp/`) |
|---|---|---|
| Compute | Lambda container (Arm) | Cloud Run (scale-to-zero) |
| Index storage (read) | S3 + SSE-KMS + versioning | GCS + CMEK + versioning |
| Audit (append-only) | S3 Object Lock (COMPLIANCE) | GCS locked retention + custom append role |
| Proposals | DynamoDB (PITR) | Firestore native |
| Secrets | Secrets Manager | Secret Manager |
| Encryption | KMS key (rotating) | KMS crypto key (rotating) |
| Runtime identity | IAM role, least privilege | Service account, least privilege |
| Leakage alarm | CloudWatch metric alarm | Cloud Logging metric |
| Cost guardrail | AWS Budgets (80/100%) | Cloud Billing budget (80/100%) |

The runtime principal in both can **write but not delete** the audit log,
**read but not write** the index, and reach a model **only** when
`enable_llm_drafting = true` - the least-privilege posture from the
[threat model](../docs/cloud/threat-model.md#iam-matrix).

## Reproducible setup

Prerequisites: Terraform ≥ 1.5, cloud CLI authenticated, a built container image
pushed to ECR (AWS) or Artifact Registry (GCP). Build the image from the repo
root (a thin FastAPI/Lambda handler over `evidence_room` - not included here).

### AWS

```bash
cd infra/aws
terraform init
terraform apply \
  -var="image_uri=<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/evidence-room:latest" \
  -var='budget_alert_emails=["you@ashesi.edu"]'
# upload the embedding index the runtime reads:
aws s3 cp src/embeddings/index.npz "s3://$(terraform output -raw index_bucket)/index.npz"
```

### GCP

```bash
cd infra/gcp
terraform init
terraform apply \
  -var="project_id=<PROJECT>" \
  -var="image=us-central1-docker.pkg.dev/<PROJECT>/evidence-room/api:latest" \
  -var="billing_account=billingAccounts/XXXXXX-XXXXXX-XXXXXX"
gsutil cp src/embeddings/index.npz "gs://$(terraform output -raw index_bucket)/index.npz"
```

### Enabling optional LLM drafting

Off by default (the deterministic composer needs no model). To turn on the
Bedrock/Vertex Claude Haiku rationale - only under an institution-approved
zero-retention agreement, since it transmits proof text:

```bash
# AWS
terraform apply -var="enable_llm_drafting=true" \
  -var="bedrock_model_arn=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5"
# GCP
terraform apply -var="enable_llm_drafting=true"
```

## Post-deploy gate

Treat the eval as a release gate on both clouds
([contract](../docs/cloud/contract.md#2-evaluation-set)):

```bash
python -m evidence_room.evaluate     # leakage must be 0; refusal accuracy at baseline
```

A deploy whose eval shows leakage > 0 or a refusal-accuracy regression must not
go live.
