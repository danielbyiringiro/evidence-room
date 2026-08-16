# Evidence Room -- AWS (portable design)
# Skeleton IaC for the contract in docs/cloud/. NOT apply-tested (no creds here);
# review before use. Portable retrieval runs in the container, so there is no
# managed vector store -- see docs/cloud/architecture.md for the tradeoff.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # backend "s3" { ... }   # configure remote state per environment
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      app       = "evidence-room"
      managed   = "terraform"
      component = "grading-agent"
    }
  }
}

data "aws_caller_identity" "me" {}

# --- encryption -------------------------------------------------------------
resource "aws_kms_key" "data" {
  description             = "${var.name} data key (index, audit, secrets)"
  enable_key_rotation     = true
  deletion_window_in_days = 14
}

# --- storage: embedding index (read), audit (append-only/WORM) --------------
resource "aws_s3_bucket" "index" {
  bucket = "${var.name}-index-${data.aws_caller_identity.me.account_id}"
}

resource "aws_s3_bucket_versioning" "index" {
  bucket = aws_s3_bucket.index.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "index" {
  bucket = aws_s3_bucket.index.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Audit bucket with Object Lock (WORM) -- runtime can write but never delete.
resource "aws_s3_bucket" "audit" {
  bucket              = "${var.name}-audit-${data.aws_caller_identity.me.account_id}"
  object_lock_enabled = true
}

resource "aws_s3_bucket_object_lock_configuration" "audit" {
  bucket = aws_s3_bucket.audit.id
  rule {
    default_retention {
      mode = "COMPLIANCE"
      days = 3650
    }
  }
}

resource "aws_s3_bucket_public_access_block" "audit" {
  bucket                  = aws_s3_bucket.audit.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- proposals store --------------------------------------------------------
resource "aws_dynamodb_table" "proposals" {
  name         = "${var.name}-proposals"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "proposal_id"
  attribute {
    name = "proposal_id"
    type = "S"
  }
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.data.arn
  }
  point_in_time_recovery { enabled = true }
}

# --- secrets ----------------------------------------------------------------
resource "aws_secretsmanager_secret" "app" {
  name       = "${var.name}/config"
  kms_key_id = aws_kms_key.data.arn
}

# --- runtime role: LEAST PRIVILEGE -----------------------------------------
resource "aws_iam_role" "runtime" {
  name = "${var.name}-runtime"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

data "aws_iam_policy_document" "runtime" {
  # read the embedding index (never write)
  statement {
    sid       = "IndexRead"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.index.arn}/*"]
  }
  # append-only audit: PutObject, but explicitly NOT Delete/Overwrite
  statement {
    sid       = "AuditAppend"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.audit.arn}/*"]
  }
  # proposals read/write
  statement {
    sid       = "Proposals"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"]
    resources = [aws_dynamodb_table.proposals.arn]
  }
  statement {
    sid       = "Secret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.app.arn]
  }
  statement {
    sid       = "Kms"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "cloudwatch:PutMetricData"]
    resources = ["*"]
  }
  # optional: model invoke, scoped to one model, only when drafting enabled
  dynamic "statement" {
    for_each = var.enable_llm_drafting ? [1] : []
    content {
      sid       = "BedrockInvoke"
      actions   = ["bedrock:InvokeModel"]
      resources = [var.bedrock_model_arn]
    }
  }
}

resource "aws_iam_role_policy" "runtime" {
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

# --- compute: Lambda container (portable retriever inside) ------------------
resource "aws_lambda_function" "api" {
  function_name = "${var.name}-api"
  role          = aws_iam_role.runtime.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["arm64"]
  timeout       = 30
  memory_size   = 1024
  environment {
    variables = {
      EVIDENCE_ROOM_INDEX_URI = "s3://${aws_s3_bucket.index.id}/index.npz"
      DRAFT_ENGINE            = var.enable_llm_drafting ? "bedrock" : "deterministic"
      PROPOSALS_TABLE         = aws_dynamodb_table.proposals.name
      AUDIT_BUCKET            = aws_s3_bucket.audit.id
    }
  }
}

# --- observability: leakage alarm (the hard invariant) ----------------------
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.name}-api"
  retention_in_days = 90
}

resource "aws_cloudwatch_metric_alarm" "leakage" {
  alarm_name          = "${var.name}-scope-leakage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ScopeLeakage" # emitted by the app on any out-of-scope citation
  namespace           = "EvidenceRoom"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "PAGE: scope/permission leakage detected (must be 0)"
  treat_missing_data  = "notBreaching"
}

# --- cost guardrail ---------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = length(var.budget_alert_emails) > 0 ? [80, 100] : []
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = var.budget_alert_emails
    }
  }
}

output "index_bucket" { value = aws_s3_bucket.index.id }
output "audit_bucket" { value = aws_s3_bucket.audit.id }
output "api_function" { value = aws_lambda_function.api.function_name }
