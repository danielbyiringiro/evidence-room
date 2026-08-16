# Evidence Room -- GCP (portable design)
# Skeleton IaC for the contract in docs/cloud/. Portable retrieval runs in the container -- no managed vector
# store. Mirror of infra/aws: same boxes, GCP labels (docs/cloud/architecture.md).

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  labels = { app = "evidence-room", managed = "terraform", component = "grading-agent" }
}

# --- encryption -------------------------------------------------------------
resource "google_kms_key_ring" "kr" {
  name     = "${var.name}-kr"
  location = var.region
}

resource "google_kms_crypto_key" "data" {
  name            = "${var.name}-data"
  key_ring        = google_kms_key_ring.kr.id
  rotation_period = "7776000s" # 90 days
}

# --- storage: index (read, CMEK, versioned), audit (retention lock) ---------
resource "google_storage_bucket" "index" {
  name                        = "${var.name}-index-${var.project_id}"
  location                    = var.region
  uniform_bucket_level_access = true
  versioning { enabled = true }
  encryption { default_kms_key_name = google_kms_crypto_key.data.id }
  labels = local.labels
}

resource "google_storage_bucket" "audit" {
  name                        = "${var.name}-audit-${var.project_id}"
  location                    = var.region
  uniform_bucket_level_access = true
  retention_policy {
    retention_period = 315360000 # 10 years, append-only via IAM below
    is_locked        = true
  }
  encryption { default_kms_key_name = google_kms_crypto_key.data.id }
  labels = local.labels
}

# --- proposals store (Firestore native) -------------------------------------
resource "google_firestore_database" "proposals" {
  name        = "${var.name}-proposals"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
}

# --- secrets ----------------------------------------------------------------
resource "google_secret_manager_secret" "app" {
  secret_id = "${var.name}-config"
  replication { auto {} }
  labels = local.labels
}

# --- runtime service account: LEAST PRIVILEGE -------------------------------
resource "google_service_account" "runtime" {
  account_id   = "${var.name}-runtime"
  display_name = "Evidence Room runtime"
}

# read the index (never write)
resource "google_storage_bucket_iam_member" "index_read" {
  bucket = google_storage_bucket.index.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

# append-only audit: a custom role with create but NOT delete/overwrite
resource "google_project_iam_custom_role" "audit_append" {
  role_id     = "evidenceRoomAuditAppend"
  title       = "Evidence Room audit append"
  permissions = ["storage.objects.create", "storage.objects.get"]
}

resource "google_storage_bucket_iam_member" "audit_append" {
  bucket = google_storage_bucket.audit.name
  role   = google_project_iam_custom_role.audit_append.id
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "proposals" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "secret" {
  secret_id = google_secret_manager_secret.app.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "obs" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.runtime.email}"
}

# optional: Vertex AI access only when drafting is enabled
resource "google_project_iam_member" "vertex" {
  count   = var.enable_llm_drafting ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# --- compute: Cloud Run (portable retriever inside) -------------------------
resource "google_cloud_run_v2_service" "api" {
  name     = "${var.name}-api"
  location = var.region
  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = 0 # scale to zero; set 1 if warm latency matters
      max_instance_count = 10
    }
    containers {
      image = var.image
      resources {
        limits = { cpu = "1", memory = "1Gi" }
      }
      env {
        name  = "EVIDENCE_ROOM_INDEX_URI"
        value = "gs://${google_storage_bucket.index.name}/index.npz"
      }
      env {
        name  = "DRAFT_ENGINE"
        value = var.enable_llm_drafting ? "vertex" : "deterministic"
      }
    }
  }
}

# --- observability: leakage alert (hard invariant) --------------------------
resource "google_logging_metric" "leakage" {
  name   = "${var.name}-scope-leakage"
  filter = "resource.type=\"cloud_run_revision\" jsonPayload.event=\"scope_leakage\""
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
  }
}

# --- cost guardrail ---------------------------------------------------------
resource "google_billing_budget" "monthly" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "${var.name}-monthly"
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }
  threshold_rules { threshold_percent = 0.8 }
  threshold_rules { threshold_percent = 1.0 }
}

output "index_bucket" { value = google_storage_bucket.index.name }
output "audit_bucket" { value = google_storage_bucket.audit.name }
output "run_service"  { value = google_cloud_run_v2_service.api.uri }
