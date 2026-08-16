variable "project_id" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "evidence-room"
}

variable "image" {
  description = "Artifact Registry image for the evidence_room container"
  type        = string
}

variable "enable_llm_drafting" {
  description = "Grant Vertex AI access for optional Claude Haiku draft rationale"
  type        = bool
  default     = false
}

variable "monthly_budget_usd" {
  description = "Cost guardrail; alerts fire at 80% and 100%"
  type        = number
  default     = 50
}

variable "billing_account" {
  description = "Billing account id for the budget (billingAccounts/XXXX)"
  type        = string
  default     = ""
}
