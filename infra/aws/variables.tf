variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "evidence-room"
}

variable "image_uri" {
  description = "ECR image URI for the evidence_room container (retrieve/draft/apply API)"
  type        = string
}

variable "enable_llm_drafting" {
  description = "Grant bedrock:InvokeModel for optional Claude Haiku draft rationale"
  type        = bool
  default     = false
}

variable "bedrock_model_arn" {
  description = "ARN of the Claude Haiku model (only used if enable_llm_drafting)"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Cost guardrail; alerts fire at 80% and 100%"
  type        = number
  default     = 50
}

variable "budget_alert_emails" {
  type    = list(string)
  default = []
}
