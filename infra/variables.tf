variable "name" {
  description = "Name prefix for all resources."
  type        = string
  default     = "closure-risk"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "ap-southeast-2"
}

variable "image_tag" {
  description = "Tag of the image in ECR to deploy (immutable tags, e.g. the git SHA)."
  type        = string
}

variable "max_instances" {
  description = "Upper bound for App Runner autoscaling."
  type        = number
  default     = 3
}
