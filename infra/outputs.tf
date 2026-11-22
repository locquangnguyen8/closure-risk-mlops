output "service_url" {
  description = "Public HTTPS URL of the prediction API."
  value       = "https://${aws_apprunner_service.api.service_url}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "mlflow_artifact_bucket" {
  value = aws_s3_bucket.mlflow.bucket
}
