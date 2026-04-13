# Deployment

## Prerequisites

1. AWS CLI configured (`aws configure`) with a profile that has SageMaker + S3 + ECR permissions.
2. A SageMaker execution role with at least:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or a policy scoped to your bucket)
3. An S3 bucket in `eu-west-1`.

## SageMaker deployment

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/<role-name>"
export AWS_DEFAULT_REGION="eu-west-1"

uv run python deploy/deploy_sagemaker.py \
  --bucket <your-bucket> \
  --region eu-west-1 \
  --endpoint-name vaultech-bath-predictor \
  --model-package-group vaultech-bath-predictor-group
```

## Resource names

| Resource            | Name                                 |
|---------------------|--------------------------------------|
| S3 bucket           | (set via --bucket)                   |
| Model Package Group | vaultech-bath-predictor-group        |
| Endpoint name       | vaultech-bath-predictor              |
| AWS region          | eu-west-1                            |

## Validate

```bash
export SAGEMAKER_MODEL_PACKAGE_GROUP="vaultech-bath-predictor-group"
export SAGEMAKER_ENDPOINT_NAME="vaultech-bath-predictor"
export AWS_DEFAULT_REGION="eu-west-1"
uv run pytest tests/test_sagemaker.py -v
```
