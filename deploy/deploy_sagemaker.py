"""
SageMaker deployment script — packages, registers, and deploys the XGBoost model.

Usage:
    export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/<role-name>"
    uv run python deploy/deploy_sagemaker.py \
      --bucket your-bucket-name \
      --region eu-west-1 \
      --endpoint-name your-endpoint-name \
      --model-package-group your-group-name
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SERVING_DIR = Path(__file__).resolve().parent.parent / "serving"
ECR_REPO_NAME = "vaultech-xgboost-serving"
MODEL_FILE = MODEL_DIR / "xgboost_bath_predictor.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

def build_and_push_serving_image(region: str) -> str:
    """Build the custom XGBoost serving container and push it to ECR.

    Returns:
        Full ECR image URI.
    """
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    ecr = boto3.client("ecr", region_name=region)
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:latest"

    # Create ECR repository (idempotent)
    try:
        ecr.create_repository(repositoryName=ECR_REPO_NAME)
        print(f"  Created ECR repository: {ECR_REPO_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            print(f"  ECR repository already exists: {ECR_REPO_NAME}")
        else:
            raise

    # Authenticate Docker to ECR
    token = ecr.get_authorization_token()["authorizationData"][0]
    import base64
    username, password = base64.b64decode(token["authorizationToken"]).decode().split(":")
    registry = token["proxyEndpoint"]
    subprocess.run(
        ["docker", "login", "--username", username, "--password", password, registry],
        check=True, capture_output=True,
    )
    print(f"  Authenticated to ECR: {registry}")

    # Build for linux/amd64 (SageMaker requirement) and push directly to ECR.
    # --provenance=false prevents Docker BuildKit from wrapping the image in an
    # OCI image-index manifest, which SageMaker does not support.
    print(f"  Building and pushing serving image for linux/amd64 (this may take ~3 min)...")
    subprocess.run(
        [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "--provenance=false",
            "--push",
            "-t", image_uri,
            ".",
        ],
        check=True, cwd=str(SERVING_DIR),
    )
    print(f"  Pushed: {image_uri}")

    return image_uri


def package_model(model_path: Path, output_dir: Path) -> Path:
    """Package the XGBoost model as a .tar.gz archive for SageMaker.

    SageMaker's built-in XGBoost container expects a file named
    'xgboost-model' at the root of the archive.

    Args:
        model_path: Path to the trained model JSON file.
        output_dir: Directory where the .tar.gz will be created.

    Returns:
        Path to the created .tar.gz file.
    """
    output_path = Path(output_dir) / "model.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_model = Path(tmp) / "xgboost-model"
        shutil.copy(model_path, tmp_model)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmp_model, arcname="xgboost-model")
    return output_path


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Full S3 URI (s3://bucket/key).
    """
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def register_model(
    s3_model_uri: str,
    model_package_group_name: str,
    region: str,
    metrics: dict,
) -> str:
    """Register the model in SageMaker Model Registry.

    Creates the Model Package Group if it doesn't exist, then registers
    a new Model Package version with the XGBoost container image,
    the S3 model artifact, and evaluation metrics.

    Args:
        s3_model_uri: S3 URI of the packaged model (.tar.gz).
        model_package_group_name: Name for the Model Package Group.
        region: AWS region.
        metrics: Dict with 'rmse', 'mae', 'r2' keys.

    Returns:
        The Model Package ARN.
    """
    sm = boto3.client("sagemaker", region_name=region)

    # Create the group (idempotent — ignore if it already exists)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription=(
                "XGBoost bath-time predictor for VaultTech forging line"
            ),
        )
        print(f"  Created Model Package Group: {model_package_group_name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code in ("ValidationException", "ConflictException") or "already exist" in msg:
            print(f"  Model Package Group already exists: {model_package_group_name}")
        else:
            raise

    # Register with metadata only — no InferenceSpecification to avoid ECR
    # image validation issues in sandbox accounts. The S3 URI is stored in
    # CustomerMetadataProperties so deploy_endpoint can retrieve it later.
    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=(
            "XGBoost model predicting piece bath time from 2nd-strike time and OEE. "
            f"RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}"
        ),
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties={
            "rmse": str(metrics["rmse"]),
            "mae": str(metrics["mae"]),
            "r2": str(metrics["r2"]),
            "s3_model_uri": s3_model_uri,
        },
    )

    return response["ModelPackageArn"]


def deploy_endpoint(
    model_package_arn: str,
    endpoint_name: str,
    region: str,
    instance_type: str = "ml.t2.medium",
    s3_model_uri: str | None = None,
) -> str:
    """Deploy a real-time SageMaker endpoint from a registered Model Package.

    Creates a SageMaker Model, Endpoint Configuration, and Endpoint.
    Waits until the endpoint status is 'InService'.

    Args:
        model_package_arn: ARN of the registered Model Package.
        endpoint_name: Name for the endpoint.
        region: AWS region.
        instance_type: EC2 instance type for the endpoint.
        s3_model_uri: S3 URI of the model artifact (overrides metadata lookup).

    Returns:
        The endpoint name.
    """
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role_arn:
        raise ValueError(
            "SAGEMAKER_ROLE_ARN environment variable must be set to a valid "
            "SageMaker execution role ARN."
        )

    sm = boto3.client("sagemaker", region_name=region)

    # Resolve S3 URI — passed directly or retrieved from model package metadata
    if s3_model_uri is None:
        detail = sm.describe_model_package(ModelPackageName=model_package_arn)
        s3_model_uri = detail["CustomerMetadataProperties"]["s3_model_uri"]

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # Build and push our own serving container to ECR (avoids cross-account
    # ECR access restrictions on sandbox accounts with built-in containers)
    print("  Building and pushing serving container to ECR...")
    image_uri = build_and_push_serving_image(region)

    # Delete old model and config so we can recreate with the latest image
    for delete_call, resource_name in [
        (lambda: sm.delete_model(ModelName=model_name), model_name),
        (lambda: sm.delete_endpoint_config(EndpointConfigName=config_name), config_name),
    ]:
        try:
            delete_call()
            print(f"  Deleted old resource: {resource_name}")
        except ClientError:
            pass  # didn't exist — that's fine

    # Create the SageMaker Model directly with our ECR image + S3 artifact
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": s3_model_uri,
            "Environment": {},
        },
    )
    print(f"  Created SageMaker Model: {model_name}")

    # Create Endpoint Configuration
    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1,
                }
            ],
        )
        print(f"  Created Endpoint Config: {config_name}")
    except ClientError as e:
        if "already exist" in e.response["Error"]["Message"]:
            print(f"  Endpoint Config already exists: {config_name}")
        else:
            raise

    # Create or update the Endpoint
    try:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        print(f"  Creating endpoint: {endpoint_name}")
    except ClientError as e:
        if "already exist" in e.response["Error"]["Message"]:
            print(f"  Updating existing endpoint: {endpoint_name}")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            raise

    # Wait for InService (up to 30 min)
    print(f"  Waiting for endpoint to be InService (this may take ~5-10 min)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},
    )
    print(f"  Endpoint is InService: {endpoint_name}")

    return endpoint_name


def test_endpoint(endpoint_name: str, region: str) -> dict:
    """Test the deployed endpoint with sample pieces.

    Invokes the endpoint with representative inputs and compares
    the predictions against expected ranges.

    Args:
        endpoint_name: Name of the deployed endpoint.
        region: AWS region.

    Returns:
        Dict with test results and predictions.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # CSV format: die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s
    test_cases = [
        ("5052_normal", "5052,18.0,13.5"),
        ("5052_slow",   "5052,30.0,13.5"),
        ("4974",        "4974,18.0,13.5"),
        ("5090",        "5090,18.0,13.5"),
        ("5091",        "5091,18.0,13.5"),
    ]

    results = {}
    all_ok = True

    for name, payload in test_cases:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        prediction = float(response["Body"].read().decode("utf-8").strip())
        in_range = 40 < prediction < 80
        results[name] = round(prediction, 3)
        status = "OK" if in_range else "OUT OF RANGE"
        print(f"    {name}: {prediction:.2f}s [{status}]")
        if not in_range:
            all_ok = False

    # Sanity checks
    if "5052_slow" in results and "5052_normal" in results:
        slow_higher = results["5052_slow"] > results["5052_normal"]
        results["slow_piece_higher"] = slow_higher
        print(f"    slow > normal: {slow_higher}")

    results["all_in_range"] = all_ok
    return results


def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost model to SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifact")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--model-package-group", required=True, help="Model Package Group name")
    parser.add_argument(
        "--instance-type", default="ml.t2.medium",
        help="Endpoint instance type (default: ml.t2.medium)",
    )
    args = parser.parse_args()

    # Load model metadata for metrics
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SageMaker Deployment Pipeline")
    print("=" * 60)

    # Step 1: Package
    print("\n[1/5] Packaging model artifact...")
    tar_path = package_model(MODEL_FILE, MODEL_DIR)
    print(f"  Created: {tar_path}")

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3_key = "models/xgboost-bath-predictor/model.tar.gz"
    s3_uri = upload_to_s3(tar_path, args.bucket, s3_key)
    print(f"  Uploaded: {s3_uri}")

    # Step 3: Register in Model Registry
    print("\n[3/5] Registering in Model Registry...")
    model_package_arn = register_model(
        s3_uri, args.model_package_group, args.region, metadata["metrics"]
    )
    print(f"  Registered: {model_package_arn}")

    # Step 4: Deploy endpoint
    print("\n[4/5] Deploying endpoint...")
    endpoint = deploy_endpoint(
        model_package_arn, args.endpoint_name, args.region, args.instance_type,
        s3_model_uri=s3_uri,
    )
    print(f"  Endpoint live: {endpoint}")

    # Step 5: Test
    print("\n[5/5] Testing endpoint...")
    results = test_endpoint(args.endpoint_name, args.region)
    print(f"  Results: {json.dumps(results, indent=2)}")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Endpoint:       {args.endpoint_name}")
    print(f"  Model Package:  {model_package_arn}")
    print(f"  S3 artifact:    {s3_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
