"""
ECS/Fargate deployment script — builds, pushes, and deploys the Streamlit app.

Usage:
    export SAGEMAKER_ENDPOINT_NAME="vaultech-bath-predictor"
    export AWS_DEFAULT_REGION="eu-west-1"
    uv run python deploy/deploy_ecs.py \
      --region eu-west-1 \
      --endpoint-name vaultech-bath-predictor \
      --cluster vaultech-cluster \
      --service vaultech-streamlit
"""

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ECR_REPO_NAME = "vaultech-streamlit"
CONTAINER_PORT = 8501


# ── ECR ───────────────────────────────────────────────────────────────────────

def build_and_push_app_image(region: str) -> str:
    """Build the Streamlit app image and push it to ECR. Returns the image URI."""
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
    username, password = base64.b64decode(token["authorizationToken"]).decode().split(":")
    registry = token["proxyEndpoint"]
    subprocess.run(
        ["docker", "login", "--username", username, "--password", password, registry],
        check=True, capture_output=True,
    )
    print(f"  Authenticated to ECR")

    # Build and push (linux/amd64 for Fargate)
    print(f"  Building app image for linux/amd64 (this may take ~5 min)...")
    subprocess.run(
        [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "--provenance=false",
            "--push",
            "-t", image_uri,
            ".",
        ],
        check=True, cwd=str(PROJECT_ROOT),
    )
    print(f"  Pushed: {image_uri}")
    return image_uri


# ── IAM ───────────────────────────────────────────────────────────────────────

def ensure_execution_role(region: str) -> str:
    """Ensure ecsTaskExecutionRole exists. Returns its ARN."""
    iam = boto3.client("iam")
    role_name = "ecsTaskExecutionRole"

    try:
        role = iam.get_role(RoleName=role_name)
        print(f"  Execution role exists: {role_name}")
        return role["Role"]["Arn"]
    except ClientError:
        pass

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    print(f"  Created execution role: {role_name}")
    return role["Role"]["Arn"]


def ensure_task_role(region: str, endpoint_name: str) -> str:
    """Ensure VaultechECSTaskRole exists with SageMaker invoke permission. Returns ARN."""
    iam = boto3.client("iam")
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    role_name = "VaultechECSTaskRole"

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    try:
        role = iam.get_role(RoleName=role_name)
        print(f"  Task role exists: {role_name}")
        role_arn = role["Role"]["Arn"]
    except ClientError:
        role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
        role_arn = role["Role"]["Arn"]
        print(f"  Created task role: {role_name}")

    # Put inline policy for SageMaker invoke
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",
            "Resource": f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{endpoint_name}",
        }],
    })
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="SageMakerInvokeEndpoint",
        PolicyDocument=policy,
    )
    print(f"  Updated task role policy: {role_name}")
    return role_arn


# ── VPC / Security Group ──────────────────────────────────────────────────────

def get_default_vpc_and_subnets(region: str):
    """Return (vpc_id, [public_subnet_ids]).

    Prefers the default VPC; falls back to any available VPC with subnets.
    """
    ec2 = boto3.client("ec2", region_name=region)

    # Try default VPC first
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]

    # Fall back to any VPC
    if not vpcs:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "state", "Values": ["available"]}])["Vpcs"]

    if not vpcs:
        raise RuntimeError("No VPC found in the region. Create one in the AWS console.")

    vpc_id = vpcs[0]["VpcId"]

    # Try public subnets first (map-public-ip-on-launch = true)
    subnets = ec2.describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]},
        {"Name": "map-public-ip-on-launch", "Values": ["true"]},
    ])["Subnets"]

    # Fall back to all subnets in the VPC
    if not subnets:
        subnets = ec2.describe_subnets(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
        ])["Subnets"]

    if not subnets:
        raise RuntimeError(f"No subnets found in VPC {vpc_id}.")

    subnet_ids = [s["SubnetId"] for s in subnets]
    print(f"  VPC: {vpc_id} | Subnets: {subnet_ids}")
    return vpc_id, subnet_ids


def ensure_security_group(region: str, vpc_id: str) -> str:
    """Ensure a security group allowing port 8501 exists. Returns its ID."""
    ec2 = boto3.client("ec2", region_name=region)
    sg_name = "vaultech-streamlit-sg"

    existing = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [sg_name]},
        {"Name": "vpc-id", "Values": [vpc_id]},
    ])["SecurityGroups"]

    if existing:
        sg_id = existing[0]["GroupId"]
        print(f"  Security group exists: {sg_id}")
        return sg_id

    sg = ec2.create_security_group(
        GroupName=sg_name,
        Description="Allow inbound on port 8501 for Streamlit",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": CONTAINER_PORT,
            "ToPort": CONTAINER_PORT,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
        }],
    )
    print(f"  Created security group: {sg_id}")
    return sg_id


# ── ECS ───────────────────────────────────────────────────────────────────────

def ensure_cluster(cluster_name: str, region: str) -> str:
    """Create ECS cluster if it doesn't exist. Returns cluster ARN."""
    ecs = boto3.client("ecs", region_name=region)
    response = ecs.describe_clusters(clusters=[cluster_name])
    active = [c for c in response["clusters"] if c["status"] == "ACTIVE"]
    if active:
        print(f"  ECS cluster exists: {cluster_name}")
        return active[0]["clusterArn"]
    response = ecs.create_cluster(clusterName=cluster_name)
    print(f"  Created ECS cluster: {cluster_name}")
    return response["cluster"]["clusterArn"]


def register_task_definition(
    family: str,
    image_uri: str,
    execution_role_arn: str,
    task_role_arn: str,
    endpoint_name: str,
    region: str,
) -> str:
    """Register (or update) the Fargate task definition. Returns its ARN."""
    ecs = boto3.client("ecs", region_name=region)
    response = ecs.register_task_definition(
        family=family,
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu="512",
        memory="1024",
        executionRoleArn=execution_role_arn,
        taskRoleArn=task_role_arn,
        containerDefinitions=[{
            "name": "streamlit",
            "image": image_uri,
            "portMappings": [{"containerPort": CONTAINER_PORT, "protocol": "tcp"}],
            "environment": [
                {"name": "SAGEMAKER_ENDPOINT_NAME", "value": endpoint_name},
                {"name": "AWS_DEFAULT_REGION",       "value": region},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group":         f"/ecs/{family}",
                    "awslogs-region":        region,
                    "awslogs-stream-prefix": "streamlit",
                    "awslogs-create-group":  "true",
                },
            },
        }],
    )
    arn = response["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered task definition: {arn}")
    return arn


def deploy_service(
    cluster_name: str,
    service_name: str,
    task_def_arn: str,
    subnet_ids: list[str],
    sg_id: str,
    region: str,
) -> str:
    """Create or update ECS service. Returns the service ARN."""
    ecs = boto3.client("ecs", region_name=region)
    network_config = {
        "awsvpcConfiguration": {
            "subnets": subnet_ids,
            "securityGroups": [sg_id],
            "assignPublicIp": "ENABLED",
        }
    }

    existing = ecs.describe_services(cluster=cluster_name, services=[service_name])
    active = [s for s in existing["services"] if s["status"] != "INACTIVE"]

    if active:
        response = ecs.update_service(
            cluster=cluster_name,
            service=service_name,
            taskDefinition=task_def_arn,
            forceNewDeployment=True,
        )
        print(f"  Updated ECS service: {service_name}")
        return response["service"]["serviceArn"]

    response = ecs.create_service(
        cluster=cluster_name,
        serviceName=service_name,
        taskDefinition=task_def_arn,
        desiredCount=1,
        launchType="FARGATE",
        networkConfiguration=network_config,
    )
    print(f"  Created ECS service: {service_name}")
    return response["service"]["serviceArn"]


def wait_for_public_ip(cluster_name: str, service_name: str, region: str) -> str:
    """Wait for the task to start and return its public IP."""
    ecs = boto3.client("ecs", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)

    print("  Waiting for task to start (up to 5 min)...")
    for _ in range(30):
        tasks = ecs.list_tasks(cluster=cluster_name, serviceName=service_name)
        if tasks["taskArns"]:
            detail = ecs.describe_tasks(cluster=cluster_name, tasks=tasks["taskArns"])
            task = detail["tasks"][0]
            if task["lastStatus"] == "RUNNING":
                for att in task.get("attachments", []):
                    for detail_item in att.get("details", []):
                        if detail_item["name"] == "networkInterfaceId":
                            eni_id = detail_item["value"]
                            eni = ec2.describe_network_interfaces(
                                NetworkInterfaceIds=[eni_id]
                            )["NetworkInterfaces"][0]
                            public_ip = eni.get("Association", {}).get("PublicIp")
                            if public_ip:
                                return public_ip
        time.sleep(10)
    return "(not yet available — check ECS console)"


def main():
    parser = argparse.ArgumentParser(description="Deploy Streamlit app to ECS/Fargate")
    parser.add_argument("--region",        default="eu-west-1")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--cluster",       default="vaultech-cluster")
    parser.add_argument("--service",       default="vaultech-streamlit")
    args = parser.parse_args()

    print("=" * 60)
    print("ECS/Fargate Deployment Pipeline")
    print("=" * 60)

    print("\n[1/7] Building and pushing Streamlit image to ECR...")
    image_uri = build_and_push_app_image(args.region)

    print("\n[2/7] Ensuring IAM roles...")
    exec_role_arn = ensure_execution_role(args.region)
    task_role_arn = ensure_task_role(args.region, args.endpoint_name)

    print("\n[3/7] Resolving VPC and subnets...")
    vpc_id, subnet_ids = get_default_vpc_and_subnets(args.region)

    print("\n[4/7] Ensuring security group (port 8501)...")
    sg_id = ensure_security_group(args.region, vpc_id)

    print("\n[5/7] Ensuring ECS cluster...")
    ensure_cluster(args.cluster, args.region)

    print("\n[6/7] Registering task definition...")
    task_def_arn = register_task_definition(
        family="vaultech-streamlit",
        image_uri=image_uri,
        execution_role_arn=exec_role_arn,
        task_role_arn=task_role_arn,
        endpoint_name=args.endpoint_name,
        region=args.region,
    )

    print("\n[7/7] Deploying ECS service...")
    deploy_service(args.cluster, args.service, task_def_arn, subnet_ids, sg_id, args.region)

    print("\n  Waiting for public IP...")
    public_ip = wait_for_public_ip(args.cluster, args.service, args.region)

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Cluster:   {args.cluster}")
    print(f"  Service:   {args.service}")
    print(f"  Image:     {image_uri}")
    print(f"  Public URL: http://{public_ip}:{CONTAINER_PORT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
