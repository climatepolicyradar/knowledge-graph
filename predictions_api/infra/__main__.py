"""
Infrastructure for the predictions visualisation API

This module represents the infrastructure for a lightweight internal service, allowing
users to visualise the predictions of concept classifier models (see predictions_api/app
for the API implementation).

Infrastructure Overview:
- ECR repository to store the container image
- A single ECS Fargate task which runs the predictions API container
- Minimal networking (one public subnet in a dedicated VPC)
- Two IAM roles for
    - accessing the predictions-visualisation S3 bucket
    - running the ECS task
- A security group which allows inbound traffic to the ECS task on port 80

NB this design prioritizes simplicity and cost over resilience. This feels like a
reasonable compromise for an internal tool where constant availability is not critical,
but might not be suitable for a public-facing service.
"""

import pulumi
import pulumi_aws as aws
import pulumi_aws.cloudwatch as cloudwatch
import pulumi_aws.ec2 as ec2
import pulumi_aws.ecr as ecr
import pulumi_aws.ecs as ecs
import pulumi_aws.iam as iam

from scripts.config import aws_region

config = pulumi.Config()
app_name = "predictions-api"

current = aws.get_caller_identity()
account_id = current.account_id

# Create CloudWatch log group
log_group = cloudwatch.LogGroup(
    f"{app_name}-log-group", name=f"/ecs/{app_name}", retention_in_days=30
)

vpc = ec2.Vpc(
    f"{app_name}-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
)

public_subnet = ec2.Subnet(
    f"{app_name}-public",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone=f"{aws_region}a",
    map_public_ip_on_launch=True,
)

internet_gateway = ec2.InternetGateway(f"{app_name}-igw", vpc_id=vpc.id)

route_table = ec2.RouteTable(
    f"{app_name}-route-table",
    vpc_id=vpc.id,
    routes=[
        ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=internet_gateway.id)
    ],
)

route_table_association = ec2.RouteTableAssociation(
    f"{app_name}-route-table-association",
    subnet_id=public_subnet.id,
    route_table_id=route_table.id,
)

repository = ecr.Repository(
    f"{app_name}-repo", image_tag_mutability="MUTABLE", force_delete=True
)

cluster = ecs.Cluster(f"{app_name}-cluster")

task_execution_role = iam.Role(
    f"{app_name}-task-execution-role",
    assume_role_policy=iam.get_policy_document(
        statements=[
            {
                "actions": ["sts:AssumeRole"],
                "principals": [
                    {
                        "type": "Service",
                        "identifiers": ["ecs-tasks.amazonaws.com"],
                    }
                ],
            }
        ]
    ).json,
)

task_execution_policy = iam.RolePolicyAttachment(
    f"{app_name}-task-execution-policy",
    role=task_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

secrets_policy = iam.RolePolicy(
    f"{app_name}-secrets-policy",
    role=task_execution_role.id,
    policy=iam.get_policy_document(
        statements=[
            {
                "actions": ["secretsmanager:GetSecretValue"],
                "resources": [
                    f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_url*",
                    f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_username*",
                    f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_password*",
                ],
            }
        ]
    ).json,
)

task_role = iam.Role(
    f"{app_name}-task-role",
    assume_role_policy=iam.get_policy_document(
        statements=[
            {
                "actions": ["sts:AssumeRole"],
                "principals": [
                    {
                        "type": "Service",
                        "identifiers": ["ecs-tasks.amazonaws.com"],
                    }
                ],
            }
        ]
    ).json,
)

s3_policy = iam.RolePolicy(
    f"{app_name}-s3-policy",
    role=task_role.id,
    policy=iam.get_policy_document(
        statements=[
            {
                "actions": [
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                "resources": [
                    "arn:aws:s3:::prediction-visualisation",
                    "arn:aws:s3:::prediction-visualisation/*",
                ],
            }
        ]
    ).json,
)

task_security_group = ec2.SecurityGroup(
    f"{app_name}-security-group",
    vpc_id=vpc.id,
    description="Security group for ECS tasks",
    ingress=[
        ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=80,
            to_port=80,
            cidr_blocks=["0.0.0.0/0"],
        )
    ],
    egress=[
        ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"],
        )
    ],
)

task_definition = ecs.TaskDefinition(
    f"{app_name}-task-definition",
    family=f"{app_name}-task",
    cpu="512",
    memory="1024",
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    execution_role_arn=task_execution_role.arn,
    task_role_arn=task_role.arn,
    container_definitions=pulumi.Output.json_dumps(
        [
            {
                "name": app_name,
                "image": repository.repository_url.apply(lambda url: f"{url}:latest"),
                "essential": True,
                "portMappings": [
                    {
                        "containerPort": 80,
                        "protocol": "tcp",
                    }
                ],
                "healthCheck": {
                    "command": [
                        "CMD-SHELL",
                        "curl -f http://localhost/health-check || exit 1",
                    ],
                    "interval": 30,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 60,
                },
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group.name,
                        "awslogs-region": aws_region,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
                "secrets": [
                    {
                        "name": "WIKIBASE_URL",
                        "valueFrom": f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_url-D41HiA",
                    },
                    {
                        "name": "WIKIBASE_USERNAME",
                        "valueFrom": f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_username-OcSuoi",
                    },
                    {
                        "name": "WIKIBASE_PASSWORD",
                        "valueFrom": f"arn:aws:secretsmanager:{aws_region}:{account_id}:secret:wikibase_password-eZj5WE",
                    },
                ],
            }
        ]
    ),
)

service = ecs.Service(
    f"{app_name}-service",
    cluster=cluster.arn,
    desired_count=1,
    launch_type="FARGATE",
    task_definition=task_definition.arn,
    health_check_grace_period_seconds=60,
    network_configuration=ecs.ServiceNetworkConfigurationArgs(
        assign_public_ip=True,
        subnets=[public_subnet.id],
        security_groups=[task_security_group.id],
    ),
)

pulumi.export("ecr_repository_url", repository.repository_url)
pulumi.export("ecs_cluster_name", cluster.name)
pulumi.export("ecs_service_name", service.name)
pulumi.export("task_definition_family", task_definition.family)
pulumi.export("aws_region", aws_region)
