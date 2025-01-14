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
- A security group which allows inbound traffic to the ECS task on port 8000

NB this design prioritizes simplicity and cost over resilience. This feels like a
reasonable compromise for an internal tool where constant availability is not critical,
but might not be suitable for a public-facing service.
"""

import pulumi
import pulumi_aws.ec2 as ec2
import pulumi_aws.ecr as ecr
import pulumi_aws.ecs as ecs
import pulumi_aws.iam as iam

from scripts.config import aws_region

config = pulumi.Config()
app_name = "predictions-api"

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

repository = ecr.Repository(f"{app_name}-repo", image_tag_mutability="MUTABLE")

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
            from_port=8000,
            to_port=8000,
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
    cpu="256",
    memory="512",
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
                        "containerPort": 8000,
                        "protocol": "tcp",
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/{app_name}",
                        "awslogs-region": aws_region,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
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
    network_configuration=ecs.ServiceNetworkConfigurationArgs(
        assign_public_ip=True,
        subnets=[public_subnet.id],
        security_groups=[task_security_group.id],
    ),
)

pulumi.export(
    "task_public_ip",
    service.id.apply(lambda _: "Access the API at http://<task_public_ip>:8000"),
)
