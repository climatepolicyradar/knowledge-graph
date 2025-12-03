import json
import os

import pulumi
from pulumi_aws import (
    acm,
    cloudwatch,
    ec2,
    ecr,
    ecs,
    get_caller_identity,
    get_region,
    iam,
    lb,
    route53,
    s3,
    ssm,
)

caller_identity = get_caller_identity()
current_region = get_region()


def get_ssm_parameter_arn(parameter_name):
    return pulumi.Output.concat(
        "arn:aws:ssm:",
        current_region.region,
        ":",
        caller_identity.account_id,
        ":parameter",
        parameter_name,
    )


app_name = "vibe-checker"
aws_region = os.getenv("AWS_REGION", "eu-west-1")
aws_profile = os.getenv("AWS_PROFILE", "labs")

# Look up the existing hosted zone for labs.climatepolicyradar.org
hosted_zone = route53.get_zone(name="labs.climatepolicyradar.org")

# Use existing wildcard certificate for *.labs.climatepolicyradar.org
certificate = acm.get_certificate(
    domain="*.labs.climatepolicyradar.org", statuses=["ISSUED"], most_recent=True
)

# s3 bucket for storing the data
bucket = s3.Bucket(f"cpr-{app_name}")

# Store bucket name in SSM Parameter Store for services to discover
bucket_param = ssm.Parameter(
    f"{app_name}-bucket-name-param",
    name=f"/{app_name}/bucket-name",
    type="String",
    value=bucket.bucket,
    description="S3 bucket name for vibe-checker data",
)

# ECR repository for storing the webapp container image
repository = ecr.Repository(app_name)

# ECR repository for storing the pipeline container image
pipeline_repository = ecr.Repository(
    f"{app_name}-pipeline",
    name=f"{app_name}-pipeline",
)

cluster = ecs.Cluster(app_name)

# VPC and subnets
vpc = ec2.Vpc(
    f"{app_name}-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
)

public_subnet_a = ec2.Subnet(
    f"{app_name}-public-subnet-a",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone=f"{aws_region}a",
)

public_subnet_b = ec2.Subnet(
    f"{app_name}-public-subnet-b",
    vpc_id=vpc.id,
    cidr_block="10.0.3.0/24",
    availability_zone=f"{aws_region}b",
)

private_subnet_a = ec2.Subnet(
    f"{app_name}-private-subnet-a",
    vpc_id=vpc.id,
    cidr_block="10.0.2.0/24",
    availability_zone=f"{aws_region}a",
)

private_subnet_b = ec2.Subnet(
    f"{app_name}-private-subnet-b",
    vpc_id=vpc.id,
    cidr_block="10.0.4.0/24",
    availability_zone=f"{aws_region}b",
)

igw = ec2.InternetGateway(f"{app_name}-igw", vpc_id=vpc.id)

# Route table
public_route_table = ec2.RouteTable(
    f"{app_name}-public-rt",
    vpc_id=vpc.id,
    routes=[
        ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
)

ec2.RouteTableAssociation(
    f"{app_name}-public-rta-a",
    subnet_id=public_subnet_a.id,
    route_table_id=public_route_table.id,
)

ec2.RouteTableAssociation(
    f"{app_name}-public-rta-b",
    subnet_id=public_subnet_b.id,
    route_table_id=public_route_table.id,
)

# NAT Gateway for private subnet internet access
nat_eip = ec2.Eip(f"{app_name}-nat-eip", domain="vpc")

nat_gateway = ec2.NatGateway(
    f"{app_name}-nat-gw",
    allocation_id=nat_eip.id,
    subnet_id=public_subnet_a.id,
)

# Route table for private subnet
private_route_table = ec2.RouteTable(
    f"{app_name}-private-rt",
    vpc_id=vpc.id,
    routes=[
        ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            nat_gateway_id=nat_gateway.id,
        )
    ],
)

# Associate private subnet with private route table
ec2.RouteTableAssociation(
    f"{app_name}-private-rta-a",
    subnet_id=private_subnet_a.id,
    route_table_id=private_route_table.id,
)

ec2.RouteTableAssociation(
    f"{app_name}-private-rta-b",
    subnet_id=private_subnet_b.id,
    route_table_id=private_route_table.id,
)

# Security groups
alb_security_group = ec2.SecurityGroup(
    f"{app_name}-alb-sg",
    vpc_id=vpc.id,
    description="Allow HTTP and HTTPS inbound traffic",
    ingress=[
        ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=80,
            to_port=80,
            cidr_blocks=["0.0.0.0/0"],
            description="HTTP",
        ),
        ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=443,
            to_port=443,
            cidr_blocks=["0.0.0.0/0"],
            description="HTTPS",
        ),
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

ecs_security_group = ec2.SecurityGroup(
    f"{app_name}-ecs-sg",
    vpc_id=vpc.id,
    description="Allow inbound traffic from ALB",
    ingress=[
        ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=3000,
            to_port=3000,
            security_groups=[alb_security_group.id],
            description="Next.js app port",
        ),
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

# IAM roles
execution_role = iam.Role(
    f"{app_name}-execution-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }""",
)

iam.RolePolicyAttachment(
    f"{app_name}-execution-role-policy",
    role=execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

task_role = iam.Role(
    f"{app_name}-task-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }""",
)

# Create a custom iam policy that only allows access to the vibe-checker bucket
s3_read_only_policy = iam.Policy(
    f"{app_name}-s3-policy",
    policy=pulumi.Output.all(bucket.arn).apply(
        lambda args: f"""{{
            "Version": "2012-10-17",
            "Statement": [
                {{
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        "{args[0]}",
                        "{args[0]}/*"
                    ]
                }}
            ]
        }}"""
    ),
)

# Attach the custom policy to the task role
iam.RolePolicyAttachment(
    f"{app_name}-s3-read-policy",
    role=task_role.name,
    policy_arn=s3_read_only_policy.arn,
)

# Attach SSM read-only policy to task role (for fetching config from Parameter Store)
iam.RolePolicyAttachment(
    f"{app_name}-ssm-read-policy",
    role=task_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess",
)

# Attach SSM read-only policy to execution role (for webapp initialization)
iam.RolePolicyAttachment(
    f"{app_name}-execution-ssm-read-policy",
    role=execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess",
)

# Application Load Balancer
alb = lb.LoadBalancer(
    f"{app_name}-alb",
    internal=False,
    load_balancer_type="application",
    security_groups=[alb_security_group.id],
    subnets=[public_subnet_a.id, public_subnet_b.id],
)

# Target group for ECS service
target_group = lb.TargetGroup(
    f"{app_name}-tg",
    port=3000,
    protocol="HTTP",
    target_type="ip",
    vpc_id=vpc.id,
    health_check=lb.TargetGroupHealthCheckArgs(
        path="/api/health",
        protocol="HTTP",
        matcher="200",
        interval=30,
        timeout=5,
    ),
)

# HTTP listener that redirects to HTTPS
http_listener = lb.Listener(
    f"{app_name}-http-listener",
    load_balancer_arn=alb.arn,
    port=80,
    protocol="HTTP",
    default_actions=[
        lb.ListenerDefaultActionArgs(
            type="redirect",
            redirect=lb.ListenerDefaultActionRedirectArgs(
                port="443",
                protocol="HTTPS",
                status_code="HTTP_301",
            ),
        )
    ],
)

# HTTPS listener
listener = lb.Listener(
    f"{app_name}-listener",
    load_balancer_arn=alb.arn,
    port=443,
    protocol="HTTPS",
    ssl_policy="ELBSecurityPolicy-TLS13-1-2-2021-06",
    certificate_arn=certificate.arn,
    default_actions=[
        lb.ListenerDefaultActionArgs(
            type="forward",
            target_group_arn=target_group.arn,
        )
    ],
)

# Route53 A record for vibe-checker.labs.climatepolicyradar.org pointing to ALB
route53_record = route53.Record(
    f"{app_name}-dns",
    zone_id=hosted_zone.zone_id,
    name="vibe-checker",
    type="A",
    aliases=[
        route53.RecordAliasArgs(
            name=alb.dns_name,
            zone_id=alb.zone_id,
            evaluate_target_health=True,
        )
    ],
)

# CloudWatch log group
log_group = cloudwatch.LogGroup(
    f"{app_name}-log-group",
    name=f"/ecs/{app_name}",
    retention_in_days=7,
)

# ECS task definition
task_definition = ecs.TaskDefinition(
    f"{app_name}-task-def",
    family=f"{app_name}-family",
    cpu="256",
    memory="512",
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    execution_role_arn=execution_role.arn,
    task_role_arn=task_role.arn,
    container_definitions=pulumi.Output.all(
        bucket.bucket, repository.repository_url
    ).apply(
        lambda args: json.dumps(
            [
                {
                    "name": app_name,
                    "image": f"{args[1]}:latest",
                    "portMappings": [
                        {"containerPort": 3000, "hostPort": 3000, "protocol": "tcp"}
                    ],
                    "environment": [
                        {"name": "BUCKET_NAME", "value": args[0]},
                        {"name": "AWS_REGION", "value": aws_region},
                        {
                            "name": "NEXT_PUBLIC_CPR_APP_URL",
                            "value": "https://app.climatepolicyradar.org",
                        },
                        {
                            "name": "NEXT_PUBLIC_WIKIBASE_URL",
                            "value": "https://climatepolicyradar.wikibase.cloud",
                        },
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
        )
    ),
)

# ECS service
service = ecs.Service(
    f"{app_name}-service",
    cluster=cluster.arn,
    task_definition=task_definition.arn,
    desired_count=1,
    launch_type="FARGATE",
    network_configuration=ecs.ServiceNetworkConfigurationArgs(
        subnets=[private_subnet_a.id, private_subnet_b.id],
        security_groups=[ecs_security_group.id],
        assign_public_ip=False,
    ),
    load_balancers=[
        ecs.ServiceLoadBalancerArgs(
            target_group_arn=target_group.arn,
            container_name=app_name,
            container_port=3000,
        )
    ],
    opts=pulumi.ResourceOptions(depends_on=[listener]),
)

# Outputs
pulumi.export("alb_dns_name", alb.dns_name)
pulumi.export("bucket_name", bucket.bucket)
pulumi.export("repository_url", repository.repository_url)
pulumi.export("pipeline_repository_url", pipeline_repository.repository_url)
pulumi.export("cluster_name", cluster.name)
pulumi.export("service_name", service.name)
pulumi.export("log_group_name", log_group.name)
pulumi.export("certificate_arn", certificate.arn)
