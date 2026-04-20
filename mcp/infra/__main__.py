import json
import subprocess

import pulumi
import pulumi_aws as aws
import pulumi_docker as docker
from pulumi_aws import acm, route53

from knowledge_graph.config import get_git_root

application_name = "concept-store-mcp"

# Get current AWS account and region to construct ARNs
caller_identity = aws.get_caller_identity()
current_region = aws.get_region()


def get_ssm_parameter_arn(parameter_name):
    return pulumi.Output.concat(
        "arn:aws:ssm:",
        current_region.region,
        ":",
        caller_identity.account_id,
        ":parameter",
        parameter_name,
    )


# Look up the existing hosted zone and certificate
hosted_zone = route53.get_zone(name="labs.climatepolicyradar.org")
certificate = acm.get_certificate(
    domain="*.labs.climatepolicyradar.org", statuses=["ISSUED"], most_recent=True
)

ssm_username_arn = get_ssm_parameter_arn("/Wikibase/Cloud/ServiceAccount/Username")
ssm_password_arn = get_ssm_parameter_arn("/Wikibase/Cloud/ServiceAccount/Password")
ssm_url_arn = get_ssm_parameter_arn("/Wikibase/Cloud/URL")

# Create a private ECR repository to store the Docker image
repo = aws.ecr.Repository(f"{application_name}-repo")

aws.ecr.LifecyclePolicy(
    f"{application_name}-ecr-lifecycle-policy",
    repository=repo.name,
    policy=json.dumps(
        {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Keep last 50 images",
                    "selection": {
                        "tagStatus": "any",
                        "countType": "imageCountMoreThan",
                        # Keeping 50 images provides roughly 14 days of history based on our busiest
                        # push frequencies (up to ~50 images pushed in a 14 day window).
                        "countNumber": 50,
                    },
                    "action": {"type": "expire"},
                }
            ]
        }
    ),
)

# Get authorization token for ECR
auth = aws.ecr.get_authorization_token()

# Build and publish the image to the ECR repository.
# The context is the root of the project.
root_dir = get_git_root()
dockerfile_path = (root_dir / "mcp" / "Dockerfile").resolve()

# use the short hash of the current git commit as the version tag for the image
git_commit_hash = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], text=True
).strip()

image = docker.Image(
    f"{application_name}-image",
    build=docker.DockerBuildArgs(
        context=str(root_dir.resolve()),
        dockerfile=str(dockerfile_path),
        platform="linux/amd64",  # Specify platform for cross-platform builds
    ),
    image_name=pulumi.Output.concat(repo.repository_url, ":", git_commit_hash),
    registry=docker.RegistryArgs(
        server=auth.proxy_endpoint,
        username=auth.user_name,
        password=auth.password,
    ),
)

# Create an ECS cluster to run the service
cluster = aws.ecs.Cluster(f"{application_name}-cluster")

# Get the default VPC and subnets to deploy into
default_vpc = aws.ec2.get_vpc(default=True)
default_subnet_ids = aws.ec2.get_subnets(
    filters=[aws.ec2.GetSubnetsFilterArgs(name="vpc-id", values=[default_vpc.id])]
).ids

# Create a security group that allows HTTP/HTTPS ingress and all egress
sg = aws.ec2.SecurityGroup(
    f"{application_name}-security-group",
    vpc_id=default_vpc.id,
    description="Allow HTTP and HTTPS ingress",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=80,
            to_port=80,
            cidr_blocks=["0.0.0.0/0"],
            description="HTTP",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=443,
            to_port=443,
            cidr_blocks=["0.0.0.0/0"],
            description="HTTPS",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=8000,
            to_port=8000,
            cidr_blocks=["0.0.0.0/0"],
            description="Application port for ALB health checks and traffic",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"],
        ),
    ],
)

# Create a load balancer to listen for HTTP traffic
alb = aws.lb.LoadBalancer(
    f"{application_name}-alb",
    internal=False,
    security_groups=[sg.id],
    subnets=default_subnet_ids,
)

target_group = aws.lb.TargetGroup(
    f"{application_name}-tg",
    port=8000,
    protocol="HTTP",
    target_type="ip",
    vpc_id=default_vpc.id,
    health_check=aws.lb.TargetGroupHealthCheckArgs(
        enabled=True,
        healthy_threshold=2,
        unhealthy_threshold=3,
        timeout=10,
        interval=30,
        path="/health",
        matcher="200",
        protocol="HTTP",
        port="traffic-port",
    ),
)

# HTTP listener that redirects to HTTPS
http_listener = aws.lb.Listener(
    f"{application_name}-http-listener",
    load_balancer_arn=alb.arn,
    port=80,
    protocol="HTTP",
    default_actions=[
        aws.lb.ListenerDefaultActionArgs(
            type="redirect",
            redirect=aws.lb.ListenerDefaultActionRedirectArgs(
                port="443",
                protocol="HTTPS",
                status_code="HTTP_301",
            ),
        )
    ],
)

# HTTPS listener
listener = aws.lb.Listener(
    f"{application_name}-listener",
    load_balancer_arn=alb.arn,
    port=443,
    protocol="HTTPS",
    ssl_policy="ELBSecurityPolicy-TLS13-1-2-2021-06",
    certificate_arn=certificate.arn,
    default_actions=[
        aws.lb.ListenerDefaultActionArgs(
            type="forward",
            target_group_arn=target_group.arn,
        )
    ],
)

# Route53 A record pointing to the ALB
route53_record = route53.Record(
    f"{application_name}-dns",
    zone_id=hosted_zone.zone_id,
    name="concept-store-mcp",
    type="A",
    aliases=[
        route53.RecordAliasArgs(
            name=alb.dns_name,
            zone_id=alb.zone_id,
            evaluate_target_health=True,
        )
    ],
)

# Create an IAM role for the ECS task execution
ecs_task_execution_role = aws.iam.Role(
    f"{application_name}-ecs-task-exec-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            }
        }]
    }""",
)

aws.iam.RolePolicyAttachment(
    f"{application_name}-ecs-exec-policy",
    role=ecs_task_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

# Create a CloudWatch log group for the application logs
log_group = aws.cloudwatch.LogGroup(
    f"{application_name}-logs",
    retention_in_days=7,
)

# Add a policy to access SSM parameters
ssm_policy = aws.iam.Policy(
    f"{application_name}-ssm-policy",
    description="Policy to allow reading Wikibase credentials from SSM Parameter Store",
    policy=pulumi.Output.all(
        ssm_username_arn,
        ssm_password_arn,
        ssm_url_arn,
    ).apply(
        lambda arns: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["ssm:GetParameters", "ssm:GetParameter"],
                        "Resource": arns,
                    },
                    {
                        "Effect": "Allow",
                        "Action": "kms:Decrypt",
                        "Resource": "*",
                    },
                ],
            }
        )
    ),
)

aws.iam.RolePolicyAttachment(
    f"{application_name}-ssm-policy-attachment",
    role=ecs_task_execution_role.name,
    policy_arn=ssm_policy.arn,
)


# Create a Fargate task definition
task_definition = aws.ecs.TaskDefinition(
    f"{application_name}-task",
    family=f"{application_name}-task-family",
    cpu="256",  # 0.25 vCPU
    memory="512",  # 512MB
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    execution_role_arn=ecs_task_execution_role.arn,
    container_definitions=pulumi.Output.all(
        image.image_name,
        ssm_username_arn,
        ssm_password_arn,
        ssm_url_arn,
        log_group.name,
    ).apply(
        lambda args: json.dumps(
            [
                {
                    "name": f"{application_name}-container",
                    "image": args[0],
                    "portMappings": [
                        {"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}
                    ],
                    "secrets": [
                        {"name": "WIKIBASE_USERNAME", "valueFrom": args[1]},
                        {"name": "WIKIBASE_PASSWORD", "valueFrom": args[2]},
                        {"name": "WIKIBASE_URL", "valueFrom": args[3]},
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": args[4],
                            "awslogs-region": current_region.region,
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                }
            ]
        )
    ),
)

# Create a Fargate service to run the task definition
service = aws.ecs.Service(
    f"{application_name}-service",
    cluster=cluster.arn,
    desired_count=1,  # Start with one task
    launch_type="FARGATE",
    task_definition=task_definition.arn,
    network_configuration=aws.ecs.ServiceNetworkConfigurationArgs(
        assign_public_ip=True,
        subnets=default_subnet_ids,
        security_groups=[sg.id],
    ),
    load_balancers=[
        aws.ecs.ServiceLoadBalancerArgs(
            target_group_arn=target_group.arn,
            container_name=f"{application_name}-container",
            container_port=8000,
        )
    ],
    opts=pulumi.ResourceOptions(depends_on=[listener]),
)

pulumi.export("alb_dns_name", alb.dns_name)
pulumi.export("url", "https://concept-store-mcp.labs.climatepolicyradar.org")
pulumi.export("mcp_url", "https://concept-store-mcp.labs.climatepolicyradar.org/mcp")
pulumi.export(
    "health_check_url",
    "https://concept-store-mcp.labs.climatepolicyradar.org/health",
)
pulumi.export("image_version", git_commit_hash)
pulumi.export("image_name", image.image_name)
pulumi.export("log_group_name", log_group.name)
