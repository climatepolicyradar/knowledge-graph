import json
import subprocess

import pulumi
import pulumi_aws as aws
import pulumi_docker as docker

from knowledge_graph.config import get_git_root

config = pulumi.Config()
application_name = config.require("application_name")


# Get current AWS account and region to construct ARNs
caller_identity = aws.get_caller_identity()
current_region = aws.get_region()


def get_ssm_parameter_arn(parameter_name):
    return pulumi.Output.concat(
        "arn:aws:ssm:",
        current_region.name,
        ":",
        caller_identity.account_id,
        ":parameter",
        parameter_name,
    )


ssm_username_arn = get_ssm_parameter_arn("/Wikibase/Cloud/ServiceAccount/Username")
ssm_password_arn = get_ssm_parameter_arn("/Wikibase/Cloud/ServiceAccount/Password")
ssm_url_arn = get_ssm_parameter_arn("/Wikibase/Cloud/URL")

# Create a private ECR repository to store the Docker image
repo = aws.ecr.Repository(f"{application_name}-repo")

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
    build=docker.DockerBuild(
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

# Create a security group that allows HTTP ingress and all egress
sg = aws.ec2.SecurityGroup(
    f"{application_name}-sg",
    vpc_id=default_vpc.id,
    description="Allow HTTP ingress",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=80,
            to_port=80,
            cidr_blocks=["0.0.0.0/0"],
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
    port=80,
    protocol="HTTP",
    target_type="ip",
    vpc_id=default_vpc.id,
)

listener = aws.lb.Listener(
    f"{application_name}-listener",
    load_balancer_arn=alb.arn,
    port=80,
    default_actions=[
        aws.lb.ListenerDefaultActionArgs(
            type="forward",
            target_group_arn=target_group.arn,
        )
    ],
)

# Create an IAM role for the ECS task execution
ecs_task_execution_role = aws.iam.Role(
    f"{application_name}-ecs-task-execution-role",
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
    f"{application_name}-ecs-task-execution-role-policy-attachment",
    role=ecs_task_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
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
                        "Resource": "*",  # Required to decrypt SecureString parameters
                    },
                ],
            }
        )
    ),
)

aws.iam.RolePolicyAttachment(
    f"{application_name}-ecs-task-ssm-policy-attachment",
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
    ).apply(
        lambda args: json.dumps(
            [
                {
                    "name": f"{application_name}-container",
                    "image": args[0],
                    "portMappings": [
                        {"containerPort": 80, "hostPort": 80, "protocol": "tcp"}
                    ],
                    "secrets": [
                        {"name": "WIKIBASE_USERNAME", "valueFrom": args[1]},
                        {"name": "WIKIBASE_PASSWORD", "valueFrom": args[2]},
                        {"name": "WIKIBASE_URL", "valueFrom": args[3]},
                    ],
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
            container_port=80,
        )
    ],
    opts=pulumi.ResourceOptions(depends_on=[listener]),
)

# Export the URL of the load balancer and version information
pulumi.export("url", alb.dns_name)
pulumi.export("image_version", git_commit_hash)
pulumi.export("image_name", image.image_name)
