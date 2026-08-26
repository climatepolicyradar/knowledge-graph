import pulumi
import pulumi_aws as aws

# One stack per AWS environment, named after that environment. Each stack must be
# run against the matching account's credentials, eg `AWS_PROFILE=labs pulumi up
# -s labs`.
stack = pulumi.get_stack()

# ECR repository holding the images that the Prefect flow deployments run from.
# It exists in every environment, and is named after this GitHub repo because
# that is where the deployment tooling gets the repository name from (see
# .github/workflows/prefect_deploy.yml).
knowledge_graph_ecr_repository = aws.ecr.Repository(
    f"{stack}-knowledge-graph-ecr-repository",
    name="knowledge-graph",
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=False,
    ),
    encryption_configurations=[
        aws.ecr.RepositoryEncryptionConfigurationArgs(
            encryption_type="AES256",
        )
    ],
    opts=pulumi.ResourceOptions(protect=True),
)

pulumi.export("ecr_repository_url", knowledge_graph_ecr_repository.repository_url)

# The feather files bucket only exists in production, where it is read by model
# training and (cross-account) by the vibe checker in labs.
if stack == "production":
    config = pulumi.Config()
    labs_aws_account_id = config.require_secret("labs_aws_account_id")

    production_knowledge_graph_feather_files_bucket = aws.s3.Bucket(
        "production-knowledge-graph-feather-files-bucket",
        bucket="cpr-kg-feather-files",
        grants=[
            aws.s3.BucketGrantArgs(
                id="0fedc730a2af259d90402b1197e87cf40c4014a20851f540cac4269c0156abb9",
                permissions=["FULL_CONTROL"],
                type="CanonicalUser",
            )
        ],
        region="eu-west-1",
        request_payer="BucketOwner",
        server_side_encryption_configuration=aws.s3.BucketServerSideEncryptionConfigurationArgs(
            rule=aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                    sse_algorithm="AES256",
                ),
                bucket_key_enabled=True,
            ),
        ),
        opts=pulumi.ResourceOptions(protect=True),
    )

    # Grant the labs AWS account read-only access to the feather files bucket.
    # This uses the _output form of the invoke so that the labs account id, which
    # is a secret Output, is resolved rather than stringified into the ARN.
    labs_cross_account_policy = aws.iam.get_policy_document_output(
        statements=[
            aws.iam.GetPolicyDocumentStatementArgs(
                sid="LabsCrossAccountReadOnly",
                effect="Allow",
                principals=[
                    aws.iam.GetPolicyDocumentStatementPrincipalArgs(
                        type="AWS",
                        identifiers=[
                            labs_aws_account_id.apply(
                                lambda account_id: f"arn:aws:iam::{account_id}:root"
                            )
                        ],
                    )
                ],
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    "arn:aws:s3:::cpr-kg-feather-files",
                    "arn:aws:s3:::cpr-kg-feather-files/*",
                ],
            ),
        ],
    )

    production_knowledge_graph_feather_files_bucket_policy = aws.s3.BucketPolicy(
        "production-knowledge-graph-feather-files-bucket-policy",
        bucket=production_knowledge_graph_feather_files_bucket.id,
        policy=labs_cross_account_policy.json,
    )
