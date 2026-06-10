import pulumi
import pulumi_aws as aws

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

# Grant the labs AWS account read-only access to the feather files bucket
labs_cross_account_policy = aws.iam.get_policy_document(
    statements=[
        aws.iam.GetPolicyDocumentStatementArgs(
            sid="LabsCrossAccountReadOnly",
            effect="Allow",
            principals=[
                aws.iam.GetPolicyDocumentStatementPrincipalArgs(
                    type="AWS",
                    identifiers=[f"arn:aws:iam::{labs_aws_account_id}:root"],
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
