"""
Infrastructure for the vibe check static site

This module represents the infrastructure for a lightweight internal service that
provides a static website.

Infrastructure Overview:
- An S3 bucket configured for storing static website content
- A CloudFront distribution, providing:
  - HTTPS termination
  - Global content delivery and caching
  - Directory index handling via CloudFront Functions
- A bucket policy restricting access to CloudFront via Origin Access Identity

Given the likely volume of traffic for an internal tool like this, the cost of this
setup should be negligible (probably pennies per month).
"""

import json

import pulumi
import pulumi_aws as aws

config = pulumi.Config()
app_name = "cpr-knowledge-graph-vibe-check"

# create an S3 bucket configured for website hosting
bucket = aws.s3.Bucket(
    f"{app_name}",
    bucket=f"{app_name}",
    website=aws.s3.BucketWebsiteArgs(
        index_document="index.html",
        error_document="index.html",
    ),
)

# create an Origin Access Identity (OAI) for cloudfront to access the bucket
cloudfront_oai = aws.cloudfront.OriginAccessIdentity(f"{app_name}-oai")

# Create a CloudFront function to handle directory indexes. For example, if a user
# requests /Q123, the function will redirect to /Q123/index.html
directory_index_function = aws.cloudfront.Function(
    f"{app_name}-directory-index",
    name=f"{app_name}-directory-index",
    runtime="cloudfront-js-1.0",
    code="""
function handler(event) {
    var request = event.request;
    var uri = request.uri;
    
    // If URI is empty or ends with a slash, append index.html
    if (uri === "" || uri === "/" || uri.endsWith("/")) {
        request.uri = uri + "index.html";
    }
    // If URI doesn't contain a file extension, append /index.html
    else if (!uri.includes(".")) {
        request.uri = uri + "/index.html";
    }
    
    return request;
}
""",
)

# update the bucket policy to allow access from our cloudfront OAI
bucket_policy = aws.s3.BucketPolicy(
    f"{app_name}-policy",
    bucket=bucket.id,
    policy=pulumi.Output.all(bucket=bucket.id, oai=cloudfront_oai.iam_arn).apply(
        lambda args: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "CloudFrontReadGetObject",
                        "Effect": "Allow",
                        "Principal": {"AWS": args["oai"]},
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{args['bucket']}/*",
                    }
                ],
            }
        )
    ),
)

# Get AWS's managed cache policy for CachingOptimized
managed_cache_policy = aws.cloudfront.get_cache_policy(name="Managed-CachingOptimized")

# create the cloudfront distribution
distribution = aws.cloudfront.Distribution(
    f"{app_name}-distribution",
    enabled=True,
    # configure how cloudfront handles requests
    default_cache_behavior=aws.cloudfront.DistributionDefaultCacheBehaviorArgs(
        # only allow get/head/options methods as this is a static site
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        cached_methods=["GET", "HEAD", "OPTIONS"],
        target_origin_id=bucket.bucket,
        # force https for all requests
        viewer_protocol_policy="redirect-to-https",
        cache_policy_id=managed_cache_policy.id,
        compress=True,
        # Add function to handle directory indexes
        function_associations=[
            aws.cloudfront.DistributionDefaultCacheBehaviorFunctionAssociationArgs(
                event_type="viewer-request",
                function_arn=directory_index_function.arn,
            )
        ],
    ),
    # define where cloudfront should fetch the content from using the S3 bucket
    origins=[
        aws.cloudfront.DistributionOriginArgs(
            domain_name=bucket.bucket_regional_domain_name,
            origin_id=bucket.bucket,
            s3_origin_config=aws.cloudfront.DistributionOriginS3OriginConfigArgs(
                origin_access_identity=cloudfront_oai.cloudfront_access_identity_path,
            ),
        ),
    ],
    # configure which edge locations cloudfront will use to serve content
    price_class="PriceClass_100",
    restrictions=aws.cloudfront.DistributionRestrictionsArgs(
        geo_restriction=aws.cloudfront.DistributionRestrictionsGeoRestrictionArgs(
            restriction_type="none",
        ),
    ),
    viewer_certificate=aws.cloudfront.DistributionViewerCertificateArgs(
        cloudfront_default_certificate=True,
    ),
    default_root_object="index.html",
    is_ipv6_enabled=True,
    http_version="http2",
)

pulumi.export("bucket_name", bucket.id)
pulumi.export("cloudfront_url", distribution.domain_name)
