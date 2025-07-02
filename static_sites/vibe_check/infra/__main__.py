"""
Infrastructure for the vibe check static site

This module represents the infrastructure for a lightweight internal service that
deploys a static website.

Infrastructure Overview:
- An S3 bucket, configured for static website hosting
- A CloudFront distribution, giving us HTTPS and clean URL rewriting via CloudFront Functions
- A bucket policy which allows CloudFront to access the content in s3

Given the likely volume of traffic for an internal tool like this, the cost of this
setup should be negligible (probably pennies per month).
"""

import json

import pulumi
import pulumi_aws as aws

config = pulumi.Config()
app_name = "cpr-knowledge-graph-vibe-check"

# Create an S3 bucket for hosting the static site
bucket = aws.s3.Bucket(
    f"{app_name}",
    bucket=f"{app_name}",
    website=aws.s3.BucketWebsiteArgs(
        index_document="index.html",
        error_document="index.html",
    ),
)

# Create an Origin Access Identity for CloudFront
cloudfront_oai = aws.cloudfront.OriginAccessIdentity(
    f"{app_name}-oai",
    comment="OAI for Vibe Check static site",
)

# Create a bucket policy that allows CloudFront to access the bucket
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

# Create a CloudFront Function for clean URLs
clean_urls_function = aws.cloudfront.Function(
    f"{app_name}-clean-urls",
    name=f"{app_name}-clean-urls",
    runtime="cloudfront-js-1.0",
    code="""
function handler(event) {
    var request = event.request;
    var uri = request.uri;
    
    // Check whether the URI is missing a file name.
    if (uri.endsWith('/')) {
        request.uri += 'index.html';
    } 
    // Check whether the URI is missing a file extension.
    else if (!uri.includes('.')) {
        request.uri += '/index.html';
    }
    
    return request;
}
""",
)

# Create a CloudFront distribution
distribution = aws.cloudfront.Distribution(
    f"{app_name}-distribution",
    enabled=True,
    is_ipv6_enabled=True,
    http_version="http2",
    # Use PriceClass_100 (cheapest) - only North America and Europe
    # Sufficient since origin is in eu-west-1 and users primarily in Europe
    price_class="PriceClass_100",
    default_root_object="index.html",
    origins=[
        aws.cloudfront.DistributionOriginArgs(
            domain_name=bucket.bucket_regional_domain_name,
            origin_id=bucket.bucket,
            origin_path="/dist",
            s3_origin_config=aws.cloudfront.DistributionOriginS3OriginConfigArgs(
                origin_access_identity=cloudfront_oai.cloudfront_access_identity_path,
            ),
        ),
    ],
    default_cache_behavior=aws.cloudfront.DistributionDefaultCacheBehaviorArgs(
        # Only allow GET/HEAD/OPTIONS methods as this is a static site
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        cached_methods=["GET", "HEAD", "OPTIONS"],
        target_origin_id=bucket.bucket,
        # Force HTTPS for all requests
        viewer_protocol_policy="redirect-to-https",
        # Don't forward query strings or cookies as they're not needed
        forwarded_values=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesArgs(
            query_string=False,
            cookies=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesCookiesArgs(
                forward="none",
            ),
        ),
        # Set reasonable TTL values for caching
        min_ttl=0,
        default_ttl=3600,
        max_ttl=86400,
        function_associations=[
            aws.cloudfront.DistributionDefaultCacheBehaviorFunctionAssociationArgs(
                event_type="viewer-request",
                function_arn=clean_urls_function.arn,
            )
        ],
    ),
    # Handle 404s by redirecting to index.html for SPA behavior
    custom_error_responses=[
        aws.cloudfront.DistributionCustomErrorResponseArgs(
            error_code=404,
            response_code=200,
            response_page_path="/index.html",
            # Setting this to 0 will ensure that we don't cache the 404 -> index.html redirects
            error_caching_min_ttl=0,
        ),
    ],
    # No geographic restrictions - accessible from anywhere
    restrictions=aws.cloudfront.DistributionRestrictionsArgs(
        geo_restriction=aws.cloudfront.DistributionRestrictionsGeoRestrictionArgs(
            restriction_type="none",
        ),
    ),
    # Use default CloudFront SSL certificate for HTTPS
    viewer_certificate=aws.cloudfront.DistributionViewerCertificateArgs(
        cloudfront_default_certificate=True,
    ),
)

# Export the bucket name and CloudFront URL
pulumi.export("bucket_name", bucket.id)
pulumi.export("cloudfront_url", distribution.domain_name)
