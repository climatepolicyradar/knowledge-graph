"""
Infrastructure for the vibe check static site

This module represents the infrastructure for a lightweight internal service that
provides a static website.

Infrastructure Overview:
- An S3 bucket, configured for static website hosting
- A CloudFront distribution, giving us HTTPS and tidy URL rewriting
- A bucket policy which allows CloudFront to access the content in s3

Given the likely volume of traffic for an internal too like this, the cost of this
setup should be negligible (probably pennies per month).
"""

import json

import pulumi
import pulumi_aws as aws

config = pulumi.Config()
app_name = "cpr-knowledge-graph-vibe-check"

# create an S3 bucket for us to dump static site assets into
bucket = aws.s3.Bucket(
    f"{app_name}",
    bucket=f"{app_name}",
    website=aws.s3.BucketWebsiteArgs(
        index_document="index.html",
        error_document="index.html",
    ),
)

# create an Origin Access Identity (OAI) for cloudfront to access the bucket
cloudfront_oai = aws.cloudfront.OriginAccessIdentity(
    f"{app_name}-oai",
    comment=f"OAI for {app_name} website",
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
        # don't forward query strings or cookies as they're not needed
        forwarded_values=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesArgs(
            query_string=False,
            cookies=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesCookiesArgs(
                forward="none",
            ),
        ),
    ),
    # define where cloudfront should fetch the content from.
    # we're using an OAI to securely access s3, keeping the bucket itself private and
    # only publicly accessible via cloudfront
    origins=[
        aws.cloudfront.DistributionOriginArgs(
            domain_name=bucket.bucket_regional_domain_name,
            origin_id=bucket.bucket,
            s3_origin_config=aws.cloudfront.DistributionOriginS3OriginConfigArgs(
                origin_access_identity=cloudfront_oai.cloudfront_access_identity_path,
            ),
        ),
    ],
    # configure how cloudfront should handle requests that don't match any of the
    # configured cache behaviors
    custom_error_responses=[
        aws.cloudfront.DistributionCustomErrorResponseArgs(
            error_code=404,
            response_code=200,
            response_page_path="/index.html",
            # setting this to 0 will ensure that we don't cache the 404 -> index.html redirects
            error_caching_min_ttl=0,
        ),
    ],
    # configure which edge locations cloudfront will use to serve content
    # we're using PriceClass_100 (which is the cheapest, only serving in North America
    # and Europe). This should be sufficient as since our origin is in eu-west-1 and
    # users are primarily in Europe.
    price_class="PriceClass_100",
    # Despite using price class 100, we're not restricting access by country as we want to
    # allow users from all over the world to access the service.
    restrictions=aws.cloudfront.DistributionRestrictionsArgs(
        geo_restriction=aws.cloudfront.DistributionRestrictionsGeoRestrictionArgs(
            restriction_type="none",
        ),
    ),
    # Use the default CloudFront SSL certificate for HTTPS
    # This gives us a domain like blah.cloudfront.net
    # For a custom domain, we would need to provide our own certificate
    viewer_certificate=aws.cloudfront.DistributionViewerCertificateArgs(
        cloudfront_default_certificate=True,
    ),
    # set index.html as the default root object so that requests to / will show the
    # content from  /index.html
    default_root_object="index.html",
    is_ipv6_enabled=True,
    http_version="http2",
)

pulumi.export("bucket_name", bucket.id)
pulumi.export("cloudfront_url", distribution.domain_name)
