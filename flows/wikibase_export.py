import os

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from pydantic import AnyHttpUrl, SecretStr

from flows.utils import SlackNotify, get_run_name
from knowledge_graph.cloud import AwsEnv, get_async_session
from knowledge_graph.operations.wikibase_export.rules import RULES, run_rules
from knowledge_graph.operations.wikibase_export.snapshot import crawl
from knowledge_graph.operations.wikibase_export.store import (
    advance_current,
    write_snapshot,
)
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"

DEFAULT_BUCKETS = {
    AwsEnv.production: "cpr-cache",
    AwsEnv.staging: "cpr-staging-cache",
}


@flow(
    persist_result=False,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def wikibase_export(
    aws_env: AwsEnv | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    bucket: str | None = None,
) -> None:
    """Export everything in Wikibase to S3, as a raw timestamped snapshot."""
    logger = get_logger()

    if aws_env is None:
        aws_env = AwsEnv(os.environ["AWS_ENV"])

    # Downstream reads the default buckets, so another Wikibase must not write there
    if wikibase_auth is not None and (
        bucket is None or bucket in DEFAULT_BUCKETS.values()
    ):
        raise ValueError("a Wikibase passed in explicitly must be given its own bucket")

    if bucket is None:
        bucket = DEFAULT_BUCKETS.get(aws_env)
    if bucket is None:
        raise ValueError(f"no default bucket for {aws_env.value}, pass a bucket")

    if wikibase_auth is None:
        wikibase_auth = WikibaseAuth(
            username=get_aws_ssm_param(WIKIBASE_USERNAME_SSM_NAME),
            password=SecretStr(get_aws_ssm_param(WIKIBASE_PASSWORD_SSM_NAME)),
            url=AnyHttpUrl(get_aws_ssm_param(WIKIBASE_URL_SSM_NAME)),
        )

    async with WikibaseSession(
        username=wikibase_auth.username,
        password=wikibase_auth.password.get_secret_value(),
        url=str(wikibase_auth.url),
    ) as wikibase:
        snapshot = await crawl(wikibase)
    logger.info(
        f"crawled {len(snapshot.items)} items and {len(snapshot.properties)} properties"
    )

    issues, failed_rules = run_rules(snapshot, RULES)
    logger.info(f"found {len(issues)} issues, {len(failed_rules)} rules failed")

    session = get_async_session(aws_env)
    async with session.client("s3") as s3_client:
        prefix = await write_snapshot(
            s3_client,
            bucket,
            snapshot,
            issues,
            failed_rules,
            flow_run_name=get_run_name(),
        )
        snapshot_uri = f"s3://{bucket}/{prefix}"
        logger.info(f"wrote snapshot to {snapshot_uri}")

        advanced = await advance_current(
            s3_client, bucket, snapshot, issues, failed_rules
        )
        logger.info(f"current pointer {'moved' if advanced else 'left in place'}")

    if failed_rules:
        raise RuntimeError(f"validation rules raised: {', '.join(failed_rules)}")
