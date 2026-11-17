import os

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from pydantic import AnyHttpUrl, SecretStr

from flows.utils import SlackNotify, get_run_name, get_slack_client
from knowledge_graph.cloud import AwsEnv, get_async_session
from knowledge_graph.operations.wikibase_export.report import (
    IssueDiff,
    diff_issues,
    summary_text,
    thread_texts,
)
from knowledge_graph.operations.wikibase_export.rules import RULES, run_rules
from knowledge_graph.operations.wikibase_export.snapshot import crawl
from knowledge_graph.operations.wikibase_export.store import (
    advance_current,
    mark_reported,
    read_reported_issues,
    write_snapshot,
)
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"

SLACK_CHANNEL = "alerts-concept-store"

DEFAULT_BUCKETS = {
    AwsEnv.production: "cpr-cache",
    AwsEnv.staging: "cpr-staging-cache",
}


async def post_issues_to_slack(diff: IssueDiff, snapshot_uri: str, advanced: bool):
    """Post a summary of changed issues, with the new ones in a thread."""
    slack_client = await get_slack_client()

    # The client raises SlackApiError if a post fails
    summary = await slack_client.chat_postMessage(
        channel=SLACK_CHANNEL, text=summary_text(diff, snapshot_uri, advanced)
    )
    for text in thread_texts(diff, snapshot_uri):
        await slack_client.chat_postMessage(
            channel=SLACK_CHANNEL, thread_ts=summary["ts"], text=text
        )


@flow(
    persist_result=False,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def wikibase_export(
    aws_env: AwsEnv | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    bucket: str | None = None,
    enable_slack_notifications: bool = False,
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

        # A failed rule's issues would otherwise read as resolved
        if enable_slack_notifications and not failed_rules:
            diff = diff_issues(await read_reported_issues(s3_client, bucket), issues)
            if diff.changed:
                await post_issues_to_slack(diff, snapshot_uri, advanced)
            # Only once posted, so a failed post is retried by the next run
            await mark_reported(s3_client, bucket, issues)

    if failed_rules:
        raise RuntimeError(f"validation rules raised: {', '.join(failed_rules)}")
