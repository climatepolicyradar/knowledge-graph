import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from flows.wikibase_export import SLACK_CHANNEL, wikibase_export
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.operations.wikibase_export.rules import Issue
from knowledge_graph.operations.wikibase_export.snapshot import IncompleteCrawlError
from knowledge_graph.operations.wikibase_export.store import (
    PREFIX,
    read_current,
    read_reported_issues,
)
from knowledge_graph.wikibase import WikibaseAuth
from tests.operations.wikibase_export.helpers import get_json, list_keys, put_json

BUCKET = "test-wikibase-export-flow"


@pytest.fixture
def auth(mock_wikibase_url: str) -> WikibaseAuth:
    return WikibaseAuth(
        username="test_user", password="test_password", url=mock_wikibase_url
    )


@pytest_asyncio.fixture
async def bucket(mock_s3_async_client):
    await mock_s3_async_client.create_bucket(
        Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    yield BUCKET
    response = await mock_s3_async_client.list_objects_v2(Bucket=BUCKET)
    for obj in response.get("Contents", []):
        await mock_s3_async_client.delete_object(Bucket=BUCKET, Key=obj["Key"])
    await mock_s3_async_client.delete_bucket(Bucket=BUCKET)


@pytest.fixture
def slack():
    client = AsyncMock()
    client.chat_postMessage.return_value = {"ok": True, "ts": "1"}
    with patch(
        "flows.wikibase_export.get_slack_client", AsyncMock(return_value=client)
    ):
        yield client


def one_issue(snapshot) -> list[Issue]:
    return [Issue(rule="one_issue", severity="warning", entity_id="Q10", message="m")]


async def snapshot_timestamps(s3, bucket: str) -> set[str]:
    keys = await list_keys(s3, bucket, f"{PREFIX}/snapshots/")
    return {key.split("/")[3] for key in keys}


@pytest.mark.asyncio
async def test_flow_writes_snapshot_and_pointer(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    await wikibase_export.fn(aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket)

    current = await read_current(mock_s3_async_client, bucket)
    assert current is not None
    manifest = await get_json(
        mock_s3_async_client,
        bucket,
        f"{PREFIX}/snapshots/{current.snapshot}/manifest.json",
    )
    assert manifest["counts"] == {"items": 10, "properties": 2, "redirects": 0}
    slack.chat_postMessage.assert_not_called()
    assert await read_reported_issues(mock_s3_async_client, bucket) is None


@pytest.mark.asyncio
async def test_flow_requires_bucket_outside_staging_and_production(
    MockedWikibaseSession,
):
    with pytest.raises(ValueError, match="no default bucket for sandbox"):
        await wikibase_export.fn(aws_env=AwsEnv.sandbox)


@pytest.mark.asyncio
async def test_flow_posts_once_then_stays_quiet(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    with patch("flows.wikibase_export.RULES", [one_issue]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=auth,
            bucket=bucket,
            enable_slack_notifications=True,
        )
        first_run_posts = slack.chat_postMessage.call_count
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=auth,
            bucket=bucket,
            enable_slack_notifications=True,
        )

    # One summary, plus one thread reply for the one rule with issues
    assert first_run_posts == 2
    summary, thread = slack.chat_postMessage.call_args_list
    assert summary.kwargs["channel"] == SLACK_CHANNEL == "alerts-concept-store"
    assert "1 new" in summary.kwargs["text"]
    assert thread.kwargs["thread_ts"] == "1"
    assert "Q10" in thread.kwargs["text"]
    assert slack.chat_postMessage.call_count == first_run_posts


@pytest.mark.asyncio
async def test_flow_reposts_after_failed_slack_post(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    slack.chat_postMessage.return_value = {"ok": False, "error": "channel_not_found"}

    with patch("flows.wikibase_export.RULES", [one_issue]):
        with pytest.raises(RuntimeError, match="channel_not_found"):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox,
                wikibase_auth=auth,
                bucket=bucket,
                enable_slack_notifications=True,
            )

    assert await read_reported_issues(mock_s3_async_client, bucket) is None
    # The snapshot still landed
    assert await read_current(mock_s3_async_client, bucket) is not None


@pytest.mark.asyncio
async def test_flow_leaves_pinned_pointer(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    await put_json(
        mock_s3_async_client,
        bucket,
        f"{PREFIX}/current.json",
        {"snapshot": "20260101T000000Z", "pinned": True},
    )

    with patch("flows.wikibase_export.RULES", [one_issue]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=auth,
            bucket=bucket,
            enable_slack_notifications=True,
        )

    current = await read_current(mock_s3_async_client, bucket)
    assert current is not None and current.snapshot == "20260101T000000Z"
    assert len(await snapshot_timestamps(mock_s3_async_client, bucket)) == 1
    assert "pinned" in slack.chat_postMessage.call_args_list[0].kwargs["text"]


@pytest.mark.asyncio
async def test_flow_fails_but_still_writes_when_a_rule_raises(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    def broken_rule(snapshot):
        raise ValueError("boom")

    with patch("flows.wikibase_export.RULES", [broken_rule]):
        with pytest.raises(RuntimeError, match="broken_rule"):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
            )

    current = await read_current(mock_s3_async_client, bucket)
    assert current is not None
    manifest = await get_json(
        mock_s3_async_client,
        bucket,
        f"{PREFIX}/snapshots/{current.snapshot}/manifest.json",
    )
    assert manifest["failed_rules"] == ["broken_rule"]


@pytest.mark.asyncio
async def test_flow_writes_nothing_when_crawl_fails(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    with patch(
        "flows.wikibase_export.crawl",
        AsyncMock(side_effect=IncompleteCrawlError("nothing listed")),
    ):
        with pytest.raises(IncompleteCrawlError):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
            )

    assert await list_keys(mock_s3_async_client, bucket, PREFIX) == set()


@pytest.mark.asyncio
async def test_flow_threads_every_rule_under_the_summary(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    def another_issue(snapshot) -> list[Issue]:
        return [
            Issue(
                rule="another_issue", severity="warning", entity_id="Q100", message="m"
            )
        ]

    slack.chat_postMessage.side_effect = [
        {"ok": True, "ts": "summary"},
        {"ok": True, "ts": "reply-1"},
        {"ok": True, "ts": "reply-2"},
    ]

    with patch("flows.wikibase_export.RULES", [one_issue, another_issue]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=auth,
            bucket=bucket,
            enable_slack_notifications=True,
        )

    _, first_reply, second_reply = slack.chat_postMessage.call_args_list
    assert first_reply.kwargs["thread_ts"] == "summary"
    assert second_reply.kwargs["thread_ts"] == "summary"


@pytest.mark.asyncio
async def test_flow_does_not_report_when_a_rule_raises(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    def broken_rule(snapshot):
        raise ValueError("boom")

    with patch("flows.wikibase_export.RULES", [one_issue]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=auth,
            bucket=bucket,
            enable_slack_notifications=True,
        )
    reported = await read_reported_issues(mock_s3_async_client, bucket)
    posts = slack.chat_postMessage.call_count
    # Snapshots are named to the second, so let the next run get its own
    await asyncio.sleep(1.1)

    # one_issue's rule now fails, which must not read as its issue being fixed
    with patch("flows.wikibase_export.RULES", [broken_rule]):
        with pytest.raises(RuntimeError, match="broken_rule"):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox,
                wikibase_auth=auth,
                bucket=bucket,
                enable_slack_notifications=True,
            )

    assert slack.chat_postMessage.call_count == posts
    assert await read_reported_issues(mock_s3_async_client, bucket) == reported


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket_name", [None, "cpr-cache", "cpr-staging-cache"])
async def test_flow_refuses_another_wikibase_in_a_default_bucket(
    MockedWikibaseSession, auth, bucket_name
):
    crawl = AsyncMock()

    with patch("flows.wikibase_export.crawl", crawl):
        with pytest.raises(ValueError, match="its own bucket"):
            await wikibase_export.fn(
                aws_env=AwsEnv.staging, wikibase_auth=auth, bucket=bucket_name
            )

    crawl.assert_not_called()


@pytest.mark.asyncio
async def test_flow_holds_pointer_back_on_an_error_issue(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth, slack
):
    def error_rule(snapshot) -> list[Issue]:
        return [
            Issue(rule="error_rule", severity="error", entity_id="Q10", message="m")
        ]

    with patch("flows.wikibase_export.RULES", [error_rule]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
        )

    assert await read_current(mock_s3_async_client, bucket) is None
    (timestamp,) = await snapshot_timestamps(mock_s3_async_client, bucket)
    manifest = await get_json(
        mock_s3_async_client, bucket, f"{PREFIX}/snapshots/{timestamp}/manifest.json"
    )
    assert manifest["issues"]["by_severity"] == {"error": 1}
