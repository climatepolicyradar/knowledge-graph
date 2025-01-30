import pytest

from flows.utils import SlackNotify, file_name_from_path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("Q1.json", "Q1"),
        ("test/Q2.json", "Q2"),
        ("test/test/test/Q3.json", "Q3"),
    ],
)
def test_file_name_from_path(path, expected):
    assert file_name_from_path(path) == expected


def test_message(mock_prefect_slack_webhook, mock_flow, mock_flow_run):
    SlackNotify.message(mock_flow, mock_flow_run, mock_flow_run.state)
    mock_SlackWebhook, mock_prefect_slack_block = mock_prefect_slack_webhook

    # `.load`
    mock_SlackWebhook.load.assert_called_once_with(
        "slack-webhook-platform-prefect-mvp-sandbox"
    )

    # `.notify`
    mock_prefect_slack_block.notify.assert_called_once()
    kwargs = mock_prefect_slack_block.notify.call_args.kwargs
    message = kwargs.get("body", "")
    assert message == (
        "Flow run TestFlow/TestFlowRun observed in state `Completed` at "
        "2025-01-28T12:00:00+00:00. For environment: sandbox. Flow run URL: "
        "None/flow-runs/flow-run/test-flow-run-id. State message: message"
    )
