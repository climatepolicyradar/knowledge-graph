import pytest
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.boundary import DocumentImportId
from scripts.audit.do_s3_passages_align_with_vespa import get_vespa_passage_counts
from tests.flows.conftest import *  # noqa: F403


@pytest.mark.vespa
def test_get_vespa_passage_counts(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """Test the get_vespa_passage_counts function."""

    expected_counts = {
        DocumentImportId("CCLW.executive.10014.4470"): 1830,
        DocumentImportId("CCLW.executive.4934.1571"): 135,
    }

    vespa_passage_counts = get_vespa_passage_counts(vespa=local_vespa_search_adapter)

    assert vespa_passage_counts == expected_counts, (
        "Passage counts don't match expected values"
    )
