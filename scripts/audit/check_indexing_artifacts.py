import asyncio
import json

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterName
from prefect.client.schemas.objects import Artifact


# @AsyncProfiler(printer=print)
async def main() -> None:
    client = get_client()

    # Get artifacts from both main flows and sub-flows
    artifacts: list[Artifact] = await client.read_artifacts(
        flow_run_filter=FlowRunFilter(
            name=FlowRunFilterName(
                any_=[
                    f"indexing-without-deprecated-classifiers-{n}" for n in range(1, 7)
                ],
            ),
        )
    )

    # Also get artifacts from the batch sub-flows which contain the detailed error tables
    batch_artifacts: list[Artifact] = await client.read_artifacts(
        # Note: no flow_run_filter here to get all artifacts that match the key pattern
    )

    # Filter for the batch-level table artifacts with detailed error information
    detailed_artifacts = [
        artifact
        for artifact in batch_artifacts
        if artifact.key == "indexing-aggregate-results-prod"
        and artifact.type == "table"
    ]

    print(f"Found {len(detailed_artifacts)} detailed table artifacts")

    # Combine all artifacts
    all_artifacts = artifacts + detailed_artifacts

    # Filter for table artifacts that contain the detailed error information
    table_artifacts = [
        artifact for artifact in detailed_artifacts if artifact.type == "table"
    ]
    print(
        f"Found {len(table_artifacts)} table artifacts with detailed error information"
    )

    all_indexing_failures = []
    all_vespa_update_failures = []
    all_other_errors = []

    for artifact in table_artifacts:
        # Table artifacts store data as JSON arrays
        if hasattr(artifact, "data") and artifact.data:
            try:
                data = json.loads(artifact.data)

                # Table artifacts contain a list of dictionaries
                if isinstance(data, list):
                    for item in data:
                        # Check if this is a document details entry with failure status
                        if isinstance(item, dict) and item.get("Status") == "✗":
                            error_info = item.get("Errors", "")
                            family_doc_id = item.get("Family document ID", "")

                            # Categorize the error based on error message content
                            if "text block not found in Vespa" in error_info:
                                all_indexing_failures.append(
                                    {
                                        "Family document ID": family_doc_id,
                                        "Error": error_info,
                                    }
                                )
                            elif "Vespa update failed" in error_info:
                                all_vespa_update_failures.append(
                                    {
                                        "Family document ID": family_doc_id,
                                        "Error": error_info,
                                    }
                                )
                            else:
                                all_other_errors.append(
                                    {
                                        "Family document ID": family_doc_id,
                                        "Error": error_info,
                                    }
                                )
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse artifact data for {artifact.key}: {e}")
                continue

    print(
        f"Total indexing failures (text block not found): {len(all_indexing_failures)}"
    )
    print(f"Total Vespa update failures: {len(all_vespa_update_failures)}")
    print(f"Total other errors: {len(all_other_errors)}")

    # Write indexing failures to file
    indexing_failure_ids = [
        error.get("Family document ID") for error in all_indexing_failures
    ]
    with open("post_run_indexing_text_block_failures.txt", "w") as f:
        for doc_id in indexing_failure_ids:
            f.write(f"{doc_id}\n")

    # Write Vespa update failures to file
    vespa_failure_ids = [
        error.get("Family document ID") for error in all_vespa_update_failures
    ]
    with open("post_run_indexing_vespa_update_failures.txt", "w") as f:
        for doc_id in vespa_failure_ids:
            f.write(f"{doc_id}\n")

    # Write other errors to file
    with open("post_run_indexing_other_errors.txt", "w") as f:
        for error in all_other_errors:
            doc_id = error.get("Family document ID")
            error_msg = error.get("Error")
            f.write(f"{doc_id}: {error_msg}\n")

    print(
        f"Wrote {len(indexing_failure_ids)} text block failure IDs to post_run_indexing_text_block_failures.txt"
    )
    print(
        f"Wrote {len(vespa_failure_ids)} Vespa update failure IDs to post_run_indexing_vespa_update_failures.txt"
    )
    print(
        f"Wrote {len(all_other_errors)} other errors to post_run_indexing_other_errors.txt"
    )


if __name__ == "__main__":
    asyncio.run(main())
