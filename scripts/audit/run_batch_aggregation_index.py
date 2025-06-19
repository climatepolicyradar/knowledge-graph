import asyncio

from prefect.deployments import run_deployment

from flows.utils import ENGLISH_TRANSLATION_SUFFIX, iterate_batch


def load_document_ids_from_file(filepath: str) -> set[str]:
    """Load document IDs from a file, extracting just the ID part from error lines."""
    document_ids = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # For error files, extract just the document ID (before the colon)
                doc_id = line.split(":")[0]
                document_ids.add(doc_id)
    return document_ids


async def main() -> None:
    flow_name = "run-indexing-from-aggregate-results"

    deployment_name = "knowledge-graph-run-indexing-from-aggregate-results-prod"

    # Load all document IDs from classifiers.txt
    all_classifiers = load_document_ids_from_file("classifiers.txt")
    print(f"Loaded {len(all_classifiers)} document IDs from classifiers.txt")

    # Load error document IDs to subtract
    post_run_errors = load_document_ids_from_file("post_run_other_errors.txt")
    post_run_missing_raw = load_document_ids_from_file(
        "post_run_document_ids_missing.txt"
    )
    # Add English translation suffix to missing document IDs
    post_run_missing = [
        doc_id + ENGLISH_TRANSLATION_SUFFIX for doc_id in post_run_missing_raw
    ]
    print(
        f"Loaded {len(post_run_errors)} error IDs and {len(post_run_missing)} missing IDs (with {ENGLISH_TRANSLATION_SUFFIX} suffix) from post_run files"
    )

    # Load translated error document IDs to subtract
    translated_errors = load_document_ids_from_file(
        "post_run_translated_other_errors.txt"
    )
    translated_missing = load_document_ids_from_file(
        "post_run_translated_document_ids_missing.txt"
    )
    print(
        f"Loaded {len(translated_errors)} translated error IDs and {len(translated_missing)} translated missing IDs"
    )

    # # Combine all error IDs to subtract
    # all_errors = (
    #     post_run_errors | set(post_run_missing) | translated_errors | translated_missing
    # )
    # print(f"Total error/missing IDs to subtract: {len(all_errors)}")

    # Calculate final document IDs (classifiers minus all errors, then add back translated missing)
    # Start with all classifiers, subtract the regular errors and missing
    remaining_classifiers = all_classifiers - post_run_errors - post_run_missing_raw
    # Add the translated missing documents (which have the _translated_en suffix)
    document_ids_or_stems = list(
        set(list(remaining_classifiers) + post_run_missing)
        - translated_errors
        - translated_missing
    )
    print(
        f"Final document count after subtracting errors: {len(document_ids_or_stems)}"
    )

    # Write document IDs to file
    output_file = "final_document_ids.txt"
    with open(output_file, "w") as f:
        for doc_id in document_ids_or_stems:
            f.write(f"{doc_id}\n")
    print(f"Wrote {len(document_ids_or_stems)} document IDs to {output_file}")

    batches = list(iterate_batch(document_ids_or_stems, 3000))
    print(f"Created {len(batches)} batches of up to 3000 documents each")

    # Record the results along the way to a file, about which batch
    # number it got to
    progress_file = "batch_progress_index.txt"

    # 1-index, so it matches the human friendly `n` from enumeration
    start_batch = 1

    # Do -1 to make it 0-index
    for n, batch in enumerate(batches[(start_batch - 1) :], start_batch):
        print(f"Processing batch {n}/{len(batches)} with {len(batch)} documents")

        try:
            # Copied params from a recent run: https://app.prefect.cloud/account/4b1558a0-3c61-4849-8b18-3e97e0516d78/workspace/1753b4f0-6221-4f6a-9233-b146518b4545/runs/flow-run/0684473b-b945-7a95-8000-1a43168a967f?g_range=%7B%22type%22%3A%22range%22%2C%22startDate%22%3A%222025-06-05T16%3A40%3A47.000Z%22%2C%22endDate%22%3A%222025-06-12T16%3A40%3A47.000Z%22%7D&entity_id=0684473b-b945-7a95-8000-1a43168a967f&state=cancelled&state=cancelling&state=completed&state=crashed&state=failed&state=paused&state=pending&state=running&state=scheduled&entity=flowRuns&entity=taskRuns&entity=events&entity=logs&entity=artifacts
            result = await run_deployment(
                name=f"{flow_name}/{deployment_name}",
                flow_run_name=f"indexing-without-deprecated-classifiers-{n}",
                parameters={
                    "run_output_identifier": "latest",
                    "document_stems": list(batch),
                    "config": None,
                    "indexer_concurrency_limit": 10,
                    "indexer_max_vespa_connections": 50,
                    "batch_size": 50,
                },
                # Rely on the flow's own timeout
                timeout=None,
            )

            with open(progress_file, "a") as f:
                # Record which documents were in each batch, though this is
                # reproducible if the input file hasn't changed
                for document_id in batch:
                    f.write(f"{document_id}\n")

            with open(f"batch_progress_index_{n}.txt", "a") as f:
                f.write(f"Batch {n}: SUCCESS - {result.state.type}\n")

            print(f"Batch {n} completed successfully")

        except Exception as exc:
            # Record failure and continue
            error_msg = f"Batch {n}: FAILED - {str(exc)}"
            with open(progress_file, "a") as f:
                f.write(f"{error_msg}\n")
            print(error_msg)

    print(f"All batches processed. Progress logged to {progress_file}")


if __name__ == "__main__":
    asyncio.run(main())
