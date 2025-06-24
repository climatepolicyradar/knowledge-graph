import asyncio
import json
import rich
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterName
from prefect.client.schemas.objects import Artifact, ArtifactCollection

from flows.utils import DocumentImportId, AsyncProfiler, iterate_batch


# @AsyncProfiler(printer=print)
async def main() -> None:
    client = get_client()

    artifacts: list[Artifact] = await client.read_artifacts(
        flow_run_filter=FlowRunFilter(
            name=FlowRunFilterName(
                any_=[
                    f"deprecating-static-classifiers-translated-{n}"
                    for n in range(1, 5)
                ],
            ),
        )
    )

    all_no_such_key_errors = []
    all_other_errors = []

    for artifact in artifacts:
        data = json.loads(artifact.data)

        for item in data:
            context = item.get("Context")
            if (
                isinstance(context, dict)
                and context.get("Error", {}).get("Code") == "NoSuchKey"
            ):
                all_no_such_key_errors.append(item)
            else:
                all_other_errors.append(item)

    print(f"Total NoSuchKey errors across all artifacts: {len(all_no_such_key_errors)}")
    print(f"Total other errors across all artifacts: {len(all_other_errors)}")

    document_ids = [error.get("Failed document ID") for error in all_no_such_key_errors]

    with open("post_run_translated_document_ids_missing.txt", "w") as f:
        for doc_id in document_ids:
            f.write(f"{doc_id}\n")

    with open("post_run_translated_other_errors.txt", "w") as f:
        for error in all_other_errors:
            doc_id = error.get("Failed document ID")
            exception = error.get("Exception")
            f.write(f"{doc_id}: {exception}\n")

    print(
        f"Wrote {len(document_ids)} document IDs to post_run_translated_document_ids_missing.txt"
    )
    print(
        f"Wrote {len(all_other_errors)} other errors to post_run_translated_other_errors.txt"
    )


if __name__ == "__main__":
    asyncio.run(main())
