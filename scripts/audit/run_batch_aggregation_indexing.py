import asyncio
from prefect.deployments import run_deployment
import typer

from flows.utils import DocumentImportId, Profiler, iterate_batch

app = typer.Typer()


async def run_all_async() -> None:
    # Load from classifiers.txt
    with open("classifiers.txt", "r") as f:
        document_ids = [DocumentImportId(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(document_ids)} document IDs from classifiers.txt")

    batches = list(iterate_batch(document_ids, 1000))
    print(f"Created {len(batches)} batches of up to 1000 documents each")

    flow_name = "aggregate-inference-results"

    deployment_name = "knowledge-graph-aggregate-inference-results-prod"

    # Record the results along the way to a file, about which batch
    # number it got to
    progress_file = "batch_progress.txt"

    # 1-index, so it matches the human friendly `n` from enumeration
    start_batch = 9

    # Do -1 to make it 0-index
    for n, batch in enumerate(batches[(start_batch - 1) :], start_batch):
        print(f"Processing batch {n}/{len(batches)} with {len(batch)} documents")

        try:
            # The classifiers specs. will be loaded from the files
            result = await run_deployment(
                name=f"{flow_name}/{deployment_name}",
                flow_run_name=f"deprecating-static-classifiers-{n}",
                idempotency_key=f"deprecating-static-classifiers-{n}",
                parameters={
                    "max_concurrent_tasks": 20,
                    "batch_size": 5,
                    "document_ids": list(batch),
                    "config": None,
                    "config": None,
                },
                # Rely on the flow's own timeout
                timeout=None,
            )

            with open(progress_file, "a") as f:
                # Record which documents were in each batch, though this is
                # reproducible if the input file hasn't changed
                for document_id in batch:
                    f.write(f"{document_id}\n")

            with open(f"batch_progress_{n}.txt", "a") as f:
                f.write(f"Batch {n}: SUCCESS - {result.state.type}\n")

            print(f"Batch {n} completed successfully")

        except Exception as exc:
            # Record failure and continue
            error_msg = f"Batch {n}: FAILED - {str(exc)}"
            with open(progress_file, "a") as f:
                f.write(f"{error_msg}\n")
            print(error_msg)

    print(f"All batches processed. Progress logged to {progress_file}")


@Profiler(printer=print)
@app.command()
def run_all() -> None:
    asyncio.run(run_all_async())


if __name__ == "__main__":
    app()
