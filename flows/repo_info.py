import os

import httpx
from opentelemetry import metrics, trace
from prefect import flow

run_counter = None


@flow(log_prints=True)
def get_repo_info(repo_name: str = "PrefectHQ/prefect"):
    global run_counter
    if run_counter is None:
        run_counter = metrics.get_meter("knowledge-graph-flows").create_counter(
            "repo_info_runs",
            description="The number of times the repo_info flow has been run",
        )

    print(f"Tracer: {trace.get_tracer_provider()}")
    print(f"OTEL_EXPORTER_OTLP_ENDPOINT: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")
    print(f"OTEL_EXPORT_OTLP_PROTOCOL: {os.getenv('OTEL_EXPORT_OTLP_PROTOCOL')}")
    print(f"OTEL_SERVICE_NAME: {os.getenv('OTEL_SERVICE_NAME')}")
    print(f"OTEL_RESOURCE_ATTRIBUTES: {os.getenv('OTEL_RESOURCE_ATTRIBUTES')}")
    url = f"https://api.github.com/repos/{repo_name}"

    response = httpx.get(url)
    response.raise_for_status()
    repo = response.json()

    run_counter.add(1)

    print(f"{repo_name} repository statistics 🤓:")
    print(f"Stars 🌠 : {repo['stargazers_count']}")
    print(f"Forks 🍴 : {repo['forks_count']}")


if __name__ == "__main__":
    # get_repo_info.serve(name="my-first-deployment")  # type: ignore
    get_repo_info()
