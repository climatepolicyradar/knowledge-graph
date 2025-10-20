import json
from typing import Annotated

import typer

app = typer.Typer()


def write_failed_doc_ids(flow_run_name):
    dir_path = "./data/audit/prefect_artifacts/{0}/"

    dir_path = dir_path.format(flow_run_name)

    failed_doc_ids = []
    with open(dir_path + "batch_inference_summary.json", "r") as file:
        data = json.load(file)

        for doc in data:
            if doc["Status"] == "\u2717":
                failed_doc_ids.append(doc["Document stem"])

    # print(failed_doc_ids)

    print(f"{len(failed_doc_ids)} Failed documents")

    report_output_file_path = dir_path + "failed_inference_docs.json"
    print(f"Written report to: {report_output_file_path}")
    with open(report_output_file_path, "w") as file:
        json.dump(failed_doc_ids, file)

    # non_placeholder_doc_ids = []

    # for doc in failed_doc_ids:
    #    if "placeholder" not in doc:
    #        non_placeholder_doc_ids.append(doc)

    # print(non_placeholder_doc_ids)
    # print(f'{len(non_placeholder_doc_ids)} Non Placeholder Failed documents' )


@app.command()
def main(
    flow_run_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="The flow run name. You must have previously run collect_prefect_artifacts.py to download this",
        ),
    ],
):
    """Generates a json file of the failed documents in a format that can be uploaded into the Prefect Cloud UI as a custom run"""

    write_failed_doc_ids(flow_run_name=flow_run_name)


if __name__ == "__main__":
    app()
