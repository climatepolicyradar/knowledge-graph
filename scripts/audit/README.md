# Audit results

To review the results of a Knowledge Graph run in Prefect and to find failed documents, you can execute [collect_prefect_artifacts.py](./audit/collect_prefect_artifacts.py) 
which will fetch the Prefect artifacts and load them into a JSON file. It requires you to authenticate your Prefect Cloud session. 

Run `uv run python ./scripts/audit/collect_prefect_artifacts.py <RUN-NAME>`


You can then extract a formatted array of the document IDs for failed documents by running the above command with the option `--write-failed-doc-ids-to-json` before the `<RUN-NAME>`, and it will write them to `./scripts/audit/data/audit/prefect_artifacts/<RUN-NAME>/failed_docs.json` 

You may paste that into the Prefect Cloud Web UI to run a new Custom Run of the Knowledge Graph but only for those documents
