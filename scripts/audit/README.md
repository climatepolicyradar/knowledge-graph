# Audit results

To review the results of a Knowledge Graph run in Prefect and to find failed documents, you can execute [collect_prefect_artifacts.py](./audit/collect_prefect_artifacts.py) 
which will fetch the Prefect artifacts and load them into a JSON file. It requires you to authenticate your Prefect Cloud session. 

Run `uv run python ./scripts/audit/collect_prefect_artifacts.py <RUN-NAME>`
