# Prefect Flows 

This directory contains prefect flows that can be developed in the knowledge graph repository. 

This allows one to seemlessy develop flows very close to the source code. 


## Running in the prefect worker in ECS

The following example runs a flow in the prefect worker that is deployed in an elastic container service cluster in the cloud. 

- Login to prefect cloud, select the api key option and use the key provided inBitwarden:
```shell 
prefect cloud login 
```

Execute the flow:
```shell
poetry run python -m flows.hello
```

