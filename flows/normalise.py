from prefect import task, flow 

from scripts.generate_merge_candidates import normalise as norm


@task
def normalise(some_string: str) -> str:
    return norm(some_string)
    
@flow
def normalise_flow(input_string: str = "hello"):
    normalise(input_string)
    
    
if __name__ == "__main__":
    normalise_flow(input_string="This is a random input string for normalisation.")