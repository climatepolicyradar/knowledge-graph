import asyncio
import time

import wandb
from flows.inference import (
    Config,
    batch_text_block_inference,
    load_classifier,
    text_block_inference,
)


# NOTE: these model aliases are to be found in STAGING.
# Initially I was debugging errors, because I was looking for versions not present in this env.
async def main():
    config = await Config.create()

    wandb.login(key=config.wandb_api_key.get_secret_value())  # type: ignore
    run = wandb.init(  # pyright: ignore[reportAttributeAccessIssue]
        entity=config.wandb_entity,
        job_type="concept_inference",
    )

    classifier = await load_classifier(
        run,
        config,
        "Q1653",
        "v0",
    )

    start_time = time.time()
    for i in list(range(1000)):
        text_block_inference(
            classifier=classifier,
            block_id=f"block_{i}",
            text="This is a test text for the classifier." * 5,
        )
    end_time = time.time()

    time_taken_Q1653 = end_time - start_time
    print(f"Time taken for Q1653: {time_taken_Q1653} seconds")

    start_time = time.time()
    all_text, all_block_ids = [], []
    for i in list(range(1000)):
        all_block_ids.append(f"block_{i}")
        all_text.append("This is a test text for the classifier." * 5)

    batch_text_block_inference(
        classifier=classifier, all_text=all_text, all_block_ids=all_block_ids
    )

    end_time = time.time()

    time_taken_Q1653_batch_10 = end_time - start_time
    print(f"Time taken for Q1653 batch 10: {time_taken_Q1653_batch_10} seconds")

    start_time = time.time()
    all_text, all_block_ids = [], []
    for i in list(range(1000)):
        all_block_ids.append(f"block_{i}")
        all_text.append("This is a test text for the classifier." * 5)

    batch_text_block_inference(
        classifier=classifier,
        all_text=all_text,
        all_block_ids=all_block_ids,
        batch_size=100,
    )

    end_time = time.time()

    time_taken_Q1653_batch_100 = end_time - start_time
    print(f"Time taken for Q1653 batch 100: {time_taken_Q1653_batch_100} seconds")

    classifier = await load_classifier(
        run,
        config,
        "Q368",
        "v7",
    )

    start_time = time.time()
    for i in list(range(1000)):
        text_block_inference(
            classifier=classifier,
            block_id=f"block_{i}",
            text="This is a test text for the classifier." * 5,
        )
    end_time = time.time()

    time_taken_Q368 = end_time - start_time

    print(f"Time taken for Q368: {time_taken_Q368} seconds")

    wandb.finish()  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    asyncio.run(main())
