import asyncio
import time

import wandb
from flows.inference import Config, load_classifier, text_block_inference


async def main():

    config = await Config.create()

    wandb.login(
        key=config.wandb_api_key.get_secret_value()
    )  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    run = wandb.init(  # pyright: ignore[reportAttributeAccessIssue]
        entity=config.wandb_entity,
        job_type="concept_inference",
    )

    classifier = await load_classifier(
        run,
        config,
        "Q1653",
        "v1",
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

    classifier = await load_classifier(
        run,
        config,
        "Q368",
        "v8",
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

    print(f"Time taken for Q1653: {time_taken_Q1653} seconds")
    print(f"Time taken for Q368: {time_taken_Q368} seconds")

    wandb.finish()  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    asyncio.run(main())
