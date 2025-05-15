import asyncio
import os
import time
from collections import Counter
from datetime import timedelta

import httpx
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.boundary import (
    get_document_passages_from_vespa__generator,
    total_milliseconds,
)

GROUPING_MAX = 10
GROUPING_PRECISION = 10000

DEFAULT_DOCUMENTS_BATCH_SIZE = 1
VESPA_MAX_TIMEOUT_MS = total_milliseconds(timedelta(minutes=50))
instance_url = os.getenv("VESPA_INSTANCE_URL")
client = VespaSearchAdapter(instance_url=instance_url).client


batches_dict = {}
batches_list = []


async def run():
    async with (
        client.asyncio(
            connections=DEFAULT_DOCUMENTS_BATCH_SIZE,  # How many tasks to have running at once
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),  # Seconds
        ) as vespa_connection_pool
    ):
        generator = get_document_passages_from_vespa__generator(
            document_import_id="CCLW.executive.1317.2153",
            vespa_connection_pool=vespa_connection_pool,
            grouping_max=GROUPING_MAX,
            grouping_precision=GROUPING_PRECISION,
        )
        async for batch in generator:
            batches_dict.update(batch)
            batches_list.extend(batch)

    print(len(batches_dict))
    print(len(batches_list))

    counter = Counter(batches_list)
    duplicates = [(pid, count) for pid, count in counter.items() if count > 1]
    print(len(duplicates))


start_time = time.time()
asyncio.run(run())
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
