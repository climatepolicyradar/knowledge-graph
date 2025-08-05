import json
import tempfile

from prefect.main import flow
from vespa.io import VespaResponse

from flows.boundary import get_vespa_search_adapter_from_aws_secrets


@flow
async def update_model_profile(id, concepts_versions):
    with tempfile.TemporaryDirectory() as temp_dir:
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir,  # Use the path directly
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )

        fields = {
            "concepts_versions": concepts_versions,
        }

        # TODO:
        data_id = id

        async with vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            response: VespaResponse = await vespa_connection_pool.update_data(
                schema="family_document",
                namespace="doc_search",
                data_id=data_id,
                create=False,
                fields=fields,
            )

    if not response.is_successful():
        # Account for when Vespa fails to include the body
        try:
            # `get_json` returns a Dict[1].
            #
            # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
            json_s = json.dumps(response.get_json())
            print(f"Vespa update failed: {json_s}")
        except Exception as e:
            print(f"failed to get JSON from Vespa response: {e}")

    return response


@flow
async def upsert_model_profile(id, fields):
    pass


@flow
async def insert_model_profile(id, fields):
    pass


@flow
async def delete_model_profile(id):
    pass


# Example:
# await update_model_profile(
#     "primaries",
#     {
#         "q880": "r5n8qz2t",
#         "q730": "w3kb7x1s",
#     },
# )
