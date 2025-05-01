import tempfile

from flows.boundary import (
    get_document_passages_from_vespa__generator,
    get_vespa_search_adapter_from_aws_secrets,
)

temp_dir = tempfile.TemporaryDirectory()
vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
    cert_dir=temp_dir.name,
    # TODO: Checked and these are read only certs used for monitoring
    vespa_private_key_param_name="VESPA_PRIVATE_KEY",
    vespa_public_cert_param_name="VESPA_PUBLIC_CERT",
)

passages_generator = get_document_passages_from_vespa__generator(
    # Note: This document has 59,000 passages (more than the max hits)
    document_import_id="UNFCCC.party.1010.0",
    vespa_search_adapter=vespa_search_adapter,
    continuation_tokens=["BKAAAAABKBGA"],
    grouping_max=100,
)
