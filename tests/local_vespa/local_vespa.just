# Setup dev instance of vespa for test and local development

# The vespa cli is required, this checks if its installed
# See: https://docs.vespa.ai/en/vespa-cli.html
vespa_confirm_cli_installed:
	@if [ ! $(which vespa) ]; then \
		echo 'ERROR: The vespa cli is not installed, please install and try again:' ; \
		echo 'https://docs.vespa.ai/en/vespa-cli.html'; \
		exit 1; \
	fi

# Starts a detached instance of vespa ready
vespa_dev_start:
	docker compose -f tests/local_vespa/docker-compose.dev.yml up --detach --wait vespadaltest

# Confirms the local vespa instance has been spun up and is healthy
vespa_healthy:
	@if [ ! $(curl -f -s 'http://localhost:19071/status.html') ]; then \
		echo 'ERROR: Bad response from local vespa cluster, is it running?'; \
		exit 1; \
	fi

# Deploys schema to dev instance
vespa_deploy_schema:
	vespa config set target local
	@vespa deploy tests/local_vespa/test_app --wait 180

# Feeds some test data into local Vespa instance
vespa_feed_data:
	vespa config set target local
	vespa feed --target local tests/local_vespa/test_documents/search_weights.json
	vespa feed --target local tests/local_vespa/test_documents/family_document.json
	vespa feed --target local tests/local_vespa/test_documents/document_passage.json
	vespa feed --target local tests/local_vespa/test_documents/concept/*
	vespa feed --target local tests/local_vespa/test_documents/classifiers_profile/*
	vespa feed --target local tests/local_vespa/test_documents/classifiers_profiles/*


# Feeds some large test data into local Vespa instance
vespa_feed_large_data:
	vespa config set target local
	vespa feed --target local tests/local_vespa/test_documents/family_document_UNFCCC.party.309.0.json
	vespa feed --target local tests/local_vespa/test_documents/document_passage_UNFCCC.party.309.0.json

# Deletes some test data from local Vespa instance
#
# Alternative approaches: https://docs.vespa.ai/en/operations/batch-delete.html
vespa_delete_data:
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/search_weights/docid?selection=true&cluster=family-document-passage"
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/family_document/docid?selection=true&cluster=family-document-passage"
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/document_passage/docid?selection=true&cluster=family-document-passage"
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/concept/docid?selection=true&cluster=family-document-passage"
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/classifiers_profile/docid?selection=true&cluster=family-document-passage"
	curl -X DELETE "http://localhost:8080/document/v1/doc_search/classifiers_profiles/docid?selection=true&cluster=family-document-passage"


copy_query_profiles_to_appn_pkg:
	cp -r tests/local_vespa/additional_query_profiles/* tests/local_vespa/test_app/search/query-profiles/

# Set up a local instance of vespa
vespa_dev_setup:
    just vespa_confirm_cli_installed
    just copy_query_profiles_to_appn_pkg
    just vespa_dev_start
    just vespa_healthy
    just vespa_deploy_schema

# Stop and remove the vespa container
vespa_dev_down:
	docker compose -f tests/local_vespa/docker-compose.dev.yml down vespadaltest
