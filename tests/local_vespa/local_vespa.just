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
	@vespa deploy tests/local_vespa/test_app --wait 300

# Loads some test data into local vespa instance
vespa_load_data:
	vespa config set target local
	vespa feed tests/local_vespa/test_documents/search_weights.json
	vespa feed tests/local_vespa/test_documents/family_document.json
	vespa feed tests/local_vespa/test_documents/document_passage.json

# Set up a local instance of vespa
vespa_dev_setup: 
    just vespa_confirm_cli_installed
    just vespa_dev_start
    just vespa_healthy
    just vespa_deploy_schema
    just vespa_load_data

# Stop and remove the vespa container
vespa_dev_down:
	docker compose -f tests/local_vespa/docker-compose.dev.yml down vespadaltest
