# Bring up the local wikibase stack
up-local-wikibase:
	docker compose --file tests/local_wikibase/docker-compose.dev.yml up --detach --wait

# Bring down the local wikibase stack
down-local-wikibase:
	docker compose --file tests/local_wikibase/docker-compose.dev.yml down
