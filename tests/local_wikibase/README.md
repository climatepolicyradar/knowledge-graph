# Local Wikibase

## Setup for tests

This folder has the config for a local wikibase instance that is used for testing.

## Reference

This was setup based on an adaption of the advice found [here](https://github.com/wmde/wikibase-release-pipeline/blob/main/deploy/README.md), and through adapting the [linked dockerfile](https://github.com/wmde/wikibase-release-pipeline/blob/main/deploy/docker-compose.yml). These adaptations where done to make it easier to use from a purely test perspective.

### What adaptations where made?

1. To remove the need to get an ssl certificate when spinning up the stack. Our use for this instance is very much a local & ci instance, being able to access it without needing a secure connection means we don't need to bother letsencrypt for certificates every run (which likely would have rate limited us).
2. Portable domain resolution. The [guidance on local setup](https://github.com/wmde/wikibase-release-pipeline/blob/main/deploy/README.md#can-i-host-wbs-deploy-locally) suggests making changes on the host machine to resolve the domains. But making it so we can skip this step makes it easier to spin up on different machines and contexts.
