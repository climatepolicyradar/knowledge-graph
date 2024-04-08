# Environment variables

Your .env file at the root of this repo should look something like this:

```bash
WIKIBASE_URL=

WIKIBASE_USERNAME=
WIKIBASE_PASSWORD=

WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID=
WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID=
WIKIBASE_RELATED_CONCEPT_PROPERTY_ID=
```

`WIKIBASE_URL` is the base URL for the Wikibase instance (including protocol). For the climate policy radar concepts store, this should be `https://climatepolicyradar.wikibase.cloud`.

`WIKIBASE_USERNAME` and `WIKIBASE_PASSWORD` are the credentials for the accessing the wikibase instance programatically. In most cases, you should use a set of bot account credentials (derived from your main account). See instructions for creating a bot user account [here](./bot-users.md).

`WIKIBASE_HAS_SUBCONCEPT_PROPERTY_ID`, `WIKIBASE_SUBCONCEPT_OF_PROPERTY_ID`, and `WIKIBASE_RELATED_CONCEPT_PROPERTY_ID` are the property IDs for the properties that define the relationships between concepts. These should be set to the IDs of the properties that you have created in your Wikibase instance. For the climate policy radar concepts store, these are `P1`, `P2`, and `P3` respectively.
