# neo4j

[Neo4j](https://neo4j.com/) is a graph database. We can use Neo4j to store our knowledge graph, with documents, passages and concepts represented as nodes, and edges between them representing their relationships.

In this repo, the neo4j instance is orchestrated by docker-compose. To start the neo4j instance, run the following command:

```bash
docker-compose up --build -d neo4j
```

When it's running, you can populate the database with the following command:

```bash
poetry run python scripts/neo4j/populate.py
```

The script will fetch all concepts from wikibase and add them (and their relationships) to the knowledge graph. It will then use the predicted passages in the `data/processed/predictions` directory to create the relationships between each concept and the passages where they're mentioned, and between each passage and the documents where they're found.

Rather than writing raw cypher, the script uses [neomodel](https://neomodel.readthedocs.io/en/stable/) to handle interactions with the database, with `Concept`, `Passage`, and `Document` nodes defined as python classes in `src/neo4j/models.py`. You can also use those classes to query the database in subsequent scripts/notebooks.
