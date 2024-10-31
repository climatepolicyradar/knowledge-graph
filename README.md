# 🕸️ Knowledge graph

This repo comprises the infrastructure, tools, and scripts for managing Climate Policy Radar's concept store and knowledge graph.

Confused about those terms? See the [concept store vs knowledge graph](./docs/docs/developers/concept-store-vs-knowledge-graph.md) documentation for more information.

## Getting started

### As a developer

This repo is orchestrated with a [justfile](./justfile) (see [just](https://github.com/casey/just)) that wraps together a number of useful commands. Make sure you have `just` installed before you get started.

Next, you'll need to install the project specific dependencies:

```bash
just install
```

Then you can run the tests:

```bash
just test
```

Or fetch everything we know about a concept from the concept store and argilla:

```bash
just get-concept Q123
```

You can then train a model, after logging in (with `aws sso login --profile labs`):

```bash
just train Q992
```

Or to also track in W&B and upload to S3:

```bash
just train Q992 --track --upload --aws-env labs
```

Afterwards, evaluate the trained model:

```bash
just evaluate Q992
```

Or to also track in W&B:

```bash
just evaluate Q992 --track
```

You can promote a model version from one AWS account/environment, to another. You can optionally promote that model to be the primary version that's used in that account.

```bash
poetry run promote Q992 --classifier RulesBasedClassifier --version v13 --from-aws-env labs --to-aws-env staging --primary
```

_or_

```bash
just promote Q992 --classifier RulesBasedClassifier --version v7 --within-aws-env staging --no-primary
```

You can see the full list of commands by running:

```bash
just --list
```

See the [docs](./docs) for more information on how to work with the knowledge graph.

### As an Editor

You can also explore and edit the graph directly through UI like the concept store. We've documented the process of getting started with the concept store and a style guide for how to structure the data in the [concept store documentation in notion](https://www.notion.so/climatepolicyradar/Concept-store-documentation-54b91a8359664cb3a9bbe3989efb7ca0?pvs=4).
