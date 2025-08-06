# Justfile

Start by installing the dependencies:

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

You can promote a model to be the primary used in that environment using its ClassifierID.

```bash
poetry run promote Q992 --classifier_id abcd2345  --aws-env labs --primary
```

You can also demote (aka disable) a promoted model in an AWS account/environment, for a concept.

```bash
just demote Q787 labs
```

You can see the full list of commands by running:

See the [docs](./docs) for more information on how to work with the knowledge graph.
