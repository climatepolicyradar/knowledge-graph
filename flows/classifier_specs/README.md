# Classifier Specs

These are used to mark which classifiers inference should run for in each environment.

They are what is looked up in wandb model registry, not to be confused with projects or s3 as the model id often varies between these three sources. The registry holds models that have been set as primary, and have links to the other two places.

To update this list after training run:

```shell
just update-inference-classifiers --aws-envs prod
```
