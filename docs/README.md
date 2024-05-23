# Documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/), giving us a simple way to write and maintain documentation in markdown, with a feature-rich site generated from it. Users are able to search and navigate the documentation easily, and the site is easy to maintain and update.

## Local development

To build the documentation, you should have installed dev dependencies with poetry:

```sh
poetry install --with dev
```

Then you can serve the documentation locally with:

```sh
poetry run mkdocs serve
```

To build the documentation, you can run:

```sh
poetry run mkdocs build
```

## Deploying the documentation

This documentation is hosted on GitHub Pages, deployed with a github action on pushes to the `main` branch.
