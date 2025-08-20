# Documentation

The documentation site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), giving us a simple way to write and maintain documentation in markdown, with a feature-rich site generated from it. Users are able to search and navigate the documentation easily, and the site is easy to maintain and update.

## Local development

To build the documentation, you should have installed dev dependencies with uv:

```sh
uv install --extra dev
```

Then you can serve the documentation locally with:

```sh
uv run mkdocs serve
```

To build the documentation, you can run:

```sh
uv run mkdocs build
```

## Deploying the documentation

This documentation is hosted on GitHub Pages, deployed with a github action on pushes to the `main` branch.
