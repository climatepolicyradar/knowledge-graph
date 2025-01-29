# Concept Librarian

A web-based tool that helps concept developers maintain the quality and consistency of our [concept store](https://climatepolicyradar.wikibase.cloud).

The tool checks for issues with concepts in the concept store and their relationships to one another, and generates a static website which:

- Lists all detected issues, with a few utilities for filtering and sorting
- Provides individual pages for each concept, with a comprehensive view of their issues
- Makes it easy to navigate to Wikibase to make each fix

The site is hosted on AWS S3 and is available at:

```
http://concept-librarian.s3-website.eu-west-2.amazonaws.com/
```

## Usage

### Generating the Report

To check for issues and generate a new set of HTML files, run:

```bash
poetry run python -m concept_librarian
```

The HTML files will be saved to the `data/build/concept_librarian` directory.

To preview the report locally, open the `index.html` file in your browser:

```bash
open data/build/concept_librarian/index.html
```

### Deploying Updates

To deploy the latest version of the report to S3:

```bash
aws s3 sync data/build/concept_librarian s3://concept-librarian --profile=labs --delete
```

The `--delete` flag will remove any files from S3 that are not present in the local build directory, keeping the hosted site in sync with your local build.
