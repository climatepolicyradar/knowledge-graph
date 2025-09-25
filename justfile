set dotenv-load
export WANDB_DIR := "./data/wandb"
import "tests/local_vespa/local_vespa.just"
import "tests/local_wikibase/local_wikibase.just"
import "scripts/scripts.just"

# Set the default command to list all available commands
default:
    @just --list

# install dependencies and set up the project
install +OPTS="":
    uv sync --locked --extra dev --extra coiled {{OPTS}}
    uv run pre-commit install
    uv run ipython kernel install --user

install-transformers:
  just install --extra transformers

# test the project
test +OPTS="":
    uv run pytest --disable-pytest-warnings --color=yes {{OPTS}}

test-concurrently +OPTS="":
    just test-without-vespa {{OPTS}}
    just test-with-vespa {{OPTS}}

# test the project, excluding tests that rely on a local Vespa instance
test-with-vespa +OPTS="":
    uv run pytest --disable-pytest-warnings --color=yes -m 'vespa' {{OPTS}}

# test the project, excluding tests that rely on a local Vespa instance
test-without-vespa +OPTS="":
    uv run pytest -n logical --disable-pytest-warnings --color=yes -m 'not vespa' {{OPTS}}

# update the snapshots for the tests
test-snapshot-update +OPTS="":
    uv run pytest --snapshot-update {{OPTS}}

# run linters and code formatters on relevant files
lint:
    uv run pre-commit run --show-diff-on-failure

# run linters and code formatters on all files
lint-all:
    uv run pre-commit run --all-files --show-diff-on-failure

build-image:
    docker build --progress=plain -t ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} .

run-image cmd="sh":
    docker run --rm -it ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} {{cmd}}

ecr-login:
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}

push-image:
    docker push ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION}

get-version:
    @grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

export-env-vars:
	export $(cat .env | xargs)

prefect-login: export-env-vars
	prefect cloud login -k ${PREFECT_API_KEY}

deploy: prefect-login
    just deploy-deployments
    just deploy-automations

deploy-deployments: prefect-login
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}
	python -m deployments

deploy-automations: prefect-login
	python -m automations

# Run a static site locally
serve-static-site tool:
    uv run python -m http.server -d static_sites/{{tool}}/dist 8080

# Generate a static site
generate-static-site tool:
    uv run python -m static_sites.{{tool}}

# Serve the concept store MCP locally
serve-mcp:
    uv run fastmcp run mcp/server.py:mcp --transport http --host 0.0.0.0 --port 8000

# Intended use is deploying to sandbox / staging for local testing
# Ensure that you have configured your .env and authenticated with aws & prefect
deploy-flows-from-local:
    echo building ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} in region: ${AWS_REGION}

    just ecr-login
    just build-image
    just push-image
    python -m deployments
    python -m automations
