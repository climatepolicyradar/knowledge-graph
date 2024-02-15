.PHONY: install start-ec2 stop-ec2 export-ec2

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

start-ec2:
	poetry run pulumi up --cwd infra; \
	export $$(poetry run pulumi stack output --shell --cwd infra)

stop-ec2:
	poetry run pulumi destroy --cwd infra

