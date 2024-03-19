.PHONY: install start-ec2 stop-ec2

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user
