.PHONY: install

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

upload-gst-data:
	poetry run python scripts/upload_gst_data.py
