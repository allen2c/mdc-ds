# Development
format:
	@isort . \
		--skip setup.py \
		--skip-glob '*/.venv/*' \
		--skip-glob '*/build/*' \
		--skip-glob '*/dist/*' \
		--skip-glob '*/__pycache__/*' \
		--skip-glob '*/docs/*' \
		--skip-glob '*/static/*' \
		--skip-glob '*/.conda/*'
	@black . \
		--exclude '/(setup\.py|\.venv|build|dist|__pycache__|docs|static|\.conda)/'

install:
	poetry install --all-extras --all-groups

update:
	poetry update
	poetry export --without-hashes -f requirements.txt --output requirements.txt
	poetry export --without-hashes -f requirements.txt --output requirements-all.txt --all-extras --all-groups

# Docs
mkdocs:
	mkdocs serve -a 0.0.0.0:8000

# Tests
pytest:
	python -m pytest
