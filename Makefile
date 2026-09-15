.PHONY: install test-fixtures test lint lint-fix typecheck run docker-build docker-run

install:
	uv sync --extra dev

test-fixtures:
	uv run python tests/fixtures/generate.py

test:
	uv run pytest -q --tb=short --cov=src/dipdca --cov-report=term-missing

typecheck:
	uv run mypy src/dipdca

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

run:
	uv run streamlit run app.py

docker-build:
	docker build -t dip-or-dca .

docker-run:
	docker run -p 8501:8501 dip-or-dca
