.PHONY: install test lint run docker-build fixtures

install:
	uv sync --extra dev

fixtures:
	uv run python data/fixtures/generate.py

test:
	uv run pytest -q

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
