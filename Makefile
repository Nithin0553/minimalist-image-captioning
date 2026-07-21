.PHONY: sync format format-check check tests prod

sync:
	uv sync --extra dev

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

check:
	uv run ruff check .
	uv run ty check
	uv run basedpyright

tests:
	uv run pytest --cov=minimal_captioning --cov-branch --cov-report=term-missing

prod: format-check check tests
