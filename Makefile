.PHONY: lint test

lint:
	ruff check src/engram

test:
	pytest
