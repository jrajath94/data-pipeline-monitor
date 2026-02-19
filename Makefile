.PHONY: install test bench run lint clean help

help:
	@echo "Available targets:"
	@echo "  make install  - Install package and dependencies"
	@echo "  make test     - Run unit and integration tests"
	@echo "  make bench    - Run performance benchmarks"
	@echo "  make run      - Run example"
	@echo "  make lint     - Run linter and type checker"
	@echo "  make clean    - Remove build artifacts"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

bench:
	python benchmarks/bench_core.py

run:
	python examples/quickstart.py

lint:
	ruff check .
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .mypy_cache/ .ruff_cache/
