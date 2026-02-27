# Contributing to Engram

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/docaohieu2808/engram.git
cd engram
pip install -e ".[dev]"
```

## Development

```bash
# Lint
ruff check src/ tests/

# Test
pytest -q

# Type check (optional)
mypy src/engram/
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` — new feature
- `fix(scope): description` — bug fix
- `docs(scope): description` — documentation
- `refactor(scope): description` — code refactor
- `test(scope): description` — tests
- `chore(scope): description` — maintenance

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Add tests for new functionality
3. Ensure `ruff check` and `pytest` pass
4. Submit a PR with a clear description

## Code Style

- Python 3.11+
- Follow existing patterns in the codebase
- Keep files under 200 lines — split into modules when needed
- Use kebab-case for file names

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
