# Contributing to BreezeML

Thank you for taking the time to contribute! BreezeML is an open-source project and we welcome improvements of all kinds — bug reports, feature suggestions, documentation edits, and code contributions.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Making a Pull Request](#making-a-pull-request)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)

---

## Code of Conduct

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before engaging with the community.

---

## How to Report a Bug

1. Check [existing issues](https://github.com/venomez-viper/breezeml/issues) to confirm the bug hasn't already been reported.
2. Open a new issue using the **Bug Report** template.
3. Include:
   - BreezeML version (`pip show breezeml`)
   - Python version and OS
   - Minimal reproducible example
   - Full error traceback

---

## How to Request a Feature

1. Check the [Roadmap](README.md#️-roadmap) and [open issues](https://github.com/venomez-viper/breezeml/issues) first.
2. Open a new issue using the **Feature Request** template.
3. Describe the use-case clearly — what problem does it solve and who benefits?

---

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/breezeml.git
cd breezeml

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify the setup
pytest examples/
```

---

## Making a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. Make your changes with clear, atomic commits.
3. Add or update tests in `examples/` covering your change.
4. Ensure all tests pass:
   ```bash
   pytest examples/
   ```
5. Push your branch and open a PR against `main`.
6. Fill out the PR template — link the related issue if applicable.

> **PRs that lack tests or break existing tests will not be merged.**

---

## Coding Standards

- **Style**: Follow [PEP 8](https://pep8.org/). Use 4-space indentation.
- **Docstrings**: All public functions must have a one-line summary docstring.
- **Type hints**: Preferred for all new public functions.
- **No magic numbers**: Use named constants or keyword arguments.
- **Backward compatibility**: Do not break the existing public API without a major version bump.

---

## Running Tests

```bash
# Run all example tests
pytest examples/ -v

# Run a specific test file
pytest examples/test_classifiers.py -v
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/venomez-viper/breezeml/discussions) or file an issue. We're happy to help.
