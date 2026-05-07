# Contributing to BreezeML

Thanks for contributing to BreezeML.

This project welcomes:

- bug reports
- feature requests
- documentation improvements
- tests
- performance improvements
- new model integrations

---

## Code of Conduct

By participating in this project, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

```bash
git clone https://github.com/<your-username>/breezeml.git
cd breezeml

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

Optional extras for broader local testing:

```bash
pip install -e ".[boost,datasets]"
```

---

## Running Checks

```bash
ruff check .
pytest tests/ -v
```

Examples can also be run manually from `examples/` when you want a quick smoke test.

---

## Pull Request Expectations

Please try to keep pull requests focused and reviewable.

All PRs should:

- include tests for new behavior when practical
- avoid breaking the current public API unless intentionally planned
- update docs when the public surface changes
- keep commit messages clear and specific

---

## Documentation

The public docs are maintained in:

- [README.md](README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md) and earlier release-note files
- [`docs/`](docs/) for the MkDocs site

If you add a new public feature, update the README and relevant docs pages in the same PR.

---

## Release Workflow

For a release branch or final release prep, the normal checklist is:

```bash
ruff check .
pytest tests/ -v
git add .
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push
git push origin vX.Y.Z
```

---

## Need Help?

Open an issue or discussion on GitHub:

- Issues: https://github.com/venomez-viper/breezeml/issues
- Repository: https://github.com/venomez-viper/breezeml
