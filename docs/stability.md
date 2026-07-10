# API stability & deprecation policy (2.0)

BreezeML 2.0 makes the public surface stable and typed.

## What "public" means

The public API is everything exported from the top-level `breezeml` package
(see `breezeml.__all__`) plus the documented methods of the `Model` object
(`fit(...)` return value). Anything prefixed with `_`, and anything under a
module not re-exported at the top level, is internal and may change without
notice.

## Typing

The package ships a `py.typed` marker (PEP 561), so type checkers (mypy,
pyright) read BreezeML's annotations directly. The public entry points
(`fit`, `predict`, `auto`, `report`, the `Model` methods) are annotated.

## Semantic versioning

- **MAJOR** (e.g. 1.x -> 2.0): breaking changes to the public API.
- **MINOR** (2.0 -> 2.1): new features, backward compatible.
- **PATCH** (2.0.0 -> 2.0.1): bug fixes only.

## Deprecation policy

A public API is only removed one MAJOR version after it is first deprecated.
Deprecations emit a `DeprecationWarning` and are listed in the CHANGELOG.

## Breaking changes in 2.0

- `report(model, df)` now returns a `Report` (the honest scorecard with a
  SHIP/WARN/STOP verdict), not a bare metrics dict. Use `model.evaluate(df)`
  for a plain metrics dict.
- The model object is now named `Model` (with `EasyModel` kept as an alias, so
  existing pickles and imports keep working).

## The sacred constraints (never broken)

- Four core dependencies: scikit-learn, pandas, numpy, joblib.
- Zero lock-in: `export()` always emits standalone scikit-learn code.
- Beginner-first: `fit(df, target)` stays a one-liner.
