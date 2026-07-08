# Recommenders: what should this user see next?

`breezeml.recommend` answers the recommendation question with collaborative
filtering on the four core dependencies. It turns a long-format interactions
table (one row per user-item event) into a user-item matrix, factorizes it with
a truncated SVD, and ranks the items a user has NOT yet touched by their
reconstructed score. Implicit feedback and explicit ratings are both supported,
and, in keeping with the rest of BreezeML, the moment it cannot personalize it
says so out loud instead of dressing popularity up as insight.

```python
from breezeml import recommend

# Explicit ratings
rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars")

# Implicit feedback (any interaction counts as 1) - just omit rating_col
rec = recommend.collaborative_filter(df, "user", "item")

rec.recommend("u_7", k=5)          # -> [(item, score), ...]
rec.recommend_report("u_7", k=5)   # -> dict with cold_start / method flags
```

`collaborative_filter()` is the one-call convenience wrapper; it builds and
returns a fitted `Recommender`. You can also construct `Recommender(...)` and
call `.fit(df, user_col, item_col, rating_col=...)` yourself.

## How it works

At fit time the interactions are pivoted into a user-item matrix. Duplicate
user-item pairs are collapsed (max for implicit presence, mean for explicit
ratings). The matrix is factorized with a `TruncatedSVD`, and the number of
latent factors is **clipped to the matrix rank** (`min(n_users, n_items) - 1`)
so the SVD never asks for more components than the data can support. The
reconstructed matrix supplies the affinity scores.

`recommend(user, k)` ranks the items a known user has not yet interacted with
by their reconstructed score; already-seen items are masked out so they are
never re-recommended. If there are not enough unseen items to fill `k`, the
list is padded with globally popular items.

`recommend_report(user, k)` returns the same recommendations plus honest
metadata: `cold_start` (bool), `method` (`"svd"` or `"popularity"`), `user`,
`k`, and a `note` explaining what the scores mean. For a known user the scores
are reconstructed affinities (not probabilities); for a cold-start user they
are raw interaction counts.

## The honesty report

`fit()` prints a report (set `show=False` to silence it) that refuses to hide
the failure modes of collaborative filtering:

- **Extremely sparse matrices** (density below 1%) get a warning that the
  latent factors will be noisy and recommendations may be little better than
  popularity.
- **Users with a single interaction** are flagged: they cannot be personalized
  reliably.
- **Items with a single interaction** are flagged: their factors rest on one
  data point.

The `stats` dict on the fitted model carries all of this: `n_users`,
`n_items`, `n_interactions`, `density`, `n_components_used`, `implicit`,
`users_with_one_interaction`, and `items_with_one_interaction`.

## Cold start is popularity, and it says so

A brand-new user has no row in the training matrix, so there are no factors to
personalize from. Rather than invent a preference, the recommender falls back
to the globally most popular items and `recommend_report` sets `cold_start:
True`, `method: "popularity"`, and a note saying the list is not personalized
and the scores are interaction counts. Honest generic beats fake personal.

## When NOT to use it

- **It is pure collaborative filtering.** It knows nothing about item or user
  side features (genre, price, text, demographics). If your signal lives in
  content, use a content-based or hybrid model instead.
- **Cold start is popularity-only.** A brand-new user or item gets the same
  globally popular list until they interact. If most of your traffic is new
  users, this will feel generic, and no amount of factors will fix that.
- **It needs enough interactions per user to personalize.** Users or items with
  a single interaction cannot be placed in latent space with any confidence;
  the fit warns about them, but it cannot invent history that is not there.
- **Scores are affinities, not probabilities.** Do not threshold them as if
  they were calibrated likelihoods; use them for ranking within a user.
