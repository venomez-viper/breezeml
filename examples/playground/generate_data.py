"""Generate the BreezeML Playground: synthetic datasets for teaching honest ML.

Every file is fully synthetic, generated deterministically by this script
(seed 7). No real people, no licensing concerns. Each dataset is designed
to exercise one part of an honest ML workflow: leakage traps, class
imbalance, forecasting baselines, conformal coverage, fairness gaps,
censored durations, confounded treatments, latent clusters, and drift.

Run:  python generate_data.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / name, index=False)
    print(f"{name:28s} {len(df):>6,} rows  {df.shape[1]:>2} cols")


# ------------------------------------------------------------------ churn
def churn():
    n = 5000
    tenure = RNG.gamma(2.0, 14, n).clip(1, 72).round()
    charges = RNG.normal(65, 22, n).clip(18, 140).round(2)
    tickets = RNG.poisson(1.1 + 2.0 * (charges > 90), n)
    contract = RNG.choice(["monthly", "one_year", "two_year"], n, p=[.55, .28, .17])
    payment = RNG.choice(["card", "bank_transfer", "check"], n, p=[.5, .3, .2])
    last_login = RNG.gamma(1.5, 6, n).round().clip(0, 90)

    risk = (
        1.6 * (contract == "monthly")
        - 0.12 * tenure
        + 0.045 * (charges - 65)
        + 0.85 * tickets
        + 0.07 * last_login
    )
    logit = -4.6 + 1.6 * risk
    churn_flag = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    base = pd.DataFrame({
        "tenure_months": tenure.astype(int),
        "monthly_charges": charges,
        "support_tickets": tickets,
        "contract_type": contract,
        "payment_method": payment,
        "days_since_login": last_login.astype(int),
        "churned": churn_flag,
    })

    # clean version
    save(base, "churn.csv")

    # raw version with the classic traps left in
    raw = base.copy()
    raw.insert(0, "customer_id", np.arange(100001, 100001 + n))
    # post-outcome column: only exists for churned customers (leakage)
    days_to_cancel = np.where(churn_flag == 1, RNG.gamma(2, 9, n).round(), np.nan)
    raw["days_until_cancellation"] = days_to_cancel
    save(raw, "churn_raw.csv")

    # next-quarter batch with drift: price rise, new contract type, longer logins
    m = 2000
    idx = RNG.choice(n, m, replace=False)
    nq = base.iloc[idx].copy().reset_index(drop=True)
    nq["monthly_charges"] = (nq["monthly_charges"] * RNG.normal(1.25, 0.05, m)).clip(18, 220).round(2)
    nq["days_since_login"] = (nq["days_since_login"] * RNG.normal(1.6, 0.2, m)).round().clip(0, 180).astype(int)
    new_contract = RNG.random(m) < 0.15
    nq.loc[new_contract, "contract_type"] = "flex"
    nq = nq.drop(columns=["churned"])
    save(nq, "churn_next_quarter.csv")


# ----------------------------------------------------------- house prices
def houses():
    n = 4000
    sqft = RNG.gamma(6, 300, n).clip(450, 6500).round()
    beds = np.clip((sqft / 850 + RNG.normal(0, 0.7, n)).round(), 1, 7)
    baths = np.clip((beds * 0.7 + RNG.normal(0, 0.5, n)).round(1), 1, 5)
    age = RNG.gamma(2, 12, n).clip(0, 110).round()
    dist = RNG.gamma(2.2, 5, n).clip(0.3, 45).round(1)
    tier = RNG.choice(["standard", "good", "premium"], n, p=[.5, .35, .15])
    garage = (RNG.random(n) < 0.62).astype(int)

    price = (
        40_000
        + 145 * sqft
        + 9_000 * beds
        + 12_000 * baths
        - 750 * age
        - 2_600 * dist
        + np.select([tier == "good", tier == "premium"], [28_000, 90_000], 0)
        + 14_000 * garage
    )
    noise = RNG.normal(0, 0.09 * price)  # heteroscedastic: pricier = noisier
    price = (price + noise).clip(45_000).round(-2)

    save(pd.DataFrame({
        "sqft": sqft.astype(int), "bedrooms": beds.astype(int), "bathrooms": baths,
        "age_years": age.astype(int), "distance_center_km": dist,
        "neighborhood_tier": tier, "garage": garage, "price": price,
    }), "house_prices.csv")


# ------------------------------------------------------------ store sales
def sales():
    days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    t = np.arange(len(days))
    weekly = 1 + 0.35 * np.isin(days.dayofweek, [4, 5]) - 0.15 * (days.dayofweek == 0)
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (days.dayofyear - 320) / 365.25)
    trend = 1 + 0.00045 * t
    promo = (RNG.random(len(days)) < 0.07).astype(int)
    base = 1800 * weekly * yearly * trend * (1 + 0.45 * promo)
    sales_v = (base * RNG.normal(1, 0.08, len(days))).round()

    save(pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "promo": promo,
        "sales": sales_v.astype(int),
    }), "store_sales.csv")


# ---------------------------------------------------------- loan approval
def loans():
    n = 6000
    gender = RNG.choice(["male", "female"], n)
    income = RNG.lognormal(10.9, 0.45, n).clip(18_000, 400_000).round(-2)
    credit = RNG.normal(680, 75, n).clip(300, 850).round()
    debt_ratio = RNG.beta(2.2, 5, n).round(3)
    emp_years = RNG.gamma(2, 3.5, n).clip(0, 40).round(1)
    amount = (income * RNG.uniform(0.1, 0.6, n)).round(-2)

    logit = (
        -1.0
        + 0.9 * (credit - 680) / 75
        + 0.55 * (np.log(income) - 10.9) / 0.45
        - 2.2 * (debt_ratio - 0.3)
        + 0.05 * emp_years
        - 1.0 * (gender == "female")  # injected bias, the point of the exercise
    )
    approved = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    save(pd.DataFrame({
        "income": income, "credit_score": credit.astype(int),
        "debt_ratio": debt_ratio, "employment_years": emp_years,
        "loan_amount": amount, "gender": gender, "approved": approved,
    }), "loan_approvals.csv")


# ------------------------------------------------------- patient survival
def survival():
    n = 2000
    group = RNG.choice(["standard", "new_drug"], n)
    age = RNG.normal(62, 11, n).clip(30, 92).round()
    stage = RNG.choice([1, 2, 3, 4], n, p=[.25, .35, .25, .15])
    scale = 34 * np.exp(0.45 * (group == "new_drug") - 0.28 * (stage - 2) - 0.012 * (age - 62))
    true_time = RNG.exponential(scale)
    censor_time = RNG.uniform(6, 60, n)
    duration = np.minimum(true_time, censor_time).round(1).clip(0.1)
    event = (true_time <= censor_time).astype(int)

    save(pd.DataFrame({
        "group": group, "age": age.astype(int), "stage": stage,
        "duration_months": duration, "event": event,
    }), "patient_survival.csv")


# ------------------------------------------------------------ ab campaign
def campaign():
    n = 8000
    engagement = RNG.beta(2.5, 3.5, n).round(3)
    prior = RNG.poisson(3 * engagement + 0.5, n)
    age = RNG.normal(38, 12, n).clip(18, 80).round()
    # confounding: engaged customers are more likely to receive the campaign
    treated = (RNG.random(n) < (0.15 + 0.55 * engagement)).astype(int)
    spend = (
        22
        + 60 * engagement
        + 6.5 * prior
        + 12.0 * treated          # true effect ~ +12
        + RNG.normal(0, 14, n)
    ).clip(0).round(2)

    save(pd.DataFrame({
        "age": age.astype(int), "engagement_score": engagement,
        "prior_purchases": prior, "treated": treated, "spend_next_month": spend,
    }), "ab_campaign.csv")


# ------------------------------------------------------ customer segments
def segments():
    centers = np.array([
        # groceries, dining, travel, electronics, fashion
        [520, 60, 30, 40, 55],     # homebodies
        [260, 340, 90, 70, 130],   # social spenders
        [180, 120, 620, 150, 90],  # travelers
        [200, 90, 60, 540, 70],    # tech enthusiasts
    ])
    sizes = [1050, 800, 550, 550]
    rows = []
    for c, s in zip(centers, sizes):
        rows.append(c * RNG.lognormal(0, 0.28, (s, 5)))
    X = np.vstack(rows)
    # 2% anomalies: extreme uniform spenders
    n_anom = 60
    anomalies = RNG.uniform(900, 2500, (n_anom, 5))
    X = np.vstack([X, anomalies])
    RNG.shuffle(X)

    save(pd.DataFrame(
        X.round(2),
        columns=["groceries", "dining", "travel", "electronics", "fashion"],
    ), "customer_segments.csv")


if __name__ == "__main__":
    churn()
    houses()
    sales()
    loans()
    survival()
    campaign()
    segments()
    print(f"\nAll files written to {OUT}")
