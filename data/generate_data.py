"""
Generates a REALISTIC, MESSY raw A/B test dataset for a checkout-page
redesign experiment (Control = old checkout, Treatment = new one-page checkout).

This is deliberately NOT a clean toy dataset. It contains the kinds of
problems a real analyst has to catch and handle:
    - duplicate user_id rows (double-logging from the event pipeline)
    - missing revenue values for users who converted
    - a small amount of bot / outlier traffic (absurd revenue spikes)
    - a mild sample ratio mismatch (allocation isn't a perfect 50/50)
    - a few users who appear in BOTH groups (contamination / crossover)
    - realistic covariates (device, country, day) that must be checked
      for balance across arms
    - timestamps spanning a 21-day experiment window

Run:  python generate_data.py
Output: raw_ab_test_data.csv  (~25,000 rows, realistic scale)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_CONTROL = 12600
N_TREATMENT = 12950          # slight SRM on purpose -> must be checked, not fatal
DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_P = [0.62, 0.33, 0.05]
COUNTRIES = ["IN", "US", "UK", "DE", "AE"]
COUNTRY_P = [0.45, 0.25, 0.12, 0.10, 0.08]

TRUE_CONV_CONTROL = 0.084     # baseline checkout conversion
TRUE_CONV_TREATMENT = 0.094   # real ~1pp lift from the new checkout flow
AOV_MEAN = 1450.0             # INR-ish average order value
AOV_SIGMA = 0.55              # lognormal shape -> realistic right-skewed spend

START = datetime(2026, 6, 1)


def make_group(n, group_name, conv_rate):
    device = RNG.choice(DEVICES, size=n, p=DEVICE_P)
    country = RNG.choice(COUNTRIES, size=n, p=COUNTRY_P)
    day_offset = RNG.integers(0, 21, size=n)
    ts = [START + timedelta(days=int(d), seconds=int(RNG.integers(0, 86400))) for d in day_offset]
    converted = RNG.binomial(1, conv_rate, size=n)

    revenue = np.zeros(n)
    converting_idx = np.where(converted == 1)[0]
    revenue[converting_idx] = RNG.lognormal(mean=np.log(AOV_MEAN), sigma=AOV_SIGMA, size=len(converting_idx))

    df = pd.DataFrame({
        "user_id": [f"{group_name[0]}_{i:06d}" for i in range(n)],
        "group": group_name,
        "device": device,
        "country": country,
        "timestamp": ts,
        "converted": converted,
        "revenue": revenue,
    })
    return df


def inject_realistic_mess(df):
    df = df.copy()

    # 1. Missing revenue for a small % of converters (tracking pixel dropped)
    conv_idx = df.index[df["converted"] == 1]
    missing_rev_idx = RNG.choice(conv_idx, size=int(0.015 * len(conv_idx)), replace=False)
    df.loc[missing_rev_idx, "revenue"] = np.nan

    # 2. Bot / outlier traffic: implausible revenue spikes (fraud / scraper checkouts)
    bot_idx = RNG.choice(df.index, size=max(1, int(0.003 * len(df))), replace=False)
    df.loc[bot_idx, "revenue"] = RNG.uniform(150000, 500000, size=len(bot_idx))
    df.loc[bot_idx, "converted"] = 1

    # 3. Duplicate rows (pipeline double-fired event)
    dup_sample = df.sample(n=int(0.008 * len(df)), random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # 4. Crossover contamination: a handful of users show up in the OTHER group too
    cross_sample = df.sample(n=int(0.002 * len(df)), random_state=2).copy()
    cross_sample["group"] = cross_sample["group"].map({"control": "treatment", "treatment": "control"})
    cross_sample["user_id"] = cross_sample["user_id"] + "_dup"
    df = pd.concat([df, cross_sample], ignore_index=True)

    # 5. A few negative/garbage revenue entries (data entry error)
    garbage_idx = RNG.choice(df.index, size=3, replace=False)
    df.loc[garbage_idx, "revenue"] = -999

    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def main():
    control = make_group(N_CONTROL, "control", TRUE_CONV_CONTROL)
    treatment = make_group(N_TREATMENT, "treatment", TRUE_CONV_TREATMENT)
    raw = pd.concat([control, treatment], ignore_index=True)
    raw = inject_realistic_mess(raw)
    out_path = "raw_ab_test_data.csv"
    raw.to_csv(out_path, index=False)
    print(f"Wrote {len(raw):,} rows to {out_path}")
    print(raw["group"].value_counts())


if __name__ == "__main__":
    main()
