"""Generate the FICTIONAL A/B test dataset used in this demo.

Story: "Northwind Outfitters" (a made-up online shop) tests a new one-page
checkout (B) against the current three-step checkout (A) for 14 days.
Nothing here comes from a real company.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261113)

N = 41_800
start = pd.Timestamp("2026-10-05")

variant = rng.choice(["A", "B"], size=N, p=[0.5, 0.5])
device = rng.choice(["mobile", "desktop"], size=N, p=[0.64, 0.36])
visitor = rng.choice(["new", "returning"], size=N, p=[0.58, 0.42])
day = rng.integers(0, 14, size=N)

# Baseline conversion by segment (fictional)
base = np.where(device == "mobile", 0.026, 0.047)
base = base * np.where(visitor == "returning", 1.35, 0.80)

# True effect of the new checkout: helps mobile, roughly neutral on desktop
lift = np.where(variant == "B", np.where(device == "mobile", 0.21, 0.01), 0.0)
# Novelty effect: a little extra lift in the first 3 days only
novelty = np.where((variant == "B") & (day < 3), 0.12, 0.0)
p = base * (1 + lift + novelty)
converted = rng.random(N) < p

# Order value: lognormal; one-page checkout shows fewer add-on suggestions,
# so order value is slightly lower in B
aov_mu = np.log(np.where(device == "mobile", 58.0, 74.0))
aov_mu = aov_mu - np.where(variant == "B", 0.04, 0.0)
revenue = np.where(converted, rng.lognormal(aov_mu, 0.55), 0.0).round(2)

df = pd.DataFrame(
    {
        "user_id": np.arange(100001, 100001 + N),
        "date": (start + pd.to_timedelta(day, unit="D")).date,
        "variant": variant,
        "device": device,
        "visitor_type": visitor,
        "converted": converted.astype(int),
        "revenue_eur": revenue,
    }
).sort_values(["date", "user_id"])

df.to_csv("data/checkout_test.csv", index=False)
print(df.shape, df.groupby("variant")[["converted", "revenue_eur"]].mean())
