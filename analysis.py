"""A/B test analysis: one-page checkout (B) vs three-step checkout (A).

Run:  python analysis.py
Output: report/results.json, report/*.png, report/report.html
Data is FICTIONAL (see make_data.py).
"""
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

OUT = Path("report")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
df = pd.read_csv("data/checkout_test.csv", parse_dates=["date"])
A, B = df[df.variant == "A"], df[df.variant == "B"]
R = {}

# 1. Sample ratio mismatch
counts = df.variant.value_counts().reindex(["A", "B"])
chi2, p_srm = stats.chisquare(counts, [len(df) / 2] * 2)
R["srm"] = {"n_A": int(counts.A), "n_B": int(counts.B), "p": float(p_srm)}

# 2. Conversion rate
x = np.array([A.converted.sum(), B.converted.sum()])
n = np.array([len(A), len(B)])
z, p_conv = proportions_ztest(x, n)
lo, hi = confint_proportions_2indep(x[1], n[1], x[0], n[0], method="newcomb")
cr = x / n
R["conversion"] = {
    "cr_A": cr[0], "cr_B": cr[1],
    "abs_diff": cr[1] - cr[0], "abs_ci": [lo, hi],
    "rel_lift": cr[1] / cr[0] - 1, "rel_ci": [lo / cr[0], hi / cr[0]],
    "p": float(p_conv),
}


# 3. Revenue per visitor (many zeros, skewed) -> bootstrap + Welch t
def boot_diff(a, b, stat=np.mean, B_=4000):
    a, b = np.asarray(a), np.asarray(b)
    d = np.empty(B_)
    for i in range(B_):
        d[i] = stat(rng.choice(b, b.size)) - stat(rng.choice(a, a.size))
    return np.percentile(d, [2.5, 97.5])


rpv = (A.revenue_eur.mean(), B.revenue_eur.mean())
rpv_ci = boot_diff(A.revenue_eur, B.revenue_eur)
_, p_rpv = stats.ttest_ind(B.revenue_eur, A.revenue_eur, equal_var=False)
R["revenue_per_visitor"] = {
    "A": rpv[0], "B": rpv[1], "diff": rpv[1] - rpv[0],
    "ci": list(rpv_ci), "rel_lift": rpv[1] / rpv[0] - 1, "p": float(p_rpv),
}

# 4. Average order value among buyers
oa, ob = A[A.converted == 1].revenue_eur, B[B.converted == 1].revenue_eur
aov_ci = boot_diff(oa, ob)
_, p_aov = stats.mannwhitneyu(ob, oa)
R["aov"] = {"A": oa.mean(), "B": ob.mean(), "diff": ob.mean() - oa.mean(),
            "ci": list(aov_ci), "p": float(p_aov)}

# 5. Segments, with Holm correction, plus a formal interaction test
seg_rows = []
for col in ["device", "visitor_type"]:
    for lvl, g in df.groupby(col):
        ga, gb = g[g.variant == "A"], g[g.variant == "B"]
        xs = np.array([ga.converted.sum(), gb.converted.sum()])
        ns = np.array([len(ga), len(gb)])
        _, ps = proportions_ztest(xs, ns)
        l, h = confint_proportions_2indep(xs[1], ns[1], xs[0], ns[0], method="newcomb")
        base_ = xs[0] / ns[0]
        seg_rows.append({"segment": f"{col}: {lvl}", "n": int(ns.sum()),
                         "cr_A": base_, "cr_B": xs[1] / ns[1],
                         "rel_lift": xs[1] / ns[1] / base_ - 1,
                         "rel_ci": [l / base_, h / base_], "p_raw": float(ps)})
holm = multipletests([r["p_raw"] for r in seg_rows], method="holm")[1]
for r, ph in zip(seg_rows, holm):
    r["p_holm"] = float(ph)
R["segments"] = seg_rows
m = smf.logit("converted ~ C(variant) * C(device) + C(visitor_type)", df).fit(disp=0)
R["interaction_p_device"] = float(m.pvalues["C(variant)[T.B]:C(device)[T.mobile]"])

# 6. Novelty: first 3 days vs the rest, and cumulative lift by day
first = df.date < df.date.min() + pd.Timedelta(days=3)
nov = {}
for name, g in [("days_1_3", df[first]), ("days_4_14", df[~first])]:
    c = g.groupby("variant").converted.mean()
    nov[name] = {"cr_A": c.A, "cr_B": c.B, "rel_lift": c.B / c.A - 1}
R["novelty"] = nov
cum = []
for d in sorted(df.date.unique()):
    g = df[df.date <= d]
    xs = g.groupby("variant").converted.sum().reindex(["A", "B"]).values
    ns = g.groupby("variant").size().reindex(["A", "B"]).values
    _, pc = proportions_ztest(xs, ns)
    l, h = confint_proportions_2indep(xs[1], ns[1], xs[0], ns[0], method="newcomb")
    b0 = xs[0] / ns[0]
    cum.append({"date": str(pd.Timestamp(d).date()), "rel_lift": xs[1] / ns[1] / b0 - 1,
                "lo": l / b0, "hi": h / b0, "p": float(pc)})
R["cumulative"] = cum

# 7. Power: smallest relative lift detectable with this sample (80% power)
pw = NormalIndPower()
es = pw.solve_power(nobs1=n[0], alpha=0.05, power=0.8, ratio=n[1] / n[0])
# convert Cohen's h back to a conversion rate above baseline
p0 = cr[0]
p1 = np.sin(np.arcsin(np.sqrt(p0)) + es / 2) ** 2
R["power"] = {"mde_rel": p1 / p0 - 1, "mde_abs": p1 - p0}

(OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))

# ---------- Charts ----------
INK, MUTED, A_C, B_C, OK = "#1f2a37", "#6b7280", "#9aa5b1", "#0f766e", "#0f766e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cbd2d9", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

# Chart 1: conversion with 95% CI
fig, ax = plt.subplots(figsize=(6.4, 4))
cis = [proportion_confint(x[i], n[i], method="wilson") for i in range(2)]
ax.bar(["A  current checkout", "B  one-page checkout"], cr * 100, color=[A_C, B_C], width=0.55)
for i in range(2):
    ax.errorbar(i, cr[i] * 100, yerr=[[(cr[i] - cis[i][0]) * 100], [(cis[i][1] - cr[i]) * 100]],
                color=INK, capsize=6, lw=1.4)
    ax.text(i, cis[i][1] * 100 + 0.12, f"{cr[i]*100:.2f}%", ha="center", color=INK, fontweight="bold")
ax.set_ylabel("Conversion rate (%)")
ax.set_ylim(0, max(cr) * 100 * 1.3)
ax.set_title("Conversion rate, 95% confidence interval", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "conversion.png", dpi=200)
plt.close(fig)

# Chart 2: lift by segment (forest plot)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
rows = [("All visitors", R["conversion"]["rel_lift"], R["conversion"]["rel_ci"], R["conversion"]["p"])]
rows += [(r["segment"].split(": ")[1].capitalize(), r["rel_lift"], r["rel_ci"], r["p_holm"]) for r in seg_rows]
for i, (lab, v, ci, pv) in enumerate(rows[::-1]):
    col = B_C if pv < 0.05 else MUTED
    ax.plot([ci[0] * 100, ci[1] * 100], [i, i], color=col, lw=2.2)
    ax.plot(v * 100, i, "o", color=col, ms=8)
    ax.text(ci[1] * 100 + 1.5, i, f"{v*100:+.0f}%", va="center", color=col, fontweight="bold")
ax.axvline(0, color="#cbd2d9", lw=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlabel("Relative lift in conversion, B vs A (%)  ·  green = significant after Holm correction", fontsize=9)
ax.set_title("Where the new checkout helps", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "segments.png", dpi=200)
plt.close(fig)

# Chart 3: cumulative lift over time
cd = pd.DataFrame(cum)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xd = np.arange(1, len(cd) + 1)
ax.fill_between(xd, cd.lo * 100, cd.hi * 100, color=B_C, alpha=0.15, lw=0)
ax.plot(xd, cd.rel_lift * 100, color=B_C, lw=2.2, marker="o", ms=4)
ax.axhline(0, color="#cbd2d9", lw=1)
ax.set_xlabel("Day of test")
ax.set_ylabel("Cumulative lift (%)")
ax.set_xticks(xd)
ax.set_title("Early results overstate the lift", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "cumulative.png", dpi=200)
plt.close(fig)

print(json.dumps({k: R[k] for k in ["srm", "conversion", "revenue_per_visitor", "aov",
                                    "interaction_p_device", "novelty", "power"]},
                 indent=1, default=float))
for r in seg_rows:
    print(r["segment"], round(r["rel_lift"], 3), [round(c, 3) for c in r["rel_ci"]],
          round(r["p_raw"], 4), round(r["p_holm"], 4))
print(cd[["date", "rel_lift", "p"]].round(4).to_string())
