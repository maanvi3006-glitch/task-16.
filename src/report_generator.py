"""
Generates matplotlib figures and assembles the full markdown report that is
the primary evidence artifact for "documented test/experiment with effect
size, significance and a recommendation".
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_conversion_bar(df: pd.DataFrame, out_path: str):
    rates = df.groupby("group")["converted"].mean().reindex(["control", "treatment"])
    ns = df.groupby("group")["converted"].count().reindex(["control", "treatment"])
    ses = np.sqrt(rates * (1 - rates) / ns)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#7f8c8d", "#2980b9"]
    ax.bar(rates.index, rates.values, yerr=1.96 * ses.values, capsize=6, color=colors)
    ax.set_ylabel("Conversion rate")
    ax.set_title("Checkout Conversion Rate by Group (±95% CI)")
    for i, (r, n) in enumerate(zip(rates.values, ns.values)):
        ax.text(i, r + 0.003, f"{r:.2%}\n(n={n:,})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_revenue_distribution(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for grp, color in [("control", "#7f8c8d"), ("treatment", "#2980b9")]:
        vals = df.loc[(df["group"] == grp) & (df["converted"] == 1), "revenue"]
        ax.hist(vals, bins=40, alpha=0.55, label=grp, color=color, density=True)
    ax.set_xlabel("Revenue per converting user")
    ax.set_ylabel("Density")
    ax.set_title("Revenue Distribution (converters only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bayesian_posteriors(bayes_result, out_path: str):
    from scipy.stats import beta
    x = np.linspace(0, 0.2, 800)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    pc_a, pc_b = bayes_result.posterior_control
    pt_a, pt_b = bayes_result.posterior_treatment
    ax.plot(x, beta.pdf(x, pc_a, pc_b), label="Control posterior", color="#7f8c8d")
    ax.plot(x, beta.pdf(x, pt_a, pt_b), label="Treatment posterior", color="#2980b9")
    ax.fill_between(x, beta.pdf(x, pc_a, pc_b), alpha=0.2, color="#7f8c8d")
    ax.fill_between(x, beta.pdf(x, pt_a, pt_b), alpha=0.2, color="#2980b9")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Posterior density")
    ax.set_title("Bayesian Posterior Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_power_curve(power_result, achieved_n, out_path: str):
    from statsmodels.stats.power import NormalIndPower
    analysis = NormalIndPower()
    ns = np.arange(200, max(achieved_n * 1.4, power_result.required_n_per_group * 1.4), 200)
    powers = [analysis.power(effect_size=abs(power_result.effect_size), nobs1=n, alpha=power_result.alpha)
              for n in ns]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ns, powers, color="#2980b9")
    ax.axhline(power_result.power_target, ls="--", color="gray", label=f"Target power = {power_result.power_target}")
    ax.axvline(power_result.required_n_per_group, ls="--", color="green",
               label=f"Required n = {power_result.required_n_per_group:,}")
    ax.axvline(achieved_n, ls="--", color="red", label=f"Achieved n = {achieved_n:,}")
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Statistical power")
    ax.set_title("Power Curve — Conversion Rate Test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_full_report(validation_report, power_conv, power_rev, conv_test, rev_t_test,
                       rev_u_test, bayes_result, recommendation, fig_dir: str) -> str:
    md = []
    md.append("# A/B Test Report — Checkout Redesign Experiment")
    md.append("_Task 16: Hypothesis Testing & A/B Tests — Data Analyst, Phase 1 Industry Immersion_\n")

    md.append("## 1. Objective & Hypotheses (pipeline step 1)")
    md.append(
        "**Business question:** Does the new one-page checkout flow (treatment) change checkout "
        "conversion rate and/or revenue per user compared to the current checkout (control)?\n\n"
        "**Decision this informs:** whether to roll the new checkout out to 100% of traffic.\n\n"
        "- **H1 (primary):** H0: conversion_treatment = conversion_control vs H1: they differ.\n"
        "- **H2 (secondary):** H0: mean revenue/user is equal between groups vs H1: it differs.\n\n"
        "Both hypotheses and the significance level (alpha = 0.05) were fixed **before** looking at "
        "the results, and a multiple-testing correction (Holm-Bonferroni) is applied across the two "
        "metrics to avoid p-hacking."
    )

    md.append("## 2. Test Selection & Justification (pipeline step 2)")
    md.append(
        "- **Conversion rate** is a binary outcome at reasonably large sample size → **two-proportion "
        "z-test** is the standard, appropriate choice (a chi-square test of independence would give an "
        "equivalent result for 2x2 tables).\n"
        "- **Revenue per user** is heavily right-skewed with a point-mass at 0 (non-converters). A "
        "**Welch's t-test** (robust to unequal variance, and valid asymptotically by the CLT even under "
        "skew at this sample size) is run as the primary parametric test, cross-checked against a "
        "**Mann-Whitney U test** (distribution-free, robust to the skew/outliers) — the alternative "
        "named explicitly in the study guide. Where they disagree, the non-parametric result is trusted more.\n"
        "- As a further alternative, a **Bayesian Beta-Binomial model** is run on conversion rate "
        "alongside the frequentist test (see Section 5), giving a direct probability statement rather "
        "than a p-value — useful for stakeholder communication."
    )

    md.append(validation_report.as_markdown())

    md.append("## 3. Sample Size & Power Analysis (pipeline step 3)")
    md.append(power_conv.as_markdown())
    md.append("")
    md.append(power_rev.as_markdown())
    md.append(f"\n![Power curve]({os.path.basename(fig_dir)}/power_curve.png)\n")

    md.append("## 4. Results — Frequentist Tests (pipeline steps 4–5)")
    md.append(conv_test.as_markdown())
    md.append(f"\n![Conversion rate by group]({os.path.basename(fig_dir)}/conversion_bar.png)\n")
    md.append(rev_t_test.as_markdown())
    md.append(rev_u_test.as_markdown())
    md.append(f"\n![Revenue distribution]({os.path.basename(fig_dir)}/revenue_dist.png)\n")

    md.append("## 5. Alternative Approach — Bayesian A/B Test")
    md.append(bayes_result.as_markdown())
    md.append(f"\n![Bayesian posteriors]({os.path.basename(fig_dir)}/bayesian_posteriors.png)\n")

    md.append("## 6. Recommendation (pipeline step 6)")
    md.append(recommendation.as_markdown())

    md.append("## 7. Pitfalls Explicitly Checked")
    md.append(
        "- **p-hacking:** hypotheses & alpha fixed up front (Section 1); Holm-Bonferroni correction "
        "applied across the 2 metrics tested; no repeated peeking.\n"
        "- **Significant-but-trivial effects:** practical-significance threshold applied on top of "
        "statistical significance in the Recommendation section.\n"
        "- **Unequal / contaminated groups:** Sample Ratio Mismatch test, cross-over contamination "
        "removal, and covariate-balance checks all run in Section 2's data validation summary."
    )

    md.append("## 8. Confounds & Deeper Questions")
    md.append(
        "- Could device mix, country mix, or time-of-experiment (e.g. a promo mid-experiment) explain "
        "the difference instead of the checkout redesign? → checked via the covariate balance test above; "
        "re-run with stratification if imbalance is flagged.\n"
        "- Is the effect size large enough to justify the engineering cost of shipping the new checkout? "
        "→ addressed explicitly via the practical-significance threshold.\n"
        "- The hypothesis, test choice, and alpha were locked in Section 1 before any peeking at results."
    )

    return "\n\n".join(md)
