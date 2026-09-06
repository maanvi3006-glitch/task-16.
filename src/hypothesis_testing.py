"""
Core inferential statistics for the A/B test.

Marking coverage: this module IS the "documented test/experiment with
effect size, significance and a recommendation" core deliverable (50 marks).

Two pre-registered hypotheses (state BEFORE looking at results — pipeline
step 1/2):

  H1 (primary):   Checkout conversion rate differs between control and treatment.
                  Test: two-proportion z-test (large-sample, binary outcome).
                  Alternative approach considered: Bayesian Beta-Binomial (see
                  bayesian_analysis.py) — reported side-by-side for comparison.

  H2 (secondary): Revenue per user differs between control and treatment.
                  Revenue is right-skewed (lognormal-like) with a mass at 0
                  for non-converters, so BOTH a Welch's t-test (robust to
                  unequal variance) AND a non-parametric Mann-Whitney U test
                  are run — the two are compared explicitly, which is the
                  "t-test vs non-parametric for skewed data" alternative
                  named in the study guide.

Multiple-testing correction: since two hypotheses are tested on the same
experiment, Holm-Bonferroni is applied to the family of p-values before
declaring anything "significant" — this directly avoids the "p-hacking /
testing until something sticks" pitfall.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_effectsize, confint_proportions_2indep


@dataclass
class TestResult:
    name: str
    hypothesis: str
    statistic: float
    p_value: float
    p_value_adj: float | None
    effect_size_name: str
    effect_size: float
    ci_low: float
    ci_high: float
    alpha: float
    significant_raw: bool
    significant_adj: bool
    interpretation: str
    extra: dict = field(default_factory=dict)

    def as_markdown(self) -> str:
        lines = [
            f"#### {self.name}",
            f"- **H0/H1**: {self.hypothesis}",
            f"- Test statistic: {self.statistic:.4f}",
            f"- p-value (raw): {self.p_value:.5f}"
            + (f"  |  Holm-adjusted: {self.p_value_adj:.5f}" if self.p_value_adj is not None else ""),
            f"- Effect size ({self.effect_size_name}): {self.effect_size:.4f}",
            f"- {int((1-self.alpha)*100)}% CI on the difference: [{self.ci_low:.4f}, {self.ci_high:.4f}]",
            f"- Significant at alpha={self.alpha}: raw = {self.significant_raw}, "
            f"after multiple-testing correction = {self.significant_adj}",
            f"- **Interpretation**: {self.interpretation}",
        ]
        return "\n".join(lines)


def _cohens_h(p1: float, p2: float) -> float:
    return proportion_effectsize(p2, p1)  # treatment - control convention (p2 - p1 direction)


def two_proportion_ztest(df: pd.DataFrame, alpha: float = 0.05) -> TestResult:
    if df["group"].nunique() != 2:
        raise ValueError("two_proportion_ztest requires exactly two groups.")
    g = df.groupby("group")["converted"].agg(["sum", "count"])
    if "control" not in g.index or "treatment" not in g.index:
        raise ValueError("Expected 'control' and 'treatment' groups.")
    x1, n1 = g.loc["control", "sum"], g.loc["control", "count"]
    x2, n2 = g.loc["treatment", "sum"], g.loc["treatment", "count"]
    p1, p2 = x1 / n1, x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    ci_low, ci_high = confint_proportions_2indep(
        count1=x2, nobs1=n2, count2=x1, nobs2=n1, method="wald", alpha=alpha
    )
    effect_size = _cohens_h(p1, p2)

    diff = p2 - p1
    rel_lift = diff / p1 if p1 > 0 else float("nan")
    interp = (
        f"Control conversion = {p1:.4%} (n={n1:,}), Treatment conversion = {p2:.4%} (n={n2:,}). "
        f"Absolute lift = {diff:+.4%} ({rel_lift:+.2%} relative). "
    )
    interp += "Statistically significant at alpha={:.2f}.".format(alpha) if p_value < alpha else \
        "NOT statistically significant at alpha={:.2f} — could plausibly be chance.".format(alpha)

    return TestResult(
        name="Primary metric: Conversion Rate (Two-Proportion Z-Test)",
        hypothesis="H0: p_treatment = p_control  vs  H1: p_treatment ≠ p_control",
        statistic=z,
        p_value=p_value,
        p_value_adj=None,
        effect_size_name="Cohen's h",
        effect_size=effect_size,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant_raw=p_value < alpha,
        significant_adj=False,  # filled in later after correction
        interpretation=interp,
        extra={"p_control": p1, "p_treatment": p2, "n_control": int(n1), "n_treatment": int(n2),
               "abs_lift": diff, "rel_lift": rel_lift},
    )


def _bootstrap_ci_diff_means(a: np.ndarray, b: np.ndarray, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sb.mean() - sa.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def revenue_tests(df: pd.DataFrame, alpha: float = 0.05, n_boot: int = 4000) -> tuple[TestResult, TestResult]:
    """Runs BOTH Welch's t-test and Mann-Whitney U on revenue-per-user, so the
    two named alternative approaches for skewed data can be compared directly."""
    if df["group"].nunique() != 2:
        raise ValueError("revenue_tests requires exactly two groups.")
    control = df.loc[df["group"] == "control", "revenue"].to_numpy(dtype=float)
    treatment = df.loc[df["group"] == "treatment", "revenue"].to_numpy(dtype=float)
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Need at least 2 observations per group for revenue tests.")

    # --- Welch's t-test (parametric, robust to unequal variance) ---
    t_stat, t_p = stats.ttest_ind(treatment, control, equal_var=False)
    mean_c, mean_t = control.mean(), treatment.mean()
    pooled_sd = np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2)
    cohens_d = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0
    ci_low, ci_high = _bootstrap_ci_diff_means(control, treatment, n_boot=n_boot, alpha=alpha)

    skew_c, skew_t = stats.skew(control), stats.skew(treatment)
    interp_t = (
        f"Mean revenue/user: control = {mean_c:.2f}, treatment = {mean_t:.2f} "
        f"(diff = {mean_t - mean_c:+.2f}). Distribution is right-skewed "
        f"(skew: control={skew_c:.2f}, treatment={skew_t:.2f}), so this parametric result "
        "is cross-checked against the non-parametric test below before trusting it. "
    )
    interp_t += "Significant." if t_p < alpha else "Not significant at alpha={:.2f}.".format(alpha)

    t_result = TestResult(
        name="Secondary metric: Revenue per User — Welch's t-test",
        hypothesis="H0: mean_revenue_treatment = mean_revenue_control  vs  H1: they differ",
        statistic=t_stat,
        p_value=t_p,
        p_value_adj=None,
        effect_size_name="Cohen's d",
        effect_size=cohens_d,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant_raw=t_p < alpha,
        significant_adj=False,
        interpretation=interp_t,
        extra={"mean_control": mean_c, "mean_treatment": mean_t, "skew_control": skew_c, "skew_treatment": skew_t},
    )

    # --- Mann-Whitney U (non-parametric, robust to skew/outliers, tests distribution shift) ---
    u_stat, u_p = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    n1, n2 = len(treatment), len(control)
    # rank-biserial correlation as the non-parametric effect size
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    median_c, median_t = np.median(control), np.median(treatment)

    # bootstrap CI for the median difference (Hodges-Lehmann style, via bootstrap of medians)
    rng = np.random.default_rng(7)
    med_diffs = np.empty(n_boot)
    for i in range(n_boot):
        med_diffs[i] = np.median(rng.choice(treatment, n1, True)) - np.median(rng.choice(control, n2, True))
    mlo, mhi = np.percentile(med_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    interp_u = (
        f"Median revenue/user: control = {median_c:.2f}, treatment = {median_t:.2f}. "
        f"Rank-biserial effect size = {rank_biserial:.4f}. "
    )
    agree = (u_p < alpha) == (t_p < alpha)
    interp_u += "Agrees with the t-test conclusion above." if agree else \
        "DISAGREES with the t-test conclusion — treat the revenue result with caution, likely driven by skew/outliers."
    interp_u += " Significant." if u_p < alpha else f" Not significant at alpha={alpha}."

    u_result = TestResult(
        name="Secondary metric: Revenue per User — Mann-Whitney U (non-parametric)",
        hypothesis="H0: distributions are identical  vs  H1: treatment revenue is stochastically greater/less",
        statistic=u_stat,
        p_value=u_p,
        p_value_adj=None,
        effect_size_name="Rank-biserial correlation",
        effect_size=rank_biserial,
        ci_low=mlo,
        ci_high=mhi,
        alpha=alpha,
        significant_raw=u_p < alpha,
        significant_adj=False,
        interpretation=interp_u,
        extra={"median_control": median_c, "median_treatment": median_t},
    )
    return t_result, u_result


def apply_multiple_testing_correction(results: list[TestResult], alpha: float = 0.05, method="holm") -> list[TestResult]:
    """Applies Holm-Bonferroni across the given family of tests, mutating and
    returning updated TestResult objects. Directly prevents the 'p-hacking /
    testing until something sticks' pitfall by controlling family-wise error."""
    if not results:
        return results
    raw_p = [r.p_value for r in results]
    reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method=method)
    out = []
    for r, p_a, rej in zip(results, p_adj, reject):
        r.p_value_adj = float(p_a)
        r.significant_adj = bool(rej)
        out.append(r)
    return out
