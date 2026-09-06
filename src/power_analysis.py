"""
Sample size & power analysis — done BEFORE looking at results (as the
pipeline step 3 in the brief requires), and re-checked achieved power
afterwards for the report.

Marking coverage: "Estimate required sample size/power" (build pipeline
step), and directly supports avoiding the "underpowered study" pitfall.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerResult:
    metric: str
    baseline_rate_or_mean: float
    mde: float
    effect_size: float
    alpha: float
    power_target: float
    required_n_per_group: int
    achieved_n_per_group: int | None = None
    achieved_power: float | None = None

    def as_markdown(self) -> str:
        lines = [
            f"**{self.metric} — a-priori power analysis**",
            f"- Baseline: {self.baseline_rate_or_mean:.4g}",
            f"- Minimum Detectable Effect (MDE): {self.mde:.4g}",
            f"- Standardized effect size: {self.effect_size:.4f}",
            f"- alpha = {self.alpha}, target power = {self.power_target}",
            f"- **Required sample size per group: {self.required_n_per_group:,}**",
        ]
        if self.achieved_n_per_group is not None:
            verdict = "sufficiently powered ✅" if self.achieved_n_per_group >= self.required_n_per_group else "UNDERPOWERED ⚠️"
            lines.append(f"- Achieved sample size per group: {self.achieved_n_per_group:,} → {verdict}")
        if self.achieved_power is not None:
            lines.append(f"- Achieved (post-hoc) power at observed n: {self.achieved_power:.3f}")
        return "\n".join(lines)


def sample_size_for_proportions(baseline_rate: float, mde_abs: float, alpha: float = 0.05,
                                  power: float = 0.8) -> PowerResult:
    """Required n per group to detect an absolute lift of `mde_abs` on a baseline
    conversion rate `baseline_rate`, using a two-proportion z-test."""
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be strictly between 0 and 1.")
    if mde_abs <= 0:
        raise ValueError("mde_abs (minimum detectable effect) must be positive.")

    p2 = baseline_rate + mde_abs
    effect_size = proportion_effectsize(baseline_rate, p2)  # Cohen's h
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0,
                              alternative="two-sided")
    return PowerResult(
        metric="Conversion rate",
        baseline_rate_or_mean=baseline_rate,
        mde=mde_abs,
        effect_size=effect_size,
        alpha=alpha,
        power_target=power,
        required_n_per_group=int(np.ceil(n)),
    )


def sample_size_for_means(std_dev: float, mde_abs: float, alpha: float = 0.05,
                            power: float = 0.8) -> PowerResult:
    """Required n per group to detect an absolute difference in means `mde_abs`
    given a (pooled) standard deviation `std_dev`, using a two-sample t-test."""
    if std_dev <= 0:
        raise ValueError("std_dev must be positive.")
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive.")
    effect_size = mde_abs / std_dev  # Cohen's d
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0,
                              alternative="two-sided")
    return PowerResult(
        metric="Revenue per user",
        baseline_rate_or_mean=std_dev,
        mde=mde_abs,
        effect_size=effect_size,
        alpha=alpha,
        power_target=power,
        required_n_per_group=int(np.ceil(n)),
    )


def achieved_power_proportions(n_per_group: int, effect_size: float, alpha: float = 0.05) -> float:
    analysis = NormalIndPower()
    return float(analysis.power(effect_size=abs(effect_size), nobs1=n_per_group, alpha=alpha, ratio=1.0))


def achieved_power_means(n_per_group: int, effect_size: float, alpha: float = 0.05) -> float:
    analysis = TTestIndPower()
    return float(analysis.power(effect_size=abs(effect_size), nobs1=n_per_group, alpha=alpha, ratio=1.0))
