"""
Bayesian Beta-Binomial A/B test — the alternative approach to the frequentist
two-proportion z-test, explicitly named in the study guide
("Frequentist A/B vs Bayesian A/B"). Reported side-by-side so a reader can
see where the two philosophies agree/disagree, rather than picking one and
hiding the comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class BayesianResult:
    prior_alpha: float
    prior_beta: float
    posterior_control: tuple
    posterior_treatment: tuple
    prob_treatment_better: float
    expected_loss_choosing_treatment: float
    credible_interval_diff: tuple

    def as_markdown(self) -> str:
        pc_a, pc_b = self.posterior_control
        pt_a, pt_b = self.posterior_treatment
        return "\n".join([
            "#### Alternative approach: Bayesian Beta-Binomial model (conversion rate)",
            f"- Prior: Beta({self.prior_alpha}, {self.prior_beta}) (weakly informative, uniform-ish)",
            f"- Posterior control: Beta({pc_a:.1f}, {pc_b:.1f})",
            f"- Posterior treatment: Beta({pt_a:.1f}, {pt_b:.1f})",
            f"- **P(treatment conversion > control conversion) = {self.prob_treatment_better:.4f}**",
            f"- 95% credible interval on the difference (treatment - control): "
            f"[{self.credible_interval_diff[0]:.4f}, {self.credible_interval_diff[1]:.4f}]",
            f"- Expected loss from choosing treatment if wrong: {self.expected_loss_choosing_treatment:.5f}",
            "- Reading: the frequentist p-value answers 'how surprising is this data if there's truly no "
            "effect', while this gives a direct probability statement about which variant is actually better "
            "— useful for explaining the decision to a non-technical stakeholder.",
        ])


def bayesian_ab_test(conversions_control: int, n_control: int, conversions_treatment: int,
                      n_treatment: int, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                      n_samples: int = 200_000, seed: int = 123) -> BayesianResult:
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Group sizes must be positive for the Bayesian model.")
    rng = np.random.default_rng(seed)

    post_c = (prior_alpha + conversions_control, prior_beta + n_control - conversions_control)
    post_t = (prior_alpha + conversions_treatment, prior_beta + n_treatment - conversions_treatment)

    samples_c = rng.beta(post_c[0], post_c[1], size=n_samples)
    samples_t = rng.beta(post_t[0], post_t[1], size=n_samples)
    diff = samples_t - samples_c

    prob_t_better = float(np.mean(diff > 0))
    ci = tuple(np.percentile(diff, [2.5, 97.5]).tolist())
    # expected loss of choosing treatment when it's actually worse
    loss = np.where(diff < 0, -diff, 0.0)
    expected_loss = float(np.mean(loss))

    return BayesianResult(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        posterior_control=post_c,
        posterior_treatment=post_t,
        prob_treatment_better=prob_t_better,
        expected_loss_choosing_treatment=expected_loss,
        credible_interval_diff=ci,
    )
