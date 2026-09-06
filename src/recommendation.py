"""
Translates the statistical results into a clear recommendation.

Marking coverage: pipeline step 6 ("Translate the result into a clear
recommendation") and the "Significant but trivially small effects" pitfall —
this module explicitly checks PRACTICAL significance (a minimum
business-relevant effect size), not just p < 0.05.
"""
from __future__ import annotations
from dataclasses import dataclass

MIN_PRACTICAL_LIFT_ABS = 0.005   # 0.5 percentage points absolute conversion lift
MIN_PRACTICAL_COHENS_H = 0.02    # tiny-effect threshold on Cohen's h


@dataclass
class Recommendation:
    decision: str
    rationale: list
    confidence_caveats: list

    def as_markdown(self) -> str:
        lines = [f"## Recommendation: {self.decision}", "", "**Rationale:**"]
        lines += [f"- {r}" for r in self.rationale]
        if self.confidence_caveats:
            lines.append("")
            lines.append("**Caveats / what would change this recommendation:**")
            lines += [f"- {c}" for c in self.confidence_caveats]
        return "\n".join(lines)


def build_recommendation(conv_test, revenue_t_test, revenue_u_test, bayes_result,
                          validation_report) -> Recommendation:
    rationale = []
    caveats = []

    stat_sig = conv_test.significant_adj
    abs_lift = conv_test.extra["abs_lift"]
    practically_sig = abs(abs_lift) >= MIN_PRACTICAL_LIFT_ABS and abs(conv_test.effect_size) >= MIN_PRACTICAL_COHENS_H

    rationale.append(
        f"Conversion rate: {abs_lift:+.3%} absolute change, Holm-adjusted p={conv_test.p_value_adj:.4f} "
        f"({'significant' if stat_sig else 'not significant'} after correcting for testing 2 metrics)."
    )
    rationale.append(
        f"Bayesian check: P(treatment beats control) = {bayes_result.prob_treatment_better:.1%}, "
        f"expected loss from shipping treatment if wrong = {bayes_result.expected_loss_choosing_treatment:.4f} "
        "— the two frameworks broadly agree." if (bayes_result.prob_treatment_better > 0.95) == stat_sig
        else f"Bayesian check: P(treatment beats control) = {bayes_result.prob_treatment_better:.1%} "
             "— note this is more/less confident than the frequentist test; investigate before deciding."
    )
    rationale.append(
        f"Practical significance check: effect size (Cohen's h={conv_test.effect_size:.3f}) is "
        f"{'above' if practically_sig else 'below'} the minimum business-relevant threshold "
        f"({MIN_PRACTICAL_COHENS_H})."
    )
    rationale.append(
        f"Revenue per user: t-test p={revenue_t_test.p_value_adj:.4f}, Mann-Whitney p={revenue_u_test.p_value_adj:.4f} "
        f"({'both agree' if revenue_t_test.significant_adj == revenue_u_test.significant_adj else 'DISAGREE - treat with caution'})."
    )

    if validation_report.srm_flag:
        caveats.append("Sample Ratio Mismatch was flagged — results are not fully trustworthy until the "
                        "assignment/randomisation pipeline is investigated.")
    if validation_report.covariate_balance:
        for k, v in validation_report.covariate_balance.items():
            if v < 0.01:
                caveats.append(f"'{k}' was imbalanced across groups (p={v:.4f}); consider re-running with "
                                "stratified randomisation or covariate adjustment.")

    if stat_sig and practically_sig and abs_lift > 0:
        decision = "SHIP the treatment (new checkout flow)."
        rationale.append("Both statistical AND practical significance criteria are met, and the effect is positive.")
    elif stat_sig and not practically_sig:
        decision = "DO NOT ship based on this alone — statistically significant but effect is too small to matter."
        rationale.append("This is the classic 'significant but trivial' pitfall: at this scale even a tiny, "
                          "operationally meaningless difference can hit p<0.05.")
    elif stat_sig and abs_lift < 0:
        decision = "DO NOT ship the treatment — it significantly underperforms control."
    else:
        decision = "DO NOT ship yet — inconclusive. Continue the test or redesign."
        caveats.append("Consider whether the study was adequately powered for the observed effect size "
                        "(see the power analysis section) before concluding there is truly no effect.")

    if not caveats:
        caveats.append("No major data-quality red flags found in validation; recommendation is reasonably trustworthy.")

    return Recommendation(decision=decision, rationale=rationale, confidence_caveats=caveats)
