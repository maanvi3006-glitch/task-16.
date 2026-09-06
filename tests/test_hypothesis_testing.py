import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import numpy as np
import pandas as pd
from hypothesis_testing import two_proportion_ztest, revenue_tests, apply_multiple_testing_correction
from bayesian_analysis import bayesian_ab_test


def make_df(n_c=1000, n_t=1000, p_c=0.10, p_t=0.10, seed=0):
    rng = np.random.default_rng(seed)
    conv_c = rng.binomial(1, p_c, n_c)
    conv_t = rng.binomial(1, p_t, n_t)
    rev_c = np.where(conv_c == 1, rng.lognormal(6, 0.5, n_c), 0.0)
    rev_t = np.where(conv_t == 1, rng.lognormal(6, 0.5, n_t), 0.0)
    df = pd.DataFrame({
        "user_id": [f"c{i}" for i in range(n_c)] + [f"t{i}" for i in range(n_t)],
        "group": ["control"] * n_c + ["treatment"] * n_t,
        "converted": np.concatenate([conv_c, conv_t]),
        "revenue": np.concatenate([rev_c, rev_t]),
    })
    return df


def test_two_proportion_ztest_no_true_effect():
    df = make_df(p_c=0.10, p_t=0.10, seed=1)
    result = two_proportion_ztest(df)
    assert result.p_value > 0.01  # should usually not be significant with no real effect
    assert result.effect_size_name == "Cohen's h"


def test_two_proportion_ztest_detects_large_effect():
    df = make_df(n_c=2000, n_t=2000, p_c=0.05, p_t=0.15, seed=2)
    result = two_proportion_ztest(df)
    assert result.p_value < 0.001
    assert result.extra["abs_lift"] > 0


def test_two_proportion_ztest_requires_two_groups():
    df = make_df()
    df["group"] = "control"
    with pytest.raises(ValueError):
        two_proportion_ztest(df)


def test_revenue_tests_basic_and_agreement():
    df = make_df(n_c=1500, n_t=1500, p_c=0.08, p_t=0.08, seed=3)
    t_res, u_res = revenue_tests(df)
    assert t_res.effect_size_name == "Cohen's d"
    assert u_res.effect_size_name == "Rank-biserial correlation"


def test_revenue_tests_too_few_observations_raises():
    df = pd.DataFrame({
        "user_id": ["a", "b"], "group": ["control", "treatment"],
        "converted": [1, 1], "revenue": [10, 20],
    })
    with pytest.raises(ValueError):
        revenue_tests(df)


def test_multiple_testing_correction_reduces_false_positives():
    df = make_df(p_c=0.10, p_t=0.10, seed=5)
    conv = two_proportion_ztest(df)
    t_res, u_res = revenue_tests(df)
    results = apply_multiple_testing_correction([conv, t_res, u_res])
    for r in results:
        assert r.p_value_adj is not None
        assert r.p_value_adj >= r.p_value - 1e-9  # Holm adjustment never decreases p


def test_multiple_testing_correction_empty_list():
    assert apply_multiple_testing_correction([]) == []


def test_bayesian_ab_test_basic():
    result = bayesian_ab_test(conversions_control=100, n_control=1000,
                               conversions_treatment=140, n_treatment=1000)
    assert 0 <= result.prob_treatment_better <= 1
    assert result.prob_treatment_better > 0.9  # treatment clearly better here


def test_bayesian_ab_test_invalid_n_raises():
    with pytest.raises(ValueError):
        bayesian_ab_test(conversions_control=10, n_control=0,
                          conversions_treatment=10, n_treatment=100)
