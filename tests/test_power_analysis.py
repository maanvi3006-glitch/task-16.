import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from power_analysis import (
    sample_size_for_proportions, sample_size_for_means,
    achieved_power_proportions, achieved_power_means,
)


def test_sample_size_for_proportions_basic():
    result = sample_size_for_proportions(baseline_rate=0.10, mde_abs=0.02)
    assert result.required_n_per_group > 0
    # smaller MDE should require MORE sample
    result_small_mde = sample_size_for_proportions(baseline_rate=0.10, mde_abs=0.005)
    assert result_small_mde.required_n_per_group > result.required_n_per_group


def test_sample_size_for_proportions_invalid_baseline_raises():
    with pytest.raises(ValueError):
        sample_size_for_proportions(baseline_rate=0.0, mde_abs=0.01)
    with pytest.raises(ValueError):
        sample_size_for_proportions(baseline_rate=1.0, mde_abs=0.01)
    with pytest.raises(ValueError):
        sample_size_for_proportions(baseline_rate=1.5, mde_abs=0.01)


def test_sample_size_for_proportions_invalid_mde_raises():
    with pytest.raises(ValueError):
        sample_size_for_proportions(baseline_rate=0.1, mde_abs=0)
    with pytest.raises(ValueError):
        sample_size_for_proportions(baseline_rate=0.1, mde_abs=-0.01)


def test_sample_size_for_means_basic():
    result = sample_size_for_means(std_dev=100, mde_abs=10)
    assert result.required_n_per_group > 0


def test_sample_size_for_means_invalid_std_raises():
    with pytest.raises(ValueError):
        sample_size_for_means(std_dev=0, mde_abs=10)
    with pytest.raises(ValueError):
        sample_size_for_means(std_dev=-5, mde_abs=10)


def test_achieved_power_increases_with_n():
    p_small = achieved_power_proportions(n_per_group=100, effect_size=0.1)
    p_large = achieved_power_proportions(n_per_group=5000, effect_size=0.1)
    assert p_large > p_small
    assert 0 <= p_small <= 1
    assert 0 <= p_large <= 1


def test_achieved_power_means_increases_with_n():
    p_small = achieved_power_means(n_per_group=50, effect_size=0.2)
    p_large = achieved_power_means(n_per_group=2000, effect_size=0.2)
    assert p_large > p_small
