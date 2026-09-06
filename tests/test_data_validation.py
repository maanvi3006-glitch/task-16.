import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import pandas as pd
import numpy as np
from data_validation import clean_and_validate, DataValidationError


def make_basic_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "group": ["control"] * (n // 2) + ["treatment"] * (n // 2),
        "converted": rng.binomial(1, 0.1, n),
        "revenue": rng.uniform(0, 1000, n),
    })


def test_empty_dataframe_raises():
    with pytest.raises(DataValidationError):
        clean_and_validate(pd.DataFrame())


def test_missing_required_columns_raises():
    df = pd.DataFrame({"user_id": [1, 2], "group": ["control", "treatment"]})
    with pytest.raises(DataValidationError):
        clean_and_validate(df)


def test_only_one_group_raises():
    df = make_basic_df()
    df["group"] = "control"
    with pytest.raises(DataValidationError):
        clean_and_validate(df)


def test_three_groups_raises():
    df = make_basic_df()
    df.loc[df.index[:10], "group"] = "holdout"
    with pytest.raises(DataValidationError):
        clean_and_validate(df)


def test_unexpected_group_label_raises():
    df = make_basic_df()
    df["group"] = df["group"].replace("control", "variant_a")
    with pytest.raises(DataValidationError):
        clean_and_validate(df)


def test_invalid_converted_values_raise():
    df = make_basic_df()
    df.loc[0, "converted"] = 5
    with pytest.raises(DataValidationError):
        clean_and_validate(df)


def test_clean_data_passes_through():
    df = make_basic_df()
    clean_df, report = clean_and_validate(df)
    assert len(clean_df) == len(df)
    assert report.duplicates_removed == 0
    assert report.n_clean == report.n_raw


def test_duplicates_are_removed():
    df = make_basic_df()
    dup = df.iloc[[0, 1]]
    df2 = pd.concat([df, dup], ignore_index=True)
    clean_df, report = clean_and_validate(df2)
    assert report.duplicates_removed == 2
    assert len(clean_df) == len(df)


def test_negative_revenue_removed():
    df = make_basic_df()
    df.loc[0, "revenue"] = -50
    clean_df, report = clean_and_validate(df)
    assert report.negative_revenue_removed == 1
    assert (clean_df["revenue"] >= 0).all()


def test_missing_revenue_for_converter_is_imputed_not_dropped():
    df = make_basic_df()
    df.loc[0, "converted"] = 1
    df.loc[0, "revenue"] = np.nan
    clean_df, report = clean_and_validate(df)
    assert report.missing_revenue_imputed >= 1
    assert len(clean_df) == len(df)  # row kept, not dropped
    assert not clean_df["revenue"].isna().any()


def test_contamination_crossover_removed():
    df = make_basic_df()
    contaminated_id = df.loc[0, "user_id"]
    extra_row = df.iloc[[0]].copy()
    extra_row["group"] = "treatment" if df.loc[0, "group"] == "control" else "control"
    extra_row["user_id"] = contaminated_id
    df2 = pd.concat([df, extra_row], ignore_index=True)
    clean_df, report = clean_and_validate(df2)
    assert report.contamination_removed >= 2
    assert contaminated_id not in clean_df["user_id"].values


def test_srm_flagged_on_extreme_imbalance():
    rng = np.random.default_rng(1)
    n_c, n_t = 5000, 500  # wildly imbalanced -> should trip SRM
    df = pd.DataFrame({
        "user_id": [f"c{i}" for i in range(n_c)] + [f"t{i}" for i in range(n_t)],
        "group": ["control"] * n_c + ["treatment"] * n_t,
        "converted": rng.binomial(1, 0.1, n_c + n_t),
        "revenue": rng.uniform(0, 1000, n_c + n_t),
    })
    _, report = clean_and_validate(df)
    assert report.srm_flag is True


def test_all_rows_removed_raises():
    df = make_basic_df(n=10)
    df["revenue"] = -1  # every row will be dropped as negative revenue
    with pytest.raises(DataValidationError):
        clean_and_validate(df)
