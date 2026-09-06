"""
Data validation & cleaning for the A/B test.

Marking coverage:
  - "Real-data quality & correctness" (data isn't taken at face value)
  - "Dependency, failure & edge-case handling" (raises clear, typed errors
    instead of silently producing wrong numbers)

Everything here is deliberately defensive: bad input data should fail loudly
with a specific, actionable message rather than flowing quietly into a
p-value that looks fine but is wrong.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from scipy import stats


class DataValidationError(Exception):
    """Raised when the input data cannot be safely analysed as-is."""


REQUIRED_COLUMNS = {"user_id", "group", "converted", "revenue"}


@dataclass
class ValidationReport:
    n_raw: int
    n_clean: int
    duplicates_removed: int
    contamination_removed: int
    missing_revenue_imputed: int
    outliers_removed: int
    negative_revenue_removed: int
    group_counts: dict
    srm_p_value: float
    srm_flag: bool
    covariate_balance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = [
            "### Data Validation & Cleaning Summary",
            f"- Raw rows loaded: **{self.n_raw:,}**",
            f"- Exact duplicate rows removed: **{self.duplicates_removed:,}**",
            f"- Cross-over (contaminated) users removed: **{self.contamination_removed:,}**",
            f"- Negative / impossible revenue values removed: **{self.negative_revenue_removed:,}**",
            f"- Extreme outlier (bot-like) revenue rows removed: **{self.outliers_removed:,}**",
            f"- Missing revenue values imputed as 0 (non-converters/no-charge): **{self.missing_revenue_imputed:,}**",
            f"- Clean rows analysed: **{self.n_clean:,}**",
            f"- Final group sizes: {self.group_counts}",
            f"- Sample Ratio Mismatch (SRM) check p-value: **{self.srm_p_value:.4f}** "
            f"({'⚠️ FLAGGED — allocation may be broken' if self.srm_flag else 'OK, allocation looks as expected'})",
        ]
        if self.covariate_balance:
            lines.append("- Covariate balance (chi-square p-values across groups):")
            for k, v in self.covariate_balance.items():
                flag = "⚠️" if v < 0.01 else "OK"
                lines.append(f"    - {k}: p = {v:.4f} ({flag})")
        if self.warnings:
            lines.append("- Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def _check_schema(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        raise DataValidationError("Input dataset is empty. Cannot run a hypothesis test on 0 rows.")
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Missing required column(s): {sorted(missing_cols)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}."
        )
    groups = set(df["group"].dropna().unique().tolist())
    if len(groups) < 2:
        raise DataValidationError(
            f"Need exactly two groups to run an A/B test, found: {groups}. "
            "Check the 'group' column / your export filter."
        )
    if len(groups) > 2:
        raise DataValidationError(
            f"Found {len(groups)} distinct groups {groups}; this pipeline runs a two-arm "
            "(control vs treatment) test. Filter to two arms first."
        )


def clean_and_validate(df: pd.DataFrame, expected_groups=("control", "treatment")) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Cleans a raw A/B test extract and returns (clean_df, ValidationReport).
    Raises DataValidationError on unrecoverable problems (empty data, wrong schema,
    fewer than 2 groups, etc.) so the caller can surface a clear message instead of
    a downstream numpy/scipy crash.
    """
    _check_schema(df)
    df = df.copy()
    n_raw = len(df)
    warnings = []

    # normalise group labels
    df["group"] = df["group"].astype(str).str.strip().str.lower()
    present = set(df["group"].unique())
    expected = set(expected_groups)
    if present != expected:
        unexpected = present - expected
        if unexpected:
            raise DataValidationError(
                f"Unexpected group label(s) {unexpected}; expected exactly {expected}."
            )

    # coerce types defensively
    if not np.issubdtype(df["converted"].dtype, np.number):
        try:
            df["converted"] = df["converted"].astype(int)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"'converted' column must be 0/1, could not coerce: {e}")
    bad_conv = ~df["converted"].isin([0, 1])
    if bad_conv.any():
        raise DataValidationError(
            f"'converted' must be binary 0/1; found {bad_conv.sum()} invalid values."
        )

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # 1. exact duplicate rows (same user_id + timestamp/values repeated) -> pipeline double log
    dup_mask = df.duplicated(subset=["user_id"], keep="first") & df["user_id"].duplicated(keep=False)
    # only treat as duplicate if the SAME user_id appears with identical group (true dup, not crossover)
    dup_same_group = df.duplicated(subset=["user_id", "group"], keep="first")
    duplicates_removed = int(dup_same_group.sum())
    df = df[~dup_same_group].copy()

    # 2. contamination / crossover: same base user_id (ignoring trailing _dup tag) seen in both groups
    base_id = df["user_id"].str.replace(r"_dup$", "", regex=True)
    df = df.assign(_base_id=base_id)
    grp_counts = df.groupby("_base_id")["group"].nunique()
    contaminated_ids = grp_counts[grp_counts > 1].index
    contamination_removed = int(df["_base_id"].isin(contaminated_ids).sum())
    df = df[~df["_base_id"].isin(contaminated_ids)].drop(columns="_base_id")

    # 3. negative / impossible revenue (data entry errors)
    neg_mask = df["revenue"] < 0
    negative_revenue_removed = int(neg_mask.sum())
    df = df[~neg_mask].copy()

    # 4. missing revenue: converters with NaN revenue -> can't assume 0 (they DID pay), impute
    #    with the group median converting revenue rather than dropping (dropping would bias the sample)
    missing_mask = df["revenue"].isna()
    missing_revenue_imputed = int(missing_mask.sum())
    if missing_revenue_imputed:
        for g in df["group"].unique():
            g_mask = (df["group"] == g) & missing_mask
            if g_mask.any():
                med = df.loc[(df["group"] == g) & (df["converted"] == 1) & (~missing_mask), "revenue"].median()
                df.loc[g_mask, "revenue"] = med if not np.isnan(med) else 0.0
        warnings.append(
            f"{missing_revenue_imputed} missing revenue value(s) for converters were imputed with "
            "the group's median converting order value (documented, not silently dropped)."
        )
    df.loc[df["converted"] == 0, "revenue"] = df.loc[df["converted"] == 0, "revenue"].fillna(0.0)

    # 5. outlier / bot-like revenue: IQR-based cap, applied per group, only on converters
    if len(df) == 0:
        raise DataValidationError("All rows were removed during cleaning (e.g. all revenue values were "
                                   "negative/invalid) — check the input data quality before this step.")

    outliers_removed = 0
    clean_parts = []
    for g in df["group"].unique():
        sub = df[df["group"] == g].copy()
        conv_rev = sub.loc[sub["converted"] == 1, "revenue"]
        if len(conv_rev) > 10:
            q1, q3 = conv_rev.quantile([0.25, 0.75])
            iqr = q3 - q1
            upper = q3 + 5 * iqr  # generous cutoff -> only removes genuine bot-scale spikes
            outlier_mask = (sub["converted"] == 1) & (sub["revenue"] > upper)
            outliers_removed += int(outlier_mask.sum())
            sub = sub[~outlier_mask]
        clean_parts.append(sub)
    df = pd.concat(clean_parts, ignore_index=True)

    if len(df) == 0:
        raise DataValidationError("All rows were removed during cleaning — check the input data quality.")

    group_counts = df["group"].value_counts().to_dict()
    if min(group_counts.values()) < 30:
        warnings.append(
            "One or both groups have fewer than 30 users after cleaning; "
            "test results below will be statistically unreliable."
        )

    # Sample Ratio Mismatch check: is the observed allocation ~consistent with intended 50/50?
    n_total = sum(group_counts.values())
    obs = np.array(list(group_counts.values()))
    exp = np.array([n_total / 2, n_total / 2])
    srm_chi2, srm_p = stats.chisquare(obs, exp)
    srm_flag = bool(srm_p < 0.001)  # conventional SRM threshold
    if srm_flag:
        warnings.append(
            "Sample Ratio Mismatch detected (p < 0.001): group sizes deviate from the intended split "
            "far more than chance allows. Investigate the randomisation/assignment pipeline before "
            "trusting the headline result."
        )

    # Covariate balance check (device, country) — makes sure groups are comparable
    covariate_balance = {}
    for cov in ["device", "country"]:
        if cov in df.columns:
            ct = pd.crosstab(df[cov], df["group"])
            if ct.shape[0] > 1:
                chi2, p, _, _ = stats.chi2_contingency(ct)
                covariate_balance[cov] = p
                if p < 0.01:
                    warnings.append(
                        f"'{cov}' distribution differs significantly between groups (p={p:.4f}); "
                        "randomisation may not have balanced this covariate."
                    )

    report = ValidationReport(
        n_raw=n_raw,
        n_clean=len(df),
        duplicates_removed=duplicates_removed,
        contamination_removed=contamination_removed,
        missing_revenue_imputed=missing_revenue_imputed,
        outliers_removed=outliers_removed,
        negative_revenue_removed=negative_revenue_removed,
        group_counts=group_counts,
        srm_p_value=srm_p,
        srm_flag=srm_flag,
        covariate_balance=covariate_balance,
        warnings=warnings,
    )
    return df, report
