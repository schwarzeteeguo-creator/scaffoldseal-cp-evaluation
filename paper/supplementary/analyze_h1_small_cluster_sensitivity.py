"""Zero-training small-cluster sensitivity analyses for the frozen H1 contrast.

This script reuses only accepted out-of-fold predictions.  It does not fit,
tune, select, replace, or recalibrate a model.  The preregistered block/seed
bootstrap remains the primary H1 interval; the analyses here address the
finite number and unequal sizes of the 18 sealed joint blocks.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from analyze_h1_block_influence import EXPECTED, load_frozen_data, source_table


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data"
ALPHA = 0.05


def paired_source_effects(sources: pd.DataFrame) -> pd.DataFrame:
    """Return one five-seed-averaged paired error contrast per source."""
    averaged = (
        sources.groupby(["arm", "sealed_block_id", "source"], sort=True)["source_mae"]
        .mean()
        .unstack("arm")
        .reset_index()
    )
    if averaged[["joint", "random"]].isna().any().any():
        raise RuntimeError("Every source must have predictions from both H1 arms")
    averaged["gap_joint_minus_random"] = averaged["joint"] - averaged["random"]
    if len(averaged) != 41 or averaged["sealed_block_id"].nunique() != 18:
        raise RuntimeError("Expected 41 sources nested in 18 sealed blocks")
    if not np.isclose(
        averaged["gap_joint_minus_random"].mean(), EXPECTED["gap"], rtol=0, atol=1e-12
    ):
        raise RuntimeError("Paired source contrast does not reproduce frozen H1")
    return averaged.sort_values(["sealed_block_id", "source"], kind="stable").reset_index(drop=True)


def cr1_intercept_interval(effects: pd.DataFrame) -> dict[str, float | int]:
    """Intercept-only CR1 interval with sealed blocks as independent clusters."""
    values = effects["gap_joint_minus_random"].to_numpy(float)
    estimate = float(values.mean())
    residuals = values - estimate
    cluster_scores = (
        pd.DataFrame(
            {
                "sealed_block_id": effects["sealed_block_id"].to_numpy(),
                "residual": residuals,
            }
        )
        .groupby("sealed_block_id", sort=True)["residual"]
        .sum()
        .to_numpy(float)
    )
    n = len(values)
    g = len(cluster_scores)
    se = float(np.sqrt((g / (g - 1)) * np.square(cluster_scores).sum() / (n * n)))
    critical = float(student_t.ppf(1 - ALPHA / 2, df=g - 1))
    return {
        "estimate": estimate,
        "standard_error": se,
        "degrees_of_freedom": g - 1,
        "t_statistic": estimate / se,
        "critical_value": critical,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
    }


def delete_one_block_jackknife(effects: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Delete-one-cluster jackknife interval and influence table."""
    estimate = float(effects["gap_joint_minus_random"].mean())
    rows: list[dict[str, object]] = []
    for block_id in sorted(effects["sealed_block_id"].unique()):
        kept = effects.loc[effects["sealed_block_id"] != block_id]
        rows.append(
            {
                "omitted_block_id": block_id,
                "omitted_n_sources": int((effects["sealed_block_id"] == block_id).sum()),
                "remaining_n_sources": int(len(kept)),
                "gap_joint_minus_random": float(kept["gap_joint_minus_random"].mean()),
            }
        )
    deletion = pd.DataFrame(rows)
    g = len(deletion)
    deletion_mean = float(deletion["gap_joint_minus_random"].mean())
    se = float(
        np.sqrt(
            ((g - 1) / g)
            * np.square(deletion["gap_joint_minus_random"] - deletion_mean).sum()
        )
    )
    critical = float(student_t.ppf(1 - ALPHA / 2, df=g - 1))
    summary: dict[str, float | int] = {
        "estimate": estimate,
        "standard_error": se,
        "degrees_of_freedom": g - 1,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "minimum_delete_one_estimate": float(deletion["gap_joint_minus_random"].min()),
        "maximum_delete_one_estimate": float(deletion["gap_joint_minus_random"].max()),
    }
    return summary, deletion


def exhaustive_block_sign_flip(effects: pd.DataFrame) -> dict[str, float | int]:
    """Exhaustive two-sided sign-flip test at the sealed-block level.

    Each block's contribution to the equal-source mean is multiplied by +1 or
    -1.  The test is exact under independent blocks and joint sign symmetry.
    """
    n = len(effects)
    contributions = (
        effects.groupby("sealed_block_id", sort=True)["gap_joint_minus_random"].sum()
        .to_numpy(float)
        / n
    )
    observed = float(contributions.sum())
    exceedances = 0
    total = 2 ** len(contributions)
    tolerance = 1e-14
    for signs in itertools.product((-1.0, 1.0), repeat=len(contributions)):
        statistic = float(np.dot(np.asarray(signs), contributions))
        exceedances += int(abs(statistic) >= abs(observed) - tolerance)
    return {
        "observed_mean_gap": observed,
        "n_blocks": len(contributions),
        "n_assignments": total,
        "n_two_sided_exceedances": exceedances,
        "two_sided_p_value": exceedances / total,
    }


def block_contributions(effects: pd.DataFrame) -> pd.DataFrame:
    result = (
        effects.groupby("sealed_block_id", sort=True)
        .agg(
            n_sources=("source", "size"),
            mean_source_gap=("gap_joint_minus_random", "mean"),
            sum_source_gap=("gap_joint_minus_random", "sum"),
        )
        .reset_index()
    )
    result["contribution_to_overall_source_macro_gap"] = result["sum_source_gap"] / len(effects)
    result["positive_mean_gap"] = result["mean_source_gap"] > 0
    return result


def write_report(summary: dict[str, object]) -> None:
    cr1 = summary["cr1_cluster_robust_t_interval"]
    jack = summary["delete_one_block_jackknife"]
    sign = summary["exhaustive_block_sign_flip"]
    report = f"""# H1 small-cluster sensitivity report

Status: post-confirmatory, zero-training sensitivity analysis of frozen H1 out-of-fold predictions.

## Result

- Frozen source-macro gap (joint minus molecule-random MAE): {summary['mean_gap']:.4f}.
- CR1 sealed-block-clustered interval with t(17) critical value: {cr1['ci_low']:.4f} to {cr1['ci_high']:.4f} (SE {cr1['standard_error']:.4f}).
- Delete-one-block jackknife interval with t(17) critical value: {jack['ci_low']:.4f} to {jack['ci_high']:.4f} (SE {jack['standard_error']:.4f}).
- Delete-one-block estimates ranged from {jack['minimum_delete_one_estimate']:.4f} to {jack['maximum_delete_one_estimate']:.4f}.
- Exhaustive two-sided sealed-block sign-flip test: p = {sign['two_sided_p_value']:.8f} ({sign['n_two_sided_exceedances']} of {sign['n_assignments']} assignments).
- Positive block-specific mean gaps: {summary['positive_blocks']} of 18.

## Interpretation boundary

These analyses address finite-cluster and leverage sensitivity; they do not replace the preregistered 10,000-replicate block/seed bootstrap. The CR1 and jackknife intervals rely on treating the 18 sealed blocks as independent clusters. The exact sign-flip result additionally relies on joint sign symmetry under the null. Unequal source and record counts remain part of the target benchmark rather than being removed by reweighting.
"""
    (HERE / "H1_SMALL_CLUSTER_SENSITIVITY_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows, _, hashes = load_frozen_data()
    effects = paired_source_effects(source_table(rows))
    cr1 = cr1_intercept_interval(effects)
    jackknife, deletion = delete_one_block_jackknife(effects)
    sign_flip = exhaustive_block_sign_flip(effects)
    contributions = block_contributions(effects)

    summary: dict[str, object] = {
        "schema_version": "scaffoldseal-h1-small-cluster-sensitivity-v1",
        "analysis_status": "post-confirmatory zero-training sensitivity analysis",
        "training_or_tuning_performed": False,
        "n_records": 6895,
        "n_sources": 41,
        "n_blocks": 18,
        "n_seeds": 5,
        "mean_gap": float(effects["gap_joint_minus_random"].mean()),
        "cr1_cluster_robust_t_interval": cr1,
        "delete_one_block_jackknife": jackknife,
        "exhaustive_block_sign_flip": sign_flip,
        "positive_blocks": int(contributions["positive_mean_gap"].sum()),
        "input_sha256": hashes,
    }

    effects.to_csv(SOURCE_DIR / "h1_paired_source_effects.csv", index=False, float_format="%.12f")
    contributions.to_csv(
        SOURCE_DIR / "h1_cluster_contributions.csv", index=False, float_format="%.12f"
    )
    deletion.to_csv(
        SOURCE_DIR / "h1_small_cluster_delete_one.csv", index=False, float_format="%.12f"
    )
    (SOURCE_DIR / "h1_small_cluster_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
