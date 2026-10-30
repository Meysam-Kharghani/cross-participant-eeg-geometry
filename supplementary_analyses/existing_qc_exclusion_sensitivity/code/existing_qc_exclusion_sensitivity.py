#!/usr/bin/env python3
"""Post-hoc exclusion sensitivity using historical EEG quality-control metrics.

Discovery-only quantiles define stricter amplitude, robust-scale, and channel-
availability thresholds. The thresholds are then applied unchanged to held-out
splits. The analysis tests dependence on visibly poor-QC epochs; it is not a new
preprocessing pipeline and does not establish absence of ocular or muscular
contamination.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

QUANTILES = [0.95, 0.975, 0.99]
METRICS = ["maximum_abs_uv", "maximum_peak_to_peak_uv", "median_robust_scale_uv"]


def boolean_series(s):
    if s.dtype == object:
        return s.astype(str).str.lower().isin(["true", "1", "yes"])
    return s.astype(bool)


def discovery_thresholds(q, quantile):
    discovery = q[(q["split"] == "discovery") & q["modeling_eligible"]].copy()
    rows = []
    for epoch_family, g in discovery.groupby("epoch_family"):
        row = {"epoch_family": epoch_family, "upper_quantile": quantile}
        for c in METRICS:
            row[c + "_max"] = float(g[c].quantile(quantile))
        row["final_good_channels_min"] = float(
            g["final_good_channels"].quantile(1 - quantile)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def trial_pass_manifest(q, thresholds):
    threshold_map = thresholds.set_index("epoch_family").to_dict("index")
    qq = q.copy()
    strict_pass = []
    for row in qq.itertuples(index=False):
        threshold = threshold_map.get(row.epoch_family)
        good = (
            threshold is not None
            and row.modeling_eligible
            and row.recording_passed
            and row.reference_converged
            and row.epoch_passed
        )
        if good:
            for c in METRICS:
                good = good and getattr(row, c) <= threshold[c + "_max"]
            good = good and row.final_good_channels >= threshold["final_good_channels_min"]
        strict_pass.append(bool(good))
    qq["strict_epoch_pass"] = strict_pass
    p = qq.pivot_table(
        index=["split", "raw_behavior_row_id", "subject_id"],
        columns="epoch_family",
        values="strict_epoch_pass",
        aggfunc="max",
    ).reset_index()
    for c in ["stimulus_early", "stimulus_late", "rating_early"]:
        if c not in p.columns:
            p[c] = False
        p[c] = p[c].fillna(False).infer_objects(copy=False).astype(bool)
    p["strict_stimulus_pair_pass"] = p["stimulus_early"] & p["stimulus_late"]
    p["strict_transition_pair_pass"] = p["stimulus_late"] & p["rating_early"]
    p["strict_three_epoch_pass"] = (
        p["stimulus_early"] & p["stimulus_late"] & p["rating_early"]
    )
    return p


def participant_balanced_support(df):
    if "in_discovery_support" not in df.columns or df.empty:
        return np.nan
    return float(df.groupby("subject_id")["in_discovery_support"].mean().mean())


def summarize_assignments(df, pass_manifest, pass_column, branch, split, quantile):
    base = df.copy()
    if "family_1based" not in base.columns and "trajectory_family_1based" in base.columns:
        base = base.rename(columns={"trajectory_family_1based": "family_1based"})
    key = pass_manifest[["raw_behavior_row_id", pass_column]].drop_duplicates()
    filtered = base.merge(key, on="raw_behavior_row_id", how="left")
    filtered[pass_column] = filtered[pass_column].fillna(False).infer_objects(copy=False).astype(bool)
    if "k" in base.columns:
        k_values = sorted(base["k"].unique())
    else:
        k_values = [2]
        base["k"] = 2
        filtered["k"] = 2

    rows = []
    for k in k_values:
        baseline = base[base["k"] == k].copy()
        strict = filtered[(filtered["k"] == k) & filtered[pass_column]].copy()
        for condition, x in [("baseline", baseline), ("strict_qc", strict)]:
            occupancy = []
            if len(x):
                for _, g in x.groupby("subject_id"):
                    vc = g["family_1based"].astype(int).value_counts(normalize=True)
                    occupancy.append([vc.get(j, 0.0) for j in range(1, int(k) + 1)])
            min_occupancy = (
                float(np.min(np.mean(np.asarray(occupancy), axis=0))) if occupancy else np.nan
            )
            rows.append({
                "quantile": quantile,
                "branch": branch,
                "split": split,
                "k": int(k),
                "condition": condition,
                "rows": len(x),
                "participants": x["subject_id"].nunique() if len(x) else 0,
                "retained_fraction_vs_baseline": (
                    len(x) / len(baseline) if len(baseline) else np.nan
                ),
                "participant_balanced_in_support_fraction": participant_balanced_support(x),
                "min_participant_balanced_family_occupancy": min_occupancy,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    input_root = Path(args.input_root) if args.input_root else (
        repo / "results" / "reproducibility_inputs" / "existing_qc_exclusion"
    )
    multires_root = repo / "results" / "reproducibility_inputs" / "multiresolution_null"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(input_root / "epoch_qc_summary.csv")
    for c in ["recording_passed", "reference_converged", "epoch_passed", "modeling_eligible"]:
        q[c] = boolean_series(q[c])

    sources = {
        ("trajectory", "validation"): input_root / "validation_trajectory_assignments.csv",
        ("trajectory", "confirmation"): input_root / "confirmation_trajectory_assignments.csv",
        ("experience", "validation"): multires_root / "experience" / "validation_assignments.csv",
        ("experience", "confirmation"): multires_root / "experience" / "confirmation_assignments.csv",
        ("transition", "validation"): multires_root / "report_transition" / "validation_assignments.csv",
        ("transition", "confirmation"): multires_root / "report_transition" / "confirmation_assignments.csv",
    }

    summary_rows, threshold_rows = [], []
    for quantile in QUANTILES:
        thresholds = discovery_thresholds(q, quantile)
        thresholds["quantile"] = quantile
        threshold_rows.append(thresholds)
        trial_pass = trial_pass_manifest(q, thresholds)
        for (branch, split), source in sources.items():
            assignments = pd.read_csv(source)
            if "k" not in assignments.columns:
                assignments["k"] = 2
            pass_column = {
                "trajectory": "strict_three_epoch_pass",
                "experience": "strict_stimulus_pair_pass",
                "transition": "strict_transition_pair_pass",
            }[branch]
            split_pass = trial_pass[trial_pass["split"] == split]
            summary_rows += summarize_assignments(
                assignments, split_pass, pass_column, branch, split, quantile
            )

    summary = pd.DataFrame(summary_rows)
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    thresholds.to_csv(out / "discovery_thresholds.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
