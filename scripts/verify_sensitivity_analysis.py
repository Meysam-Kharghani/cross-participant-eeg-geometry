#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import pandas as pd

root = Path(__file__).resolve().parents[1]
primary = root / "supplementary_analyses" / "trajectory_geometry_sensitivity"
axis = root / "supplementary_analyses" / "cluster_independent_axis_sensitivity"
multires = root / "supplementary_analyses" / "multiresolution_continuous_null_sensitivity"
strict_qc = root / "supplementary_analyses" / "existing_qc_exclusion_sensitivity"
inputs = root / "results" / "reproducibility_inputs" / "trajectory_geometry"
multires_inputs = root / "results" / "reproducibility_inputs" / "multiresolution_null"
qc_inputs = root / "results" / "reproducibility_inputs" / "existing_qc_exclusion"

required = [
    inputs / "continuous_trajectories.csv",
    inputs / "trajectory_family_assignments.csv",
    inputs / "trajectory_family_candidate_metrics.csv",
    inputs / "trajectory_family_lopo_diagnostics.csv",
    inputs / "trajectory_family_stability_replicates.csv",
    inputs / "trial_mapping_manifest.csv",
    inputs / "heldout" / "validation_raw_trajectory_geometry.csv",
    inputs / "heldout" / "validation_eligible_trajectory_keys.csv",
    inputs / "heldout" / "confirmation_raw_trajectory_geometry.csv",
    inputs / "heldout" / "confirmation_eligible_trajectory_keys.csv",
    primary / "tables" / "core" / "00_primary_reproduction.json",
    primary / "tables" / "core" / "02_feature_sensitivity.csv",
    primary / "tables" / "context_preserving" / "09b_context_stratified_deranged_null_summary_400rep.csv",
    primary / "tables" / "context_preserving" / "11c_continuous_axis_LOPO_summary.csv",
    axis / "tables" / "cross_split_pca_summary.csv",
    axis / "tables" / "cross_split_axis_angles.csv",
    axis / "tables" / "context_preserving_axis_null_summary.csv",
    axis / "code" / "cluster_independent_axis_sensitivity.py",
    multires / "tables" / "summary.csv",
    multires / "tables" / "null_replicates.csv",
    multires / "code" / "multiresolution_continuous_null_sensitivity.py",
    strict_qc / "tables" / "summary.csv",
    strict_qc / "tables" / "discovery_thresholds.csv",
    strict_qc / "code" / "existing_qc_exclusion_sensitivity.py",
    multires_inputs / "experience" / "discovery_signatures.csv",
    multires_inputs / "experience" / "validation_signatures.csv",
    multires_inputs / "experience" / "confirmation_signatures.csv",
    multires_inputs / "report_transition" / "discovery_signatures.csv",
    multires_inputs / "report_transition" / "validation_signatures.csv",
    multires_inputs / "report_transition" / "confirmation_signatures.csv",
    qc_inputs / "epoch_qc_summary.csv",
    qc_inputs / "validation_trajectory_assignments.csv",
    qc_inputs / "confirmation_trajectory_assignments.csv",
]
ok = all(path.exists() for path in required)

if ok:
    q = json.loads((primary / "tables" / "core" / "00_primary_reproduction.json").read_text())
    ok &= q["n_trials"] == 1237
    ok &= q["n_participants"] == 31
    ok &= q["prespecified_counts"].get("1") == 749
    ok &= q["prespecified_counts"].get("2") == 488
    ok &= math.isclose(q["recomputed_weighted_silhouette"], 0.2892249518335221, abs_tol=1e-12)
    ok &= math.isclose(q["released_LOPO_agreement"], 0.994341147938561, abs_tol=1e-12)

    f = pd.read_csv(primary / "tables" / "core" / "02_feature_sensitivity.csv")
    rr = f.loc[f.variant == "primitive3_d1_d2_cos"].iloc[0]
    ok &= float(rr.aligned_agreement_vs_prespecified6D) > 0.96
    ok &= float(rr.own_LOPO_agreement) > 0.995

    n = pd.read_csv(primary / "tables" / "context_preserving" / "09b_context_stratified_deranged_null_summary_400rep.csv")
    ok &= len(n) == 3
    ok &= set(n.null_type) == {"late_rating_pair", "late_rating_independent", "rating_only"}
    ok &= int(n.reps.min()) == 400

    a = pd.read_csv(primary / "tables" / "context_preserving" / "11c_continuous_axis_LOPO_summary.csv")
    ok &= float(a.loc[a.representation == "prespecified6D"].iloc[0].mean_axis_cosine) > 0.999
    ok &= float(a.loc[a.representation == "primitive3D"].iloc[0].mean_axis_cosine) > 0.9999

    cs = pd.read_csv(axis / "tables" / "cross_split_pca_summary.csv")
    ang = pd.read_csv(axis / "tables" / "cross_split_axis_angles.csv")
    nu = pd.read_csv(axis / "tables" / "context_preserving_axis_null_summary.csv")

    def value(frame, **query):
        mask = pd.Series(True, index=frame.index)
        for col, val in query.items():
            mask &= frame[col].astype(str) == str(val)
        return frame.loc[mask].iloc[0]

    ok &= len(cs) == 6
    ok &= len(ang) == 6
    ok &= len(nu) == 6
    ok &= math.isclose(
        float(value(cs, representation="six_coordinate", split="Discovery").pc1_variance_fraction),
        0.5074470157028963, abs_tol=1e-12
    )
    ok &= math.isclose(
        float(value(cs, representation="primitive_three_coordinate", split="Discovery").pc1_variance_fraction),
        0.5671383067611793, abs_tol=1e-12
    )
    ok &= math.isclose(
        float(value(ang, representation="six_coordinate", split_a="Discovery", split_b="Validation").acute_axis_angle_deg),
        5.341326002820875, abs_tol=1e-10
    )
    ok &= math.isclose(
        float(value(ang, representation="six_coordinate", split_a="Discovery", split_b="Confirmation").acute_axis_angle_deg),
        8.288286444134847, abs_tol=1e-10
    )
    ok &= math.isclose(
        float(value(ang, representation="primitive_three_coordinate", split_a="Discovery", split_b="Validation").acute_axis_angle_deg),
        2.9552081092528826, abs_tol=1e-10
    )
    ok &= int(nu.reps.min()) == 400
    ok &= float(value(nu, representation="six_coordinate", null_type="rating_only").mean_axis_angle_to_observed_deg) < 6.0
    ok &= float(value(nu, representation="primitive_three_coordinate", null_type="rating_only").mean_axis_angle_to_observed_deg) < 4.0

    mr = pd.read_csv(multires / "tables" / "summary.csv")
    ok &= len(mr) == 5
    ok &= set(mr.branch) == {"experience", "transition"}
    ok &= float(mr.p_null_silhouette_ge_observed.max()) <= 0.006
    ok &= float(mr.null_validation_gate_pass_fraction.min()) == 1.0
    ok &= float(mr.null_confirmation_gate_pass_fraction.min()) == 1.0
    ok &= float(mr.null_validation_pb_support_mean.min()) > 0.9996
    ok &= float(mr.null_confirmation_pb_support_mean.min()) > 0.9996

    mrrep = pd.read_csv(multires / "tables" / "null_replicates.csv")
    ok &= len(mrrep) == 2500
    ok &= int(mrrep.replicate.max()) == 500

    sq = pd.read_csv(strict_qc / "tables" / "summary.csv")
    th = pd.read_csv(strict_qc / "tables" / "discovery_thresholds.csv")
    ok &= set(sq["quantile"].astype(float)) == {0.95, 0.975, 0.99}
    ok &= set(sq.branch) == {"trajectory", "experience", "transition"}
    strict95 = sq[(sq["quantile"].astype(float) == 0.95) & (sq.condition == "strict_qc")]
    ok &= float(strict95.participant_balanced_in_support_fraction.min()) > 0.993
    ok &= float(strict95.retained_fraction_vs_baseline.min()) > 0.65
    ok &= set(th["quantile"].astype(float)) == {0.95, 0.975, 0.99}
    qci = pd.read_csv(qc_inputs / "epoch_qc_summary.csv")
    ok &= len(qci) == 6964
    ok &= set(qci.split) == {"discovery", "validation", "confirmation"}

print(
    "PASS: repository post-hoc sensitivity checks, including trajectory geometry, cluster-independent axis, multiresolution null calibration, and existing-QC exclusion."
    if ok else
    "FAIL: one or more repository post-hoc sensitivity checks failed."
)
sys.exit(0 if ok else 1)
