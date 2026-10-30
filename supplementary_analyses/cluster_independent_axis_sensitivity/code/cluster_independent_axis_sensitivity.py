#!/usr/bin/env python3
"""
Post-hoc cluster-independent continuous-axis sensitivity analysis.

This labeled sensitivity asks whether the leading participant-balanced
trajectory-geometry direction reproduces across the internal Discovery,
Validation, and Confirmation splits without using cluster assignments or
K-means centroids. It also tests whether the Discovery axis is specific to
intact same-trial temporal pairing with participant-by-context derangement
nulls.

The analysis is post hoc and does not redefine the frozen primary criteria.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES6 = [
    "early_to_late_norm", "late_to_rating_norm",
    "early_to_rating_direct_norm", "total_path_length",
    "directness_ratio", "segment_cosine",
]
PRIMITIVE3 = ["early_to_late_norm", "late_to_rating_norm", "segment_cosine"]
FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES6)}
E_COLS = [f"stimulus_early_pc{i}" for i in range(1, 9)]
L_COLS = [f"stimulus_late_pc{i}" for i in range(1, 9)]
R_COLS = [f"rating_early_pc{i}" for i in range(1, 9)]
KEYS = ["subject_id", "raw_behavior_row_id", "trial_number"]


def participant_indices(subjects: np.ndarray):
    return [np.flatnonzero(subjects == s) for s in pd.unique(subjects)]


def participant_weights(subjects: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(subjects, return_counts=True)
    lookup = dict(zip(unique, counts))
    w = np.array([1.0 / lookup[s] for s in subjects], float)
    return w / w.sum()


def geometry_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES6].to_numpy(float, copy=True)


def transform_standardize_matrix(
    geometry: np.ndarray,
    feature_indices: list[int],
    subject_rows: list[np.ndarray],
) -> np.ndarray:
    x = geometry[:, feature_indices].copy()
    for j, original_index in enumerate(feature_indices):
        if original_index <= 3:
            x[:, j] = np.log1p(x[:, j])
        elif original_index == 4:
            v = np.clip(x[:, j], 1e-6, 1.0 - 1e-6)
            x[:, j] = np.log(v / (1.0 - v))
        elif original_index == 5:
            v = np.clip(x[:, j], -1.0 + 1e-6, 1.0 - 1e-6)
            x[:, j] = np.arctanh(v)

    z = np.empty_like(x)
    pooled_iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    for idx in subject_rows:
        xs = x[idx]
        median = np.median(xs, axis=0)
        iqr = np.percentile(xs, 75, axis=0) - np.percentile(xs, 25, axis=0)
        bad = ~(np.isfinite(iqr) & (iqr > 1e-8))
        if bad.any():
            iqr[bad] = np.where(pooled_iqr[bad] > 1e-8, pooled_iqr[bad], 1.0)
        z[idx] = np.clip((xs - median) / iqr, -6.0, 6.0)
    return z


def weighted_pc1(z: np.ndarray, weights: np.ndarray):
    mean = np.sum(z * weights[:, None], axis=0)
    centered = z - mean
    covariance = (centered * weights[:, None]).T @ centered
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    axis = vectors[:, order[0]]
    axis = axis / np.linalg.norm(axis)
    return axis, float(values[0] / values.sum())


def acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(n)
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != base):
            return p
    return np.roll(base, int(rng.integers(1, n)))


def restrict_eligible(raw: pd.DataFrame, key_file: str) -> pd.DataFrame:
    keys = pd.read_csv(key_file)[KEYS]
    return raw.merge(keys, on=KEYS, how="inner", validate="one_to_one")


def prepare_split(df: pd.DataFrame):
    subjects = df["subject_id"].astype(str).to_numpy()
    subject_rows = participant_indices(subjects)
    weights = participant_weights(subjects)
    geometry = geometry_matrix(df)
    return subjects, subject_rows, weights, geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-trajectories", required=True)
    ap.add_argument("--discovery-manifest", required=True)
    ap.add_argument("--validation-trajectories", required=True)
    ap.add_argument("--validation-keys", required=True)
    ap.add_argument("--confirmation-trajectories", required=True)
    ap.add_argument("--confirmation-keys", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--null-reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    discovery = pd.read_csv(args.discovery_trajectories, low_memory=False)
    manifest = pd.read_csv(args.discovery_manifest, low_memory=False)
    discovery = discovery.merge(
        manifest[KEYS + ["context"]], on=KEYS, how="left", validate="one_to_one"
    )
    validation = restrict_eligible(
        pd.read_csv(args.validation_trajectories, low_memory=False), args.validation_keys
    )
    confirmation = restrict_eligible(
        pd.read_csv(args.confirmation_trajectories, low_memory=False), args.confirmation_keys
    )

    frames = {"Discovery": discovery, "Validation": validation, "Confirmation": confirmation}
    reps = {
        "six_coordinate": [0, 1, 2, 3, 4, 5],
        "primitive_three_coordinate": [0, 1, 5],
    }

    axes = {}
    summaries = []
    prepared = {}
    for split, frame in frames.items():
        prepared[split] = prepare_split(frame)

    for representation, indices in reps.items():
        for split, frame in frames.items():
            _, subject_rows, weights, geometry = prepared[split]
            z = transform_standardize_matrix(geometry, indices, subject_rows)
            axis, fraction = weighted_pc1(z, weights)
            axes[(representation, split)] = axis
            row = {
                "representation": representation,
                "split": split,
                "n_trials": len(frame),
                "n_participants": frame["subject_id"].nunique(),
                "pc1_variance_fraction": fraction,
            }
            for i, value in enumerate(axis, 1):
                row[f"axis_loading_{i}"] = float(value)
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / "cross_split_pca_summary.csv", index=False)

    angle_rows = []
    for representation in reps:
        for a, b in [
            ("Discovery", "Validation"),
            ("Discovery", "Confirmation"),
            ("Validation", "Confirmation"),
        ]:
            angle_rows.append({
                "representation": representation,
                "split_a": a,
                "split_b": b,
                "acute_axis_angle_deg": acute_angle_deg(
                    axes[(representation, a)], axes[(representation, b)]
                ),
            })
    pd.DataFrame(angle_rows).to_csv(out / "cross_split_axis_angles.csv", index=False)

    subjects, subject_rows, weights, _ = prepared["Discovery"]
    context = discovery["context"].astype(str).to_numpy()
    strata = []
    for subject in pd.unique(subjects):
        for ctx in pd.unique(context[subjects == subject]):
            strata.append(np.flatnonzero((subjects == subject) & (context == ctx)))

    early = discovery[E_COLS].to_numpy(float)
    late = discovery[L_COLS].to_numpy(float)
    rating = discovery[R_COLS].to_numpy(float)
    observed_axes = {r: axes[(r, "Discovery")] for r in reps}

    rng = np.random.default_rng(args.seed)
    null_rows = []
    # Order is frozen here to keep the seeded Monte Carlo output reproducible.
    null_types = ["late_rating_pair", "late_rating_independent", "rating_only"]
    for null_type in null_types:
        for replicate in range(args.null_reps):
            late2 = late.copy()
            rating2 = rating.copy()
            for idx in strata:
                n = len(idx)
                if null_type == "rating_only":
                    rating2[idx] = rating[idx][derangement(n, rng)]
                elif null_type == "late_rating_pair":
                    perm = derangement(n, rng)
                    late2[idx] = late[idx][perm]
                    rating2[idx] = rating[idx][perm]
                elif null_type == "late_rating_independent":
                    late2[idx] = late[idx][derangement(n, rng)]
                    rating2[idx] = rating[idx][derangement(n, rng)]

            first = late2 - early
            second = rating2 - late2
            direct_vector = rating2 - early
            d1 = np.linalg.norm(first, axis=1)
            d2 = np.linalg.norm(second, axis=1)
            direct = np.linalg.norm(direct_vector, axis=1)
            path = d1 + d2
            cosine = np.sum(first * second, axis=1) / np.maximum(d1 * d2, 1e-15)
            directness = direct / np.maximum(path, 1e-15)
            geometry = np.column_stack([
                d1, d2, direct, path, directness, np.clip(cosine, -1.0, 1.0)
            ])

            for representation, indices in reps.items():
                z = transform_standardize_matrix(geometry, indices, subject_rows)
                axis, fraction = weighted_pc1(z, weights)
                null_rows.append({
                    "null_type": null_type,
                    "replicate": replicate,
                    "representation": representation,
                    "axis_angle_to_observed_deg": acute_angle_deg(
                        axis, observed_axes[representation]
                    ),
                    "pc1_variance_fraction": fraction,
                })

    null_table = pd.DataFrame(null_rows)
    null_table.to_csv(out / "context_preserving_axis_null_replicates.csv", index=False)
    null_summary = (
        null_table.groupby(["representation", "null_type"], sort=False)
        .agg(
            reps=("replicate", "size"),
            mean_axis_angle_to_observed_deg=("axis_angle_to_observed_deg", "mean"),
            sd_axis_angle_to_observed_deg=("axis_angle_to_observed_deg", "std"),
            median_axis_angle_to_observed_deg=("axis_angle_to_observed_deg", "median"),
            mean_pc1_variance_fraction=("pc1_variance_fraction", "mean"),
            sd_pc1_variance_fraction=("pc1_variance_fraction", "std"),
        )
        .reset_index()
    )
    null_summary.to_csv(out / "context_preserving_axis_null_summary.csv", index=False)

    print("PASS: cluster-independent cross-split axis sensitivity completed.")


if __name__ == "__main__":
    main()
