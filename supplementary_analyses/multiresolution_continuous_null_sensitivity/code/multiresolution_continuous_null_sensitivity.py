#!/usr/bin/env python3
"""Post-hoc continuous-null calibration of multiresolution EEG structure.

The analysis compares the reported multiresolution geometry with a single
multivariate Gaussian reference matched to the participant-weighted Discovery
mean and covariance in the analyzed standardized signature space. It does not
redefine the historical structural-transfer criteria and is not a test for a
unique or intrinsically discrete latent state count.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score

SEED = 540


def participant_weights(df):
    n = df.groupby("subject_id").size()
    return df["subject_id"].map(lambda s: 1.0 / n[s]).to_numpy(float)


def feature_columns(df):
    cols = [c for c in df.columns if c.startswith("standardized__")]
    if len(cols) != 14:
        raise ValueError(f"Expected 14 standardized signature columns; found {len(cols)}")
    return cols


def weighted_covariance(X, w):
    w = np.asarray(w, float)
    w = w / w.sum()
    mu = np.sum(X * w[:, None], axis=0)
    Z = X - mu
    denom = max(1e-12, 1.0 - np.sum(w * w))
    cov = ((Z * w[:, None]).T @ Z) / denom
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov = (eigvecs * eigvals) @ eigvecs.T
    return mu, cov


def participant_balanced_fraction(subjects, values):
    d = pd.DataFrame({"subject": subjects, "value": np.asarray(values, float)})
    return float(d.groupby("subject")["value"].mean().mean())


def participant_balanced_occupancy(subjects, labels, k):
    d = pd.DataFrame({"subject": subjects, "label": labels})
    rows = []
    for _, g in d.groupby("subject"):
        vc = g["label"].value_counts(normalize=True)
        rows.append([vc.get(j, 0.0) for j in range(k)])
    return np.mean(np.asarray(rows), axis=0)


def transfer_metrics(X, subjects, centroids, threshold):
    squared_distance = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = squared_distance.argmin(axis=1)
    min_distance = squared_distance[np.arange(len(X)), labels]
    support = min_distance <= threshold
    occupancy = participant_balanced_occupancy(subjects, labels, centroids.shape[0])
    trials = np.bincount(labels, minlength=centroids.shape[0])
    participants = np.array([
        len(set(np.asarray(subjects)[labels == j])) for j in range(centroids.shape[0])
    ])
    return {
        "pb_support": participant_balanced_fraction(subjects, support),
        "min_pb_occupancy": float(occupancy.min()),
        "min_trials": int(trials.min()),
        "min_participants": int(participants.min()),
    }


def historical_metrics(assignments, k):
    x = assignments[assignments["k"] == k].copy()
    subjects = x["subject_id"].astype(str).to_numpy()
    support = x["in_discovery_support"].astype(bool).to_numpy()
    labels = x["family_1based"].astype(int).to_numpy() - 1
    occupancy = participant_balanced_occupancy(subjects, labels, k)
    trials = np.bincount(labels, minlength=k)
    participants = np.array([
        len(set(subjects[labels == j])) for j in range(k)
    ])
    return {
        "pb_support": participant_balanced_fraction(subjects, support),
        "min_pb_occupancy": float(occupancy.min()),
        "min_trials": int(trials.min()),
        "min_participants": int(participants.min()),
    }


def structural_gate(metrics):
    return bool(
        metrics["pb_support"] >= 0.80
        and metrics["min_pb_occupancy"] >= 0.05
        and metrics["min_trials"] >= 30
        and metrics["min_participants"] >= 5
    )


def branch_paths(input_root, branch):
    d = input_root / ("experience" if branch == "experience" else "report_transition")
    return {
        "Dsig": d / "discovery_signatures.csv",
        "Dass": d / "discovery_assignments.csv",
        "Vsig": d / "validation_signatures.csv",
        "Vass": d / "validation_assignments.csv",
        "Csig": d / "confirmation_signatures.csv",
        "Cass": d / "confirmation_assignments.csv",
        "thr": d / "support_thresholds.csv",
    }


def run_branch(input_root, branch, n_null, n_init):
    pp = branch_paths(input_root, branch)
    D = pd.read_csv(pp["Dsig"])
    V = pd.read_csv(pp["Vsig"])
    C = pd.read_csv(pp["Csig"])
    DA = pd.read_csv(pp["Dass"])
    VA = pd.read_csv(pp["Vass"])
    CA = pd.read_csv(pp["Cass"])
    thresholds = pd.read_csv(pp["thr"])

    cols = feature_columns(D)
    XD = D[cols].to_numpy(float)
    XV = V[cols].to_numpy(float)
    XC = C[cols].to_numpy(float)
    discovery_weights = participant_weights(D)
    mu, covariance = weighted_covariance(XD, discovery_weights)
    rng = np.random.default_rng(SEED + (0 if branch == "experience" else 10000))
    K_values = sorted(thresholds["k"].astype(int).unique())

    summaries, replicates = [], []
    for k in K_values:
        threshold = float(
            thresholds.loc[thresholds["k"] == k, "in_support_squared_distance_max"].iloc[0]
        )
        observed_fit = KMeans(
            n_clusters=k, n_init=100, max_iter=500, tol=1e-7, random_state=SEED
        ).fit(XD, sample_weight=discovery_weights)
        observed_labels = observed_fit.labels_
        silhouette = float(
            np.sum(silhouette_samples(XD, observed_labels) * discovery_weights)
            / np.sum(discovery_weights)
        )
        historical = DA[DA["k"] == k].sort_values("raw_behavior_row_id")
        refitted = pd.DataFrame({
            "raw_behavior_row_id": D["raw_behavior_row_id"],
            "refit_label": observed_labels,
        }).sort_values("raw_behavior_row_id")
        merged = historical.merge(refitted, on="raw_behavior_row_id")
        historical_ari = float(adjusted_rand_score(
            merged["family_1based"].astype(int) - 1, merged["refit_label"]
        ))
        observed_V = historical_metrics(VA, k)
        observed_C = historical_metrics(CA, k)

        null_sils, null_vs, null_cs, null_vgate, null_cgate = [], [], [], [], []
        for replicate in range(n_null):
            ND = rng.multivariate_normal(mu, covariance, size=len(D))
            NV = rng.multivariate_normal(mu, covariance, size=len(V))
            NC = rng.multivariate_normal(mu, covariance, size=len(C))
            null_fit = KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=500,
                tol=1e-7,
                random_state=SEED + replicate + 1000 * k,
            ).fit(ND, sample_weight=discovery_weights)
            null_labels = null_fit.labels_
            null_sil = float(
                np.sum(silhouette_samples(ND, null_labels) * discovery_weights)
                / np.sum(discovery_weights)
            )
            V_metrics = transfer_metrics(
                NV, V["subject_id"].astype(str).to_numpy(), null_fit.cluster_centers_, threshold
            )
            C_metrics = transfer_metrics(
                NC, C["subject_id"].astype(str).to_numpy(), null_fit.cluster_centers_, threshold
            )
            V_gate, C_gate = structural_gate(V_metrics), structural_gate(C_metrics)
            null_sils.append(null_sil)
            null_vs.append(V_metrics["pb_support"])
            null_cs.append(C_metrics["pb_support"])
            null_vgate.append(V_gate)
            null_cgate.append(C_gate)
            replicates.append({
                "branch": branch,
                "k": k,
                "replicate": replicate + 1,
                "discovery_silhouette": null_sil,
                "validation_pb_support": V_metrics["pb_support"],
                "confirmation_pb_support": C_metrics["pb_support"],
                "validation_gate_pass": V_gate,
                "confirmation_gate_pass": C_gate,
                "validation_min_pb_occupancy": V_metrics["min_pb_occupancy"],
                "confirmation_min_pb_occupancy": C_metrics["min_pb_occupancy"],
            })

        null_sils = np.asarray(null_sils)
        null_vs = np.asarray(null_vs)
        null_cs = np.asarray(null_cs)
        summaries.append({
            "branch": branch,
            "k": k,
            "historical_assignment_ari_vs_refit": historical_ari,
            "observed_discovery_weighted_silhouette": silhouette,
            "null_silhouette_mean": float(null_sils.mean()),
            "p_null_silhouette_ge_observed": float(
                (1 + (null_sils >= silhouette).sum()) / (n_null + 1)
            ),
            "observed_validation_pb_support": observed_V["pb_support"],
            "null_validation_pb_support_mean": float(null_vs.mean()),
            "p_null_validation_support_ge_observed": float(
                (1 + (null_vs >= observed_V["pb_support"]).sum()) / (n_null + 1)
            ),
            "null_validation_gate_pass_fraction": float(np.mean(null_vgate)),
            "observed_confirmation_pb_support": observed_C["pb_support"],
            "null_confirmation_pb_support_mean": float(null_cs.mean()),
            "p_null_confirmation_support_ge_observed": float(
                (1 + (null_cs >= observed_C["pb_support"]).sum()) / (n_null + 1)
            ),
            "null_confirmation_gate_pass_fraction": float(np.mean(null_cgate)),
            "support_threshold_squared_distance": threshold,
        })
    return summaries, replicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--null-reps", type=int, default=500)
    parser.add_argument("--null-n-init", type=int, default=20)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    input_root = Path(args.input_root) if args.input_root else (
        repo / "results" / "reproducibility_inputs" / "multiresolution_null"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, replicates = [], []
    for branch in ["experience", "transition"]:
        s, r = run_branch(input_root, branch, args.null_reps, args.null_n_init)
        summaries += s
        replicates += r
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(replicates).to_csv(out / "null_replicates.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
