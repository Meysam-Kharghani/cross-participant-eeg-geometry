#!/usr/bin/env python3
"""
Post-hoc robustness / stress tests for primary analysis trajectory geometry.

Purpose
-------
This script does NOT alter the fixed primary analysis. It adds post-hoc
sensitivity analyses aimed at a specific interpretive question:

    Is the very stable two-family trajectory solution robust to reasonable
    reparameterization, or could the apparent partition be driven mainly by
    redundant geometry features / preprocessing / generic K=2 quantization?

Analyses
--------
1) Exact raw-geometry redundancy audit.
2) Non-redundant 3-feature geometry: [early->late, late->rating, segment cosine].
3) Leave-one-feature-out K=2 sensitivity.
4) Scaling / transformation sensitivity.
5) K=1-vs-K=2 clusterability against a matched single-Gaussian null.
6) Alternative clustering algorithms (descriptive).
7) 1D separation-axis GMM diagnostic with participant-balanced subsampling.
8) Same-trial trajectory-destruction nulls, if per-epoch PCA coordinates are
   present in the released continuous trajectory table.

Expected input
--------------
An extracted PROJECT_ROOT project tree containing, at minimum:
- continuous_trajectories.csv
- S081_validation_geometry_transfer_reference.json

The script searches recursively for these files, so the exact repository
packaging level does not matter.

Example
-------
python posthoc_geometry_sensitivity_portable.py \
    --project /path/to/PROJECT_ROOT \
    --out /path/to/trajectory_geometry_sensitivity \
    --gaussian-reps 500 \
    --destruction-reps 300

Important interpretation boundary
---------------------------------
These analyses are post-hoc sensitivity analyses. They can strengthen or weaken
confidence that the fixed geometry is representation-robust, but they should
not be presented as preregistered confirmation and should not overwrite the
fixed primary results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.mixture import GaussianMixture


FEATURES6 = [
    "early_to_late_norm",
    "late_to_rating_norm",
    "early_to_rating_direct_norm",
    "total_path_length",
    "directness_ratio",
    "segment_cosine",
]

PRIMITIVE3 = [
    "early_to_late_norm",
    "late_to_rating_norm",
    "segment_cosine",
]

FIRST4 = {
    "early_to_late_norm",
    "late_to_rating_norm",
    "early_to_rating_direct_norm",
    "total_path_length",
}

IQR_FLOOR = 1e-8
CLIP_EPS = 1e-6
WINSOR_LIMIT = 6.0
SEED = 540


def find_one(root: Path, basename: str, required: bool = True) -> Path | None:
    hits = list(root.rglob(basename))
    if not hits:
        if required:
            raise FileNotFoundError(f"Could not find {basename} under {root}")
        return None
    if len(hits) > 1:
        hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def participant_weights(subjects: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects).astype(str)
    u, c = np.unique(subjects, return_counts=True)
    lookup = dict(zip(u, c))
    return np.array([1.0 / lookup[s] for s in subjects], dtype=float)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.average(x, axis=0, weights=w)


def weighted_sse(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, w: np.ndarray) -> float:
    d = x - centers[labels]
    return float(np.sum(w * np.sum(d * d, axis=1)))


def weighted_one_cluster_sse(x: np.ndarray, w: np.ndarray) -> float:
    mu = weighted_mean(x, w)
    d = x - mu
    return float(np.sum(w * np.sum(d * d, axis=1)))


def weighted_silhouette(x: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    s = silhouette_samples(x, labels, metric="euclidean")
    return float(np.average(s, weights=w))


def align_labels(candidate: np.ndarray, reference: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, dict[int, int]]:
    candidate = np.asarray(candidate, int)
    reference = np.asarray(reference, int)
    kc = int(candidate.max()) + 1
    kr = int(reference.max()) + 1
    k = max(kc, kr)
    if w is None:
        w = np.ones(len(candidate), float)
    table = np.zeros((k, k), float)
    for a, b, ww in zip(candidate, reference, w):
        table[a, b] += ww
    rr, cc = linear_sum_assignment(-table)
    mapping = {int(a): int(b) for a, b in zip(rr, cc)}
    mapped = np.array([mapping.get(int(v), int(v)) for v in candidate], int)
    return mapped, mapping


def transform_values(df: pd.DataFrame, features: list[str], mode: str = "prespecified") -> np.ndarray:
    x = df[features].to_numpy(float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite raw geometry values in features {features}")

    if mode == "identity":
        return x
    if mode != "prespecified":
        raise ValueError(mode)

    for j, f in enumerate(features):
        if f in FIRST4:
            if np.any(x[:, j] < 0):
                raise ValueError(f"{f} has negative values; log1p invalid for prespecified transform.")
            x[:, j] = np.log1p(x[:, j])
        elif f == "directness_ratio":
            d = np.clip(x[:, j], CLIP_EPS, 1.0 - CLIP_EPS)
            x[:, j] = np.log(d / (1.0 - d))
        elif f == "segment_cosine":
            c = np.clip(x[:, j], -1.0 + CLIP_EPS, 1.0 - CLIP_EPS)
            x[:, j] = np.arctanh(c)
        else:
            raise ValueError(f"Unknown geometry feature: {f}")
    return x


@dataclass
class Standardized:
    z: np.ndarray
    fallback_uses: int
    transformed: np.ndarray


def standardize_by_participant(
    df: pd.DataFrame,
    features: list[str],
    transform_mode: str = "prespecified",
    scaling: str = "median_iqr",
    winsor: bool = True,
    prespecified_fallback: dict[str, float] | None = None,
) -> Standardized:
    x = transform_values(df, features, transform_mode)
    subjects = df["subject_id"].astype(str).to_numpy()
    z = np.full_like(x, np.nan, dtype=float)
    fallback_uses = 0

    for s in pd.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        xs = x[idx]

        if scaling == "median_iqr":
            center = np.median(xs, axis=0)
            q75 = np.percentile(xs, 75, axis=0)
            q25 = np.percentile(xs, 25, axis=0)
            scale = q75 - q25
        elif scaling == "mean_sd":
            center = np.mean(xs, axis=0)
            scale = np.std(xs, axis=0, ddof=1)
        else:
            raise ValueError(scaling)

        bad = ~(np.isfinite(scale) & (scale > IQR_FLOOR))
        if bad.any():
            for j, is_bad in enumerate(bad):
                if not is_bad:
                    continue
                fallback_uses += 1
                f = features[j]
                if prespecified_fallback is not None and f in prespecified_fallback:
                    scale[j] = float(prespecified_fallback[f])
                else:
                    pooled = x[:, j]
                    if scaling == "median_iqr":
                        repl = np.percentile(pooled, 75) - np.percentile(pooled, 25)
                    else:
                        repl = np.std(pooled, ddof=1)
                    if not np.isfinite(repl) or repl <= IQR_FLOOR:
                        repl = 1.0
                    scale[j] = repl

        zz = (xs - center) / scale
        if winsor:
            zz = np.clip(zz, -WINSOR_LIMIT, WINSOR_LIMIT)
        z[idx] = zz

    if not np.isfinite(z).all():
        raise ValueError("Non-finite standardized geometry")
    return Standardized(z=z, fallback_uses=fallback_uses, transformed=x)


def fit_k2(z: np.ndarray, w: np.ndarray, seed: int = SEED, n_init: int = 100):
    km = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=n_init,
        max_iter=500,
        tol=1e-7,
        algorithm="lloyd",
    )
    km.fit(z, sample_weight=w)
    return km.labels_.astype(int), km.cluster_centers_.astype(float), km


def assign_centroids(z: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dist, axis=1)
    return labels.astype(int), dist


def lopo_stability(
    df: pd.DataFrame,
    features: list[str],
    full_labels: np.ndarray,
    transform_mode: str = "prespecified",
    scaling: str = "median_iqr",
    winsor: bool = True,
    prespecified_fallback: dict[str, float] | None = None,
    n_init: int = 50,
) -> dict:
    subjects = df["subject_id"].astype(str).to_numpy()
    agree = np.zeros(len(df), bool)
    per_subject = []
    all_fallback_uses = 0

    for s in sorted(pd.unique(subjects)):
        tr = subjects != s
        te = ~tr
        dtr = df.loc[tr].reset_index(drop=True)
        dte = df.loc[te].reset_index(drop=True)

        prep_tr = standardize_by_participant(
            dtr, features, transform_mode, scaling, winsor, prespecified_fallback
        )
        prep_te = standardize_by_participant(
            dte, features, transform_mode, scaling, winsor, prespecified_fallback
        )
        all_fallback_uses += prep_tr.fallback_uses + prep_te.fallback_uses

        sub_tr = dtr["subject_id"].astype(str).to_numpy()
        wtr = participant_weights(sub_tr)
        lab_tr, ctr, _ = fit_k2(prep_tr.z, wtr, seed=SEED + sum(map(ord, s)), n_init=n_init)

        # Align training labels to this representation's full-fit labels.
        ref_tr = full_labels[tr]
        mapped_tr, mapping = align_labels(lab_tr, ref_tr, wtr)

        raw_te, _ = assign_centroids(prep_te.z, ctr)
        mapped_te = np.array([mapping[int(v)] for v in raw_te], int)
        ref_te = full_labels[te]

        ok = mapped_te == ref_te
        agree[np.flatnonzero(te)] = ok
        per_subject.append((s, float(ok.mean()), int(ok.sum()), int(ok.size)))

    per_subject_df = pd.DataFrame(
        per_subject, columns=["subject_id", "agreement", "agree_n", "n"]
    )
    return {
        "lopo_agreement": float(agree.mean()),
        "minimum_subject_lopo_agreement": float(per_subject_df["agreement"].min()),
        "subjects_at_or_above_0_95": int((per_subject_df["agreement"] >= 0.95).sum()),
        "fallback_uses": int(all_fallback_uses),
        "per_subject": per_subject_df,
    }


def representation_summary(
    df: pd.DataFrame,
    features: list[str],
    reference_labels: np.ndarray,
    name: str,
    transform_mode: str = "prespecified",
    scaling: str = "median_iqr",
    winsor: bool = True,
    prespecified_fallback: dict[str, float] | None = None,
    lopo: bool = True,
) -> tuple[dict, pd.DataFrame | None]:
    prep = standardize_by_participant(
        df, features, transform_mode, scaling, winsor, prespecified_fallback
    )
    w = participant_weights(df["subject_id"].astype(str).to_numpy())
    labels, centers, _ = fit_k2(prep.z, w, n_init=100)
    mapped, _ = align_labels(labels, reference_labels, w)

    ari = adjusted_rand_score(reference_labels, labels)
    agreement = float(np.mean(mapped == reference_labels))
    sil = weighted_silhouette(prep.z, labels, w)
    sse1 = weighted_one_cluster_sse(prep.z, w)
    sse2 = weighted_sse(prep.z, labels, centers, w)
    gain = float((sse1 - sse2) / sse1)

    # Participant-balanced occupancy, after mapping to prespecified family IDs.
    tmp = pd.DataFrame({
        "subject_id": df["subject_id"].astype(str).to_numpy(),
        "family": mapped + 1,
    })
    occ = (
        tmp.assign(one=1)
        .groupby(["subject_id", "family"])["one"].sum()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
    )
    pb_f1 = float(occ.get(1, pd.Series(0, index=occ.index)).mean())
    pb_f2 = float(occ.get(2, pd.Series(0, index=occ.index)).mean())

    row = {
        "analysis": name,
        "features": ";".join(features),
        "n_features": len(features),
        "transform": transform_mode,
        "scaling": scaling,
        "winsor": winsor,
        "prespecified_assignment_ARI": ari,
        "prespecified_assignment_aligned_agreement": agreement,
        "participant_weighted_silhouette": sil,
        "relative_SSE_gain_K1_to_K2": gain,
        "participant_balanced_family1_occupancy": pb_f1,
        "participant_balanced_family2_occupancy": pb_f2,
        "full_fit_fallback_uses": prep.fallback_uses,
    }

    ps = None
    if lopo:
        stab = lopo_stability(
            df, features, labels,
            transform_mode=transform_mode,
            scaling=scaling,
            winsor=winsor,
            prespecified_fallback=prespecified_fallback,
            n_init=50,
        )
        row.update({
            "own_solution_LOPO_agreement": stab["lopo_agreement"],
            "own_solution_min_subject_LOPO_agreement": stab["minimum_subject_lopo_agreement"],
            "own_solution_subjects_ge_0_95": stab["subjects_at_or_above_0_95"],
            "lopo_fallback_uses": stab["fallback_uses"],
        })
        ps = stab["per_subject"]
        ps.insert(0, "analysis", name)
    return row, ps


def raw_redundancy_audit(df: pd.DataFrame) -> pd.DataFrame:
    a = df["early_to_late_norm"].to_numpy(float)
    b = df["late_to_rating_norm"].to_numpy(float)
    d = df["early_to_rating_direct_norm"].to_numpy(float)
    p = df["total_path_length"].to_numpy(float)
    r = df["directness_ratio"].to_numpy(float)
    c = df["segment_cosine"].to_numpy(float)

    eps = 1e-12
    errs = {
        "path_minus_a_plus_b": p - (a + b),
        "directness_minus_d_over_path": r - d / np.maximum(p, eps),
        "law_of_cosines_residual": d*d - (a*a + b*b + 2*a*b*c),
    }
    rows = []
    for name, e in errs.items():
        ae = np.abs(e[np.isfinite(e)])
        rows.append({
            "identity": name,
            "n": len(ae),
            "median_abs_error": float(np.median(ae)),
            "p99_abs_error": float(np.percentile(ae, 99)),
            "max_abs_error": float(np.max(ae)),
        })
    return pd.DataFrame(rows)


def gaussian_clusterability_null(z: np.ndarray, w: np.ndarray, reps: int, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    obs_lab, obs_ctr, _ = fit_k2(z, w, n_init=100)
    obs_sil = weighted_silhouette(z, obs_lab, w)
    obs_sse1 = weighted_one_cluster_sse(z, w)
    obs_sse2 = weighted_sse(z, obs_lab, obs_ctr, w)
    obs_gain = (obs_sse1 - obs_sse2) / obs_sse1

    mu = np.average(z, axis=0, weights=w)
    # np.cov with aweights gives a covariance matched to participant-balanced weights.
    cov = np.cov(z, rowvar=False, aweights=w)
    # Small ridge for numerical safety.
    cov = cov + np.eye(cov.shape[0]) * 1e-8

    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        zn = rng.multivariate_normal(mu, cov, size=len(z))
        lab, ctr, _ = fit_k2(zn, w, seed=seed + rep + 1, n_init=20)
        sil = weighted_silhouette(zn, lab, w)
        s1 = weighted_one_cluster_sse(zn, w)
        s2 = weighted_sse(zn, lab, ctr, w)
        rows.append({
            "replicate": rep,
            "participant_weighted_silhouette": sil,
            "relative_SSE_gain_K1_to_K2": (s1 - s2) / s1,
        })
    null = pd.DataFrame(rows)
    p_sil = (1 + np.sum(null["participant_weighted_silhouette"] >= obs_sil)) / (reps + 1)
    p_gain = (1 + np.sum(null["relative_SSE_gain_K1_to_K2"] >= obs_gain)) / (reps + 1)
    summary = {
        "observed_participant_weighted_silhouette": float(obs_sil),
        "gaussian_null_silhouette_mean": float(null["participant_weighted_silhouette"].mean()),
        "gaussian_null_silhouette_p95": float(null["participant_weighted_silhouette"].quantile(.95)),
        "gaussian_null_silhouette_empirical_p": float(p_sil),
        "observed_relative_SSE_gain_K1_to_K2": float(obs_gain),
        "gaussian_null_gain_mean": float(null["relative_SSE_gain_K1_to_K2"].mean()),
        "gaussian_null_gain_p95": float(null["relative_SSE_gain_K1_to_K2"].quantile(.95)),
        "gaussian_null_gain_empirical_p": float(p_gain),
    }
    return null, summary


def alternative_clustering(z: np.ndarray, reference_labels: np.ndarray) -> pd.DataFrame:
    rows = []

    # Unweighted K-means: diagnostic for whether participant balancing is driving the split.
    km = KMeans(n_clusters=2, random_state=SEED, n_init=100, max_iter=500, tol=1e-7).fit(z)
    rows.append({
        "algorithm": "unweighted_kmeans",
        "ARI_vs_prespecified": adjusted_rand_score(reference_labels, km.labels_),
    })

    # Ward: no sample weights in sklearn; descriptive only.
    ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(z)
    rows.append({
        "algorithm": "ward_hierarchical_unweighted",
        "ARI_vs_prespecified": adjusted_rand_score(reference_labels, ward.labels_),
    })

    for cov in ["full", "diag"]:
        gmm = GaussianMixture(
            n_components=2, covariance_type=cov,
            random_state=SEED, n_init=20, reg_covar=1e-6
        ).fit(z)
        rows.append({
            "algorithm": f"gmm2_{cov}_unweighted",
            "ARI_vs_prespecified": adjusted_rand_score(reference_labels, gmm.predict(z)),
        })
    return pd.DataFrame(rows)


def projection_gmm_diagnostic(
    z: np.ndarray,
    subjects: np.ndarray,
    centroids: np.ndarray,
    reps: int = 300,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    direction = centroids[1] - centroids[0]
    direction = direction / np.linalg.norm(direction)
    score = z @ direction

    # Equal-participant subsampling: same number of rows from each participant.
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects).astype(str)
    idx_by_sub = {s: np.flatnonzero(subjects == s) for s in pd.unique(subjects)}
    m = min(len(v) for v in idx_by_sub.values())

    rows = []
    for rep in range(reps):
        take = np.concatenate([rng.choice(v, size=m, replace=False) for v in idx_by_sub.values()])
        x = score[take, None]
        g1 = GaussianMixture(n_components=1, random_state=seed + rep, n_init=10).fit(x)
        g2 = GaussianMixture(n_components=2, random_state=seed + rep, n_init=20).fit(x)
        rows.append({
            "replicate": rep,
            "equal_rows_per_participant": m,
            "BIC1_minus_BIC2": float(g1.bic(x) - g2.bic(x)),
        })
    tab = pd.DataFrame(rows)
    summary = {
        "median_BIC1_minus_BIC2": float(tab["BIC1_minus_BIC2"].median()),
        "fraction_BIC_favors_2": float((tab["BIC1_minus_BIC2"] > 0).mean()),
        "fraction_BIC_strongly_favors_2_gt10": float((tab["BIC1_minus_BIC2"] > 10).mean()),
        "equal_rows_per_participant": int(m),
    }
    return tab, summary


def detect_epoch_pc_columns(df: pd.DataFrame) -> dict[str, dict[int, str]]:
    out = {"stimulus_early": {}, "stimulus_late": {}, "rating_early": {}}
    pat = re.compile(r"^(stimulus_early|stimulus_late|rating_early)_pc(\d+)$")
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            out[m.group(1)][int(m.group(2))] = col
    return out


def recompute_geometry_from_epoch_pcs(
    df: pd.DataFrame,
    rng: np.random.Generator,
    null_type: str,
    context_col: str | None = None,
) -> pd.DataFrame:
    pcs = detect_epoch_pc_columns(df)
    common_all = set(pcs["stimulus_early"]) & set(pcs["stimulus_late"]) & set(pcs["rating_early"])
    if not common_all:
        raise RuntimeError("No per-epoch PCA coordinate columns found in continuous trajectory table.")

    group_cols = ["subject_id"] + ([context_col] if context_col and context_col in df.columns else [])
    out_rows = []

    for _, g in df.groupby(group_cols, sort=False, dropna=False):
        g = g.copy().reset_index(drop=True)
        # Determine PCs valid for this group across all 3 epochs and all rows.
        valid_pcs = []
        for pc in sorted(common_all):
            cols = [pcs[e][pc] for e in ["stimulus_early","stimulus_late","rating_early"]]
            if np.isfinite(g[cols].to_numpy(float)).all():
                valid_pcs.append(pc)
        if not valid_pcs:
            raise RuntimeError(f"No common finite epoch-PC coordinates for group {g[group_cols].iloc[0].to_dict()}")

        E = g[[pcs["stimulus_early"][pc] for pc in valid_pcs]].to_numpy(float)
        L = g[[pcs["stimulus_late"][pc] for pc in valid_pcs]].to_numpy(float)
        R = g[[pcs["rating_early"][pc] for pc in valid_pcs]].to_numpy(float)

        if null_type == "rating_shuffle":
            R = R[rng.permutation(len(R))]
        elif null_type == "late_rating_independent_shuffle":
            L = L[rng.permutation(len(L))]
            R = R[rng.permutation(len(R))]
        else:
            raise ValueError(null_type)

        d1v = L - E
        d2v = R - L
        dv = R - E
        n1 = np.linalg.norm(d1v, axis=1)
        n2 = np.linalg.norm(d2v, axis=1)
        nd = np.linalg.norm(dv, axis=1)
        path = n1 + n2
        cos = np.sum(d1v * d2v, axis=1) / np.maximum(n1*n2, 1e-12)
        cos = np.clip(cos, -1, 1)
        directness = nd / np.maximum(path, 1e-12)

        h = pd.DataFrame({
            "subject_id": g["subject_id"].astype(str),
            "raw_behavior_row_id": g["raw_behavior_row_id"].astype(str),
            "trial_number": g["trial_number"].to_numpy(),
            "early_to_late_norm": n1,
            "late_to_rating_norm": n2,
            "early_to_rating_direct_norm": nd,
            "total_path_length": path,
            "directness_ratio": directness,
            "segment_cosine": cos,
        })
        if context_col and context_col in g.columns:
            h[context_col] = g[context_col].to_numpy()
        out_rows.append(h)

    return pd.concat(out_rows, ignore_index=True)


def trajectory_destruction_null(
    df: pd.DataFrame,
    prespecified_fallback: dict[str, float],
    observed_sil: float,
    observed_gain: float,
    reps: int,
    context_col: str | None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []

    for null_type in ["rating_shuffle", "late_rating_independent_shuffle"]:
        for rep in range(reps):
            dnull = recompute_geometry_from_epoch_pcs(df, rng, null_type, context_col)
            prep = standardize_by_participant(
                dnull, FEATURES6, "prespecified", "median_iqr", True, prespecified_fallback
            )
            w = participant_weights(dnull["subject_id"].astype(str).to_numpy())
            lab, ctr, _ = fit_k2(prep.z, w, seed=seed + rep, n_init=20)
            sil = weighted_silhouette(prep.z, lab, w)
            s1 = weighted_one_cluster_sse(prep.z, w)
            s2 = weighted_sse(prep.z, lab, ctr, w)
            gain = (s1 - s2) / s1
            rows.append({
                "null_type": null_type,
                "replicate": rep,
                "participant_weighted_silhouette": sil,
                "relative_SSE_gain_K1_to_K2": gain,
            })

    tab = pd.DataFrame(rows)
    summ = []
    for nt, g in tab.groupby("null_type"):
        p_sil = (1 + np.sum(g["participant_weighted_silhouette"] >= observed_sil)) / (len(g) + 1)
        p_gain = (1 + np.sum(g["relative_SSE_gain_K1_to_K2"] >= observed_gain)) / (len(g) + 1)
        summ.append({
            "null_type": nt,
            "reps": len(g),
            "observed_silhouette": observed_sil,
            "null_silhouette_mean": float(g["participant_weighted_silhouette"].mean()),
            "null_silhouette_p95": float(g["participant_weighted_silhouette"].quantile(.95)),
            "empirical_p_silhouette": float(p_sil),
            "observed_relative_SSE_gain": observed_gain,
            "null_gain_mean": float(g["relative_SSE_gain_K1_to_K2"].mean()),
            "null_gain_p95": float(g["relative_SSE_gain_K1_to_K2"].quantile(.95)),
            "empirical_p_gain": float(p_gain),
            "stratified_by_context": bool(context_col),
        })
    return tab, pd.DataFrame(summ)


def maybe_merge_context(df: pd.DataFrame, context_csv: Path | None) -> tuple[pd.DataFrame, str | None]:
    # If context already exists, use it.
    for c in ["context", "elicitation_context"]:
        if c in df.columns:
            return df, c

    if context_csv is None:
        return df, None

    cdf = pd.read_csv(context_csv, low_memory=False)
    ctx_col = next((c for c in ["context", "elicitation_context"] if c in cdf.columns), None)
    if ctx_col is None:
        raise ValueError("Context CSV supplied, but no 'context' or 'elicitation_context' column found.")

    keys = [k for k in ["subject_id", "raw_behavior_row_id"] if k in df.columns and k in cdf.columns]
    if len(keys) < 2:
        keys = [k for k in ["subject_id", "trial_number"] if k in df.columns and k in cdf.columns]
    if len(keys) < 2:
        raise ValueError("Could not find merge keys for context CSV.")

    csmall = cdf[keys + [ctx_col]].drop_duplicates(keys)
    merged = df.merge(csmall, on=keys, how="left", validate="many_to_one")
    if merged[ctx_col].isna().mean() > 0.05:
        warnings.warn(f"Context merge missingness is {merged[ctx_col].isna().mean():.1%}; using subject-only null instead.")
        return df, None
    return merged, ctx_col


def write_markdown_summary(out: Path, sections: dict):
    lines = [
        "# primary analysis trajectory-geometry stress tests",
        "",
        "**Status:** post-hoc sensitivity analysis; does not modify the fixed primary result.",
        "",
    ]

    red = sections["redundancy"]
    lines += ["## 1. Raw-geometry redundancy", "", "```text", red.to_string(index=False), "```", ""]

    reps = sections["representations"]
    lines += ["## 2. Representation / feature sensitivity", "", "```text",
              reps.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", ""]

    scal = sections["scaling"]
    lines += ["## 3. Scaling sensitivity", "", "```text",
              scal.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", ""]

    gsum = sections["gaussian_summary"]
    lines += ["## 4. Matched single-Gaussian clusterability null", ""]
    for k, v in gsum.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")

    alt = sections["alternative"]
    lines += ["## 5. Alternative clustering", "", "```text",
              alt.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", ""]

    psum = sections["projection_summary"]
    lines += ["## 6. Separation-axis GMM diagnostic", ""]
    for k, v in psum.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")

    if sections.get("destruction_summary") is not None:
        ds = sections["destruction_summary"]
        lines += ["## 7. Same-trial trajectory-destruction nulls", "", "```text",
                  ds.to_string(index=False, float_format=lambda x: f"{x:.6f}"), "```", ""]
    else:
        lines += [
            "## 7. Same-trial trajectory-destruction nulls",
            "",
            "Skipped: per-epoch PCA coordinate columns were not available in the supplied continuous-trajectory table.",
            "",
        ]

    lines += [
        "## Interpretation guide",
        "",
        "- Strong agreement of the primitive 3-feature solution with the prespecified 6-feature solution would argue that the result is not dependent on duplicating derived geometric quantities.",
        "- Strong leave-one-feature-out and scaling robustness would argue against a fragile preprocessing-specific partition.",
        "- An observed K=2 silhouette / SSE gain well above the matched one-Gaussian null would argue that the data are more clusterable than a generic single elliptical cloud.",
        "- If the trajectory-destruction null recreates the observed clusterability, the two-family structure may be largely induced by marginal epoch distributions or feature construction; if the observed structure is much stronger than the destroyed-trial null, same-trial temporal organization matters.",
        "- A 2-component GMM preference along the centroid-separation axis is only a diagnostic for distributional bimodality; it is not evidence for two biological states.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True, help="Extracted PROJECT_ROOT project root (or a parent directory).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gaussian-reps", type=int, default=500)
    ap.add_argument("--gmm-reps", type=int, default=300)
    ap.add_argument("--destruction-reps", type=int, default=300)
    ap.add_argument("--context-csv", type=Path, default=None, help="Optional table containing subject_id + raw_behavior_row_id/trial_number + context.")
    args = ap.parse_args()

    root = args.project.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    traj_path = find_one(root, "continuous_trajectories.csv", required=False)
    if traj_path is None:
        traj_path = find_one(root, "latent_state_fit_continuous_trajectories.csv")
    ref_path = find_one(root, "S081_validation_geometry_transfer_reference.json")

    print("Using trajectories:", traj_path)
    print("Using prespecified reference:", ref_path)

    df = pd.read_csv(traj_path, low_memory=False)
    required = {"subject_id", "raw_behavior_row_id", "trial_number", *FEATURES6}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Trajectory table missing required columns: {missing}")

    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    ctr = np.array([ref["fixed_centroids"]["1"], ref["fixed_centroids"]["2"]], float)
    prespecified_fallback = {k: float(v) for k, v in ref.get("pooled_fallback_iqr", {}).items()}
    # Some reference JSON versions store fallback IQR elsewhere; use known prespecified values if absent.
    if not prespecified_fallback:
        prespecified_fallback = {
            "early_to_late_norm": 0.4754957722315247,
            "late_to_rating_norm": 0.43253425451665484,
            "early_to_rating_direct_norm": 0.4071658093620769,
            "total_path_length": 0.41793243255608914,
            "directness_ratio": 1.0449506338050496,
            "segment_cosine": 0.6616508651779611,
        }

    # prespecified baseline assignment from fixed centroids.
    base_prep = standardize_by_participant(
        df, FEATURES6, "prespecified", "median_iqr", True, prespecified_fallback
    )
    prespecified_labels, dmat = assign_centroids(base_prep.z, ctr)
    w = participant_weights(df["subject_id"].astype(str).to_numpy())

    # Baseline metrics.
    base_sil = weighted_silhouette(base_prep.z, prespecified_labels, w)
    base_sse1 = weighted_one_cluster_sse(base_prep.z, w)
    base_sse2 = weighted_sse(base_prep.z, prespecified_labels, ctr, w)
    base_gain = (base_sse1 - base_sse2) / base_sse1

    print(f"prespecified assignment family counts: {np.bincount(prespecified_labels, minlength=2).tolist()}")
    print(f"prespecified participant-weighted silhouette: {base_sil:.6f}")
    print(f"prespecified relative SSE gain K1->K2: {base_gain:.6f}")

    # 1) Exact redundancy.
    redundancy = raw_redundancy_audit(df)
    redundancy.to_csv(out / "01_raw_geometry_redundancy_audit.csv", index=False)

    # 2) Primitive + leave-one-feature-out.
    rep_rows = []
    per_subject_rows = []

    baseline_row = {
        "analysis": "prespecified_fixed_6D_reference",
        "features": ";".join(FEATURES6),
        "n_features": 6,
        "transform": "prespecified",
        "scaling": "median_iqr",
        "winsor": True,
        "prespecified_assignment_ARI": 1.0,
        "prespecified_assignment_aligned_agreement": 1.0,
        "participant_weighted_silhouette": base_sil,
        "relative_SSE_gain_K1_to_K2": base_gain,
        "participant_balanced_family1_occupancy": np.nan,
        "participant_balanced_family2_occupancy": np.nan,
        "full_fit_fallback_uses": base_prep.fallback_uses,
        "own_solution_LOPO_agreement": np.nan,
        "own_solution_min_subject_LOPO_agreement": np.nan,
        "own_solution_subjects_ge_0_95": np.nan,
        "lopo_fallback_uses": np.nan,
    }
    rep_rows.append(baseline_row)

    row, ps = representation_summary(
        df, PRIMITIVE3, prespecified_labels, "primitive_3D_d1_d2_cos",
        prespecified_fallback=prespecified_fallback, lopo=True
    )
    rep_rows.append(row)
    if ps is not None:
        per_subject_rows.append(ps)

    for drop in FEATURES6:
        feats = [f for f in FEATURES6 if f != drop]
        row, ps = representation_summary(
            df, feats, prespecified_labels, f"drop_{drop}",
            prespecified_fallback=prespecified_fallback, lopo=True
        )
        rep_rows.append(row)
        if ps is not None:
            per_subject_rows.append(ps)

    reps_df = pd.DataFrame(rep_rows)
    reps_df.to_csv(out / "02_representation_feature_sensitivity.csv", index=False)
    if per_subject_rows:
        pd.concat(per_subject_rows, ignore_index=True).to_csv(
            out / "02b_representation_LOPO_by_subject.csv", index=False
        )

    # 3) Scaling sensitivity.
    scaling_specs = [
        ("prespecified_transform_medianIQR_winsor", "prespecified", "median_iqr", True),
        ("prespecified_transform_medianIQR_no_winsor", "prespecified", "median_iqr", False),
        ("prespecified_transform_meanSD_winsor", "prespecified", "mean_sd", True),
        ("raw_geometry_medianIQR_winsor", "identity", "median_iqr", True),
    ]
    scal_rows = []
    scal_ps = []
    for name, trmode, scale, wins in scaling_specs:
        row, ps = representation_summary(
            df, FEATURES6, prespecified_labels, name,
            transform_mode=trmode, scaling=scale, winsor=wins,
            prespecified_fallback=prespecified_fallback, lopo=True
        )
        scal_rows.append(row)
        if ps is not None:
            scal_ps.append(ps)
    scaling_df = pd.DataFrame(scal_rows)
    scaling_df.to_csv(out / "03_scaling_sensitivity.csv", index=False)
    if scal_ps:
        pd.concat(scal_ps, ignore_index=True).to_csv(
            out / "03b_scaling_LOPO_by_subject.csv", index=False
        )

    # 4) Gaussian clusterability null.
    gnull, gsum = gaussian_clusterability_null(
        base_prep.z, w, reps=args.gaussian_reps, seed=SEED
    )
    gnull.to_csv(out / "04_gaussian_clusterability_null_replicates.csv", index=False)
    pd.DataFrame([gsum]).to_csv(out / "04_gaussian_clusterability_null_summary.csv", index=False)

    # 5) Alternative clustering.
    alt = alternative_clustering(base_prep.z, prespecified_labels)
    alt.to_csv(out / "05_alternative_clustering.csv", index=False)

    # 6) Projection GMM.
    pgmm, psum = projection_gmm_diagnostic(
        base_prep.z,
        df["subject_id"].astype(str).to_numpy(),
        ctr,
        reps=args.gmm_reps,
        seed=SEED,
    )
    pgmm.to_csv(out / "06_separation_axis_gmm_replicates.csv", index=False)
    pd.DataFrame([psum]).to_csv(out / "06_separation_axis_gmm_summary.csv", index=False)

    # 7) Trajectory destruction null.
    destruction_summary = None
    pcs = detect_epoch_pc_columns(df)
    if all(len(v) > 0 for v in pcs.values()):
        dfx, ctx = maybe_merge_context(df, args.context_csv)
        drep, dsum = trajectory_destruction_null(
            dfx,
            prespecified_fallback,
            observed_sil=base_sil,
            observed_gain=base_gain,
            reps=args.destruction_reps,
            context_col=ctx,
            seed=SEED + 10000,
        )
        drep.to_csv(out / "07_trajectory_destruction_null_replicates.csv", index=False)
        dsum.to_csv(out / "07_trajectory_destruction_null_summary.csv", index=False)
        destruction_summary = dsum
    else:
        print("WARNING: per-epoch PCA coordinate columns were not found; trajectory-destruction null skipped.")

    # Compact machine-readable overview.
    overview = {
        "trajectory_csv": str(traj_path),
        "prespecified_reference_json": str(ref_path),
        "n_rows": int(len(df)),
        "n_participants": int(df["subject_id"].nunique()),
        "prespecified_family_counts": {
            "family1": int(np.sum(prespecified_labels == 0)),
            "family2": int(np.sum(prespecified_labels == 1)),
        },
        "prespecified_participant_weighted_silhouette": float(base_sil),
        "prespecified_relative_SSE_gain_K1_to_K2": float(base_gain),
        "gaussian_null": gsum,
        "separation_axis_gmm": psum,
        "trajectory_destruction_executed": destruction_summary is not None,
    }
    with open(out / "00_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    write_markdown_summary(
        out / "trajectory_geometry_sensitivity_summary.md",
        {
            "redundancy": redundancy,
            "representations": reps_df,
            "scaling": scaling_df,
            "gaussian_summary": gsum,
            "alternative": alt,
            "projection_summary": psum,
            "destruction_summary": destruction_summary,
        },
    )

    print("\nDone. Outputs written to:", out)


if __name__ == "__main__":
    main()
