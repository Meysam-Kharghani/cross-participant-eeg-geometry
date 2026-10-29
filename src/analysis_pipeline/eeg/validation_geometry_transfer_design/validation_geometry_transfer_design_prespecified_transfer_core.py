# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""prespecified pure transfer core for validation_geometry_transfer_design/validation_fixed_assignment.

This module contains only deterministic outcome-free geometry transformations
and assignment to the two fixed Discovery centroids. It performs no I/O and
must not be modified after the validation_geometry_transfer_design analysis specification."""
from __future__ import annotations
import numpy as np
import pandas as pd
FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
CLIP_EPS = 1e-06

def _iqr(x: np.ndarray) -> np.ndarray:
    return np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)

def monotone_transform(frame: pd.DataFrame) -> np.ndarray:
    """Apply the exact cross_person_alignment_design prespecified monotone transforms in feature order."""
    x = frame[FEATURES].to_numpy(float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite raw geometry values')
    x[:, 0:4] = np.log1p(x[:, 0:4])
    d = np.clip(x[:, 4], CLIP_EPS, 1.0 - CLIP_EPS)
    x[:, 4] = np.log(d / (1.0 - d))
    c = np.clip(x[:, 5], -1.0 + CLIP_EPS, 1.0 - CLIP_EPS)
    x[:, 5] = np.arctanh(c)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite transformed geometry values')
    return x

def standardize_one_participant(frame: pd.DataFrame, pooled_fallback_iqr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR standardize one participant exactly as prespecified in cross_person_alignment_design."""
    x = monotone_transform(frame)
    fallback = np.asarray(pooled_fallback_iqr, float)
    if fallback.shape != (len(FEATURES),) or not np.isfinite(fallback).all() or np.any(fallback <= IQR_FLOOR):
        raise ValueError('invalid prespecified pooled fallback IQR')
    med = np.nanmedian(x, axis=0)
    local = _iqr(x)
    used_fallback = ~(np.isfinite(local) & (local > IQR_FLOOR))
    scale = np.where(used_fallback, fallback, local)
    z = np.clip((x - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
    if not np.isfinite(z).all():
        raise ValueError('nonfinite standardized geometry values')
    return (z, med, scale, used_fallback)

def assign_fixed_families(standardized: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign to fixed centroids by squared Euclidean distance.

    Returns family labels (1-based), assigned distance, assignment margin, and
    the full n-by-2 squared-distance matrix.
    """
    z = np.asarray(standardized, float)
    ctr = np.asarray(centroids, float)
    if z.ndim != 2 or z.shape[1] != len(FEATURES):
        raise ValueError('standardized geometry has wrong shape')
    if ctr.shape != (2, len(FEATURES)) or not np.isfinite(ctr).all():
        raise ValueError('fixed centroids have wrong shape')
    dist = ((z[:, None, :] - ctr[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(dist, axis=1, kind='mergesort')
    label = order[:, 0] + 1
    assigned = dist[np.arange(len(z)), order[:, 0]]
    margin = dist[np.arange(len(z)), order[:, 1]] - assigned
    return (label.astype(np.int16), assigned, margin, dist)

def transfer_one_participant(frame: pd.DataFrame, pooled_fallback_iqr: np.ndarray, centroids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform and assign one participant; no outcome/context access."""
    required = {'subject_id', 'raw_behavior_row_id', 'trial_number', *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'missing required columns: {missing}')
    if frame['subject_id'].astype(str).nunique() != 1:
        raise ValueError('one participant required')
    z, med, scale, used = standardize_one_participant(frame, pooled_fallback_iqr)
    label, assigned, margin, all_dist = assign_fixed_families(z, centroids)
    out = frame[['subject_id', 'raw_behavior_row_id', 'trial_number']].copy()
    for j, name in enumerate(FEATURES):
        out[f'standardized_{name}'] = z[:, j]
    out['trajectory_family_1based'] = label
    out['assigned_squared_distance'] = assigned
    out['assignment_margin'] = margin
    out['distance_to_family1'] = all_dist[:, 0]
    out['distance_to_family2'] = all_dist[:, 1]
    prep = pd.DataFrame({'subject_id': [str(frame['subject_id'].iloc[0])] * len(FEATURES), 'feature_order_0based': np.arange(len(FEATURES), dtype=int), 'feature_name': FEATURES, 'within_participant_median': med, 'scale_used': scale, 'pooled_fallback_used': used})
    return (out, prep)