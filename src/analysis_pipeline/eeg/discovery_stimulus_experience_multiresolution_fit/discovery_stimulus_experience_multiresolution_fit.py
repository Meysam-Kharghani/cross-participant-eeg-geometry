# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase discovery_stimulus_experience_multiresolution_fit — Discovery-only stimulus-experience multi-resolution fit.

This phase executes the exact discovery_stimulus_experience_multiresolution_fit policy prespecified by multiresolution_experience_report_design. It reads only:
  * the multiresolution_experience_report_design prespecified Discovery stimulus_early/stimulus_late pair manifest; and
  * the already-prespecified Discovery 169-cell compact EEG epoch store.

It does NOT read rating_early EEG, behavioral reports, valence, liking, arousal,
emotion/category labels, context, stimulus identity, physiology, Validation
signal values, Confirmation signal values, or the historical F1/F2 assignments.

The analysis is intentionally multi-resolution. K=2..8 are all evaluated under
the same participant-weighted KMeans and two-level stability/collapse criteria.
Every passing resolution is retained. If none pass, no discrete taxonomy is
invented and validation_experience_transfer_design must close the discrete branch while retaining the continuous
14-cell experience-reconfiguration signature.

Numerical implementation note (non-semantic):
The fixed 14-cell coordinate order is:
  for each multiresolution_experience_report_design block in prespecified order: [RMS displacement, cosine similarity],
  then [global RMS displacement, global cosine similarity].
Cosines with one or both near-zero vectors use the deterministic convention
both-zero -> 1.0, one-zero -> 0.0. Counts are audited. This only resolves the
standard numerical zero-norm edge case; no outcome information is involved."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
PHASE = 'discovery_stimulus_experience_multiresolution_fit'
NEXT_PHASE = 'validation_experience_transfer_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
OUT_REL = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit')
CACHE_REL = Path('outputs/large/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_cache_v1')
multiresolution_experience_report_design_OUT = Path('outputs/interpretability/multiresolution_experience_report_design_multiresolution_experience_report_design_specification')
multiresolution_experience_report_design_LOCK = multiresolution_experience_report_design_OUT / 'qc/multiresolution_experience_report_design_final_output_lock_hashes.json'
multiresolution_experience_report_design_LOCK_SHA256 = 'ae65247138d1aa19cbafa05542702031d273b77503e569146710e2db1d8d467a'
multiresolution_experience_report_design_SUMMARY = multiresolution_experience_report_design_OUT / 'qc/multiresolution_experience_report_design_summary.json'
multiresolution_experience_report_design_PAIR_MANIFEST = multiresolution_experience_report_design_OUT / 'tables/multiresolution_experience_report_design_discovery_experience_pair_manifest.csv'
multiresolution_experience_report_design_SOURCE_REHASH = multiresolution_experience_report_design_OUT / 'qc/multiresolution_experience_report_design_source_rehash.csv'
multiresolution_experience_report_design_POLICY = Path('analysis_specifications/S046_multiresolution_experience_report_discovery_stimulus.json')
DISCOVERY_COMPACT = Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy')
DISCOVERY_COMPACT_SHA256 = '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'
EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256 = '4a3c90404eea73798fe8aaebe8fb48c8c4c81971fc7f8e2b1cc1f9ded4c3de79'
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ZERO_NORM_EPS = 1e-12
SIGNATURE_COORDINATES = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
TRANSFORMED_COORDINATES = [f'transformed__{x}' for x in SIGNATURE_COORDINATES]
STANDARDIZED_COORDINATES = [f'standardized__{x}' for x in SIGNATURE_COORDINATES]

@dataclass
class FitResult:
    labels: np.ndarray
    centers: np.ndarray
    objective: float
    iterations: int
    converged: bool
    assigned_squared_distance: np.ndarray
    assignment_margin: np.ndarray

@dataclass
class SignatureResult:
    raw: np.ndarray
    transformed: np.ndarray
    standardized: np.ndarray
    weights: np.ndarray
    stage_audit: pd.DataFrame
    cross_person_audit: pd.DataFrame
    zero_norm_counts: dict[str, int]

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def readj(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def stable_seed(*parts: Any) -> int:
    payload = '|'.join(['540', *[str(x) for x in parts]]).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little', signed=False)

def verify_exact(path: Path, expected: str, label: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f'{label} SHA-256 mismatch:\n  path: {path}\n  expected: {expected}\n  observed: {observed}')

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv.tmp', dir=path.parent, delete=False, encoding='utf-8') as f:
        tmp = Path(f.name)
        df.to_csv(f, index=False)
    os.replace(tmp, path)

def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json.tmp', dir=path.parent, delete=False, encoding='utf-8') as f:
        tmp = Path(f.name)
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '')
    os.replace(tmp, path)

def weighted_quantile_1d(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return float('nan')
    x = x[ok]
    w = w[ok]
    order = np.argsort(x, kind='mergesort')
    x = x[order]
    w = w[order]
    midpoint_cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, midpoint_cdf, x, left=x[0], right=x[-1]))

def weighted_iqr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.array([weighted_quantile_1d(x[:, j], w, 0.75) - weighted_quantile_1d(x[:, j], w, 0.25) for j in range(x.shape[1])], dtype=float)

def canonicalize(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(len(centers)), key=lambda i: tuple((float(v) for v in centers[i])))
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[int(v)] for v in labels], dtype=int)
    return (new_labels, centers[order].copy())

def assigned_distances_and_margins(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assigned = d[np.arange(len(x)), labels]
    alt = d.copy()
    alt[np.arange(len(x)), labels] = np.inf
    margin = np.min(alt, axis=1) - assigned
    return (assigned, margin)

def fit_weighted_kmeans(x: np.ndarray, weights: np.ndarray, k: int, seed: int, n_init: int, max_iter: int, tol: float) -> FitResult:
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite KMeans input')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError('Invalid participant weights')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = KMeans(n_clusters=k, init='k-means++', n_init=n_init, max_iter=max_iter, tol=tol, random_state=seed, algorithm='lloyd')
        labels = model.fit_predict(x, sample_weight=weights)
    labels, centers = canonicalize(labels, model.cluster_centers_)
    dist, margin = assigned_distances_and_margins(x, labels, centers)
    return FitResult(labels=labels, centers=centers, objective=float(np.sum(weights * dist)), iterations=int(model.n_iter_), converged=bool(model.n_iter_ < max_iter), assigned_squared_distance=dist, assignment_margin=margin)

def weighted_silhouette(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(x) <= len(np.unique(labels)):
        return float('nan')
    s = silhouette_samples(x, labels, metric='euclidean')
    return float(np.average(s, weights=weights))

def collapse_metrics(labels: np.ndarray, subjects: np.ndarray, weights: np.ndarray, k: int, min_participants: int, min_trials: int, min_occupancy: float) -> tuple[bool, int, int, float]:
    total = float(weights.sum())
    pcounts: list[int] = []
    ncounts: list[int] = []
    occs: list[float] = []
    for j in range(k):
        idx = labels == j
        pcounts.append(int(len(set(subjects[idx]))) if np.any(idx) else 0)
        ncounts.append(int(np.sum(idx)))
        occs.append(float(weights[idx].sum() / total) if np.any(idx) else 0.0)
    collapsed = min(pcounts) < min_participants or min(ncounts) < min_trials or min(occs) < min_occupancy
    return (collapsed, int(min(pcounts)), int(min(ncounts)), float(min(occs)))

def participant_weights(subjects: np.ndarray) -> np.ndarray:
    counts = pd.Series(subjects.astype(str)).value_counts().to_dict()
    return np.array([1.0 / counts[str(s)] for s in subjects], dtype=float)

def stage_standardize(block: np.ndarray, pooled_iqr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    block = np.asarray(block, dtype=float).copy()
    med = np.nanmedian(block, axis=0)
    if not np.isfinite(med).all():
        bad = np.flatnonzero(~np.isfinite(med))
        raise RuntimeError('Participant/stage median undefined for compact cell(s): ' + ','.join(map(str, bad[:20])))
    missing = ~np.isfinite(block)
    n_imputed = int(missing.sum())
    if n_imputed:
        block[missing] = np.broadcast_to(med, block.shape)[missing]
    q1 = np.quantile(block, 0.25, axis=0)
    q3 = np.quantile(block, 0.75, axis=0)
    iq = q3 - q1
    fallback = ~np.isfinite(iq) | (iq <= IQR_FLOOR)
    scale = np.where(fallback, pooled_iqr, iq)
    constant = ~np.isfinite(scale) | (scale <= IQR_FLOOR)
    scale = np.where(constant, 1.0, scale)
    z = np.clip((block - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite participant/stage standardized EEG values')
    return (z, med, iq, fallback, n_imputed)

def row_cosine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int, int]:
    num = np.sum(a * b, axis=1)
    na = np.sqrt(np.sum(a * a, axis=1))
    nb = np.sqrt(np.sum(b * b, axis=1))
    za = na <= ZERO_NORM_EPS
    zb = nb <= ZERO_NORM_EPS
    both = za & zb
    one = za ^ zb
    out = np.empty(len(a), dtype=float)
    regular = ~(za | zb)
    out[regular] = num[regular] / (na[regular] * nb[regular])
    out[both] = 1.0
    out[one] = 0.0
    out = np.clip(out, -1.0, 1.0)
    return (out, int(np.sum(both)), int(np.sum(one)))

def raw_signatures(early_z: np.ndarray, late_z: np.ndarray, blocks: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    if early_z.shape != late_z.shape or early_z.shape[1] != 169:
        raise RuntimeError('Unexpected standardized early/late shape')
    cols: list[np.ndarray] = []
    zero_counts: dict[str, int] = {}
    for b in blocks:
        name = str(b['name'])
        start = int(b['start'])
        stop = int(b['stop'])
        ea = early_z[:, start:stop]
        la = late_z[:, start:stop]
        rms = np.sqrt(np.mean((la - ea) ** 2, axis=1))
        cos, both, one = row_cosine(ea, la)
        cols.extend([rms, cos])
        zero_counts[f'{name}__both_zero'] = both
        zero_counts[f'{name}__one_zero'] = one
    rms = np.sqrt(np.mean((late_z - early_z) ** 2, axis=1))
    cos, both, one = row_cosine(early_z, late_z)
    cols.extend([rms, cos])
    zero_counts['global__both_zero'] = both
    zero_counts['global__one_zero'] = one
    out = np.column_stack(cols)
    if out.shape[1] != 14 or not np.isfinite(out).all():
        raise RuntimeError('Invalid raw 14-cell experience signature')
    return (out, zero_counts)

def transform_signatures(raw: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw, dtype=float)
    for j in range(0, raw.shape[1], 2):
        if np.any(raw[:, j] < 0):
            raise RuntimeError('Negative RMS displacement encountered')
        out[:, j] = np.log1p(raw[:, j])
        c = np.clip(raw[:, j + 1], -1 + 1e-06, 1 - 1e-06)
        out[:, j + 1] = np.arctanh(c)
    if not np.isfinite(out).all():
        raise RuntimeError('Nonfinite transformed experience signature')
    return out

def build_signatures(base: pd.DataFrame, early_raw: np.ndarray, late_raw: np.ndarray, blocks: list[dict[str, Any]]) -> SignatureResult:
    subjects = base.subject_id.astype(str).to_numpy()
    n = len(base)
    if early_raw.shape != (n, 169) or late_raw.shape != (n, 169):
        raise RuntimeError('Raw compact arrays do not align with pair manifest')
    early_z = np.empty_like(early_raw, dtype=float)
    late_z = np.empty_like(late_raw, dtype=float)
    stage_rows: list[dict[str, Any]] = []
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        e = np.asarray(early_raw[idx], dtype=float)
        l = np.asarray(late_raw[idx], dtype=float)
        pooled = np.vstack([e, l])
        pooled_iqr = np.nanquantile(pooled, 0.75, axis=0) - np.nanquantile(pooled, 0.25, axis=0)
        pooled_iqr = np.where(np.isfinite(pooled_iqr), pooled_iqr, 0.0)
        ez, em, eiq, efb, eimp = stage_standardize(e, pooled_iqr)
        lz, lm, liq, lfb, limp = stage_standardize(l, pooled_iqr)
        early_z[idx] = ez
        late_z[idx] = lz
        for stage, med, iq, fb, imp in [('stimulus_early', em, eiq, efb, eimp), ('stimulus_late', lm, liq, lfb, limp)]:
            stage_rows.append({'subject_id': str(subject), 'stage': stage, 'paired_trials': int(len(idx)), 'imputed_cells_total': int(imp), 'stage_iqr_fallback_cells': int(np.sum(fb)), 'pooled_iqr_constant_cells': int(np.sum(~np.isfinite(pooled_iqr) | (pooled_iqr <= IQR_FLOOR))), 'median_abs_stage_median': float(np.median(np.abs(med))), 'median_stage_iqr': float(np.median(iq[np.isfinite(iq)])) if np.isfinite(iq).any() else float('nan')})
    raw, zero_counts = raw_signatures(early_z, late_z, blocks)
    transformed = transform_signatures(raw)
    weights = participant_weights(subjects)
    pooled_iq = weighted_iqr(transformed, weights)
    pooled_iq = np.where(np.isfinite(pooled_iq), pooled_iq, 0.0)
    standardized = np.empty_like(transformed)
    xp_rows: list[dict[str, Any]] = []
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        x = transformed[idx]
        med = np.median(x, axis=0)
        iq = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        fb = ~np.isfinite(iq) | (iq <= IQR_FLOOR)
        scale = np.where(fb, pooled_iq, iq)
        constant = ~np.isfinite(scale) | (scale <= IQR_FLOOR)
        scale = np.where(constant, 1.0, scale)
        standardized[idx] = np.clip((x - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
        xp_rows.append({'subject_id': str(subject), 'paired_trials': int(len(idx)), 'signature_iqr_fallback_cells': int(np.sum(fb)), 'signature_constant_cells_after_pooled_fallback': int(np.sum(constant))})
    if not np.isfinite(standardized).all():
        raise RuntimeError('Nonfinite cross-person standardized signatures')
    return SignatureResult(raw=raw, transformed=transformed, standardized=standardized, weights=weights, stage_audit=pd.DataFrame(stage_rows), cross_person_audit=pd.DataFrame(xp_rows), zero_norm_counts=zero_counts)

def cache_meta(script_sha: str, compact_sha: str, pair_sha: str, policy_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'script_sha256': script_sha, 'Discovery_compact_sha256': compact_sha, 'multiresolution_experience_report_design_pair_manifest_sha256': pair_sha, 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': policy_sha}

def prepare_cache(root: Path, expected: dict[str, Any], recompute: bool) -> None:
    root = Path(root)
    meta_path = root / 'meta.json'
    if recompute and root.exists():
        shutil.rmtree(root)
    if root.exists():
        try:
            if readj(meta_path) != expected:
                shutil.rmtree(root)
        except Exception:
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        atomic_json(meta_path, expected)

def save_fit(path: Path, fit: FitResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, labels=fit.labels, centers=fit.centers, objective=np.array([fit.objective]), iterations=np.array([fit.iterations]), converged=np.array([int(fit.converged)]), assigned_squared_distance=fit.assigned_squared_distance, assignment_margin=fit.assignment_margin)
    os.replace(tmp, path)

def load_fit(path: Path) -> FitResult:
    d = np.load(path, allow_pickle=False)
    return FitResult(labels=d['labels'].astype(int), centers=d['centers'].astype(float), objective=float(d['objective'][0]), iterations=int(d['iterations'][0]), converged=bool(d['converged'][0]), assigned_squared_distance=d['assigned_squared_distance'].astype(float), assignment_margin=d['assignment_margin'].astype(float))

def verify_multiresolution_experience_report_design(project: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    lock_path = project / multiresolution_experience_report_design_LOCK
    verify_exact(lock_path, multiresolution_experience_report_design_LOCK_SHA256, 'multiresolution_experience_report_design final lock')
    lock = readj(lock_path)
    if lock.get('phase') != 'multiresolution_experience_report_design':
        raise RuntimeError('multiresolution_experience_report_design phase mismatch')
    if lock.get('next_phase') != PHASE:
        raise RuntimeError(f'multiresolution_experience_report_design does not authorize {PHASE}')
    if lock.get('new_signal_inference_performed') is not False:
        raise RuntimeError('multiresolution_experience_report_design boundary drift: unexpected signal inference')
    if lock.get('behavioral_values_read') is not False:
        raise RuntimeError('multiresolution_experience_report_design boundary drift: behavior was read')
    if lock.get('Validation_signal_values_read') is not False:
        raise RuntimeError('multiresolution_experience_report_design boundary drift: Validation was read')
    if lock.get('Confirmation_signal_values_read') is not False:
        raise RuntimeError('multiresolution_experience_report_design boundary drift: Confirmation was read')
    for rel, expected in lock.get('hashes', {}).items():
        verify_exact(project / rel, expected, f'multiresolution_experience_report_design fixed child {rel}')
    summary = readj(project / multiresolution_experience_report_design_SUMMARY)
    if summary.get('next_phase') != PHASE:
        raise RuntimeError('multiresolution_experience_report_design summary next-phase drift')
    if summary.get('discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256') != EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256:
        raise RuntimeError('multiresolution_experience_report_design summary policy canonical SHA drift')
    policy = readj(project / multiresolution_experience_report_design_POLICY)
    if canonical_sha(policy) != EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256:
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit prespecified policy canonical SHA mismatch')
    if policy.get('phase') != PHASE or policy.get('discovery_only') is not True:
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit prespecified policy identity drift')
    permitted = policy['data_values_permitted']
    allowed_true = {k for k, v in permitted.items() if bval(v)}
    if allowed_true != {'stimulus_early_compact_EEG', 'stimulus_late_compact_EEG'}:
        raise RuntimeError(f'Unexpected discovery_stimulus_experience_multiresolution_fit data permission set: {sorted(allowed_true)}')
    if policy['multi_resolution_family_discovery']['candidate_K'] != [2, 3, 4, 5, 6, 7, 8]:
        raise RuntimeError('Candidate-K drift')
    if int(policy['representation']['source_cells_per_epoch']) != 169:
        raise RuntimeError('Compact width drift')
    blocks = policy['representation']['blocks']
    if [(int(b['start']), int(b['stop'])) for b in blocks] != [(0, 60), (60, 72), (72, 84), (84, 124), (124, 145), (145, 169)]:
        raise RuntimeError('prespecified compact block boundary drift')
    rehash = pd.read_csv(project / multiresolution_experience_report_design_SOURCE_REHASH)
    row = rehash[rehash.source.astype(str).eq('Discovery_compact_EEG_array')]
    if len(row) != 1 or str(row.iloc[0].sha256) != DISCOVERY_COMPACT_SHA256:
        raise RuntimeError('multiresolution_experience_report_design Discovery compact source rehash drift')
    verify_exact(project / DISCOVERY_COMPACT, DISCOVERY_COMPACT_SHA256, 'Discovery compact EEG array')
    return (lock, policy, rehash)

def load_prespecified_pairs(project: Path) -> pd.DataFrame:
    need = ['raw_behavior_row_id', 'subject_id', 'stimulus_early_row_0based', 'stimulus_late_row_0based', 'experience_paired_trials', 'experience_eligible']
    header = pd.read_csv(project / multiresolution_experience_report_design_PAIR_MANIFEST, nrows=0)
    missing = set(need) - set(header.columns)
    if missing:
        raise RuntimeError(f'multiresolution_experience_report_design pair-manifest schema drift: {sorted(missing)}')
    df = pd.read_csv(project / multiresolution_experience_report_design_PAIR_MANIFEST, usecols=need, low_memory=False)
    df['raw_behavior_row_id'] = df['raw_behavior_row_id'].astype(str)
    df['subject_id'] = df['subject_id'].astype(str)
    df = df[df.experience_eligible.map(bval)].copy()
    if df.empty:
        raise RuntimeError('No multiresolution_experience_report_design experience-eligible pairs')
    if df.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Duplicate raw_behavior_row_id in prespecified experience cohort')
    if df.subject_id.nunique() != 31 or len(df) != 1261:
        raise RuntimeError(f'prespecified multiresolution_experience_report_design cohort drift: n={len(df)}, subjects={df.subject_id.nunique()}')
    if int(df.groupby('subject_id').size().min()) < 30:
        raise RuntimeError('prespecified multiresolution_experience_report_design minimum paired-trial rule drift')
    return df.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def cross_resolution_tables(base: pd.DataFrame, weights: np.ndarray, labels_by_k: dict[int, np.ndarray], eligible_k: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    purity_rows: list[dict[str, Any]] = []
    for a_i, ka in enumerate(eligible_k):
        for kb in eligible_k[a_i + 1:]:
            la = labels_by_k[ka]
            lb = labels_by_k[kb]
            metrics_rows.append({'k_a': ka, 'k_b': kb, 'ARI': float(adjusted_rand_score(la, lb)), 'NMI': float(normalized_mutual_info_score(la, lb))})
            for fa in range(ka):
                for fb in range(kb):
                    idx = (la == fa) & (lb == fb)
                    overlap_rows.append({'k_a': ka, 'family_a_1based': fa + 1, 'k_b': kb, 'family_b_1based': fb + 1, 'trial_count': int(idx.sum()), 'participant_weighted_mass': float(weights[idx].sum())})
    for coarse in eligible_k:
        for fine in eligible_k:
            if fine <= coarse:
                continue
            lc = labels_by_k[coarse]
            lf = labels_by_k[fine]
            for ff in range(fine):
                idxf = lf == ff
                if not np.any(idxf):
                    continue
                trial_counts = np.array([np.sum(idxf & (lc == cc)) for cc in range(coarse)], dtype=float)
                weighted_mass = np.array([weights[idxf & (lc == cc)].sum() for cc in range(coarse)], dtype=float)
                best_count = int(np.argmax(trial_counts))
                best_weight = int(np.argmax(weighted_mass))
                purity_rows.append({'coarse_k': coarse, 'fine_k': fine, 'fine_family_1based': ff + 1, 'best_parent_by_trial_count_1based': best_count + 1, 'trial_count_purity': float(trial_counts[best_count] / trial_counts.sum()), 'best_parent_by_participant_weight_1based': best_weight + 1, 'participant_weighted_purity': float(weighted_mass[best_weight] / weighted_mass.sum()), 'fine_family_trials': int(idxf.sum()), 'fine_family_participant_weighted_mass': float(weights[idxf].sum())})
    return (pd.DataFrame(metrics_rows), pd.DataFrame(overlap_rows), pd.DataFrame(purity_rows))

def self_test() -> None:
    blocks = [{'name': 'b1', 'start': 0, 'stop': 60}, {'name': 'b2', 'start': 60, 'stop': 72}, {'name': 'b3', 'start': 72, 'stop': 84}, {'name': 'b4', 'start': 84, 'stop': 124}, {'name': 'b5', 'start': 124, 'stop': 145}, {'name': 'b6', 'start': 145, 'stop': 169}]
    rng = np.random.default_rng(123)
    nsub, ntrial = (10, 12)
    subjects = np.repeat([f's{i:02d}' for i in range(nsub)], ntrial)
    base = pd.DataFrame({'raw_behavior_row_id': [f'r{i}' for i in range(nsub * ntrial)], 'subject_id': subjects})
    early = rng.normal(size=(len(base), 169))
    late = early + rng.normal(scale=0.4, size=early.shape)
    late[np.arange(len(base)) % 3 == 0, :60] += 1.2
    sig = build_signatures(base, early, late, blocks)
    assert sig.raw.shape == (len(base), 14)
    assert sig.transformed.shape == (len(base), 14)
    assert sig.standardized.shape == (len(base), 14)
    assert np.isfinite(sig.standardized).all()
    assert abs(sig.weights.sum() - nsub) < 1e-09
    fit = fit_weighted_kmeans(sig.standardized, sig.weights, 3, 540, n_init=10, max_iter=200, tol=1e-07)
    assert set(np.unique(fit.labels)) == {0, 1, 2}
    assert fit.centers.shape == (3, 14)
    tuples = [tuple(x) for x in fit.centers]
    assert tuples == sorted(tuples)
    z = np.zeros((2, 3))
    o = np.ones((2, 3))
    c, both, one = row_cosine(z, z)
    assert np.allclose(c, 1) and both == 2 and (one == 0)
    c, both, one = row_cosine(z, o)
    assert np.allclose(c, 0) and both == 0 and (one == 2)
    print('discovery_stimulus_experience_multiresolution_fit self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--recompute-cache', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    print(f'[{now()}] verifying exact multiresolution_experience_report_design analysis specification and Discovery compact store', flush=True)
    multiresolution_experience_report_design_lock, policy, _ = verify_multiresolution_experience_report_design(project)
    pair_manifest_sha = sha256(project / multiresolution_experience_report_design_PAIR_MANIFEST)
    pair_expected = multiresolution_experience_report_design_lock['hashes'][str(multiresolution_experience_report_design_PAIR_MANIFEST)]
    if pair_manifest_sha != pair_expected:
        raise RuntimeError('multiresolution_experience_report_design pair-manifest lock mismatch')
    base = load_prespecified_pairs(project)
    compact = np.load(project / DISCOVERY_COMPACT, mmap_mode='r', allow_pickle=False)
    if compact.ndim != 2 or compact.shape[1] != 169:
        raise RuntimeError(f'Unexpected Discovery compact shape: {compact.shape}')
    erow = base.stimulus_early_row_0based.to_numpy(int)
    lrow = base.stimulus_late_row_0based.to_numpy(int)
    if np.min(erow) < 0 or np.min(lrow) < 0 or np.max(erow) >= len(compact) or (np.max(lrow) >= len(compact)):
        raise RuntimeError('prespecified pair manifest references compact rows outside the store')
    print(f'[{now()}] opening only prespecified Discovery stimulus_early/stimulus_late compact EEG values', flush=True)
    early = np.asarray(compact[erow], dtype=np.float64)
    late = np.asarray(compact[lrow], dtype=np.float64)
    del compact
    blocks = policy['representation']['blocks']
    print(f'[{now()}] constructing the prespecified 14-cell stimulus-only reconfiguration signature', flush=True)
    full_sig = build_signatures(base, early, late, blocks)
    mres = policy['multi_resolution_family_discovery']
    ks = [int(x) for x in mres['candidate_K']]
    n_init = int(mres['n_init'])
    max_iter = int(mres['max_iter'])
    tol = float(mres['tol'])
    stab = mres['stability']
    repeats = int(stab['repeats'])
    pfrac = float(stab['participant_fraction_without_replacement'])
    tfrac = float(stab['trial_fraction_within_retained_participant_without_replacement'])
    thr = stab['eligibility_thresholds_reused_from_cross_person_alignment_fit']
    median_min = float(thr['median_ARI_at_least'])
    p10_min = float(thr['ARI_10th_percentile_at_least'])
    collapse_max = float(thr['collapse_rate_at_most'])
    min_participants = int(thr['minimum_participants_per_family'])
    min_trials = int(thr['minimum_trials_per_family'])
    min_occ = float(thr['minimum_participant_weighted_occupancy'])
    script_sha = sha256(Path(__file__).resolve())
    cache_root = project / CACHE_REL
    prepare_cache(cache_root, cache_meta(script_sha, DISCOVERY_COMPACT_SHA256, pair_manifest_sha, EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256), args.recompute_cache)
    out = project / OUT_REL
    final_lock = out / 'qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json'
    if out.exists():
        if final_lock.is_file() and (not args.force):
            raise RuntimeError(f'Completed discovery_stimulus_experience_multiresolution_fit output exists: {out}\nUse --force only if intentionally rematerializing identical analysis.')
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    participants = np.array(sorted(base.subject_id.astype(str).unique()), dtype=object)
    subjects = base.subject_id.astype(str).to_numpy()
    n_keep_p = max(max(ks), int(round(pfrac * len(participants))))
    full_fits: dict[int, FitResult] = {}
    stability_parts: list[pd.DataFrame] = []
    for k in ks:
        print(f'[{now()}] K={k}: full participant-weighted fit + {repeats} prespecified two-level stability resamples', flush=True)
        kroot = cache_root / f'k{k}'
        kroot.mkdir(parents=True, exist_ok=True)
        fit_path = kroot / 'full_fit.npz'
        if fit_path.is_file():
            fit = load_fit(fit_path)
        else:
            fit = fit_weighted_kmeans(full_sig.standardized, full_sig.weights, k, stable_seed('full', k), n_init=n_init, max_iter=max_iter, tol=tol)
            save_fit(fit_path, fit)
        full_fits[k] = fit
        stab_path = kroot / 'stability.csv'
        existing = pd.read_csv(stab_path) if stab_path.is_file() else pd.DataFrame()
        done = set(existing['repeat'].astype(int)) if len(existing) else set()
        rows = existing.to_dict('records') if len(existing) else []
        for r in range(repeats):
            if r in done:
                continue
            rng = np.random.default_rng(stable_seed('resample', k, r))
            kept_p = rng.choice(participants, size=n_keep_p, replace=False)
            idx_parts: list[np.ndarray] = []
            for subject in kept_p:
                idx_s = np.flatnonzero(subjects == str(subject))
                n_keep_t = max(2, int(round(tfrac * len(idx_s))))
                idx_parts.append(rng.choice(idx_s, size=n_keep_t, replace=False))
            idx = np.sort(np.concatenate(idx_parts))
            sub_base = base.iloc[idx].reset_index(drop=True)
            try:
                sub_sig = build_signatures(sub_base, early[idx], late[idx], blocks)
                rf = fit_weighted_kmeans(sub_sig.standardized, sub_sig.weights, k, stable_seed('resample-fit', k, r), n_init=n_init, max_iter=max_iter, tol=tol)
                ari = float(adjusted_rand_score(fit.labels[idx], rf.labels))
                collapsed, minp, minn, mino = collapse_metrics(rf.labels, sub_base.subject_id.astype(str).to_numpy(), sub_sig.weights, k, min_participants, min_trials, min_occ)
                row = {'k': k, 'repeat': r, 'retained_participants': int(len(kept_p)), 'retained_trials': int(len(idx)), 'ari_vs_full': ari, 'collapsed': bool(collapsed), 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_participant_weighted_occupancy': mino, 'objective': rf.objective, 'converged': rf.converged, 'failure': ''}
            except Exception as exc:
                row = {'k': k, 'repeat': r, 'retained_participants': int(len(kept_p)), 'retained_trials': int(len(idx)), 'ari_vs_full': np.nan, 'collapsed': True, 'minimum_participants_per_family': 0, 'minimum_trials_per_family': 0, 'minimum_participant_weighted_occupancy': 0.0, 'objective': np.nan, 'converged': False, 'failure': f'{type(exc).__name__}: {exc}'}
            rows.append(row)
            if (r + 1) % 5 == 0 or r + 1 == repeats:
                atomic_csv(stab_path, pd.DataFrame(rows).sort_values('repeat'))
                print(f'  K={k}: stability checkpoint {r + 1}/{repeats}', flush=True)
        stability_parts.append(pd.DataFrame(rows).sort_values('repeat'))
    stability = pd.concat(stability_parts, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    assignment_rows: list[pd.DataFrame] = []
    centroid_rows: list[dict[str, Any]] = []
    for k in ks:
        fit = full_fits[k]
        labels_by_k[k] = fit.labels
        reps = stability[stability.k.eq(k)].copy()
        valid = reps[np.isfinite(reps.ari_vs_full)]
        collapsed, minp, minn, mino = collapse_metrics(fit.labels, subjects, full_sig.weights, k, min_participants, min_trials, min_occ)
        median_ari = float(valid.ari_vs_full.median()) if len(valid) else float('nan')
        p10 = float(valid.ari_vs_full.quantile(0.1)) if len(valid) else float('nan')
        collapse_rate = float(reps.collapsed.map(bval).mean()) if len(reps) else 1.0
        silhouette = weighted_silhouette(full_sig.standardized, fit.labels, full_sig.weights)
        eligible = bool(fit.converged and (not collapsed) and (len(valid) == repeats) and (median_ari >= median_min) and (p10 >= p10_min) and (collapse_rate <= collapse_max))
        metric_rows.append({'k': k, 'full_fit_converged': fit.converged, 'full_fit_iterations': fit.iterations, 'objective': fit.objective, 'participant_weighted_silhouette': silhouette, 'full_fit_collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_participant_weighted_occupancy': mino, 'stability_valid_repeats': int(len(valid)), 'stability_failed_repeats': int(repeats - len(valid)), 'median_ari': median_ari, 'ari_10th_percentile': p10, 'collapse_rate': collapse_rate, 'eligible': eligible})
        adf = base[['raw_behavior_row_id', 'subject_id']].copy()
        adf['k'] = k
        adf['family_1based'] = fit.labels + 1
        adf['assigned_squared_distance'] = fit.assigned_squared_distance
        adf['assignment_margin'] = fit.assignment_margin
        adf['participant_weight'] = full_sig.weights
        assignment_rows.append(adf)
        for fam in range(k):
            idxf = fit.labels == fam
            row: dict[str, Any] = {'k': k, 'family_1based': fam + 1, 'trial_count': int(idxf.sum()), 'participant_count': int(len(set(subjects[idxf]))), 'participant_weighted_occupancy': float(full_sig.weights[idxf].sum() / full_sig.weights.sum())}
            for j, name in enumerate(STANDARDIZED_COORDINATES):
                row[name] = float(fit.centers[fam, j])
            centroid_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values('k').reset_index(drop=True)
    eligible_k = metrics.loc[metrics.eligible.map(bval), 'k'].astype(int).tolist()
    assignments = pd.concat(assignment_rows, ignore_index=True)
    centroids = pd.DataFrame(centroid_rows)
    cross_metrics, overlap, purity = cross_resolution_tables(base, full_sig.weights, labels_by_k, eligible_k)
    sig_table = base[['raw_behavior_row_id', 'subject_id']].copy()
    for j, name in enumerate(SIGNATURE_COORDINATES):
        sig_table[name] = full_sig.raw[:, j]
    for j, name in enumerate(TRANSFORMED_COORDINATES):
        sig_table[name] = full_sig.transformed[:, j]
    for j, name in enumerate(STANDARDIZED_COORDINATES):
        sig_table[name] = full_sig.standardized[:, j]
    sig_table['participant_weight'] = full_sig.weights
    coord_table = pd.DataFrame({'coordinate_0based': np.arange(14, dtype=int), 'raw_name': SIGNATURE_COORDINATES, 'transformed_name': TRANSFORMED_COORDINATES, 'standardized_name': STANDARDIZED_COORDINATES, 'raw_transform': ['log1p' if i % 2 == 0 else 'arctanh_clip_1e-6' for i in range(14)]})
    zero_table = pd.DataFrame([{'signature_scope': k, 'trial_count': v} for k, v in sorted(full_sig.zero_norm_counts.items())])
    source_rehash = pd.DataFrame([{'source': 'multiresolution_experience_report_design_final_lock', 'path': str(multiresolution_experience_report_design_LOCK), 'sha256': multiresolution_experience_report_design_LOCK_SHA256, 'verified': True, 'values_opened': False}, {'source': 'multiresolution_experience_report_design_prespecified_experience_pair_manifest', 'path': str(multiresolution_experience_report_design_PAIR_MANIFEST), 'sha256': pair_manifest_sha, 'verified': True, 'values_opened': 'structural_identifiers_only'}, {'source': 'Discovery_compact_EEG_array', 'path': str(DISCOVERY_COMPACT), 'sha256': DISCOVERY_COMPACT_SHA256, 'verified': True, 'values_opened': 'stimulus_early_and_stimulus_late_rows_only'}])
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv', sig_table)
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_candidate_metrics.csv', metrics)
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_stability_resamples.csv', stability.sort_values(['k', 'repeat']))
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv', assignments.sort_values(['k', 'subject_id', 'raw_behavior_row_id']))
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_centroids_all_k.csv', centroids.sort_values(['k', 'family_1based']))
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_cross_resolution_metrics_eligible_k.csv', cross_metrics)
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_overlap_eligible_k.csv', overlap)
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_fine_to_coarse_parent_purity.csv', purity)
    atomic_csv(out / 'tables/discovery_stimulus_experience_multiresolution_fit_signature_coordinate_schema.csv', coord_table)
    atomic_csv(out / 'qc/discovery_stimulus_experience_multiresolution_fit_participant_stage_preprocessing_audit.csv', full_sig.stage_audit.sort_values(['subject_id', 'stage']))
    atomic_csv(out / 'qc/discovery_stimulus_experience_multiresolution_fit_cross_person_standardization_audit.csv', full_sig.cross_person_audit.sort_values('subject_id'))
    atomic_csv(out / 'qc/discovery_stimulus_experience_multiresolution_fit_zero_norm_cosine_audit.csv', zero_table)
    atomic_csv(out / 'qc/discovery_stimulus_experience_multiresolution_fit_source_rehash.csv', source_rehash)
    status = 'one_or_more_stable_experience_resolutions' if eligible_k else 'no_stable_discrete_experience_resolution_continuous_signature_only'
    summary = {'phase': PHASE, 'generated_at': now(), 'status': status, 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256, 'cohort': {'Discovery_experience_eligible_participants': int(base.subject_id.nunique()), 'Discovery_experience_pairs': int(len(base)), 'minimum_pairs_per_participant': int(base.groupby('subject_id').size().min()), 'maximum_pairs_per_participant': int(base.groupby('subject_id').size().max())}, 'representation': {'source_compact_cells_per_epoch': 169, 'experience_signature_cells': 14, 'signature_coordinate_order': SIGNATURE_COORDINATES, 'rating_early_included': False}, 'candidate_K': ks, 'eligible_K': eligible_k, 'candidate_metrics': metrics.to_dict('records'), 'scientific_boundaries': {'Discovery_stimulus_early_values_read': True, 'Discovery_stimulus_late_values_read': True, 'Discovery_rating_early_values_read': False, 'behavioral_report_values_read': False, 'emotion_category_values_read': False, 'context_values_read': False, 'stimulus_identity_values_read': False, 'physiology_values_read': False, 'historical_F1_F2_assignments_read_or_used_as_targets': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE, 'next_phase_role': 'specification Discovery templates/support thresholds and one-shot Validation stimulus-only structural transfer; if no K is eligible, formally close the discrete experience-family branch and retain continuous signatures'}
    writej(out / 'qc/discovery_stimulus_experience_multiresolution_fit_summary.json', summary)
    report_lines = ['# discovery_stimulus_experience_multiresolution_fit — Discovery stimulus-only multi-resolution experience fit', '', f'Status: **{status}**', '', '## prespecified boundary', 'Only Discovery `stimulus_early` and `stimulus_late` compact EEG values were opened. `rating_early`, behavior, category, context, stimulus identity, physiology, Validation, Confirmation, and the historical F1/F2 assignments were not read.', '', '## Cohort', f'- eligible participants: **{base.subject_id.nunique()}**', f'- paired experience trials: **{len(base)}**', '', '## Resolution scan', '```text', metrics.to_string(index=False), '```', '', f"Eligible K resolutions: **{(eligible_k if eligible_k else 'none')}**", '', 'All passing K values are retained. No single K is promoted merely for simplicity, and the old three-epoch K=2 family solution was not reused as a target.', '', '## Claim boundary', 'These are outcome-free stimulus-experience reconfiguration families, not emotion labels. No family may be called joy, fear, positive, negative, valence, liking, arousal, or another psychological construct from this phase.', '', '## Next', f'**{NEXT_PHASE}** — specification the held-out structural transfer rules before opening Validation signal values.', '']
    (out / 'reports/discovery_stimulus_experience_multiresolution_fit_report.md').write_text(''.join(report_lines), encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'status': status, 'source_code_sha256': script_sha, 'multiresolution_experience_report_design_final_lock_sha256': multiresolution_experience_report_design_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256, 'Discovery_compact_sha256': DISCOVERY_COMPACT_SHA256, 'multiresolution_experience_report_design_experience_pair_manifest_sha256': pair_manifest_sha, 'candidate_K': ks, 'eligible_K': eligible_k, 'family_label_rule': 'lexicographic centroid order in the fixed 14-cell signature coordinate order', 'signature_coordinate_order': SIGNATURE_COORDINATES, 'zero_norm_cosine_convention': {'both_zero': 1.0, 'one_zero': 0.0, 'epsilon': ZERO_NORM_EPS}, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'behavioral_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/discovery_stimulus_experience_multiresolution_fit_contract.json', contract)
    lock_targets = [out / 'qc/discovery_stimulus_experience_multiresolution_fit_contract.json', out / 'qc/discovery_stimulus_experience_multiresolution_fit_summary.json', out / 'qc/discovery_stimulus_experience_multiresolution_fit_participant_stage_preprocessing_audit.csv', out / 'qc/discovery_stimulus_experience_multiresolution_fit_cross_person_standardization_audit.csv', out / 'qc/discovery_stimulus_experience_multiresolution_fit_zero_norm_cosine_audit.csv', out / 'qc/discovery_stimulus_experience_multiresolution_fit_source_rehash.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_candidate_metrics.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_stability_resamples.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_centroids_all_k.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_cross_resolution_metrics_eligible_k.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_family_overlap_eligible_k.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_fine_to_coarse_parent_purity.csv', out / 'tables/discovery_stimulus_experience_multiresolution_fit_signature_coordinate_schema.csv', out / 'reports/discovery_stimulus_experience_multiresolution_fit_report.md']
    hashes = {str(p.relative_to(project)): sha256(p) for p in lock_targets if p.is_file()}
    final = {'phase': PHASE, 'generated_at': now(), 'status': status, 'hashes': hashes, 'eligible_K': eligible_k, 'candidate_K': ks, 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': EXPECTED_discovery_stimulus_experience_multiresolution_fit_POLICY_CANONICAL_SHA256, 'Discovery_stimulus_values_read': True, 'rating_early_EEG_values_read': False, 'behavioral_values_read': False, 'emotion_category_values_read': False, 'context_values_read': False, 'stimulus_identity_values_read': False, 'physiology_values_read': False, 'historical_F1_F2_assignments_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final)
    phase_dir = Path(__file__).resolve().parent
    print('', flush=True)
    print('discovery_stimulus_experience_multiresolution_fit PASS', flush=True)
    print(f'Discovery experience participants: {base.subject_id.nunique()}', flush=True)
    print(f'Discovery experience pairs: {len(base)}', flush=True)
    print(f'Candidate K: {ks}', flush=True)
    print(f"Eligible K: {(eligible_k if eligible_k else 'NONE')}", flush=True)
    print('rating_early EEG opened: False', flush=True)
    print('Behavior/category/context/stimulus identity opened: False', flush=True)
    print('Validation opened: False', flush=True)
    print('Confirmation opened: False', flush=True)
    print('Semantic relabeling permitted: False', flush=True)
    print(f'Next phase: {NEXT_PHASE}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())