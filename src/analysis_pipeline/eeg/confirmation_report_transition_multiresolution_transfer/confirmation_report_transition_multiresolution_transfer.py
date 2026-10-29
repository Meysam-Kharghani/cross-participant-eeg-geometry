# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PHASE = 'confirmation_report_transition_multiresolution_transfer'
NEXT_PHASE = 'cross_split_report_transition_structure_synthesis'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ZERO_NORM_EPS = 1e-12
AUTHORITATIVE_CONFIRMATION_TRIALS = 494
EXPECTED_CONFIRMATION_REPORT_TRANSITION_PAIRS = 459
EXPECTED_CONFIRMATION_SUBJECTS = 12
ELIGIBLE_K = [2, 3]
confirmation_report_transition_transfer_design_OUT = Path('outputs/interpretability/confirmation_report_transition_transfer_design_confirmation_report_transition_transfer_design_specification')
confirmation_report_transition_transfer_design_LOCK = confirmation_report_transition_transfer_design_OUT / 'qc/confirmation_report_transition_transfer_design_final_output_lock_hashes.json'
confirmation_report_transition_transfer_design_LOCK_SHA256 = 'b614f8d7f8bc8261a6dd03fe20e2a380715c6af9576432cf7021070bc8a0085d'
confirmation_report_transition_transfer_design_POLICY = Path('analysis_specifications/S006_confirmation_report_transition_transfer.json')
confirmation_report_transition_transfer_design_POLICY_SHA256 = '4e2f10ccc688776f704704c25fe6c8e70de7d0411090dba5d664f70899305c32'
confirmation_report_transition_transfer_design_POLICY_CANONICAL_SHA256 = '5ae129c8119f8f930133ce59d764a7c1aff537c170e654226725970f6b4804db'
confirmation_report_transition_transfer_design_CENTROIDS = confirmation_report_transition_transfer_design_OUT / 'tables/confirmation_report_transition_transfer_design_prespecified_centroids.csv'
confirmation_report_transition_transfer_design_FALLBACK = confirmation_report_transition_transfer_design_OUT / 'tables/confirmation_report_transition_transfer_design_discovery_fallback_iqr.csv'
confirmation_report_transition_transfer_design_THRESHOLDS = confirmation_report_transition_transfer_design_OUT / 'tables/confirmation_report_transition_transfer_design_support_thresholds.csv'
confirmation_report_transition_transfer_design_GATE_SPEC = confirmation_report_transition_transfer_design_OUT / 'tables/confirmation_report_transition_transfer_design_confirmation_gate_spec.csv'
confirmation_feature_trajectory_LOCK = Path('outputs/qc/confirmation_feature_trajectory_final_output_lock_hashes.json')
confirmation_feature_trajectory_LOCK_SHA256 = '0aae3cc2f447f8383c9ec5656c9a5ba9b4299316f3a19248f34b43850b8246df'
CONFIRMATION_STORE_ROOT = Path('outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1')
CONFIRMATION_COMPACT = CONFIRMATION_STORE_ROOT / 'global/epoch_compact_float32.npy'
CONFIRMATION_MANIFEST = CONFIRMATION_STORE_ROOT / 'global/epoch_manifest.csv'
CONFIRMATION_COMPACT_SHA256 = '734fa3d2581d95adc172ea854d4f1fccc2432eaeb167960fdf1e76da0c1509d3'
CONFIRMATION_MANIFEST_SHA256 = 'ccd83881685da351d79c8de5d199b650d17ed0f6e899c332a7c6d731f7becbd2'
OUT = Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer')
NEXT_CONFIG = Path('analysis_specifications/S005_confirmation_report_transition_multiresolution_transfer_cross.json')
BLOCKS = [{'name': 'time_frequency_channel_median_iqr', 'start': 0, 'stop': 60}, {'name': 'morphology_channel_median_iqr', 'start': 60, 'stop': 72}, {'name': 'plv_channel_median_iqr', 'start': 72, 'stop': 84}, {'name': 'waveform_dct_channel_median_iqr', 'start': 84, 'stop': 124}, {'name': 'covariance_summary', 'start': 124, 'stop': 145}, {'name': 'plv_global', 'start': 145, 'stop': 169}]
COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
RAW = [f'raw__{x}' for x in COORDS]
TR = [f'transformed__{x}' for x in COORDS]
STD = [f'standardized__{x}' for x in COORDS]

def now() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'1', 'true', 't', 'yes', 'y'}

def verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')

def stage_standardize(block: np.ndarray, pooled_iqr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
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
    return (z, med, iq, fallback, n_imputed, constant)

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
    return (np.clip(out, -1.0, 1.0), int(both.sum()), int(one.sum()))

def raw_signatures(late_z: np.ndarray, rating_z: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if late_z.shape != rating_z.shape or late_z.shape[1] != 169:
        raise RuntimeError('Unexpected standardized stimulus_late/rating_early shape')
    cols: list[np.ndarray] = []
    zero_counts: dict[str, int] = {}
    for block in BLOCKS:
        name = str(block['name'])
        start, stop = (int(block['start']), int(block['stop']))
        a = late_z[:, start:stop]
        b = rating_z[:, start:stop]
        rms = np.sqrt(np.mean((b - a) ** 2, axis=1))
        cos, both, one = row_cosine(a, b)
        cols.extend([rms, cos])
        zero_counts[f'{name}__both_zero'] = both
        zero_counts[f'{name}__one_zero'] = one
    rms = np.sqrt(np.mean((rating_z - late_z) ** 2, axis=1))
    cos, both, one = row_cosine(late_z, rating_z)
    cols.extend([rms, cos])
    zero_counts['global__both_zero'] = both
    zero_counts['global__one_zero'] = one
    out = np.column_stack(cols)
    if out.shape[1] != 14 or not np.isfinite(out).all():
        raise RuntimeError('Invalid raw 14-cell report-transition signature')
    return (out, zero_counts)

def transform_signatures(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    out = np.empty_like(raw)
    for j in range(0, raw.shape[1], 2):
        if np.any(raw[:, j] < 0):
            raise RuntimeError('Negative RMS displacement encountered')
        out[:, j] = np.log1p(raw[:, j])
        c = np.clip(raw[:, j + 1], -1 + 1e-06, 1 - 1e-06)
        out[:, j + 1] = np.arctanh(c)
    if not np.isfinite(out).all():
        raise RuntimeError('Nonfinite transformed report-transition signature')
    return out

def build_confirmation_signatures(base: pd.DataFrame, late_raw: np.ndarray, rating_raw: np.ndarray, discovery_fallback_iqr: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    subjects = base.subject_id.astype(str).to_numpy()
    n = len(base)
    if late_raw.shape != (n, 169) or rating_raw.shape != (n, 169):
        raise RuntimeError('Confirmation compact arrays do not align with report-transition pair manifest')
    late_z = np.empty_like(late_raw, dtype=float)
    rating_z = np.empty_like(rating_raw, dtype=float)
    stage_rows: list[dict[str, Any]] = []
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        late = np.asarray(late_raw[idx], dtype=float)
        rating = np.asarray(rating_raw[idx], dtype=float)
        pooled = np.vstack([late, rating])
        pooled_iqr = np.nanquantile(pooled, 0.75, axis=0) - np.nanquantile(pooled, 0.25, axis=0)
        pooled_iqr = np.where(np.isfinite(pooled_iqr), pooled_iqr, 0.0)
        lz, lm, liq, lfb, limp, lconst = stage_standardize(late, pooled_iqr)
        rz, rm, riq, rfb, rimp, rconst = stage_standardize(rating, pooled_iqr)
        late_z[idx] = lz
        rating_z[idx] = rz
        for stage, med, iq, fb, imp, const in [('stimulus_late', lm, liq, lfb, limp, lconst), ('rating_early', rm, riq, rfb, rimp, rconst)]:
            stage_rows.append({'subject_id': str(subject), 'stage': stage, 'paired_trials': int(len(idx)), 'imputed_cells_total': int(imp), 'stage_iqr_fallback_cells': int(np.sum(fb)), 'constant_cells_after_pooled_fallback': int(np.sum(const)), 'median_abs_stage_median': float(np.median(np.abs(med))), 'median_stage_iqr': float(np.median(iq[np.isfinite(iq)])) if np.isfinite(iq).any() else float('nan')})
    raw, zero_counts = raw_signatures(late_z, rating_z)
    transformed = transform_signatures(raw)
    standardized = np.empty_like(transformed)
    scaling_rows: list[dict[str, Any]] = []
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        x = transformed[idx]
        med = np.median(x, axis=0)
        iq = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        fallback = ~np.isfinite(iq) | (iq <= IQR_FLOOR)
        scale = np.where(fallback, discovery_fallback_iqr, iq)
        constant = ~np.isfinite(scale) | (scale <= IQR_FLOOR)
        scale = np.where(constant, 1.0, scale)
        standardized[idx] = np.clip((x - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
        scaling_rows.append({'subject_id': str(subject), 'paired_trials': int(len(idx)), 'signature_iqr_fallback_cells': int(np.sum(fallback)), 'signature_constant_cells_after_discovery_fallback': int(np.sum(constant))})
    if not np.isfinite(standardized).all():
        raise RuntimeError('Nonfinite Confirmation standardized report-transition signatures')
    sig = base[['raw_behavior_row_id', 'subject_id', 'stimulus_late_row_0based', 'rating_early_row_0based']].copy()
    for j, c in enumerate(COORDS):
        sig[f'raw__{c}'] = raw[:, j]
        sig[f'transformed__{c}'] = transformed[:, j]
        sig[f'standardized__{c}'] = standardized[:, j]
    return (sig, pd.DataFrame(stage_rows), pd.DataFrame(scaling_rows), zero_counts)

def assign_fixed(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    lab = np.argmin(d, axis=1)
    assigned = d[np.arange(len(x)), lab]
    alt = d.copy()
    alt[np.arange(len(x)), lab] = np.inf
    margin = np.min(alt, axis=1) - assigned
    return (lab, assigned, margin, d)

def build_pair_manifest(epoch_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {'raw_behavior_row_id', 'subject_id', 'epoch_family', 'epoch_compact_row_0based'}
    missing = required - set(epoch_manifest.columns)
    if missing:
        raise RuntimeError(f'Confirmation manifest schema drift: {sorted(missing)}')
    m = epoch_manifest[list(required)].copy()
    m['raw_behavior_row_id'] = m.raw_behavior_row_id.astype(str)
    m['subject_id'] = m.subject_id.astype(str)
    m['epoch_family'] = m.epoch_family.astype(str)
    m = m[m.epoch_family.isin(['stimulus_late', 'rating_early'])].copy()
    dup = m.duplicated(['raw_behavior_row_id', 'epoch_family'], keep=False)
    if dup.any():
        ex = m.loc[dup, ['raw_behavior_row_id', 'epoch_family']].head().to_dict('records')
        raise RuntimeError(f'Duplicate Confirmation epoch family per trial: {ex}')
    late = m[m.epoch_family.eq('stimulus_late')][['raw_behavior_row_id', 'subject_id', 'epoch_compact_row_0based']].rename(columns={'epoch_compact_row_0based': 'stimulus_late_row_0based'})
    rating = m[m.epoch_family.eq('rating_early')][['raw_behavior_row_id', 'subject_id', 'epoch_compact_row_0based']].rename(columns={'epoch_compact_row_0based': 'rating_early_row_0based'})
    pair = late.merge(rating, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
    pair = pair.sort_values(['subject_id', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    if pair.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('Duplicate Confirmation report-transition raw_behavior_row_id')
    pair['paired_trials_subject'] = pair.groupby('subject_id').raw_behavior_row_id.transform('size').astype(int)
    pair['participant_eligible'] = pair.paired_trials_subject.ge(16)
    return pair

def verify_inputs(project: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    rehash_rows: list[dict[str, Any]] = []
    verify(project / confirmation_report_transition_transfer_design_LOCK, confirmation_report_transition_transfer_design_LOCK_SHA256, 'confirmation_report_transition_transfer_design final lock')
    lock = readj(project / confirmation_report_transition_transfer_design_LOCK)
    if lock.get('phase') != 'confirmation_report_transition_transfer_design':
        raise RuntimeError('confirmation_report_transition_transfer_design final-lock phase drift')
    if lock.get('advance_K') != ELIGIBLE_K:
        raise RuntimeError(f"confirmation_report_transition_transfer_design final-lock K drift: {lock.get('advance_K')}")
    if lock.get('next_phase') != PHASE:
        raise RuntimeError('confirmation_report_transition_transfer_design final-lock next-phase drift')
    for key in ['Confirmation_compact_values_read', 'Confirmation_manifest_parsed', 'Confirmation_behavioral_values_read', 'Confirmation_emotion_category_values_read', 'semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
        if bool(lock.get(key)):
            raise RuntimeError(f'confirmation_report_transition_transfer_design boundary violation: {key}=True')
    for rel, expected in sorted(lock.get('hashes', {}).items()):
        p = project / rel
        verify(p, expected, f'confirmation_report_transition_transfer_design child {rel}')
        rehash_rows.append({'source': 'confirmation_report_transition_transfer_design_fixed_artifact', 'path': rel, 'expected_sha256': expected, 'observed_sha256': sha256(p), 'passed': True})
    verify(project / confirmation_report_transition_transfer_design_POLICY, confirmation_report_transition_transfer_design_POLICY_SHA256, 'confirmation_report_transition_transfer_design prespecified policy')
    policy = readj(project / confirmation_report_transition_transfer_design_POLICY)
    if canonical_sha(policy) != confirmation_report_transition_transfer_design_POLICY_CANONICAL_SHA256:
        raise RuntimeError('confirmation_report_transition_transfer_design policy canonical SHA drift')
    if policy.get('phase') != 'confirmation_report_transition_transfer_design':
        raise RuntimeError('confirmation_report_transition_transfer_design policy phase drift')
    if policy.get('next_phase') != PHASE:
        raise RuntimeError('confirmation_report_transition_transfer_design policy next-phase drift')
    if policy.get('advance_K') != ELIGIBLE_K:
        raise RuntimeError('confirmation_report_transition_transfer_design policy advance-K drift')
    if policy.get('fixed_before_confirmation_signal_access_in_this_branch') is not True:
        raise RuntimeError('confirmation_report_transition_transfer_design was not fixed before Confirmation signal access')
    if policy.get('representation', {}).get('stages') != ['stimulus_late', 'rating_early']:
        raise RuntimeError('confirmation_report_transition_transfer_design report-transition stage drift')
    if policy.get('representation', {}).get('source_cells_per_epoch') != 169:
        raise RuntimeError('confirmation_report_transition_transfer_design source width drift')
    boundary = policy.get('confirmation_report_transition_transfer_design_data_boundary', {})
    for key in ['Confirmation_compact_values_read', 'Confirmation_manifest_parsed', 'Confirmation_behavioral_values_read', 'Confirmation_emotion_category_values_read', 'Confirmation_context_target_values_read', 'Confirmation_stimulus_identity_values_read', 'semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
        if bool(boundary.get(key)):
            raise RuntimeError(f'confirmation_report_transition_transfer_design policy boundary violation: {key}=True')
    store = policy.get('confirmation_store', {})
    expected_store = {'compact': str(CONFIRMATION_COMPACT), 'compact_sha256': CONFIRMATION_COMPACT_SHA256, 'manifest': str(CONFIRMATION_MANIFEST), 'manifest_sha256': CONFIRMATION_MANIFEST_SHA256, 'historical_confirmation_feature_trajectory_final_lock_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'authoritative_confirmation_trials': AUTHORITATIVE_CONFIRMATION_TRIALS, 'expected_confirmation_subjects': EXPECTED_CONFIRMATION_SUBJECTS}
    for key, expected in expected_store.items():
        if store.get(key) != expected:
            raise RuntimeError(f'confirmation_report_transition_transfer_design Confirmation store reference drift for {key}: {store.get(key)!r} != {expected!r}')
    cent = pd.read_csv(project / confirmation_report_transition_transfer_design_CENTROIDS)
    thr = pd.read_csv(project / confirmation_report_transition_transfer_design_THRESHOLDS)
    if sorted(cent.k.astype(int).unique().tolist()) != ELIGIBLE_K:
        raise RuntimeError('confirmation_report_transition_transfer_design centroid K-set drift')
    if sorted(thr.k.astype(int).unique().tolist()) != ELIGIBLE_K:
        raise RuntimeError('confirmation_report_transition_transfer_design threshold K-set drift')
    for c in STD:
        if c not in cent.columns:
            raise RuntimeError(f'Missing confirmation_report_transition_transfer_design centroid coordinate: {c}')
    for k in ELIGIBLE_K:
        pnode = policy['fixed_assignment'][str(k)]
        cdf = cent[cent.k.astype(int).eq(k)].sort_values('family_1based')
        table_centers = cdf[STD].to_numpy(float)
        policy_centers = np.asarray(pnode['centroids'], dtype=float)
        if table_centers.shape != (k, 14) or not np.allclose(table_centers, policy_centers, rtol=0, atol=1e-12):
            raise RuntimeError(f'confirmation_report_transition_transfer_design policy/table centroid mismatch for K={k}')
        trow = thr[thr.k.astype(int).eq(k)]
        if len(trow) != 1:
            raise RuntimeError(f'confirmation_report_transition_transfer_design support-threshold row drift for K={k}')
        if not np.isclose(float(trow.iloc[0].in_support_squared_distance_max), float(pnode['support']['in_support_squared_distance_max']), rtol=0, atol=1e-12):
            raise RuntimeError(f'confirmation_report_transition_transfer_design distance threshold mismatch for K={k}')
        if not np.isclose(float(trow.iloc[0].low_margin_threshold), float(pnode['support']['low_margin_threshold']), rtol=0, atol=1e-12):
            raise RuntimeError(f'confirmation_report_transition_transfer_design margin threshold mismatch for K={k}')
    verify(project / confirmation_feature_trajectory_LOCK, confirmation_feature_trajectory_LOCK_SHA256, 'historical confirmation_feature_trajectory final lock')
    rehash_rows.append({'source': 'confirmation_feature_trajectory_historical_final_lock', 'path': str(confirmation_feature_trajectory_LOCK), 'expected_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'observed_sha256': sha256(project / confirmation_feature_trajectory_LOCK), 'passed': True})
    verify(project / CONFIRMATION_COMPACT, CONFIRMATION_COMPACT_SHA256, 'Confirmation compact EEG store')
    rehash_rows.append({'source': 'Confirmation_compact_EEG_array', 'path': str(CONFIRMATION_COMPACT), 'expected_sha256': CONFIRMATION_COMPACT_SHA256, 'observed_sha256': sha256(project / CONFIRMATION_COMPACT), 'passed': True})
    verify(project / CONFIRMATION_MANIFEST, CONFIRMATION_MANIFEST_SHA256, 'Confirmation compact manifest')
    rehash_rows.append({'source': 'Confirmation_compact_manifest', 'path': str(CONFIRMATION_MANIFEST), 'expected_sha256': CONFIRMATION_MANIFEST_SHA256, 'observed_sha256': sha256(project / CONFIRMATION_MANIFEST), 'passed': True})
    return (lock, policy, pd.DataFrame(rehash_rows))

def evaluate_k(k: int, sig: pd.DataFrame, pair: pd.DataFrame, centers: np.ndarray, distance_threshold: float, low_margin_threshold: float, gate_policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x = sig[STD].to_numpy(float)
    lab, assigned, margin, d = assign_fixed(x, centers)
    a = sig[['raw_behavior_row_id', 'subject_id']].copy()
    a.insert(0, 'k', int(k))
    a['family_1based'] = lab + 1
    a['assigned_squared_distance'] = assigned
    a['assignment_margin'] = margin
    a['in_discovery_support'] = assigned <= distance_threshold
    a['low_margin'] = margin < low_margin_threshold
    for j in range(k):
        a[f'distance_to_family{j + 1}'] = d[:, j]
    pair_subject = pair[['raw_behavior_row_id', 'paired_trials_subject', 'participant_eligible']]
    a = a.merge(pair_subject, on='raw_behavior_row_id', how='left', validate='one_to_one')
    participant_rows: list[dict[str, Any]] = []
    for subject, g in a.groupby('subject_id', sort=True):
        n = len(g)
        eligible = bool(g.participant_eligible.iloc[0]) if n else False
        row: dict[str, Any] = {'k': int(k), 'subject_id': str(subject), 'paired_report_transition_trials': int(n), 'eligible_for_transfer_gate': bool(eligible), 'in_support_fraction': float(g.in_discovery_support.mean()) if n else float('nan'), 'low_margin_fraction': float(g.low_margin.mean()) if n else float('nan'), 'distinct_families_present': int(g.family_1based.nunique()) if n else 0}
        for fam in range(1, k + 1):
            count = int(g.family_1based.eq(fam).sum())
            row[f'family{fam}_trials'] = count
            row[f'family{fam}_occupancy'] = float(count / n) if n else float('nan')
        participant_rows.append(row)
    participant = pd.DataFrame(participant_rows)
    eligible_p = participant[participant.eligible_for_transfer_gate].copy()
    eligible_subjects = set(eligible_p.subject_id.astype(str))
    eligible_a = a[a.subject_id.astype(str).isin(eligible_subjects)].copy()
    paired_fraction = float(len(pair) / AUTHORITATIVE_CONFIRMATION_TRIALS)
    pb_support = float(eligible_p.in_support_fraction.mean()) if len(eligible_p) else float('nan')
    family_stats: list[dict[str, Any]] = []
    for fam in range(1, k + 1):
        trial_count = int(eligible_a.family_1based.eq(fam).sum())
        distinct_p = int(eligible_a.loc[eligible_a.family_1based.eq(fam), 'subject_id'].astype(str).nunique())
        occ_col = f'family{fam}_occupancy'
        pb_occ = float(eligible_p[occ_col].mean()) if len(eligible_p) else float('nan')
        family_stats.append({'family_1based': fam, 'trial_count_eligible_participants': trial_count, 'distinct_eligible_participants': distinct_p, 'participant_balanced_occupancy': pb_occ})
    gates: list[dict[str, Any]] = []

    def add_gate(gid: str, metric: str, observed: Any, threshold: Any, comparison: str, passed: bool) -> None:
        gates.append({'k': int(k), 'gate_id': gid, 'metric': metric, 'observed': observed, 'threshold': threshold, 'comparison': comparison, 'pass': bool(passed)})
    add_gate('source_hashes', 'all source/store hash gates pass', True, True, '==', True)
    min_ep = int(gate_policy['minimum_eligible_participants'])
    add_gate('eligible_participants', 'eligible Confirmation participants', int(len(eligible_p)), min_ep, '>=', len(eligible_p) >= min_ep)
    min_pair = float(gate_policy['minimum_paired_report_transition_fraction_of_authoritative_confirmation_trials'])
    add_gate('paired_fraction', 'paired report-transition trials / authoritative Confirmation trials', paired_fraction, min_pair, '>=', paired_fraction >= min_pair)
    min_support = float(gate_policy['minimum_participant_balanced_in_support_fraction'])
    add_gate('in_support', 'participant-balanced in-support fraction', pb_support, min_support, '>=', np.isfinite(pb_support) and pb_support >= min_support)
    min_fp = int(gate_policy['minimum_distinct_participants_per_family'])
    min_ft = int(gate_policy['minimum_trials_per_family'])
    min_occ = float(gate_policy['minimum_participant_balanced_occupancy_per_family'])
    for fs in family_stats:
        fam = int(fs['family_1based'])
        add_gate(f'family{fam}_participants', f'family {fam} distinct eligible participants', int(fs['distinct_eligible_participants']), min_fp, '>=', int(fs['distinct_eligible_participants']) >= min_fp)
        add_gate(f'family{fam}_trials', f'family {fam} trials among eligible participants', int(fs['trial_count_eligible_participants']), min_ft, '>=', int(fs['trial_count_eligible_participants']) >= min_ft)
        add_gate(f'family{fam}_occupancy', f'family {fam} participant-balanced occupancy', float(fs['participant_balanced_occupancy']), min_occ, '>=', np.isfinite(float(fs['participant_balanced_occupancy'])) and float(fs['participant_balanced_occupancy']) >= min_occ)
    add_gate('no_behavior', 'behavior/category/context/stimulus target mapping executed', False, False, '==', True)
    gate = pd.DataFrame(gates)
    passed = bool(gate['pass'].all())
    summary = {'k': int(k), 'replicated': passed, 'eligible_participants': int(len(eligible_p)), 'paired_report_transition_trials_all_participants': int(len(pair)), 'paired_fraction_of_494': paired_fraction, 'assigned_trials_eligible_participants': int(len(eligible_a)), 'participant_balanced_in_support_fraction': pb_support, 'overall_in_support_fraction': float(eligible_a.in_discovery_support.mean()) if len(eligible_a) else float('nan'), 'participant_balanced_low_margin_fraction': float(eligible_p.low_margin_fraction.mean()) if len(eligible_p) else float('nan'), 'overall_low_margin_fraction': float(eligible_a.low_margin.mean()) if len(eligible_a) else float('nan'), 'families': family_stats}
    return (a, participant, gate, summary)

def self_test() -> int:
    rng = np.random.default_rng(540)
    subjects = np.repeat([f'sub-{i:02d}' for i in range(1, 11)], 20)
    n = len(subjects)
    base = pd.DataFrame({'raw_behavior_row_id': [f'v{i:04d}' for i in range(n)], 'subject_id': subjects, 'stimulus_late_row_0based': np.arange(n), 'rating_early_row_0based': np.arange(n) + n})
    late = rng.normal(size=(n, 169))
    rating = late + rng.normal(scale=0.7, size=(n, 169))
    fallback = np.ones(14, dtype=float)
    sig, stage, scaling, zero = build_confirmation_signatures(base, late, rating, fallback)
    assert len(sig) == n and sig[STD].shape == (n, 14)
    assert set(stage.stage) == {'stimulus_late', 'rating_early'}
    assert len(stage) == 20 and len(scaling) == 10
    centers = sig[STD].to_numpy(float)[[0, 55, 110]].copy()
    lab, dist, margin, d = assign_fixed(sig[STD].to_numpy(float), centers)
    assert lab.min() >= 0 and lab.max() < 3
    assert np.isfinite(dist).all() and np.isfinite(margin).all()
    assert d.shape == (n, 3)
    synthetic_manifest = pd.DataFrame({'raw_behavior_row_id': ['a', 'a', 'b', 'b'], 'subject_id': ['sub-01'] * 4, 'epoch_family': ['stimulus_late', 'rating_early', 'stimulus_late', 'rating_early'], 'epoch_compact_row_0based': [1, 2, 3, 4]})
    p = build_pair_manifest(synthetic_manifest)
    assert len(p) == 2
    assert p[['stimulus_late_row_0based', 'rating_early_row_0based']].to_numpy().tolist() == [[1, 2], [3, 4]]
    assert not zero or isinstance(zero, dict)
    print('confirmation_report_transition_multiresolution_transfer self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    started = time.time()
    print(f'[{now()}] verifying prespecified confirmation_report_transition_transfer_design transfer design and exact Confirmation compact store', flush=True)
    _, policy, source_rehash = verify_inputs(project)
    out = project / OUT
    final_lock = out / 'qc/confirmation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'
    if out.exists():
        if final_lock.is_file() and (not args.force):
            raise RuntimeError(f'Completed confirmation_report_transition_multiresolution_transfer output exists: {out}\nUse --force only if intentionally rematerializing the identical fixed transfer.')
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(project / CONFIRMATION_MANIFEST, low_memory=False)
    pair = build_pair_manifest(manifest)
    if manifest.subject_id.astype(str).nunique() != EXPECTED_CONFIRMATION_SUBJECTS:
        raise RuntimeError(f'Confirmation compact manifest subject-count drift: {manifest.subject_id.astype(str).nunique()}')
    if len(pair) != EXPECTED_CONFIRMATION_REPORT_TRANSITION_PAIRS:
        raise RuntimeError(f'Confirmation report-transition pair-count drift: expected {EXPECTED_CONFIRMATION_REPORT_TRANSITION_PAIRS}, got {len(pair)}')
    if pair.subject_id.astype(str).nunique() != EXPECTED_CONFIRMATION_SUBJECTS:
        raise RuntimeError(f'Confirmation report-transition paired-subject count drift: {pair.subject_id.astype(str).nunique()}')
    pair_fraction = len(pair) / AUTHORITATIVE_CONFIRMATION_TRIALS
    print(f'[{now()}] Confirmation report-transition pairs: {len(pair)}/{AUTHORITATIVE_CONFIRMATION_TRIALS} ({pair_fraction:.4f}); participants={pair.subject_id.nunique()}', flush=True)
    compact = np.load(project / CONFIRMATION_COMPACT, mmap_mode='r', allow_pickle=False)
    if compact.ndim != 2 or compact.shape[1] != 169:
        raise RuntimeError(f'Unexpected Confirmation compact shape: {compact.shape}')
    lrow = pair.stimulus_late_row_0based.to_numpy(int)
    rrow = pair.rating_early_row_0based.to_numpy(int)
    if len(pair) == 0:
        raise RuntimeError('No Confirmation stimulus_late/rating_early pairs')
    if lrow.min() < 0 or rrow.min() < 0 or lrow.max() >= len(compact) or (rrow.max() >= len(compact)):
        raise RuntimeError('Confirmation report-transition pair manifest references compact rows outside store')
    print(f'[{now()}] opening only Confirmation stimulus_late/rating_early compact EEG values', flush=True)
    late = np.asarray(compact[lrow], dtype=np.float64)
    rating = np.asarray(compact[rrow], dtype=np.float64)
    del compact
    fallback_df = pd.read_csv(project / confirmation_report_transition_transfer_design_FALLBACK)
    if fallback_df.coordinate.astype(str).tolist() != COORDS:
        raise RuntimeError('confirmation_report_transition_transfer_design fallback-IQR coordinate order drift')
    fallback = fallback_df.discovery_participant_balanced_pooled_iqr.to_numpy(float)
    if fallback.shape != (14,) or not np.isfinite(fallback).all():
        raise RuntimeError('Invalid prespecified Discovery report-transition fallback IQR')
    sig, stage_audit, scaling_audit, zero_counts = build_confirmation_signatures(pair, late, rating, fallback)
    print(f'[{now()}] constructed prespecified 14-cell Confirmation report-transition signatures; all behavioral/category/context/stimulus targets remain unopened', flush=True)
    cent = pd.read_csv(project / confirmation_report_transition_transfer_design_CENTROIDS)
    thr = pd.read_csv(project / confirmation_report_transition_transfer_design_THRESHOLDS)
    all_assign: list[pd.DataFrame] = []
    all_participant: list[pd.DataFrame] = []
    all_gate: list[pd.DataFrame] = []
    resolution_rows: list[dict[str, Any]] = []
    gate_policy = policy['structural_transfer_gate']
    for k in ELIGIBLE_K:
        cdf = cent[cent.k.astype(int).eq(k)].sort_values('family_1based')
        if len(cdf) != k or cdf.family_1based.astype(int).tolist() != list(range(1, k + 1)):
            raise RuntimeError(f'prespecified centroid family numbering drift for K={k}')
        centers = cdf[STD].to_numpy(float)
        trow = thr[thr.k.astype(int).eq(k)]
        if len(trow) != 1:
            raise RuntimeError(f'Missing prespecified support threshold for K={k}')
        distance_threshold = float(trow.iloc[0].in_support_squared_distance_max)
        low_margin_threshold = float(trow.iloc[0].low_margin_threshold)
        a, psummary, gate, summary = evaluate_k(k, sig, pair, centers, distance_threshold, low_margin_threshold, gate_policy)
        all_assign.append(a)
        all_participant.append(psummary)
        all_gate.append(gate)
        row = {key: value for key, value in summary.items() if key != 'families'}
        row['family_stats_json'] = json.dumps(summary['families'], sort_keys=True)
        resolution_rows.append(row)
        print(f"[{now()}] K={k}: replicated={summary['replicated']}; eligible_participants={summary['eligible_participants']}; PB support={summary['participant_balanced_in_support_fraction']:.6f}", flush=True)
    assignments = pd.concat(all_assign, ignore_index=True)
    participant = pd.concat(all_participant, ignore_index=True)
    gates = pd.concat(all_gate, ignore_index=True)
    resolution = pd.DataFrame(resolution_rows).sort_values('k').reset_index(drop=True)
    passing_k = resolution.loc[resolution.replicated.map(bval), 'k'].astype(int).tolist()
    pair_out = pair.copy()
    pair_out['authoritative_confirmation_trials'] = AUTHORITATIVE_CONFIRMATION_TRIALS
    pair_out['paired_fraction_of_authoritative_confirmation_trials'] = pair_fraction
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_pair_manifest.csv', pair_out)
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_stage_preprocessing_audit.csv', stage_audit)
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_signature_scaling_audit.csv', scaling_audit)
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv', sig)
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv', assignments)
    atomic_csv(out / 'qc/confirmation_report_transition_multiresolution_transfer_confirmation_participant_transfer_summary_all_k.csv', participant)
    atomic_csv(out / 'qc/confirmation_report_transition_multiresolution_transfer_confirmation_structural_transfer_gate_all_k.csv', gates)
    atomic_csv(out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_resolution_summary.csv', resolution)
    atomic_csv(out / 'qc/confirmation_report_transition_multiresolution_transfer_source_rehash.csv', source_rehash)
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'synthesize the one-shot Confirmation report-transition structural transfer without opening any new signal or behavioral values', 'tested_K': ELIGIBLE_K, 'passing_K_confirmation': passing_k, 'any_K_replicated_in_confirmation': bool(passing_k), 'cross_split_report_transition_structure_synthesis_new_signal_opening_permitted': False, 'cross_split_report_transition_structure_synthesis_behavioral_opening_permitted': False, 'cross_split_report_transition_structure_synthesis_semantic_test_execution_permitted': False, 'if_no_K_pass': 'record Confirmation failure of the discrete transition-family branch; retain the prespecified continuous 14-cell report-transition signature and prohibit rescue or retuning', 'if_one_or_more_K_pass': 'record all passing K as internally cross-split structurally replicated; do not rank them or assign semantics', 'forbidden': ['new EEG signal access in cross_split_report_transition_structure_synthesis', 'behavioral report access in cross_split_report_transition_structure_synthesis', 'emotion category target mapping in cross_split_report_transition_structure_synthesis', 'context target mapping in cross_split_report_transition_structure_synthesis', 'stimulus identity target mapping in cross_split_report_transition_structure_synthesis', 'centroid refit', 'threshold tuning', 'K reselection', 'family relabeling', 'semantic naming', 'claims of distortion/dishonesty/suppression/exaggeration/conscious editing']}
    writej(project / NEXT_CONFIG, next_policy)
    summary_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'confirmation_report_transition_multiresolution_transfer_complete', 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'any_K_replicated': bool(passing_k), 'counts': {'confirmation_manifest_subjects': int(manifest.subject_id.astype(str).nunique()), 'authoritative_confirmation_trials': AUTHORITATIVE_CONFIRMATION_TRIALS, 'paired_stimulus_late_rating_early_trials': int(len(pair)), 'paired_participants': int(pair.subject_id.nunique()), 'participants_with_at_least_16_pairs': int(pair.loc[pair.participant_eligible, 'subject_id'].nunique())}, 'paired_fraction_of_494': float(pair_fraction), 'resolution_results': resolution.to_dict('records'), 'zero_norm_counts': zero_counts, 'boundary': {'Confirmation_stimulus_late_values_read': True, 'Confirmation_rating_early_values_read': True, 'Confirmation_behavioral_values_read': False, 'Confirmation_emotion_category_values_read': False, 'Confirmation_context_values_read': False, 'Confirmation_stimulus_identity_values_read': False, 'Confirmation_compact_values_read': True, 'Confirmation_manifest_parsed': True, 'centroids_refit': False, 'thresholds_tuned': False, 'K_reselected': False, 'family_semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE, 'next_phase_role': next_policy['next_phase_role'], 'runtime_seconds': time.time() - started}
    writej(out / 'qc/confirmation_report_transition_multiresolution_transfer_summary.json', summary_obj)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'one_shot_confirmation_report_transition_transfer_executed', 'source_code_sha256': sha256(Path(__file__).resolve()), 'confirmation_report_transition_transfer_design_final_lock_sha256': confirmation_report_transition_transfer_design_LOCK_SHA256, 'confirmation_report_transition_transfer_design_policy_file_sha256': confirmation_report_transition_transfer_design_POLICY_SHA256, 'confirmation_report_transition_transfer_design_policy_canonical_sha256': confirmation_report_transition_transfer_design_POLICY_CANONICAL_SHA256, 'confirmation_feature_trajectory_historical_final_lock_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'Confirmation_compact_sha256': CONFIRMATION_COMPACT_SHA256, 'Confirmation_manifest_sha256': CONFIRMATION_MANIFEST_SHA256, 'expected_confirmation_report_transition_pairs': EXPECTED_CONFIRMATION_REPORT_TRANSITION_PAIRS, 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'prespecified_gate_policy': gate_policy, 'boundary': summary_obj['boundary'], 'next_phase': NEXT_PHASE}
    writej(out / 'qc/confirmation_report_transition_multiresolution_transfer_contract.json', contract)
    lines = ['# confirmation_report_transition_multiresolution_transfer — One-shot Confirmation report-transition multi-resolution transfer', '', '**Confirmation `stimulus_late -> rating_early` EEG was opened under the confirmation_report_transition_transfer_design-prespecified transfer contract.**', '', f'- Authoritative Confirmation trials: **{AUTHORITATIVE_CONFIRMATION_TRIALS}**', f'- Paired stimulus_late→rating_early trials: **{len(pair)}** ({pair_fraction:.4f})', f"- Participants with ≥16 paired trials: **{int(pair.loc[pair.participant_eligible, 'subject_id'].nunique())}**", f'- Tested Discovery transition resolutions: **{ELIGIBLE_K}**', f"- Passing Confirmation transition resolutions: **{(passing_k if passing_k else 'NONE')}**", '', '## prespecified structural-transfer results', resolution.to_string(index=False), '', '## Interpretation boundary', '- Replication, if present, supports only transfer of the prespecified report/evaluation-stage EEG reconfiguration partition.', '- It does not establish emotion semantics, report truthfulness, distortion, suppression, exaggeration, or conscious editing.', '- Confirmation behavioral/category/context/stimulus targets remained unopened.', '- Centroids were not refit, K was not reselected, support thresholds were not tuned, and families were not renamed.', '- The prespecified Confirmation compact EEG/manifest were opened one-shot for this structural transfer; no behavioral/category/context/stimulus targets were opened.', '', f'## Next', f'**{NEXT_PHASE}** — synthesize this Confirmation result without opening any new signal or behavioral values.', '']
    report_path = out / 'reports/confirmation_report_transition_multiresolution_transfer_report.md'
    report_path.write_text(''.join(lines), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, out / 'qc/confirmation_report_transition_multiresolution_transfer_contract.json', out / 'qc/confirmation_report_transition_multiresolution_transfer_summary.json', out / 'qc/confirmation_report_transition_multiresolution_transfer_source_rehash.csv', out / 'qc/confirmation_report_transition_multiresolution_transfer_confirmation_participant_transfer_summary_all_k.csv', out / 'qc/confirmation_report_transition_multiresolution_transfer_confirmation_structural_transfer_gate_all_k.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_pair_manifest.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_stage_preprocessing_audit.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_signature_scaling_audit.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv', out / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_resolution_summary.csv', report_path]
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'confirmation_report_transition_multiresolution_transfer_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'any_K_replicated': bool(passing_k), 'Confirmation_behavioral_values_read': False, 'Confirmation_stimulus_late_values_read': True, 'Confirmation_rating_early_values_read': True, 'Confirmation_compact_values_read': True, 'Confirmation_manifest_parsed': True, 'centroids_refit': False, 'thresholds_tuned': False, 'K_reselected': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final_obj)
    script_dir = Path(__file__).resolve().parent
    files = lock_targets + [final_lock]
    for p in sorted(script_dir.iterdir()):
        if p.is_file():
            files.append(p)
    files += [project / confirmation_report_transition_transfer_design_POLICY, project / confirmation_report_transition_transfer_design_LOCK, project / confirmation_report_transition_transfer_design_CENTROIDS, project / confirmation_report_transition_transfer_design_FALLBACK, project / confirmation_report_transition_transfer_design_THRESHOLDS, project / confirmation_report_transition_transfer_design_GATE_SPEC]
    print('')
    print('confirmation_report_transition_multiresolution_transfer PASS')
    print(f'Confirmation report-transition pairs: {len(pair)}/{AUTHORITATIVE_CONFIRMATION_TRIALS}')
    print(f'Tested K: {ELIGIBLE_K}')
    print(f"Passing K: {(passing_k if passing_k else 'NONE')}")
    for r in resolution.itertuples(index=False):
        print(f'K={int(r.k)} replicated={bool(r.replicated)}; eligible_participants={int(r.eligible_participants)}; PB_support={float(r.participant_balanced_in_support_fraction):.6f}')
    print('Confirmation stimulus_late EEG opened: True')
    print('Confirmation rating_early EEG opened: True')
    print('Confirmation behavioral values opened: False')
    print('Confirmation compact values opened: True')
    print('Confirmation manifest parsed: True')
    print('Centroids refit: False')
    print('Thresholds tuned: False')
    print('K reselected: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())