# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PHASE = 'validation_stimulus_experience_multiresolution_transfer'
NEXT_PHASE = 'validation_experience_transfer_synthesis'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ZERO_NORM_EPS = 1e-12
AUTHORITATIVE_VALIDATION_TRIALS = 494
ELIGIBLE_K = [2, 3, 4]
validation_experience_transfer_design_OUT = Path('outputs/interpretability/validation_experience_transfer_design_validation_experience_transfer_design_specification')
validation_experience_transfer_design_LOCK = validation_experience_transfer_design_OUT / 'qc/validation_experience_transfer_design_final_output_lock_hashes.json'
validation_experience_transfer_design_LOCK_SHA256 = '17d7804ee252da28ab35dee675d30eece1b44105ecd5abb96d06e6b7dbc0a58f'
validation_experience_transfer_design_POLICY = Path('analysis_specifications/S077_validation_experience_transfer.json')
validation_experience_transfer_design_POLICY_CANONICAL_SHA256 = '5b72d75a9f59183fe69f88e1a7168a1390bc64875adf54a75b245dae1dd4c589'
validation_experience_transfer_design_CENTROIDS = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_prespecified_centroids.csv'
validation_experience_transfer_design_FALLBACK = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_discovery_fallback_iqr.csv'
validation_experience_transfer_design_THRESHOLDS = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_support_thresholds.csv'
validation_experience_transfer_design_GATE_SPEC = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_validation_gate_spec.csv'
validation_feature_trajectory_LOCK = Path('outputs/qc/validation_feature_trajectory_final_output_lock_hashes.json')
validation_feature_trajectory_LOCK_SHA256 = 'ab18935f81020fcc0d8596d63f8a6d3021c3e19285ac075d3c56c2b566cbfabe'
VALIDATION_STORE_ROOT = Path('outputs/large/validation_feature_trajectory_validation_compact_features_v1')
VALIDATION_COMPACT = VALIDATION_STORE_ROOT / 'global/epoch_compact_float32.npy'
VALIDATION_MANIFEST = VALIDATION_STORE_ROOT / 'global/epoch_manifest.csv'
VALIDATION_COMPACT_SHA256 = '6f8ab1755f8d950a33351d2032ccd190092656923b09b701ab04e78430fec402'
VALIDATION_MANIFEST_SHA256 = '24643e775d90d30011fbbba7d40defedcbdb699632603c2d838b420adb49c1a8'
OUT = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer')
NEXT_CONFIG = Path('analysis_specifications/S091_validation_stimulus_experience_multiresolution_transfer.json')
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
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

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

def participant_weights(subjects: np.ndarray) -> np.ndarray:
    vc = pd.Series(subjects.astype(str)).value_counts().to_dict()
    return np.array([1.0 / vc[str(s)] for s in subjects], dtype=float)

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

def raw_signatures(early_z: np.ndarray, late_z: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if early_z.shape != late_z.shape or early_z.shape[1] != 169:
        raise RuntimeError('Unexpected standardized early/late shape')
    cols: list[np.ndarray] = []
    zero_counts: dict[str, int] = {}
    for block in BLOCKS:
        name = str(block['name'])
        start, stop = (int(block['start']), int(block['stop']))
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
    raw = np.asarray(raw, dtype=float)
    out = np.empty_like(raw)
    for j in range(0, raw.shape[1], 2):
        if np.any(raw[:, j] < 0):
            raise RuntimeError('Negative RMS displacement encountered')
        out[:, j] = np.log1p(raw[:, j])
        c = np.clip(raw[:, j + 1], -1 + 1e-06, 1 - 1e-06)
        out[:, j + 1] = np.arctanh(c)
    if not np.isfinite(out).all():
        raise RuntimeError('Nonfinite transformed experience signature')
    return out

def build_validation_signatures(base: pd.DataFrame, early_raw: np.ndarray, late_raw: np.ndarray, discovery_fallback_iqr: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    subjects = base.subject_id.astype(str).to_numpy()
    n = len(base)
    if early_raw.shape != (n, 169) or late_raw.shape != (n, 169):
        raise RuntimeError('Validation compact arrays do not align with pair manifest')
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
        ez, em, eiq, efb, eimp, econst = stage_standardize(e, pooled_iqr)
        lz, lm, liq, lfb, limp, lconst = stage_standardize(l, pooled_iqr)
        early_z[idx] = ez
        late_z[idx] = lz
        for stage, med, iq, fb, imp, const in [('stimulus_early', em, eiq, efb, eimp, econst), ('stimulus_late', lm, liq, lfb, limp, lconst)]:
            stage_rows.append({'subject_id': str(subject), 'stage': stage, 'paired_trials': int(len(idx)), 'imputed_cells_total': int(imp), 'stage_iqr_fallback_cells': int(np.sum(fb)), 'constant_cells_after_pooled_fallback': int(np.sum(const)), 'median_abs_stage_median': float(np.median(np.abs(med))), 'median_stage_iqr': float(np.median(iq[np.isfinite(iq)])) if np.isfinite(iq).any() else float('nan')})
    raw, zero_counts = raw_signatures(early_z, late_z)
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
        raise RuntimeError('Nonfinite Validation standardized experience signatures')
    sig = base[['raw_behavior_row_id', 'subject_id', 'stimulus_early_row_0based', 'stimulus_late_row_0based']].copy()
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
        raise RuntimeError(f'Validation manifest schema drift: {sorted(missing)}')
    m = epoch_manifest[list(required)].copy()
    m['raw_behavior_row_id'] = m.raw_behavior_row_id.astype(str)
    m['subject_id'] = m.subject_id.astype(str)
    m['epoch_family'] = m.epoch_family.astype(str)
    m = m[m.epoch_family.isin(['stimulus_early', 'stimulus_late'])].copy()
    dup = m.duplicated(['raw_behavior_row_id', 'epoch_family'], keep=False)
    if dup.any():
        ex = m.loc[dup, ['raw_behavior_row_id', 'epoch_family']].head().to_dict('records')
        raise RuntimeError(f'Duplicate Validation epoch family per trial: {ex}')
    early = m[m.epoch_family.eq('stimulus_early')][['raw_behavior_row_id', 'subject_id', 'epoch_compact_row_0based']].rename(columns={'epoch_compact_row_0based': 'stimulus_early_row_0based'})
    late = m[m.epoch_family.eq('stimulus_late')][['raw_behavior_row_id', 'subject_id', 'epoch_compact_row_0based']].rename(columns={'epoch_compact_row_0based': 'stimulus_late_row_0based'})
    pair = early.merge(late, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
    pair = pair.sort_values(['subject_id', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    if pair.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('Duplicate Validation paired raw_behavior_row_id')
    pair['paired_trials_subject'] = pair.groupby('subject_id').raw_behavior_row_id.transform('size').astype(int)
    pair['participant_eligible'] = pair.paired_trials_subject.ge(16)
    return pair

def verify_inputs(project: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    verify(project / validation_experience_transfer_design_LOCK, validation_experience_transfer_design_LOCK_SHA256, 'validation_experience_transfer_design final lock')
    validation_experience_transfer_design_lock = readj(project / validation_experience_transfer_design_LOCK)
    if validation_experience_transfer_design_lock.get('phase') != 'validation_experience_transfer_design':
        raise RuntimeError('validation_experience_transfer_design final-lock phase drift')
    if validation_experience_transfer_design_lock.get('eligible_K') != ELIGIBLE_K or validation_experience_transfer_design_lock.get('next_phase') != PHASE:
        raise RuntimeError('validation_experience_transfer_design eligible-K/next-phase drift')
    for key in ['Validation_signal_values_read', 'Validation_behavioral_values_read', 'Confirmation_signal_values_read', 'semantic_relabeling_permitted']:
        if validation_experience_transfer_design_lock.get(key) is not False:
            raise RuntimeError(f'validation_experience_transfer_design boundary drift: {key}')
    rehash_rows = []
    for rel, expected in validation_experience_transfer_design_lock.get('hashes', {}).items():
        p = project / rel
        verify(p, expected, 'validation_experience_transfer_design child ' + rel)
        rehash_rows.append({'source': 'validation_experience_transfer_design_fixed_artifact', 'path': rel, 'expected_sha256': expected, 'observed_sha256': sha256(p), 'passed': True})
    policy = readj(project / validation_experience_transfer_design_POLICY)
    if canonical_sha(policy) != validation_experience_transfer_design_POLICY_CANONICAL_SHA256:
        raise RuntimeError('validation_experience_transfer_design policy canonical SHA drift')
    if policy.get('eligible_K') != ELIGIBLE_K:
        raise RuntimeError('validation_experience_transfer_design policy eligible-K drift')
    if policy.get('next_phase') != PHASE:
        raise RuntimeError('validation_experience_transfer_design policy next-phase drift')
    if policy.get('fixed_before_validation_signal_access') is not True:
        raise RuntimeError('validation_experience_transfer_design was not fixed before Validation signal access')
    if policy.get('representation', {}).get('stages') != ['stimulus_early', 'stimulus_late']:
        raise RuntimeError('validation_experience_transfer_design representation-stage drift')
    if policy.get('representation', {}).get('rating_early_forbidden') is not True:
        raise RuntimeError('validation_experience_transfer_design rating boundary drift')
    verify(project / validation_feature_trajectory_LOCK, validation_feature_trajectory_LOCK_SHA256, 'historical validation_feature_trajectory final lock')
    rehash_rows.append({'source': 'validation_feature_trajectory_historical_final_lock', 'path': str(validation_feature_trajectory_LOCK), 'expected_sha256': validation_feature_trajectory_LOCK_SHA256, 'observed_sha256': sha256(project / validation_feature_trajectory_LOCK), 'passed': True})
    verify(project / VALIDATION_COMPACT, VALIDATION_COMPACT_SHA256, 'Validation compact EEG store')
    rehash_rows.append({'source': 'Validation_compact_EEG_array', 'path': str(VALIDATION_COMPACT), 'expected_sha256': VALIDATION_COMPACT_SHA256, 'observed_sha256': sha256(project / VALIDATION_COMPACT), 'passed': True})
    verify(project / VALIDATION_MANIFEST, VALIDATION_MANIFEST_SHA256, 'Validation compact manifest')
    rehash_rows.append({'source': 'Validation_compact_manifest', 'path': str(VALIDATION_MANIFEST), 'expected_sha256': VALIDATION_MANIFEST_SHA256, 'observed_sha256': sha256(project / VALIDATION_MANIFEST), 'passed': True})
    return (validation_experience_transfer_design_lock, policy, pd.DataFrame(rehash_rows))

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
        row: dict[str, Any] = {'k': int(k), 'subject_id': str(subject), 'paired_experience_trials': int(n), 'eligible_for_transfer_gate': bool(eligible), 'in_support_fraction': float(g.in_discovery_support.mean()) if n else float('nan'), 'low_margin_fraction': float(g.low_margin.mean()) if n else float('nan'), 'distinct_families_present': int(g.family_1based.nunique()) if n else 0}
        for fam in range(1, k + 1):
            count = int(g.family_1based.eq(fam).sum())
            row[f'family{fam}_trials'] = count
            row[f'family{fam}_occupancy'] = float(count / n) if n else float('nan')
        participant_rows.append(row)
    participant = pd.DataFrame(participant_rows)
    eligible_p = participant[participant.eligible_for_transfer_gate].copy()
    eligible_subjects = set(eligible_p.subject_id.astype(str))
    eligible_a = a[a.subject_id.astype(str).isin(eligible_subjects)].copy()
    paired_fraction = float(len(pair) / AUTHORITATIVE_VALIDATION_TRIALS)
    pb_support = float(eligible_p.in_support_fraction.mean()) if len(eligible_p) else float('nan')
    family_stats: list[dict[str, Any]] = []
    for fam in range(1, k + 1):
        trial_count = int(eligible_a.family_1based.eq(fam).sum())
        distinct_p = int(eligible_a.loc[eligible_a.family_1based.eq(fam), 'subject_id'].astype(str).nunique())
        occ_col = f'family{fam}_occupancy'
        pb_occ = float(eligible_p[occ_col].mean()) if len(eligible_p) else float('nan')
        family_stats.append({'family_1based': fam, 'trial_count_eligible_participants': trial_count, 'distinct_eligible_participants': distinct_p, 'participant_balanced_occupancy': pb_occ})
    gates = []

    def add_gate(gid: str, metric: str, observed: Any, threshold: Any, comparison: str, passed: bool) -> None:
        gates.append({'k': int(k), 'gate_id': gid, 'metric': metric, 'observed': observed, 'threshold': threshold, 'comparison': comparison, 'pass': bool(passed)})
    add_gate('source_hashes', 'all source/store hash gates pass', True, True, '==', True)
    min_ep = int(gate_policy['minimum_eligible_participants'])
    add_gate('eligible_participants', 'eligible Validation participants', int(len(eligible_p)), min_ep, '>=', len(eligible_p) >= min_ep)
    min_pair = float(gate_policy['minimum_paired_experience_fraction_of_authoritative_validation_trials'])
    add_gate('paired_fraction', 'paired experience trials / authoritative Validation trials', paired_fraction, min_pair, '>=', paired_fraction >= min_pair)
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
    summary = {'k': int(k), 'replicated': passed, 'eligible_participants': int(len(eligible_p)), 'paired_experience_trials_all_participants': int(len(pair)), 'paired_fraction_of_494': paired_fraction, 'assigned_trials_eligible_participants': int(len(eligible_a)), 'participant_balanced_in_support_fraction': pb_support, 'overall_in_support_fraction': float(eligible_a.in_discovery_support.mean()) if len(eligible_a) else float('nan'), 'participant_balanced_low_margin_fraction': float(eligible_p.low_margin_fraction.mean()) if len(eligible_p) else float('nan'), 'overall_low_margin_fraction': float(eligible_a.low_margin.mean()) if len(eligible_a) else float('nan'), 'families': family_stats}
    return (a, participant, gate, summary)

def self_test() -> None:
    rng = np.random.default_rng(540)
    subjects = np.repeat([f'sub-{i:02d}' for i in range(1, 11)], 20)
    n = len(subjects)
    base = pd.DataFrame({'raw_behavior_row_id': [f'v{i:04d}' for i in range(n)], 'subject_id': subjects, 'stimulus_early_row_0based': np.arange(n), 'stimulus_late_row_0based': np.arange(n) + n})
    early = rng.normal(size=(n, 169))
    late = early + rng.normal(scale=0.7, size=(n, 169))
    fallback = np.ones(14, dtype=float)
    sig, stage, scaling, zero = build_validation_signatures(base, early, late, fallback)
    assert len(sig) == n and sig[STD].shape == (n, 14)
    assert len(stage) == 20 and len(scaling) == 10
    centers = sig[STD].to_numpy(float)[[0, 55, 110]].copy()
    lab, dist, margin, d = assign_fixed(sig[STD].to_numpy(float), centers)
    assert lab.min() >= 0 and lab.max() < 3
    assert np.isfinite(dist).all() and np.isfinite(margin).all() and (d.shape == (n, 3))
    mid = ((centers[0] + centers[1]) / 2.0)[None, :]
    mlab, *_ = assign_fixed(mid, centers[:2])
    assert int(mlab[0]) == 0
    assert not zero or isinstance(zero, dict)
    print('validation_stimulus_experience_multiresolution_transfer self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    started = time.time()
    print(f'[{now()}] verifying prespecified validation_experience_transfer_design transfer design and exact Validation compact store', flush=True)
    validation_experience_transfer_design_lock, policy, source_rehash = verify_inputs(project)
    out = project / OUT
    final_lock = out / 'qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
    if out.exists():
        if final_lock.is_file() and (not args.force):
            raise RuntimeError(f'Completed validation_stimulus_experience_multiresolution_transfer output exists: {out}\nUse --force only if intentionally rematerializing the identical fixed transfer.')
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(project / VALIDATION_MANIFEST, low_memory=False)
    pair = build_pair_manifest(manifest)
    if manifest.subject_id.astype(str).nunique() != 12:
        raise RuntimeError(f'Validation compact manifest subject-count drift: {manifest.subject_id.astype(str).nunique()}')
    if len(pair) > AUTHORITATIVE_VALIDATION_TRIALS:
        raise RuntimeError('Paired Validation experience trials exceed authoritative trial count')
    pair_fraction = len(pair) / AUTHORITATIVE_VALIDATION_TRIALS
    print(f'[{now()}] Validation stimulus-only pairs: {len(pair)}/{AUTHORITATIVE_VALIDATION_TRIALS} ({pair_fraction:.4f}); participants={pair.subject_id.nunique()}', flush=True)
    compact = np.load(project / VALIDATION_COMPACT, mmap_mode='r', allow_pickle=False)
    if compact.ndim != 2 or compact.shape[1] != 169:
        raise RuntimeError(f'Unexpected Validation compact shape: {compact.shape}')
    erow = pair.stimulus_early_row_0based.to_numpy(int)
    lrow = pair.stimulus_late_row_0based.to_numpy(int)
    if len(pair) == 0:
        raise RuntimeError('No Validation stimulus_early/stimulus_late pairs')
    if erow.min() < 0 or lrow.min() < 0 or erow.max() >= len(compact) or (lrow.max() >= len(compact)):
        raise RuntimeError('Validation pair manifest references compact rows outside store')
    print(f'[{now()}] opening only Validation stimulus_early/stimulus_late compact EEG values', flush=True)
    early = np.asarray(compact[erow], dtype=np.float64)
    late = np.asarray(compact[lrow], dtype=np.float64)
    del compact
    fallback_df = pd.read_csv(project / validation_experience_transfer_design_FALLBACK)
    if fallback_df.coordinate.astype(str).tolist() != COORDS:
        raise RuntimeError('validation_experience_transfer_design fallback-IQR coordinate order drift')
    fallback = fallback_df.discovery_participant_balanced_pooled_iqr.to_numpy(float)
    if fallback.shape != (14,) or not np.isfinite(fallback).all():
        raise RuntimeError('Invalid prespecified Discovery fallback IQR')
    sig, stage_audit, scaling_audit, zero_counts = build_validation_signatures(pair, early, late, fallback)
    print(f'[{now()}] constructed prespecified 14-cell Validation stimulus-only signatures; rating_early and all behavioral targets remain unopened', flush=True)
    cent = pd.read_csv(project / validation_experience_transfer_design_CENTROIDS)
    thr = pd.read_csv(project / validation_experience_transfer_design_THRESHOLDS)
    if sorted(cent.k.astype(int).unique().tolist()) != ELIGIBLE_K:
        raise RuntimeError('validation_experience_transfer_design centroid K-set drift')
    if sorted(thr.k.astype(int).unique().tolist()) != ELIGIBLE_K:
        raise RuntimeError('validation_experience_transfer_design threshold K-set drift')
    for c in STD:
        if c not in cent.columns:
            raise RuntimeError(f'Missing prespecified centroid coordinate: {c}')
    all_assign = []
    all_participant = []
    all_gate = []
    resolution_rows = []
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
    pair_out['authoritative_validation_trials'] = AUTHORITATIVE_VALIDATION_TRIALS
    pair_out['paired_fraction_of_authoritative_validation_trials'] = pair_fraction
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_pair_manifest.csv', pair_out)
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_stage_preprocessing_audit.csv', stage_audit)
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_signature_scaling_audit.csv', scaling_audit)
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv', sig)
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv', assignments)
    atomic_csv(out / 'qc/validation_stimulus_experience_multiresolution_transfer_validation_participant_transfer_summary_all_k.csv', participant)
    atomic_csv(out / 'qc/validation_stimulus_experience_multiresolution_transfer_validation_structural_transfer_gate_all_k.csv', gates)
    atomic_csv(out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_resolution_summary.csv', resolution)
    atomic_csv(out / 'qc/validation_stimulus_experience_multiresolution_transfer_source_rehash.csv', source_rehash)
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'synthesize the one-shot Validation stimulus-only structural transfer; specification whether/which K may advance unchanged to Confirmation, without opening Confirmation', 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'any_K_replicated': bool(passing_k), 'confirmation_signal_opening_in_validation_experience_transfer_synthesis': False, 'confirmation_design_specification_permitted': bool(passing_k), 'if_no_K_pass': 'close the discrete stimulus-experience transfer branch and move to the separately preplanned report-transition/continuous route; no K rescue or retuning', 'if_one_or_more_K_pass': 'only passing K may be considered for a separately prespecified Confirmation transfer; validation_experience_transfer_synthesis itself must not open Confirmation signal values', 'forbidden': ['Validation behavioral reports', 'emotion category target mapping', 'context target mapping', 'stimulus identity target mapping', 'rating_early EEG', 'centroid refit', 'threshold tuning', 'K reselection', 'family relabeling', 'semantic naming', 'Confirmation signal access in validation_experience_transfer_synthesis']}
    writej(project / NEXT_CONFIG, next_policy)
    summary_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_stimulus_experience_multiresolution_transfer_complete', 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'any_K_replicated': bool(passing_k), 'counts': {'validation_manifest_subjects': int(manifest.subject_id.astype(str).nunique()), 'authoritative_validation_trials': AUTHORITATIVE_VALIDATION_TRIALS, 'paired_stimulus_early_late_trials': int(len(pair)), 'paired_participants': int(pair.subject_id.nunique()), 'participants_with_at_least_16_pairs': int(pair.loc[pair.participant_eligible, 'subject_id'].nunique())}, 'paired_fraction_of_494': float(pair_fraction), 'resolution_results': resolution.to_dict('records'), 'zero_norm_counts': zero_counts, 'boundary': {'Validation_stimulus_early_values_read': True, 'Validation_stimulus_late_values_read': True, 'Validation_rating_early_values_read': False, 'Validation_behavioral_values_read': False, 'Validation_emotion_category_values_read': False, 'Validation_context_values_read': False, 'Validation_stimulus_identity_values_read': False, 'Confirmation_signal_values_read': False, 'centroids_refit': False, 'thresholds_tuned': False, 'K_reselected': False, 'family_semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE, 'next_phase_role': next_policy['next_phase_role'], 'runtime_seconds': time.time() - started}
    writej(out / 'qc/validation_stimulus_experience_multiresolution_transfer_summary.json', summary_obj)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'one_shot_validation_transfer_executed', 'source_code_sha256': sha256(Path(__file__).resolve()), 'validation_experience_transfer_design_final_lock_sha256': validation_experience_transfer_design_LOCK_SHA256, 'validation_experience_transfer_design_policy_canonical_sha256': validation_experience_transfer_design_POLICY_CANONICAL_SHA256, 'validation_feature_trajectory_historical_final_lock_sha256': validation_feature_trajectory_LOCK_SHA256, 'Validation_compact_sha256': VALIDATION_COMPACT_SHA256, 'Validation_manifest_sha256': VALIDATION_MANIFEST_SHA256, 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'prespecified_gate_policy': gate_policy, 'boundary': summary_obj['boundary'], 'next_phase': NEXT_PHASE}
    writej(out / 'qc/validation_stimulus_experience_multiresolution_transfer_contract.json', contract)
    lines = ['# validation_stimulus_experience_multiresolution_transfer — One-shot Validation stimulus-experience multi-resolution transfer', '', '**Validation stimulus-only EEG was opened exactly once under the validation_experience_transfer_design-prespecified transfer contract.**', '', f'- Authoritative Validation trials: **{AUTHORITATIVE_VALIDATION_TRIALS}**', f'- Paired stimulus_early→stimulus_late trials: **{len(pair)}** ({pair_fraction:.4f})', f"- Participants with ≥16 paired trials: **{int(pair.loc[pair.participant_eligible, 'subject_id'].nunique())}**", f'- Tested Discovery resolutions: **{ELIGIBLE_K}**', f"- Passing Validation resolutions: **{(passing_k if passing_k else 'NONE')}**", '', '## prespecified structural-transfer results', resolution.to_string(index=False), '', '## Interpretation boundary', '- Replication, if present, supports only transfer of the prespecified stimulus-only reconfiguration partition.', '- It does not identify joy, fear, valence, liking, arousal, context, category, or stimulus semantics.', '- `rating_early` EEG and all Validation behavioral/category/context/stimulus targets remained unopened.', '- Centroids were not refit, K was not reselected, support thresholds were not tuned, and families were not renamed.', '- Confirmation remains closed.', '', f'## Next', f'**{NEXT_PHASE}** — synthesize this Validation result and specification the Confirmation/closure decision without opening Confirmation.', '']
    (out / 'reports/validation_stimulus_experience_multiresolution_transfer_report.md').write_text(''.join(lines), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, out / 'qc/validation_stimulus_experience_multiresolution_transfer_contract.json', out / 'qc/validation_stimulus_experience_multiresolution_transfer_summary.json', out / 'qc/validation_stimulus_experience_multiresolution_transfer_source_rehash.csv', out / 'qc/validation_stimulus_experience_multiresolution_transfer_validation_participant_transfer_summary_all_k.csv', out / 'qc/validation_stimulus_experience_multiresolution_transfer_validation_structural_transfer_gate_all_k.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_pair_manifest.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_stage_preprocessing_audit.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_signature_scaling_audit.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv', out / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_resolution_summary.csv', out / 'reports/validation_stimulus_experience_multiresolution_transfer_report.md']
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_stimulus_experience_multiresolution_transfer_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'tested_K': ELIGIBLE_K, 'passing_K': passing_k, 'any_K_replicated': bool(passing_k), 'Validation_behavioral_values_read': False, 'Validation_rating_early_values_read': False, 'Confirmation_signal_values_read': False, 'centroids_refit': False, 'thresholds_tuned': False, 'K_reselected': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final_obj)
    script_dir = Path(__file__).resolve().parent
    files = lock_targets + [final_lock]
    for p in sorted(script_dir.iterdir()):
        if p.is_file():
            files.append(p)
    files += [project / validation_experience_transfer_design_POLICY, project / validation_experience_transfer_design_LOCK, project / validation_experience_transfer_design_CENTROIDS, project / validation_experience_transfer_design_FALLBACK, project / validation_experience_transfer_design_THRESHOLDS, project / validation_experience_transfer_design_GATE_SPEC]
    print('')
    print('validation_stimulus_experience_multiresolution_transfer PASS')
    print(f'Validation stimulus-only pairs: {len(pair)}/{AUTHORITATIVE_VALIDATION_TRIALS}')
    print(f'Tested K: {ELIGIBLE_K}')
    print(f"Passing K: {(passing_k if passing_k else 'NONE')}")
    for r in resolution.itertuples(index=False):
        print(f'K={int(r.k)} replicated={bool(r.replicated)}; eligible_participants={int(r.eligible_participants)}; PB_support={float(r.participant_balanced_in_support_fraction):.6f}')
    print('Validation behavioral values opened: False')
    print('Validation rating_early EEG opened: False')
    print('Confirmation opened: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())