# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase latent_state_fit: execute the prespecified outcome-free latent-state design.

This phase fits one independent within-person PCA + diagonal Gaussian-mixture
model for every latent_state_design-eligible discovery participant, executes every prespecified
trial-coupled stability resample and K-selection rule, writes continuous
three-epoch trajectories for all eligible participants, writes discrete
trajectories only for stable selected solutions, and runs the prespecified
within-person trial-linkage nulls.

Valence, liking, arousal, emotion reports, context, report-derived factors,
validation participants, and confirmation participants are not read. QC is
loaded only after primary state assignments and trajectories have been written
and SHA-256 fixed; it is used descriptively and can never alter a state."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.mixture import GaussianMixture
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'latent_state_fit'
NEXT_PHASE = 'cross_person_alignment_design'
SEED = 540
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
FAMILY_TO_INDEX = {name: i for i, name in enumerate(FAMILIES)}
PER_EPOCH_COMPACT = 169
PER_EPOCH_QC = 65
EPOCH_MANIFEST_USECOLS = ['epoch_store_row_0based', 'raw_behavior_row_id', 'subject_id', 'epoch_family']
CACHE_REL = Path('outputs/large/latent_state_fit_latent_state_cache_v1')
REL = {'latent_state_design_summary': Path('outputs/qc/latent_state_design_latent_state_design_specification_summary.json'), 'latent_state_design_contract': Path('config/latent_state_design_latent_state_design_contract.json'), 'latent_state_design_latent_policy': Path('analysis_specifications/S037_latent_state.json'), 'latent_state_design_trajectory_policy': Path('analysis_specifications/S038_latent_state_trajectory.json'), 'latent_state_design_next': Path('analysis_specifications/S036_latent_state_fit.json'), 'trial_manifest': Path('outputs/tables/latent_state_design_common_trial_manifest.csv'), 'subject_eligibility': Path('outputs/qc/latent_state_design_subject_eligibility.csv'), 'participant_feature_audit': Path('outputs/tables/latent_state_design_participant_feature_audit.csv'), 'compact_feature_schema': Path('outputs/tables/latent_state_design_compact_feature_schema.csv'), 'latent_state_design_epoch_rehash': Path('outputs/qc/latent_state_design_epoch_store_rehash.csv'), 'epoch_store_summary': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_store_summary.json'), 'epoch_manifest': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_manifest.csv'), 'epoch_compact': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy'), 'epoch_qc': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_qc_float32.npy')}
latent_state_design_OUTPUT_MAP = {'epoch_store_rehash': REL['latent_state_design_epoch_rehash'], 'subject_eligibility': REL['subject_eligibility'], 'summary': REL['latent_state_design_summary'], 'trial_manifest': REL['trial_manifest'], 'feature_schema': REL['compact_feature_schema'], 'participant_feature_audit': REL['participant_feature_audit'], 'latent_policy': REL['latent_state_design_latent_policy'], 'trajectory_policy': REL['latent_state_design_trajectory_policy']}
BANNED_EXACT_COLUMNS = {'valence', 'liking', 'arousal', 'emotion', 'emotion_label', 'emotion_category', 'context', 'condition', 'stimulus_category', 'stimulus_type', 'pleasantness', 'unpleasantness', 'wanting', 'pain_intensity', 'pain_unpleasantness', 'subjective_factor_1', 'subjective_factor_2', 'report_factor_1', 'report_factor_2'}

@dataclass
class prespecifiedPolicy:
    policy_sha256: str
    candidate_k: list[int]
    pca_components: int
    iqr_floor: float
    winsor_limit: float
    covariance_type: str
    reg_covar: float
    n_init: int
    max_iter: int
    stability_repeats: int
    trial_fraction: float
    median_ari_min: float
    p10_ari_min: float
    collapse_rate_max: float
    min_state_observations: int
    min_state_unique_trials: int
    bic_delta: float
    null_repeats: int

@dataclass
class SubjectData:
    subject_id: str
    trial_ids: np.ndarray
    trial_numbers: np.ndarray
    epoch_rows: np.ndarray
    epoch_families: np.ndarray
    compact: np.ndarray
    retained_features: np.ndarray
    input_sha256: str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--recompute-cache', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def csha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def jread(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def bval(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {'true', '1', 'yes'}:
        return True
    if s in {'false', '0', 'no'}:
        return False
    raise RuntimeError(f'Invalid boolean value: {value!r}')

def required(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(map(str, missing)))

def atom_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '', encoding='utf-8')
    os.replace(tmp, path)

def ensure_no_banned_columns(df: pd.DataFrame, source: str) -> None:
    bad = sorted({str(c).strip().lower() for c in df.columns} & BANNED_EXACT_COLUMNS)
    if bad:
        raise RuntimeError(f'Forbidden outcome/context columns found in {source}: {bad}')

def output_paths(project: Path) -> dict[str, Path]:
    return {'log': project / 'outputs/logs/latent_state_fit_latent_state_fit.log', 'epoch_store_rehash': project / 'outputs/qc/latent_state_fit_epoch_store_rehash.csv', 'checkpoint_inventory': project / 'outputs/qc/latent_state_fit_checkpoint_inventory.csv', 'primary_lock': project / 'outputs/qc/latent_state_fit_primary_lock_hashes.json', 'qc_state_descriptives': project / 'outputs/qc/latent_state_fit_qc_state_descriptives.csv', 'epoch_state_association': project / 'outputs/qc/latent_state_fit_epoch_state_association.csv', 'summary': project / 'outputs/qc/latent_state_fit_latent_state_fit_summary.json', 'report': project / 'outputs/reports/latent_state_fit_latent_state_fit_report.md', 'preprocessing': project / 'outputs/tables/latent_state_fit_participant_preprocessing_summary.csv', 'candidate_metrics': project / 'outputs/tables/latent_state_fit_state_candidate_metrics.csv', 'stability': project / 'outputs/tables/latent_state_fit_stability_replicates.csv', 'selection': project / 'outputs/tables/latent_state_fit_participant_state_selection.csv', 'pca_scores': project / 'outputs/tables/latent_state_fit_epoch_pca_scores.csv', 'pca_loadings': project / 'outputs/tables/latent_state_fit_pca_loadings.csv', 'assignments': project / 'outputs/tables/latent_state_fit_state_assignments.csv', 'continuous_trajectories': project / 'outputs/tables/latent_state_fit_continuous_trajectories.csv', 'discrete_trajectories': project / 'outputs/tables/latent_state_fit_discrete_trajectories.csv', 'occupancy': project / 'outputs/tables/latent_state_fit_state_occupancy.csv', 'transitions': project / 'outputs/tables/latent_state_fit_transition_matrices.csv', 'transition_summary': project / 'outputs/tables/latent_state_fit_participant_transition_summaries.csv', 'null_replicates': project / 'outputs/tables/latent_state_fit_trajectory_null_replicates.csv', 'null_tests': project / 'outputs/tables/latent_state_fit_trajectory_null_tests.csv', 'centroids_pca': project / 'outputs/tables/latent_state_fit_state_centroids_pca.csv', 'centroids_common': project / 'outputs/tables/latent_state_fit_state_centroids_common_axes.csv', 'contract': project / 'config/latent_state_fit_latent_state_fit_contract.json', 'next_policy': project / 'analysis_specifications/S039_latent_state_fit_cross_person_alignment.json'}

def load_and_verify_specification(project: Path, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], prespecifiedPolicy]:
    summary = jread(paths['latent_state_design_summary'])
    contract = jread(paths['latent_state_design_contract'])
    latent = jread(paths['latent_state_design_latent_policy'])
    trajectory = jread(paths['latent_state_design_trajectory_policy'])
    nxt = jread(paths['latent_state_design_next'])
    if summary.get('phase') != 'latent_state_design' or contract.get('phase') != 'latent_state_design':
        raise RuntimeError('latent_state_design phase mismatch')
    if summary.get('status') != 'outcome_free_within_person_latent_state_and_trajectory_design_prespecified_candidate':
        raise RuntimeError(f"Unexpected latent_state_design status: {summary.get('status')}")
    expected_policy_sha = str(summary.get('design_policy_sha256', ''))
    if not expected_policy_sha or csha(latent) != expected_policy_sha:
        raise RuntimeError('prespecified latent-state policy canonical SHA-256 mismatch')
    if str(trajectory.get('policy_sha256')) != expected_policy_sha:
        raise RuntimeError('prespecified trajectory policy references the wrong design SHA-256')
    if csha(contract.get('design_policy')) != expected_policy_sha:
        raise RuntimeError('latent_state_design contract embeds a different design policy')
    decision = summary.get('decision', {})
    required_true = ['secondary_models_verified_complete', 'epoch_store_byte_verified', 'latent_state_fit_design_prespecified']
    bad = [k for k in required_true if not bool(decision.get(k))]
    if bad:
        raise RuntimeError(f'latent_state_design incomplete decision flags: {bad}')
    forbidden_true = ['outcome_or_report_columns_read', 'latent_state_models_fitted', 'state_count_selected', 'cross_person_alignment_performed', 'validation_opened', 'confirmation_opened']
    bad = [k for k in forbidden_true if bool(decision.get(k))]
    if bad:
        raise RuntimeError(f'latent_state_design reports forbidden prior actions: {bad}')
    if not bool(nxt.get('next_phase_permitted')) or nxt.get('next_phase') != PHASE:
        raise RuntimeError('latent_state_design did not authorize latent_state_fit')
    if bool(nxt.get('validation_opened')) or bool(nxt.get('confirmation_opened')):
        raise RuntimeError('latent_state_design next policy opened validation/confirmation')
    output_hashes = contract.get('output_hashes', {})
    for key, rel in latent_state_design_OUTPUT_MAP.items():
        p = project / rel
        if key not in output_hashes:
            raise RuntimeError(f'latent_state_design contract missing output hash: {key}')
        if not p.is_file() or sha(p) != str(output_hashes[key]):
            raise RuntimeError(f'latent_state_design output integrity failure: {key} -> {p}')
    pca = latent['primary_representation']['pca']
    state = latent['state_model']
    stability = latent['state_count_and_stability']
    reqs = stability['eligible_k_requirements']
    null = latent['trajectory_outputs']['null']
    policy = prespecifiedPolicy(policy_sha256=expected_policy_sha, candidate_k=[int(x) for x in state['candidate_state_counts']], pca_components=int(pca['fixed_components']), iqr_floor=float(latent['primary_representation']['iqr_floor']), winsor_limit=float(latent['primary_representation']['winsor_limit']), covariance_type=str(state['covariance_type']), reg_covar=float(state['reg_covar']), n_init=int(state['n_init']), max_iter=int(state['max_iter']), stability_repeats=int(stability['repeats']), trial_fraction=float(stability['trial_fraction_without_replacement']), median_ari_min=float(reqs['median_ari_at_least']), p10_ari_min=float(reqs['ari_10th_percentile_at_least']), collapse_rate_max=float(reqs['collapse_rate_at_most']), min_state_observations=int(str(stability['collapse_definition']).split('fewer than ')[1].split(' observations')[0]), min_state_unique_trials=int(str(stability['collapse_definition']).split('or fewer than ')[1].split(' unique')[0]), bic_delta=float(str(stability['selection_rule']).split(' + ')[1]), null_repeats=int(null['repeats']))
    if policy.candidate_k != [2, 3, 4, 5]:
        raise RuntimeError(f'Unexpected candidate K: {policy.candidate_k}')
    if FAMILIES != list(latent['observation_unit']['epoch_order']):
        raise RuntimeError('Epoch order differs from prespecified design')
    if int(latent['primary_representation']['cells_per_epoch']) != PER_EPOCH_COMPACT:
        raise RuntimeError('Compact representation width differs from prespecified design')
    if bool(latent['primary_representation']['qc_cells_in_primary_state_fit']):
        raise RuntimeError('prespecified policy unexpectedly permits QC in the primary fit')
    return (summary, contract, latent, policy)

def verify_epoch_store(paths: dict[str, Path], latent_state_design_contract: dict[str, Any]) -> pd.DataFrame:
    summary = jread(paths['epoch_store_summary'])
    if not bool(summary.get('complete')):
        raise RuntimeError('Epoch store is not marked complete')
    if int(summary.get('rows', -1)) != 3853:
        raise RuntimeError(f"Unexpected epoch-store rows: {summary.get('rows')}")
    if int(summary.get('compact_cells_per_epoch', -1)) != PER_EPOCH_COMPACT:
        raise RuntimeError('Unexpected compact epoch width')
    if int(summary.get('qc_cells_per_epoch', -1)) != PER_EPOCH_QC:
        raise RuntimeError('Unexpected QC epoch width')
    expected_inputs = latent_state_design_contract.get('input_hashes', {})
    key_map = {'epoch_manifest': 'epoch_manifest', 'epoch_compact': 'epoch_compact', 'epoch_qc': 'epoch_qc'}
    rows = []
    for key, path_key in key_map.items():
        p = paths[path_key]
        expected = str(expected_inputs.get(key, ''))
        observed = sha(p)
        if not expected or observed != expected:
            raise RuntimeError(f'latent_state_design-prespecified epoch-store hash mismatch: {key}')
        rows.append({'relative_path': p.name, 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': observed, 'passed': True})
    summary_expected = str(expected_inputs.get('epoch_store_summary', ''))
    summary_observed = sha(paths['epoch_store_summary'])
    if summary_expected and summary_observed != summary_expected:
        raise RuntimeError('Epoch-store summary hash mismatch')
    rows.append({'relative_path': paths['epoch_store_summary'].name, 'stored_path': str(paths['epoch_store_summary']), 'size_bytes': paths['epoch_store_summary'].stat().st_size, 'sha256': summary_observed, 'passed': True})
    return pd.DataFrame(rows)

def validate_inputs(trial: pd.DataFrame, eligibility: pd.DataFrame, audit: pd.DataFrame, schema: pd.DataFrame, epoch_manifest: pd.DataFrame, compact: np.ndarray) -> list[str]:
    for name, df in [('trial_manifest', trial), ('subject_eligibility', eligibility), ('participant_feature_audit', audit), ('compact_feature_schema', schema), ('epoch_manifest', epoch_manifest)]:
        ensure_no_banned_columns(df, name)
    trial_required = {'raw_behavior_row_id', 'subject_id', 'trial_number', 'stimulus_early_epoch_store_row_0based', 'stimulus_late_epoch_store_row_0based', 'rating_early_epoch_store_row_0based', 'latent_state_fit_eligible_subject'}
    if not trial_required.issubset(trial.columns):
        raise RuntimeError(f'Trial manifest missing columns: {sorted(trial_required - set(trial.columns))}')
    elig_required = {'subject_id', 'latent_state_fit_eligible', 'common_trials', 'retained_features_after_finite_rule'}
    if not elig_required.issubset(eligibility.columns):
        raise RuntimeError(f'Eligibility table missing columns: {sorted(elig_required - set(eligibility.columns))}')
    audit_required = {'subject_id', 'compact_cell_0based', 'passes_finite_fraction_rule'}
    if not audit_required.issubset(audit.columns):
        raise RuntimeError(f'Feature audit missing columns: {sorted(audit_required - set(audit.columns))}')
    if schema.shape[0] != PER_EPOCH_COMPACT or not np.array_equal(schema['compact_cell_0based'].to_numpy(int), np.arange(PER_EPOCH_COMPACT)):
        raise RuntimeError('Compact feature schema is incomplete or out of order')
    if compact.shape != (3853, PER_EPOCH_COMPACT):
        raise RuntimeError(f'Unexpected compact array shape: {compact.shape}')
    eligible_subjects = eligibility.loc[eligibility.latent_state_fit_eligible.map(bval), 'subject_id'].astype(str).tolist()
    trial_eligible = trial[trial.latent_state_fit_eligible_subject.map(bval)].copy()
    if sorted(trial_eligible.subject_id.astype(str).unique()) != sorted(eligible_subjects):
        raise RuntimeError('Eligible subject mismatch between trial manifest and eligibility table')
    if trial_eligible.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Duplicate raw_behavior_row_id in eligible trial manifest')
    if len(eligible_subjects) != 31 or len(trial_eligible) != 1237:
        raise RuntimeError(f'prespecified cohort mismatch: {len(eligible_subjects)} participants, {len(trial_eligible)} trials')
    em_required = {'epoch_store_row_0based', 'raw_behavior_row_id', 'subject_id', 'epoch_family'}
    if not em_required.issubset(epoch_manifest.columns):
        raise RuntimeError(f'Epoch manifest missing columns: {sorted(em_required - set(epoch_manifest.columns))}')
    em = epoch_manifest.set_index('epoch_store_row_0based', verify_integrity=True)
    for family in FAMILIES:
        col = f'{family}_epoch_store_row_0based'
        idx = trial_eligible[col].to_numpy(int)
        if idx.min() < 0 or idx.max() >= compact.shape[0]:
            raise RuntimeError(f'{family} epoch row outside compact array')
        chk = em.loc[idx]
        if not chk.epoch_family.astype(str).eq(family).all():
            raise RuntimeError(f'Epoch-family mismatch in {family} rows')
        if not np.array_equal(chk.raw_behavior_row_id.astype(str).to_numpy(), trial_eligible.raw_behavior_row_id.astype(str).to_numpy()):
            raise RuntimeError(f'Trial identity mismatch in {family} rows')
        if not np.array_equal(chk.subject_id.astype(str).to_numpy(), trial_eligible.subject_id.astype(str).to_numpy()):
            raise RuntimeError(f'Subject identity mismatch in {family} rows')
    return eligible_subjects

def stable_seed(subject_id: str, k: int=0, repeat: int=0, stream: int=0) -> int:
    payload = f'{SEED}|{subject_id}|{k}|{repeat}|{stream}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder='little', signed=False)

def subject_input_sha(subject_id: str, trial_ids: np.ndarray, epoch_rows: np.ndarray, retained: np.ndarray, raw: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(subject_id.encode('utf-8'))
    for value in trial_ids.astype(str):
        h.update(value.encode('utf-8') + b'\x00')
    h.update(np.asarray(epoch_rows, dtype=np.int64).tobytes())
    h.update(np.asarray(retained, dtype=np.int16).tobytes())
    h.update(np.ascontiguousarray(raw, dtype=np.float32).tobytes())
    return h.hexdigest()

def build_subject_data(subject_id: str, trial: pd.DataFrame, audit: pd.DataFrame, compact_store: np.ndarray) -> SubjectData:
    t = trial[(trial.subject_id.astype(str) == subject_id) & trial.latent_state_fit_eligible_subject.map(bval)].copy()
    t = t.sort_values(['subject_acquisition_order_1based', 'trial_number', 'raw_behavior_row_id']).reset_index(drop=True)
    if t.empty:
        raise RuntimeError(f'No eligible trials for {subject_id}')
    a = audit[audit.subject_id.astype(str).eq(subject_id)].copy()
    retained = np.sort(a.loc[a.passes_finite_fraction_rule.map(bval), 'compact_cell_0based'].to_numpy(int))
    if retained.size < 30:
        raise RuntimeError(f'{subject_id} has only {retained.size} retained compact cells')
    trial_ids: list[str] = []
    trial_numbers: list[int] = []
    rows: list[int] = []
    families: list[str] = []
    raw_rows: list[np.ndarray] = []
    for rec in t.itertuples(index=False):
        for family in FAMILIES:
            row = int(getattr(rec, f'{family}_epoch_store_row_0based'))
            trial_ids.append(str(rec.raw_behavior_row_id))
            trial_numbers.append(int(rec.trial_number))
            rows.append(row)
            families.append(family)
            raw_rows.append(np.asarray(compact_store[row, retained], dtype=np.float32))
    raw = np.vstack(raw_rows).astype(np.float32, copy=False)
    trial_ids_arr = np.asarray(trial_ids, dtype=str)
    trial_numbers_arr = np.asarray(trial_numbers, dtype=np.int32)
    rows_arr = np.asarray(rows, dtype=np.int32)
    families_arr = np.asarray(families, dtype=str)
    digest = subject_input_sha(subject_id, trial_ids_arr, rows_arr, retained, raw)
    return SubjectData(subject_id, trial_ids_arr, trial_numbers_arr, rows_arr, families_arr, raw, retained, digest)

def iqr(values: np.ndarray, axis: int=0) -> np.ndarray:
    return np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)

def preprocess_subject(data: SubjectData, policy: prespecifiedPolicy) -> dict[str, Any]:
    raw = np.asarray(data.compact, dtype=np.float64)
    f = raw.shape[1]
    pooled_iqr = iqr(raw, axis=0)
    keep = np.isfinite(pooled_iqr) & (pooled_iqr > policy.iqr_floor)
    if int(keep.sum()) < 30:
        raise RuntimeError(f'{data.subject_id}: fewer than 30 features survive pooled-IQR rule')
    raw = raw[:, keep]
    final_features = data.retained_features[keep]
    pooled_iqr = pooled_iqr[keep]
    medians = np.empty((3, raw.shape[1]), dtype=np.float64)
    scales = np.empty_like(medians)
    standardized = np.empty_like(raw)
    for family, fam_idx in FAMILY_TO_INDEX.items():
        mask = data.epoch_families == family
        block = raw[mask].copy()
        med = np.nanmedian(block, axis=0)
        if not np.isfinite(med).all():
            raise RuntimeError(f'{data.subject_id}: nonfinite family median after prespecified finite rule')
        missing = ~np.isfinite(block)
        if missing.any():
            block[missing] = np.broadcast_to(med, block.shape)[missing]
        fam_iqr = iqr(block, axis=0)
        scale = np.where(np.isfinite(fam_iqr) & (fam_iqr > policy.iqr_floor), fam_iqr, pooled_iqr)
        if not np.isfinite(scale).all() or np.any(scale <= policy.iqr_floor):
            raise RuntimeError(f'{data.subject_id}: invalid robust scale after pooled fallback')
        z = (block - med) / scale
        z = np.clip(z, -policy.winsor_limit, policy.winsor_limit)
        standardized[mask] = z
        medians[fam_idx] = med
        scales[fam_idx] = scale
    n_components = min(policy.pca_components, standardized.shape[1], standardized.shape[0] - 1)
    if n_components < 2:
        raise RuntimeError(f'{data.subject_id}: fewer than two PCA components available')
    pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
    scores = pca.fit_transform(standardized)
    if not np.isfinite(scores).all():
        raise RuntimeError(f'{data.subject_id}: nonfinite PCA scores')
    return {'final_features': final_features.astype(np.int16), 'family_medians': medians, 'family_scales': scales, 'pooled_iqr': pooled_iqr, 'standardized': standardized, 'pca_scores': scores, 'pca_mean': pca.mean_, 'pca_components': pca.components_, 'pca_explained_variance': pca.explained_variance_, 'pca_explained_variance_ratio': pca.explained_variance_ratio_}

def canonicalize_gmm(model: GaussianMixture, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    means = np.asarray(model.means_, dtype=float)
    order = sorted(range(means.shape[0]), key=lambda idx: tuple((float(x) for x in means[idx])))
    mapping = {old: new + 1 for new, old in enumerate(order)}
    canonical_labels = np.asarray([mapping[int(v)] for v in labels], dtype=np.int16)
    return (canonical_labels, means[order], np.asarray(model.covariances_, dtype=float)[order], np.asarray(model.weights_, dtype=float)[order], mapping)

def state_collapse(labels: np.ndarray, trial_ids: np.ndarray, k: int, min_obs: int, min_trials: int) -> tuple[bool, int, int]:
    counts = []
    unique_trials = []
    for state in range(1, k + 1):
        mask = labels == state
        counts.append(int(mask.sum()))
        unique_trials.append(int(np.unique(trial_ids[mask]).size))
    return (min(counts) < min_obs or min(unique_trials) < min_trials, min(counts), min(unique_trials))

def fit_gmm(scores: np.ndarray, k: int, policy: prespecifiedPolicy, random_state: int) -> GaussianMixture:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        model = GaussianMixture(n_components=k, covariance_type=policy.covariance_type, reg_covar=policy.reg_covar, n_init=policy.n_init, max_iter=policy.max_iter, random_state=random_state, init_params='kmeans')
        model.fit(scores)
    return model

def fit_subject_models(data: SubjectData, prep: dict[str, Any], policy: prespecifiedPolicy, repeats_override: int | None=None) -> dict[str, Any]:
    scores = np.asarray(prep['pca_scores'], dtype=float)
    candidate_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    labels_by_k = np.full((len(policy.candidate_k), scores.shape[0]), -1, dtype=np.int16)
    gmm_arrays: dict[str, np.ndarray] = {}
    repeats = policy.stability_repeats if repeats_override is None else int(repeats_override)
    unique_trials = np.unique(data.trial_ids)
    n_sample_trials = max(1, int(math.floor(policy.trial_fraction * unique_trials.size)))
    for k_idx, k in enumerate(policy.candidate_k):
        fit_error = ''
        converged = False
        bic = np.nan
        full_collapse = True
        min_obs = 0
        min_trials = 0
        full_labels = np.full(scores.shape[0], -1, dtype=np.int16)
        try:
            model = fit_gmm(scores, k, policy, stable_seed(data.subject_id, k, stream=1))
            raw_labels = model.predict(scores)
            full_labels, means, covariances, weights, _ = canonicalize_gmm(model, raw_labels)
            converged = bool(model.converged_)
            bic = float(model.bic(scores))
            full_collapse, min_obs, min_trials = state_collapse(full_labels, data.trial_ids, k, policy.min_state_observations, policy.min_state_unique_trials)
            labels_by_k[k_idx] = full_labels
            gmm_arrays[f'k{k}_means_pca'] = means.astype(np.float64)
            gmm_arrays[f'k{k}_covariances'] = covariances.astype(np.float64)
            gmm_arrays[f'k{k}_weights'] = weights.astype(np.float64)
        except Exception as exc:
            fit_error = f'{type(exc).__name__}: {exc}'
        for repeat in range(repeats):
            row = {'subject_id': data.subject_id, 'candidate_k': k, 'repeat_0based': repeat, 'sampled_trial_count': n_sample_trials, 'sampled_epoch_observations': n_sample_trials * 3, 'fit_converged': False, 'adjusted_rand_index': np.nan, 'collapsed': True, 'min_state_observations': 0, 'min_state_unique_trials': 0, 'fit_error': ''}
            try:
                rng = np.random.default_rng(stable_seed(data.subject_id, k, repeat, stream=2))
                sampled_trials = rng.choice(unique_trials, size=n_sample_trials, replace=False)
                mask = np.isin(data.trial_ids, sampled_trials)
                sub_trial_ids = data.trial_ids[mask]
                sub_data = SubjectData(subject_id=data.subject_id, trial_ids=sub_trial_ids, trial_numbers=data.trial_numbers[mask], epoch_rows=data.epoch_rows[mask], epoch_families=data.epoch_families[mask], compact=data.compact[mask], retained_features=data.retained_features, input_sha256='')
                sub_prep = preprocess_subject(sub_data, policy)
                sub_scores = np.asarray(sub_prep['pca_scores'], dtype=float)
                sub_model = fit_gmm(sub_scores, k, policy, stable_seed(data.subject_id, k, repeat, stream=3))
                sub_raw = sub_model.predict(sub_scores)
                sub_labels, _, _, _, _ = canonicalize_gmm(sub_model, sub_raw)
                sub_collapse, sub_min_obs, sub_min_trials = state_collapse(sub_labels, sub_trial_ids, k, policy.min_state_observations, policy.min_state_unique_trials)
                sub_converged = bool(sub_model.converged_)
                if sub_converged and np.all(full_labels[mask] > 0):
                    ari = float(adjusted_rand_score(full_labels[mask], sub_labels))
                else:
                    ari = np.nan
                    sub_collapse = True
                row.update({'fit_converged': sub_converged, 'adjusted_rand_index': ari, 'collapsed': bool(sub_collapse), 'min_state_observations': sub_min_obs, 'min_state_unique_trials': sub_min_trials})
            except Exception as exc:
                row['fit_error'] = f'{type(exc).__name__}: {exc}'
            stability_rows.append(row)
        stab = pd.DataFrame(stability_rows)
        current = stab[stab.candidate_k.eq(k)]
        valid = current.adjusted_rand_index.dropna().to_numpy(float)
        median_ari = float(np.median(valid)) if valid.size else np.nan
        p10_ari = float(np.quantile(valid, 0.1)) if valid.size else np.nan
        collapse_rate = float(current.collapsed.map(bval).mean())
        valid_repeats = int(valid.size)
        eligible = converged and (not full_collapse) and (not fit_error) and (valid_repeats == repeats) and (median_ari >= policy.median_ari_min) and (p10_ari >= policy.p10_ari_min) and (collapse_rate <= policy.collapse_rate_max)
        candidate_rows.append({'subject_id': data.subject_id, 'candidate_k': k, 'full_fit_converged': converged, 'full_fit_bic': bic, 'full_fit_collapsed': full_collapse, 'full_fit_min_state_observations': min_obs, 'full_fit_min_state_unique_trials': min_trials, 'stability_repeats_required': repeats, 'stability_valid_ari_repeats': valid_repeats, 'stability_median_ari': median_ari, 'stability_p10_ari': p10_ari, 'stability_collapse_rate': collapse_rate, 'eligible_k': eligible, 'full_fit_error': fit_error})
    candidate = pd.DataFrame(candidate_rows)
    eligible = candidate[candidate.eligible_k.map(bval)].copy()
    selected_k: int | None = None
    selection_reason = 'no_eligible_k_continuous_trajectory_only'
    if not eligible.empty:
        min_bic = float(eligible.full_fit_bic.min())
        within = eligible[eligible.full_fit_bic <= min_bic + policy.bic_delta]
        selected_k = int(within.candidate_k.min())
        selection_reason = f'smallest_eligible_k_within_delta_bic_{policy.bic_delta:g}'
    return {'candidate_metrics': candidate, 'stability': pd.DataFrame(stability_rows), 'labels_by_k': labels_by_k, 'gmm_arrays': gmm_arrays, 'selected_k': selected_k, 'selection_reason': selection_reason}

def checkpoint_files(folder: Path) -> dict[str, Path]:
    return {'metadata': folder / 'metadata.csv', 'arrays': folder / 'arrays.npz', 'candidate_metrics': folder / 'candidate_metrics.csv', 'stability': folder / 'stability_replicates.csv', 'selection': folder / 'selection.json', 'complete': folder / 'complete.json'}

def save_checkpoint(folder: Path, data: SubjectData, prep: dict[str, Any], fit: dict[str, Any], policy: prespecifiedPolicy) -> None:
    tmp = folder.with_name(folder.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    files = checkpoint_files(tmp)
    pd.DataFrame({'subject_id': data.subject_id, 'observation_row_0based': np.arange(data.trial_ids.size, dtype=int), 'raw_behavior_row_id': data.trial_ids, 'trial_number': data.trial_numbers, 'epoch_family': data.epoch_families, 'epoch_store_row_0based': data.epoch_rows}).to_csv(files['metadata'], index=False)
    arrays = {'retained_features': np.asarray(prep['final_features'], dtype=np.int16), 'family_medians': np.asarray(prep['family_medians'], dtype=np.float64), 'family_scales': np.asarray(prep['family_scales'], dtype=np.float64), 'pooled_iqr': np.asarray(prep['pooled_iqr'], dtype=np.float64), 'pca_scores': np.asarray(prep['pca_scores'], dtype=np.float32), 'pca_mean': np.asarray(prep['pca_mean'], dtype=np.float64), 'pca_components': np.asarray(prep['pca_components'], dtype=np.float64), 'pca_explained_variance': np.asarray(prep['pca_explained_variance'], dtype=np.float64), 'pca_explained_variance_ratio': np.asarray(prep['pca_explained_variance_ratio'], dtype=np.float64), 'labels_by_k': np.asarray(fit['labels_by_k'], dtype=np.int16)}
    arrays.update(fit['gmm_arrays'])
    np.savez_compressed(files['arrays'], **arrays)
    fit['candidate_metrics'].to_csv(files['candidate_metrics'], index=False)
    fit['stability'].to_csv(files['stability'], index=False)
    atom_json(files['selection'], {'subject_id': data.subject_id, 'selected_k': fit['selected_k'], 'stable_discrete_solution': fit['selected_k'] is not None, 'selection_reason': fit['selection_reason']})
    file_hashes = {key: sha(path) for key, path in files.items() if key != 'complete'}
    atom_json(files['complete'], {'generated_at': now(), 'subject_id': data.subject_id, 'phase': PHASE, 'policy_sha256': policy.policy_sha256, 'subject_input_sha256': data.input_sha256, 'file_hashes': file_hashes, 'complete': True})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)

def load_checkpoint(folder: Path, data: SubjectData, policy: prespecifiedPolicy) -> dict[str, Any] | None:
    files = checkpoint_files(folder)
    if not files['complete'].is_file():
        return None
    try:
        complete = jread(files['complete'])
        if not bool(complete.get('complete')) or complete.get('phase') != PHASE:
            return None
        if str(complete.get('policy_sha256')) != policy.policy_sha256:
            return None
        if str(complete.get('subject_input_sha256')) != data.input_sha256:
            return None
        for key, expected in complete.get('file_hashes', {}).items():
            p = files[key]
            if not p.is_file() or sha(p) != str(expected):
                return None
        arrays_npz = np.load(files['arrays'], allow_pickle=False)
        arrays = {k: arrays_npz[k] for k in arrays_npz.files}
        metadata = pd.read_csv(files['metadata'], low_memory=False)
        if len(metadata) != data.trial_ids.size:
            return None
        selection = jread(files['selection'])
        return {'metadata': metadata, 'arrays': arrays, 'candidate_metrics': pd.read_csv(files['candidate_metrics'], low_memory=False), 'stability': pd.read_csv(files['stability'], low_memory=False), 'selected_k': None if selection.get('selected_k') is None else int(selection['selected_k']), 'selection_reason': str(selection['selection_reason'])}
    except Exception:
        return None

def compute_or_load_subject(data: SubjectData, policy: prespecifiedPolicy, cache_root: Path, recompute: bool, note) -> dict[str, Any]:
    folder = cache_root / data.subject_id
    if recompute and folder.exists():
        shutil.rmtree(folder)
    cached = load_checkpoint(folder, data, policy)
    if cached is not None:
        note(f'{data.subject_id}: verified checkpoint reused.')
        return cached
    note(f'{data.subject_id}: fitting prespecified PCA/GMM/stability grid.')
    prep = preprocess_subject(data, policy)
    fit = fit_subject_models(data, prep, policy)
    save_checkpoint(folder, data, prep, fit, policy)
    cached = load_checkpoint(folder, data, policy)
    if cached is None:
        raise RuntimeError(f'{data.subject_id}: checkpoint failed verification after write')
    return cached

def inverse_pca_centroids(means_pca: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    scaled = means_pca * np.sqrt(np.asarray(arrays['pca_explained_variance'], dtype=float))[None, :]
    return scaled @ np.asarray(arrays['pca_components'], dtype=float) + np.asarray(arrays['pca_mean'], dtype=float)[None, :]

def vector_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.dot(x, x)))

def continuous_trajectory_rows(subject_id: str, metadata: pd.DataFrame, scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_cols = [f'pc{i + 1}' for i in range(scores.shape[1])]
    frame = metadata.copy()
    for i, col in enumerate(score_cols):
        frame[col] = scores[:, i]
    for trial_id, group in frame.groupby('raw_behavior_row_id', sort=False):
        by_family = {str(row.epoch_family): row for row in group.itertuples(index=False)}
        if set(by_family) != set(FAMILIES):
            raise RuntimeError(f'{subject_id} {trial_id}: incomplete three-epoch trajectory')
        vectors = {family: np.asarray([getattr(by_family[family], c) for c in score_cols], dtype=float) for family in FAMILIES}
        early, late, rating = (vectors[f] for f in FAMILIES)
        d1 = late - early
        d2 = rating - late
        direct = rating - early
        n1, n2, nd = (vector_norm(d1), vector_norm(d2), vector_norm(direct))
        path_length = n1 + n2
        cosine = float(np.dot(d1, d2) / (n1 * n2)) if n1 > 0 and n2 > 0 else np.nan
        directness = float(nd / path_length) if path_length > 0 else np.nan
        row: dict[str, Any] = {'subject_id': subject_id, 'raw_behavior_row_id': str(trial_id), 'trial_number': int(group.trial_number.iloc[0]), 'pca_components': scores.shape[1], 'early_to_late_norm': n1, 'late_to_rating_norm': n2, 'early_to_rating_direct_norm': nd, 'total_path_length': path_length, 'directness_ratio': directness, 'segment_cosine': cosine}
        for family, vec in vectors.items():
            for i, value in enumerate(vec, start=1):
                row[f'{family}_pc{i}'] = float(value)
        for prefix, vec in [('late_minus_early', d1), ('rating_minus_late', d2), ('rating_minus_early', direct)]:
            for i, value in enumerate(vec, start=1):
                row[f'{prefix}_pc{i}'] = float(value)
        rows.append(row)
    return rows

def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def transition_statistics(early: np.ndarray, late: np.ndarray, rating: np.ndarray) -> dict[str, float]:
    mi1 = float(mutual_info_score(early, late))
    mi2 = float(mutual_info_score(late, rating))
    paths = pd.Series([f'{a}>{b}>{c}' for a, b, c in zip(early, late, rating)])
    return {'early_to_late_persistence': float(np.mean(early == late)), 'late_to_rating_persistence': float(np.mean(late == rating)), 'all_three_same_state_rate': float(np.mean((early == late) & (late == rating))), 'transition_mutual_information': float((mi1 + mi2) / 2.0), 'early_to_late_mutual_information': mi1, 'late_to_rating_mutual_information': mi2, 'path_entropy_bits': entropy_from_counts(paths.value_counts().to_numpy()), 'unique_path_count': int(paths.nunique())}

def discrete_outputs(subject_id: str, metadata: pd.DataFrame, labels: np.ndarray, selected_k: int) -> dict[str, Any]:
    assignments = metadata.copy()
    assignments['selected_k'] = selected_k
    assignments['state_1based'] = labels.astype(int)
    assignments['state_label'] = assignments.state_1based.map(lambda x: f'S{x}')
    traj_rows: list[dict[str, Any]] = []
    for trial_id, group in assignments.groupby('raw_behavior_row_id', sort=False):
        state = dict(zip(group.epoch_family.astype(str), group.state_1based.astype(int)))
        e, l, r = (state['stimulus_early'], state['stimulus_late'], state['rating_early'])
        traj_rows.append({'subject_id': subject_id, 'raw_behavior_row_id': str(trial_id), 'trial_number': int(group.trial_number.iloc[0]), 'selected_k': selected_k, 'stimulus_early_state_1based': e, 'stimulus_late_state_1based': l, 'rating_early_state_1based': r, 'early_to_late_transition': f'S{e}>S{l}', 'late_to_rating_transition': f'S{l}>S{r}', 'three_state_path': f'S{e}>S{l}>S{r}', 'early_to_late_persistent': e == l, 'late_to_rating_persistent': l == r, 'all_three_same_state': e == l == r})
    trajectories = pd.DataFrame(traj_rows)
    occupancy_rows = []
    for epoch_family in ['all_epochs'] + FAMILIES:
        x = assignments if epoch_family == 'all_epochs' else assignments[assignments.epoch_family.eq(epoch_family)]
        for state in range(1, selected_k + 1):
            count = int((x.state_1based == state).sum())
            occupancy_rows.append({'subject_id': subject_id, 'selected_k': selected_k, 'epoch_family': epoch_family, 'state_1based': state, 'observations': count, 'proportion': count / len(x), 'unique_trials': int(x.loc[x.state_1based.eq(state), 'raw_behavior_row_id'].nunique())})
    transition_rows = []
    for transition_name, source_family, target_family in [('stimulus_early_to_stimulus_late', 'stimulus_early', 'stimulus_late'), ('stimulus_late_to_rating_early', 'stimulus_late', 'rating_early')]:
        source_col = f'{source_family}_state_1based'
        target_col = f'{target_family}_state_1based'
        for source_state in range(1, selected_k + 1):
            denom = int((trajectories[source_col] == source_state).sum())
            for target_state in range(1, selected_k + 1):
                count = int(((trajectories[source_col] == source_state) & (trajectories[target_col] == target_state)).sum())
                transition_rows.append({'subject_id': subject_id, 'selected_k': selected_k, 'transition': transition_name, 'source_state_1based': source_state, 'target_state_1based': target_state, 'count': count, 'conditional_probability': count / denom if denom else np.nan})
    early = trajectories.stimulus_early_state_1based.to_numpy(int)
    late = trajectories.stimulus_late_state_1based.to_numpy(int)
    rating = trajectories.rating_early_state_1based.to_numpy(int)
    stats = transition_statistics(early, late, rating)
    summary = {'subject_id': subject_id, 'selected_k': selected_k, 'trial_count': len(trajectories), **stats}
    summary['early_to_late_switch_rate'] = 1.0 - summary['early_to_late_persistence']
    summary['late_to_rating_switch_rate'] = 1.0 - summary['late_to_rating_persistence']
    return {'assignments': assignments, 'trajectories': trajectories, 'occupancy': pd.DataFrame(occupancy_rows), 'transitions': pd.DataFrame(transition_rows), 'summary': summary}

def trajectory_nulls(subject_id: str, trajectories: pd.DataFrame, selected_k: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    early = trajectories.stimulus_early_state_1based.to_numpy(int)
    late = trajectories.stimulus_late_state_1based.to_numpy(int)
    rating = trajectories.rating_early_state_1based.to_numpy(int)
    observed = transition_statistics(early, late, rating)
    stat_names = ['early_to_late_persistence', 'late_to_rating_persistence', 'all_three_same_state_rate', 'transition_mutual_information']
    rows = []
    for repeat in range(repeats):
        rng_late = np.random.default_rng(stable_seed(subject_id, selected_k, repeat, stream=10))
        rng_rating = np.random.default_rng(stable_seed(subject_id, selected_k, repeat, stream=11))
        late_perm = late[rng_late.permutation(len(late))]
        rating_perm = rating[rng_rating.permutation(len(rating))]
        stats = transition_statistics(early, late_perm, rating_perm)
        rows.append({'subject_id': subject_id, 'selected_k': selected_k, 'repeat_0based': repeat, **{k: stats[k] for k in stat_names}})
    null_df = pd.DataFrame(rows)
    tests = []
    for stat in stat_names:
        obs = float(observed[stat])
        null_values = null_df[stat].to_numpy(float)
        exceed = int(np.sum(null_values >= obs))
        tests.append({'subject_id': subject_id, 'selected_k': selected_k, 'statistic': stat, 'observed': obs, 'null_mean': float(np.mean(null_values)), 'null_sd': float(np.std(null_values, ddof=1)), 'null_q025': float(np.quantile(null_values, 0.025)), 'null_q975': float(np.quantile(null_values, 0.975)), 'exceedances_greater_or_equal': exceed, 'null_repeats': repeats, 'one_sided_add_one_p': (exceed + 1.0) / (repeats + 1.0)})
    return (null_df, pd.DataFrame(tests))

def checkpoint_inventory(cache_root: Path) -> pd.DataFrame:
    rows = []
    if not cache_root.exists():
        return pd.DataFrame(columns=['relative_path', 'size_bytes', 'sha256'])
    for path in sorted((p for p in cache_root.rglob('*') if p.is_file())):
        rows.append({'relative_path': str(path.relative_to(cache_root)), 'size_bytes': path.stat().st_size, 'sha256': sha(path)})
    return pd.DataFrame(rows)

def primary_file_keys() -> list[str]:
    return ['preprocessing', 'candidate_metrics', 'stability', 'selection', 'pca_scores', 'pca_loadings', 'assignments', 'continuous_trajectories', 'discrete_trajectories', 'occupancy', 'transitions', 'transition_summary', 'null_replicates', 'null_tests', 'centroids_pca', 'centroids_common']

def qc_diagnostics(assignments: pd.DataFrame, qc_store: np.ndarray, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows = []
    assoc_rows = []
    if assignments.empty:
        return (pd.DataFrame(columns=['subject_id', 'selected_k', 'state_1based', 'qc_cell_0based', 'observations', 'mean', 'median', 'iqr']), pd.DataFrame(columns=['subject_id', 'selected_k', 'chi2', 'degrees_of_freedom', 'p_value', 'cramers_v']))
    for subject_id, group in assignments.groupby('subject_id', sort=True):
        k = int(group.selected_k.iloc[0])
        for state, state_group in group.groupby('state_1based', sort=True):
            values = np.asarray(qc_store[state_group.epoch_store_row_0based.to_numpy(int)], dtype=float)
            for cell in range(values.shape[1]):
                x = values[:, cell]
                qc_rows.append({'subject_id': subject_id, 'selected_k': k, 'state_1based': int(state), 'qc_cell_0based': cell, 'observations': len(x), 'mean': float(np.mean(x)), 'median': float(np.median(x)), 'iqr': float(np.quantile(x, 0.75) - np.quantile(x, 0.25))})
        table = pd.crosstab(group.epoch_family, group.state_1based).reindex(index=FAMILIES, columns=range(1, k + 1), fill_value=0)
        try:
            chi2, p, dof, _ = chi2_contingency(table.to_numpy(), correction=False)
            denom = len(group) * min(table.shape[0] - 1, table.shape[1] - 1)
            cramers_v = math.sqrt(chi2 / denom) if denom > 0 else np.nan
        except ValueError:
            chi2, p, dof, cramers_v = (np.nan, np.nan, np.nan, np.nan)
        assoc_rows.append({'subject_id': subject_id, 'selected_k': k, 'chi2': chi2, 'degrees_of_freedom': dof, 'p_value': p, 'cramers_v': cramers_v, 'diagnostic_only_after_primary_lock': True})
    return (pd.DataFrame(qc_rows), pd.DataFrame(assoc_rows))

def write_report(path: Path, summary: dict[str, Any], selection: pd.DataFrame, null_tests: pd.DataFrame) -> None:
    stable = selection[selection.stable_discrete_solution.map(bval)]
    lines = ['# Phase latent_state_fit — outcome-free within-person latent-state fit', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f"- prespecified design SHA-256: `{summary['design_policy_sha256']}`", '- Reports/outcomes/context read: **no**', '- Validation/confirmation opened: **no**', '', '## Execution completeness', '', f"- Eligible participants fitted: **{summary['counts']['eligible_participants_fitted']}**", f"- Candidate full fits: **{summary['counts']['candidate_full_fits']}**", f"- Stability resamples: **{summary['counts']['stability_resamples']}**", f"- Continuous trial trajectories: **{summary['counts']['continuous_trajectories']}**", f"- Participants with stable discrete states: **{summary['counts']['stable_discrete_participants']}**", f"- Discrete trial trajectories: **{summary['counts']['discrete_trajectories']}**", f"- Trajectory-null replicates: **{summary['counts']['trajectory_null_replicates']}**", '', '## prespecified K decisions', '', '| Subject | Trials | Stable discrete solution | Selected K | Reason |', '|---|---:|---|---:|---|']
    for row in selection.itertuples(index=False):
        selected_k = '' if pd.isna(row.selected_k) else str(int(row.selected_k))
        lines.append(f'| `{row.subject_id}` | {int(row.trial_count)} | {bool(row.stable_discrete_solution)} | {selected_k} | `{row.selection_reason}` |')
    lines += ['', '## Outcome-free trajectory nulls', '']
    if null_tests.empty:
        lines.append('No participant had a stable discrete-state solution, so no discrete transition null was applicable.')
    else:
        sig = null_tests[null_tests.one_sided_add_one_p < 0.05]
        lines += [f'- prespecified participant × statistic tests: **{len(null_tests)}**', f'- Nominal one-sided p < 0.05 tests: **{len(sig)}**', '- These tests describe outcome-free temporal structure only; no report or context interpretation is made here.']
    lines += ['', '## Phase boundary', '', '- Primary assignments and trajectories were written and SHA-256 fixed before QC diagnostics.', '- QC and residual epoch association are descriptive only and cannot revise K or assignments.', '- Cross-person alignment has not been performed.', '- Report/context mapping has not been performed.', '- Validation and confirmation remain closed.', '', 'A stable latent state is a reproducible partition of this representation, not ground truth for subjective experience.']
    path.write_text(''.join(lines) + '', encoding='utf-8')

def synthetic_subject(subject_id: str, n_trials: int=36, n_features: int=40) -> SubjectData:
    rng = np.random.default_rng(123)
    trial_ids, trial_numbers, rows, families, raw = ([], [], [], [], [])
    row = 0
    for trial in range(n_trials):
        latent = -3.0 if trial < n_trials // 2 else 3.0
        for fam_idx, family in enumerate(FAMILIES):
            x = rng.normal(scale=0.35, size=n_features)
            x[:8] += latent
            x[8:12] += 0.25 * fam_idx * latent
            trial_ids.append(f'{subject_id}|trial{trial + 1}')
            trial_numbers.append(trial + 1)
            rows.append(row)
            families.append(family)
            raw.append(x.astype(np.float32))
            row += 1
    trial_ids_arr = np.asarray(trial_ids, dtype=str)
    rows_arr = np.asarray(rows, dtype=np.int32)
    retained = np.arange(n_features, dtype=int)
    raw_arr = np.vstack(raw).astype(np.float32)
    return SubjectData(subject_id=subject_id, trial_ids=trial_ids_arr, trial_numbers=np.asarray(trial_numbers, dtype=np.int32), epoch_rows=rows_arr, epoch_families=np.asarray(families, dtype=str), compact=raw_arr, retained_features=retained, input_sha256=subject_input_sha(subject_id, trial_ids_arr, rows_arr, retained, raw_arr))

def self_test() -> None:
    policy = prespecifiedPolicy(policy_sha256='self-test', candidate_k=[2, 3], pca_components=8, iqr_floor=1e-08, winsor_limit=6.0, covariance_type='diag', reg_covar=0.0001, n_init=5, max_iter=500, stability_repeats=5, trial_fraction=0.8, median_ari_min=0.6, p10_ari_min=0.3, collapse_rate_max=0.2, min_state_observations=8, min_state_unique_trials=4, bic_delta=10.0, null_repeats=20)
    data = synthetic_subject('sub-test')
    prep = preprocess_subject(data, policy)
    fit = fit_subject_models(data, prep, policy, repeats_override=5)
    assert prep['pca_scores'].shape == (108, 8)
    assert len(fit['candidate_metrics']) == 2
    assert len(fit['stability']) == 10
    assert fit['selected_k'] == 2
    metadata = pd.DataFrame({'subject_id': data.subject_id, 'observation_row_0based': np.arange(len(data.trial_ids)), 'raw_behavior_row_id': data.trial_ids, 'trial_number': data.trial_numbers, 'epoch_family': data.epoch_families, 'epoch_store_row_0based': data.epoch_rows})
    labels = fit['labels_by_k'][policy.candidate_k.index(2)]
    disc = discrete_outputs(data.subject_id, metadata, labels, 2)
    assert len(disc['trajectories']) == 36
    null, tests = trajectory_nulls(data.subject_id, disc['trajectories'], 2, 20)
    assert len(null) == 20 and len(tests) == 4
    cont = continuous_trajectory_rows(data.subject_id, metadata, prep['pca_scores'])
    assert len(cont) == 36
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / data.subject_id
        save_checkpoint(folder, data, prep, fit, policy)
        loaded = load_checkpoint(folder, data, policy)
        assert loaded is not None and loaded['selected_k'] == 2
    print('latent_state_fit self-test: PASS')

def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    started = time.time()
    project = args.project.resolve()
    paths = {key: project / rel for key, rel in REL.items()}
    out = output_paths(project)
    required(paths.values())
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    cache_root = project / CACHE_REL
    cache_root.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"Output already exists; review or use --force: {out['contract']}")
    log_lines: list[str] = []

    def note(message: str) -> None:
        line = f'[{now()}] {message}'
        log_lines.append(line)
        print(line, flush=True)
        out['log'].write_text(''.join(log_lines) + '', encoding='utf-8')
    note('Verifying latent_state_design specification, contract hashes, and authorization.')
    latent_state_design_summary, latent_state_design_contract, latent_policy, policy = load_and_verify_specification(project, paths)
    note('Rehashing prespecified outcome-free epoch-store inputs.')
    epoch_rehash = verify_epoch_store(paths, latent_state_design_contract)
    epoch_rehash.to_csv(out['epoch_store_rehash'], index=False)
    note('Loading outcome-free manifests and compact array only; QC remains unopened until primary lock.')
    trial = pd.read_csv(paths['trial_manifest'], low_memory=False)
    eligibility = pd.read_csv(paths['subject_eligibility'], low_memory=False)
    audit = pd.read_csv(paths['participant_feature_audit'], low_memory=False)
    schema = pd.read_csv(paths['compact_feature_schema'], low_memory=False)
    epoch_manifest = pd.read_csv(paths['epoch_manifest'], usecols=EPOCH_MANIFEST_USECOLS, low_memory=False)
    compact = np.load(paths['epoch_compact'], mmap_mode='r', allow_pickle=False)
    eligible_subjects = validate_inputs(trial, eligibility, audit, schema, epoch_manifest, compact)
    all_candidate, all_stability, all_selection, preprocessing_rows = ([], [], [], [])
    all_pca_scores, all_pca_loadings, all_continuous = ([], [], [])
    all_assignments, all_discrete, all_occupancy, all_transitions, all_transition_summary = ([], [], [], [], [])
    all_null_replicates, all_null_tests, all_centroids_pca, all_centroids_common = ([], [], [], [])
    subject_results: dict[str, dict[str, Any]] = {}
    for subject_id in eligible_subjects:
        data = build_subject_data(subject_id, trial, audit, compact)
        result = compute_or_load_subject(data, policy, cache_root, args.recompute_cache, note)
        subject_results[subject_id] = {'data': data, 'result': result}
        metadata = result['metadata'].copy()
        arrays = result['arrays']
        scores = np.asarray(arrays['pca_scores'], dtype=float)
        selected_k = result['selected_k']
        stable_solution = selected_k is not None
        candidate = result['candidate_metrics'].copy()
        candidate['selected_k'] = selected_k
        candidate['selected_candidate'] = candidate.candidate_k.eq(selected_k) if selected_k is not None else False
        all_candidate.append(candidate)
        all_stability.append(result['stability'].copy())
        all_selection.append({'subject_id': subject_id, 'trial_count': int(np.unique(data.trial_ids).size), 'epoch_observations': len(data.trial_ids), 'retained_features_after_finite_rule': len(data.retained_features), 'retained_features_after_pooled_iqr_rule': len(arrays['retained_features']), 'pca_components': scores.shape[1], 'stable_discrete_solution': stable_solution, 'selected_k': selected_k, 'selection_reason': result['selection_reason']})
        all_pre = {'subject_id': subject_id, 'trial_count': int(np.unique(data.trial_ids).size), 'epoch_observations': len(data.trial_ids), 'initial_retained_compact_cells': len(data.retained_features), 'final_retained_compact_cells': len(arrays['retained_features']), 'dropped_by_pooled_iqr_rule': len(data.retained_features) - len(arrays['retained_features']), 'pca_components': scores.shape[1], 'pca_explained_variance_ratio_sum': float(np.sum(arrays['pca_explained_variance_ratio'])), 'subject_input_sha256': data.input_sha256, 'checkpoint_reused_or_verified': True}
        all_pca_scores.append(metadata.assign(**{f'pc{i + 1}': scores[:, i] for i in range(scores.shape[1])}))
        for pc in range(scores.shape[1]):
            for local_idx, compact_cell in enumerate(arrays['retained_features'].astype(int)):
                all_pca_loadings.append({'subject_id': subject_id, 'pc_1based': pc + 1, 'compact_cell_0based': compact_cell, 'feature_name': str(schema.loc[schema.compact_cell_0based.eq(compact_cell), 'feature_name'].iloc[0]), 'pca_component_loading': float(arrays['pca_components'][pc, local_idx]), 'explained_variance': float(arrays['pca_explained_variance'][pc]), 'explained_variance_ratio': float(arrays['pca_explained_variance_ratio'][pc])})
        all_continuous.extend(continuous_trajectory_rows(subject_id, metadata, scores))
        preprocessing_rows.append(all_pre)
        if stable_solution:
            k_idx = policy.candidate_k.index(int(selected_k))
            labels = np.asarray(arrays['labels_by_k'][k_idx], dtype=np.int16)
            disc = discrete_outputs(subject_id, metadata, labels, int(selected_k))
            all_assignments.append(disc['assignments'])
            all_discrete.append(disc['trajectories'])
            all_occupancy.append(disc['occupancy'])
            all_transitions.append(disc['transitions'])
            all_transition_summary.append(disc['summary'])
            null_rep, null_test = trajectory_nulls(subject_id, disc['trajectories'], int(selected_k), policy.null_repeats)
            all_null_replicates.append(null_rep)
            all_null_tests.append(null_test)
            means_pca = np.asarray(arrays[f'k{selected_k}_means_pca'], dtype=float)
            means_common_local = inverse_pca_centroids(means_pca, arrays)
            occupancy_all = disc['occupancy'][disc['occupancy'].epoch_family.eq('all_epochs')].set_index('state_1based')
            for state_idx in range(int(selected_k)):
                state = state_idx + 1
                for pc in range(means_pca.shape[1]):
                    all_centroids_pca.append({'subject_id': subject_id, 'selected_k': int(selected_k), 'state_1based': state, 'pc_1based': pc + 1, 'centroid_value': float(means_pca[state_idx, pc]), 'state_weight': float(arrays[f'k{selected_k}_weights'][state_idx])})
                local_lookup = {int(cell): float(means_common_local[state_idx, j]) for j, cell in enumerate(arrays['retained_features'].astype(int))}
                for compact_cell in range(PER_EPOCH_COMPACT):
                    all_centroids_common.append({'subject_id': subject_id, 'selected_k': int(selected_k), 'state_1based': state, 'compact_cell_0based': compact_cell, 'feature_name': str(schema.loc[schema.compact_cell_0based.eq(compact_cell), 'feature_name'].iloc[0]), 'within_person_stage_standardized_centroid': local_lookup.get(compact_cell, np.nan), 'feature_retained': compact_cell in local_lookup, 'state_observations': int(occupancy_all.loc[state, 'observations']), 'state_proportion': float(occupancy_all.loc[state, 'proportion'])})
    preprocessing_df = pd.DataFrame(preprocessing_rows)
    candidate_df = pd.concat(all_candidate, ignore_index=True)
    stability_df = pd.concat(all_stability, ignore_index=True)
    selection_df = pd.DataFrame(all_selection)
    pca_scores_df = pd.concat(all_pca_scores, ignore_index=True)
    pca_loadings_df = pd.DataFrame(all_pca_loadings)
    continuous_df = pd.DataFrame(all_continuous)
    assignments_df = pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame(columns=['subject_id', 'observation_row_0based', 'raw_behavior_row_id', 'trial_number', 'epoch_family', 'epoch_store_row_0based', 'selected_k', 'state_1based', 'state_label'])
    discrete_df = pd.concat(all_discrete, ignore_index=True) if all_discrete else pd.DataFrame(columns=['subject_id', 'raw_behavior_row_id', 'trial_number', 'selected_k', 'stimulus_early_state_1based', 'stimulus_late_state_1based', 'rating_early_state_1based', 'early_to_late_transition', 'late_to_rating_transition', 'three_state_path', 'early_to_late_persistent', 'late_to_rating_persistent', 'all_three_same_state'])
    occupancy_df = pd.concat(all_occupancy, ignore_index=True) if all_occupancy else pd.DataFrame(columns=['subject_id', 'selected_k', 'epoch_family', 'state_1based', 'observations', 'proportion', 'unique_trials'])
    transitions_df = pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame(columns=['subject_id', 'selected_k', 'transition', 'source_state_1based', 'target_state_1based', 'count', 'conditional_probability'])
    transition_summary_df = pd.DataFrame(all_transition_summary, columns=['subject_id', 'selected_k', 'trial_count', 'early_to_late_persistence', 'late_to_rating_persistence', 'all_three_same_state_rate', 'transition_mutual_information', 'early_to_late_mutual_information', 'late_to_rating_mutual_information', 'path_entropy_bits', 'unique_path_count', 'early_to_late_switch_rate', 'late_to_rating_switch_rate'])
    null_replicates_df = pd.concat(all_null_replicates, ignore_index=True) if all_null_replicates else pd.DataFrame(columns=['subject_id', 'selected_k', 'repeat_0based', 'early_to_late_persistence', 'late_to_rating_persistence', 'all_three_same_state_rate', 'transition_mutual_information'])
    null_tests_df = pd.concat(all_null_tests, ignore_index=True) if all_null_tests else pd.DataFrame(columns=['subject_id', 'selected_k', 'statistic', 'observed', 'null_mean', 'null_sd', 'null_q025', 'null_q975', 'exceedances_greater_or_equal', 'null_repeats', 'one_sided_add_one_p'])
    centroids_pca_df = pd.DataFrame(all_centroids_pca, columns=['subject_id', 'selected_k', 'state_1based', 'pc_1based', 'centroid_value', 'state_weight'])
    centroids_common_df = pd.DataFrame(all_centroids_common, columns=['subject_id', 'selected_k', 'state_1based', 'compact_cell_0based', 'feature_name', 'within_person_stage_standardized_centroid', 'feature_retained', 'state_observations', 'state_proportion'])
    note('Writing all primary outcome-free state and trajectory outputs.')
    primary_frames = {'preprocessing': preprocessing_df, 'candidate_metrics': candidate_df, 'stability': stability_df, 'selection': selection_df, 'pca_scores': pca_scores_df, 'pca_loadings': pca_loadings_df, 'assignments': assignments_df, 'continuous_trajectories': continuous_df, 'discrete_trajectories': discrete_df, 'occupancy': occupancy_df, 'transitions': transitions_df, 'transition_summary': transition_summary_df, 'null_replicates': null_replicates_df, 'null_tests': null_tests_df, 'centroids_pca': centroids_pca_df, 'centroids_common': centroids_common_df}
    for key, df in primary_frames.items():
        df.to_csv(out[key], index=False)
    note('SHA-256 locking primary assignments and trajectories before opening QC diagnostics.')
    lock_hashes = {key: sha(out[key]) for key in primary_file_keys()}
    primary_lock = {'generated_at': now(), 'phase': PHASE, 'design_policy_sha256': policy.policy_sha256, 'primary_outputs_fixed_before_qc_diagnostics': True, 'context_or_reports_read_before_lock': False, 'qc_values_loaded_or_interpreted_before_lock': False, 'qc_file_integrity_hash_verified_before_lock': True, 'hashes': lock_hashes}
    atom_json(out['primary_lock'], primary_lock)
    note('Primary lock complete. Loading prespecified QC array for descriptive post-lock diagnostics only.')
    qc_store = np.load(paths['epoch_qc'], mmap_mode='r', allow_pickle=False)
    if qc_store.shape != (3853, PER_EPOCH_QC) or not np.isfinite(np.asarray(qc_store)).all():
        raise RuntimeError('prespecified QC array failed post-lock integrity/finite check')
    qc_desc_df, epoch_assoc_df = qc_diagnostics(assignments_df, qc_store, selection_df)
    qc_desc_df.to_csv(out['qc_state_descriptives'], index=False)
    epoch_assoc_df.to_csv(out['epoch_state_association'], index=False)
    note('Hashing verified participant checkpoints.')
    cache_inv = checkpoint_inventory(cache_root)
    cache_inv.to_csv(out['checkpoint_inventory'], index=False)
    stable_count = int(selection_df.stable_discrete_solution.map(bval).sum())
    selected_k_counts = {str(int(k)): int(v) for k, v in selection_df.dropna(subset=['selected_k']).selected_k.astype(int).value_counts().sort_index().items()}
    decision = {'all_eligible_participants_fitted': len(selection_df) == len(eligible_subjects), 'all_candidate_k_fitted_or_failure_recorded': len(candidate_df) == len(eligible_subjects) * len(policy.candidate_k), 'all_stability_resamples_executed_or_failure_recorded': len(stability_df) == len(eligible_subjects) * len(policy.candidate_k) * policy.stability_repeats, 'continuous_trajectories_written_for_all_eligible_trials': len(continuous_df) == 1237, 'discrete_states_forced_when_no_eligible_k': False, 'primary_outputs_fixed_before_qc_diagnostics': True, 'qc_used_in_primary_fit_or_k_selection': False, 'outcome_report_or_context_columns_read': False, 'cross_person_alignment_performed': False, 'posthoc_report_mapping_performed': False, 'validation_opened': False, 'confirmation_opened': False}
    if not all((decision[k] for k in ['all_eligible_participants_fitted', 'all_candidate_k_fitted_or_failure_recorded', 'all_stability_resamples_executed_or_failure_recorded', 'continuous_trajectories_written_for_all_eligible_trials', 'primary_outputs_fixed_before_qc_diagnostics'])):
        raise RuntimeError(f'latent_state_fit completeness failure: {decision}')
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'prespecified_outcome_free_within_person_latent_states_and_three_epoch_trajectories_complete_candidate', 'design_policy_sha256': policy.policy_sha256, 'counts': {'eligible_participants_fitted': len(selection_df), 'eligible_trials': len(continuous_df), 'eligible_epoch_observations': len(pca_scores_df), 'candidate_full_fits': len(candidate_df), 'stability_resamples': len(stability_df), 'stable_discrete_participants': stable_count, 'continuous_only_participants': len(selection_df) - stable_count, 'continuous_trajectories': len(continuous_df), 'discrete_state_assignments': len(assignments_df), 'discrete_trajectories': len(discrete_df), 'trajectory_null_replicates': len(null_replicates_df), 'trajectory_null_tests': len(null_tests_df), 'checkpoint_files': len(cache_inv), 'checkpoint_bytes': int(cache_inv.size_bytes.sum()) if not cache_inv.empty else 0}, 'selected_k_counts': selected_k_counts, 'selection': selection_df.to_dict(orient='records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
    atom_json(out['summary'], summary)
    write_report(out['report'], summary, selection_df, null_tests_df)
    output_hashes = {}
    for key, path in out.items():
        if key not in {'contract', 'next_policy', 'report', 'log'} and path.is_file():
            output_hashes[key] = sha(path)
    contract = {**summary, 'prespecified_design_policy': latent_policy, 'input_hashes': {key: sha(path) for key, path in paths.items()}, 'latent_state_design_contract_sha256': sha(paths['latent_state_design_contract']), 'latent_state_design_next_policy_sha256': sha(paths['latent_state_design_next']), 'primary_lock_sha256': sha(out['primary_lock']), 'primary_output_hashes_before_diagnostics': lock_hashes, 'output_hashes': output_hashes, 'cache_root': str(cache_root), 'large_cache_embedded_in_release': False, 'cache_reuse_policy': 'subject checkpoint SHA-256 and subject-input SHA-256 must both verify before reuse', 'outcomes_modified': False, 'feature_store_modified': False, 'secondary_models_cache_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'specification an outcome-free cross-person alignment design using only latent_state_fit-fixed participant state centroids and continuous trajectory geometry', 'required_inputs': {'latent_state_fit_contract': str(out['contract']), 'primary_lock': str(out['primary_lock']), 'participant_state_selection': str(out['selection']), 'state_centroids_common_axes': str(out['centroids_common']), 'state_centroids_pca': str(out['centroids_pca']), 'continuous_trajectories': str(out['continuous_trajectories']), 'state_assignments': str(out['assignments']), 'prespecified_latent_state_design_latent_policy': str(paths['latent_state_design_latent_policy'])}, 'required_design_boundary': ['cross-person alignment remains outcome-free', 'no valence, liking, arousal, emotion report, context, or report-derived factor may be read', 'stable discrete centroids and continuous-only participant trajectories must both be represented without forcing missing states', 'alignment choices must be prespecified before report/context mapping'], 'forbidden': latent_policy['forbidden'], 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['next_policy'], next_policy)
    note('latent_state_fit complete. Primary state/trajectory outputs are fixed; no report/context mapping or cross-person alignment was performed.')
    print('' + '=' * 100)
    print('PHASE latent_state_fit OUTCOME-FREE LATENT-STATE FIT COMPLETED')
    print('=' * 100)
    print(f'Design-policy SHA-256: {policy.policy_sha256}')
    print(f'Eligible participants fitted: {len(selection_df)}')
    print(f'Stable discrete participants: {stable_count}')
    print(f'Continuous trajectories: {len(continuous_df)}')
    print(f'Discrete trajectories: {len(discrete_df)}')
    print('Reports/outcomes/context read: False')
    print('Cross-person alignment performed: False')
    print('Validation opened: False')
    print('Confirmation opened: False')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    raise SystemExit(main())