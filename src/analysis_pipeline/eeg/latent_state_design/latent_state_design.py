# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase latent_state_design: specification an outcome-free within-person latent-state design.

This phase does not fit latent-state models and does not inspect valence, liking,
rehashes the outcome-free epoch store, specifications the exact common-trial cohort and
compact feature schema, and writes a complete execution policy for latent_state_fit.

Validation and confirmation remain closed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'latent_state_design'
NEXT_PHASE = 'latent_state_fit'
SEED = 540
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
PER_EPOCH_COMPACT = 169
PER_EPOCH_QC = 65
MIN_COMMON_TRIALS = 30
MIN_RETAINED_FEATURES = 30
FINITE_FRACTION_PER_FAMILY = 0.9
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
PCA_COMPONENTS = 8
STATE_COUNTS = [2, 3, 4, 5]
GMM_COVARIANCE = 'diag'
GMM_REG_COVAR = 0.0001
GMM_N_INIT = 20
GMM_MAX_ITER = 1000
STABILITY_REPEATS = 30
STABILITY_TRIAL_FRACTION = 0.8
STABILITY_MEDIAN_ARI = 0.6
STABILITY_P10_ARI = 0.3
STABILITY_MAX_COLLAPSE_RATE = 0.1
MIN_STATE_OBSERVATIONS = 8
MIN_STATE_UNIQUE_TRIALS = 4
BIC_PARSIMONY_DELTA = 10.0
TRAJECTORY_NULL_REPEATS = 1000
REL = {'secondary_models_summary': Path('outputs/qc/secondary_models_secondary_models_summary.json'), 'secondary_models_contract': Path('config/secondary_models_secondary_models_contract.json'), 'secondary_models_next': Path('analysis_specifications/S059_secondary_models_latent_state.json'), 'secondary_models_reg_predictions': Path('outputs/tables/secondary_models_secondary_regression_outer_predictions.csv'), 'secondary_models_class_metrics': Path('outputs/tables/secondary_models_diagnostic_classification_metrics.csv'), 'primary_analysis_reference_analysis_rows': Path('outputs/tables/outcome_model_integrity_analysis_set_row_manifest.csv'), 'primary_analysis_reference_subject_eligibility': Path('outputs/qc/outcome_model_integrity_analysis_set_subject_eligibility.csv'), 'epoch_store_summary': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_store_summary.json'), 'epoch_manifest': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_manifest.csv'), 'epoch_compact': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy'), 'epoch_qc': Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_qc_float32.npy')}
secondary_models_OUTPUT_MAP = {'feature_rehash': Path('outputs/qc/secondary_models_feature_store_rehash.csv'), 'primary_models_cache_rehash': Path('outputs/qc/secondary_models_primary_models_cache_rehash.csv'), 'reg_predictions': Path('outputs/tables/secondary_models_secondary_regression_outer_predictions.csv'), 'reg_selected': Path('outputs/tables/secondary_models_secondary_regression_selected_hyperparameters.csv'), 'reg_inner': Path('outputs/tables/secondary_models_secondary_regression_inner_scores.csv'), 'reg_metrics': Path('outputs/tables/secondary_models_secondary_regression_metrics.csv'), 'reg_contrasts': Path('outputs/tables/secondary_models_secondary_regression_contrasts.csv'), 'p2_subject_metrics': Path('outputs/tables/secondary_models_person_specific_subject_metrics.csv'), 'p2_sign_tests': Path('outputs/tables/secondary_models_person_specific_sign_tests.csv'), 'u1_bootstrap_replicates': Path('outputs/tables/secondary_models_unseen_person_bootstrap_replicates.csv'), 'u1_bootstrap_intervals': Path('outputs/tables/secondary_models_unseen_person_bootstrap_intervals.csv'), 'class_predictions': Path('outputs/tables/secondary_models_diagnostic_classification_predictions.csv'), 'class_selected': Path('outputs/tables/secondary_models_diagnostic_classification_selected_hyperparameters.csv'), 'class_inner': Path('outputs/tables/secondary_models_diagnostic_classification_inner_scores.csv'), 'class_metrics': Path('outputs/tables/secondary_models_diagnostic_classification_metrics.csv'), 'cache_inventory': Path('outputs/tables/secondary_models_cache_inventory.csv'), 'summary': Path('outputs/qc/secondary_models_secondary_models_summary.json')}
COMPACT_BLOCKS = [('time_frequency_channel_median_iqr', 0, 60, 'channel-median and channel-IQR summaries of time-frequency cells'), ('morphology_channel_median_iqr', 60, 72, 'channel-median and channel-IQR summaries of morphology cells'), ('plv_channel_median_iqr', 72, 84, 'channel-median and channel-IQR summaries of channel PLV cells'), ('waveform_dct_channel_median_iqr', 84, 124, 'channel-median and channel-IQR summaries of waveform DCT cells'), ('covariance_summary', 124, 145, 'direct covariance-summary cells'), ('plv_global', 145, 169, 'direct global-PLV cells')]
DISCOVERY_POLICY = {'phase': PHASE, 'seed': SEED, 'scientific_role': 'specification outcome-free within-person state discovery and three-epoch trajectory analysis before any latent-state fit', 'discovery_only': True, 'validation_opened': False, 'confirmation_opened': False, 'observation_unit': {'state_observation': 'one epoch observation within one participant and one trial', 'trajectory_unit': 'one common trial with stimulus_early, stimulus_late, and rating_early', 'epoch_order': FAMILIES, 'participant_modeling': 'one independent latent-state model per eligible discovery participant'}, 'cohort': {'source_analysis_set': 'all_three_epoch_families', 'minimum_common_trials': MIN_COMMON_TRIALS, 'must_be_previously_within_person_eligible': True, 'must_have_exactly_one_row_per_trial_per_epoch_family': True, 'outcome_columns_read_by_latent_state_design_or_latent_state_fit': []}, 'primary_representation': {'source': 'secondary_models outcome-free compact epoch representation', 'cells_per_epoch': PER_EPOCH_COMPACT, 'qc_cells_in_primary_state_fit': False, 'feature_inclusion': f'finite fraction >= {FINITE_FRACTION_PER_FAMILY:.2f} separately in every epoch family for that participant', 'minimum_retained_features': MIN_RETAINED_FEATURES, 'preprocessing_order': ['within participant and epoch family, median-impute retained features', 'within participant and epoch family, subtract feature median', 'divide by epoch-family IQR; if an epoch-family IQR is <= floor, use participant-pooled IQR for that feature', 'drop a feature if pooled IQR is also <= floor', f'winsorize standardized values to [-{WINSOR_LIMIT:g}, {WINSOR_LIMIT:g}]', 'fit PCA within participant after stacking all three epoch families'], 'iqr_floor': IQR_FLOOR, 'winsor_limit': WINSOR_LIMIT, 'pca': {'solver': 'full', 'whiten': True, 'fixed_components': PCA_COMPONENTS, 'actual_components': 'minimum of fixed_components, retained_features, and observations_minus_one', 'minimum_components_required': 2}, 'rationale': 'epoch-family-specific robust centering prevents trivial stage location/scale differences from defining states while retaining within-stage trial variation and cross-stage trajectories'}, 'state_model': {'model': 'GaussianMixture', 'covariance_type': GMM_COVARIANCE, 'reg_covar': GMM_REG_COVAR, 'n_init': GMM_N_INIT, 'max_iter': GMM_MAX_ITER, 'candidate_state_counts': STATE_COUNTS, 'state_count_never_selected_from_reports_or_context': True, 'canonical_label_rule': 'sort full-fit state centroids lexicographically by PCA coordinate 1, then 2, etc.; never by report association'}, 'state_count_and_stability': {'resampling_unit': 'trial; all three epoch observations remain coupled in every resample', 'repeats': STABILITY_REPEATS, 'trial_fraction_without_replacement': STABILITY_TRIAL_FRACTION, 'comparison': 'adjusted Rand index between full-fit labels and resample-fit labels on retained observations', 'collapse_definition': f'any state has fewer than {MIN_STATE_OBSERVATIONS} observations or fewer than {MIN_STATE_UNIQUE_TRIALS} unique trials', 'eligible_k_requirements': {'full_fit_converged': True, 'full_fit_min_state_observations': MIN_STATE_OBSERVATIONS, 'full_fit_min_state_unique_trials': MIN_STATE_UNIQUE_TRIALS, 'median_ari_at_least': STABILITY_MEDIAN_ARI, 'ari_10th_percentile_at_least': STABILITY_P10_ARI, 'collapse_rate_at_most': STABILITY_MAX_COLLAPSE_RATE}, 'selection_rule': f'among eligible K, choose the smallest K with BIC <= minimum eligible BIC + {BIC_PARSIMONY_DELTA:g}', 'no_eligible_k_rule': 'declare no stable discrete-state solution for that participant; retain continuous PCA trajectories and do not force clusters'}, 'trajectory_outputs': {'discrete_when_available': ['state at stimulus_early, stimulus_late, and rating_early', 'stimulus_early_to_late transition', 'stimulus_late_to_rating transition', 'three-state path identifier', 'per-epoch occupancy, transition matrices, persistence, switching, and path entropy'], 'continuous_for_every_eligible_participant': ['PCA score vectors at each epoch', 'late-minus-early displacement', 'rating-minus-late displacement', 'rating-minus-early displacement', 'segment norms, total path length, direct displacement, directness ratio, and segment cosine when defined'], 'null': {'repeats': TRAJECTORY_NULL_REPEATS, 'scheme': 'within participant, independently permute trial linkage for stimulus_late and rating_early while preserving each epoch occupancy and keeping stimulus_early fixed', 'statistics': ['early_to_late_persistence', 'late_to_rating_persistence', 'all_three_same_state_rate', 'transition_mutual_information'], 'p_value': 'one-sided add-one randomization p-value for excess structured persistence/information'}}, 'diagnostic_only_after_state_lock': {'qc_association': 'summarize QC by prespecified state after assignments are written; QC never enters primary state fit or K selection', 'epoch_association': 'summarize residual epoch-family association after stage-centering; never revise states', 'context_and_reports': 'forbidden in latent_state_fit; deferred to a separate post-hoc mapping phase after state assignments and trajectories are hashed'}, 'cross_person_alignment': {'performed_in_latent_state_fit': False, 'deferred_role': 'later outcome-free layer using prespecified participant state centroids in the common 169-cell compact feature axes', 'global_preprocessing_before_within_person_discovery': False}, 'claim_guardrails': ['a latent state is a reproducible partition of this representation, not ground truth for subjective experience', 'failure to obtain stable discrete states does not imply absent neurophysiological structure', 'context or participant separability is not emotion decoding', 'no best epoch is selected from prior outer-test results', 'state discovery and post-hoc report mapping remain separate phases'], 'forbidden': ['reading valence, liking, arousal, emotion reports, labels, or report-derived factors during latent_state_design or latent_state_fit', 'choosing K, PCA dimensionality, features, participant inclusion, or state labels from report/context association', 'using secondary_models outer-test performance to select an epoch family', 'including QC cells in the primary state representation', 'global outcome-informed or context-informed alignment', 'opening validation or confirmation participants']}
POLICY_SHA = hashlib.sha256(json.dumps(DISCOVERY_POLICY, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
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
    return hashlib.sha256(payload.encode()).hexdigest()

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
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '', encoding='utf-8')
    os.replace(tmp, path)

def output_paths(project: Path) -> dict[str, Path]:
    return {'log': project / 'outputs/logs/latent_state_design_latent_state_design_specification.log', 'epoch_store_rehash': project / 'outputs/qc/latent_state_design_epoch_store_rehash.csv', 'subject_eligibility': project / 'outputs/qc/latent_state_design_subject_eligibility.csv', 'summary': project / 'outputs/qc/latent_state_design_latent_state_design_specification_summary.json', 'report': project / 'outputs/reports/latent_state_design_latent_state_design_specification_report.md', 'trial_manifest': project / 'outputs/tables/latent_state_design_common_trial_manifest.csv', 'feature_schema': project / 'outputs/tables/latent_state_design_compact_feature_schema.csv', 'participant_feature_audit': project / 'outputs/tables/latent_state_design_participant_feature_audit.csv', 'latent_policy': project / 'analysis_specifications/S037_latent_state.json', 'trajectory_policy': project / 'analysis_specifications/S038_latent_state_trajectory.json', 'contract': project / 'config/latent_state_design_latent_state_design_contract.json', 'next_policy': project / 'analysis_specifications/S036_latent_state_fit.json'}

def verify_secondary_models(project: Path, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = jread(paths['secondary_models_summary'])
    contract = jread(paths['secondary_models_contract'])
    nxt = jread(paths['secondary_models_next'])
    if summary.get('phase') != 'secondary_models' or contract.get('phase') != 'secondary_models':
        raise RuntimeError('secondary_models phase mismatch')
    decision = summary.get('decision', {})
    required_true = ['all_mandatory_secondary_models_executed', 'all_outer_predictions_complete', 'all_transforms_training_only', 'person_specific_support_decided', 'unseen_person_support_decided', 'identity_probe_complete', 'context_probe_complete', 'conditional_liking_model_complete']
    bad = [k for k in required_true if not bool(decision.get(k))]
    if bad:
        raise RuntimeError(f'secondary_models incomplete decision flags: {bad}')
    if bool(decision.get('outer_test_family_or_epoch_selection_performed')):
        raise RuntimeError('secondary_models reports forbidden outer-test family/epoch selection')
    if bool(decision.get('latent_state_models_fitted')):
        raise RuntimeError('Latent-state models were unexpectedly fitted before latent_state_design')
    if bool(decision.get('validation_opened')) or bool(decision.get('confirmation_opened')):
        raise RuntimeError('Validation or confirmation was opened')
    if not bool(nxt.get('next_phase_permitted')) or nxt.get('next_phase') != PHASE:
        raise RuntimeError('secondary_models did not authorize latent_state_design')
    if bool(nxt.get('validation_opened')) or bool(nxt.get('confirmation_opened')):
        raise RuntimeError('secondary_models next policy opened validation/confirmation')
    output_hashes = contract.get('output_hashes', {})
    for key, rel in secondary_models_OUTPUT_MAP.items():
        p = project / rel
        if key not in output_hashes:
            raise RuntimeError(f'secondary_models contract missing output hash: {key}')
        if not p.is_file() or sha(p) != str(output_hashes[key]):
            raise RuntimeError(f'secondary_models output integrity failure: {key} -> {p}')
    return (summary, contract, nxt)

def verify_epoch_store(project: Path, paths: dict[str, Path]) -> pd.DataFrame:
    summary = jread(paths['epoch_store_summary'])
    if not bool(summary.get('complete')):
        raise RuntimeError('Epoch store is not marked complete')
    if int(summary.get('rows', -1)) != 3853:
        raise RuntimeError(f"Unexpected epoch-store rows: {summary.get('rows')}")
    if int(summary.get('compact_cells_per_epoch', -1)) != PER_EPOCH_COMPACT:
        raise RuntimeError('Unexpected compact epoch width')
    if int(summary.get('qc_cells_per_epoch', -1)) != PER_EPOCH_QC:
        raise RuntimeError('Unexpected QC epoch width')
    root = paths['epoch_store_summary'].parent
    rows = []
    stored = summary.get('stored_files', [])
    if not isinstance(stored, list) or not stored:
        raise RuntimeError('Epoch-store stored_files inventory is missing')
    for item in stored:
        rel = Path(str(item['relative_path']))
        p = root / rel
        expected_size = int(item['size_bytes'])
        expected_sha = str(item['sha256'])
        if not p.is_file() or p.stat().st_size != expected_size or sha(p) != expected_sha:
            raise RuntimeError(f'Epoch-store integrity failure: {p}')
        rows.append({'relative_path': str(rel), 'stored_path': str(p), 'size_bytes': expected_size, 'sha256': expected_sha, 'passed': True})
    return pd.DataFrame(rows)

def compact_feature_schema() -> pd.DataFrame:
    rows = []
    for block, start, stop, description in COMPACT_BLOCKS:
        for idx in range(start, stop):
            rows.append({'compact_cell_0based': idx, 'feature_name': f'{block}_{idx - start:03d}', 'feature_block': block, 'block_cell_0based': idx - start, 'block_start_inclusive': start, 'block_stop_exclusive': stop, 'description': description, 'primary_state_representation': True, 'outcome_derived': False, 'qc_cell': False})
    df = pd.DataFrame(rows)
    if len(df) != PER_EPOCH_COMPACT or not np.array_equal(df.compact_cell_0based.to_numpy(), np.arange(PER_EPOCH_COMPACT)):
        raise RuntimeError('Compact feature schema is incomplete')
    return df

def build_trial_and_subject_audit(analysis_rows: pd.DataFrame, old_eligibility: pd.DataFrame, epoch_manifest: pd.DataFrame, compact: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_cols = {'analysis_set', 'raw_behavior_row_id', 'subject_id', 'trial_number', 'within_subject_block_fold', 'subject_acquisition_order_1based', 'within_person_eligible'}
    if not required_cols.issubset(analysis_rows.columns):
        raise RuntimeError(f'Analysis row manifest missing columns: {sorted(required_cols - set(analysis_rows.columns))}')
    ar = analysis_rows[analysis_rows.analysis_set.eq('all_three_epoch_families')].copy()
    if ar.empty:
        raise RuntimeError('No all-three analysis rows')
    old = old_eligibility[old_eligibility.analysis_set.eq('all_three_epoch_families')].copy()
    old = old[['subject_id', 'rows', 'imagery_rows', 'video_rows', 'within_person_eligible']].rename(columns={'within_person_eligible': 'previous_within_person_eligible'})
    ar = ar.merge(old, on='subject_id', how='left', validate='many_to_one')
    if ar.previous_within_person_eligible.isna().any():
        raise RuntimeError('Missing prior subject eligibility')
    em_required = {'raw_behavior_row_id', 'subject_id', 'epoch_family', 'epoch_store_row_0based'}
    if not em_required.issubset(epoch_manifest.columns):
        raise RuntimeError(f'Epoch manifest missing columns: {sorted(em_required - set(epoch_manifest.columns))}')
    em = epoch_manifest[list(em_required)].copy()
    if em.duplicated(['raw_behavior_row_id', 'epoch_family']).any():
        raise RuntimeError('Duplicate trial/epoch rows in epoch manifest')
    trial = ar[['raw_behavior_row_id', 'subject_id', 'trial_number', 'within_subject_block_fold', 'subject_acquisition_order_1based', 'previous_within_person_eligible']].drop_duplicates('raw_behavior_row_id').copy()
    if len(trial) != len(ar):
        raise RuntimeError('All-three analysis rows are not one row per trial')
    for family in FAMILIES:
        x = em[em.epoch_family.eq(family)][['raw_behavior_row_id', 'subject_id', 'epoch_store_row_0based']].rename(columns={'subject_id': f'{family}_subject_id', 'epoch_store_row_0based': f'{family}_epoch_store_row_0based'})
        trial = trial.merge(x, on='raw_behavior_row_id', how='left', validate='one_to_one')
        if trial[f'{family}_epoch_store_row_0based'].isna().any():
            raise RuntimeError(f'Missing {family} for common trial')
        if not trial[f'{family}_subject_id'].astype(str).eq(trial.subject_id.astype(str)).all():
            raise RuntimeError(f'Subject mismatch for {family}')
        trial.drop(columns=[f'{family}_subject_id'], inplace=True)
        trial[f'{family}_epoch_store_row_0based'] = trial[f'{family}_epoch_store_row_0based'].astype(int)
    row_indices = np.concatenate([trial[f'{f}_epoch_store_row_0based'].to_numpy(int) for f in FAMILIES])
    if row_indices.min() < 0 or row_indices.max() >= compact.shape[0]:
        raise RuntimeError('Epoch-store row index outside compact array')
    audit_rows = []
    subject_rows = []
    for subject_id, sg in trial.groupby('subject_id', sort=True):
        family_arrays = {}
        finite_by_family = {}
        for family in FAMILIES:
            idx = sg[f'{family}_epoch_store_row_0based'].to_numpy(int)
            arr = np.asarray(compact[idx], dtype=np.float32)
            if arr.shape != (len(sg), PER_EPOCH_COMPACT):
                raise RuntimeError(f'Compact shape mismatch for {subject_id}/{family}: {arr.shape}')
            if np.isinf(arr).any():
                raise RuntimeError(f'Infinite compact value for {subject_id}/{family}')
            family_arrays[family] = arr
            finite_by_family[family] = np.isfinite(arr).mean(axis=0)
        retained = np.ones(PER_EPOCH_COMPACT, dtype=bool)
        for family in FAMILIES:
            retained &= finite_by_family[family] >= FINITE_FRACTION_PER_FAMILY
        for j in range(PER_EPOCH_COMPACT):
            audit_rows.append({'subject_id': subject_id, 'compact_cell_0based': j, **{f'finite_fraction_{f}': float(finite_by_family[f][j]) for f in FAMILIES}, 'passes_finite_fraction_rule': bool(retained[j])})
        previous = bool(bval(sg.previous_within_person_eligible.iloc[0]))
        enough_trials = len(sg) >= MIN_COMMON_TRIALS
        enough_features = int(retained.sum()) >= MIN_RETAINED_FEATURES
        exact_three = all((sg[f'{f}_epoch_store_row_0based'].notna().all() for f in FAMILIES))
        eligible = previous and enough_trials and enough_features and exact_three
        subject_rows.append({'subject_id': subject_id, 'common_trials': len(sg), 'epoch_observations': len(sg) * 3, 'previous_within_person_eligible': previous, 'minimum_common_trials_required': MIN_COMMON_TRIALS, 'passes_trial_count': enough_trials, 'retained_features_after_finite_rule': int(retained.sum()), 'minimum_retained_features_required': MIN_RETAINED_FEATURES, 'passes_feature_availability': enough_features, 'has_exact_three_epoch_rows': exact_three, 'latent_state_fit_eligible': eligible, 'exclusion_reason': '' if eligible else ';'.join([x for x, bad in [('not_previously_within_person_eligible', not previous), ('fewer_than_minimum_common_trials', not enough_trials), ('insufficient_available_compact_features', not enough_features), ('missing_or_duplicate_epoch_rows', not exact_three)] if bad])})
    subject_df = pd.DataFrame(subject_rows).sort_values('subject_id').reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values(['subject_id', 'compact_cell_0based']).reset_index(drop=True)
    eligible_subjects = set(subject_df.loc[subject_df.latent_state_fit_eligible, 'subject_id'].astype(str))
    trial['latent_state_fit_eligible_subject'] = trial.subject_id.astype(str).isin(eligible_subjects)
    trial = trial.sort_values(['subject_id', 'subject_acquisition_order_1based', 'raw_behavior_row_id']).reset_index(drop=True)
    trial['latent_state_design_trial_row_0based'] = np.arange(len(trial), dtype=int)
    columns = ['latent_state_design_trial_row_0based', 'raw_behavior_row_id', 'subject_id', 'trial_number', 'subject_acquisition_order_1based', 'within_subject_block_fold', 'stimulus_early_epoch_store_row_0based', 'stimulus_late_epoch_store_row_0based', 'rating_early_epoch_store_row_0based', 'previous_within_person_eligible', 'latent_state_fit_eligible_subject']
    return (trial[columns], subject_df, audit_df)

def write_report(path: Path, summary: dict[str, Any], subject_df: pd.DataFrame, feature_schema: pd.DataFrame) -> None:
    eligible = subject_df[subject_df.latent_state_fit_eligible]
    excluded = subject_df[~subject_df.latent_state_fit_eligible]
    lines = ['# Phase latent_state_design — outcome-free latent-state analysis specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f'- prespecified design SHA-256: `{POLICY_SHA}`', '- Latent-state models fitted: **no**', '- Reports/outcomes read: **no**', '- Validation/confirmation opened: **no**', '', '## prespecified cohort', '', f"- Common all-three trials audited: **{summary['counts']['common_trials']:,}**", f'- Eligible participants for latent_state_fit: **{len(eligible)}**', f"- Eligible trial trajectories: **{int(summary['counts']['eligible_common_trials']):,}**", f"- Epoch observations in eligible trajectories: **{int(summary['counts']['eligible_epoch_observations']):,}**", '', '| Subject | Trials | Retained compact cells | Eligible | Exclusion |', '|---|---:|---:|---|---|']
    for _, r in subject_df.iterrows():
        lines.append(f'| `{r.subject_id}` | {int(r.common_trials)} | {int(r.retained_features_after_finite_rule)} | {bool(r.latent_state_fit_eligible)} | `{r.exclusion_reason}` |')
    lines += ['', '## Primary outcome-free representation', '', f'- Compact cells per epoch: **{PER_EPOCH_COMPACT}** across **{feature_schema.feature_block.nunique()}** prespecified blocks.', '- QC cells are excluded from the primary state fit and retained only for post-lock diagnostics.', '- Features must meet the finite-fraction rule separately in all three epoch families within each participant.', '- Median imputation and robust centering/scaling are performed within participant and epoch family.', '- PCA is fit independently within participant on the stacked three-epoch observations.', '', '## prespecified state-count rule', '', f'- Candidate K: `{STATE_COUNTS}`.', f'- Stability: {STABILITY_REPEATS} trial-coupled {int(STABILITY_TRIAL_FRACTION * 100)}% subsamples.', f'- Required: median ARI ≥ {STABILITY_MEDIAN_ARI}, 10th-percentile ARI ≥ {STABILITY_P10_ARI}, collapse rate ≤ {STABILITY_MAX_COLLAPSE_RATE}.', f'- Selection: smallest eligible K within ΔBIC ≤ {BIC_PARSIMONY_DELTA:g} of the best eligible BIC.', '- If no K is eligible, no discrete state solution is forced; continuous trajectories remain valid.', '', '## prespecified trajectory analysis', '', '- Every eligible trial follows `stimulus_early → stimulus_late → rating_early`.', '- Discrete transition summaries are produced only when that participant has a stable state solution.', '- Continuous PCA-space displacement and path-geometry summaries are produced for every eligible participant.', f'- Structured transitions are compared with {TRAJECTORY_NULL_REPEATS} within-participant trial-linkage permutations.', '', '## Phase boundary', '', '- latent_state_fit may fit only the prespecified outcome-free models described here.', '- Context, valence, liking, arousal, emotion reports, and report-derived factors cannot be used in latent_state_fit.', '- Post-hoc report/context mapping begins only after latent_state_fit assignments and trajectories are written and hashed.', '- Cross-person alignment is deferred to a later outcome-free layer.', '- Validation and confirmation remain closed.', '']
    if not excluded.empty:
        lines.append(f"Excluded participants at this specification: {', '.join(excluded.subject_id.astype(str))}.")
        lines.append('')
    path.write_text(''.join(lines), encoding='utf-8')

def self_test() -> None:
    rng = np.random.default_rng(1)
    subjects = ['sub-a', 'sub-b', 'sub-c']
    trials = []
    epochs = []
    compact = []
    row = 0
    for s_idx, subject in enumerate(subjects):
        n = [30, 31, 15][s_idx]
        for t in range(n):
            trial_id = f'{subject}-{t:03d}'
            trials.append({'analysis_set': 'all_three_epoch_families', 'raw_behavior_row_id': trial_id, 'subject_id': subject, 'trial_number': t + 1, 'within_subject_block_fold': t % 5 + 1, 'subject_acquisition_order_1based': t + 1, 'within_person_eligible': n >= 30})
            for family in FAMILIES:
                epochs.append({'raw_behavior_row_id': trial_id, 'subject_id': subject, 'epoch_family': family, 'epoch_store_row_0based': row})
                x = rng.normal(size=PER_EPOCH_COMPACT).astype(np.float32)
                compact.append(x)
                row += 1
    old = pd.DataFrame([{'subject_id': s, 'rows': n, 'imagery_rows': n // 2, 'video_rows': n - n // 2, 'within_person_eligible': n >= 30, 'analysis_set': 'all_three_epoch_families'} for s, n in zip(subjects, [30, 31, 15])])
    trial_df, subject_df, audit_df = build_trial_and_subject_audit(pd.DataFrame(trials), old, pd.DataFrame(epochs), np.asarray(compact, dtype=np.float32))
    assert len(trial_df) == 76
    assert int(subject_df.latent_state_fit_eligible.sum()) == 2
    assert len(audit_df) == len(subjects) * PER_EPOCH_COMPACT
    schema = compact_feature_schema()
    assert len(schema) == PER_EPOCH_COMPACT
    assert DISCOVERY_POLICY['forbidden'][0].startswith('reading valence')
    print('latent_state_design self-test: PASS')

def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    paths = {k: project / v for k, v in REL.items()}
    out = output_paths(project)
    required(paths.values())
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"Output already exists; review or use --force: {out['contract']}")
    log_lines: list[str] = []

    def note(message: str) -> None:
        line = f'[{now()}] {message}'
        log_lines.append(line)
        print(line, flush=True)
        out['log'].write_text(''.join(log_lines) + '', encoding='utf-8')
    note('Verifying completed secondary_models contract and authorization.')
    secondary_models_summary, secondary_models_contract, secondary_models_next = verify_secondary_models(project, paths)
    note('Rehashing outcome-free secondary_models epoch store.')
    epoch_rehash = verify_epoch_store(project, paths)
    epoch_rehash.to_csv(out['epoch_store_rehash'], index=False)
    note('Loading only outcome-free manifests and compact/QC arrays.')
    analysis_rows = pd.read_csv(paths['primary_analysis_reference_analysis_rows'], low_memory=False)
    old_eligibility = pd.read_csv(paths['primary_analysis_reference_subject_eligibility'], low_memory=False)
    epoch_manifest = pd.read_csv(paths['epoch_manifest'], low_memory=False)
    compact = np.load(paths['epoch_compact'], mmap_mode='r', allow_pickle=False)
    qc = np.load(paths['epoch_qc'], mmap_mode='r', allow_pickle=False)
    if compact.shape != (3853, PER_EPOCH_COMPACT):
        raise RuntimeError(f'Unexpected compact array shape: {compact.shape}')
    if qc.shape != (3853, PER_EPOCH_QC) or not np.isfinite(np.asarray(qc)).all():
        raise RuntimeError(f'Unexpected or nonfinite QC array: {qc.shape}')
    note('Freezing common-trial cohort and participant-level feature availability.')
    trial_df, subject_df, audit_df = build_trial_and_subject_audit(analysis_rows, old_eligibility, epoch_manifest, compact)
    if int(subject_df.latent_state_fit_eligible.sum()) < 1:
        raise RuntimeError('No participant is eligible for latent_state_fit')
    trial_df.to_csv(out['trial_manifest'], index=False)
    subject_df.to_csv(out['subject_eligibility'], index=False)
    audit_df.to_csv(out['participant_feature_audit'], index=False)
    schema = compact_feature_schema()
    schema.to_csv(out['feature_schema'], index=False)
    atom_json(out['latent_policy'], DISCOVERY_POLICY)
    atom_json(out['trajectory_policy'], {'phase': PHASE, 'policy_sha256': POLICY_SHA, 'epoch_order': FAMILIES, 'trajectory_outputs': DISCOVERY_POLICY['trajectory_outputs'], 'diagnostic_only_after_state_lock': DISCOVERY_POLICY['diagnostic_only_after_state_lock'], 'context_and_reports_forbidden_in_latent_state_fit': True, 'validation_opened': False, 'confirmation_opened': False})
    eligible = subject_df[subject_df.latent_state_fit_eligible]
    eligible_subjects = set(eligible.subject_id.astype(str))
    eligible_trials = trial_df[trial_df.subject_id.astype(str).isin(eligible_subjects)]
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'outcome_free_within_person_latent_state_and_trajectory_design_prespecified_candidate', 'design_policy_sha256': POLICY_SHA, 'counts': {'epoch_store_files_rehashed': len(epoch_rehash), 'epoch_store_bytes_rehashed': int(epoch_rehash.size_bytes.sum()), 'epoch_store_rows': int(compact.shape[0]), 'compact_cells_per_epoch': int(compact.shape[1]), 'qc_cells_per_epoch': int(qc.shape[1]), 'common_trials': len(trial_df), 'common_trial_participants': int(trial_df.subject_id.nunique()), 'eligible_participants': len(eligible), 'eligible_common_trials': len(eligible_trials), 'eligible_epoch_observations': len(eligible_trials) * 3, 'feature_audit_rows': len(audit_df)}, 'eligibility': subject_df.to_dict(orient='records'), 'decision': {'secondary_models_verified_complete': True, 'epoch_store_byte_verified': True, 'outcome_or_report_columns_read': False, 'latent_state_models_fitted': False, 'state_count_selected': False, 'cross_person_alignment_performed': False, 'latent_state_fit_design_prespecified': True, 'validation_opened': False, 'confirmation_opened': False}}
    atom_json(out['summary'], summary)
    output_hashes = {}
    for key, p in out.items():
        if key not in {'contract', 'next_policy', 'report', 'log'} and p.is_file():
            output_hashes[key] = sha(p)
    contract = {**summary, 'design_policy': DISCOVERY_POLICY, 'input_hashes': {k: sha(p) for k, p in paths.items()}, 'secondary_models_contract_sha256': sha(paths['secondary_models_contract']), 'secondary_models_next_policy_sha256': sha(paths['secondary_models_next']), 'output_hashes': output_hashes, 'outcomes_modified': False, 'feature_store_modified': False, 'secondary_models_cache_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'fit only the prespecified outcome-free within-person latent-state and three-epoch trajectory models', 'required_inputs': {'latent_state_design_contract': str(out['contract']), 'prespecified_latent_state_policy': str(out['latent_policy']), 'prespecified_trajectory_policy': str(out['trajectory_policy']), 'common_trial_manifest': str(out['trial_manifest']), 'subject_eligibility': str(out['subject_eligibility']), 'participant_feature_audit': str(out['participant_feature_audit']), 'compact_feature_schema': str(out['feature_schema']), 'epoch_store': str(paths['epoch_store_summary'].parent)}, 'required_execution': ['fit every eligible participant independently under all candidate K values', 'execute every prespecified stability resample and state-count rule', 'write continuous trajectories for every eligible participant', 'write discrete trajectories only for participants with a stable selected K', 'run prespecified outcome-free trajectory nulls', 'hash assignments and trajectories before any report/context mapping'], 'forbidden': DISCOVERY_POLICY['forbidden'], 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['next_policy'], next_policy)
    write_report(out['report'], summary, subject_df, schema)
    note('latent_state_design complete. Outcome-free design is prespecified; no latent-state model was fitted.')
    print('' + '=' * 100)
    print('PHASE latent_state_design LATENT-STATE analysis specification COMPLETED')
    print('=' * 100)
    print(f'Design-policy SHA-256: {POLICY_SHA}')
    print(f'Eligible participants: {len(eligible)}')
    print(f'Eligible trajectories: {len(eligible_trials)}')
    print('Latent-state models fitted: False')
    print('Reports/outcomes read: False')
    print('Validation opened: False')
    print('Confirmation opened: False')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    raise SystemExit(main())