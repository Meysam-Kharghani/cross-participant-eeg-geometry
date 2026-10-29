# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase cross_person_alignment_design: specification an outcome-free cross-person alignment design.

This phase verifies the fixed latent_state_fit within-person latent-state outputs and
specifications, before any alignment fit, two complementary cross-person layers:

1. participant-balanced alignment of stable local-state centroids in the common
   169-cell axes, with injective within-participant assignment to shared state
   families; and
2. participant-balanced alignment of rotation/sign-invariant continuous
   three-epoch trajectory geometry for every eligible participant, including
   participants without a stable discrete state solution.

No valence, liking, arousal, emotion report, context, QC value, validation
participant, or confirmation participant is read. No alignment model is fit in
this phase."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'cross_person_alignment_design'
NEXT_PHASE = 'cross_person_alignment_fit'
SEED = 540
PER_EPOCH_COMPACT = 169
STATE_ALIGNMENT_PCS = 8
STATE_FAMILY_COUNTS = [3, 4, 5, 6]
TRAJECTORY_GEOMETRY_FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
TRAJECTORY_FAMILY_COUNTS = [2, 3, 4, 5, 6]
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ALIGNMENT_N_INIT = 100
ALIGNMENT_MAX_ITER = 500
ALIGNMENT_TOL = 1e-07
STABILITY_REPEATS = 200
STABILITY_PARTICIPANT_FRACTION = 0.8
STABILITY_TRIAL_FRACTION = 0.8
STABILITY_MEDIAN_ARI = 0.6
STABILITY_P10_ARI = 0.3
STABILITY_MAX_COLLAPSE_RATE = 0.1
SILHOUETTE_PARSIMONY_DELTA = 0.05
STATE_MIN_PARTICIPANTS_PER_FAMILY = 3
STATE_MIN_WEIGHTED_OCCUPANCY = 0.08
TRAJECTORY_MIN_PARTICIPANTS_PER_FAMILY = 5
TRAJECTORY_MIN_TRIALS_PER_FAMILY = 30
TRAJECTORY_MIN_WEIGHTED_OCCUPANCY = 0.05
LINKAGE_NULL_REPEATS = 1000
REL = {'summary': Path('outputs/qc/latent_state_fit_latent_state_fit_summary.json'), 'contract': Path('config/latent_state_fit_latent_state_fit_contract.json'), 'next_policy': Path('analysis_specifications/S039_latent_state_fit_cross_person_alignment.json'), 'primary_lock': Path('outputs/qc/latent_state_fit_primary_lock_hashes.json'), 'selection': Path('outputs/tables/latent_state_fit_participant_state_selection.csv'), 'candidate_metrics': Path('outputs/tables/latent_state_fit_state_candidate_metrics.csv'), 'stability': Path('outputs/tables/latent_state_fit_stability_replicates.csv'), 'preprocessing': Path('outputs/tables/latent_state_fit_participant_preprocessing_summary.csv'), 'pca_scores': Path('outputs/tables/latent_state_fit_epoch_pca_scores.csv'), 'pca_loadings': Path('outputs/tables/latent_state_fit_pca_loadings.csv'), 'assignments': Path('outputs/tables/latent_state_fit_state_assignments.csv'), 'continuous_trajectories': Path('outputs/tables/latent_state_fit_continuous_trajectories.csv'), 'discrete_trajectories': Path('outputs/tables/latent_state_fit_discrete_trajectories.csv'), 'occupancy': Path('outputs/tables/latent_state_fit_state_occupancy.csv'), 'transitions': Path('outputs/tables/latent_state_fit_transition_matrices.csv'), 'transition_summary': Path('outputs/tables/latent_state_fit_participant_transition_summaries.csv'), 'null_replicates': Path('outputs/tables/latent_state_fit_trajectory_null_replicates.csv'), 'null_tests': Path('outputs/tables/latent_state_fit_trajectory_null_tests.csv'), 'centroids_pca': Path('outputs/tables/latent_state_fit_state_centroids_pca.csv'), 'centroids_common': Path('outputs/tables/latent_state_fit_state_centroids_common_axes.csv')}
PRIMARY_OUTPUT_MAP = {'preprocessing': REL['preprocessing'], 'candidate_metrics': REL['candidate_metrics'], 'stability': REL['stability'], 'selection': REL['selection'], 'pca_scores': REL['pca_scores'], 'pca_loadings': REL['pca_loadings'], 'assignments': REL['assignments'], 'continuous_trajectories': REL['continuous_trajectories'], 'discrete_trajectories': REL['discrete_trajectories'], 'occupancy': REL['occupancy'], 'transitions': REL['transitions'], 'transition_summary': REL['transition_summary'], 'null_replicates': REL['null_replicates'], 'null_tests': REL['null_tests'], 'centroids_pca': REL['centroids_pca'], 'centroids_common': REL['centroids_common']}
BANNED_EXACT_COLUMNS = {'valence', 'liking', 'arousal', 'emotion', 'emotion_label', 'emotion_category', 'context', 'condition', 'stimulus_category', 'stimulus_type', 'pleasantness', 'unpleasantness', 'wanting', 'pain_intensity', 'pain_unpleasantness', 'subjective_factor_1', 'subjective_factor_2', 'report_factor_1', 'report_factor_2'}
ALIGNMENT_POLICY: dict[str, Any] = {'phase': PHASE, 'seed': SEED, 'scientific_role': 'specification outcome-free cross-person alignment of local state centroids and continuous three-epoch trajectory geometry before any report/context mapping', 'discovery_only': True, 'validation_opened': False, 'confirmation_opened': False, 'input_boundary': {'fixed_source_phase': 'latent_state_fit', 'reports_outcomes_context_read': [], 'qc_values_read': [], 'allowed_primary_inputs': ['participant_state_selection', 'state_centroids_common_axes', 'state_centroids_pca for integrity/schema audit only', 'state_assignments for lineage/count audit only', 'continuous_trajectories restricted to identifiers and six invariant geometry columns'], 'epoch_family_is_not_an_alignment_feature': True, 'prior_decoder_performance_is_not_an_alignment_feature': True}, 'two_branch_design': {'stable_discrete_state_branch': {'population': 'participants with a latent_state_fit-stable discrete solution only', 'object': 'one fixed local-state centroid', 'common_axes': PER_EPOCH_COMPACT, 'missing_state_rule': 'participants without stable discrete states contribute no pseudo-state and are never forced into this branch', 'participant_balance': 'state sample weights equal prespecified within-participant state proportions; weights sum to one within every participant', 'preprocessing': ['require finite fixed common-axis centroid values for every retained cell', 'compute participant-balanced weighted median and weighted IQR separately for each common-axis cell', f'drop a cell only if weighted IQR <= {IQR_FLOOR:g}', f'winsorize robust standardized values to [-{WINSOR_LIMIT:g}, {WINSOR_LIMIT:g}]', f'fit participant-weighted PCA with at most {STATE_ALIGNMENT_PCS} components', 'orient every component deterministically so its largest-absolute loading is positive'], 'alignment_model': {'model': 'participant-balanced constrained k-means', 'candidate_shared_state_family_counts': STATE_FAMILY_COUNTS, 'distance': 'squared Euclidean distance in prespecified aligned-state PCA space', 'within_participant_constraint': 'injective Hungarian assignment: two local states from the same participant cannot map to the same shared family', 'initializations': ALIGNMENT_N_INIT, 'maximum_iterations': ALIGNMENT_MAX_ITER, 'tolerance': ALIGNMENT_TOL, 'canonical_family_label_rule': 'sort final prototypes lexicographically by deterministically oriented aligned-state PC1, then PC2, etc.'}, 'stability': {'resampling_unit': 'participant', 'repeats': STABILITY_REPEATS, 'participant_fraction_without_replacement': STABILITY_PARTICIPANT_FRACTION, 'entire_chain_refit': 'weighted robust scaling, weighted PCA, and constrained alignment are all refit inside every resample', 'comparison': 'adjusted Rand index on retained local-state objects against full-fit assignments', 'collapse_definition': {'minimum_distinct_participants_per_family': STATE_MIN_PARTICIPANTS_PER_FAMILY, 'minimum_participant-balanced_weighted_occupancy': STATE_MIN_WEIGHTED_OCCUPANCY}, 'eligible_family_count_requirements': {'full_fit_converged': True, 'median_ari_at_least': STABILITY_MEDIAN_ARI, 'ari_10th_percentile_at_least': STABILITY_P10_ARI, 'collapse_rate_at_most': STABILITY_MAX_COLLAPSE_RATE}, 'selection_rule': f'among eligible family counts, choose the smallest count with participant-weighted silhouette >= maximum eligible silhouette - {SILHOUETTE_PARSIMONY_DELTA:g}', 'no_eligible_count_rule': 'do not force shared discrete state families; retain the continuous aligned-state embedding and pairwise distances only'}, 'leave_one_participant_out_diagnostic': 'refit on all other stable participants, assign held-out local states injectively, and report assignment margins/agreement; never revise family-count selection'}, 'continuous_trajectory_geometry_branch': {'population': 'all latent_state_fit-eligible participants, including continuous-only participants', 'object': 'one fixed three-epoch trial trajectory', 'features': TRAJECTORY_GEOMETRY_FEATURES, 'rotation_sign_invariance': True, 'transformations_before_within_participant_scaling': {'early_to_late_norm': 'log1p', 'late_to_rating_norm': 'log1p', 'early_to_rating_direct_norm': 'log1p', 'total_path_length': 'log1p', 'directness_ratio': 'logit after clipping to [1e-6, 1-1e-6]', 'segment_cosine': 'atanh after clipping to [-1+1e-6, 1-1e-6]'}, 'participant_balance': 'each trial has weight 1 divided by that participant trial count; weights sum to one within every participant', 'preprocessing': ['apply prespecified monotone transformations', 'within each participant, median-center and divide by IQR for each geometry feature', f'if within-participant IQR <= {IQR_FLOOR:g}, use pooled participant-balanced IQR for that feature', f'winsorize standardized values to [-{WINSOR_LIMIT:g}, {WINSOR_LIMIT:g}]'], 'alignment_model': {'model': 'participant-balanced weighted k-means', 'candidate_shared_trajectory_family_counts': TRAJECTORY_FAMILY_COUNTS, 'distance': 'squared Euclidean distance in the six prespecified standardized geometry coordinates', 'initializations': ALIGNMENT_N_INIT, 'maximum_iterations': ALIGNMENT_MAX_ITER, 'tolerance': ALIGNMENT_TOL, 'canonical_family_label_rule': 'sort final centroids lexicographically in the prespecified geometry-feature order'}, 'stability': {'resampling_unit': 'two-level participant then trial resampling', 'repeats': STABILITY_REPEATS, 'participant_fraction_without_replacement': STABILITY_PARTICIPANT_FRACTION, 'trial_fraction_without_replacement_within_retained_participant': STABILITY_TRIAL_FRACTION, 'entire_chain_refit': 'transform fallback statistics, within-person scaling, and weighted k-means are all refit inside every resample', 'comparison': 'adjusted Rand index on retained trajectories against full-fit assignments', 'collapse_definition': {'minimum_distinct_participants_per_family': TRAJECTORY_MIN_PARTICIPANTS_PER_FAMILY, 'minimum_trial_count_per_family': TRAJECTORY_MIN_TRIALS_PER_FAMILY, 'minimum_participant-balanced_weighted_occupancy': TRAJECTORY_MIN_WEIGHTED_OCCUPANCY}, 'eligible_family_count_requirements': {'full_fit_converged': True, 'median_ari_at_least': STABILITY_MEDIAN_ARI, 'ari_10th_percentile_at_least': STABILITY_P10_ARI, 'collapse_rate_at_most': STABILITY_MAX_COLLAPSE_RATE}, 'selection_rule': f'among eligible family counts, choose the smallest count with participant-weighted silhouette >= maximum eligible silhouette - {SILHOUETTE_PARSIMONY_DELTA:g}', 'no_eligible_count_rule': 'do not force trajectory families; retain the six-dimensional continuous geometry representation'}, 'leave_one_participant_out_diagnostic': 'fit on all other participants, assign held-out trajectories to prespecified training centroids, and report confidence/occupancy; never revise family-count selection'}}, 'outcome_free_linkage_after_both_branches_are_fixed': {'eligible_rows': 'discrete trajectories belonging to participants with stable local states', 'global_path': 'replace each local state by its prespecified shared state-family assignment when available', 'cross_branch_summary': 'cross-tabulate shared state-family paths against shared continuous-trajectory families', 'null_repeats': LINKAGE_NULL_REPEATS, 'null_scheme': 'within participant, permute continuous-trajectory family labels across trials while preserving local/global path frequencies', 'statistics': ['mutual_information', 'normalized_mutual_information', 'path_by_geometry_family_cramers_v'], 'failure_rule': 'if either branch has no stable discrete family solution, omit discrete cross-branch linkage and retain both continuous outputs'}, 'execution_order_and_locks': ['verify every latent_state_fit primary hash before reading alignment inputs', 'fit and write both outcome-free alignment branches', 'write selected family counts, embeddings, assignments, prototypes, stability, and leave-one-participant-out diagnostics', 'SHA-256 lock all alignment outputs before any context/report table may be opened', 'perform only the prespecified outcome-free cross-branch linkage after both branch locks'], 'claim_guardrails': ['a shared family is a reproducible alignment of this representation, not a universal emotion or valence category', 'absence of stable shared families does not imply absence of individual neurophysiological structure', 'participant identity, epoch association, or context separability is not emotion decoding', 'continuous-only participants remain represented through invariant trajectory geometry without fabricated states', 'report/context mapping is strictly post-hoc and begins only after cross_person_alignment_fit outputs are hashed'], 'forbidden': ['reading valence, liking, arousal, emotion reports, labels, context, or report-derived factors during cross_person_alignment_design or cross_person_alignment_fit', 'using QC values, epoch family, stimulus identity, participant identity targets, or prior decoding performance as alignment features', 'changing latent_state_fit local K, local assignments, centroids, trajectories, or participant inclusion', 'creating pseudo-states for continuous-only participants', 'selecting shared-family counts from later report/context association', 'opening validation or confirmation participants']}
POLICY_SHA = hashlib.sha256(json.dumps(ALIGNMENT_POLICY, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

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
        raise RuntimeError(f'Forbidden report/context columns found in {source}: {bad}')

def output_paths(project: Path) -> dict[str, Path]:
    return {'log': project / 'outputs/logs/cross_person_alignment_design_cross_person_alignment_design_specification.log', 'latent_state_fit_rehash': project / 'outputs/qc/cross_person_alignment_design_latent_state_fit_primary_rehash.csv', 'input_audit': project / 'outputs/qc/cross_person_alignment_design_alignment_input_audit.csv', 'summary': project / 'outputs/qc/cross_person_alignment_design_alignment_design_specification_summary.json', 'report': project / 'outputs/reports/cross_person_alignment_design_alignment_design_specification_report.md', 'state_manifest': project / 'outputs/tables/cross_person_alignment_design_stable_local_state_manifest.csv', 'trajectory_manifest': project / 'outputs/tables/cross_person_alignment_design_continuous_geometry_manifest.csv', 'feature_schema': project / 'outputs/tables/cross_person_alignment_design_alignment_feature_schema.csv', 'alignment_policy': project / 'analysis_specifications/S012_cross_person_alignment.json', 'contract': project / 'config/cross_person_alignment_design_alignment_design_contract.json', 'next_policy': project / 'analysis_specifications/S011_cross_person_alignment.json'}

def verify_latent_state_fit(project: Path, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    summary = jread(paths['summary'])
    contract = jread(paths['contract'])
    nxt = jread(paths['next_policy'])
    lock = jread(paths['primary_lock'])
    expected_status = 'prespecified_outcome_free_within_person_latent_states_and_three_epoch_trajectories_complete_candidate'
    if summary.get('phase') != 'latent_state_fit' or contract.get('phase') != 'latent_state_fit':
        raise RuntimeError('latent_state_fit phase mismatch')
    if summary.get('status') != expected_status or contract.get('status') != expected_status:
        raise RuntimeError('latent_state_fit status is not complete candidate')
    if nxt.get('next_phase') != PHASE or not bval(nxt.get('next_phase_permitted')):
        raise RuntimeError('latent_state_fit did not authorize cross_person_alignment_design')
    if lock.get('phase') != 'latent_state_fit':
        raise RuntimeError('Primary-lock phase mismatch')
    if not bval(lock.get('primary_outputs_fixed_before_qc_diagnostics')):
        raise RuntimeError('latent_state_fit primary outputs were not fixed before diagnostics')
    if bval(lock.get('context_or_reports_read_before_lock')):
        raise RuntimeError('latent_state_fit reports/context boundary violated')
    if bval(lock.get('qc_values_loaded_or_interpreted_before_lock')):
        raise RuntimeError('latent_state_fit QC boundary violated')
    decision = summary.get('decision', {})
    required_true = ['all_eligible_participants_fitted', 'all_candidate_k_fitted_or_failure_recorded', 'all_stability_resamples_executed_or_failure_recorded', 'continuous_trajectories_written_for_all_eligible_trials', 'primary_outputs_fixed_before_qc_diagnostics']
    for key in required_true:
        if not bval(decision.get(key)):
            raise RuntimeError(f'latent_state_fit decision flag is false: {key}')
    required_false = ['discrete_states_forced_when_no_eligible_k', 'qc_used_in_primary_fit_or_k_selection', 'outcome_report_or_context_columns_read', 'cross_person_alignment_performed', 'posthoc_report_mapping_performed', 'validation_opened', 'confirmation_opened']
    for key in required_false:
        if bval(decision.get(key)):
            raise RuntimeError(f'latent_state_fit forbidden decision flag is true: {key}')
    expected_hashes = lock.get('hashes', {})
    if set(expected_hashes) != set(PRIMARY_OUTPUT_MAP):
        raise RuntimeError(f'Primary-lock keys mismatch: {sorted(expected_hashes)}')
    contract_fixed = contract.get('primary_output_hashes_before_diagnostics', {})
    if contract_fixed != expected_hashes:
        raise RuntimeError('latent_state_fit contract and primary lock disagree')
    if contract.get('primary_lock_sha256') != sha(paths['primary_lock']):
        raise RuntimeError('latent_state_fit primary-lock SHA mismatch in contract')
    rows = []
    for key, rel in PRIMARY_OUTPUT_MAP.items():
        p = project / rel
        observed = sha(p)
        expected = str(expected_hashes[key])
        rows.append({'primary_output_key': key, 'relative_path': str(rel), 'size_bytes': p.stat().st_size, 'expected_sha256': expected, 'observed_sha256': observed, 'sha256_match': observed == expected})
        if observed != expected:
            raise RuntimeError(f'latent_state_fit primary output hash mismatch: {key}')
    return (summary, contract, nxt, lock, pd.DataFrame(rows))

def build_state_manifest(selection: pd.DataFrame, centroids: pd.DataFrame, centroids_pca: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    ensure_no_banned_columns(selection, 'participant_state_selection')
    ensure_no_banned_columns(centroids, 'state_centroids_common_axes')
    ensure_no_banned_columns(centroids_pca, 'state_centroids_pca')
    ensure_no_banned_columns(assignments, 'state_assignments')
    required_sel = {'subject_id', 'trial_count', 'epoch_observations', 'pca_components', 'stable_discrete_solution', 'selected_k', 'selection_reason'}
    if not required_sel.issubset(selection.columns):
        raise RuntimeError(f'Selection schema missing: {sorted(required_sel - set(selection.columns))}')
    if len(selection) != 31 or selection.subject_id.nunique() != 31:
        raise RuntimeError(f'Unexpected selection cohort: rows={len(selection)}, participants={selection.subject_id.nunique()}')
    stable = selection[selection.stable_discrete_solution.map(bval)].copy()
    if len(stable) != 19:
        raise RuntimeError(f'Expected 19 stable participants, found {len(stable)}')
    stable['selected_k'] = stable.selected_k.astype(int)
    if stable.selected_k.sum() != 40:
        raise RuntimeError(f'Expected 40 stable local states, found K sum={stable.selected_k.sum()}')
    req_cent = {'subject_id', 'selected_k', 'state_1based', 'compact_cell_0based', 'feature_name', 'within_person_stage_standardized_centroid', 'feature_retained', 'state_observations', 'state_proportion'}
    if not req_cent.issubset(centroids.columns):
        raise RuntimeError(f'Common-centroid schema missing: {sorted(req_cent - set(centroids.columns))}')
    if len(centroids) != 40 * PER_EPOCH_COMPACT:
        raise RuntimeError(f'Unexpected common-centroid rows: {len(centroids)}')
    if centroids[['subject_id', 'state_1based', 'compact_cell_0based']].duplicated().any():
        raise RuntimeError('Duplicate common-centroid cells')
    if centroids.subject_id.nunique() != 19:
        raise RuntimeError('Common centroids do not contain 19 participants')
    if not centroids.feature_retained.map(bval).all():
        raise RuntimeError('A fixed local-state centroid contains an unretained common-axis cell')
    if not np.isfinite(centroids.within_person_stage_standardized_centroid.to_numpy(float)).all():
        raise RuntimeError('Nonfinite common-axis centroid value')
    per_state_cells = centroids.groupby(['subject_id', 'state_1based']).compact_cell_0based.nunique()
    if not (per_state_cells == PER_EPOCH_COMPACT).all():
        raise RuntimeError('Every local state must have exactly 169 common-axis cells')
    req_pca = {'subject_id', 'selected_k', 'state_1based', 'pc_1based', 'centroid_value', 'state_weight'}
    if not req_pca.issubset(centroids_pca.columns):
        raise RuntimeError(f'PCA-centroid schema missing: {sorted(req_pca - set(centroids_pca.columns))}')
    if len(centroids_pca) != 40 * 8 or not np.isfinite(centroids_pca.centroid_value.to_numpy(float)).all():
        raise RuntimeError('Unexpected PCA-centroid content')
    req_assn = {'subject_id', 'raw_behavior_row_id', 'epoch_family', 'selected_k', 'state_1based'}
    if not req_assn.issubset(assignments.columns):
        raise RuntimeError(f'Assignment schema missing: {sorted(req_assn - set(assignments.columns))}')
    if len(assignments) != 2319 or assignments.subject_id.nunique() != 19:
        raise RuntimeError(f'Unexpected fixed assignments: {assignments.shape}')
    state = centroids[['subject_id', 'selected_k', 'state_1based', 'state_observations', 'state_proportion']].drop_duplicates().sort_values(['subject_id', 'state_1based']).reset_index(drop=True)
    if len(state) != 40:
        raise RuntimeError(f'Expected 40 local-state objects, found {len(state)}')
    sums = state.groupby('subject_id').state_proportion.sum()
    if not np.allclose(sums.to_numpy(float), 1.0, atol=1e-08):
        raise RuntimeError('prespecified state proportions do not sum to one within participant')
    assn_counts = assignments.groupby(['subject_id', 'state_1based']).size().rename('assignment_observations').reset_index()
    state = state.merge(assn_counts, on=['subject_id', 'state_1based'], how='left', validate='one_to_one')
    if state.assignment_observations.isna().any() or not np.array_equal(state.state_observations.to_numpy(int), state.assignment_observations.to_numpy(int)):
        raise RuntimeError('State centroid and assignment observation counts disagree')
    state['state_object_id'] = state.subject_id.astype(str) + '|local_state_' + state.state_1based.astype(int).astype(str)
    state['participant_total_alignment_weight'] = 1.0
    state['state_alignment_weight'] = state.state_proportion.astype(float)
    state['common_axis_cells'] = PER_EPOCH_COMPACT
    state['eligible_for_discrete_cross_person_alignment'] = True
    return state[['state_object_id', 'subject_id', 'selected_k', 'state_1based', 'state_observations', 'assignment_observations', 'state_proportion', 'state_alignment_weight', 'participant_total_alignment_weight', 'common_axis_cells', 'eligible_for_discrete_cross_person_alignment']]

def build_trajectory_manifest(continuous: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    ensure_no_banned_columns(continuous, 'continuous_trajectories')
    required_cols = {'subject_id', 'raw_behavior_row_id', 'trial_number', 'pca_components', *TRAJECTORY_GEOMETRY_FEATURES}
    if not required_cols.issubset(continuous.columns):
        raise RuntimeError(f'Continuous-trajectory schema missing: {sorted(required_cols - set(continuous.columns))}')
    if len(continuous) != 1237 or continuous.subject_id.nunique() != 31:
        raise RuntimeError(f'Unexpected continuous trajectory cohort: {continuous.shape}, participants={continuous.subject_id.nunique()}')
    if continuous[['subject_id', 'raw_behavior_row_id']].duplicated().any():
        raise RuntimeError('Duplicate continuous trajectory identifiers')
    x = continuous[TRAJECTORY_GEOMETRY_FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite continuous trajectory geometry')
    if (continuous[['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length']] < 0).any().any():
        raise RuntimeError('Negative path norm/length')
    if not continuous.directness_ratio.between(0, 1, inclusive='both').all():
        raise RuntimeError('Directness ratio outside [0,1]')
    if not continuous.segment_cosine.between(-1, 1, inclusive='both').all():
        raise RuntimeError('Segment cosine outside [-1,1]')
    stable_map = selection.set_index('subject_id').stable_discrete_solution.map(bval).to_dict()
    counts = continuous.groupby('subject_id').size().rename('participant_trial_count')
    out = continuous[['subject_id', 'raw_behavior_row_id', 'trial_number', 'pca_components', *TRAJECTORY_GEOMETRY_FEATURES]].copy()
    out['participant_trial_count'] = out.subject_id.map(counts).astype(int)
    out['trajectory_alignment_weight'] = 1.0 / out.participant_trial_count.astype(float)
    out['participant_total_alignment_weight'] = out.groupby('subject_id').trajectory_alignment_weight.transform('sum')
    out['has_stable_local_discrete_states'] = out.subject_id.map(stable_map).map(bool)
    out['eligible_for_continuous_geometry_alignment'] = True
    if not np.allclose(out.groupby('subject_id').trajectory_alignment_weight.sum().to_numpy(float), 1.0, atol=1e-12):
        raise RuntimeError('Trajectory weights do not sum to one within participant')
    return out.sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id']).reset_index(drop=True)

def feature_schema(centroids: pd.DataFrame) -> pd.DataFrame:
    common = centroids[['compact_cell_0based', 'feature_name']].drop_duplicates().sort_values('compact_cell_0based')
    if len(common) != PER_EPOCH_COMPACT or not np.array_equal(common.compact_cell_0based.to_numpy(int), np.arange(PER_EPOCH_COMPACT)):
        raise RuntimeError('Common-axis feature schema is not exactly 0..168')
    common = common.copy()
    common['alignment_branch'] = 'stable_discrete_state_centroids'
    common['input_feature'] = common.feature_name
    common['prespecified_transform'] = 'participant-balanced weighted median/IQR; winsorize; weighted PCA'
    common['feature_order_0based'] = common.compact_cell_0based.astype(int)
    common['outcome_context_qc_free'] = True
    common = common[['alignment_branch', 'feature_order_0based', 'input_feature', 'prespecified_transform', 'outcome_context_qc_free']]
    transforms = {'early_to_late_norm': 'log1p then within-participant median/IQR', 'late_to_rating_norm': 'log1p then within-participant median/IQR', 'early_to_rating_direct_norm': 'log1p then within-participant median/IQR', 'total_path_length': 'log1p then within-participant median/IQR', 'directness_ratio': 'clipped logit then within-participant median/IQR', 'segment_cosine': 'clipped atanh then within-participant median/IQR'}
    geom = pd.DataFrame([{'alignment_branch': 'continuous_trajectory_geometry', 'feature_order_0based': i, 'input_feature': name, 'prespecified_transform': transforms[name], 'outcome_context_qc_free': True} for i, name in enumerate(TRAJECTORY_GEOMETRY_FEATURES)])
    return pd.concat([common, geom], ignore_index=True)

def build_input_audit(selection: pd.DataFrame, state_manifest: pd.DataFrame, trajectory_manifest: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    stable = selection[selection.stable_discrete_solution.map(bval)]
    continuous_only = selection[~selection.stable_discrete_solution.map(bval)]
    rows = [('latent_state_fit_participants', len(selection), 31, len(selection) == 31), ('stable_discrete_participants', len(stable), 19, len(stable) == 19), ('continuous_only_participants', len(continuous_only), 12, len(continuous_only) == 12), ('stable_local_state_objects', len(state_manifest), 40, len(state_manifest) == 40), ('common_axis_centroid_rows', len(centroids), 40 * PER_EPOCH_COMPACT, len(centroids) == 40 * PER_EPOCH_COMPACT), ('common_axis_features', centroids.compact_cell_0based.nunique(), PER_EPOCH_COMPACT, centroids.compact_cell_0based.nunique() == PER_EPOCH_COMPACT), ('continuous_trial_trajectories', len(trajectory_manifest), 1237, len(trajectory_manifest) == 1237), ('continuous_geometry_participants', trajectory_manifest.subject_id.nunique(), 31, trajectory_manifest.subject_id.nunique() == 31), ('continuous_geometry_features', len(TRAJECTORY_GEOMETRY_FEATURES), 6, len(TRAJECTORY_GEOMETRY_FEATURES) == 6), ('reports_outcomes_context_columns_loaded', 0, 0, True), ('alignment_models_fitted_in_cross_person_alignment_design', 0, 0, True)]
    return pd.DataFrame(rows, columns=['audit_item', 'observed', 'expected', 'passes'])

def write_report(path: Path, summary: dict[str, Any], selection: pd.DataFrame) -> None:
    stable = selection[selection.stable_discrete_solution.map(bval)].copy()
    stable['selected_k'] = stable.selected_k.astype(int)
    kcounts = stable.selected_k.value_counts().sort_index().to_dict()
    lines = ['# Phase cross_person_alignment_design — outcome-free cross-person alignment analysis specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f'- prespecified alignment-policy SHA-256: `{POLICY_SHA}`', '- Cross-person alignment models fitted: **no**', '- Reports/outcomes/context/QC values read: **no**', '- Validation/confirmation opened: **no**', '', '## Verified latent_state_fit inputs', '', f"- Discovery participants represented: **{summary['counts']['all_participants']}**", f"- Stable discrete-state participants: **{summary['counts']['stable_discrete_participants']}**", f"- Continuous-only participants: **{summary['counts']['continuous_only_participants']}**", f"- fixed local-state objects: **{summary['counts']['stable_local_state_objects']}**", f"- fixed continuous trajectories: **{summary['counts']['continuous_trajectories']:,}**", f'- Selected local K distribution: `{kcounts}`', '', '## prespecified branch 1: stable local-state alignment', '', '- Uses only fixed local-state centroids in the common 169-cell outcome-free axes.', '- Every participant contributes total weight one; local states are weighted by their prespecified occupancy.', f'- Candidate shared state-family counts: `{STATE_FAMILY_COUNTS}`.', '- Assignment is injective within participant, so two local states from one participant cannot collapse into the same shared family.', f'- Entire participant-balanced robust scaling/PCA/alignment chain is refit in {STABILITY_REPEATS} participant resamples.', '- If no candidate is stable, no shared discrete family is forced.', '', '## prespecified branch 2: continuous trajectory-geometry alignment', '', '- Includes all 31 participants, including the 12 without stable discrete states.', f'- Uses only six rotation/sign-invariant path features: `{TRAJECTORY_GEOMETRY_FEATURES}`.', '- Each participant contributes total weight one regardless of trial count.', f'- Candidate shared trajectory-family counts: `{TRAJECTORY_FAMILY_COUNTS}`.', f'- Entire two-level participant/trial resampling chain is refit in {STABILITY_REPEATS} resamples.', '- If no candidate is stable, the continuous six-dimensional geometry is retained without forced families.', '', '## Phase boundary', '', '- cross_person_alignment_fit may execute only the prespecified outcome-free alignment policy.', '- Local latent_state_fit K values and assignments cannot be changed.', '- Epoch family, QC, stimulus identity, context, participant-label targets, and subjective reports cannot be alignment features.', '- All cross_person_alignment_fit alignment outputs must be SHA-256 fixed before post-hoc report/context mapping.', '- Validation and confirmation remain closed.', '', 'A shared family is a reproducible alignment of this representation, not a universal emotion category.', '']
    path.write_text(''.join(lines), encoding='utf-8')

def self_test() -> None:
    assert csha(ALIGNMENT_POLICY) == POLICY_SHA
    assert min(STATE_FAMILY_COUNTS) >= 3
    assert TRAJECTORY_GEOMETRY_FEATURES == ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
    sel = pd.DataFrame([{'subject_id': 's1', 'trial_count': 30, 'epoch_observations': 90, 'pca_components': 8, 'stable_discrete_solution': True, 'selected_k': 2, 'selection_reason': 'x'}, {'subject_id': 's2', 'trial_count': 30, 'epoch_observations': 90, 'pca_components': 8, 'stable_discrete_solution': True, 'selected_k': 3, 'selection_reason': 'x'}])
    w = pd.DataFrame({'subject_id': ['s1', 's1', 's2', 's2', 's2'], 'p': [0.4, 0.6, 0.2, 0.3, 0.5]})
    assert np.allclose(w.groupby('subject_id').p.sum().to_numpy(), 1.0)
    x = np.array([0.2, 0.8])
    assert np.all(np.isfinite(np.log1p(x)))
    assert np.all(np.isfinite(np.arctanh(np.clip(np.array([-0.9, 0.9]), -1 + 1e-06, 1 - 1e-06))))
    assert ALIGNMENT_POLICY['forbidden'][0].startswith('reading valence')
    print('cross_person_alignment_design self-test: PASS')

def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    paths = {k: project / v for k, v in REL.items()}
    required(paths.values())
    out = output_paths(project)
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
    note('Verifying latent_state_fit completion, authorization, and every primary SHA-256 lock.')
    summary_b, contract_b, next_b, lock_b, rehash = verify_latent_state_fit(project, paths)
    rehash.to_csv(out['latent_state_fit_rehash'], index=False)
    note('Loading only fixed outcome-free state-centroid and invariant trajectory-geometry columns.')
    selection = pd.read_csv(paths['selection'], low_memory=False)
    centroids = pd.read_csv(paths['centroids_common'], low_memory=False)
    centroids_pca = pd.read_csv(paths['centroids_pca'], low_memory=False)
    assignments = pd.read_csv(paths['assignments'], usecols=['subject_id', 'raw_behavior_row_id', 'epoch_family', 'selected_k', 'state_1based'], low_memory=False)
    continuous = pd.read_csv(paths['continuous_trajectories'], usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', 'pca_components', *TRAJECTORY_GEOMETRY_FEATURES], low_memory=False)
    note('Auditing stable local-state objects and participant-balanced state weights.')
    state_manifest = build_state_manifest(selection, centroids, centroids_pca, assignments)
    state_manifest.to_csv(out['state_manifest'], index=False)
    note('Auditing all-participant invariant trajectory geometry and participant-balanced trial weights.')
    trajectory_manifest = build_trajectory_manifest(continuous, selection)
    trajectory_manifest.to_csv(out['trajectory_manifest'], index=False)
    schema = feature_schema(centroids)
    schema.to_csv(out['feature_schema'], index=False)
    audit = build_input_audit(selection, state_manifest, trajectory_manifest, centroids)
    if not audit.passes.all():
        raise RuntimeError('One or more cross_person_alignment_design input-audit checks failed')
    audit.to_csv(out['input_audit'], index=False)
    atom_json(out['alignment_policy'], ALIGNMENT_POLICY)
    stable_count = int(selection.stable_discrete_solution.map(bval).sum())
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'outcome_free_cross_person_alignment_design_prespecified_candidate', 'alignment_policy_sha256': POLICY_SHA, 'counts': {'latent_state_fit_primary_files_rehashed': len(rehash), 'latent_state_fit_primary_bytes_rehashed': int(rehash.size_bytes.sum()), 'all_participants': int(selection.subject_id.nunique()), 'stable_discrete_participants': stable_count, 'continuous_only_participants': int(len(selection) - stable_count), 'stable_local_state_objects': len(state_manifest), 'common_axis_centroid_rows': len(centroids), 'common_axis_features': int(centroids.compact_cell_0based.nunique()), 'continuous_trajectories': len(trajectory_manifest), 'continuous_geometry_features': len(TRAJECTORY_GEOMETRY_FEATURES)}, 'selected_local_k_counts': {str(int(k)): int(v) for k, v in selection.loc[selection.stable_discrete_solution.map(bval), 'selected_k'].astype(int).value_counts().sort_index().items()}, 'decision': {'latent_state_fit_verified_complete': True, 'all_latent_state_fit_primary_hashes_verified': True, 'reports_outcomes_context_columns_read': False, 'qc_values_read': False, 'cross_person_alignment_models_fitted': False, 'shared_family_count_selected': False, 'continuous_only_participants_represented_without_pseudo_states': True, 'cross_person_alignment_fit_execution_design_prespecified': True, 'validation_opened': False, 'confirmation_opened': False}}
    atom_json(out['summary'], summary)
    output_hashes = {}
    for key, p in out.items():
        if key not in {'contract', 'next_policy', 'report', 'log'} and p.is_file():
            output_hashes[key] = sha(p)
    contract = {**summary, 'alignment_policy': ALIGNMENT_POLICY, 'input_hashes': {k: sha(p) for k, p in paths.items()}, 'latent_state_fit_contract_sha256': sha(paths['contract']), 'latent_state_fit_next_policy_sha256': sha(paths['next_policy']), 'latent_state_fit_primary_lock_sha256': sha(paths['primary_lock']), 'latent_state_fit_primary_hashes': lock_b['hashes'], 'output_hashes': output_hashes, 'outcomes_modified': False, 'latent_state_fit_outputs_modified': False, 'feature_store_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'execute only the prespecified outcome-free cross-person state-centroid and continuous-trajectory alignment models', 'alignment_policy_sha256': POLICY_SHA, 'required_inputs': {'cross_person_alignment_design_contract': str(out['contract']), 'prespecified_alignment_policy': str(out['alignment_policy']), 'stable_local_state_manifest': str(out['state_manifest']), 'continuous_geometry_manifest': str(out['trajectory_manifest']), 'alignment_feature_schema': str(out['feature_schema']), 'latent_state_fit_primary_lock': str(paths['primary_lock']), 'latent_state_fit_state_centroids_common_axes': str(paths['centroids_common']), 'latent_state_fit_continuous_trajectories': str(paths['continuous_trajectories']), 'latent_state_fit_state_assignments': str(paths['assignments'])}, 'required_execution': ['fit every prespecified candidate shared state-family count with participant-balanced constrained alignment', 'fit every prespecified candidate shared trajectory-family count with participant-balanced geometry alignment', 'execute every prespecified participant-level and two-level stability resample', 'apply prespecified collapse, silhouette-parsimony, no-force, and canonical-label rules', 'execute leave-one-participant-out diagnostics without revising selections', 'write and SHA-256 lock both alignment branches before any report/context mapping', 'execute only the prespecified outcome-free cross-branch linkage after both branch locks'], 'forbidden': ALIGNMENT_POLICY['forbidden'], 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['next_policy'], next_policy)
    write_report(out['report'], summary, selection)
    note('cross_person_alignment_design complete. Alignment design is prespecified; no cross-person model was fitted.')
    print('' + '=' * 104)
    print('PHASE cross_person_alignment_design CROSS-PERSON ALIGNMENT analysis specification COMPLETED')
    print('=' * 104)
    print(f'Alignment-policy SHA-256: {POLICY_SHA}')
    print(f'Stable discrete participants: {stable_count}')
    print(f'fixed local-state objects: {len(state_manifest)}')
    print(f'All-participant continuous trajectories: {len(trajectory_manifest)}')
    print('Alignment models fitted: False')
    print('Reports/outcomes/context/QC read: False')
    print('Validation opened: False')
    print('Confirmation opened: False')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    raise SystemExit(main())