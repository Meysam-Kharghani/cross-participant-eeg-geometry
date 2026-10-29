# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase mapping_design: specification post-hoc Discovery-only mapping design.

This phase is deliberately a design/attachment specification, not an association phase.
It verifies the fixed cross_person_alignment_fit outcome-free alignment, writes and hashes the full
mapping policy *before* opening report/context/stimulus values, then creates
immutable trial- and local-state-level mapping manifests for mapping_inference.

cross_person_alignment_fit family counts, embeddings, assignments, preprocessing, and labels are
never refit or relabeled here. Validation and Confirmation remain closed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'mapping_design'
SOURCE_PHASE = 'cross_person_alignment_fit'
NEXT_PHASE = 'mapping_inference'
SEED = 540
PERMUTATION_REPEATS_GLOBAL = 10000
PERMUTATION_REPEATS_CV = 1000
BOOTSTRAP_REPEATS = 5000
ALPHA = 0.05
RIDGE_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
REPORT_COLUMNS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
CORE_REPORTS = ['valence', 'liking']
COMPONENT_REPORTS = [c for c in REPORT_COLUMNS if c not in CORE_REPORTS]
GEOMETRY_COLUMNS = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']
STATE_PC_COLUMNS = [f'aligned_state_pc{i}' for i in range(1, 9)]
CATEGORICAL_MAPPING_COLUMNS = ['context', 'emotion_category', 'stimulus_id']
EXPECTED_CORRECTED_MASTER_SHA = '79aac098573fada34c2dd1247e7ce91e82633130f43cfdea82e2f59b41445039'
EXPECTED_primary_analysis_reference_TRIAL_OUTCOME_SHA = 'a3cf2292d31749b96a0e3729bb5860d69bb642219ce6e478f4f794fb48c832ec'

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

def require(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Missing prerequisite(s):\n  ' + '\n  '.join(missing))

def bval(x: Any) -> bool:
    return bool(x) and str(x).strip().lower() not in {'false', '0', 'none', 'nan', ''}

def output_paths(project: Path) -> dict[str, Path]:
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    c = project / 'config'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'log': l / 'mapping_design_mapping_design_specification.log', 'policy': project / 'analysis_specifications/S041_mapping.json', 'policy_lock': q / 'mapping_design_policy_lock_hashes.json', 'cross_person_alignment_fit_rehash': q / 'mapping_design_cross_person_alignment_fit_final_rehash.csv', 'source_audit': q / 'mapping_design_source_attachment_audit.csv', 'variable_schema': t / 'mapping_design_mapping_variable_schema.csv', 'trial_manifest': t / 'mapping_design_trial_mapping_manifest.csv', 'state_manifest': t / 'mapping_design_state_mapping_manifest.csv', 'summary': q / 'mapping_design_mapping_design_specification_summary.json', 'contract': c / 'mapping_design_mapping_design_contract.json', 'next_policy': project / 'analysis_specifications/S040_mapping_inference.json', 'final_lock': q / 'mapping_design_final_output_lock_hashes.json', 'report': r / 'mapping_design_mapping_design_specification_report.md'}

def input_paths(project: Path) -> dict[str, Path]:
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    c = project / 'config'
    return {'cross_person_alignment_fit_contract': c / 'cross_person_alignment_fit_cross_person_alignment_contract.json', 'cross_person_alignment_fit_next_policy': project / 'analysis_specifications/S013_cross_person_alignment_fit_mapping.json', 'cross_person_alignment_fit_summary': q / 'cross_person_alignment_fit_cross_person_alignment_summary.json', 'cross_person_alignment_fit_final_lock': q / 'cross_person_alignment_fit_final_output_lock_hashes.json', 'state_preprocessing': t / 'cross_person_alignment_fit_state_alignment_preprocessing.csv', 'state_pca_loadings': t / 'cross_person_alignment_fit_state_alignment_pca_loadings.csv', 'state_embedding': t / 'cross_person_alignment_fit_state_alignment_embedding.csv', 'state_pairwise': t / 'cross_person_alignment_fit_state_pairwise_distances.csv', 'state_candidates': t / 'cross_person_alignment_fit_state_family_candidate_metrics.csv', 'state_candidate_assignments': t / 'cross_person_alignment_fit_state_family_candidate_assignments.csv', 'state_candidate_prototypes': t / 'cross_person_alignment_fit_state_family_candidate_prototypes.csv', 'state_stability': t / 'cross_person_alignment_fit_state_family_stability_replicates.csv', 'state_selection': t / 'cross_person_alignment_fit_state_family_selection.csv', 'state_selected_assignments': t / 'cross_person_alignment_fit_state_family_assignments.csv', 'state_selected_prototypes': t / 'cross_person_alignment_fit_state_family_prototypes.csv', 'state_lopo': t / 'cross_person_alignment_fit_state_family_lopo_diagnostics.csv', 'trajectory_preprocessing': t / 'cross_person_alignment_fit_trajectory_preprocessing.csv', 'trajectory_standardized': t / 'cross_person_alignment_fit_trajectory_standardized_geometry.csv', 'trajectory_candidates': t / 'cross_person_alignment_fit_trajectory_family_candidate_metrics.csv', 'trajectory_candidate_assignments': t / 'cross_person_alignment_fit_trajectory_family_candidate_assignments.csv', 'trajectory_candidate_centroids': t / 'cross_person_alignment_fit_trajectory_family_candidate_centroids.csv', 'trajectory_stability': t / 'cross_person_alignment_fit_trajectory_family_stability_replicates.csv', 'trajectory_selection': t / 'cross_person_alignment_fit_trajectory_family_selection.csv', 'trajectory_selected_assignments': t / 'cross_person_alignment_fit_trajectory_family_assignments.csv', 'trajectory_selected_centroids': t / 'cross_person_alignment_fit_trajectory_family_centroids.csv', 'trajectory_lopo': t / 'cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv', 'linkage_rows': t / 'cross_person_alignment_fit_outcome_free_cross_branch_linkage.csv', 'linkage_crosstab': t / 'cross_person_alignment_fit_outcome_free_cross_branch_crosstab.csv', 'linkage_nulls': t / 'cross_person_alignment_fit_outcome_free_cross_branch_null_replicates.csv', 'linkage_tests': t / 'cross_person_alignment_fit_outcome_free_cross_branch_tests.csv', 'latent_state_fit_primary_lock': q / 'latent_state_fit_primary_lock_hashes.json', 'latent_state_fit_state_assignments': t / 'latent_state_fit_state_assignments.csv', 'primary_analysis_reference_contract': c / 'outcome_model_integrity_outcome_model_integrity_contract.json', 'trial_outcomes': t / 'outcome_model_integrity_discovery_trial_outcome_manifest.csv', 'derivative_closure_contract': c / 'derivative_closure_derivative_closure_canonical_master_contract.json', 'canonical_master': t / 'derivative_closure_corrected_discovery_master_v1.csv'}
cross_person_alignment_fit_HASH_PATH_KEYS = ['state_preprocessing', 'state_pca_loadings', 'state_embedding', 'state_pairwise', 'state_candidates', 'state_candidate_assignments', 'state_candidate_prototypes', 'state_stability', 'state_selection', 'state_selected_assignments', 'state_selected_prototypes', 'state_lopo', 'trajectory_preprocessing', 'trajectory_standardized', 'trajectory_candidates', 'trajectory_candidate_assignments', 'trajectory_candidate_centroids', 'trajectory_stability', 'trajectory_selection', 'trajectory_selected_assignments', 'trajectory_selected_centroids', 'trajectory_lopo', 'linkage_rows', 'linkage_crosstab', 'linkage_nulls', 'linkage_tests']
MAPPING_POLICY: dict[str, Any] = {'phase': PHASE, 'seed': SEED, 'scientific_role': 'strictly post-hoc Discovery-only mapping of immutable cross_person_alignment_fit trajectory families/geometry and continuous aligned local-state embeddings to subjective reports, context, and stimulus variables', 'discovery_only': True, 'validation_opened': False, 'confirmation_opened': False, 'immutable_source_boundary': {'source_phase': SOURCE_PHASE, 'alignment_refit_forbidden': True, 'family_count_revision_forbidden': True, 'family_relabeling_from_reports_forbidden': True, 'trajectory_family_labels_remain_geometric_labels_1_and_2': True, 'state_branch_rule': 'cross_person_alignment_fit selected no stable shared discrete state family; use only the fixed continuous aligned-state embedding and never fabricate state families', 'continuous_only_participants_retained': True}, 'mapping_population': {'trial_level': 'all 1,237 fixed cross_person_alignment_fit trajectories from all 31 eligible Discovery participants', 'state_level': 'all local states from the 19 participants with stable latent_state_fit within-person states; report values are attached to epoch observations only after policy lock', 'trial_to_state_weight': 'each of the three epoch observations in a trial receives weight 1/3 so every trial contributes total weight one'}, 'subjective_report_vector': {'ordered_columns': REPORT_COLUMNS, 'core_reports': CORE_REPORTS, 'component_reports': COMPONENT_REPORTS, 'interpretation': 'dimensions overlap but are not interchangeable; no single valence label substitutes for the full profile'}, 'nuisance_adjustment': {'primary_report_residualization': 'fit report ~ participant fixed effect + context + acquisition_fraction + acquisition_fraction_squared without any family/geometry/state predictor; use fold-contained fits for cross-validation and full-Discovery fits only for within-participant family contrasts', 'raw_descriptives': 'also report unadjusted family means and participant-level contrasts, explicitly labeled descriptive', 'emotion_category_not_a_nuisance_covariate': 'emotion_category is a mapping target and is never used to residualize reports'}, 'global_hypothesis_family': {'H1_trajectory_family_report_profile': {'predictor': 'fixed two-family trajectory assignment', 'outcome': '10-dimensional nuisance-adjusted report vector', 'statistic': 'shrinkage Hotelling-style quadratic statistic on the 31 participant-specific family-2 minus family-1 contrast vectors', 'null': f'{PERMUTATION_REPEATS_GLOBAL} joint participant sign flips; the full 10-dimensional vector is flipped together', 'effect_reporting': 'participant-bootstrap confidence intervals for the global norm and every report contrast'}, 'H2_continuous_geometry_incremental_report_mapping': {'predictor': GEOMETRY_COLUMNS, 'baseline': 'context + acquisition_fraction + acquisition_fraction_squared', 'full_model': 'baseline plus all six fixed standardized trajectory-geometry coordinates', 'estimator': 'multivariate ridge; alpha selected only inside training participants from the prespecified ridge grid', 'outer_validation': 'fixed cross_person_fold, with no test-participant report calibration', 'global_metric': 'multivariate out-of-fold SSE improvement and variance-weighted Delta-R2 versus baseline', 'null': f'{PERMUTATION_REPEATS_CV} within-participant permutations of the entire 10-dimensional report row vector, preserving report covariance', 'bootstrap': f'{BOOTSTRAP_REPEATS} participant bootstrap replicates'}, 'H3_continuous_state_embedding_report_mapping': {'predictor': STATE_PC_COLUMNS, 'state_profile_construction': 'aggregate nuisance-adjusted trial report vectors to each local state with epoch weight 1/3, then occupancy-weight center state profiles within participant', 'estimator': 'leave-one-participant-out multivariate ridge from the fixed aligned-state PCs to the 10-dimensional state report profile', 'baseline': 'zero after within-participant weighted centering', 'global_metric': 'held-out multivariate SSE improvement and variance-weighted Delta-R2', 'null': f'{PERMUTATION_REPEATS_GLOBAL} within-participant permutations of complete state-profile vectors across that participant local states', 'no_discrete_state_claim': True}, 'H4_trajectory_family_context': {'predictor': 'fixed trajectory family', 'target': 'context', 'statistic': 'participant-balanced normalized mutual information and Cramers V', 'null': f'{PERMUTATION_REPEATS_GLOBAL} within-participant family-label permutations preserving each participant family counts'}, 'H5_trajectory_family_emotion_category': {'analysis_set': 'all fixed trajectories with prespecified nonmissing emotion_category; no imputation and no exclusion from other hypotheses', 'predictor': 'fixed trajectory family', 'target': 'emotion_category', 'statistic': 'participant-balanced normalized mutual information and Cramers V', 'null': f'{PERMUTATION_REPEATS_GLOBAL} within-participant family-label permutations'}, 'H6_trajectory_family_stimulus_identity': {'analysis_set': 'all fixed trajectories with prespecified nonmissing stimulus_id; no imputation and no exclusion from other hypotheses', 'predictor': 'fixed trajectory family', 'target': 'stimulus_id', 'statistic': 'participant-balanced normalized mutual information', 'null': f'{PERMUTATION_REPEATS_GLOBAL} within-participant family-label permutations'}}, 'multiplicity_and_followups': {'global_control': 'Holm family-wise correction across H1-H6', 'report_followups': 'for H1, H2, or H3, report all 10 dimension-specific results; claim dimension-specific support only when the parent global hypothesis passes Holm and the 10 child p-values pass Holm within that parent', 'core_reports': 'valence and liking remain explicitly identified but are not exempt from multiplicity control', 'categorical_followups': 'cell residuals and category-specific proportions are descriptive only; no post-hoc category p-value fishing', 'threshold': ALPHA}, 'effect_size_and_uncertainty': {'participant_balance': 'every participant has equal total weight in all trial-level statistics', 'bootstrap_unit': 'participant', 'bootstrap_repeats': BOOTSTRAP_REPEATS, 'interval': 'percentile 95 percent', 'family_contrast_direction': 'family 2 minus family 1; labels are not renamed after mapping', 'report_scale': 'original rating scale plus standardized effect size based on participant-level contrast SD'}, 'execution_order': ['verify cross_person_alignment_fit contract, authorization, branch decisions, and every final output hash', 'write and SHA-256 lock this complete mapping policy before opening reports/context/stimulus values', 'verify the prespecified primary_analysis_reference outcome-manifest hash and derivative_closure corrected layered-master hash', 'attach values one-to-one to fixed cross_person_alignment_fit trial IDs and latent_state_fit local-state epoch assignments', 'write immutable mapping manifests and source/missingness audits without calculating any association', 'mapping_inference may execute only the prespecified H1-H6 analyses and may not revise alignment or mapping design'], 'forbidden': ['opening Validation or Confirmation participants', 'refitting cross_person_alignment_fit alignment after any report/context/stimulus value is opened', 'choosing family count, state count, embedding dimension, report subset, covariates, or test from observed mapping strength', 'renaming family 1/2 as positive, negative, pleasant, unpleasant, engaged, disengaged, or emotion categories before inference', 'dropping continuous-only participants or low-margin trials to improve association', 'globally dropping trials that lack prespecified stimulus identity; those rows remain in report/context hypotheses and are excluded only from the predeclared stimulus-target test', 'using report variables to alter any EEG preprocessing, PCA, GMM, state assignment, trajectory geometry, or family assignment'], 'claim_guardrails': ['trajectory families are geometrical trajectory regimes, not emotion classes', 'a significant report association is post-hoc Discovery evidence and not independent validation', 'a null state-embedding mapping does not erase stable within-person states', 'context or stimulus association may explain part of report association and must be reported alongside it', 'Validation and Confirmation remain required for any generalization claim']}
MAPPING_POLICY_SHA = canonical_sha(MAPPING_POLICY)

def verify_cross_person_alignment_fit(inputs: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    require([inputs['cross_person_alignment_fit_contract'], inputs['cross_person_alignment_fit_next_policy'], inputs['cross_person_alignment_fit_summary'], inputs['cross_person_alignment_fit_final_lock']] + [inputs[k] for k in cross_person_alignment_fit_HASH_PATH_KEYS])
    contract = json.loads(inputs['cross_person_alignment_fit_contract'].read_text())
    next_policy = json.loads(inputs['cross_person_alignment_fit_next_policy'].read_text())
    summary = json.loads(inputs['cross_person_alignment_fit_summary'].read_text())
    lock = json.loads(inputs['cross_person_alignment_fit_final_lock'].read_text())
    if contract.get('phase') != SOURCE_PHASE:
        raise RuntimeError('cross_person_alignment_fit contract phase mismatch')
    if next_policy.get('next_phase') != PHASE or not bval(next_policy.get('next_phase_permitted')):
        raise RuntimeError('cross_person_alignment_fit did not authorize mapping_design')
    if bval(summary.get('decision', {}).get('reports_outcomes_context_qc_read')):
        raise RuntimeError('cross_person_alignment_fit unexpectedly opened reports/context/QC')
    if bval(summary.get('decision', {}).get('validation_opened')) or bval(summary.get('decision', {}).get('confirmation_opened')):
        raise RuntimeError('Validation or Confirmation was opened in cross_person_alignment_fit')
    if summary.get('state_branch', {}).get('stable_discrete_family_solution') is not False:
        raise RuntimeError('Expected no stable shared discrete state family in the prespecified cross_person_alignment_fit result')
    if summary.get('trajectory_branch', {}).get('selected_shared_k') != 2 or summary.get('trajectory_branch', {}).get('stable_discrete_family_solution') is not True:
        raise RuntimeError('Expected the prespecified stable K=2 trajectory-family solution')
    hashes = lock.get('hashes', {})
    rows = []
    for key in cross_person_alignment_fit_HASH_PATH_KEYS:
        expected = hashes.get(key)
        if not expected:
            raise RuntimeError(f'cross_person_alignment_fit final lock lacks hash key: {key}')
        observed = sha256(inputs[key])
        ok = observed == expected
        rows.append({'source_phase': SOURCE_PHASE, 'object': key, 'path': str(inputs[key]), 'expected_sha256': expected, 'observed_sha256': observed, 'match': ok})
        if not ok:
            raise RuntimeError(f'cross_person_alignment_fit hash mismatch: {key}')
    return (summary, lock, pd.DataFrame(rows))

def verify_source_hashes(inputs: dict[str, Path]) -> list[dict[str, Any]]:
    require([inputs['latent_state_fit_primary_lock'], inputs['latent_state_fit_state_assignments'], inputs['primary_analysis_reference_contract'], inputs['trial_outcomes'], inputs['derivative_closure_contract'], inputs['canonical_master']])
    rows: list[dict[str, Any]] = []
    latent_state_fit_lock = json.loads(inputs['latent_state_fit_primary_lock'].read_text())
    latent_state_fit_expected = latent_state_fit_lock.get('hashes', {}).get('assignments')
    latent_state_fit_observed = sha256(inputs['latent_state_fit_state_assignments'])
    if latent_state_fit_expected != latent_state_fit_observed:
        raise RuntimeError('latent_state_fit state-assignment hash mismatch')
    rows.append({'source': 'latent_state_fit_state_assignments', 'path': str(inputs['latent_state_fit_state_assignments']), 'expected_sha256': latent_state_fit_expected, 'observed_sha256': latent_state_fit_observed, 'match': True})
    primary_analysis_reference_contract = json.loads(inputs['primary_analysis_reference_contract'].read_text())
    primary_analysis_reference_expected = primary_analysis_reference_contract.get('output_hashes', {}).get('trial_outcomes')
    if primary_analysis_reference_expected != EXPECTED_primary_analysis_reference_TRIAL_OUTCOME_SHA:
        raise RuntimeError('primary_analysis_reference contract trial-outcome authority differs from prespecified mapping_design expectation')
    primary_analysis_reference_observed = sha256(inputs['trial_outcomes'])
    if primary_analysis_reference_observed != primary_analysis_reference_expected:
        raise RuntimeError('primary_analysis_reference trial-outcome manifest hash mismatch')
    rows.append({'source': 'primary_analysis_reference_discovery_trial_outcomes', 'path': str(inputs['trial_outcomes']), 'expected_sha256': primary_analysis_reference_expected, 'observed_sha256': primary_analysis_reference_observed, 'match': True})
    derivative_closure_contract = json.loads(inputs['derivative_closure_contract'].read_text())
    corrected_expected = derivative_closure_contract.get('output_hashes', {}).get('master')
    if corrected_expected != EXPECTED_CORRECTED_MASTER_SHA:
        raise RuntimeError('derivative_closure corrected-master authority differs from prespecified mapping_design expectation')
    corrected_observed = sha256(inputs['canonical_master'])
    if corrected_observed != corrected_expected:
        raise RuntimeError('derivative_closure corrected layered-master hash mismatch')
    rows.append({'source': 'derivative_closure_corrected_layered_discovery_master', 'path': str(inputs['canonical_master']), 'expected_sha256': corrected_expected, 'observed_sha256': corrected_observed, 'match': True})
    return rows

def load_and_attach(inputs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_assign_cols = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'selected_shared_k', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'trajectory_alignment_weight']
    traj_std_cols = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_alignment_weight'] + GEOMETRY_COLUMNS
    outcome_cols = ['raw_behavior_row_id', 'subject_id', 'split', 'trial_number', *REPORT_COLUMNS, 'report_complete', 'context', 'raw_run', 'within_subject_block_fold', 'cross_person_fold', 'acquisition_fraction']
    corrected_cols = ['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'context_authoritative_for_analysis', 'stimulus_id', 'emotion_category', 'stimulus_authoritative_for_analysis']
    state_assign_cols = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'epoch_family', 'selected_k', 'state_1based']
    state_emb_cols = ['state_object_id', 'subject_id', 'state_1based', 'state_alignment_weight', *STATE_PC_COLUMNS]
    ta = pd.read_csv(inputs['trajectory_selected_assignments'], usecols=traj_assign_cols)
    tg = pd.read_csv(inputs['trajectory_standardized'], usecols=traj_std_cols)
    outcomes = pd.read_csv(inputs['trial_outcomes'], usecols=outcome_cols)
    corrected = pd.read_csv(inputs['canonical_master'], usecols=corrected_cols)
    sa = pd.read_csv(inputs['latent_state_fit_state_assignments'], usecols=state_assign_cols)
    se = pd.read_csv(inputs['state_embedding'], usecols=state_emb_cols)
    for name, df, keys in [('trajectory assignments', ta, ['raw_behavior_row_id']), ('trajectory geometry', tg, ['raw_behavior_row_id']), ('trial outcomes', outcomes, ['raw_behavior_row_id']), ('corrected layered master', corrected, ['raw_behavior_row_id']), ('state embedding', se, ['subject_id', 'state_1based'])]:
        if df.duplicated(keys).any():
            raise RuntimeError(f'Duplicate key in {name}: {keys}')
    trial = ta.merge(tg.drop(columns=['trajectory_alignment_weight']), on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='one_to_one')
    trial = trial.merge(outcomes, on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='one_to_one')
    trial = trial.merge(corrected, on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='one_to_one', suffixes=('', '_corrected'))
    if len(trial) != len(ta):
        raise RuntimeError(f'Trial attachment changed rows: {len(ta)} -> {len(trial)}')
    if not (trial['split'].astype(str) == 'discovery').all():
        raise RuntimeError('Non-Discovery row entered trial mapping manifest')
    if not trial['report_complete'].map(bval).all():
        raise RuntimeError('Incomplete report row entered trial mapping manifest')
    if not trial['context_authoritative_for_analysis'].map(bval).all():
        raise RuntimeError('A cross_person_alignment_fit trial lacks authoritative corrected context')
    if not (trial['context'].astype(str) == trial['context_corrected'].astype(str)).all():
        raise RuntimeError('Context mismatch between primary_analysis_reference and derivative_closure corrected layered master')
    trial['stimulus_mapping_available'] = trial['stimulus_authoritative_for_analysis'].map(bval) & trial['stimulus_id'].notna() & trial['emotion_category'].notna()
    trial = trial.drop(columns=['context_corrected', 'context_authoritative_for_analysis', 'stimulus_authoritative_for_analysis', 'split', 'report_complete'])
    ordered_trial = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'raw_run', 'within_subject_block_fold', 'cross_person_fold', 'acquisition_fraction', 'context', 'stimulus_mapping_available', 'stimulus_id', 'emotion_category', *REPORT_COLUMNS, 'selected_shared_k', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'trajectory_alignment_weight', *GEOMETRY_COLUMNS]
    trial = trial[ordered_trial].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    stable_subjects = set(se['subject_id'].astype(str))
    sa = sa[sa['subject_id'].astype(str).isin(stable_subjects)].copy()
    if sa.empty:
        raise RuntimeError('No stable local-state epoch assignments found')
    state = sa.merge(se, on=['subject_id', 'state_1based'], how='inner', validate='many_to_one')
    trial_meta_cols = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'acquisition_fraction', 'context', 'stimulus_mapping_available', 'stimulus_id', 'emotion_category', *REPORT_COLUMNS]
    state = state.merge(trial[trial_meta_cols], on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='many_to_one')
    if len(state) != len(sa):
        raise RuntimeError(f'State attachment lost rows: {len(sa)} -> {len(state)}')
    state['trial_epoch_weight'] = 1.0 / 3.0
    per_trial_n = state.groupby(['subject_id', 'raw_behavior_row_id']).size()
    if not (per_trial_n == 3).all():
        raise RuntimeError('Every retained state-mapping trial must contribute exactly three epoch observations')
    ordered_state = ['state_object_id', 'subject_id', 'state_1based', 'selected_k', 'raw_behavior_row_id', 'trial_number', 'epoch_family', 'trial_epoch_weight', 'state_alignment_weight', *STATE_PC_COLUMNS, 'acquisition_fraction', 'context', 'stimulus_mapping_available', 'stimulus_id', 'emotion_category', *REPORT_COLUMNS]
    state = state[ordered_state].sort_values(['subject_id', 'state_1based', 'trial_number', 'epoch_family']).reset_index(drop=True)
    schema_rows: list[dict[str, Any]] = []

    def add(vars_: list[str], level: str, role: str, family: str, primary_use: str) -> None:
        for v in vars_:
            schema_rows.append({'variable': v, 'level': level, 'role': role, 'multiplicity_family': family, 'prespecified_primary_use': primary_use})
    add(REPORT_COLUMNS, 'trial_and_state_aggregate', 'subjective_report', 'H1_H2_H3_child_reports', 'joint report vector and dimension-specific gated follow-up')
    add(['trajectory_family_1based'], 'trial', 'fixed_discrete_predictor', 'H1_H4_H5_H6', 'immutable geometric family label')
    add(GEOMETRY_COLUMNS, 'trial', 'fixed_continuous_predictor', 'H2', 'incremental geometry mapping beyond context/order baseline')
    add(STATE_PC_COLUMNS, 'local_state', 'fixed_continuous_predictor', 'H3', 'continuous state embedding only; no discrete shared state family')
    add(CATEGORICAL_MAPPING_COLUMNS, 'trial', 'categorical_mapping_target', 'H4_H5_H6', 'participant-stratified association target; stimulus targets use prespecified target-specific complete cases only')
    add(['stimulus_mapping_available'], 'trial', 'prespecified_availability_flag', 'H5_H6_analysis_set', 'predeclared target-specific complete-case flag; never used to filter report/context analyses')
    add(['acquisition_fraction', 'context'], 'trial', 'prespecified_nuisance_or_target', 'nuisance_and_H4', 'context/order adjustment for report mapping; context also separately tested')
    schema = pd.DataFrame(schema_rows)
    return (trial, state, schema)

def source_audit(trial: pd.DataFrame, state: pd.DataFrame, source_hash_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in source_hash_rows:
        rows.append({'audit_type': 'source_hash', 'object': r['source'], 'value': r['observed_sha256'], 'status': 'PASS', 'detail': r['path']})
    rows.extend([{'audit_type': 'mapping_population', 'object': 'trial_rows', 'value': int(len(trial)), 'status': 'PASS' if len(trial) == 1237 else 'FAIL', 'detail': 'fixed cross_person_alignment_fit trajectories'}, {'audit_type': 'mapping_population', 'object': 'trial_participants', 'value': int(trial['subject_id'].nunique()), 'status': 'PASS' if trial['subject_id'].nunique() == 31 else 'FAIL', 'detail': 'all cross_person_alignment_fit participants'}, {'audit_type': 'mapping_population', 'object': 'trajectory_families', 'value': int(trial['trajectory_family_1based'].nunique()), 'status': 'PASS' if set(trial['trajectory_family_1based']) == {1, 2} else 'FAIL', 'detail': 'prespecified K=2 labels'}, {'audit_type': 'mapping_population', 'object': 'state_epoch_rows', 'value': int(len(state)), 'status': 'PASS', 'detail': 'stable local-state participants only'}, {'audit_type': 'mapping_population', 'object': 'state_participants', 'value': int(state['subject_id'].nunique()), 'status': 'PASS' if state['subject_id'].nunique() == 19 else 'FAIL', 'detail': 'latent_state_fit stable discrete participants'}, {'audit_type': 'mapping_population', 'object': 'local_state_objects', 'value': int(state['state_object_id'].nunique()), 'status': 'PASS' if state['state_object_id'].nunique() == 40 else 'FAIL', 'detail': 'continuous aligned-state objects'}])
    for name, df in [('trial_manifest', trial), ('state_manifest', state)]:
        for col in REPORT_COLUMNS + CATEGORICAL_MAPPING_COLUMNS:
            if col in df.columns:
                miss = int(df[col].isna().sum())
                allowed = 0
                if col in {'stimulus_id', 'emotion_category'}:
                    allowed = int((~df.get('stimulus_mapping_available', pd.Series(False, index=df.index)).fillna(False)).sum())
                status = 'PASS' if miss == allowed else 'FAIL'
                detail = 'no association calculated; stimulus missingness is prespecified target-specific complete-case only' if col in {'stimulus_id', 'emotion_category'} else 'no association calculated'
                rows.append({'audit_type': 'missingness', 'object': f'{name}:{col}', 'value': miss, 'status': status, 'detail': detail})
    out = pd.DataFrame(rows)
    if (out['status'] == 'FAIL').any():
        bad = out[out['status'] == 'FAIL'].to_dict('records')
        raise RuntimeError(f'Source/mapping audit failed: {bad}')
    return out

def write_report(path: Path, summary: dict[str, Any]) -> None:
    c = summary['counts']
    text = f"# Phase mapping_design — prespecified post-hoc mapping design\n\n- Generated: `{summary['generated_at']}`\n- Status: **`{summary['status']}`**\n- prespecified mapping-policy canonical SHA-256: `{summary['mapping_policy_sha256']}`\n- Mapping policy fixed before report/context/stimulus values opened: **yes**\n- Associations, models, p-values, or family relabeling performed: **no**\n- Validation/Confirmation opened: **no**\n\n## prespecified mapping populations\n\n- Trial-level trajectories: **{c['trial_rows']:,}** from **{c['trial_participants']}** participants.\n- fixed trajectory families: **2**; labels remain geometric family 1 and family 2.\n- Local-state epoch observations: **{c['state_epoch_rows']:,}**.\n- Continuous aligned local-state objects: **{c['local_state_objects']}** from **{c['state_participants']}** participants.\n- Shared discrete state families: **none forced**, exactly as decided in cross_person_alignment_fit.\n\n## prespecified global hypotheses\n\n1. H1: two-family trajectory assignment and the nuisance-adjusted 10-dimensional report profile.\n2. H2: incremental cross-person report mapping from six continuous trajectory-geometry coordinates beyond context/order.\n3. H3: continuous aligned local-state embedding and state-aggregated report profiles; no discrete state-family claim.\n4. H4: trajectory family and context.\n5. H5: trajectory family and stimulus emotion category.\n6. H6: trajectory family and exact stimulus identity.\n\nHolm family-wise correction is prespecified across H1-H6. Dimension-specific report claims are gated by their parent global test and Holm correction across all ten reports.\n\n## Guardrails\n\n- cross_person_alignment_fit preprocessing, embeddings, family count, and assignments are immutable.\n- Family 1/2 may not be renamed from observed reports.\n- Continuous-only participants remain in every trajectory analysis.\n- This is Discovery-only post-hoc mapping; it is not validation."
    atomic_text(path, text)

def self_test() -> int:
    assert len(REPORT_COLUMNS) == 10
    assert MAPPING_POLICY_SHA == canonical_sha(MAPPING_POLICY)
    assert set(CORE_REPORTS).issubset(REPORT_COLUMNS)
    ta = pd.DataFrame({'subject_id': ['s'], 'raw_behavior_row_id': ['s|trial1'], 'trial_number': [1], 'trajectory_family_1based': [1]})
    assert not ta.duplicated(['raw_behavior_row_id']).any()
    ep = pd.DataFrame({'subject_id': ['s'] * 3, 'raw_behavior_row_id': ['s|trial1'] * 3, 'epoch_family': ['stimulus_early', 'stimulus_late', 'rating_early']})
    assert (ep.groupby(['subject_id', 'raw_behavior_row_id']).size() == 3).all()
    print('mapping_design self-test: PASS')
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
    inputs = input_paths(project)
    outputs = output_paths(project)
    if outputs['contract'].exists() and (not args.force):
        raise RuntimeError(f"Output exists; rerun with --force: {outputs['contract']}")
    log_lines: list[str] = []

    def note(msg: str) -> None:
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        log_lines.append(line)
        atomic_text(outputs['log'], ''.join(log_lines) + '')
    note('Verifying cross_person_alignment_fit authorization, branch decisions, and every final output hash without opening reports/context/stimulus values.')
    cross_person_alignment_fit_summary, cross_person_alignment_fit_lock, rehash = verify_cross_person_alignment_fit(inputs)
    atomic_csv(outputs['cross_person_alignment_fit_rehash'], rehash)
    note('Writing and locking the complete mapping_design mapping policy before opening any report/context/stimulus values.')
    atomic_json(outputs['policy'], MAPPING_POLICY)
    if canonical_sha(json.loads(outputs['policy'].read_text())) != MAPPING_POLICY_SHA:
        raise RuntimeError('Written mapping policy canonical SHA mismatch')
    policy_lock = {'generated_at': now(), 'phase': PHASE, 'mapping_policy_canonical_sha256': MAPPING_POLICY_SHA, 'mapping_policy_file_sha256': sha256(outputs['policy']), 'reports_context_stimulus_opened_before_lock': False, 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['policy_lock'], policy_lock)
    note('Policy lock complete. Verifying prespecified report and canonical-stimulus source hashes.')
    source_hash_rows = verify_source_hashes(inputs)
    note('Attaching reports/context/stimulus values one-to-one to immutable trajectory IDs and local-state epoch assignments; no association is calculated.')
    trial, state, schema = load_and_attach(inputs)
    audit = source_audit(trial, state, source_hash_rows)
    atomic_csv(outputs['trial_manifest'], trial)
    atomic_csv(outputs['state_manifest'], state)
    atomic_csv(outputs['variable_schema'], schema)
    atomic_csv(outputs['source_audit'], audit)
    counts = {'trial_rows': int(len(trial)), 'trial_participants': int(trial['subject_id'].nunique()), 'trajectory_families': int(trial['trajectory_family_1based'].nunique()), 'trajectory_family_1_rows': int((trial['trajectory_family_1based'] == 1).sum()), 'trajectory_family_2_rows': int((trial['trajectory_family_1based'] == 2).sum()), 'state_epoch_rows': int(len(state)), 'state_participants': int(state['subject_id'].nunique()), 'local_state_objects': int(state['state_object_id'].nunique()), 'report_variables': len(REPORT_COLUMNS), 'categorical_mapping_targets': len(CATEGORICAL_MAPPING_COLUMNS), 'stimulus_mapped_trial_rows': int(trial['stimulus_mapping_available'].sum()), 'stimulus_unmapped_trial_rows': int((~trial['stimulus_mapping_available']).sum()), 'global_hypotheses': 6}
    decision = {'cross_person_alignment_fit_verified_complete': True, 'all_cross_person_alignment_fit_final_hashes_verified': bool(rehash['match'].all()), 'mapping_policy_written_and_fixed_before_report_context_stimulus_open': True, 'primary_analysis_reference_trial_outcome_hash_verified': True, 'derivative_closure_canonical_master_hash_verified': True, 'trial_mapping_attachment_exact': True, 'state_mapping_attachment_exact': True, 'associations_calculated': False, 'mapping_models_fitted': False, 'p_values_calculated': False, 'family_count_or_assignment_revised': False, 'family_labels_renamed': False, 'validation_opened': False, 'confirmation_opened': False, 'mapping_inference_execution_permitted': True}
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'posthoc_discovery_mapping_design_and_attachment_prespecified_candidate', 'mapping_policy_sha256': MAPPING_POLICY_SHA, 'source_alignment_policy_sha256': cross_person_alignment_fit_summary.get('alignment_policy_sha256'), 'counts': counts, 'source_branch_decisions': {'state_stable_discrete_family_solution': False, 'trajectory_selected_shared_k': 2, 'trajectory_lopo_agreement': cross_person_alignment_fit_summary.get('trajectory_branch', {}).get('lopo_agreement')}, 'decision': decision}
    atomic_json(outputs['summary'], summary)
    output_hashes = {'mapping_policy': sha256(outputs['policy']), 'policy_lock': sha256(outputs['policy_lock']), 'cross_person_alignment_fit_rehash': sha256(outputs['cross_person_alignment_fit_rehash']), 'source_audit': sha256(outputs['source_audit']), 'variable_schema': sha256(outputs['variable_schema']), 'trial_manifest': sha256(outputs['trial_manifest']), 'state_manifest': sha256(outputs['state_manifest']), 'summary': sha256(outputs['summary'])}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': 'posthoc_discovery_mapping_design_and_attachment_prespecified_candidate', 'mapping_policy_sha256': MAPPING_POLICY_SHA, 'source_cross_person_alignment_fit_final_lock_sha256': sha256(inputs['cross_person_alignment_fit_final_lock']), 'source_hashes': {r['source']: r['observed_sha256'] for r in source_hash_rows}, 'counts': counts, 'decision': decision, 'output_hashes': output_hashes, 'source_dataset_modified': False, 'cross_person_alignment_fit_outputs_modified': False}
    atomic_json(outputs['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'execute exactly the prespecified H1-H6 Discovery-only post-hoc mapping analyses with participant-balanced inference, fixed multiplicity control, and no alignment revision', 'mapping_policy_sha256': MAPPING_POLICY_SHA, 'required_inputs': {'mapping_design_contract': str(outputs['contract']), 'mapping_design_final_lock': str(outputs['final_lock']), 'mapping_policy': str(outputs['policy']), 'trial_mapping_manifest': str(outputs['trial_manifest']), 'state_mapping_manifest': str(outputs['state_manifest']), 'variable_schema': str(outputs['variable_schema'])}, 'required_boundary': ['verify every mapping_design output hash before analysis', 'execute all six global hypotheses regardless of observed direction', 'apply Holm across H1-H6 and gated Holm across all ten report dimensions', 'retain geometric family labels 1 and 2 and all continuous-only participants', 'state branch remains continuous only because cross_person_alignment_fit selected no stable shared discrete state family', 'Discovery only; Validation and Confirmation remain closed'], 'forbidden': MAPPING_POLICY['forbidden'], 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['next_policy'], next_policy)
    write_report(outputs['report'], summary)
    final_hashes = {**output_hashes, 'contract': sha256(outputs['contract']), 'next_policy': sha256(outputs['next_policy']), 'report': sha256(outputs['report'])}
    final_lock = {'generated_at': now(), 'phase': PHASE, 'mapping_policy_sha256': MAPPING_POLICY_SHA, 'reports_context_stimulus_opened_only_after_policy_lock': True, 'associations_calculated': False, 'validation_opened': False, 'confirmation_opened': False, 'hashes': final_hashes}
    atomic_json(outputs['final_lock'], final_lock)
    note('mapping_design design/attachment specification complete; all outputs fixed. No association or model was calculated.')
    print(f'Mapping policy canonical SHA-256: {MAPPING_POLICY_SHA}')
    print(f'Trial rows: {len(trial):,}; participants: {trial.subject_id.nunique()}')
    print(f'State epoch rows: {len(state):,}; local state objects: {state.state_object_id.nunique()}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())