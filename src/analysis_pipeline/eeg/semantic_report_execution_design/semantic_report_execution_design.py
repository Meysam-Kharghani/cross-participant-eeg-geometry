# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
PHASE = 'semantic_report_execution_design'
NEXT_PHASE = 'discovery_semantic_report_execution'
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 540
EXPERIENCE_K = [2, 3, 4]
TRANSITION_K = [2, 3]
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
FULL_CATEGORIES = ['sadness', 'disgust', 'fear', 'neutral', 'joy', 'tenderness', 'inspiration']
CATEGORY_PREFIX_MAP = {'sad': 'sadness', 'dis': 'disgust', 'fear': 'fear', 'neu': 'neutral', 'joy': 'joy', 'ten': 'tenderness', 'ins': 'inspiration'}
SIGNATURE_COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
STD_SIGNATURE_COLUMNS = [f'standardized__{c}' for c in SIGNATURE_COORDS]
cross_split_report_transition_structure_synthesis_OUT = Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis')
cross_split_report_transition_structure_synthesis_LOCK = cross_split_report_transition_structure_synthesis_OUT / 'qc/cross_split_report_transition_structure_synthesis_final_output_lock_hashes.json'
cross_split_report_transition_structure_synthesis_LOCK_SHA256 = '4a2131505761db9a64e888294eb9daf58b30797da42fe486ef9c4ec52d1c5439'
cross_split_report_transition_structure_synthesis_SUMMARY = cross_split_report_transition_structure_synthesis_OUT / 'qc/cross_split_report_transition_structure_synthesis_summary.json'
cross_split_report_transition_structure_synthesis_SUMMARY_SHA256 = 'a4b0e4be66789bbe40d8ddad8ccb533bf06e30513e477f68e15fa8036d93703b'
cross_split_report_transition_structure_synthesis_NEXT_POLICY = Path('analysis_specifications/S015_cross_split_report_transition_structure_synthesis.json')
cross_split_report_transition_structure_synthesis_NEXT_POLICY_SHA256 = '97bbe572ea174bcc993acc0d66044edf4decd82f2461ae0174aafb49d881221d'
semantic_report_dissociation_design_SEMANTIC_POLICY = Path('analysis_specifications/S061_semantic_report_dissociation_future.json')
semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256 = 'dbf9fbca05f50d58c3ce561fa73a0c66edc12219ba6f745360c446ef3762723d'
STRUCTURAL_REFS = [('experience_assignments', 'Discovery', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'), '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'), ('experience_assignments', 'Validation', Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv'), 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'), ('experience_assignments', 'Confirmation', Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'), '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'), ('experience_signature', 'Discovery', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'), 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'), ('experience_signature', 'Validation', Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv'), '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'), ('experience_signature', 'Confirmation', Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'), '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'), ('transition_assignments', 'Discovery', Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'), '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'), ('transition_assignments', 'Validation', Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv'), '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'), ('transition_assignments', 'Confirmation', Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'), 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'), ('transition_signature', 'Discovery', Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv'), '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83'), ('transition_signature', 'Validation', Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv'), 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067'), ('transition_signature', 'Confirmation', Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv'), '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6')]
ANALYSIS_specification = Path('config/analysis_specification.v1.json')
ANALYSIS_specification_SHA256 = 'cfd175b2ef3de731355d7378650ffa1b5d31a06171f4600a379c504fe2a85dfa'
SPLIT_MANIFEST = Path('config/participant_split_manifest.v1.csv')
SPLIT_MANIFEST_SHA256 = '0f6156e2a0de5af59dd3a7cd15b2a4a914dc0a0540b9d941fa6676a697f8d453'
CANONICAL_TRIAL_TABLE = Path('data/canonical_behavioral_table.csv')
CANONICAL_TRIAL_TABLE_SHA256 = 'e31f96f918e806d466fd2bccc85b94d5c9dc96002bf97a97a442e8059a0365ed'
DISCOVERY_CORRECTED_MASTER = Path('outputs/tables/derivative_closure_corrected_discovery_master_v1.csv')
DISCOVERY_CORRECTED_MASTER_SHA256 = '79aac098573fada34c2dd1247e7ce91e82633130f43cfdea82e2f59b41445039'
DISCOVERY_OUTCOME_MANIFEST = Path('outputs/tables/outcome_model_integrity_discovery_trial_outcome_manifest.csv')
DISCOVERY_OUTCOME_MANIFEST_SHA256 = 'a3cf2292d31749b96a0e3729bb5860d69bb642219ce6e478f4f794fb48c832ec'
VAL_TIMING = Path('outputs/tables/validation_source_epoch_lineage_validation_authoritative_trial_timing_manifest.csv')
VAL_TIMING_SHA256 = '08a5f19fefb0b7371592f7d9ab796774168a05b64295bfa8019a8cdf1338d145'
VAL_LINEAGE_LOCK = Path('outputs/qc/validation_source_epoch_lineage_final_output_lock_hashes.json')
VAL_LINEAGE_LOCK_SHA256 = '5d8ab8375fcbe90995efcbeb7b3cb707b3c7efc779d38ffc62557942f88c0e31'
CONF_TIMING = Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_authoritative_trial_timing_manifest.csv')
CONF_TIMING_SHA256 = 'dcb34cdd525faffb9fc38d865589d598d436a8429f225554290ae5c731dd693f'
CONF_LINEAGE_LOCK = Path('outputs/qc/confirmation_source_epoch_lineage_final_output_lock_hashes.json')
CONF_LINEAGE_LOCK_SHA256 = 'd502dbc7eb5557d33a6e8e23f54127271b136180100704de477b5951b5e91f82'
README_SHA256 = '379cfc97fce08b4c54d3dc26e95c35d504ca130666bbcf02eff59638f0ff7e3a'
README_RELATIVE = Path('README.md')
README_SEQUENCE_CORE = Path('src/analysis_pipeline/data_preparation/event_grammar_reconstruction.py')
README_SEQUENCE_CORE_SHA256 = '39b690bdbd692cd5a8df6f370e7ad96ddff135ba9f96e6817c6e7e77c23474ac'
OUT = Path('outputs/interpretability/semantic_report_execution_design_semantic_report_execution_design_specification')
PROGRAM_POLICY = Path('analysis_specifications/S063_semantic_report.json')
NEXT_POLICY = Path('analysis_specifications/S062_semantic_report_discovery.json')

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

def verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')
    return got

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')
    os.replace(tmp, path)

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)

def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def verify_parent_lock(project: Path) -> list[dict[str, Any]]:
    lock_path = project / cross_split_report_transition_structure_synthesis_LOCK
    verify(lock_path, cross_split_report_transition_structure_synthesis_LOCK_SHA256, 'cross_split_report_transition_structure_synthesis final lock')
    lock = readj(lock_path)
    if lock.get('phase') != 'cross_split_report_transition_structure_synthesis':
        raise RuntimeError('cross_split_report_transition_structure_synthesis final-lock phase drift')
    if lock.get('next_phase') != PHASE:
        raise RuntimeError(f'cross_split_report_transition_structure_synthesis does not authorize {PHASE}')
    if lock.get('experience_K') != EXPERIENCE_K:
        raise RuntimeError('cross_split_report_transition_structure_synthesis experience-K drift')
    if lock.get('report_transition_K') != TRANSITION_K:
        raise RuntimeError('cross_split_report_transition_structure_synthesis transition-K drift')
    if bool(lock.get('new_behavioral_values_opened')) or bool(lock.get('new_signal_values_opened')):
        raise RuntimeError('cross_split_report_transition_structure_synthesis boundary drift')
    if bool(lock.get('semantic_relabeling_permitted')):
        raise RuntimeError('cross_split_report_transition_structure_synthesis semantic boundary drift')
    rows = []
    for rel, expected in sorted(lock.get('hashes', {}).items()):
        p = project / rel
        got = verify(p, expected, f'cross_split_report_transition_structure_synthesis fixed child {rel}')
        rows.append({'source': 'cross_split_report_transition_structure_synthesis_fixed_child', 'role': 'parent_lock_child', 'split': '', 'path': rel, 'expected_sha256': expected, 'observed_sha256': got, 'values_parsed_in_semantic_report_execution_design': False})
    return rows

def read_split_manifest_safe(project: Path) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    with (project / SPLIT_MANIFEST).open('r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        required = {'subject_id', 'split'}
        if not required.issubset(set(r.fieldnames or [])):
            raise RuntimeError('prespecified split manifest lacks subject_id/split')
        rows = [{'subject_id': str(x['subject_id']).strip(), 'split': str(x['split']).strip()} for x in r]
    if len(rows) != 60 or len({x['subject_id'] for x in rows}) != 60:
        raise RuntimeError('prespecified split manifest participant-count drift')
    by_split = {}
    for split, expected_n in [('discovery', 36), ('validation', 12), ('confirmation', 12)]:
        vals = sorted((x['subject_id'] for x in rows if x['split'] == split))
        if len(vals) != expected_n:
            raise RuntimeError(f'prespecified split size drift for {split}: {len(vals)}')
        by_split[split] = vals
    return (rows, by_split)

def raw_source_inventory(dataset: Path, split_subjects: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ['validation', 'confirmation']:
        for subject in split_subjects[split]:
            beh = dataset / subject / 'beh' / f'{subject}_task-emotion_beh.tsv'
            meta = dataset / 'derivatives' / 'metadata' / f'{subject}_trials_with_context.csv'
            for role, path in [('raw_behavior_tsv', beh), ('safe_metadata_schedule', meta)]:
                if not path.is_file():
                    raise RuntimeError(f'Missing held-out source file: {path}')
                rows.append({'split': split, 'subject_id': subject, 'role': role, 'path': str(path), 'size_bytes': int(path.stat().st_size), 'sha256': sha256(path), 'values_parsed_in_semantic_report_execution_design': False})
    return rows

def program_policy(source_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    return {'phase': PHASE, 'generated_at': now(), 'status': 'semantic_report_execution_program_prespecified_before_target_opening', 'seed': SEED, 'structural_prerequisites': {'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'B1_crosssplit_low_overlap_already_fixed': True}, 'target_columns': {'ordered_reports': REPORTS, 'nominal_categories_full_names': FULL_CATEGORIES, 'readme_prefix_to_full_category': CATEGORY_PREFIX_MAP}, 'absolute_lineage_rules': {'reports': 'Discovery: primary_analysis_reference outcome manifest. Validation/Confirmation: only the ten report columns from the prespecified canonical trial table, keyed by subject_id + trial_number_global and source-audited against the raw behavior TSV inventory. Canonical context/stimulus/category columns are forbidden.', 'discovery_context_category': 'derivative_closure corrected Discovery master only; require the layer-specific authoritative flag needed by the test.', 'heldout_context': 'validation_source_epoch_lineage/confirmation_source_epoch_lineage authoritative timing manifest event-grammar context only. During target materialization, it must agree with the corresponding safe metadata schedule context for that global trial; disagreement makes context/category unavailable.', 'heldout_stimulus_category': 'Never use canonical stimulus_id/emotion_category. Recompute context-specific trial index from the complete safe metadata schedule, then map (subject_id, authoritative context, context-specific index) through the exact README sequence parser prespecified by README_SEQUENCE_CORE_SHA256. Missing or ambiguous sequence item => unavailable; no fallback.', 'acquisition_fraction': 'Use subject_acquisition_order_1based / subject-specific maximum order in the authoritative timing/outcome source before EEG-semantic intersection. This reproduces the historical primary_analysis_reference nuisance convention; do not recompute the denominator after target/EEG exclusions.', 'join_keys': {'Discovery': 'raw_behavior_row_id + subject_id integrity check', 'Validation': 'subject_id + trial_number; retain prespecified structural raw_behavior_row_id', 'Confirmation': 'subject_id + trial_number; retain prespecified structural raw_behavior_row_id'}, 'no_global_exclusion_mask': True, 'per_test_complete_case_only': True}, 'structural_predictor_rules': {'experience_discrete': {'K': EXPERIENCE_K, 'family_ids': 'immutable 1-based IDs from fixed discovery_stimulus_experience_multiresolution_fit/validation_stimulus_experience_multiresolution_transfer/confirmation_stimulus_experience_multiresolution_transfer assignment files', 'heldout_rows': 'participant_eligible=True only for discrete held-out family tests'}, 'transition_discrete': {'K': TRANSITION_K, 'family_ids': 'immutable 1-based IDs from fixed discovery_report_transition_multiresolution_fit/validation_report_transition_multiresolution_transfer/confirmation_report_transition_multiresolution_transfer assignment files', 'heldout_rows': 'participant_eligible=True only for B3 discrete held-out family tests'}, 'continuous_signatures': {'experience': 'exact 14 standardized__ coordinates from prespecified experience signature files', 'report_transition': 'exact 14 standardized__ coordinates from prespecified transition signature files', 'B2_rows': 'intersection of finite experience and transition continuous signatures plus complete target/nuisance rows; participant_eligible discrete flag is not required', 'coordinate_order': STD_SIGNATURE_COLUMNS}}, 'A1_nominal_category_experience_family': {'tested_K': EXPERIENCE_K, 'analysis_rows': 'experience discrete assignment + authoritative emotion_category; no report or context requirement', 'participant_weight': 'within each K/test, each participant total weight = 1', 'weighted_contingency': 'sum participant-balanced trial weights by category x family', 'NMI': 'weighted mutual information / sqrt(weighted H(category)*weighted H(family)); 0 if denominator is zero', 'Cramers_V': 'weighted Pearson chi-square V = sqrt(chi2/(total_weight*min(r-1,c-1)))', 'permutations': 10000, 'null': 'permute family labels within participant, preserving each participant family counts', 'tail': 'upper', 'metric_specific_p': '(1 + count(null >= observed))/(B+1)', 'co_primary_parent_p': "Bonferroni omnibus across NMI and Cramer's V: min(1, 2*min(p_NMI,p_V))", 'mixing_outputs': ['participant-balanced P(family|category)', 'participant-balanced P(category|family)', 'H(family|category)', 'H(category|family)', 'normalized H(family|category)/H(family)', 'normalized H(category|family)/H(category)', 'per-category maximum-family concentration', 'per-family maximum-category concentration'], 'claim_boundary': 'non-significance is not positive proof of label mismatch; weak/diffuse mapping is reported with exact held-out effect sizes and mixing matrices'}, 'A2_joint_reports_experience_family': {'tested_K': EXPERIENCE_K, 'analysis_rows': 'experience discrete assignment + all ten finite reports + authoritative context + prespecified acquisition_fraction', 'baseline_X': ['context_video', 'acquisition_fraction', 'acquisition_fraction_squared'], 'full_X': ['baseline_X', 'all_10_reports_in_prespecified_order'], 'outer': 'leave-one-participant-out', 'model': 'multinomial logistic regression, L2, lbfgs, intercept, max_iter=4000, tol=1e-8', 'C_grid': [0.01, 0.1, 1.0, 10.0, 100.0], 'inner_selection': '5 deterministic participant-group folds inside outer training data; choose C maximizing participant-balanced balanced accuracy; ties choose smaller C', 'predictor_scaling': 'fit mean/std on outer/inner training rows only; zero std -> 1; never fit scaler on held-out participant', 'training_weights': 'each training participant total weight 1, multiplied by inverse training class frequency class balancing', 'primary_statistic': 'participant-balanced held-out BA(full)-BA(nuisance)', 'permutations': 1000, 'null': 'permute complete 10-report row vectors within participant, preserving report covariance; rerun nested pipeline', 'tail': 'upper', 'single_report_followups': 'nuisance+one-report vs nuisance, same permutation draws; Holm across ten; interpret only if the corresponding A2 parent survives A-family Holm'}, 'A_family_multiplicity': {'parents': ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4'], 'correction': 'Holm within split across the six parent p-values', 'alpha': 0.05, 'no_cross_K_winner_selection': True}, 'B2_incremental_reports_from_transition_signature': {'analysis_rows': 'complete 10-report vector + authoritative context + acquisition_fraction + finite 14D experience signature + finite 14D report-transition signature', 'M0': 'context_video + acquisition_fraction + acquisition_fraction_squared', 'M1': 'M0 + 14D prespecified continuous experience signature', 'M2': 'M1 + 14D prespecified continuous report-transition signature', 'outer': 'leave-one-participant-out', 'estimator': 'participant-weighted multivariate ridge with intercept', 'alpha_grid': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], 'inner_selection': '5 deterministic participant-group folds inside outer training data; select alpha separately for M0/M1/M2 minimizing participant-balanced multivariate MSE; ties choose larger alpha', 'X_scaling': 'training-only mean/std per predictor; zero std -> 1', 'Y_scaling': 'training-only mean/std separately for each of ten reports; zero std -> 1; no held-out target centering/calibration', 'primary_statistic': 'participant-balanced mean of Delta-R2=M2_R2-M1_R2; within held-out participant, R2=1-SSE_model/SSE_zero where SSE_zero predicts zero in training-standardized target space', 'permutations': 1000, 'null': 'permute complete 10-report row vectors within participant and rerun the full nested ridge pipeline', 'tail': 'upper', 'dimension_followups': 'dimension-specific Delta-R2 from the same held-out predictions; Holm across ten; interpret only if B2 parent p<0.05'}, 'B3_discrete_transition_family_reports': {'tested_K': TRANSITION_K, 'status': 'secondary; structural activation gate already satisfied cross-split', 'model': 'exact A2 joint-10-report multinomial-logistic framework, but outcome is transition family', 'heldout_rows': 'participant_eligible=True', 'permutations': 1000, 'parent_multiplicity': 'Holm across B3_K2 and B3_K3 within split', 'single_report_followups': 'same one-report framework, Holm across ten, only after that B3 parent survives B3 Holm'}, 'split_execution_and_lockbox': {'all_prespecified_tests_run_in_each_split': True, 'Discovery_results_do_not_select_Validation_tests': True, 'Validation_results_do_not_select_Confirmation_tests': True, 'Confirmation_opening_rule': 'Confirmation semantic targets may open only after all Validation semantic outputs are final-fixed and a separate validation_confirmation_semantic_design Confirmation opening/analysis specification passes; opening is not contingent on p-value direction because the entire program is already prespecified', 'replication_language': 'internal held-out replication within this dataset only; never external replication'}, 'semantic_naming_rule': 'No family is renamed automatically. Any future semantic name requires a separate pre-prespecified naming rule and strong, specific, directionally stable held-out mapping.', 'forbidden': ['canonical context, canonical stimulus_id, or canonical emotion_category as held-out target', 'post-hoc report subset selection', 'selecting one K by semantic strength', 'changing structural family IDs', 'changing EEG signatures/centroids/support thresholds', 'split-specific family relabeling', 'using behavior to alter any structural decision', 'calling report-transition activity distortion, dishonesty, suppression, exaggeration, or conscious editing', 'external replication claim'], 'source_inventory_hashed_without_value_parsing': source_inventory, 'execution_order': [{'order': 1, 'phase': 'discovery_semantic_report_execution', 'role': 'Discovery semantic/report execution A1/A2/B2/B3'}, {'order': 2, 'phase': 'validation_semantic_report_execution', 'role': 'one-shot Validation target materialization + semantic/report execution, all prespecified tests'}, {'order': 3, 'phase': 'validation_confirmation_semantic_design', 'role': 'Validation synthesis + Confirmation semantic opening/analysis specification; no Confirmation values'}, {'order': 4, 'phase': 'confirmation_semantic_report_execution', 'role': 'one-shot Confirmation target materialization + semantic/report execution, all prespecified tests'}, {'order': 5, 'phase': 'cross_split_semantic_report_synthesis', 'role': 'cross-split semantic/report synthesis and claim specification'}]}

def next_policy(program_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'program_policy_canonical_sha256': program_sha, 'role': 'Discovery-only execution of the already-prespecified A1/A2/B2/B3 semantic/report tests', 'Discovery_behavior_opening_permitted': True, 'Validation_behavior_opening_permitted': False, 'Confirmation_behavior_opening_permitted': False, 'new_EEG_signal_opening_permitted': False, 'required_Discovery_sources': {'reports_and_acquisition_fraction': {'path': str(DISCOVERY_OUTCOME_MANIFEST), 'sha256': DISCOVERY_OUTCOME_MANIFEST_SHA256}, 'authoritative_context_and_category': {'path': str(DISCOVERY_CORRECTED_MASTER), 'sha256': DISCOVERY_CORRECTED_MASTER_SHA256}}, 'required_structural_sources': [{'role': role, 'split': split, 'path': str(path), 'sha256': h} for role, split, path, h in STRUCTURAL_REFS if split == 'Discovery'], 'tests': ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3'], 'behavior_cannot_change': ['K', 'family IDs', 'EEG signatures', 'centroids', 'support thresholds', 'test definitions', 'multiplicity families', 'held-out opening order'], 'next_after_discovery_semantic_report_execution': 'validation_semantic_report_execution regardless of Discovery p-values, after discovery_semantic_report_execution final lock verification'}

def self_test() -> int:
    assert EXPERIENCE_K == [2, 3, 4]
    assert TRANSITION_K == [2, 3]
    assert len(REPORTS) == 10
    assert len(SIGNATURE_COORDS) == 14
    assert len(STD_SIGNATURE_COLUMNS) == 14
    assert set(CATEGORY_PREFIX_MAP.values()) == set(FULL_CATEGORIES)
    assert PHASE == 'semantic_report_execution_design'
    assert NEXT_PHASE == 'discovery_semantic_report_execution'
    print('semantic_report_execution_design self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    dataset = args.dataset.resolve()
    project = args.project.resolve()
    if not dataset.is_dir():
        raise RuntimeError(f'Dataset root not found: {dataset}')
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    out = project / OUT
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] verifying cross_split_report_transition_structure_synthesis structural closure and design authorization', flush=True)
    rehash_rows = verify_parent_lock(project)
    verify(project / cross_split_report_transition_structure_synthesis_SUMMARY, cross_split_report_transition_structure_synthesis_SUMMARY_SHA256, 'cross_split_report_transition_structure_synthesis summary')
    verify(project / cross_split_report_transition_structure_synthesis_NEXT_POLICY, cross_split_report_transition_structure_synthesis_NEXT_POLICY_SHA256, 'cross_split_report_transition_structure_synthesis→semantic_report_execution_design policy')
    verify(project / semantic_report_dissociation_design_SEMANTIC_POLICY, semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256, 'semantic_report_dissociation_design semantic policy')
    pnext = readj(project / cross_split_report_transition_structure_synthesis_NEXT_POLICY)
    if pnext.get('structural_prerequisites', {}).get('experience_resolutions_replicated') != EXPERIENCE_K:
        raise RuntimeError('cross_split_report_transition_structure_synthesis next-policy experience K drift')
    if pnext.get('structural_prerequisites', {}).get('report_transition_resolutions_replicated') != TRANSITION_K:
        raise RuntimeError('cross_split_report_transition_structure_synthesis next-policy transition K drift')
    if pnext.get('semantic_execution_in_semantic_report_execution_design_permitted') is not False:
        raise RuntimeError('cross_split_report_transition_structure_synthesis policy unexpectedly permits semantic execution in semantic_report_execution_design')
    print(f'[{now()}] verifying immutable structural assignment/signature references', flush=True)
    source_inventory_rows: list[dict[str, Any]] = []
    for role, split, rel, expected in STRUCTURAL_REFS:
        got = verify(project / rel, expected, f'{split} {role}')
        row = {'source': 'structural_reference', 'role': role, 'split': split, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'values_parsed_in_semantic_report_execution_design': False}
        source_inventory_rows.append(row)
        rehash_rows.append(row.copy())
    print(f'[{now()}] verifying target/split lineage by hashes only', flush=True)
    fixed_refs = [('analysis_specification', '', ANALYSIS_specification, ANALYSIS_specification_SHA256), ('participant_split', '', SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256), ('canonical_trial_table_REPORTS_ONLY_FUTURE', '', CANONICAL_TRIAL_TABLE, CANONICAL_TRIAL_TABLE_SHA256), ('Discovery_canonical_master_FUTURE', 'Discovery', DISCOVERY_CORRECTED_MASTER, DISCOVERY_CORRECTED_MASTER_SHA256), ('Discovery_outcome_manifest_FUTURE', 'Discovery', DISCOVERY_OUTCOME_MANIFEST, DISCOVERY_OUTCOME_MANIFEST_SHA256), ('Validation_event_timing', 'Validation', VAL_TIMING, VAL_TIMING_SHA256), ('Validation_lineage_lock', 'Validation', VAL_LINEAGE_LOCK, VAL_LINEAGE_LOCK_SHA256), ('Confirmation_event_timing', 'Confirmation', CONF_TIMING, CONF_TIMING_SHA256), ('Confirmation_lineage_lock', 'Confirmation', CONF_LINEAGE_LOCK, CONF_LINEAGE_LOCK_SHA256), ('README_sequence_core', '', README_SEQUENCE_CORE, README_SEQUENCE_CORE_SHA256)]
    for role, split, rel, expected in fixed_refs:
        got = verify(project / rel, expected, role)
        row = {'source': 'prespecified_lineage_reference', 'role': role, 'split': split, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'values_parsed_in_semantic_report_execution_design': False}
        source_inventory_rows.append(row)
        rehash_rows.append(row.copy())
    verify(dataset / README_RELATIVE, README_SHA256, 'dataset README')
    source_inventory_rows.append({'source': 'prespecified_lineage_reference', 'role': 'dataset_README_sequence_source', 'split': '', 'path': str(dataset / README_RELATIVE), 'expected_sha256': README_SHA256, 'observed_sha256': README_SHA256, 'values_parsed_in_semantic_report_execution_design': False})
    _, by_split = read_split_manifest_safe(project)
    print(f'[{now()}] hashing held-out raw behavior/metadata sources without parsing values', flush=True)
    raw_inventory = raw_source_inventory(dataset, by_split)
    prog = program_policy(raw_inventory)
    writej(project / PROGRAM_POLICY, prog)
    prog_sha = canonical_sha(prog)
    nxt = next_policy(prog_sha)
    writej(project / NEXT_POLICY, nxt)
    hypotheses = [{'branch': 'A', 'test_id': 'A1_K2/A1_K3/A1_K4', 'target': 'authoritative nominal emotion category', 'predictor_or_outcome': 'experience family K=2,3,4', 'primary_statistic': "participant-balanced NMI + Cramer's V co-primary", 'permutations': 10000, 'parent_multiplicity': 'A1/A2 x K=2,3,4 Holm after A1 2-metric Bonferroni omnibus', 'heldout_execution': 'all tests regardless Discovery p'}, {'branch': 'A', 'test_id': 'A2_K2/A2_K3/A2_K4', 'target': 'experience family K=2,3,4', 'predictor_or_outcome': 'all ten reports jointly beyond nuisance', 'primary_statistic': 'participant-balanced LOPO Delta balanced accuracy', 'permutations': 1000, 'parent_multiplicity': 'A1/A2 x K=2,3,4 Holm', 'heldout_execution': 'all tests regardless Discovery p'}, {'branch': 'B', 'test_id': 'B2', 'target': 'joint ten-report vector', 'predictor_or_outcome': 'incremental 14D report-transition signature beyond 14D experience signature', 'primary_statistic': 'participant-balanced LOPO Delta-R2 M2-M1', 'permutations': 1000, 'parent_multiplicity': 'single primary B2 parent', 'heldout_execution': 'all tests regardless Discovery p'}, {'branch': 'B', 'test_id': 'B3_K2/B3_K3', 'target': 'transition family K=2,3', 'predictor_or_outcome': 'all ten reports jointly beyond nuisance', 'primary_statistic': 'participant-balanced LOPO Delta balanced accuracy', 'permutations': 1000, 'parent_multiplicity': 'Holm across B3 K=2,3', 'heldout_execution': 'all tests regardless Discovery p'}]
    write_csv(out / 'tables/semantic_report_execution_design_prespecified_hypothesis_execution_registry.csv', hypotheses, ['branch', 'test_id', 'target', 'predictor_or_outcome', 'primary_statistic', 'permutations', 'parent_multiplicity', 'heldout_execution'])
    order_rows = []
    for x in prog['execution_order']:
        order_rows.append({'order': x['order'], 'phase': x['phase'], 'role': x['role'], 'result_contingent_selection_permitted': False})
    write_csv(out / 'tables/semantic_report_execution_design_prespecified_execution_order.csv', order_rows, ['order', 'phase', 'role', 'result_contingent_selection_permitted'])
    lineage_rows = [{'split': 'Discovery', 'reports': str(DISCOVERY_OUTCOME_MANIFEST), 'context': str(DISCOVERY_CORRECTED_MASTER), 'emotion_category': str(DISCOVERY_CORRECTED_MASTER), 'canonical_context_category_permitted': False, 'join_rule': 'raw_behavior_row_id + subject_id integrity'}, {'split': 'Validation', 'reports': f'{CANONICAL_TRIAL_TABLE} REPORT COLUMNS ONLY', 'context': str(VAL_TIMING), 'emotion_category': 'README sequence after event-context + safe-metadata context-index reconstruction', 'canonical_context_category_permitted': False, 'join_rule': 'subject_id + trial_number -> retain prespecified structural raw_behavior_row_id'}, {'split': 'Confirmation', 'reports': f'{CANONICAL_TRIAL_TABLE} REPORT COLUMNS ONLY', 'context': str(CONF_TIMING), 'emotion_category': 'README sequence after event-context + safe-metadata context-index reconstruction', 'canonical_context_category_permitted': False, 'join_rule': 'subject_id + trial_number -> retain prespecified structural raw_behavior_row_id'}]
    write_csv(out / 'tables/semantic_report_execution_design_prespecified_target_lineage_spec.csv', lineage_rows, ['split', 'reports', 'context', 'emotion_category', 'canonical_context_category_permitted', 'join_rule'])
    write_csv(out / 'tables/semantic_report_execution_design_heldout_raw_source_inventory.csv', raw_inventory, ['split', 'subject_id', 'role', 'path', 'size_bytes', 'sha256', 'values_parsed_in_semantic_report_execution_design'])
    write_csv(out / 'tables/semantic_report_execution_design_source_reference_inventory.csv', source_inventory_rows, ['source', 'role', 'split', 'path', 'expected_sha256', 'observed_sha256', 'values_parsed_in_semantic_report_execution_design'])
    write_csv(out / 'qc/semantic_report_execution_design_source_rehash.csv', rehash_rows, ['source', 'role', 'split', 'path', 'expected_sha256', 'observed_sha256', 'values_parsed_in_semantic_report_execution_design'])
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'semantic_report_execution_design_prespecified', 'program_policy_canonical_sha256': prog_sha, 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'A_parent_tests': 6, 'B2_parent_tests': 1, 'B3_parent_tests': 2, 'heldout_raw_behavior_files_hashed_not_parsed': 24, 'heldout_safe_metadata_files_hashed_not_parsed': 24, 'new_EEG_signal_values_opened': False, 'behavioral_report_values_opened': False, 'emotion_category_values_opened': False, 'context_target_values_opened': False, 'stimulus_identity_target_values_opened': False, 'canonical_context_category_use_permitted': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_role': 'Discovery-only A1/A2/B2/B3 execution under the prespecified semantic_report_execution_design program', 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/semantic_report_execution_design_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': sha256(Path(__file__).resolve()), 'cross_split_report_transition_structure_synthesis_final_lock_sha256': cross_split_report_transition_structure_synthesis_LOCK_SHA256, 'cross_split_report_transition_structure_synthesis_next_policy_sha256': cross_split_report_transition_structure_synthesis_NEXT_POLICY_SHA256, 'semantic_report_dissociation_design_semantic_policy_sha256': semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256, 'program_policy_canonical_sha256': prog_sha, 'boundary': {'new_EEG_signal_values_opened': False, 'behavioral_report_values_opened': False, 'emotion_category_values_opened': False, 'context_target_values_opened': False, 'stimulus_identity_target_values_opened': False, 'raw_behavior_bytes_hashed_only': True, 'raw_behavior_values_parsed': False, 'canonical_behavior_values_parsed': False, 'semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/semantic_report_execution_design_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/semantic_report_execution_design_report.md'
    report_path.write_text(''.join(['# semantic_report_execution_design — Semantic/report execution analysis specification', '', '**semantic_report_execution_design opened no behavioral values and no new EEG signal values.**', '', '## Structural starting point', '', '- Experience partitions: **K=2, K=3, K=4** replicated across Discovery/Validation/Confirmation.', '- Report-transition partitions: **K=2, K=3** replicated across Discovery/Validation/Confirmation.', '- Outcome-free B1 experience/transition overlap is already fixed as very low across all three splits.', '', '## Critical lineage correction', '', 'The old canonical table remains permitted only as a source-faithful report-row authority for held-out participants.', '**Its context, stimulus_id, and emotion_category columns are explicitly forbidden.**', '', 'Held-out context must come from the authoritative event-grammar timing lineage. Held-out stimulus/category must be reconstructed from authoritative context + full safe metadata context index + the prespecified README context-specific sequence parser. Any disagreement or missing sequence item becomes unavailable; there is no fallback to old canonical semantics.', '', '## prespecified tests', '', '- **A1:** participant-balanced category ↔ experience-family dependence/mixing, K=2/3/4.', '- **A2:** all ten reports jointly predicting experience family beyond context/acquisition nuisance, K=2/3/4.', '- **B2:** incremental joint-report information in the continuous report-transition signature beyond the continuous experience signature.', '- **B3:** all ten reports jointly predicting replicated transition family K=2/3; secondary, structurally activated.', '', 'All K values and all parent tests are carried to held-out splits regardless of Discovery significance. No semantic winner is selected.', '', '## Multiplicity', '', "- A1 has a two-metric Bonferroni omnibus (NMI + Cramer's V).", '- Holm across the six A1/A2 × K=2/3/4 parents within each split.', '- B2 is one pre-specified parent.', '- Holm across B3 K=2/3.', '- Ten-report child follow-ups are gated by their parent and Holm-corrected across ten.', '', '## Lockbox sequence', '', 'discovery_semantic_report_execution executes Discovery. validation_semantic_report_execution executes Validation under the already-prespecified design regardless of Discovery p-values. validation_confirmation_semantic_design is a no-Confirmation-value synthesis/analysis specification. confirmation_semantic_report_execution then opens Confirmation one-shot and executes the identical prespecified tests regardless of Validation direction.', '', '## Claim boundary', '', 'Non-significance alone is not proof that a human label is wrong. A report-transition association cannot be called distortion, dishonesty, suppression, exaggeration, or conscious editing. Any replication remains internal held-out replication within this dataset.', '', f'## Next', '', f'**{NEXT_PHASE}** — Discovery-only semantic/report execution.', '']) + '', encoding='utf-8')
    lock_targets = [project / PROGRAM_POLICY, project / NEXT_POLICY, contract_path, summary_path, out / 'qc/semantic_report_execution_design_source_rehash.csv', out / 'tables/semantic_report_execution_design_prespecified_hypothesis_execution_registry.csv', out / 'tables/semantic_report_execution_design_prespecified_execution_order.csv', out / 'tables/semantic_report_execution_design_prespecified_target_lineage_spec.csv', out / 'tables/semantic_report_execution_design_heldout_raw_source_inventory.csv', out / 'tables/semantic_report_execution_design_source_reference_inventory.csv', report_path]
    final_lock = out / 'qc/semantic_report_execution_design_final_output_lock_hashes.json'
    writej(final_lock, {'phase': PHASE, 'generated_at': now(), 'status': 'semantic_report_execution_design_prespecified', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'program_policy_canonical_sha256': prog_sha, 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'new_EEG_signal_values_opened': False, 'behavioral_report_values_opened': False, 'emotion_category_values_opened': False, 'context_target_values_opened': False, 'canonical_context_category_use_permitted': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE})
    script_dir = Path(__file__).resolve().parent
    print('')
    print('semantic_report_execution_design PASS')
    print(f'Experience K prespecified: {EXPERIENCE_K}')
    print(f'Report-transition K prespecified: {TRANSITION_K}')
    print('Behavioral report values opened: False')
    print('Emotion-category values opened: False')
    print('Context target values opened: False')
    print('New EEG signal values opened: False')
    print('Canonical context/category use permitted: False')
    print('Held-out raw behavior files hashed, not parsed: 24')
    print('Held-out metadata files hashed, not parsed: 24')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())