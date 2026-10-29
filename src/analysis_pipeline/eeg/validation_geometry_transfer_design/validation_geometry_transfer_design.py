# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase validation_geometry_transfer_design: specification Validation trajectory-geometry transfer.

This phase reads only fixed Discovery artifacts, prespecified code/policies, and the
Phase-0 participant split roster. It does not read any Validation EEG, event,
feature, report, context, stimulus, or outcome file. It specifications the complete
multi-stage boundary for later outcome-free Validation transfer."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'validation_geometry_transfer_design'
SOURCE_PHASE = 'discovery_robustness_audit'
NEXT_PHASE = 'validation_source_epoch_lineage'
SEED = 540
EXPECTED = {'raw_eeg_epoch_folds_script': '93ca3cc67f7fb72f008845941a16f8a802e4b0aee601d53fbeb5c84c8e25aa8b', 'raw_eeg_preprocessing_qc_script': 'bd6a50669f0a6ba17965a4497433f1d4583d4a98f840dff29a73866d801e2ecb', 'outcome_free_features_script': 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba', 'raw_eeg_epoch_folds_policy': '70ed998214bff8fed6bdb9fba30f3aad5239c973b596ccfa558eec8af450df46', 'raw_eeg_preprocessing_qc_contract': 'b0aa79f1da532c99e780bcc0740041fc7429a051ac76df1d202e465f0fb19b6f', 'outcome_free_features_contract': '63a76ebcc83b5d282045771130123c194bc98556e0216bc5a855a55a2f4be36d', 'split_manifest': '0f6156e2a0de5af59dd3a7cd15b2a4a914dc0a0540b9d941fa6676a697f8d453'}
FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
EXPECTED_VALIDATION_SUBJECTS = ['sub-07', 'sub-09', 'sub-11', 'sub-17', 'sub-21', 'sub-25', 'sub-28', 'sub-37', 'sub-39', 'sub-51', 'sub-53', 'sub-57']
EXPECTED_VALIDATION_TRIALS = 494
prespecified_FALLBACK_IQR = {'early_to_late_norm': 0.4754957722315247, 'late_to_rating_norm': 0.43253425451665484, 'early_to_rating_direct_norm': 0.4071658093620769, 'total_path_length': 0.41793243255608914, 'directness_ratio': 1.0449506338050496, 'segment_cosine': 0.6616508651779611}
prespecified_CENTROIDS = {1: [-0.35414370231882775, -0.3054190864870062, 0.0023303355418339416, -0.3729072257696148, 0.36312713605666835, 0.3454466033852239], 2: [0.6825998665814916, 0.5559751984416603, 0.19227691522592993, 0.7145652013742286, -0.5821176422158368, -0.6172840398179816]}
prespecified_REFERENCE_THRESHOLDS = {'in_support_squared_distance_max': 12.825359640315092, 'low_margin_threshold': 0.1061596662041992, 'reference_rule_distance': '99.5th percentile of cross_person_alignment_fit LOPO-agreeing assigned squared distance', 'reference_rule_margin': '1st percentile of cross_person_alignment_fit LOPO-agreeing assignment margin'}

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

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

def require(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Missing prerequisite(s):\n  ' + '\n  '.join(missing))

def bval(x: Any) -> bool:
    return bool(x) and str(x).strip().lower() not in {'false', '0', 'none', 'nan', ''}

def input_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    s = project / 'src/analysis_pipeline/eeg'
    return {'h_contract': c / 'discovery_robustness_audit_discovery_robustness_contract.json', 'h_next': project / 'analysis_specifications/S024_discovery_robustness_audit_validation_geometry_transfer.json', 'h_summary': q / 'discovery_robustness_audit_discovery_robustness_summary.json', 'h_final_lock': q / 'discovery_robustness_audit_final_output_lock_hashes.json', 'h_log': project / 'outputs/logs/discovery_robustness_audit_discovery_robustness_audit.log', 'h_source_rehash': q / 'discovery_robustness_audit_source_rehash.csv', 'h_r1': t / 'discovery_robustness_audit_r1_integrity_reproduction.csv', 'h_r2': t / 'discovery_robustness_audit_r2_h1_fixed_contrast_jackknife.csv', 'h_r3': t / 'discovery_robustness_audit_r3_h2_fixed_oof_jackknife.csv', 'h_r4': t / 'discovery_robustness_audit_r4_h3_fixed_oof_jackknife.csv', 'h_r5': t / 'discovery_robustness_audit_r5_categorical_fixed_assignment_jackknife.csv', 'h_r6_tests': t / 'discovery_robustness_audit_r6_lopo_subset_sensitivity_tests.csv', 'h_r6_null': t / 'discovery_robustness_audit_r6_lopo_subset_null_replicates.csv', 'h_r7': t / 'discovery_robustness_audit_r7_null_calibration_audit.csv', 'h_r8_subjects': t / 'discovery_robustness_audit_r8_trajectory_structural_gate_subjects.csv', 'h_r8_summary': q / 'discovery_robustness_audit_r8_trajectory_structural_gate_summary.json', 'h_r9': t / 'discovery_robustness_audit_r9_label_swap_invariance.csv', 'h_check_summary': t / 'discovery_robustness_audit_robustness_check_summary.csv', 'h_report': project / 'outputs/reports/discovery_robustness_audit_discovery_robustness_report.md', 'g_validation_policy': project / 'analysis_specifications/S030_discovery_synthesis_validation_gate.json', 'g_validation_manifest': t / 'discovery_synthesis_prespecified_validation_gate_manifest.csv', 'd_contract': c / 'cross_person_alignment_fit_cross_person_alignment_contract.json', 'd_final_lock': q / 'cross_person_alignment_fit_final_output_lock_hashes.json', 'd_preprocessing': t / 'cross_person_alignment_fit_trajectory_preprocessing.csv', 'd_centroids': t / 'cross_person_alignment_fit_trajectory_family_centroids.csv', 'd_assignments': t / 'cross_person_alignment_fit_trajectory_family_assignments.csv', 'd_lopo': t / 'cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv', 'split_manifest': c / 'participant_split_manifest.v1.csv', 'raw_eeg_epoch_folds_script': s / 'raw_eeg_epoch_folds/raw_eeg_epoch_folds.py', 'raw_eeg_preprocessing_qc_script': s / 'raw_eeg_preprocessing_qc/raw_eeg_preprocessing_qc.py', 'outcome_free_features_script': s / 'outcome_free_features/outcome_free_features.py', 'raw_eeg_epoch_folds_policy': project / 'analysis_specifications/S057_raw_eeg_epoch_folds.json', 'raw_eeg_preprocessing_qc_contract': c / 'raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_contract.json', 'outcome_free_features_contract': c / 'outcome_free_features_outcome_free_feature_contract.json'}

def output_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'policy': project / 'analysis_specifications/S082_validation_geometry_transfer.json', 'reference': project / 'analysis_specifications/S081_validation_geometry_transfer_reference.json', 'core': project / 'analysis_specifications/reference_code/validation_geometry_transfer_core.py', 'next_policy': project / 'analysis_specifications/S083_validation_geometry_transfer_source_epoch.json', 'contract': c / 'validation_geometry_transfer_design_validation_geometry_transfer_design_contract.json', 'roster': t / 'validation_geometry_transfer_design_validation_roster_manifest.csv', 'preprocessing': t / 'validation_geometry_transfer_design_discovery_reference_preprocessing.csv', 'centroids': t / 'validation_geometry_transfer_design_discovery_reference_centroids.csv', 'schema': t / 'validation_geometry_transfer_design_prespecified_transfer_output_schema.csv', 'gates': t / 'validation_geometry_transfer_design_prespecified_transfer_gate_manifest.csv', 'source_rehash': q / 'validation_geometry_transfer_design_source_rehash.csv', 'summary': q / 'validation_geometry_transfer_design_validation_geometry_transfer_design_summary.json', 'final_lock': q / 'validation_geometry_transfer_design_final_output_lock_hashes.json', 'report': r / 'validation_geometry_transfer_design_validation_geometry_transfer_design_report.md', 'log': l / 'validation_geometry_transfer_design_validation_geometry_transfer_design_specification.log'}

def verify_h(inputs: dict[str, Path]) -> tuple[dict, dict, dict, list[dict[str, Any]]]:
    h_contract = json.loads(inputs['h_contract'].read_text())
    h_next = json.loads(inputs['h_next'].read_text())
    h_summary = json.loads(inputs['h_summary'].read_text())
    lock = json.loads(inputs['h_final_lock'].read_text())
    if h_contract.get('phase') != SOURCE_PHASE or h_summary.get('phase') != SOURCE_PHASE:
        raise RuntimeError('discovery_robustness_audit phase mismatch')
    if h_next.get('next_phase') != PHASE or not bval(h_next.get('next_phase_permitted')):
        raise RuntimeError('discovery_robustness_audit did not permit validation_geometry_transfer_design')
    dec = h_summary.get('decision', {})
    for key in ['R1_pass', 'R7_pass', 'R8_pass', 'validation_geometry_transfer_design_specification_permitted']:
        if not bval(dec.get(key)):
            raise RuntimeError(f'discovery_robustness_audit gate failed: {key}')
    if bval(dec.get('validation_opened')) or bval(dec.get('confirmation_opened')):
        raise RuntimeError('discovery_robustness_audit opened a later cohort')
    expected = lock.get('hashes', {})
    hmap = {'log': 'h_log', 'source_rehash': 'h_source_rehash', 'r1': 'h_r1', 'r2': 'h_r2', 'r3': 'h_r3', 'r4': 'h_r4', 'r5': 'h_r5', 'r6_tests': 'h_r6_tests', 'r6_null': 'h_r6_null', 'r7': 'h_r7', 'r8_subjects': 'h_r8_subjects', 'r8_summary': 'h_r8_summary', 'r9': 'h_r9', 'check_summary': 'h_check_summary', 'summary': 'h_summary', 'report': 'h_report', 'contract': 'h_contract', 'next_policy': 'h_next'}
    audit = []
    for key, path_key in hmap.items():
        exp = expected.get(key)
        obs = sha256(inputs[path_key])
        ok = bool(exp and obs == exp)
        audit.append({'source_key': f'discovery_robustness_audit_{key}', 'path': str(inputs[path_key]), 'observed_sha256': obs, 'expected_sha256': exp or '', 'pass': ok, 'size_bytes': inputs[path_key].stat().st_size})
        if not ok:
            raise RuntimeError(f'discovery_robustness_audit final-lock mismatch: {key}')
    if h_contract.get('output_hashes', {}).get('source_rehash') != expected.get('source_rehash'):
        raise RuntimeError('discovery_robustness_audit contract/final-lock mismatch')
    return (h_contract, h_summary, lock, audit)

def verify_discovery_reference(inputs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract = json.loads(inputs['d_contract'].read_text())
    lock = json.loads(inputs['d_final_lock'].read_text())
    if contract.get('phase') != 'cross_person_alignment_fit':
        raise RuntimeError('cross_person_alignment_fit contract phase mismatch')
    h = lock.get('hashes', lock.get('final_output_hashes', {}))
    expected = contract.get('final_output_hashes', {})
    for key, path_key in [('trajectory_preprocessing', 'd_preprocessing'), ('trajectory_selected_centroids', 'd_centroids'), ('trajectory_selected_assignments', 'd_assignments'), ('trajectory_lopo', 'd_lopo')]:
        exp = expected.get(key) or h.get(key)
        if not exp or sha256(inputs[path_key]) != exp:
            raise RuntimeError(f'cross_person_alignment_fit fixed reference mismatch: {key}')
    pre = pd.read_csv(inputs['d_preprocessing'])
    ctr = pd.read_csv(inputs['d_centroids'])
    ass = pd.read_csv(inputs['d_assignments'])
    lopo = pd.read_csv(inputs['d_lopo'])
    if len(ass) != 1237 or ass.subject_id.nunique() != 31 or set(ass.trajectory_family_1based.astype(int)) != {1, 2}:
        raise RuntimeError('Unexpected cross_person_alignment_fit assignment reference')
    if len(ctr) != 12 or ctr.trajectory_family_1based.nunique() != 2:
        raise RuntimeError('Unexpected cross_person_alignment_fit centroid reference')
    return (pre, ctr, ass, lopo)

def build_policy(core_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'seed': SEED, 'scientific_role': 'specification outcome-free transfer of fixed Discovery three-epoch trajectory geometry and two centroids to Validation before any Validation file is read', 'fixed_before_validation_access': True, 'validation_opened': False, 'confirmation_opened': False, 'scope': {'permitted': 'structural outcome-free trajectory geometry transfer only', 'prohibited': ['H1-H6 report/context/category/stimulus mapping in Validation', 'shared discrete local-state taxonomy transfer', 'family renaming', 'centroid refitting', 'threshold tuning', 'Confirmation access']}, 'staged_execution': [{'phase': 'validation_source_epoch_lineage', 'role': 'specification Validation source/epoch lineage and coverage using split roster, raw headers, and event timing only', 'eeg_amplitude_read': False}, {'phase': 'validation_preprocessing_qc', 'role': 'apply exact prespecified raw_eeg_preprocessing_qc preprocessing/QC to Validation raw EEG', 'reports_outcomes_read': False}, {'phase': 'validation_feature_trajectory', 'role': 'extract exact prespecified outcome_free_features outcome-free compact features and construct per-participant three-epoch PCA trajectories', 'reports_outcomes_read': False}, {'phase': 'validation_fixed_assignment', 'role': 'apply prespecified geometry transforms and fixed centroids without refitting; evaluate structural transfer gates', 'reports_outcomes_context_targets_read': False}], 'source_code_lineage': {'raw_eeg_epoch_folds_epoch_specification_script_sha256': EXPECTED['raw_eeg_epoch_folds_script'], 'raw_eeg_preprocessing_qc_preprocessing_script_sha256': EXPECTED['raw_eeg_preprocessing_qc_script'], 'outcome_free_features_feature_script_sha256': EXPECTED['outcome_free_features_script'], 'prespecified_transfer_core_sha256': core_sha, 'adapter_rule': 'future cohort-specific executors may change only cohort selection, output namespace, and prespecified expected counts; signal-processing and feature algorithms must remain byte-identical or be proven functionally identical by prespecified synthetic regression fixtures'}, 'validation_roster': {'split_version': 'v1', 'subjects': EXPECTED_VALIDATION_SUBJECTS, 'expected_subjects': 12, 'expected_authoritative_trials': EXPECTED_VALIDATION_TRIALS, 'all_have_raw_eeg_in_participant_roster': True}, 'operationally_allowed_metadata': {'subject_id': True, 'trial_number': True, 'raw_behavior_row_id': True, 'raw_run': True, 'epoch_family': True, 'context': 'allowed only to apply already-prespecified epoch endpoint rules; must be dropped before geometry and never tested against family'}, 'forbidden_validation_columns_or_targets': ['valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion_category', 'stimulus_id', 'subjective factors', 'report-derived factors'], 'trajectory_construction': {'epoch_order': ['stimulus_early', 'stimulus_late', 'rating_early'], 'within_person_feature_preprocessing': 'exact latent_state_fit epoch-family robust standardization and whitened PCA, with no cohort pooling and no outcome access', 'geometry_features': FEATURES, 'geometry_transformations': {'first_four': 'log1p', 'directness_ratio': 'logit clip [1e-6,1-1e-6]', 'segment_cosine': 'atanh clip [-1+1e-6,1-1e-6]'}, 'within_validation_participant_scaling': 'median center and IQR divide using all eligible trajectories for that participant; use prespecified Discovery pooled participant-balanced IQR only when local IQR <=1e-8; winsorize [-6,6]', 'pooled_fallback_iqr': prespecified_FALLBACK_IQR}, 'fixed_assignment': {'family_count': 2, 'centroids': prespecified_CENTROIDS, 'distance': 'squared Euclidean in six standardized geometry coordinates', 'tie_rule': 'stable lowest numeric family via mergesort', 'refit': False, 'rename': False, 'support_thresholds': prespecified_REFERENCE_THRESHOLDS}, 'participant_eligibility': {'minimum_complete_three_epoch_trajectories': 16, 'nonfinite_geometry_rule': 'exclude trajectory before scaling and record reason', 'participant_below_minimum_rule': 'exclude participant from transfer gate but retain source/QC audit'}, 'structural_transfer_gate': {'minimum_eligible_participants': 9, 'minimum_complete_trajectory_fraction_of_participant_roster': 0.75, 'minimum_participant_balanced_in_support_fraction': 0.8, 'minimum_participants_with_both_families': 8, 'minimum_trials_per_family': 30, 'minimum_participant_balanced_occupancy_per_family': 0.1, 'all_source_preprocessing_feature_hash_gates_pass': True, 'no_report_outcome_mapping_executed': True, 'gate_failure_rule': 'stop structural transfer claim; do not retune preprocessing, thresholds, feature set, family count, or centroids'}, 'output_schema': {'required_assignment_columns': ['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2', 'in_discovery_support', 'low_margin'], 'required_participant_columns': ['subject_id', 'authoritative_trials', 'complete_trajectories', 'eligible_for_transfer', 'family1_trials', 'family2_trials', 'in_support_fraction', 'low_margin_fraction', 'both_families_present'], 'required_gate_columns': ['gate_id', 'metric', 'observed', 'threshold', 'comparison', 'pass']}, 'decision_boundary': 'A successful validation_fixed_assignment may support only replication of the two-family outcome-free geometric partition. It cannot support hedonic, emotional, contextual, or stimulus interpretation.'}

def schema_frame() -> pd.DataFrame:
    rows = []
    for table, cols in {'trajectory_assignments': ['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2', 'in_discovery_support', 'low_margin'], 'participant_transfer_summary': ['subject_id', 'authoritative_trials', 'complete_trajectories', 'eligible_for_transfer', 'family1_trials', 'family2_trials', 'in_support_fraction', 'low_margin_fraction', 'both_families_present'], 'structural_transfer_gate': ['gate_id', 'metric', 'observed', 'threshold', 'comparison', 'pass']}.items():
        for i, c in enumerate(cols):
            rows.append({'table_name': table, 'column_order_0based': i, 'column_name': c})
    return pd.DataFrame(rows)

def gate_frame() -> pd.DataFrame:
    return pd.DataFrame([{'gate_id': 'V1', 'metric': 'eligible_participants', 'threshold': 9, 'comparison': '>=', 'failure_action': 'stop; no tuning'}, {'gate_id': 'V2', 'metric': 'complete_trajectory_fraction_of_494', 'threshold': 0.75, 'comparison': '>=', 'failure_action': 'stop; no alternate windows'}, {'gate_id': 'V3', 'metric': 'participant_balanced_in_support_fraction', 'threshold': 0.8, 'comparison': '>=', 'failure_action': 'stop; no distance threshold change'}, {'gate_id': 'V4', 'metric': 'participants_with_both_families', 'threshold': 8, 'comparison': '>=', 'failure_action': 'stop; no centroid refit'}, {'gate_id': 'V5a', 'metric': 'family1_trial_count', 'threshold': 30, 'comparison': '>=', 'failure_action': 'stop; no family merging'}, {'gate_id': 'V5b', 'metric': 'family2_trial_count', 'threshold': 30, 'comparison': '>=', 'failure_action': 'stop; no family merging'}, {'gate_id': 'V6a', 'metric': 'family1_participant_balanced_occupancy', 'threshold': 0.1, 'comparison': '>=', 'failure_action': 'stop; no relabeling'}, {'gate_id': 'V6b', 'metric': 'family2_participant_balanced_occupancy', 'threshold': 0.1, 'comparison': '>=', 'failure_action': 'stop; no relabeling'}, {'gate_id': 'V7', 'metric': 'source_and_algorithm_hash_gates', 'threshold': 1, 'comparison': '==', 'failure_action': 'abort before interpretation'}, {'gate_id': 'V8', 'metric': 'report_outcome_mapping_executed', 'threshold': 0, 'comparison': '==', 'failure_action': 'invalidate transfer phase'}])

def write_report(path: Path, summary: dict, policy_sha: str, core_sha: str) -> None:
    lines = ['# Phase validation_geometry_transfer_design — Validation geometry-transfer analysis specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Validation data read: **no**', '- Confirmation read: **no**', '', '## prespecified scope', '', '- Only the fixed two-family outcome-free trajectory geometry may transfer to Validation.', '- H1-H6, report/context/category/stimulus mapping, shared discrete state taxonomy, family renaming, and centroid refitting are prohibited.', '', '## prespecified lineage', '', f'- Transfer-policy canonical SHA-256: `{policy_sha}`', f'- prespecified pure transfer-core SHA-256: `{core_sha}`', f'- Validation roster: 12 participants / {EXPECTED_VALIDATION_TRIALS} authoritative trials.', '', '## Next stage', '', '- validation_source_epoch_lineage may read Validation source metadata and EDF headers only to specification source/epoch lineage.', '- EEG amplitude remains unread in validation_source_epoch_lineage; later stages remain bound to the prespecified code, centroids, support thresholds, and failure rules.']
    atomic_text(path, ''.join(lines) + '')

def load_core(path: Path):
    spec = importlib.util.spec_from_file_location('prespecified_transfer_core', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod

def self_test(core_path: Path) -> None:
    core = load_core(core_path)
    df = pd.DataFrame({'subject_id': ['s'] * 4, 'raw_behavior_row_id': [f's|trial{i}' for i in range(4)], 'trial_number': range(1, 5), 'early_to_late_norm': [1, 2, 3, 4], 'late_to_rating_norm': [2, 2.5, 3, 3.5], 'early_to_rating_direct_norm': [1, 1.5, 2, 2.5], 'total_path_length': [3, 4.5, 6, 7.5], 'directness_ratio': [0.3, 0.4, 0.5, 0.6], 'segment_cosine': [-0.5, -0.2, 0.2, 0.5]})
    fb = np.array([prespecified_FALLBACK_IQR[x] for x in FEATURES])
    ctr = np.array([prespecified_CENTROIDS[1], prespecified_CENTROIDS[2]])
    out, prep = core.transfer_one_participant(df, fb, ctr)
    assert len(out) == 4 and len(prep) == 6 and set(out.trajectory_family_1based).issubset({1, 2})
    assert canonical_sha({'a': 1, 'b': 2}) == canonical_sha({'b': 2, 'a': 1})
    print('validation_geometry_transfer_design self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--core-source', type=Path, default=Path(__file__).with_name('validation_geometry_transfer_design_prespecified_transfer_core.py'))
    args = ap.parse_args()
    core_source = args.core_source.resolve()
    if args.self_test:
        self_test(core_source)
        return 0
    project = args.project.resolve()
    inputs = input_paths(project)
    outputs = output_paths(project)
    require([*inputs.values(), core_source])
    if outputs['final_lock'].exists() and (not args.force):
        raise RuntimeError('validation_geometry_transfer_design final lock exists; use --force to reproduce')
    outputs['log'].parent.mkdir(parents=True, exist_ok=True)
    atomic_text(outputs['log'], '')

    def note(msg: str):
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        with outputs['log'].open('a', encoding='utf-8') as f:
            f.write(line + '')
    note('Verifying discovery_robustness_audit gates and authorization without reading any Validation data file.')
    h_contract, h_summary, h_lock, h_audit = verify_h(inputs)
    note('Verifying fixed Discovery trajectory preprocessing, centroids, assignments, and LOPO diagnostics.')
    pre, ctr, ass, lopo = verify_discovery_reference(inputs)
    note('Verifying prespecified Phase-0 roster and upstream EEG/feature code lineage.')
    gpol = json.loads(inputs['g_validation_policy'].read_text())
    gman = pd.read_csv(inputs['g_validation_manifest'])
    if bval(gpol.get('validation_currently_open')) or bval(gpol.get('confirmation_currently_open')):
        raise RuntimeError('discovery_synthesis validation gate was already open')
    if gpol.get('conditional_future_scope') != 'outcome-free transfer of the fixed two-family trajectory geometry only':
        raise RuntimeError('discovery_synthesis validation scope mismatch')
    if set(gman.gate_id.astype(str)) != {f'G{i}' for i in range(1, 9)}:
        raise RuntimeError('discovery_synthesis validation gate manifest mismatch')
    rows = list(h_audit)
    for key in ['split_manifest', 'raw_eeg_epoch_folds_script', 'raw_eeg_preprocessing_qc_script', 'outcome_free_features_script', 'raw_eeg_epoch_folds_policy', 'raw_eeg_preprocessing_qc_contract', 'outcome_free_features_contract']:
        obs = sha256(inputs[key])
        exp = EXPECTED[key]
        rows.append({'source_key': key, 'path': str(inputs[key]), 'observed_sha256': obs, 'expected_sha256': exp, 'pass': obs == exp, 'size_bytes': inputs[key].stat().st_size})
    if not all((r['pass'] for r in rows)):
        raise RuntimeError('One or more prespecified upstream source hashes failed')
    core_sha = sha256(core_source)
    outputs['core'].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(core_source, outputs['core'])
    if sha256(outputs['core']) != core_sha:
        raise RuntimeError('prespecified transfer core copy mismatch')
    rows.append({'source_key': 'prespecified_transfer_core', 'path': str(core_source), 'observed_sha256': core_sha, 'expected_sha256': core_sha, 'pass': True, 'size_bytes': core_source.stat().st_size})
    atomic_csv(outputs['source_rehash'], pd.DataFrame(rows))
    roster = pd.read_csv(inputs['split_manifest'], encoding='utf-8-sig', usecols=['subject_id', 'split', 'n_trials', 'has_raw_eeg'])
    val = roster.loc[roster['split'].astype(str).eq('validation')].copy().sort_values('subject_id').reset_index(drop=True)
    if val.subject_id.tolist() != EXPECTED_VALIDATION_SUBJECTS or len(val) != 12 or int(val.n_trials.sum()) != EXPECTED_VALIDATION_TRIALS or (not val.has_raw_eeg.astype(int).eq(1).all()):
        raise RuntimeError('prespecified Validation roster differs from expected Phase-0 split')
    val['validation_geometry_transfer_design_roster_only_not_opened'] = True
    atomic_csv(outputs['roster'], val)
    pooled = pre.groupby('feature_name', sort=False)['pooled_participant_balanced_iqr'].agg(['min', 'max', 'nunique']).reset_index()
    for name, exp in prespecified_FALLBACK_IQR.items():
        row = pooled.loc[pooled.feature_name.eq(name)]
        if len(row) != 1 or int(row['nunique'].iloc[0]) != 1 or abs(float(row['min'].iloc[0]) - exp) > 1e-12 or (abs(float(row['max'].iloc[0]) - exp) > 1e-12):
            raise RuntimeError(f'prespecified fallback IQR mismatch: {name}')
    ref_pre = pd.DataFrame([{'feature_order_0based': i, 'feature_name': name, 'pooled_participant_balanced_iqr': prespecified_FALLBACK_IQR[name]} for i, name in enumerate(FEATURES)])
    atomic_csv(outputs['preprocessing'], ref_pre)
    ctr2 = ctr.sort_values(['trajectory_family_1based', 'feature_order_0based']).reset_index(drop=True)
    expected_ctr = np.array([prespecified_CENTROIDS[1], prespecified_CENTROIDS[2]])
    if not np.allclose(ctr2.centroid_value.to_numpy(float).reshape(2, 6), expected_ctr, rtol=0, atol=1e-12):
        raise RuntimeError('prespecified centroid values mismatch')
    atomic_csv(outputs['centroids'], ctr2)
    agree = lopo.loc[lopo.assignment_agrees_with_full_fit.map(bval)]
    qdist = float(agree.assigned_squared_distance.quantile(0.995))
    qmargin = float(agree.assignment_margin.quantile(0.01))
    if abs(qdist - prespecified_REFERENCE_THRESHOLDS['in_support_squared_distance_max']) > 1e-12 or abs(qmargin - prespecified_REFERENCE_THRESHOLDS['low_margin_threshold']) > 1e-12:
        raise RuntimeError('prespecified support threshold mismatch')
    policy = build_policy(core_sha)
    policy_sha = canonical_sha(policy)
    atomic_json(outputs['policy'], policy)
    reference = {'phase': PHASE, 'feature_order': FEATURES, 'pooled_fallback_iqr': prespecified_FALLBACK_IQR, 'fixed_centroids': prespecified_CENTROIDS, 'support_thresholds': prespecified_REFERENCE_THRESHOLDS, 'discovery_reference_counts': {'participants': 31, 'trajectories': 1237, 'family1': int((ass.trajectory_family_1based == 1).sum()), 'family2': int((ass.trajectory_family_1based == 2).sum()), 'lopo_agreeing_rows': int(agree.shape[0])}, 'transfer_policy_canonical_sha256': policy_sha, 'prespecified_transfer_core_sha256': core_sha}
    atomic_json(outputs['reference'], reference)
    atomic_csv(outputs['schema'], schema_frame())
    atomic_csv(outputs['gates'], gate_frame())
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'validation_geometry_transfer_design_prespecified_candidate', 'transfer_policy_canonical_sha256': policy_sha, 'prespecified_transfer_core_sha256': core_sha, 'counts': {'validation_roster_subjects': 12, 'validation_roster_trials': EXPECTED_VALIDATION_TRIALS, 'discovery_reference_participants': 31, 'discovery_reference_trajectories': 1237, 'source_hashes_verified': len(rows), 'prespecified_gates': len(gate_frame()), 'prespecified_output_schema_rows': len(schema_frame())}, 'decision': {'discovery_robustness_audit_integrity_null_structural_gates_passed': True, 'validation_files_read': False, 'validation_eeg_amplitude_read': False, 'validation_reports_outcomes_context_targets_read': False, 'confirmation_opened': False, 'family_count_or_centroids_changed': False, 'validation_source_epoch_lineage_source_epoch_specification_permitted': True, 'future_scope': 'outcome_free_trajectory_geometry_transfer_only'}}
    atomic_json(outputs['summary'], summary)
    write_report(outputs['report'], summary, policy_sha, core_sha)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'read only prespecified Validation roster/source metadata, raw EDF headers, and event timing to create a source/epoch lineage specification; do not read EEG amplitude or reports/outcomes', 'transfer_policy_canonical_sha256': policy_sha, 'prespecified_transfer_core_sha256': core_sha, 'required_boundary': ['verify every validation_geometry_transfer_design hash', 'use exactly the 12 prespecified Validation participants', 'read context only to apply prespecified epoch endpoint rules and never as a family target', 'do not read EEG amplitude in validation_source_epoch_lineage', 'do not read reports/outcomes/stimulus identity', 'do not alter epoch windows, preprocessing, feature schema, centroids, support thresholds, gates, or output schema', 'Confirmation remains closed'], 'later_prespecified_sequence': ['validation_preprocessing_qc', 'validation_feature_trajectory', 'validation_fixed_assignment'], 'validation_opened_for_source_lineage_only': False, 'confirmation_opened': False}
    atomic_json(outputs['next_policy'], next_policy)
    primary = {k: sha256(p) for k, p in outputs.items() if k not in {'contract', 'final_lock', 'log'} and p.exists()}
    contract = {'generated_at': now(), 'phase': PHASE, 'source_phase': SOURCE_PHASE, 'status': summary['status'], 'transfer_policy': policy, 'transfer_policy_canonical_sha256': policy_sha, 'prespecified_transfer_core_sha256': core_sha, 'source_discovery_robustness_audit_final_lock_sha256': sha256(inputs['h_final_lock']), 'source_cross_person_alignment_fit_final_lock_sha256': sha256(inputs['d_final_lock']), 'participant_split_manifest_sha256': sha256(inputs['split_manifest']), 'output_hashes': primary, 'validation_files_read': False, 'validation_opened': False, 'confirmation_opened': False, 'source_dataset_modified': False}
    atomic_json(outputs['contract'], contract)
    note('validation_geometry_transfer_design complete. Validation remains unread; only validation_source_epoch_lineage source/epoch lineage access is now permitted.')
    primary['contract'] = sha256(outputs['contract'])
    primary['next_policy'] = sha256(outputs['next_policy'])
    primary['log'] = sha256(outputs['log'])
    atomic_json(outputs['final_lock'], {'generated_at': now(), 'phase': PHASE, 'transfer_policy_canonical_sha256': policy_sha, 'validation_opened': False, 'confirmation_opened': False, 'hashes': primary})
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())