# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase validation_confirmation_synthesis — Validation synthesis and Confirmation analysis specification.

This phase performs no new EEG fitting, no family remapping, and no access to
Confirmation source files. It verifies the fixed Discovery interpretation and
robustness boundary, verifies the complete fixed Validation transfer result,
synthesizes the structural evidence, and specifications every operational detail of a
single final Confirmation transfer of the same outcome-free two-family
trajectory geometry.

The Confirmation scope is structural only. H1-H6, reports, outcomes, context
as a target, emotion category, stimulus identity, family renaming, centroid
refitting, threshold tuning, and discrete local-state taxonomy transfer remain
forbidden."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'validation_confirmation_synthesis'
NEXT_PHASE = 'confirmation_source_epoch_lineage'
PREFIX = 'validation_confirmation_synthesis'
EXPECTED_SPLIT_SHA = '0f6156e2a0de5af59dd3a7cd15b2a4a914dc0a0540b9d941fa6676a697f8d453'
EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA = 'bd3e7a5f4bd0cba00cfa66a0cd09208c07e0b1a9edab4b2a9493060200e1b37e'
EXPECTED_TRANSFER_CORE_SHA = '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b'
EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA = 'c088554da2135e5f397edc4301913f2fc2b6f15932ffbbc9384471719d669bf7'
EXPECTED_ATTACHMENT_RESOLUTION_SHA = 'db78b6578d8ea81ac40ebc6f6abd310a45ed1749d80d258dce0905213914a6d7'
EXPECTED_raw_eeg_epoch_folds_SHA = '93ca3cc67f7fb72f008845941a16f8a802e4b0aee601d53fbeb5c84c8e25aa8b'
EXPECTED_raw_eeg_preprocessing_qc_SHA = 'bd6a50669f0a6ba17965a4497433f1d4583d4a98f840dff29a73866d801e2ecb'
EXPECTED_outcome_free_features_SHA = 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba'
EXPECTED_validation_feature_trajectory_SHA = '444748f05f3c85eeacc183540df0180023ab5c4dba6c7c41b29c36a84d3c9f3c'
EXPECTED_validation_fixed_assignment_SHA = 'b94bd329b199aaa53cd2fd6d5dc46c9c37d6440519b689a73c7f8fc1ca7299e7'
CONFIRMATION_SUBJECTS = ['sub-03', 'sub-13', 'sub-19', 'sub-23', 'sub-27', 'sub-31', 'sub-35', 'sub-46', 'sub-47', 'sub-48', 'sub-49', 'sub-59']
EXPECTED_CONFIRMATION_TRIALS = 494
FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
G_REL = {'interpretation_policy': Path('analysis_specifications/S027_discovery_synthesis_fixed_interpretation.json'), 'validation_policy': Path('analysis_specifications/S030_discovery_synthesis_validation_gate.json'), 'summary': Path('outputs/qc/discovery_synthesis_discovery_synthesis_summary.json'), 'lock': Path('outputs/qc/discovery_synthesis_final_output_lock_hashes.json')}
H_REL = {'summary': Path('outputs/qc/discovery_robustness_audit_discovery_robustness_summary.json'), 'r8_summary': Path('outputs/qc/discovery_robustness_audit_r8_trajectory_structural_gate_summary.json'), 'next_policy': Path('analysis_specifications/S024_discovery_robustness_audit_validation_geometry_transfer.json'), 'lock': Path('outputs/qc/discovery_robustness_audit_final_output_lock_hashes.json')}
I_REL = {'policy': Path('analysis_specifications/S082_validation_geometry_transfer.json'), 'reference': Path('analysis_specifications/S081_validation_geometry_transfer_reference.json'), 'core': Path('analysis_specifications/reference_code/validation_geometry_transfer_core.py'), 'preprocessing': Path('outputs/tables/validation_geometry_transfer_design_discovery_reference_preprocessing.csv'), 'centroids': Path('outputs/tables/validation_geometry_transfer_design_discovery_reference_centroids.csv'), 'schema': Path('outputs/tables/validation_geometry_transfer_design_prespecified_transfer_output_schema.csv'), 'gates': Path('outputs/tables/validation_geometry_transfer_design_prespecified_transfer_gate_manifest.csv'), 'lock': Path('outputs/qc/validation_geometry_transfer_design_final_output_lock_hashes.json')}
J_REL = {'contract': Path('config/validation_source_epoch_lineage_validation_source_epoch_lineage_contract.json'), 'summary': Path('outputs/qc/validation_source_epoch_lineage_validation_source_epoch_lineage_summary.json'), 'lock': Path('outputs/qc/validation_source_epoch_lineage_final_output_lock_hashes.json'), 'executor': Path('src/analysis_pipeline/eeg/validation_source_epoch_lineage/validation_source_epoch_lineage.py')}
M_REL = {'source_rehash': Path('outputs/qc/validation_fixed_assignment_source_rehash.csv'), 'geometry_preprocessing': Path('outputs/tables/validation_fixed_assignment_validation_geometry_preprocessing.csv'), 'standardized_geometry': Path('outputs/tables/validation_fixed_assignment_validation_standardized_geometry.csv'), 'assignments': Path('outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv'), 'participant_summary': Path('outputs/qc/validation_fixed_assignment_validation_participant_transfer_summary.csv'), 'gate': Path('outputs/qc/validation_fixed_assignment_validation_structural_transfer_gate.csv'), 'diagnostics': Path('outputs/qc/validation_fixed_assignment_validation_assignment_diagnostics.csv'), 'summary': Path('outputs/qc/validation_fixed_assignment_validation_structural_transfer_summary.json'), 'next': Path('analysis_specifications/S080_validation_fixed_assignment_confirmation_synthesis.json'), 'contract': Path('config/validation_fixed_assignment_validation_structural_transfer_contract.json'), 'report': Path('outputs/reports/validation_fixed_assignment_validation_structural_transfer_report.md'), 'lock': Path('outputs/qc/validation_fixed_assignment_final_output_lock_hashes.json')}

def parse_args():
    p = argparse.ArgumentParser(description='Synthesize fixed Validation transfer and specification Confirmation design.')
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def csha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def jread(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def atom_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    os.replace(tmp, path)

def require(paths: Iterable[Path]):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n  - ' + '\n  - '.join(missing))

def bval(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def output_paths(project: Path):
    return {'log': project / 'outputs/logs/validation_confirmation_synthesis_validation_confirmation_synthesis.log', 'source_rehash': project / 'outputs/qc/validation_confirmation_synthesis_source_rehash.csv', 'evidence': project / 'outputs/tables/validation_confirmation_synthesis_validation_confirmation_evidence_matrix.csv', 'roster': project / 'outputs/tables/validation_confirmation_synthesis_confirmation_roster_manifest.csv', 'preprocessing': project / 'outputs/tables/validation_confirmation_synthesis_discovery_reference_preprocessing.csv', 'centroids': project / 'outputs/tables/validation_confirmation_synthesis_discovery_reference_centroids.csv', 'schema': project / 'outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_output_schema.csv', 'gates': project / 'outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_gate_manifest.csv', 'reference': project / 'analysis_specifications/S074_validation_confirmation_synthesis_geometry_reference.json', 'policy': project / 'analysis_specifications/S075_validation_confirmation_synthesis_geometry_transfer.json', 'attachment_policy': project / 'analysis_specifications/S073_validation_confirmation_synthesis_event_attachment.json', 'next': project / 'analysis_specifications/S076_validation_confirmation_synthesis_source_epoch.json', 'core': project / 'analysis_specifications/reference_code/validation_confirmation_transfer_core.py', 'lineage_reference': project / 'analysis_specifications/reference_code/validation_lineage_reference.py', 'summary': project / 'outputs/qc/validation_confirmation_synthesis_validation_confirmation_synthesis_summary.json', 'report': project / 'outputs/reports/validation_confirmation_synthesis_validation_confirmation_synthesis_report.md', 'contract': project / 'config/validation_confirmation_synthesis_validation_confirmation_synthesis_contract.json', 'lock': project / 'outputs/qc/validation_confirmation_synthesis_final_output_lock_hashes.json'}

def verify_lock_selected(project: Path, rels: dict[str, Path], phase: str, lock_key_map: dict[str, str] | None=None):
    paths = {k: project / v for k, v in rels.items()}
    require(paths.values())
    lock = jread(paths['lock'])
    if lock.get('phase') != phase:
        raise RuntimeError(f'{phase} final-lock phase mismatch')
    fixed = lock.get('hashes', {})
    rows = []
    for key, path in paths.items():
        if key == 'lock':
            continue
        lk = lock_key_map.get(key, key) if lock_key_map else key
        if lk not in fixed:
            raise RuntimeError(f'{phase} final lock missing {lk}')
        observed = sha(path)
        expected = str(fixed[lk])
        ok = observed == expected
        rows.append({'source_phase': phase, 'artifact_key': key, 'relative_path': str(rels[key]), 'expected_sha256': expected, 'observed_sha256': observed, 'passed': ok})
        if not ok:
            raise RuntimeError(f'{phase} fixed output mismatch: {key}')
    rows.append({'source_phase': phase, 'artifact_key': 'lock', 'relative_path': str(rels['lock']), 'expected_sha256': sha(paths['lock']), 'observed_sha256': sha(paths['lock']), 'passed': True})
    return (paths, pd.DataFrame(rows))

def verify_sources(project: Path):
    gpaths, gr = verify_lock_selected(project, G_REL, 'discovery_synthesis')
    hpaths, hr = verify_lock_selected(project, H_REL, 'discovery_robustness_audit')
    ipaths, ir = verify_lock_selected(project, I_REL, 'validation_geometry_transfer_design')
    jpaths, jr = verify_lock_selected(project, {k: v for k, v in J_REL.items() if k != 'executor'}, 'validation_source_epoch_lineage')
    mpaths, mr = verify_lock_selected(project, M_REL, 'validation_fixed_assignment')
    require([project / J_REL['executor'], project / 'config/participant_split_manifest.v1.csv'])
    if sha(project / J_REL['executor']) != EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA:
        raise RuntimeError('Unexpected final validation_source_epoch_lineage executor hash')
    split = project / 'config/participant_split_manifest.v1.csv'
    if sha(split) != EXPECTED_SPLIT_SHA:
        raise RuntimeError('Participant split manifest hash mismatch')
    extra = pd.DataFrame([{'source_role': 'participant_split_manifest', 'artifact_key': 'participant_split_manifest', 'relative_path': 'config/participant_split_manifest.v1.csv', 'expected_sha256': EXPECTED_SPLIT_SHA, 'observed_sha256': sha(split), 'passed': True}, {'source_phase': 'validation_source_epoch_lineage', 'artifact_key': 'executor', 'relative_path': str(J_REL['executor']), 'expected_sha256': EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA, 'observed_sha256': sha(project / J_REL['executor']), 'passed': True}])
    return (gpaths, hpaths, ipaths, jpaths, mpaths, split, pd.concat([gr, hr, ir, jr, mr, extra], ignore_index=True))

def validate_boundaries(gpaths, hpaths, ipaths, jpaths, mpaths):
    interpretation = jread(gpaths['interpretation_policy'])
    gsummary = jread(gpaths['summary'])
    hsummary = jread(hpaths['summary'])
    r8 = jread(hpaths['r8_summary'])
    hnext = jread(hpaths['next_policy'])
    ipolicy = jread(ipaths['policy'])
    iref = jread(ipaths['reference'])
    jcontract = jread(jpaths['contract'])
    jsummary = jread(jpaths['summary'])
    msummary = jread(mpaths['summary'])
    mnext = jread(mpaths['next'])
    mcontract = jread(mpaths['contract'])
    if interpretation.get('renaming_permitted') or interpretation.get('global_mapping_support'):
        raise RuntimeError('Discovery interpretation boundary changed')
    if not interpretation.get('trajectory_geometry_supported_as_structure_only') or interpretation.get('confirmation_opened'):
        raise RuntimeError('Discovery structure-only boundary invalid')
    hd = hsummary.get('decision', {})
    for k in ['R1_pass', 'R7_pass', 'R8_pass', 'R9_pass', 'all_prespecified_checks_executed']:
        if not hd.get(k):
            raise RuntimeError(f'discovery_robustness_audit required decision failed: {k}')
    if not r8.get('R8_pass') or int(r8.get('participants', -1)) != 31:
        raise RuntimeError('discovery_robustness_audit R8 structural gate invalid')
    if hnext.get('permitted_scope') != 'outcome_free_trajectory_geometry_transfer_only':
        raise RuntimeError('discovery_robustness_audit scope changed')
    if csha(ipolicy) != EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_geometry_transfer_design canonical transfer-policy mismatch')
    if sha(ipaths['core']) != EXPECTED_TRANSFER_CORE_SHA:
        raise RuntimeError('validation_geometry_transfer_design transfer core mismatch')
    if str(iref.get('prespecified_transfer_core_sha256')) != EXPECTED_TRANSFER_CORE_SHA:
        raise RuntimeError('validation_geometry_transfer_design reference core lineage mismatch')
    if str(jcontract.get('attachment_resolution_sha256')) != EXPECTED_ATTACHMENT_RESOLUTION_SHA:
        raise RuntimeError('validation_source_epoch_lineage attachment resolution mismatch')
    if jcontract.get('source_code_lineage', {}).get('raw_eeg_epoch_folds') != EXPECTED_raw_eeg_epoch_folds_SHA:
        raise RuntimeError('raw_eeg_epoch_folds lineage mismatch')
    if jcontract.get('source_code_lineage', {}).get('raw_eeg_preprocessing_qc') != EXPECTED_raw_eeg_preprocessing_qc_SHA:
        raise RuntimeError('raw_eeg_preprocessing_qc lineage mismatch')
    if jcontract.get('source_code_lineage', {}).get('outcome_free_features') != EXPECTED_outcome_free_features_SHA:
        raise RuntimeError('outcome_free_features lineage mismatch')
    jd = jsummary.get('decision', {})
    if not jd.get('all_three_epoch_branches_permitted_for_validation_preprocessing_qc') or jd.get('reports_outcomes_read') or jd.get('eeg_amplitude_read'):
        raise RuntimeError('validation_source_epoch_lineage boundary invalid')
    md = msummary.get('decision', {})
    counts = msummary.get('counts', {})
    metrics = msummary.get('metrics', {})
    if not md.get('all_structural_transfer_gates_passed') or not md.get('two_family_outcome_free_geometry_replication_supported'):
        raise RuntimeError('Validation structural transfer did not pass')
    forbidden_true = ['hedonic_emotional_contextual_or_stimulus_interpretation_supported', 'centroids_refit', 'thresholds_tuned', 'families_renamed', 'reports_outcomes_read', 'context_target_read', 'stimulus_identity_read', 'confirmation_opened']
    if any((md.get(k) for k in forbidden_true)):
        raise RuntimeError('Validation boundary violation')
    expected_counts = {'validation_subjects': 12, 'authoritative_trials': 494, 'assigned_trajectories': 437, 'eligible_participants': 12, 'family1_trials': 263, 'family2_trials': 174, 'participants_with_both_families': 12, 'in_support_trajectories': 432, 'low_margin_trajectories': 5}
    for k, v in expected_counts.items():
        if int(counts.get(k, -1)) != v:
            raise RuntimeError(f'validation_fixed_assignment count mismatch: {k}')
    if abs(float(metrics.get('participant_balanced_in_support_fraction', -1)) - 0.9885616120645948) > 1e-12:
        raise RuntimeError('Validation support metric changed')
    if mnext.get('next_phase') != PHASE or not mnext.get('next_phase_permitted') or (not mnext.get('confirmation_design_specification_permitted')):
        raise RuntimeError('validation_fixed_assignment did not authorize validation_confirmation_synthesis')
    if str(mcontract.get('executor_sha256')) != EXPECTED_validation_fixed_assignment_SHA:
        raise RuntimeError('Unexpected validation_fixed_assignment executor')
    return (interpretation, gsummary, hsummary, r8, ipolicy, iref, jcontract, msummary)

def confirmation_roster(split_path: Path):
    df = pd.read_csv(split_path, low_memory=False)
    required = ['subject_id', 'split', 'seed', 'split_version', 'n_trials', 'has_raw_eeg']
    if any((c not in df.columns for c in required)):
        raise RuntimeError('Split manifest schema changed')
    c = df[df['split'].astype(str).eq('confirmation')][required].copy().sort_values('subject_id', kind='mergesort').reset_index(drop=True)
    if c.subject_id.astype(str).tolist() != CONFIRMATION_SUBJECTS:
        raise RuntimeError('Confirmation roster changed')
    if len(c) != 12 or int(c.n_trials.sum()) != EXPECTED_CONFIRMATION_TRIALS or (not c.has_raw_eeg.map(bval).all()):
        raise RuntimeError('Invalid Confirmation roster')
    c['confirmation_accessed_in_validation_confirmation_synthesis'] = False
    c['planned_attachment_branch'] = np.where(c.n_trials.astype(int).eq(42), 'complete_42_trial_global_event_grammar', 'retained_trial_exact_rating_anchor')
    return c

def self_test():
    obj = {'b': 2, 'a': [1, 3]}
    assert csha(obj) == csha({'a': [1, 3], 'b': 2})
    x = pd.DataFrame({'subject_id': CONFIRMATION_SUBJECTS, 'n_trials': [42] * 11 + [32], 'split': ['confirmation'] * 12, 'seed': [540] * 12, 'split_version': ['v1'] * 12, 'has_raw_eeg': [1] * 12})
    assert len(x) == 12 and int(x.n_trials.sum()) == 494
    print('validation_confirmation_synthesis self-test: PASS')

def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    out = output_paths(project)
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not args.force):
        raise FileExistsError('validation_confirmation_synthesis outputs already exist; use --force')
    gpaths, hpaths, ipaths, jpaths, mpaths, split, rehash = verify_sources(project)
    print(f'[{now()}] Verified Discovery interpretation/robustness, Validation specification/lineage, and complete validation_fixed_assignment structural transfer.')
    interpretation, gsummary, hsummary, r8, ipolicy, iref, jcontract, msummary = validate_boundaries(gpaths, hpaths, ipaths, jpaths, mpaths)
    roster = confirmation_roster(split)
    print(f'[{now()}] Confirmation roster prespecified from Phase 0 split only; no Confirmation source file opened.')
    shutil.copyfile(ipaths['preprocessing'], out['preprocessing'])
    shutil.copyfile(ipaths['centroids'], out['centroids'])
    shutil.copyfile(ipaths['schema'], out['schema'])
    shutil.copyfile(ipaths['gates'], out['gates'])
    shutil.copyfile(ipaths['core'], out['core'])
    shutil.copyfile(project / J_REL['executor'], out['lineage_reference'])
    roster.to_csv(out['roster'], index=False)
    rehash.to_csv(out['source_rehash'], index=False)
    evidence = pd.DataFrame([{'stage': 'Discovery', 'evidence_id': 'D1', 'scope': 'outcome-free two-family trajectory geometry', 'participants': 31, 'trajectories': 1237, 'family1_trials': 749, 'family2_trials': 488, 'structural_metric_name': 'overall_LOPO_agreement', 'structural_metric_value': float(r8['overall_lopo_agreement']), 'gate_passed': True, 'interpretation': 'structure_only'}, {'stage': 'Discovery', 'evidence_id': 'D2', 'scope': 'global report/context/stimulus mapping H1-H6', 'participants': 31, 'trajectories': 1237, 'family1_trials': 749, 'family2_trials': 488, 'structural_metric_name': 'supported_parent_tests', 'structural_metric_value': 0.0, 'gate_passed': False, 'interpretation': 'no_hedonic_emotional_contextual_or_stimulus_label'}, {'stage': 'Validation', 'evidence_id': 'V1', 'scope': 'fixed centroid transfer without refit', 'participants': 12, 'trajectories': 437, 'family1_trials': 263, 'family2_trials': 174, 'structural_metric_name': 'participant_balanced_in_support_fraction', 'structural_metric_value': float(msummary['metrics']['participant_balanced_in_support_fraction']), 'gate_passed': True, 'interpretation': 'structure_only'}, {'stage': 'Validation', 'evidence_id': 'V2', 'scope': 'participants with both fixed families', 'participants': 12, 'trajectories': 437, 'family1_trials': 263, 'family2_trials': 174, 'structural_metric_name': 'participants_with_both_families', 'structural_metric_value': 12.0, 'gate_passed': True, 'interpretation': 'structure_only'}, {'stage': 'Confirmation_plan', 'evidence_id': 'outcome_free_features', 'scope': 'one final fixed structural transfer', 'participants': 12, 'trajectories': np.nan, 'family1_trials': np.nan, 'family2_trials': np.nan, 'structural_metric_name': 'authoritative_trial_roster', 'structural_metric_value': 494.0, 'gate_passed': True, 'interpretation': 'not_yet_executed'}])
    evidence.to_csv(out['evidence'], index=False)
    gates = pd.read_csv(ipaths['gates'], low_memory=False).copy()
    gates.insert(0, 'source_specification', 'validation_confirmation_synthesis')
    gates['target_cohort'] = 'confirmation'
    gates['threshold_changed_from_validation'] = False
    gates.to_csv(out['gates'], index=False)
    attachment_policy = {'phase': PHASE, 'fixed_before_confirmation_access': True, 'source_resolution': 'validation_source_epoch_lineage_validated_attachment_rule_EVENT_GRAMMAR_FINAL_GENERALIZED_BY_prespecified_ROSTER_SIZE', 'validation_reference_executor_sha256': EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA, 'validation_attachment_resolution_sha256': EXPECTED_ATTACHMENT_RESOLUTION_SHA, 'rules': [{'condition': 'authoritative n_trials == 42', 'method': 'exact README TypeIDs -> global primary non-overlapping event grammar -> fixed capacity 21 imagery plus 21 video -> chronological behavior mapping'}, {'condition': 'authoritative n_trials < 42', 'method': 'exact retained behavioral trial ID -> exact metadata rating anchor -> unique exact source stimulus between adjacent raw rating events; preserve original retained trial IDs'}], 'rating_anchor_join_key': 'metadata.trial_number, never metadata.global_trial', 'context_source': 'exact selected EEG trigger; metadata context is diagnostic only', 'within_subject_block_fold': 'five contiguous folds using the exact raw_eeg_epoch_folds rule', 'forbidden': ['Confirmation outcomes/reports', 'canonical context as selection target', 'metadata global_trial as behavioral join key', 'adjacent-rating heuristic for complete 42-trial subjects', 'alternate window selection after seeing Confirmation coverage']}
    atom_json(out['attachment_policy'], attachment_policy)
    attachment_canonical = csha(attachment_policy)
    policy = {'phase': PHASE, 'seed': 540, 'scientific_role': 'specification one final outcome-free transfer of the fixed Discovery two-family trajectory geometry to Confirmation after successful fixed Validation transfer', 'fixed_before_confirmation_access': True, 'confirmation_opened': False, 'scope': {'permitted': 'structural outcome-free trajectory geometry transfer only', 'prohibited': ['H1-H6 report/context/category/stimulus mapping in Confirmation', 'shared discrete local-state taxonomy transfer', 'family renaming', 'centroid refitting', 'threshold tuning', 'new feature selection', 'Validation-informed threshold adaptation']}, 'evidence_boundary': {'discovery_structure_supported': True, 'validation_fixed_structure_replication_supported': True, 'hedonic_emotional_contextual_or_stimulus_interpretation_supported': False}, 'staged_execution': [{'phase': 'confirmation_source_epoch_lineage', 'role': 'specification Confirmation source/epoch lineage and coverage using split roster, raw headers, annotations, and event timing only', 'eeg_amplitude_read': False}, {'phase': 'confirmation_preprocessing_qc', 'role': 'apply exact prespecified raw_eeg_preprocessing_qc preprocessing/QC to Confirmation raw EEG', 'reports_outcomes_read': False}, {'phase': 'confirmation_feature_trajectory', 'role': 'extract exact prespecified outcome_free_features compact features and construct within-participant three-epoch trajectories', 'reports_outcomes_read': False}, {'phase': 'confirmation_fixed_assignment', 'role': 'apply prespecified transforms and fixed Discovery centroids without refit; evaluate unchanged structural transfer gates', 'reports_outcomes_context_targets_read': False}, {'phase': 'three_cohort_structural_synthesis', 'role': 'final structural synthesis; no post hoc semantic mapping', 'confirmation_reports_outcomes_read': False}], 'source_code_lineage': {'raw_eeg_epoch_folds_epoch_specification_script_sha256': EXPECTED_raw_eeg_epoch_folds_SHA, 'raw_eeg_preprocessing_qc_preprocessing_script_sha256': EXPECTED_raw_eeg_preprocessing_qc_SHA, 'outcome_free_features_feature_script_sha256': EXPECTED_outcome_free_features_SHA, 'validation_feature_trajectory_executor_sha256': EXPECTED_validation_feature_trajectory_SHA, 'prespecified_transfer_core_sha256': EXPECTED_TRANSFER_CORE_SHA, 'validation_lineage_reference_sha256': EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA, 'attachment_policy_canonical_sha256': attachment_canonical}, 'confirmation_roster': {'split_version': 'v1', 'subjects': CONFIRMATION_SUBJECTS, 'expected_subjects': 12, 'expected_authoritative_trials': 494, 'all_have_raw_eeg_in_participant_roster': True, 'partial_roster_subjects': roster.loc[roster.n_trials.astype(int).lt(42), 'subject_id'].astype(str).tolist()}, 'operationally_allowed_metadata': {'subject_id': True, 'trial_number': True, 'raw_behavior_row_id': True, 'raw_run': True, 'epoch_family': True, 'context': 'allowed only as exact EEG-trigger lineage and prespecified epoch endpoint metadata; never as family target'}, 'forbidden_confirmation_columns_or_targets': ['valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion_category', 'stimulus_id', 'subjective factors', 'report-derived factors'], 'trajectory_construction': ipolicy['trajectory_construction'], 'fixed_assignment': ipolicy['fixed_assignment'], 'participant_eligibility': ipolicy['participant_eligibility'], 'structural_transfer_gate': ipolicy['structural_transfer_gate'], 'output_schema': ipolicy['output_schema'], 'decision_boundary': 'A successful Confirmation transfer may support only a third-cohort replication of the fixed two-family outcome-free geometric partition. It cannot support hedonic, emotional, contextual, stimulus, or causal interpretation.'}
    atom_json(out['policy'], policy)
    policy_canonical = csha(policy)
    reference = {'phase': PHASE, 'confirmation_transfer_policy_canonical_sha256': policy_canonical, 'attachment_policy_canonical_sha256': attachment_canonical, 'prespecified_transfer_core_sha256': sha(out['core']), 'validation_lineage_reference_sha256': sha(out['lineage_reference']), 'discovery_reference_preprocessing_sha256': sha(out['preprocessing']), 'discovery_reference_centroids_sha256': sha(out['centroids']), 'confirmation_roster_sha256': sha(out['roster']), 'prespecified_gate_manifest_sha256': sha(out['gates']), 'prespecified_output_schema_sha256': sha(out['schema']), 'family_labels': ['Family 1', 'Family 2'], 'renaming_permitted': False, 'centroid_refit_permitted': False, 'threshold_tuning_permitted': False}
    atom_json(out['reference'], reference)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'read only the prespecified Confirmation roster, safe source metadata, raw EDF headers, and annotations/event timing to create source/epoch lineage; do not read EEG amplitude, reports, outcomes, emotion category, stimulus identity, or Validation targets', 'required_confirmation_transfer_policy_canonical_sha256': policy_canonical, 'required_attachment_policy_canonical_sha256': attachment_canonical, 'required_prespecified_transfer_core_sha256': sha(out['core']), 'required_validation_lineage_reference_sha256': sha(out['lineage_reference']), 'required_boundary': ['verify every validation_confirmation_synthesis hash', 'use exactly the 12 prespecified Confirmation participants and 494 authoritative trials', 'apply the prespecified two-branch event attachment policy', 'read no EEG sample amplitude in confirmation_source_epoch_lineage', 'read no reports/outcomes/stimulus identity', 'do not alter epoch windows, preprocessing, features, centroids, support thresholds, structural gates, or output schema'], 'later_prespecified_sequence': ['confirmation_preprocessing_qc', 'confirmation_feature_trajectory', 'confirmation_fixed_assignment', 'three_cohort_structural_synthesis'], 'confirmation_opened_for_source_lineage_only': False}
    atom_json(out['next'], next_policy)
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'validation_structural_replication_synthesized_confirmation_design_prespecified_candidate', 'counts': {'discovery_participants': 31, 'discovery_trajectories': 1237, 'validation_participants': 12, 'validation_authoritative_trials': 494, 'validation_assigned_trajectories': 437, 'validation_family1_trials': 263, 'validation_family2_trials': 174, 'validation_in_support_trajectories': 432, 'confirmation_roster_subjects': 12, 'confirmation_roster_trials': 494, 'source_hashes_verified': int(len(rehash)), 'prespecified_confirmation_gates': int(len(gates))}, 'validation_metrics': msummary['metrics'], 'decision': {'discovery_structure_only_boundary_preserved': True, 'validation_fixed_structural_transfer_passed': True, 'two_family_geometry_replication_across_discovery_and_validation_supported': True, 'hedonic_emotional_contextual_or_stimulus_interpretation_supported': False, 'family_labels_renamed': False, 'centroids_refit': False, 'thresholds_tuned': False, 'validation_reports_outcomes_read': False, 'confirmation_files_read': False, 'confirmation_opened': False, 'confirmation_source_epoch_lineage_source_epoch_lineage_permitted': True, 'future_scope': 'outcome_free_trajectory_geometry_transfer_only'}, 'confirmation_transfer_policy_canonical_sha256': policy_canonical, 'attachment_policy_canonical_sha256': attachment_canonical}
    atom_json(out['summary'], summary)
    report = f"# Phase {PHASE}: Validation synthesis and Confirmation analysis specification\n\n## fixed result\n\nThe Discovery two-family trajectory geometry passed the complete pre-prespecified structural transfer gate in Validation without centroid refitting, threshold tuning, feature selection, family renaming, or report/outcome access. Validation contained 437 complete trajectories from all 12 participants: 263 assigned to Family 1 and 174 to Family 2. The participant-balanced in-support fraction was {msummary['metrics']['participant_balanced_in_support_fraction']:.6f}; all 12 participants contained both families.\n\nThis supports replication of an outcome-free geometric partition only. Discovery H1-H6 remained unsupported, so no hedonic, emotional, contextual, stimulus, or causal interpretation is licensed.\n\n## Confirmation specification\n\nConfirmation remains unopened in this phase. The Phase 0 split fixes 12 participants and 494 authoritative trials. The exact Discovery preprocessing, six geometry coordinates, monotone transformations, within-participant scaling, two centroids, distance rule, support thresholds, eligibility rules, structural gates, and output schema are prespecified unchanged.\n\nThe final event-lineage policy incorporates the resolved Validation event grammar: complete 42-trial rosters use the global non-overlapping 21/21 event grammar; incomplete retained rosters use exact retained rating anchors while preserving original trial IDs. Metadata `global_trial` is never used as the behavioral join key.\n\n## Permitted next phase\n{NEXT_PHASE} may open Confirmation only for source/header/annotation lineage. It may not read EEG amplitude, reports, outcomes, emotion category, or stimulus identity."
    out['report'].write_text(report, encoding='utf-8')
    primary_keys = ['source_rehash', 'evidence', 'roster', 'preprocessing', 'centroids', 'schema', 'gates', 'reference', 'policy', 'attachment_policy', 'next', 'core', 'lineage_reference', 'summary', 'report']
    hashes = {k: sha(out[k]) for k in primary_keys}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'executor_sha256': sha(Path(__file__).resolve()), 'source_lineage': {'participant_split_sha256': EXPECTED_SPLIT_SHA, 'validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA, 'validation_geometry_transfer_design_prespecified_transfer_core_sha256': EXPECTED_TRANSFER_CORE_SHA, 'validation_source_epoch_lineage_executor_sha256': EXPECTED_validation_source_epoch_lineage_validated_attachment_rule_EXECUTOR_SHA, 'validation_source_epoch_lineage_attachment_resolution_sha256': EXPECTED_ATTACHMENT_RESOLUTION_SHA, 'validation_fixed_assignment_executor_sha256': EXPECTED_validation_fixed_assignment_SHA}, 'output_hashes': hashes, 'boundary': summary['decision']}
    atom_json(out['contract'], contract)
    hashes['contract'] = sha(out['contract'])
    lock = {'generated_at': now(), 'phase': PHASE, 'confirmation_transfer_policy_canonical_sha256': policy_canonical, 'attachment_policy_canonical_sha256': attachment_canonical, 'hashes': hashes, 'confirmation_files_read': False, 'confirmation_opened': False, 'reports_outcomes_read': False, 'family_refit_or_rename': False}
    atom_json(out['lock'], lock)
    out['log'].write_text(f'[{now()}] validation_confirmation_synthesis complete. Validation structural transfer synthesized; Confirmation design prespecified without Confirmation access.', encoding='utf-8')
    stamp = tag()
    stage = project / '' / f'{PREFIX}_validation_confirmation_synthesis_{stamp}'
    if stage.exists():
        shutil.rmtree(stage)
    include = [out[k] for k in [*primary_keys, 'contract', 'lock']]
    for src in include:
        rel = src.relative_to(project)
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.rmtree(stage)
    print(f'[{now()}] Confirmation transfer policy canonical SHA-256: {policy_canonical}')
    print(f'[{now()}] Attachment policy canonical SHA-256: {attachment_canonical}')
    print(f'[{now()}] Confirmation roster: 12 participants / 494 trials; Confirmation files read: False')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())