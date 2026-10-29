# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase validation_fixed_assignment — fixed Validation trajectory-family transfer.

This executor applies only the validation_geometry_transfer_design-prespecified, outcome-free transfer pipeline:

1. verify every fixed validation_geometry_transfer_design design file and validation_feature_trajectory trajectory output;
2. import the byte-prespecified pure transfer core;
3. apply the prespecified monotone transforms and within-participant median/IQR
   scaling independently to each eligible Validation participant;
4. assign trajectories to the two fixed Discovery centroids by squared
   Euclidean distance, without refitting, relabeling, or threshold tuning;
5. evaluate the complete pre-prespecified structural transfer gate.

No report, outcome, emotion category, stimulus identity, or context target is
read. Confirmation remains closed. A successful gate supports only replication
of the outcome-free two-family geometric partition; it does not support any
hedonic, emotional, contextual, or stimulus interpretation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'validation_fixed_assignment'
NEXT_PHASE = 'validation_confirmation_synthesis'
PREFIX = 'validation_fixed_assignment'
EXPECTED_SUBJECTS = 12
EXPECTED_AUTHORITATIVE_TRIALS = 494
EXPECTED_TRAJECTORIES = 437
MIN_COMPLETE_PER_PARTICIPANT = 16
EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA = 'bd3e7a5f4bd0cba00cfa66a0cd09208c07e0b1a9edab4b2a9493060200e1b37e'
EXPECTED_validation_geometry_transfer_design_CORE_SHA = '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b'
EXPECTED_validation_feature_trajectory_EXECUTOR_SHA = '444748f05f3c85eeacc183540df0180023ab5c4dba6c7c41b29c36a84d3c9f3c'
FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
FORBIDDEN = {'valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion_category', 'stimulus_id', 'context', 'subjective_factor', 'subjective_factors', 'report_derived_factor'}
I_REL = {'policy': Path('analysis_specifications/S082_validation_geometry_transfer.json'), 'reference': Path('analysis_specifications/S081_validation_geometry_transfer_reference.json'), 'core': Path('analysis_specifications/reference_code/validation_geometry_transfer_core.py'), 'next_policy': Path('analysis_specifications/S083_validation_geometry_transfer_source_epoch.json'), 'roster': Path('outputs/tables/validation_geometry_transfer_design_validation_roster_manifest.csv'), 'preprocessing': Path('outputs/tables/validation_geometry_transfer_design_discovery_reference_preprocessing.csv'), 'centroids': Path('outputs/tables/validation_geometry_transfer_design_discovery_reference_centroids.csv'), 'schema': Path('outputs/tables/validation_geometry_transfer_design_prespecified_transfer_output_schema.csv'), 'gates': Path('outputs/tables/validation_geometry_transfer_design_prespecified_transfer_gate_manifest.csv'), 'source_rehash': Path('outputs/qc/validation_geometry_transfer_design_source_rehash.csv'), 'summary': Path('outputs/qc/validation_geometry_transfer_design_validation_geometry_transfer_design_summary.json'), 'report': Path('outputs/reports/validation_geometry_transfer_design_validation_geometry_transfer_design_report.md'), 'contract': Path('config/validation_geometry_transfer_design_validation_geometry_transfer_design_contract.json'), 'log': Path('outputs/logs/validation_geometry_transfer_design_validation_geometry_transfer_design_specification.log'), 'lock': Path('outputs/qc/validation_geometry_transfer_design_final_output_lock_hashes.json')}
L_REL = {'large_rehash': Path('outputs/qc/validation_feature_trajectory_validation_validation_preprocessing_qc_large_store_rehash.csv'), 'source_inventory': Path('outputs/qc/validation_feature_trajectory_validation_feature_checkpoint_inventory.csv'), 'feature_qc': Path('outputs/qc/validation_feature_trajectory_validation_compact_feature_qc.csv'), 'participant_feature_audit': Path('outputs/tables/validation_feature_trajectory_validation_participant_feature_audit.csv'), 'feature_schema': Path('outputs/tables/validation_feature_trajectory_validation_compact_feature_schema.csv'), 'epoch_manifest': Path('outputs/tables/validation_feature_trajectory_validation_compact_epoch_manifest.csv'), 'pca_scores': Path('outputs/tables/validation_feature_trajectory_validation_epoch_pca_scores.csv'), 'pca_loadings': Path('outputs/tables/validation_feature_trajectory_validation_pca_loadings.csv'), 'preprocessing': Path('outputs/tables/validation_feature_trajectory_validation_participant_preprocessing.csv'), 'trajectories': Path('outputs/tables/validation_feature_trajectory_validation_raw_trajectory_geometry.csv'), 'participant_qc': Path('outputs/qc/validation_feature_trajectory_validation_participant_trajectory_qc.csv'), 'gate': Path('outputs/qc/validation_feature_trajectory_validation_pre_assignment_transfer_gate.csv'), 'global_store_inventory': Path('outputs/tables/validation_feature_trajectory_validation_compact_store_inventory.csv'), 'summary': Path('outputs/qc/validation_feature_trajectory_validation_feature_trajectory_summary.json'), 'next': Path('analysis_specifications/S079_validation_feature_trajectory_fixed_assignment.json'), 'contract': Path('config/validation_feature_trajectory_validation_feature_trajectory_contract.json'), 'report': Path('outputs/reports/validation_feature_trajectory_validation_feature_trajectory_report.md'), 'lock': Path('outputs/qc/validation_feature_trajectory_final_output_lock_hashes.json')}

def parse_args():
    p = argparse.ArgumentParser(description='Apply fixed Discovery trajectory centroids to Validation.')
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

def json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)

def atom_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + '', encoding='utf-8')
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

def import_verified(path: Path, expected_sha: str, name: str):
    require([path])
    observed = sha(path)
    if observed != expected_sha:
        raise RuntimeError(f'{name} hash mismatch: expected {expected_sha}, observed {observed}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {name}: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def output_paths(project: Path):
    return {'log': project / 'outputs/logs/validation_fixed_assignment_validation_fixed_assignment.log', 'source_rehash': project / 'outputs/qc/validation_fixed_assignment_source_rehash.csv', 'geometry_preprocessing': project / 'outputs/tables/validation_fixed_assignment_validation_geometry_preprocessing.csv', 'standardized_geometry': project / 'outputs/tables/validation_fixed_assignment_validation_standardized_geometry.csv', 'assignments': project / 'outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv', 'participant_summary': project / 'outputs/qc/validation_fixed_assignment_validation_participant_transfer_summary.csv', 'gate': project / 'outputs/qc/validation_fixed_assignment_validation_structural_transfer_gate.csv', 'diagnostics': project / 'outputs/qc/validation_fixed_assignment_validation_assignment_diagnostics.csv', 'summary': project / 'outputs/qc/validation_fixed_assignment_validation_structural_transfer_summary.json', 'report': project / 'outputs/reports/validation_fixed_assignment_validation_structural_transfer_report.md', 'next': project / 'analysis_specifications/S080_validation_fixed_assignment_confirmation_synthesis.json', 'contract': project / 'config/validation_fixed_assignment_validation_structural_transfer_contract.json', 'lock': project / 'outputs/qc/validation_fixed_assignment_final_output_lock_hashes.json'}

def verify_fixed_phase(project: Path, rels: dict[str, Path], phase: str):
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
        if key not in fixed:
            raise RuntimeError(f'{phase} final lock missing {key}')
        observed = sha(path)
        expected = str(fixed[key])
        ok = observed == expected
        rows.append({'source_phase': phase, 'artifact_key': key, 'relative_path': str(rels[key]), 'expected_sha256': expected, 'observed_sha256': observed, 'passed': ok})
        if not ok:
            raise RuntimeError(f'{phase} fixed output mismatch: {key}')
    rows.append({'source_phase': phase, 'artifact_key': 'lock', 'relative_path': str(rels['lock']), 'expected_sha256': sha(paths['lock']), 'observed_sha256': sha(paths['lock']), 'passed': True})
    return (paths, pd.DataFrame(rows))

def verify_inputs(project: Path):
    ipaths, irh = verify_fixed_phase(project, I_REL, 'validation_geometry_transfer_design')
    lpaths, lrh = verify_fixed_phase(project, L_REL, 'validation_feature_trajectory')
    policy = jread(ipaths['policy'])
    reference = jread(ipaths['reference'])
    isummary = jread(ipaths['summary'])
    lsummary = jread(lpaths['summary'])
    lnext = jread(lpaths['next'])
    lcontract = jread(lpaths['contract'])
    if csha(policy) != EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_geometry_transfer_design transfer-policy canonical hash mismatch')
    if sha(ipaths['core']) != EXPECTED_validation_geometry_transfer_design_CORE_SHA:
        raise RuntimeError('validation_geometry_transfer_design prespecified transfer-core hash mismatch')
    if str(reference.get('transfer_policy_canonical_sha256')) != EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_geometry_transfer_design reference-policy lineage mismatch')
    if str(reference.get('prespecified_transfer_core_sha256')) != EXPECTED_validation_geometry_transfer_design_CORE_SHA:
        raise RuntimeError('validation_geometry_transfer_design reference-core lineage mismatch')
    if isummary.get('decision', {}).get('validation_files_read') or isummary.get('decision', {}).get('confirmation_opened'):
        raise RuntimeError('validation_geometry_transfer_design boundary violation')
    if lsummary.get('phase') != 'validation_feature_trajectory' or not lsummary.get('decision', {}).get('validation_fixed_assignment_assignment_permitted'):
        raise RuntimeError('validation_feature_trajectory did not permit validation_fixed_assignment')
    if lnext.get('next_phase') != PHASE or not lnext.get('next_phase_permitted'):
        raise RuntimeError('validation_feature_trajectory next-phase policy does not authorize validation_fixed_assignment')
    if str(lnext.get('required_validation_geometry_transfer_design_transfer_policy_canonical_sha256')) != EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_feature_trajectory transfer-policy lineage mismatch')
    if str(lnext.get('required_validation_geometry_transfer_design_prespecified_core_sha256')) != EXPECTED_validation_geometry_transfer_design_CORE_SHA:
        raise RuntimeError('validation_feature_trajectory core lineage mismatch')
    if str(lcontract.get('executor_sha256')) != EXPECTED_validation_feature_trajectory_EXECUTOR_SHA:
        raise RuntimeError('Unexpected validation_feature_trajectory executor')
    counts = lsummary.get('counts', {})
    expected = {'validation_subjects': 12, 'authoritative_trials': 494, 'complete_three_epoch_trajectories': 437, 'finite_geometry_trajectories': 437, 'eligible_participants_for_validation_fixed_assignment': 12}
    for k, v in expected.items():
        if int(counts.get(k, -1)) != v:
            raise RuntimeError(f'validation_feature_trajectory count mismatch for {k}: {counts.get(k)}')
    decision = lsummary.get('decision', {})
    if decision.get('reports_outcomes_read') or decision.get('context_target_read') or decision.get('stimulus_identity_read') or decision.get('family_assignment_performed') or decision.get('confirmation_opened'):
        raise RuntimeError('validation_feature_trajectory boundary violation')
    return (ipaths, lpaths, pd.concat([irh, lrh], ignore_index=True), policy, reference)

def read_fixed_inputs(ipaths: dict[str, Path], lpaths: dict[str, Path], core):
    roster = pd.read_csv(ipaths['roster'], usecols=['subject_id', 'split', 'n_trials', 'has_raw_eeg'], low_memory=False)
    if len(roster) != EXPECTED_SUBJECTS or roster.subject_id.nunique() != EXPECTED_SUBJECTS or int(roster.n_trials.sum()) != EXPECTED_AUTHORITATIVE_TRIALS:
        raise RuntimeError('validation_geometry_transfer_design Validation roster changed')
    if set(roster.split.astype(str)) != {'validation'} or not roster.has_raw_eeg.map(bval).all():
        raise RuntimeError('Invalid Validation roster')
    prep = pd.read_csv(ipaths['preprocessing'], low_memory=False).sort_values('feature_order_0based')
    ctr = pd.read_csv(ipaths['centroids'], low_memory=False).sort_values(['trajectory_family_1based', 'feature_order_0based'])
    if prep.feature_name.astype(str).tolist() != FEATURES:
        raise RuntimeError('Discovery preprocessing feature order changed')
    fallback = prep.pooled_participant_balanced_iqr.to_numpy(float)
    if fallback.shape != (6,) or not np.isfinite(fallback).all() or np.any(fallback <= 1e-08):
        raise RuntimeError('Invalid prespecified pooled fallback IQR')
    if len(ctr) != 12 or ctr.trajectory_family_1based.nunique() != 2:
        raise RuntimeError('Invalid fixed centroid table')
    for fam in [1, 2]:
        g = ctr[ctr.trajectory_family_1based.eq(fam)].sort_values('feature_order_0based')
        if g.standardized_geometry_feature.astype(str).tolist() != FEATURES:
            raise RuntimeError(f'Centroid feature order changed for family {fam}')
    centroids = np.vstack([ctr[ctr.trajectory_family_1based.eq(f)].sort_values('feature_order_0based').centroid_value.to_numpy(float) for f in [1, 2]])
    if centroids.shape != (2, 6) or not np.isfinite(centroids).all():
        raise RuntimeError('Invalid prespecified centroids')
    if list(core.FEATURES) != FEATURES:
        raise RuntimeError('prespecified core feature order changed')
    header = pd.read_csv(lpaths['trajectories'], nrows=0).columns.tolist()
    bad = sorted(set(header) & FORBIDDEN)
    if bad:
        raise RuntimeError(f'Forbidden target columns unexpectedly present in fixed trajectory source: {bad}')
    usecols = ['subject_id', 'raw_behavior_row_id', 'trial_number', *FEATURES, 'geometry_finite']
    traj = pd.read_csv(lpaths['trajectories'], usecols=usecols, low_memory=False)
    if len(traj) != EXPECTED_TRAJECTORIES or traj.raw_behavior_row_id.astype(str).duplicated().any():
        raise RuntimeError('validation_feature_trajectory trajectory table invalid')
    if set(traj.subject_id.astype(str)) != set(roster.subject_id.astype(str)):
        raise RuntimeError('Trajectory roster differs from prespecified Validation roster')
    if not traj.geometry_finite.map(bval).all() or not np.isfinite(traj[FEATURES].to_numpy(float)).all():
        raise RuntimeError('Nonfinite Validation geometry reached assignment')
    pqc = pd.read_csv(lpaths['participant_qc'], usecols=['subject_id', 'finite_geometry_trajectories', 'eligible_for_validation_fixed_assignment'], low_memory=False)
    merged = roster[['subject_id', 'n_trials']].merge(pqc, on='subject_id', how='left', validate='one_to_one')
    observed = traj.groupby('subject_id').size().rename('n').reset_index()
    merged = merged.merge(observed, on='subject_id', how='left', validate='one_to_one')
    if merged.isna().any().any() or not (merged.finite_geometry_trajectories.astype(int) == merged.n.astype(int)).all() or (not merged.eligible_for_validation_fixed_assignment.map(bval).all()):
        raise RuntimeError('validation_feature_trajectory participant eligibility mismatch')
    return (roster, traj, fallback, centroids)

def perform_transfer(roster: pd.DataFrame, traj: pd.DataFrame, fallback: np.ndarray, centroids: np.ndarray, core, reference: dict):
    assign_parts = []
    z_parts = []
    prep_parts = []
    for subject in roster.subject_id.astype(str):
        g = traj[traj.subject_id.astype(str).eq(subject)].copy().sort_values(['trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
        if len(g) < MIN_COMPLETE_PER_PARTICIPANT:
            continue
        transferred, prep = core.transfer_one_participant(g[['subject_id', 'raw_behavior_row_id', 'trial_number', *FEATURES]], fallback, centroids)
        assignments = transferred[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2']].copy()
        threshold = float(reference['support_thresholds']['in_support_squared_distance_max'])
        margin_threshold = float(reference['support_thresholds']['low_margin_threshold'])
        assignments['in_discovery_support'] = assignments.assigned_squared_distance.le(threshold)
        assignments['low_margin'] = assignments.assignment_margin.lt(margin_threshold)
        z = transferred[['subject_id', 'raw_behavior_row_id', 'trial_number', *[f'standardized_{x}' for x in FEATURES]]].copy()
        assign_parts.append(assignments)
        z_parts.append(z)
        prep_parts.append(prep)
    if not assign_parts:
        raise RuntimeError('No eligible Validation participant for transfer')
    assignments = pd.concat(assign_parts, ignore_index=True)
    standardized = pd.concat(z_parts, ignore_index=True)
    preprocessing = pd.concat(prep_parts, ignore_index=True)
    assignments = assignments.sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    standardized = standardized.sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    preprocessing = preprocessing.sort_values(['subject_id', 'feature_order_0based'], kind='mergesort').reset_index(drop=True)
    if len(assignments) != EXPECTED_TRAJECTORIES or assignments.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('Assignment output row integrity failure')
    if not assignments.trajectory_family_1based.isin([1, 2]).all():
        raise RuntimeError('Invalid family label')
    if not np.isfinite(assignments[['assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2']].to_numpy(float)).all():
        raise RuntimeError('Nonfinite assignment diagnostics')
    return (assignments, standardized, preprocessing)

def evaluate_gate(roster: pd.DataFrame, assignments: pd.DataFrame, source_hash_gates_pass: bool):
    rows = []
    for r in roster.sort_values('subject_id').itertuples(index=False):
        g = assignments[assignments.subject_id.astype(str).eq(str(r.subject_id))]
        n = len(g)
        eligible = n >= MIN_COMPLETE_PER_PARTICIPANT
        rows.append({'subject_id': str(r.subject_id), 'authoritative_trials': int(r.n_trials), 'complete_trajectories': n, 'eligible_for_transfer': eligible, 'family1_trials': int(g.trajectory_family_1based.eq(1).sum()), 'family2_trials': int(g.trajectory_family_1based.eq(2).sum()), 'in_support_fraction': float(g.in_discovery_support.mean()) if n else np.nan, 'low_margin_fraction': float(g.low_margin.mean()) if n else np.nan, 'both_families_present': bool(g.trajectory_family_1based.nunique() == 2) if n else False})
    participant = pd.DataFrame(rows)
    eligible = participant[participant.eligible_for_transfer].copy()
    if eligible.empty:
        raise RuntimeError('No eligible participant at transfer gate')
    complete_fraction = len(assignments) / EXPECTED_AUTHORITATIVE_TRIALS
    pb_support = float(eligible.in_support_fraction.mean())
    both = int(eligible.both_families_present.sum())
    f1 = int(assignments.trajectory_family_1based.eq(1).sum())
    f2 = int(assignments.trajectory_family_1based.eq(2).sum())
    occ1 = float((eligible.family1_trials / eligible.complete_trajectories).mean())
    occ2 = float((eligible.family2_trials / eligible.complete_trajectories).mean())
    gates = [('V1', 'eligible_participants', len(eligible), 9.0, '>=', len(eligible) >= 9), ('V2', 'complete_trajectory_fraction_of_494', complete_fraction, 0.75, '>=', complete_fraction >= 0.75), ('V3', 'participant_balanced_in_support_fraction', pb_support, 0.8, '>=', pb_support >= 0.8), ('V4', 'participants_with_both_families', both, 8.0, '>=', both >= 8), ('V5a', 'family1_trial_count', f1, 30.0, '>=', f1 >= 30), ('V5b', 'family2_trial_count', f2, 30.0, '>=', f2 >= 30), ('V6a', 'family1_participant_balanced_occupancy', occ1, 0.1, '>=', occ1 >= 0.1), ('V6b', 'family2_participant_balanced_occupancy', occ2, 0.1, '>=', occ2 >= 0.1), ('V7', 'source_and_algorithm_hash_gates', 1.0 if source_hash_gates_pass else 0.0, 1.0, '==', bool(source_hash_gates_pass)), ('V8', 'report_outcome_mapping_executed', 0.0, 0.0, '==', True)]
    gate = pd.DataFrame([{'gate_id': a, 'metric': b, 'observed': c, 'threshold': d, 'comparison': e, 'pass': f} for a, b, c, d, e, f in gates])
    diagnostics = pd.DataFrame([{'metric': 'overall_in_support_fraction', 'value': float(assignments.in_discovery_support.mean())}, {'metric': 'participant_balanced_in_support_fraction', 'value': pb_support}, {'metric': 'overall_low_margin_fraction', 'value': float(assignments.low_margin.mean())}, {'metric': 'participant_balanced_low_margin_fraction', 'value': float(eligible.low_margin_fraction.mean())}, {'metric': 'assigned_squared_distance_median', 'value': float(assignments.assigned_squared_distance.median())}, {'metric': 'assigned_squared_distance_max', 'value': float(assignments.assigned_squared_distance.max())}, {'metric': 'assignment_margin_median', 'value': float(assignments.assignment_margin.median())}, {'metric': 'assignment_margin_min', 'value': float(assignments.assignment_margin.min())}])
    return (participant, gate, diagnostics)

def self_test(project: Path):
    core = import_verified(project / I_REL['core'], EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'validation_geometry_transfer_design_transfer_core_selftest')
    rng = np.random.default_rng(540)
    n = 24
    frame = pd.DataFrame({'subject_id': ['sub-test'] * n, 'raw_behavior_row_id': [f'test-{i:03d}' for i in range(n)], 'trial_number': np.arange(1, n + 1), 'early_to_late_norm': rng.lognormal(0.9, 0.25, n), 'late_to_rating_norm': rng.lognormal(1.0, 0.25, n), 'early_to_rating_direct_norm': rng.lognormal(0.95, 0.2, n), 'total_path_length': rng.lognormal(1.7, 0.2, n), 'directness_ratio': rng.uniform(0.15, 0.9, n), 'segment_cosine': rng.uniform(-0.9, 0.8, n)})
    fallback = np.array([0.4, 0.4, 0.4, 0.4, 1.0, 0.6])
    centroids = np.array([[-0.3, -0.3, 0, -0.3, 0.3, 0.3], [0.7, 0.6, 0.2, 0.7, -0.6, -0.6]])
    out, prep = core.transfer_one_participant(frame, fallback, centroids)
    assert len(out) == n and len(prep) == 6 and out.trajectory_family_1based.isin([1, 2]).all()
    midpoint = (centroids[0] + centroids[1]) / 2
    label, *_ = core.assign_fixed_families(midpoint[None, :], centroids)
    assert int(label[0]) == 1
    print('validation_fixed_assignment self-test: PASS')

def main():
    args = parse_args()
    project = args.project.resolve()
    if args.self_test:
        self_test(project)
        return 0
    out = output_paths(project)
    out['log'].parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"validation_fixed_assignment outputs already exist: {out['contract']}; use --force after review")
    started = time.time()
    logf = out['log'].open('a', encoding='utf-8')

    def note(msg):
        print(f'[{now()}] {msg}', flush=True)
        print(f'[{now()}] {msg}', file=logf, flush=True)
    script_sha = sha(Path(__file__).resolve())
    try:
        ipaths, lpaths, rehash, policy, reference = verify_inputs(project)
        core = import_verified(ipaths['core'], EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'prespecified_validation_geometry_transfer_design_transfer_core')
        roster, traj, fallback, centroids = read_fixed_inputs(ipaths, lpaths, core)
        note('Verified validation_geometry_transfer_design analysis specification, validation_feature_trajectory trajectories, exact prespecified transfer core, centroids, thresholds, and all fixed boundaries.')
        assignments, standardized, preprocessing = perform_transfer(roster, traj, fallback, centroids, core, reference)
        participant, gate, diagnostics = evaluate_gate(roster, assignments, source_hash_gates_pass=True)
        gate_passed = bool(gate['pass'].all())
        for p in [out['source_rehash'], out['geometry_preprocessing'], out['standardized_geometry'], out['assignments'], out['participant_summary'], out['gate'], out['diagnostics']]:
            p.parent.mkdir(parents=True, exist_ok=True)
        rehash.to_csv(out['source_rehash'], index=False)
        preprocessing.to_csv(out['geometry_preprocessing'], index=False)
        standardized.to_csv(out['standardized_geometry'], index=False)
        assignments.to_csv(out['assignments'], index=False)
        participant.to_csv(out['participant_summary'], index=False)
        gate.to_csv(out['gate'], index=False)
        diagnostics.to_csv(out['diagnostics'], index=False)
        primary = {'source_rehash': out['source_rehash'], 'geometry_preprocessing': out['geometry_preprocessing'], 'standardized_geometry': out['standardized_geometry'], 'assignments': out['assignments'], 'participant_summary': out['participant_summary'], 'gate': out['gate'], 'diagnostics': out['diagnostics']}
        ph = {k: sha(p) for k, p in primary.items()}
        fam_counts = assignments.trajectory_family_1based.value_counts().sort_index().to_dict()
        eligible = participant[participant.eligible_for_transfer]
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'validation_fixed_two_family_geometry_transfer_complete_candidate', 'source_lineage': {'validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA, 'validation_geometry_transfer_design_prespecified_transfer_core_sha256': EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'validation_feature_trajectory_executor_sha256': EXPECTED_validation_feature_trajectory_EXECUTOR_SHA}, 'counts': {'validation_subjects': int(roster.subject_id.nunique()), 'authoritative_trials': int(roster.n_trials.sum()), 'assigned_trajectories': len(assignments), 'eligible_participants': int(len(eligible)), 'family1_trials': int(fam_counts.get(1, 0)), 'family2_trials': int(fam_counts.get(2, 0)), 'participants_with_both_families': int(eligible.both_families_present.sum()), 'in_support_trajectories': int(assignments.in_discovery_support.sum()), 'low_margin_trajectories': int(assignments.low_margin.sum())}, 'metrics': {'complete_trajectory_fraction_of_494': float(len(assignments) / EXPECTED_AUTHORITATIVE_TRIALS), 'participant_balanced_in_support_fraction': float(eligible.in_support_fraction.mean()), 'participant_balanced_family1_occupancy': float((eligible.family1_trials / eligible.complete_trajectories).mean()), 'participant_balanced_family2_occupancy': float((eligible.family2_trials / eligible.complete_trajectories).mean()), 'overall_low_margin_fraction': float(assignments.low_margin.mean())}, 'decision': {'all_structural_transfer_gates_passed': gate_passed, 'two_family_outcome_free_geometry_replication_supported': gate_passed, 'hedonic_emotional_contextual_or_stimulus_interpretation_supported': False, 'centroids_refit': False, 'thresholds_tuned': False, 'families_renamed': False, 'reports_outcomes_read': False, 'context_target_read': False, 'stimulus_identity_read': False, 'confirmation_opened': False, 'validation_confirmation_synthesis_permitted': True, 'confirmation_design_specification_permitted_in_validation_confirmation_synthesis': gate_passed}, 'runtime_seconds': time.time() - started}
        atom_json(out['summary'], summary)
        next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'synthesize the fixed Validation structural-transfer result and specification the Confirmation decision without opening reports/outcomes or Confirmation data', 'structural_transfer_gate_passed': gate_passed, 'confirmation_design_specification_permitted': gate_passed, 'required_inputs': {**ph, 'summary': sha(out['summary'])}, 'required_validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA, 'required_validation_geometry_transfer_design_prespecified_core_sha256': EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'forbidden': ['family relabeling', 'centroid refit', 'threshold tuning', 'H1-H6 mapping', 'Validation reports/outcomes/context targets', 'Confirmation data access'], 'validation_opened_for_fixed_geometry_assignment_only': True, 'confirmation_opened': False}
        atom_json(out['next'], next_policy)
        contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'executor_sha256': script_sha, 'validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_POLICY_CANONICAL_SHA, 'validation_geometry_transfer_design_prespecified_transfer_core_sha256': EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'validation_geometry_transfer_design_input_hashes': jread(ipaths['lock'])['hashes'], 'validation_feature_trajectory_input_hashes': jread(lpaths['lock'])['hashes'], 'output_hashes': {**ph, 'summary': sha(out['summary']), 'next': sha(out['next'])}, 'boundary': summary['decision']}
        atom_json(out['contract'], contract)
        report = ''.join(['# Phase validation_fixed_assignment — Validation fixed trajectory-family transfer', '', f'- Generated: `{now()}`', f'- Assigned trajectories: **{len(assignments)}/{EXPECTED_AUTHORITATIVE_TRIALS}**', f'- Eligible participants: **{len(eligible)}/{EXPECTED_SUBJECTS}**', f'- Family 1 trials: **{int(fam_counts.get(1, 0))}**', f'- Family 2 trials: **{int(fam_counts.get(2, 0))}**', f'- Participant-balanced in-support fraction: **{float(eligible.in_support_fraction.mean()):.6f}**', f'- Participants with both families: **{int(eligible.both_families_present.sum())}/{len(eligible)}**', f'- All prespecified structural-transfer gates passed: **{gate_passed}**', '', '## Interpretation boundary', '', '- A pass supports replication of the fixed two-family outcome-free geometric partition only.', '- It does not support hedonic, emotional, contextual, category, or stimulus interpretation.', '- Centroids, thresholds, feature set, family count, and family names were not changed.', '- Reports/outcomes/context targets were not read; Confirmation remains closed.', ''])
        out['report'].parent.mkdir(parents=True, exist_ok=True)
        out['report'].write_text(report, encoding='utf-8')
        lock_files = {**primary, 'summary': out['summary'], 'next': out['next'], 'contract': out['contract'], 'report': out['report']}
        atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': {k: sha(p) for k, p in lock_files.items()}, 'structural_transfer_gate_passed': gate_passed, 'reports_outcomes_read': False, 'context_target_read': False, 'family_refit_or_rename': False, 'confirmation_opened': False})
        files = list(lock_files.values()) + [out['lock'], Path(__file__).resolve(), ipaths['policy'], ipaths['reference'], ipaths['core'], ipaths['roster'], ipaths['preprocessing'], ipaths['centroids'], ipaths['gates'], ipaths['schema'], ipaths['lock'], lpaths['trajectories'], lpaths['participant_qc'], lpaths['gate'], lpaths['summary'], lpaths['next'], lpaths['contract'], lpaths['lock']]
        note(f'validation_fixed_assignment complete. Gate passed={gate_passed}; Family1={int(fam_counts.get(1, 0))}; Family2={int(fam_counts.get(2, 0))}; participant-balanced support={float(eligible.in_support_fraction.mean()):.6f}.')
        return 0
    finally:
        logf.close()
if __name__ == '__main__':
    raise SystemExit(main())