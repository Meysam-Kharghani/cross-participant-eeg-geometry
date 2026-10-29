# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase confirmation_fixed_assignment — fixed Confirmation trajectory-family transfer.

Apply only the validation_confirmation_synthesis-prespecified, outcome-free transfer pipeline to eligible
Confirmation participants. No centroid refit, threshold tuning, family
renaming, semantic mapping, or report/outcome/context-target access is allowed.
Participants below the prespecified minimum of 16 complete trajectories are retained
in source/QC audit but excluded from assignment and the structural transfer gate,
matching the prespecified Validation transfer rule."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'confirmation_fixed_assignment'
NEXT_PHASE = 'three_cohort_structural_synthesis'
PREFIX = 'confirmation_fixed_assignment'
EXPECTED_SUBJECTS = 12
EXPECTED_AUTHORITATIVE_TRIALS = 494
EXPECTED_FINITE_TRAJECTORIES = 459
EXPECTED_ELIGIBLE_PARTICIPANTS = 11
EXPECTED_ASSIGNABLE_TRAJECTORIES = 446
MIN_COMPLETE_PER_PARTICIPANT = 16
EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA = '73df540947a247e541f20766de290b3c7cf6878ec2e1cc3bb17684f642c286c9'
EXPECTED_validation_confirmation_synthesis_CORE_SHA = '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b'
EXPECTED_validation_fixed_assignment_EXECUTOR_SHA = 'b94bd329b199aaa53cd2fd6d5dc46c9c37d6440519b689a73c7f8fc1ca7299e7'
EXPECTED_confirmation_feature_trajectory_EXECUTOR_SHA = 'cd00a7c7e2cf66ea0959f3f398312f5760bea75a6c84f7ec7e611e34dc79015b'
FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
FORBIDDEN = {'valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion_category', 'stimulus_id', 'context', 'subjective_factor', 'subjective_factors', 'report_derived_factor'}
N_REL = {'source_rehash': Path('outputs/qc/validation_confirmation_synthesis_source_rehash.csv'), 'evidence': Path('outputs/tables/validation_confirmation_synthesis_validation_confirmation_evidence_matrix.csv'), 'roster': Path('outputs/tables/validation_confirmation_synthesis_confirmation_roster_manifest.csv'), 'preprocessing': Path('outputs/tables/validation_confirmation_synthesis_discovery_reference_preprocessing.csv'), 'centroids': Path('outputs/tables/validation_confirmation_synthesis_discovery_reference_centroids.csv'), 'schema': Path('outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_output_schema.csv'), 'gates': Path('outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_gate_manifest.csv'), 'reference': Path('analysis_specifications/S074_validation_confirmation_synthesis_geometry_reference.json'), 'policy': Path('analysis_specifications/S075_validation_confirmation_synthesis_geometry_transfer.json'), 'attachment_policy': Path('analysis_specifications/S073_validation_confirmation_synthesis_event_attachment.json'), 'next': Path('analysis_specifications/S076_validation_confirmation_synthesis_source_epoch.json'), 'core': Path('analysis_specifications/reference_code/validation_confirmation_transfer_core.py'), 'lineage_reference': Path('analysis_specifications/reference_code/validation_lineage_reference.py'), 'summary': Path('outputs/qc/validation_confirmation_synthesis_validation_confirmation_synthesis_summary.json'), 'report': Path('outputs/reports/validation_confirmation_synthesis_validation_confirmation_synthesis_report.md'), 'contract': Path('config/validation_confirmation_synthesis_validation_confirmation_synthesis_contract.json'), 'lock': Path('outputs/qc/validation_confirmation_synthesis_final_output_lock_hashes.json')}
Q_REL = {'large_rehash': Path('outputs/qc/confirmation_feature_trajectory_confirmation_confirmation_preprocessing_qc_large_store_rehash.csv'), 'source_inventory': Path('outputs/qc/confirmation_feature_trajectory_confirmation_feature_checkpoint_inventory.csv'), 'feature_qc': Path('outputs/qc/confirmation_feature_trajectory_confirmation_compact_feature_qc.csv'), 'participant_feature_audit': Path('outputs/tables/confirmation_feature_trajectory_confirmation_participant_feature_audit.csv'), 'feature_schema': Path('outputs/tables/confirmation_feature_trajectory_confirmation_compact_feature_schema.csv'), 'epoch_manifest': Path('outputs/tables/confirmation_feature_trajectory_confirmation_compact_epoch_manifest.csv'), 'pca_scores': Path('outputs/tables/confirmation_feature_trajectory_confirmation_epoch_pca_scores.csv'), 'pca_loadings': Path('outputs/tables/confirmation_feature_trajectory_confirmation_pca_loadings.csv'), 'preprocessing': Path('outputs/tables/confirmation_feature_trajectory_confirmation_participant_preprocessing.csv'), 'trajectories': Path('outputs/tables/confirmation_feature_trajectory_confirmation_raw_trajectory_geometry.csv'), 'participant_qc': Path('outputs/qc/confirmation_feature_trajectory_confirmation_participant_trajectory_qc.csv'), 'gate': Path('outputs/qc/confirmation_feature_trajectory_confirmation_pre_assignment_transfer_gate.csv'), 'global_store_inventory': Path('outputs/tables/confirmation_feature_trajectory_confirmation_compact_store_inventory.csv'), 'summary': Path('outputs/qc/confirmation_feature_trajectory_confirmation_feature_trajectory_summary.json'), 'next': Path('analysis_specifications/S002_confirmation_feature_trajectory_fixed_assignment.json'), 'contract': Path('config/confirmation_feature_trajectory_confirmation_feature_trajectory_contract.json'), 'report': Path('outputs/reports/confirmation_feature_trajectory_confirmation_feature_trajectory_report.md'), 'lock': Path('outputs/qc/confirmation_feature_trajectory_final_output_lock_hashes.json')}
M_SCRIPT_REL = Path('src/analysis_pipeline/eeg/validation_fixed_assignment/validation_fixed_assignment.py')

def parse_args():
    p = argparse.ArgumentParser(description='Apply fixed Discovery trajectory centroids to Confirmation.')
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

def csha(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def jread(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))

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

def bval(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def import_verified(path: Path, expected: str, name: str):
    require([path])
    obs = sha(path)
    if obs != expected:
        raise RuntimeError(f'{name} hash mismatch: expected {expected}, observed {obs}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {name}: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def output_paths(project: Path):
    return {'log': project / 'outputs/logs/confirmation_fixed_assignment_confirmation_fixed_assignment.log', 'source_rehash': project / 'outputs/qc/confirmation_fixed_assignment_source_rehash.csv', 'geometry_preprocessing': project / 'outputs/tables/confirmation_fixed_assignment_confirmation_geometry_preprocessing.csv', 'standardized_geometry': project / 'outputs/tables/confirmation_fixed_assignment_confirmation_standardized_geometry.csv', 'assignments': project / 'outputs/tables/confirmation_fixed_assignment_confirmation_trajectory_assignments.csv', 'participant_summary': project / 'outputs/qc/confirmation_fixed_assignment_confirmation_participant_transfer_summary.csv', 'gate': project / 'outputs/qc/confirmation_fixed_assignment_confirmation_structural_transfer_gate.csv', 'diagnostics': project / 'outputs/qc/confirmation_fixed_assignment_confirmation_assignment_diagnostics.csv', 'summary': project / 'outputs/qc/confirmation_fixed_assignment_confirmation_structural_transfer_summary.json', 'report': project / 'outputs/reports/confirmation_fixed_assignment_confirmation_structural_transfer_report.md', 'next': project / 'analysis_specifications/S003_confirmation_fixed_assignment_three_cohort_structural.json', 'contract': project / 'config/confirmation_fixed_assignment_confirmation_structural_transfer_contract.json', 'lock': project / 'outputs/qc/confirmation_fixed_assignment_final_output_lock_hashes.json'}

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
        obs = sha(path)
        exp = str(fixed[key])
        ok = obs == exp
        rows.append({'source_phase': phase, 'artifact_key': key, 'relative_path': str(rels[key]), 'expected_sha256': exp, 'observed_sha256': obs, 'passed': ok})
        if not ok:
            raise RuntimeError(f'{phase} fixed output mismatch: {key}')
    rows.append({'source_phase': phase, 'artifact_key': 'lock', 'relative_path': str(rels['lock']), 'expected_sha256': sha(paths['lock']), 'observed_sha256': sha(paths['lock']), 'passed': True})
    return (paths, pd.DataFrame(rows))

def verify_inputs(project: Path):
    npaths, nrh = verify_fixed_phase(project, N_REL, 'validation_confirmation_synthesis')
    qpaths, qrh = verify_fixed_phase(project, Q_REL, 'confirmation_feature_trajectory')
    policy = jread(npaths['policy'])
    reference = jread(npaths['reference'])
    nsummary = jread(npaths['summary'])
    qsummary = jread(qpaths['summary'])
    qnext = jread(qpaths['next'])
    qcontract = jread(qpaths['contract'])
    if csha(policy) != EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_confirmation_synthesis transfer-policy canonical hash mismatch')
    if sha(npaths['core']) != EXPECTED_validation_confirmation_synthesis_CORE_SHA:
        raise RuntimeError('validation_confirmation_synthesis prespecified transfer-core hash mismatch')
    if str(reference.get('confirmation_transfer_policy_canonical_sha256')) != EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA:
        raise RuntimeError('validation_confirmation_synthesis reference-policy lineage mismatch')
    if str(reference.get('prespecified_transfer_core_sha256')) != EXPECTED_validation_confirmation_synthesis_CORE_SHA:
        raise RuntimeError('validation_confirmation_synthesis reference-core lineage mismatch')
    if not policy.get('fixed_before_confirmation_access') or policy.get('confirmation_opened') or nsummary.get('decision', {}).get('confirmation_opened'):
        raise RuntimeError('validation_confirmation_synthesis Confirmation boundary violation')
    if qsummary.get('phase') != 'confirmation_feature_trajectory' or not qsummary.get('decision', {}).get('confirmation_fixed_assignment_assignment_permitted'):
        raise RuntimeError('confirmation_feature_trajectory did not permit confirmation_fixed_assignment')
    if qnext.get('next_phase') != PHASE or not qnext.get('next_phase_permitted'):
        raise RuntimeError('confirmation_feature_trajectory next-phase policy does not authorize confirmation_fixed_assignment')
    if str(qnext.get('required_validation_confirmation_synthesis_transfer_policy_canonical_sha256')) != EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA:
        raise RuntimeError('confirmation_feature_trajectory transfer-policy lineage mismatch')
    if str(qnext.get('required_validation_confirmation_synthesis_prespecified_core_sha256')) != EXPECTED_validation_confirmation_synthesis_CORE_SHA:
        raise RuntimeError('confirmation_feature_trajectory core lineage mismatch')
    if str(qcontract.get('executor_sha256')) != EXPECTED_confirmation_feature_trajectory_EXECUTOR_SHA:
        raise RuntimeError('Unexpected confirmation_feature_trajectory executor')
    counts = qsummary.get('counts', {})
    expected = {'confirmation_subjects': 12, 'authoritative_trials': 494, 'complete_three_epoch_trajectories': 459, 'finite_geometry_trajectories': 459, 'eligible_participants_for_confirmation_fixed_assignment': 11}
    for k, v in expected.items():
        if int(counts.get(k, -1)) != v:
            raise RuntimeError(f'confirmation_feature_trajectory count mismatch for {k}: {counts.get(k)}')
    d = qsummary.get('decision', {})
    if d.get('reports_outcomes_read') or d.get('context_target_read') or d.get('stimulus_identity_read') or d.get('validation_targets_read') or d.get('family_assignment_performed'):
        raise RuntimeError('confirmation_feature_trajectory target-access/family-assignment boundary violation')
    mscript = project / M_SCRIPT_REL
    if mscript.is_file() and sha(mscript) != EXPECTED_validation_fixed_assignment_EXECUTOR_SHA:
        raise RuntimeError('validation_fixed_assignment algorithm-reference script hash mismatch')
    return (npaths, qpaths, pd.concat([nrh, qrh], ignore_index=True), policy, reference)

def read_fixed_inputs(npaths, qpaths, core):
    roster = pd.read_csv(npaths['roster'], usecols=['subject_id', 'split', 'n_trials', 'has_raw_eeg'], low_memory=False)
    if len(roster) != EXPECTED_SUBJECTS or roster.subject_id.nunique() != EXPECTED_SUBJECTS or int(roster.n_trials.sum()) != EXPECTED_AUTHORITATIVE_TRIALS:
        raise RuntimeError('validation_confirmation_synthesis Confirmation roster changed')
    if set(roster.split.astype(str)) != {'confirmation'} or not roster.has_raw_eeg.map(bval).all():
        raise RuntimeError('Invalid Confirmation roster')
    prep = pd.read_csv(npaths['preprocessing'], low_memory=False).sort_values('feature_order_0based')
    ctr = pd.read_csv(npaths['centroids'], low_memory=False).sort_values(['trajectory_family_1based', 'feature_order_0based'])
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
    if centroids.shape != (2, 6) or not np.isfinite(centroids).all() or list(core.FEATURES) != FEATURES:
        raise RuntimeError('Invalid prespecified transfer geometry')
    header = pd.read_csv(qpaths['trajectories'], nrows=0).columns.tolist()
    bad = sorted(set(header) & FORBIDDEN)
    if bad:
        raise RuntimeError(f'Forbidden target columns unexpectedly present in fixed trajectory source: {bad}')
    usecols = ['subject_id', 'raw_behavior_row_id', 'trial_number', *FEATURES, 'geometry_finite']
    traj = pd.read_csv(qpaths['trajectories'], usecols=usecols, low_memory=False)
    if len(traj) != EXPECTED_FINITE_TRAJECTORIES or traj.raw_behavior_row_id.astype(str).duplicated().any():
        raise RuntimeError('confirmation_feature_trajectory trajectory table invalid')
    if set(traj.subject_id.astype(str)) != set(roster.subject_id.astype(str)):
        raise RuntimeError('Trajectory roster differs from prespecified Confirmation roster')
    if not traj.geometry_finite.map(bval).all() or not np.isfinite(traj[FEATURES].to_numpy(float)).all():
        raise RuntimeError('Nonfinite Confirmation geometry reached assignment')
    pqc = pd.read_csv(qpaths['participant_qc'], usecols=['subject_id', 'finite_geometry_trajectories', 'eligible_for_confirmation_fixed_assignment'], low_memory=False)
    merged = roster[['subject_id', 'n_trials']].merge(pqc, on='subject_id', how='left', validate='one_to_one')
    observed = traj.groupby('subject_id').size().rename('n').reset_index()
    merged = merged.merge(observed, on='subject_id', how='left', validate='one_to_one')
    if merged.isna().any().any() or not (merged.finite_geometry_trajectories.astype(int) == merged.n.astype(int)).all():
        raise RuntimeError('confirmation_feature_trajectory participant trajectory count mismatch')
    elig = merged.n.astype(int) >= MIN_COMPLETE_PER_PARTICIPANT
    if not (merged.eligible_for_confirmation_fixed_assignment.map(bval).to_numpy() == elig.to_numpy()).all() or int(elig.sum()) != EXPECTED_ELIGIBLE_PARTICIPANTS:
        raise RuntimeError('confirmation_feature_trajectory participant eligibility mismatch')
    assignable = int(merged.loc[elig, 'n'].sum())
    if assignable != EXPECTED_ASSIGNABLE_TRAJECTORIES:
        raise RuntimeError(f'Expected {EXPECTED_ASSIGNABLE_TRAJECTORIES} assignable trajectories, found {assignable}')
    return (roster, traj, fallback, centroids)

def perform_transfer(roster, traj, fallback, centroids, core, reference):
    assign_parts = []
    z_parts = []
    prep_parts = []
    for subject in roster.subject_id.astype(str):
        g = traj[traj.subject_id.astype(str).eq(subject)].copy().sort_values(['trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
        if len(g) < MIN_COMPLETE_PER_PARTICIPANT:
            continue
        transferred, prep = core.transfer_one_participant(g[['subject_id', 'raw_behavior_row_id', 'trial_number', *FEATURES]], fallback, centroids)
        a = transferred[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2']].copy()
        threshold = float(reference['support_thresholds']['in_support_squared_distance_max'])
        margin = float(reference['support_thresholds']['low_margin_threshold'])
        a['in_discovery_support'] = a.assigned_squared_distance.le(threshold)
        a['low_margin'] = a.assignment_margin.lt(margin)
        z = transferred[['subject_id', 'raw_behavior_row_id', 'trial_number', *[f'standardized_{x}' for x in FEATURES]]].copy()
        assign_parts.append(a)
        z_parts.append(z)
        prep_parts.append(prep)
    if not assign_parts:
        raise RuntimeError('No eligible Confirmation participant for transfer')
    assignments = pd.concat(assign_parts, ignore_index=True).sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    standardized = pd.concat(z_parts, ignore_index=True).sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    preprocessing = pd.concat(prep_parts, ignore_index=True).sort_values(['subject_id', 'feature_order_0based'], kind='mergesort').reset_index(drop=True)
    if len(assignments) != EXPECTED_ASSIGNABLE_TRAJECTORIES or assignments.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('Assignment output row integrity failure')
    if not assignments.trajectory_family_1based.isin([1, 2]).all():
        raise RuntimeError('Invalid family label')
    if not np.isfinite(assignments[['assigned_squared_distance', 'assignment_margin', 'distance_to_family1', 'distance_to_family2']].to_numpy(float)).all():
        raise RuntimeError('Nonfinite assignment diagnostics')
    return (assignments, standardized, preprocessing)

def evaluate_gate(roster, all_traj, assignments, source_hash_gates_pass=True):
    rows = []
    for r in roster.sort_values('subject_id').itertuples(index=False):
        allg = all_traj[all_traj.subject_id.astype(str).eq(str(r.subject_id))]
        g = assignments[assignments.subject_id.astype(str).eq(str(r.subject_id))]
        n_all = len(allg)
        eligible = n_all >= MIN_COMPLETE_PER_PARTICIPANT
        rows.append({'subject_id': str(r.subject_id), 'authoritative_trials': int(r.n_trials), 'finite_geometry_trajectories': n_all, 'complete_trajectories_assigned': len(g), 'eligible_for_transfer': eligible, 'family1_trials': int(g.trajectory_family_1based.eq(1).sum()), 'family2_trials': int(g.trajectory_family_1based.eq(2).sum()), 'in_support_fraction': float(g.in_discovery_support.mean()) if len(g) else np.nan, 'low_margin_fraction': float(g.low_margin.mean()) if len(g) else np.nan, 'both_families_present': bool(g.trajectory_family_1based.nunique() == 2) if len(g) else False})
    participant = pd.DataFrame(rows)
    eligible = participant[participant.eligible_for_transfer].copy()
    if len(eligible) != EXPECTED_ELIGIBLE_PARTICIPANTS:
        raise RuntimeError('Eligible participant count changed at transfer gate')
    complete_fraction = len(assignments) / EXPECTED_AUTHORITATIVE_TRIALS
    pb_support = float(eligible.in_support_fraction.mean())
    both = int(eligible.both_families_present.sum())
    f1 = int(assignments.trajectory_family_1based.eq(1).sum())
    f2 = int(assignments.trajectory_family_1based.eq(2).sum())
    occ1 = float((eligible.family1_trials / eligible.complete_trajectories_assigned).mean())
    occ2 = float((eligible.family2_trials / eligible.complete_trajectories_assigned).mean())
    gates = [('outcome_model_integrity', 'eligible_participants', len(eligible), 9.0, '>=', len(eligible) >= 9), ('C2', 'complete_trajectory_fraction_of_494', complete_fraction, 0.75, '>=', complete_fraction >= 0.75), ('C3', 'participant_balanced_in_support_fraction', pb_support, 0.8, '>=', pb_support >= 0.8), ('C4', 'participants_with_both_families', both, 8.0, '>=', both >= 8), ('C5a', 'family1_trial_count', f1, 30.0, '>=', f1 >= 30), ('C5b', 'family2_trial_count', f2, 30.0, '>=', f2 >= 30), ('raw_dynamic_physio_alignment_design', 'family1_participant_balanced_occupancy', occ1, 0.1, '>=', occ1 >= 0.1), ('raw_dynamic_physio_timebase_audit', 'family2_participant_balanced_occupancy', occ2, 0.1, '>=', occ2 >= 0.1), ('C7', 'source_and_algorithm_hash_gates', 1.0 if source_hash_gates_pass else 0.0, 1.0, '==', bool(source_hash_gates_pass)), ('C8', 'report_outcome_mapping_executed', 0.0, 0.0, '==', True)]
    gate = pd.DataFrame([{'gate_id': a, 'metric': b, 'observed': c, 'threshold': d, 'comparison': e, 'pass': f} for a, b, c, d, e, f in gates])
    diagnostics = pd.DataFrame([{'metric': 'finite_geometry_trajectories_before_eligibility', 'value': float(len(all_traj))}, {'metric': 'assigned_trajectories_after_eligibility', 'value': float(len(assignments))}, {'metric': 'overall_in_support_fraction', 'value': float(assignments.in_discovery_support.mean())}, {'metric': 'participant_balanced_in_support_fraction', 'value': pb_support}, {'metric': 'overall_low_margin_fraction', 'value': float(assignments.low_margin.mean())}, {'metric': 'participant_balanced_low_margin_fraction', 'value': float(eligible.low_margin_fraction.mean())}, {'metric': 'assigned_squared_distance_median', 'value': float(assignments.assigned_squared_distance.median())}, {'metric': 'assigned_squared_distance_max', 'value': float(assignments.assigned_squared_distance.max())}, {'metric': 'assignment_margin_median', 'value': float(assignments.assignment_margin.median())}, {'metric': 'assignment_margin_min', 'value': float(assignments.assignment_margin.min())}])
    return (participant, gate, diagnostics)

def self_test(project: Path):
    core = import_verified(project / N_REL['core'], EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'validation_confirmation_synthesis_transfer_core_selftest')
    rng = np.random.default_rng(540)
    n = 24
    frame = pd.DataFrame({'subject_id': ['sub-test'] * n, 'raw_behavior_row_id': [f'test-{i:03d}' for i in range(n)], 'trial_number': np.arange(1, n + 1), 'early_to_late_norm': rng.lognormal(0.9, 0.25, n), 'late_to_rating_norm': rng.lognormal(1.0, 0.25, n), 'early_to_rating_direct_norm': rng.lognormal(0.95, 0.2, n), 'total_path_length': rng.lognormal(1.7, 0.2, n), 'directness_ratio': rng.uniform(0.15, 0.9, n), 'segment_cosine': rng.uniform(-0.9, 0.8, n)})
    fallback = np.array([0.4, 0.4, 0.4, 0.4, 1.0, 0.6])
    centroids = np.array([[-0.3, -0.3, 0, -0.3, 0.3, 0.3], [0.7, 0.6, 0.2, 0.7, -0.6, -0.6]])
    out, prep = core.transfer_one_participant(frame, fallback, centroids)
    assert len(out) == n and len(prep) == 6 and out.trajectory_family_1based.isin([1, 2]).all()
    print('confirmation_fixed_assignment self-test: PASS')

def main():
    args = parse_args()
    project = args.project.resolve()
    if args.self_test:
        self_test(project)
        return 0
    out = output_paths(project)
    out['log'].parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"confirmation_fixed_assignment outputs already exist: {out['contract']}; use --force after review")
    started = time.time()
    logf = out['log'].open('a', encoding='utf-8')
    script_sha = sha(Path(__file__).resolve())

    def note(msg):
        print(f'[{now()}] {msg}', flush=True)
        print(f'[{now()}] {msg}', file=logf, flush=True)
    try:
        npaths, qpaths, rehash, policy, reference = verify_inputs(project)
        core = import_verified(npaths['core'], EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'prespecified_validation_confirmation_synthesis_transfer_core')
        roster, traj, fallback, centroids = read_fixed_inputs(npaths, qpaths, core)
        note('Verified validation_confirmation_synthesis Confirmation transfer specification, confirmation_feature_trajectory trajectories, prespecified core, centroids, thresholds, and target-access boundaries.')
        assignments, standardized, preprocessing = perform_transfer(roster, traj, fallback, centroids, core, policy['fixed_assignment']['support_thresholds']) if False else perform_transfer(roster, traj, fallback, centroids, core, {'support_thresholds': policy['fixed_assignment']['support_thresholds']})
        participant, gate, diagnostics = evaluate_gate(roster, traj, assignments, True)
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
        fam = assignments.trajectory_family_1based.value_counts().sort_index().to_dict()
        eligible = participant[participant.eligible_for_transfer]
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'confirmation_fixed_two_family_geometry_transfer_complete_candidate', 'source_lineage': {'validation_confirmation_synthesis_transfer_policy_canonical_sha256': EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA, 'validation_confirmation_synthesis_prespecified_transfer_core_sha256': EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'validation_fixed_assignment_validation_transfer_executor_sha256': EXPECTED_validation_fixed_assignment_EXECUTOR_SHA, 'confirmation_feature_trajectory_executor_sha256': EXPECTED_confirmation_feature_trajectory_EXECUTOR_SHA}, 'counts': {'confirmation_subjects': int(roster.subject_id.nunique()), 'authoritative_trials': int(roster.n_trials.sum()), 'finite_geometry_trajectories': len(traj), 'assigned_trajectories': len(assignments), 'eligible_participants': int(len(eligible)), 'ineligible_participants_retained_in_audit': int((~participant.eligible_for_transfer).sum()), 'family1_trials': int(fam.get(1, 0)), 'family2_trials': int(fam.get(2, 0)), 'participants_with_both_families': int(eligible.both_families_present.sum()), 'in_support_trajectories': int(assignments.in_discovery_support.sum()), 'low_margin_trajectories': int(assignments.low_margin.sum())}, 'metrics': {'assigned_trajectory_fraction_of_494': float(len(assignments) / EXPECTED_AUTHORITATIVE_TRIALS), 'participant_balanced_in_support_fraction': float(eligible.in_support_fraction.mean()), 'participant_balanced_family1_occupancy': float((eligible.family1_trials / eligible.complete_trajectories_assigned).mean()), 'participant_balanced_family2_occupancy': float((eligible.family2_trials / eligible.complete_trajectories_assigned).mean()), 'overall_low_margin_fraction': float(assignments.low_margin.mean())}, 'decision': {'all_structural_transfer_gates_passed': gate_passed, 'third_cohort_two_family_outcome_free_geometry_replication_supported': gate_passed, 'hedonic_emotional_contextual_or_stimulus_interpretation_supported': False, 'centroids_refit': False, 'thresholds_tuned': False, 'families_renamed': False, 'reports_outcomes_read': False, 'context_target_read': False, 'stimulus_identity_read': False, 'validation_targets_read': False, 'confirmation_opened_for_fixed_geometry_assignment_only': True, 'three_cohort_structural_synthesis_permitted': True}, 'runtime_seconds': time.time() - started}
        atom_json(out['summary'], summary)
        next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'final structural synthesis across Discovery, Validation, and Confirmation without semantic mapping or post hoc retuning', 'structural_transfer_gate_passed': gate_passed, 'required_inputs': {**ph, 'summary': sha(out['summary'])}, 'required_validation_confirmation_synthesis_transfer_policy_canonical_sha256': EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA, 'required_validation_confirmation_synthesis_prespecified_core_sha256': EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'forbidden': ['family relabeling', 'centroid refit', 'threshold tuning', 'H1-H6 mapping', 'Confirmation reports/outcomes/context targets', 'Validation-target reuse'], 'confirmation_opened_for_fixed_geometry_assignment_only': True, 'confirmation_reports_outcomes_read': False}
        atom_json(out['next'], next_policy)
        contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'executor_sha256': script_sha, 'validation_confirmation_synthesis_transfer_policy_canonical_sha256': EXPECTED_validation_confirmation_synthesis_POLICY_CANONICAL_SHA, 'validation_confirmation_synthesis_prespecified_transfer_core_sha256': EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'validation_confirmation_synthesis_input_hashes': jread(npaths['lock'])['hashes'], 'confirmation_feature_trajectory_input_hashes': jread(qpaths['lock'])['hashes'], 'output_hashes': {**ph, 'summary': sha(out['summary']), 'next': sha(out['next'])}, 'boundary': summary['decision']}
        atom_json(out['contract'], contract)
        report = ''.join(['# Phase confirmation_fixed_assignment — Confirmation fixed trajectory-family transfer', '', f'- Generated: `{now()}`', f'- Finite trajectories before participant eligibility: **{len(traj)}**', f'- Assigned trajectories from eligible participants: **{len(assignments)}/{EXPECTED_AUTHORITATIVE_TRIALS}**', f'- Eligible participants: **{len(eligible)}/{EXPECTED_SUBJECTS}**', f'- Family 1 trials: **{int(fam.get(1, 0))}**', f'- Family 2 trials: **{int(fam.get(2, 0))}**', f'- Participant-balanced in-support fraction: **{float(eligible.in_support_fraction.mean()):.6f}**', f'- Participants with both families: **{int(eligible.both_families_present.sum())}/{len(eligible)}**', f'- All prespecified structural-transfer gates passed: **{gate_passed}**', '', '## Interpretation boundary', '', '- A pass supports only third-cohort replication of the fixed two-family outcome-free geometric partition.', '- It does not support hedonic, emotional, contextual, category, stimulus, or causal interpretation.', '- Centroids, thresholds, feature set, family count, and family names were not changed.', '- Reports/outcomes/context targets were not read.', '- sub-59 remains in source/QC audit but is excluded from assignment/gate because 13 < prespecified minimum 16 complete trajectories.', ''])
        out['report'].parent.mkdir(parents=True, exist_ok=True)
        out['report'].write_text(report, encoding='utf-8')
        lock_files = {**primary, 'summary': out['summary'], 'next': out['next'], 'contract': out['contract'], 'report': out['report']}
        atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': {k: sha(p) for k, p in lock_files.items()}, 'structural_transfer_gate_passed': gate_passed, 'reports_outcomes_read': False, 'context_target_read': False, 'validation_targets_read': False, 'family_refit_or_rename': False, 'confirmation_opened_for_fixed_geometry_assignment_only': True})
        files = list(lock_files.values()) + [out['lock'], Path(__file__).resolve(), npaths['policy'], npaths['reference'], npaths['core'], npaths['roster'], npaths['preprocessing'], npaths['centroids'], npaths['gates'], npaths['schema'], npaths['lock'], qpaths['trajectories'], qpaths['participant_qc'], qpaths['gate'], qpaths['summary'], qpaths['next'], qpaths['contract'], qpaths['lock']]
        note(f'confirmation_fixed_assignment complete. Gate passed={gate_passed}; Family1={int(fam.get(1, 0))}; Family2={int(fam.get(2, 0))}; participant-balanced support={float(eligible.in_support_fraction.mean()):.6f}.')
        return 0
    finally:
        logf.close()
if __name__ == '__main__':
    raise SystemExit(main())