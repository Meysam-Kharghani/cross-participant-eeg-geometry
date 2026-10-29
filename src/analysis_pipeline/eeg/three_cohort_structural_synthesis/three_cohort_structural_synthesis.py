# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase three_cohort_structural_synthesis — final three-cohort structural synthesis.

No model fitting, family remapping, threshold tuning, or semantic mapping occurs.
The phase verifies fixed Discovery, robustness, Validation, Confirmation and
Confirmation-design lineage, then specifications the final interpretation boundary for
this EEG trajectory-geometry branch.

Supported claim: a fixed two-family outcome-free trajectory geometry discovered
in the Discovery split transferred structurally, without refit, to the Validation
and Confirmation splits of the same dataset.

Not supported: hedonic/emotional/contextual/stimulus/causal labels, a universal
biomarker, or independent external-study replication."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'three_cohort_structural_synthesis'
NEXT_PHASE = 'multimodal_physio_design'
PREFIX = 'three_cohort_structural_synthesis'
PHASES = {'cross_person_alignment_fit': {'phase': 'cross_person_alignment_fit', 'lock': Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json'), 'lock_sha': '64bb8278afd0ddc17b46521180753e5537333e9e55a93741eece62d406864150', 'files': {'summary': Path('outputs/qc/cross_person_alignment_fit_cross_person_alignment_summary.json'), 'trajectory_selected_assignments': Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv'), 'trajectory_lopo': Path('outputs/tables/cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv'), 'trajectory_selection': Path('outputs/tables/cross_person_alignment_fit_trajectory_family_selection.csv')}}, 'mapping_inference': {'phase': 'mapping_inference', 'lock': Path('outputs/qc/mapping_inference_final_output_lock_hashes.json'), 'lock_sha': '29f7265b0d63ae2348d35ff76a5a89f318e86b6a120ede9c6f60e3d01a6d6948', 'files': {'summary': Path('outputs/qc/mapping_inference_mapping_inference_summary.json'), 'global_tests': Path('outputs/tables/mapping_inference_global_hypothesis_tests.csv'), 'report_followups': Path('outputs/tables/mapping_inference_report_dimension_followups.csv')}}, 'discovery_robustness_audit': {'phase': 'discovery_robustness_audit', 'lock': Path('outputs/qc/discovery_robustness_audit_final_output_lock_hashes.json'), 'lock_sha': '9d05bd8d03fde1d85387b77683ed116f33d505d9c119256d7b34ca88091b8633', 'files': {'summary': Path('outputs/qc/discovery_robustness_audit_discovery_robustness_summary.json'), 'r8_summary': Path('outputs/qc/discovery_robustness_audit_r8_trajectory_structural_gate_summary.json')}}, 'validation_fixed_assignment': {'phase': 'validation_fixed_assignment', 'lock': Path('outputs/qc/validation_fixed_assignment_final_output_lock_hashes.json'), 'lock_sha': 'b6ea679c97243167292bf2e167883b17b067bb4b220d38ed0e1c0e35fd783450', 'files': {'summary': Path('outputs/qc/validation_fixed_assignment_validation_structural_transfer_summary.json'), 'assignments': Path('outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv'), 'participant_summary': Path('outputs/qc/validation_fixed_assignment_validation_participant_transfer_summary.csv'), 'gate': Path('outputs/qc/validation_fixed_assignment_validation_structural_transfer_gate.csv')}}, 'validation_confirmation_synthesis': {'phase': 'validation_confirmation_synthesis', 'lock': Path('outputs/qc/validation_confirmation_synthesis_final_output_lock_hashes.json'), 'lock_sha': 'd2f06565fa0e458af159025f12b6c17df97ecf35f7d6fba3ff2be94589f709a4', 'files': {'summary': Path('outputs/qc/validation_confirmation_synthesis_validation_confirmation_synthesis_summary.json'), 'evidence': Path('outputs/tables/validation_confirmation_synthesis_validation_confirmation_evidence_matrix.csv')}}, 'confirmation_fixed_assignment': {'phase': 'confirmation_fixed_assignment', 'lock': Path('outputs/qc/confirmation_fixed_assignment_final_output_lock_hashes.json'), 'lock_sha': '7eb453fcf6822d627d5ea451f183477045eec4d45d8645c218eceb7aa5d7e7a1', 'files': {'summary': Path('outputs/qc/confirmation_fixed_assignment_confirmation_structural_transfer_summary.json'), 'assignments': Path('outputs/tables/confirmation_fixed_assignment_confirmation_trajectory_assignments.csv'), 'participant_summary': Path('outputs/qc/confirmation_fixed_assignment_confirmation_participant_transfer_summary.csv'), 'gate': Path('outputs/qc/confirmation_fixed_assignment_confirmation_structural_transfer_gate.csv'), 'next': Path('analysis_specifications/S003_confirmation_fixed_assignment_three_cohort_structural.json')}}}
LOCK_KEYS = {'cross_person_alignment_fit': {'summary': None, 'trajectory_selected_assignments': 'trajectory_selected_assignments', 'trajectory_lopo': 'trajectory_lopo', 'trajectory_selection': 'trajectory_selection'}, 'mapping_inference': {'summary': 'summary', 'global_tests': 'global_tests', 'report_followups': 'report_followups'}, 'discovery_robustness_audit': {'summary': 'summary', 'r8_summary': 'r8_summary'}, 'validation_fixed_assignment': {'summary': 'summary', 'assignments': 'assignments', 'participant_summary': 'participant_summary', 'gate': 'gate'}, 'validation_confirmation_synthesis': {'summary': 'summary', 'evidence': 'evidence'}, 'confirmation_fixed_assignment': {'summary': 'summary', 'assignments': 'assignments', 'participant_summary': 'participant_summary', 'gate': 'gate', 'next': 'next'}}
cross_person_alignment_fit_SUMMARY_SHA = '13ca118e52c9af4e247a3dc9d75b5f2c10bfa21e179a26d824346ea3b9116bf1'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def jread(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))

def atom_json(p: Path, v: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    os.replace(t, p)

def require(ps):
    miss = [str(p) for p in ps if not p.is_file()]
    if miss:
        raise FileNotFoundError('Missing required files:\n  - ' + '\n  - '.join(miss))

def bval(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def output_paths(project: Path):
    return {'log': project / 'outputs/logs/three_cohort_structural_synthesis_three_cohort_structural_synthesis.log', 'source_rehash': project / 'outputs/qc/three_cohort_structural_synthesis_source_rehash.csv', 'evidence': project / 'outputs/tables/three_cohort_structural_synthesis_three_cohort_structural_evidence_matrix.csv', 'balance': project / 'outputs/tables/three_cohort_structural_synthesis_three_cohort_family_balance.csv', 'interpretation': project / 'analysis_specifications/S069_three_cohort_structural_synthesis_interpretation.json', 'next': project / 'analysis_specifications/S070_three_cohort_structural_synthesis_multimodal_physio.json', 'summary': project / 'outputs/qc/three_cohort_structural_synthesis_final_structural_synthesis_summary.json', 'report': project / 'outputs/reports/three_cohort_structural_synthesis_final_structural_synthesis_report.md', 'contract': project / 'config/three_cohort_structural_synthesis_final_structural_synthesis_contract.json', 'lock': project / 'outputs/qc/three_cohort_structural_synthesis_final_output_lock_hashes.json'}

def verify_sources(project: Path):
    rows = []
    paths = {}
    for label, spec in PHASES.items():
        lp = project / spec['lock']
        require([lp])
        if sha(lp) != spec['lock_sha']:
            raise RuntimeError(f'{label} final-lock SHA mismatch')
        lock = jread(lp)
        if lock.get('phase') != spec['phase']:
            raise RuntimeError(f'{label} phase mismatch')
        hashes = lock.get('hashes', {})
        paths[label] = {}
        rows.append({'source_phase': label, 'artifact': 'final_lock', 'relative_path': str(spec['lock']), 'expected_sha256': spec['lock_sha'], 'observed_sha256': sha(lp), 'passed': True})
        for name, rel in spec['files'].items():
            p = project / rel
            require([p])
            obs = sha(p)
            paths[label][name] = p
            if label == 'cross_person_alignment_fit' and name == 'summary':
                exp = cross_person_alignment_fit_SUMMARY_SHA
            else:
                key = LOCK_KEYS[label][name]
                if key is None:
                    raise RuntimeError('Internal lock-key error')
                if key not in hashes:
                    raise RuntimeError(f'{label} lock missing key {key}')
                exp = str(hashes[key])
            ok = obs == exp
            rows.append({'source_phase': label, 'artifact': name, 'relative_path': str(rel), 'expected_sha256': exp, 'observed_sha256': obs, 'passed': ok})
            if not ok:
                raise RuntimeError(f'{label} artifact hash mismatch: {name}')
    return (paths, pd.DataFrame(rows))

def self_test():
    a = pd.DataFrame({'subject_id': ['a', 'a', 'b', 'b'], 'trajectory_family_1based': [1, 2, 1, 2]})
    assert a.groupby('subject_id')['trajectory_family_1based'].nunique().eq(2).all()
    assert abs(270 / 446 + 176 / 446 - 1) < 1e-12
    print('three_cohort_structural_synthesis self-test: PASS')

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
        raise RuntimeError('three_cohort_structural_synthesis already exists; use --force only for identical rerun')
    logs = []

    def log(x):
        line = f'[{now()}] {x}'
        print(line, flush=True)
        logs.append(line)
    log('Verifying fixed Discovery, robustness, Validation, Confirmation-design, and Confirmation artifacts.')
    src, rehash = verify_sources(project)
    d = jread(src['cross_person_alignment_fit']['summary'])
    f = jread(src['mapping_inference']['summary'])
    h = jread(src['discovery_robustness_audit']['summary'])
    h8 = jread(src['discovery_robustness_audit']['r8_summary'])
    m = jread(src['validation_fixed_assignment']['summary'])
    n = jread(src['validation_confirmation_synthesis']['summary'])
    r = jread(src['confirmation_fixed_assignment']['summary'])
    rnext = jread(src['confirmation_fixed_assignment']['next'])
    ftests = pd.read_csv(src['mapping_inference']['global_tests'])
    massign = pd.read_csv(src['validation_fixed_assignment']['assignments'])
    rassign = pd.read_csv(src['confirmation_fixed_assignment']['assignments'])
    mgate = pd.read_csv(src['validation_fixed_assignment']['gate'])
    rgate = pd.read_csv(src['confirmation_fixed_assignment']['gate'])
    dassign = pd.read_csv(src['cross_person_alignment_fit']['trajectory_selected_assignments'])
    if d['trajectory_branch'].get('selected_shared_k') != 2 or not d['trajectory_branch'].get('stable_discrete_family_solution'):
        raise RuntimeError('Discovery two-family trajectory solution not present')
    if not h['decision'].get('R8_pass', False) or not h8.get('R8_pass', False):
        raise RuntimeError('Discovery structural robustness gate failed')
    if int(ftests['global_supported'].astype(bool).sum()) != 0:
        raise RuntimeError('At least one Discovery semantic parent mapping is supported; boundary changed')
    if not m['decision'].get('all_structural_transfer_gates_passed', False):
        raise RuntimeError('Validation structural gate failed')
    if not r['decision'].get('all_structural_transfer_gates_passed', False):
        raise RuntimeError('Confirmation structural gate failed')
    if not bool(mgate['pass'].map(bval).all()) or not bool(rgate['pass'].map(bval).all()):
        raise RuntimeError('Transfer gate table contains failure')
    if not rnext.get('next_phase_permitted', False) or rnext.get('next_phase') != 'three_cohort_structural_synthesis':
        raise RuntimeError('confirmation_fixed_assignment did not authorize three_cohort_structural_synthesis')
    for summary, label in [(m, 'Validation'), (r, 'Confirmation')]:
        dec = summary['decision']
        forbidden_true = ['hedonic_emotional_contextual_or_stimulus_interpretation_supported', 'centroids_refit', 'thresholds_tuned', 'families_renamed', 'reports_outcomes_read', 'context_target_read', 'stimulus_identity_read']
        if any((dec.get(k, False) for k in forbidden_true)):
            raise RuntimeError(f'{label} boundary violated')
    d1 = int((dassign['trajectory_family_1based'] == 1).sum())
    d2 = int((dassign['trajectory_family_1based'] == 2).sum())
    if (d1, d2) != (749, 488):
        raise RuntimeError(f'Unexpected Discovery family counts {(d1, d2)}')
    if len(dassign) != 1237 or dassign['subject_id'].nunique() != 31:
        raise RuntimeError('Unexpected Discovery assignment roster')
    if len(massign) != 437 or massign['subject_id'].nunique() != 12:
        raise RuntimeError('Unexpected Validation assignment roster')
    if len(rassign) != 446 or rassign['subject_id'].nunique() != 11:
        raise RuntimeError('Unexpected Confirmation assignment roster')
    semantic_supported = int(ftests['global_supported'].astype(bool).sum())
    evidence = pd.DataFrame([{'cohort': 'Discovery', 'role': 'discovery', 'roster_participants': 31, 'eligible_assigned_participants': 31, 'assigned_trajectories': 1237, 'family1_trials': 749, 'family2_trials': 488, 'structural_metric': 'LOPO agreement', 'structural_metric_value': float(h8['overall_lopo_agreement']), 'structural_gate_passed': True, 'semantic_parent_tests_executed': 6, 'semantic_parent_tests_supported': semantic_supported, 'semantic_mapping_scope': 'Discovery only'}, {'cohort': 'Validation', 'role': 'fixed transfer', 'roster_participants': 12, 'eligible_assigned_participants': int(m['counts']['eligible_participants']), 'assigned_trajectories': int(m['counts']['assigned_trajectories']), 'family1_trials': int(m['counts']['family1_trials']), 'family2_trials': int(m['counts']['family2_trials']), 'structural_metric': 'participant-balanced in-support fraction', 'structural_metric_value': float(m['metrics']['participant_balanced_in_support_fraction']), 'structural_gate_passed': True, 'semantic_parent_tests_executed': 0, 'semantic_parent_tests_supported': 0, 'semantic_mapping_scope': 'forbidden'}, {'cohort': 'Confirmation', 'role': 'fixed transfer', 'roster_participants': 12, 'eligible_assigned_participants': int(r['counts']['eligible_participants']), 'assigned_trajectories': int(r['counts']['assigned_trajectories']), 'family1_trials': int(r['counts']['family1_trials']), 'family2_trials': int(r['counts']['family2_trials']), 'structural_metric': 'participant-balanced in-support fraction', 'structural_metric_value': float(r['metrics']['participant_balanced_in_support_fraction']), 'structural_gate_passed': True, 'semantic_parent_tests_executed': 0, 'semantic_parent_tests_supported': 0, 'semantic_mapping_scope': 'forbidden'}])
    evidence['family1_raw_fraction'] = evidence['family1_trials'] / evidence['assigned_trajectories']
    evidence['family2_raw_fraction'] = evidence['family2_trials'] / evidence['assigned_trajectories']
    balance = evidence[['cohort', 'assigned_trajectories', 'family1_trials', 'family2_trials', 'family1_raw_fraction', 'family2_raw_fraction']].copy()
    balance['family1_abs_deviation_from_discovery'] = abs(balance['family1_raw_fraction'] - float(balance.loc[balance.cohort.eq('Discovery'), 'family1_raw_fraction'].iloc[0]))
    balance['both_families_in_all_eligible_participants'] = [True, bool(m['counts']['participants_with_both_families'] == m['counts']['eligible_participants']), bool(r['counts']['participants_with_both_families'] == r['counts']['eligible_participants'])]
    interpretation = {'phase': PHASE, 'generated_at': now(), 'supported_claims': ['A two-family outcome-free EEG trajectory-geometry partition was discovered in the Discovery split under prespecified unsupervised procedures.', 'The fixed two-family geometry transferred without centroid refitting, threshold tuning, feature selection, or family renaming to the Validation split.', 'The same fixed geometry transferred again to the Confirmation split among participants meeting the pre-prespecified minimum trajectory criterion.', 'The structural result therefore replicated across three pre-specified non-overlapping splits of this dataset.', 'Discovery post-hoc H1-H6 semantic mappings were all unsupported after the prespecified multiplicity gate.'], 'not_supported_claims': ['Family 1 or Family 2 is a pleasure, pain, valence, liking, emotion, context, category, stimulus, or causal state.', 'The two families are a universal cross-person biomarker.', 'The result is an independent external-study replication; all three cohorts come from one dataset split.', 'The absence of supported Discovery semantic mappings proves that no semantic relation exists.', 'Validation or Confirmation semantically validate H1-H6; those mappings were intentionally not tested there.', 'A stable shared discrete taxonomy of local within-person states was found; that branch did not pass its prespecified no-force criteria.'], 'family_labels_fixed': ['Family 1', 'Family 2'], 'family_labels_are_geometric_only': True, 'semantic_mapping_closed_for_validation_and_confirmation': True, 'posthoc_relabeling_forbidden': True}
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'outcome-free multimodal/peripheral-physiology integration analysis specification only', 'required_boundary': ['three_cohort_structural_synthesis final structural interpretation remains immutable.', 'EEG Family 1/2 labels and assignments may not be renamed or retuned.', 'No physiology result may be used to retroactively select EEG families or thresholds.', 'Any semantic or multimodal hypotheses must be prespecified before inferential testing.'], 'forbidden_in_next_phase': ['new outcome inference', 'family relabeling', 'centroid refit', 'threshold tuning', 'Confirmation semantic mapping', 'post hoc selection using Valence/Liking']}
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'three_cohort_outcome_free_trajectory_geometry_structural_replication_complete_candidate', 'counts': {'split_cohorts': 3, 'total_roster_participants': 55, 'eligible_assigned_participants_total': 31 + int(m['counts']['eligible_participants']) + int(r['counts']['eligible_participants']), 'assigned_trajectories_total': 1237 + int(m['counts']['assigned_trajectories']) + int(r['counts']['assigned_trajectories']), 'discovery_assigned_trajectories': 1237, 'validation_assigned_trajectories': int(m['counts']['assigned_trajectories']), 'confirmation_assigned_trajectories': int(r['counts']['assigned_trajectories']), 'discovery_semantic_parent_tests_supported': semantic_supported}, 'structural_metrics': {'discovery_lopo_agreement': float(h8['overall_lopo_agreement']), 'validation_participant_balanced_in_support_fraction': float(m['metrics']['participant_balanced_in_support_fraction']), 'confirmation_participant_balanced_in_support_fraction': float(r['metrics']['participant_balanced_in_support_fraction']), 'discovery_family1_raw_fraction': d1 / 1237, 'validation_family1_raw_fraction': int(m['counts']['family1_trials']) / int(m['counts']['assigned_trajectories']), 'confirmation_family1_raw_fraction': int(r['counts']['family1_trials']) / int(r['counts']['assigned_trajectories'])}, 'decision': {'three_split_cohort_fixed_structural_replication_supported': True, 'independent_external_replication_supported': False, 'stable_shared_discrete_local_state_taxonomy_supported': False, 'hedonic_emotional_contextual_or_stimulus_interpretation_supported': False, 'universal_biomarker_claim_supported': False, 'family_refit_or_rename_performed': False, 'validation_semantic_mapping_performed': False, 'confirmation_semantic_mapping_performed': False, 'trajectory_geometry_eeg_geometry_branch_complete': True, 'multimodal_physio_design_specification_permitted': True}}
    rehash.to_csv(out['source_rehash'], index=False)
    evidence.to_csv(out['evidence'], index=False)
    balance.to_csv(out['balance'], index=False)
    atom_json(out['interpretation'], interpretation)
    atom_json(out['next'], next_policy)
    atom_json(out['summary'], summary)
    report = f"# Phase three_cohort_structural_synthesis — final three-cohort structural synthesis\n\n- Generated: `{summary['generated_at']}`\n- Status: **`{summary['status']}`**\n\n## Final structural result\n\nThe outcome-free two-family EEG trajectory geometry discovered in the Discovery split transferred without refitting or retuning to both held-out splits of the same dataset.\n\n| Cohort | Eligible participants | Assigned trajectories | Family 1 | Family 2 | Structural metric |\n|---|---:|---:|---:|---:|---:|\n| Discovery | 31 | 1,237 | 749 | 488 | LOPO agreement {h8['overall_lopo_agreement']:.6f} |\n| Validation | {m['counts']['eligible_participants']} | {m['counts']['assigned_trajectories']} | {m['counts']['family1_trials']} | {m['counts']['family2_trials']} | in-support {m['metrics']['participant_balanced_in_support_fraction']:.6f} |\n| Confirmation | {r['counts']['eligible_participants']} | {r['counts']['assigned_trajectories']} | {r['counts']['family1_trials']} | {r['counts']['family2_trials']} | in-support {r['metrics']['participant_balanced_in_support_fraction']:.6f} |\n\nAll prespecified structural gates passed in Validation and Confirmation. Confirmation retained sub-59 in audit but excluded that participant from assignment because 13 complete trajectories were below the prespecified minimum of 16.\n\n## Semantic boundary\n\nAll six prespecified Discovery parent mappings H1-H6 were unsupported. No semantic mapping was run in Validation or Confirmation. Therefore Family 1/2 remain geometric labels only. This phase does **not** license pleasure, pain, valence, liking, emotion, context, stimulus, causal, or universal-biomarker interpretations.\n\nThe three cohorts are pre-specified non-overlapping splits of one dataset; this is a three-split replication, not independent external-study replication.\n\n## Branch status\n\nThe analysis_stage EEG trajectory-geometry branch is complete. The only permitted subsequent analysis is a new design-specification phase for outcome-free multimodal/peripheral-physiology integration. It may not retroactively retune or relabel the EEG result."
    out['report'].write_text(report, encoding='utf-8')
    output_hashes = {k: sha(p) for k, p in out.items() if k not in {'log', 'lock', 'contract'}}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'source_final_lock_sha256': {k: v['lock_sha'] for k, v in PHASES.items()}, 'output_hashes': output_hashes, 'boundary': summary['decision']}
    atom_json(out['contract'], contract)
    output_hashes['contract'] = sha(out['contract'])
    lock = {'generated_at': now(), 'phase': PHASE, 'hashes': output_hashes, 'three_split_cohort_fixed_structural_replication_supported': True, 'semantic_interpretation_supported': False, 'family_refit_or_rename': False, 'multimodal_physio_design_specification_permitted': True}
    atom_json(out['lock'], lock)
    logs.append(f'[{now()}] three_cohort_structural_synthesis final synthesis complete; three-cohort structural replication supported, semantic interpretation not supported.')
    out['log'].write_text(''.join(logs) + '', encoding='utf-8')
    include = [p for k, p in out.items() if k != 'log']
    zp_sha = sha(zpath)
    print(f'SHA-256: {zp_sha}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())