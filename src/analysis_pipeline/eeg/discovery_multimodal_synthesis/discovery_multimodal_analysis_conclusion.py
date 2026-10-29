# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase discovery_multimodal_synthesis — Discovery multimodal synthesis and analysis conclusion.

No new model fitting, permutation, bootstrap, physiology feature selection, held-out
physiology access, family remapping, or semantic relabeling occurs in this phase.

The phase verifies the fixed three-cohort EEG structural result and the complete
Discovery-only EEG×physiology inference, then specifications the multimodal interpretation
boundary. Because none of M1-M3 passed the pre-prespecified Holm + robustness gate,
Validation and Confirmation physiology remain unopened and this multimodal branch
is closed without held-out physiology testing."""
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'discovery_multimodal_synthesis'
NEXT_PHASE = 'physiology_followup_design'
PREFIX = 'discovery_multimodal_synthesis'
SOURCES = {'three_cohort_structural_synthesis': {'phase': 'three_cohort_structural_synthesis', 'lock': Path('outputs/qc/three_cohort_structural_synthesis_final_output_lock_hashes.json'), 'lock_sha': '3c0b3b2f0998672a3f6521f7b32489d301b3f06561a7e1855b9fbf8f27b5e216', 'files': {'summary': Path('outputs/qc/three_cohort_structural_synthesis_final_structural_synthesis_summary.json'), 'interpretation': Path('analysis_specifications/S069_three_cohort_structural_synthesis_interpretation.json'), 'evidence': Path('outputs/tables/three_cohort_structural_synthesis_three_cohort_structural_evidence_matrix.csv')}, 'keys': {'summary': 'summary', 'interpretation': 'interpretation', 'evidence': 'evidence'}}, 'multimodal_physio_design': {'phase': 'multimodal_physio_design', 'lock': Path('outputs/qc/multimodal_physio_design_final_output_lock_hashes.json'), 'lock_sha': '2e2792d21f49620be426b1a61c70dcd4d2655b03c952c86dd015a58183a25ac2', 'files': {'summary': Path('outputs/qc/multimodal_physio_design_multimodal_physio_design_summary.json'), 'hypotheses': Path('outputs/tables/multimodal_physio_design_prespecified_multimodal_hypotheses.csv'), 'gates': Path('outputs/tables/multimodal_physio_design_prespecified_multimodal_gate_manifest.csv'), 'policy': Path('analysis_specifications/S044_multimodal_physio.json')}, 'keys': {'summary': 'summary', 'hypotheses': 'hyp', 'gates': 'gates', 'policy': 'policy'}}, 'discovery_multimodal_attachment_qc': {'phase': 'discovery_multimodal_attachment_qc', 'lock': Path('outputs/qc/discovery_multimodal_attachment_qc_final_output_lock_hashes.json'), 'lock_sha': '2cc5d2964e130ce6e3afee028ff762399b54b211a86c3a530d5d593b445803f6', 'files': {'summary': Path('outputs/qc/discovery_multimodal_attachment_qc_attachment_summary.json'), 'roster': Path('outputs/tables/discovery_multimodal_attachment_qc_prespecified_eligible_roster.csv'), 'bridge_addendum': Path('config/discovery_multimodal_attachment_qc_bridge_correction_addendum.json'), 'next': Path('analysis_specifications/S018_discovery_multimodal_attachment_qc.json')}, 'keys': {'summary': 'summary', 'roster': 'roster', 'bridge_addendum': 'bridge_addendum', 'next': 'next'}}, 'discovery_multimodal_inference': {'phase': 'discovery_multimodal_inference', 'lock': Path('outputs/qc/discovery_multimodal_inference_final_output_lock_hashes.json'), 'lock_sha': '6fb90c3bcce0422cdba055a67f5fe7c091bf0bd07bbfabc52457ecf82ac244e2', 'files': {'summary': Path('outputs/qc/discovery_multimodal_inference_inference_summary.json'), 'global': Path('outputs/tables/discovery_multimodal_inference_global_tests.csv'), 'bootstrap': Path('outputs/tables/discovery_multimodal_inference_bootstrap_summary.csv'), 'robustness': Path('outputs/tables/discovery_multimodal_inference_robustness_summary.csv'), 'complete_case': Path('outputs/tables/discovery_multimodal_inference_complete_case_sensitivity.csv'), 'next': Path('analysis_specifications/S019_discovery_multimodal_inference_synthesis.json'), 'resolution': Path('config/discovery_multimodal_inference_implementation_resolution.json')}, 'keys': {'summary': 'summary', 'global': 'global', 'bootstrap': 'boot', 'robustness': 'robust', 'complete_case': 'cc', 'next': 'next', 'resolution': 'resolution'}}}

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
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '', encoding='utf-8')
    os.replace(t, p)

def require(paths):
    miss = [str(p) for p in paths if not p.is_file()]
    if miss:
        raise FileNotFoundError('Missing required files:\n  - ' + '\n  - '.join(miss))

def bval(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def output_paths(project: Path):
    return {'rehash': project / 'outputs/qc/discovery_multimodal_synthesis_source_rehash_audit.csv', 'evidence': project / 'outputs/tables/discovery_multimodal_synthesis_multimodal_evidence_matrix.csv', 'robustness': project / 'outputs/tables/discovery_multimodal_synthesis_multimodal_robustness_closure.csv', 'interpretation': project / 'analysis_specifications/S021_discovery_multimodal_synthesis_interpretation.json', 'heldout': project / 'analysis_specifications/S020_discovery_multimodal_synthesis_held_out_physio.json', 'next': project / 'analysis_specifications/S022_discovery_multimodal_synthesis_integrated.json', 'summary': project / 'outputs/qc/discovery_multimodal_synthesis_multimodal_analysis_conclusion_summary.json', 'report': project / 'outputs/reports/discovery_multimodal_synthesis_multimodal_analysis_conclusion_report.md', 'contract': project / 'config/discovery_multimodal_synthesis_multimodal_analysis_conclusion_contract.json', 'lock': project / 'outputs/qc/discovery_multimodal_synthesis_final_output_lock_hashes.json', 'log': project / 'outputs/logs/discovery_multimodal_synthesis_multimodal_analysis_conclusion.log'}

def verify_sources(project: Path):
    rows = []
    paths = {}
    for label, spec in SOURCES.items():
        lp = project / spec['lock']
        require([lp])
        obs_lock = sha(lp)
        if obs_lock != spec['lock_sha']:
            raise RuntimeError(f'{label} final-lock SHA mismatch: {obs_lock}')
        lock = jread(lp)
        if lock.get('phase') != spec['phase']:
            raise RuntimeError(f'{label} phase mismatch')
        hashes = lock.get('hashes', {})
        rows.append({'source_phase': label, 'artifact': 'final_lock', 'relative_path': str(spec['lock']), 'expected_sha256': spec['lock_sha'], 'observed_sha256': obs_lock, 'passed': True})
        paths[label] = {}
        for name, rel in spec['files'].items():
            p = project / rel
            require([p])
            obs = sha(p)
            key = spec['keys'][name]
            if key not in hashes:
                raise RuntimeError(f'{label} final lock missing key {key}')
            exp = str(hashes[key])
            ok = obs == exp
            rows.append({'source_phase': label, 'artifact': name, 'relative_path': str(rel), 'expected_sha256': exp, 'observed_sha256': obs, 'passed': ok})
            if not ok:
                raise RuntimeError(f'{label} artifact SHA mismatch: {name}')
            paths[label][name] = p
    return (paths, pd.DataFrame(rows))

def self_test():
    p = np.array([0.6373626373626373, 0.6373626373626373, 0.3722627737226277])
    assert np.all(p >= 0.05)
    assert not bool(np.any(p < 0.05))
    f1 = np.array([749 / 1237, 263 / 437, 270 / 446])
    assert float(np.max(f1) - np.min(f1)) < 0.01
    print('discovery_multimodal_synthesis self-test: PASS')

def main():
    a = parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    out = output_paths(project)
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not a.force):
        raise RuntimeError('discovery_multimodal_synthesis already exists; use --force only for an identical rerun')
    logs = []

    def note(x):
        line = f'[{now()}] {x}'
        print(line, flush=True)
        logs.append(line)
    note('Verifying fixed three_cohort_structural_synthesis and multimodal_physio_design-C source artifacts; no held-out physiology values are opened.')
    src, rehash = verify_sources(project)
    three_cohort_structural_synthesis = jread(src['three_cohort_structural_synthesis']['summary'])
    validation_geometry_transfer_design = jread(src['three_cohort_structural_synthesis']['interpretation'])
    a4 = jread(src['multimodal_physio_design']['summary'])
    a_policy = jread(src['multimodal_physio_design']['policy'])
    b4 = jread(src['discovery_multimodal_attachment_qc']['summary'])
    bnext = jread(src['discovery_multimodal_attachment_qc']['next'])
    c4 = jread(src['discovery_multimodal_inference']['summary'])
    cnext = jread(src['discovery_multimodal_inference']['next'])
    resolution = jread(src['discovery_multimodal_inference']['resolution'])
    gt = pd.read_csv(src['discovery_multimodal_inference']['global'])
    bt = pd.read_csv(src['discovery_multimodal_inference']['bootstrap'])
    rb = pd.read_csv(src['discovery_multimodal_inference']['robustness'])
    cc = pd.read_csv(src['discovery_multimodal_inference']['complete_case'])
    if not three_cohort_structural_synthesis['decision'].get('three_split_cohort_fixed_structural_replication_supported', False):
        raise RuntimeError('three_cohort_structural_synthesis structural replication was not supported')
    if three_cohort_structural_synthesis['decision'].get('hedonic_emotional_contextual_or_stimulus_interpretation_supported', True):
        raise RuntimeError('Unexpected semantic support in three_cohort_structural_synthesis')
    if not bnext.get('next_phase_permitted', False) or bnext.get('next_phase') != 'discovery_multimodal_inference':
        raise RuntimeError('discovery_multimodal_attachment_qc did not authorize discovery_multimodal_inference')
    if cnext.get('next_phase') != PHASE or not cnext.get('next_phase_permitted', False):
        raise RuntimeError('discovery_multimodal_inference did not authorize discovery_multimodal_synthesis')
    if cnext.get('Validation_physio_may_open_next_phase', True) or cnext.get('Confirmation_physio_may_open', True):
        raise RuntimeError('Held-out physiology boundary unexpectedly opened')
    if c4['boundary'].get('Validation_physio_values_opened', True) or c4['boundary'].get('Confirmation_physio_values_opened', True):
        raise RuntimeError('Held-out physiology values were opened in discovery_multimodal_inference')
    if int(c4['counts']['M1_M3_permutations']) != 10000 or int(c4['counts']['M2_permutations']) != 1000 or int(c4['counts']['bootstraps']) != 5000:
        raise RuntimeError('prespecified inference replicate counts changed')
    gt = gt.sort_values('test').reset_index(drop=True)
    rb = rb.sort_values('test').reset_index(drop=True)
    bt = bt.sort_values('test').reset_index(drop=True)
    if gt.test.tolist() != ['M1', 'M2', 'M3']:
        raise RuntimeError('Expected M1-M3 exactly')
    if gt.primary_supported.map(bval).any() or gt.robustness_pass.map(bval).any() or gt.validation_gate_pass.map(bval).any():
        raise RuntimeError('discovery_multimodal_synthesis closure is valid only when M1-M3 all fail the prespecified held-out gate')
    if (pd.to_numeric(gt.holm_p, errors='coerce') < 0.05).any():
        raise RuntimeError('Unexpected Holm-supported multimodal test')
    evid = gt.merge(bt[['test', 'bootstrap_ci_low', 'bootstrap_ci_high']], on='test', how='left', validate='one_to_one')
    evid = evid.merge(rb[['test', 'unadjusted_p', 'complete_case_p_or_gain', 'min_lopo_cosine', 'participant_gain_median', 'participant_gain_nonnegative_fraction', 'block_followup_any_supported', 'robustness_pass']], on='test', how='left', validate='one_to_one', suffixes=('', '_rob'))
    descriptions = {'M1': 'fixed EEG Family 1/2 difference in the 40D trial-level physiology profile after nuisance adjustment.', 'M2': 'Out-of-person prediction of fixed EEG family from physiology above context + normalized trial-order baseline.', 'M3': 'Participant-balanced residual coupling between 40D physiology and the immutable six-dimensional EEG trajectory geometry.'}
    evid.insert(1, 'prespecified_question', [descriptions[x] for x in evid.test])
    evid['heldout_transfer_permitted'] = False
    evid['branch_conclusion'] = 'unsupported_under_prespecified_discovery_gate'
    rob_rows = []
    for _, r in rb.iterrows():
        test = str(r.test)
        if test == 'M1':
            checks = [('unadjusted_p_lt_0.05', float(r.unadjusted_p) < 0.05), ('complete_case_p_lt_0.05', float(r.complete_case_p_or_gain) < 0.05), ('min_lopo_cosine_ge_0.90', float(r.min_lopo_cosine) >= 0.9), ('gated_block_supported', bval(r.block_followup_any_supported))]
        elif test == 'M2':
            checks = [('bootstrap_ci_low_gt_0', float(r.bootstrap_ci_low) > 0), ('complete_case_gain_gt_0', float(r.complete_case_p_or_gain) > 0), ('participant_gain_median_gt_0', float(r.participant_gain_median) > 0), ('participant_gain_nonnegative_fraction_ge_0.60', float(r.participant_gain_nonnegative_fraction) >= 0.6)]
        else:
            checks = [('unadjusted_p_lt_0.05', float(r.unadjusted_p) < 0.05), ('complete_case_p_lt_0.05', float(r.complete_case_p_or_gain) < 0.05), ('min_lopo_cosine_ge_0.90', float(r.min_lopo_cosine) >= 0.9), ('gated_block_supported', bval(r.block_followup_any_supported))]
        for check, passed in checks:
            rob_rows.append({'test': test, 'prespecified_robustness_check': check, 'passed': bool(passed)})
    robustness = pd.DataFrame(rob_rows)
    interpretation = {'phase': PHASE, 'generated_at': now(), 'supported_claims': ['The fixed outcome-free two-family EEG trajectory geometry replicated structurally across the Discovery, Validation, and Confirmation splits of this dataset.', 'In Discovery participants with joined trial-level P0.2 physiology, none of the three pre-prespecified multimodal tests M1-M3 passed Holm correction.', 'No M1-M3 test passed the pre-prespecified combined primary-plus-robustness gate required to open held-out physiology.', 'Therefore Validation and Confirmation physiology remain unopened for this branch, preserving the held-out boundary.'], 'not_supported_claims': ['Trial-level peripheral physiology has no relation to EEG trajectory geometry in general.', 'Raw time-resolved autonomic dynamics have been tested; only the prespecified P0.2 trial-level 40-feature representation was tested.', 'Family 1/2 are pleasure, suffering, valence, liking, emotion, context, stimulus, or causal states.', 'Specific GSR or HR features are negative or positive biomarkers; parent gates did not license feature-level follow-ups.', 'Held-out physiology replicated a null result; held-out physiology was intentionally not opened.', 'The absence of Discovery support proves biological independence between brain and body signals.'], 'family_labels_fixed': ['Family 1', 'Family 2'], 'family_labels_are_geometric_only': True, 'posthoc_physio_feature_selection_forbidden': True, 'heldout_physio_testing_for_unsupported_M1_M3_forbidden': True, 'raw_dynamic_physio_out_of_scope': True}
    heldout = {'phase': PHASE, 'generated_at': now(), 'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'Validation_physio_may_open_for_M1_M3': False, 'Confirmation_physio_may_open_for_M1_M3': False, 'reason': 'No Discovery M1-M3 test passed both Holm p<0.05 and all applicable prespecified robustness rules.', 'supported_tests_for_validation': [], 'closure_is_successful_boundary_preservation': True, 'forbidden': ['testing M1-M3 in held-out physiology', 'retuning physiology preprocessing', 'outcome-driven physiology feature selection', 'redefining EEG families using physiology', 'opening held-out physiology to seek a null replication']}
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'integrated project evidence synthesis analysis specification; no new inferential testing', 'required_inputs': ['fixed subjective-architecture evidence', 'three_cohort_structural_synthesis final three-split EEG structural synthesis', 'discovery_multimodal_synthesis multimodal analysis conclusion'], 'required_boundary': ['Family 1/2 remain geometric labels only.', 'Validation and Confirmation physiology remain unopened for unsupported M1-M3.', 'No post-hoc feature selection or semantic relabeling is permitted.', 'Any future new physiology analysis, especially raw dynamic physiology, requires a new prospective analysis specification and must not be presented as confirmation of M1-M3.'], 'forbidden_in_next_phase': ['new outcome inference', 'new physiology inference', 'family relabeling', 'held-out physiology opening', 'retroactive threshold tuning']}
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_multimodal_analysis_concluded_without_heldout_physio_opening', 'counts': {'discovery_joined_trials': int(c4['counts']['joined_trials']), 'eligible_M1_M2_participants': int(c4['counts']['eligible_M1_M2_participants']), 'eligible_M3_participants': int(c4['counts']['eligible_M3_participants']), 'biological_features': int(c4['counts']['biological_features']), 'M1_M3_permutations': int(c4['counts']['M1_M3_permutations']), 'M2_permutations': int(c4['counts']['M2_permutations']), 'bootstraps': int(c4['counts']['bootstraps']), 'primary_tests': 3, 'primary_tests_holm_supported': 0, 'primary_tests_passing_full_heldout_gate': 0}, 'tests': evid[['test', 'observed_statistic', 'raw_p', 'holm_p', 'primary_supported', 'robustness_pass', 'validation_gate_pass']].to_dict(orient='records'), 'decision': {'trajectory_geometry_eeg_structural_result_preserved': True, 'multimodal_discovery_support_any': False, 'Validation_physio_opened': False, 'Confirmation_physio_opened': False, 'heldout_physio_transfer_permitted': False, 'multimodal_analysis_concluded': True, 'negative_result_interpreted_as_representation_specific_not_proof_of_independence': True, 'integrated_synthesis_integrated_synthesis_specification_permitted': True}}
    rehash.to_csv(out['rehash'], index=False)
    evid.to_csv(out['evidence'], index=False)
    robustness.to_csv(out['robustness'], index=False)
    atom_json(out['interpretation'], interpretation)
    atom_json(out['heldout'], heldout)
    atom_json(out['next'], next_policy)
    atom_json(out['summary'], summary)
    report = f"# Phase discovery_multimodal_synthesis — Discovery multimodal synthesis and analysis conclusion\n\nStatus: **{summary['status']}**\n\n## prespecified Discovery multimodal result\n\n| Test | Observed statistic | Raw p | Holm p | Full held-out gate |\n|---|---:|---:|---:|---|\n| M1 | {float(gt.loc[gt.test.eq('M1'), 'observed_statistic'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M1'), 'raw_p'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M1'), 'holm_p'].iloc[0]):.6f} | FAIL |\n| M2 | {float(gt.loc[gt.test.eq('M2'), 'observed_statistic'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M2'), 'raw_p'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M2'), 'holm_p'].iloc[0]):.6f} | FAIL |\n| M3 | {float(gt.loc[gt.test.eq('M3'), 'observed_statistic'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M3'), 'raw_p'].iloc[0]):.6f} | {float(gt.loc[gt.test.eq('M3'), 'holm_p'].iloc[0]):.6f} | FAIL |\n\nM2 improved participant-balanced accuracy from {float(gt.loc[gt.test.eq('M2'), 'baseline_balanced_accuracy'].iloc[0]):.6f} to {float(gt.loc[gt.test.eq('M2'), 'full_balanced_accuracy'].iloc[0]):.6f}, a gain of only {float(gt.loc[gt.test.eq('M2'), 'observed_statistic'].iloc[0]):.6f}; its participant-bootstrap 95% CI was [{float(bt.loc[bt.test.eq('M2'), 'bootstrap_ci_low'].iloc[0]):.6f}, {float(bt.loc[bt.test.eq('M2'), 'bootstrap_ci_high'].iloc[0]):.6f}].\n\n## Interpretation boundary\n\nThe three-split EEG structural result remains intact. The Discovery-only P0.2 physiology representation did not provide prespecified-gate support for (M1) a Family 1/2 physiology-profile difference, (M2) out-of-person family prediction above context/trial-order baseline, or (M3) continuous EEG-geometry/physiology coupling.\n\nThis is a representation-specific negative result, not proof that brain and body are independent. Raw time-resolved autonomic dynamics were not tested. Feature-level GSR/HR follow-ups were not licensed because their parent gates failed.\n\n## Held-out boundary\n\nValidation and Confirmation physiology remain unopened. Testing unsupported M1-M3 in those held-out splits is forbidden under the prespecified policy.\n\n## subsequent analysis\n\nThe multimodal branch is closed. The permitted next phase is {NEXT_PHASE}: an integrated evidence synthesis analysis specification using already fixed evidence only, with no new inferential testing."
    out['report'].write_text(report, encoding='utf-8')
    output_hashes = {k: sha(p) for k, p in out.items() if k not in {'log', 'lock', 'contract'}}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'source_final_lock_sha256': {k: v['lock_sha'] for k, v in SOURCES.items()}, 'output_hashes': output_hashes, 'boundary': summary['decision']}
    atom_json(out['contract'], contract)
    output_hashes['contract'] = sha(out['contract'])
    lock = {'generated_at': now(), 'phase': PHASE, 'hashes': output_hashes, 'multimodal_analysis_concluded': True, 'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'integrated_synthesis_integrated_synthesis_specification_permitted': True}
    atom_json(out['lock'], lock)
    note('discovery_multimodal_synthesis complete: M1-M3 unsupported under prespecified gates; held-out physiology remains closed; multimodal branch closed.')
    out['log'].write_text(''.join(logs) + '', encoding='utf-8')
    include = [p for k, p in out.items() if k != 'log']
    zsha = sha(z)
    sp = Path(str(z) + '.sha256')
    print(f'SHA-256: {zsha}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())