# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, hashlib, json, math
from datetime import datetime
from pathlib import Path
import pandas as pd
PHASE = 'dynamic_gsr_synthesis'
UPSTREAM_PHASE = 'discovery_dynamic_gsr_inference'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
J_LOCK = 'outputs/qc/discovery_dynamic_gsr_inference_final_output_lock_hashes.json'
J_SUM = 'outputs/qc/discovery_dynamic_gsr_inference_dynamic_gsr_inference_summary.json'
J_GLOBAL = 'outputs/tables/discovery_dynamic_gsr_inference_global_tests.csv'
J_ROBUST = 'outputs/tables/discovery_dynamic_gsr_inference_robustness_summary.csv'
J_BOOT = 'outputs/tables/discovery_dynamic_gsr_inference_bootstrap_summary.csv'
J_POLICY = 'analysis_specifications/S017_discovery_dynamic_gsr_inference.json'
LOCK_KEY_PATH = {'blocks': 'outputs/tables/discovery_dynamic_gsr_inference_gated_scr_scl_followups.csv', 'boot': 'outputs/tables/discovery_dynamic_gsr_inference_bootstrap_summary.csv', 'bootrep': 'outputs/tables/discovery_dynamic_gsr_inference_bootstrap_replicates.csv', 'cc': 'outputs/tables/discovery_dynamic_gsr_inference_complete_case_sensitivity.csv', 'contract': 'config/discovery_dynamic_gsr_inference_dynamic_gsr_inference_contract.json', 'dg1null': 'outputs/tables/discovery_dynamic_gsr_inference_dg1_null_statistics.csv', 'dg1part': 'outputs/tables/discovery_dynamic_gsr_inference_dg1_participant_contrasts.csv', 'dg1vec': 'outputs/tables/discovery_dynamic_gsr_inference_dg1_contrast_vector.csv', 'dg2null': 'outputs/tables/discovery_dynamic_gsr_inference_dg2_null_statistics.csv', 'dg2pred': 'outputs/tables/discovery_dynamic_gsr_inference_dg2_oof_predictions.csv', 'dg2sel': 'outputs/tables/discovery_dynamic_gsr_inference_dg2_hyperparameter_selection.csv', 'dg3mat': 'outputs/tables/discovery_dynamic_gsr_inference_dg3_correlation_matrix.csv', 'dg3null': 'outputs/tables/discovery_dynamic_gsr_inference_dg3_null_statistics.csv', 'dg3part': 'outputs/tables/discovery_dynamic_gsr_inference_dg3_participant_matrices.csv', 'global': 'outputs/tables/discovery_dynamic_gsr_inference_global_tests.csv', 'iscr': 'outputs/tables/discovery_dynamic_gsr_inference_iscr_augmented_sensitivity.csv', 'lopo': 'outputs/tables/discovery_dynamic_gsr_inference_lopo_diagnostics.csv', 'next': 'analysis_specifications/S017_discovery_dynamic_gsr_inference.json', 'rehash': 'outputs/qc/discovery_dynamic_gsr_inference_source_rehash_audit.csv', 'report': 'outputs/reports/discovery_dynamic_gsr_inference_discovery_dynamic_gsr_inference.md', 'resolution': 'config/discovery_dynamic_gsr_inference_implementation_resolution.json', 'robust': 'outputs/tables/discovery_dynamic_gsr_inference_robustness_summary.csv', 'roster': 'outputs/tables/discovery_dynamic_gsr_inference_inferential_eligibility_roster.csv', 'summary': 'outputs/qc/discovery_dynamic_gsr_inference_dynamic_gsr_inference_summary.json'}

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def verify_discovery_dynamic_gsr_inference(project: Path):
    lp = project / J_LOCK
    if not lp.is_file():
        raise RuntimeError(f'Missing discovery_dynamic_gsr_inference final lock: {lp}')
    lock = readj(lp)
    if lock.get('phase') != UPSTREAM_PHASE:
        raise RuntimeError(f"Wrong upstream phase: {lock.get('phase')}")
    if set(lock.get('hashes', {})) != set(LOCK_KEY_PATH):
        raise RuntimeError('discovery_dynamic_gsr_inference lock key roster drift')
    mismatches = []
    rows = []
    for key, rel in LOCK_KEY_PATH.items():
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing fixed discovery_dynamic_gsr_inference artifact: {rel}')
        got = sha256(p)
        exp = lock['hashes'][key]
        ok = got == exp
        rows.append({'lock_key': key, 'relative_path': rel, 'expected_sha256': exp, 'observed_sha256': got, 'match': ok})
        if not ok:
            mismatches.append((key, rel, got, exp))
    if mismatches:
        raise RuntimeError(f'discovery_dynamic_gsr_inference output lock mismatch: {mismatches[:3]}')
    return (lock, pd.DataFrame(rows))

def self_test():
    fake = pd.DataFrame([{'test': 'DG1', 'primary_supported': False, 'robustness_pass': False, 'heldout_gate_pass': False}, {'test': 'DG2', 'primary_supported': False, 'robustness_pass': False, 'heldout_gate_pass': False}, {'test': 'DG3', 'primary_supported': False, 'robustness_pass': False, 'heldout_gate_pass': False}])
    if fake['heldout_gate_pass'].any():
        raise RuntimeError('closure self-test failed')
    print('dynamic_gsr_synthesis self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    lock, rehash = verify_discovery_dynamic_gsr_inference(project)
    summary = readj(project / J_SUM)
    policy = readj(project / J_POLICY)
    gt = pd.read_csv(project / J_GLOBAL)
    rb = pd.read_csv(project / J_ROBUST)
    bt = pd.read_csv(project / J_BOOT)
    required_tests = ['DG1', 'DG2', 'DG3']
    if gt['test'].tolist() != required_tests:
        raise RuntimeError(f"Unexpected global-test roster: {gt['test'].tolist()}")
    if gt['primary_supported'].astype(str).str.lower().isin(['true', '1']).any():
        raise RuntimeError('dynamic_gsr_synthesis closure is invalid because a primary test is supported')
    if gt['heldout_gate_pass'].astype(str).str.lower().isin(['true', '1']).any():
        raise RuntimeError('dynamic_gsr_synthesis closure is invalid because held-out gate passed')
    if summary.get('decision', {}).get('Discovery_support_plus_robustness_any') is not False:
        raise RuntimeError('discovery_dynamic_gsr_inference summary does not authorize analysis conclusion')
    if summary.get('decision', {}).get('Validation_GSR_derivative_values_may_open_after_dynamic_gsr_synthesis_specification') is not False:
        raise RuntimeError('Validation opening was unexpectedly authorized')
    if policy.get('next_phase') != PHASE:
        raise RuntimeError(f"discovery_dynamic_gsr_inference policy does not route to dynamic_gsr_synthesis: {policy.get('next_phase')}")
    if policy.get('supported_tests_for_possible_validation') != []:
        raise RuntimeError('Unexpected supported test list')
    if lock.get('Validation_GSR_derivative_values_may_open_after_dynamic_gsr_synthesis_specification') is not False:
        raise RuntimeError('discovery_dynamic_gsr_inference final lock boundary mismatch')
    if lock.get('Confirmation_GSR_derivative_values_may_open') is not False:
        raise RuntimeError('Confirmation unexpectedly open')
    if lock.get('behavioral_outcomes_read') is not False:
        raise RuntimeError('Behavioral outcome boundary mismatch')
    if lock.get('rating_period_physio_tested') is not False:
        raise RuntimeError('Rating physiology boundary mismatch')
    g = {r['test']: r for _, r in gt.iterrows()}
    boot = {r['test']: r for _, r in bt.iterrows()}
    evidence_rows = [{'claim_id': 'dynamic_gsr_synthesis_DG1', 'claim': 'Author-provided 10x3-s stimulus-period GSR dynamics did not support a Discovery difference between fixed EEG trajectory families beyond prespecified nuisance adjustment.', 'test': 'DG1', 'observed_statistic': float(g['DG1']['observed_statistic']), 'raw_p': float(g['DG1']['raw_p']), 'holm_p': float(g['DG1']['holm_p']), 'supported': False, 'scope': 'Discovery stimulus-period GSR only'}, {'claim_id': 'dynamic_gsr_synthesis_DG2', 'claim': 'Author-provided stimulus-period dynamic GSR did not improve unseen-person prediction of fixed EEG trajectory family beyond context and acquisition order.', 'test': 'DG2', 'observed_statistic': float(g['DG2']['observed_statistic']), 'raw_p': float(g['DG2']['raw_p']), 'holm_p': float(g['DG2']['holm_p']), 'supported': False, 'scope': 'Discovery stimulus-period GSR only'}, {'claim_id': 'dynamic_gsr_synthesis_DG3', 'claim': 'Author-provided stimulus-period dynamic GSR did not support coupling to the fixed continuous 6D EEG trajectory geometry beyond prespecified nuisance adjustment.', 'test': 'DG3', 'observed_statistic': float(g['DG3']['observed_statistic']), 'raw_p': float(g['DG3']['raw_p']), 'holm_p': float(g['DG3']['holm_p']), 'supported': False, 'scope': 'Discovery stimulus-period GSR only'}]
    decision = {'phase': PHASE, 'generated_at': now(), 'status': 'dynamic_gsr_discovery_analysis_concluded_without_heldout_opening', 'reason': 'No DG1-DG3 primary test was Holm-supported; therefore the prespecified support+robustness gate could not pass.', 'DG1_primary_supported': False, 'DG2_primary_supported': False, 'DG3_primary_supported': False, 'Discovery_support_plus_robustness_any': False, 'Validation_GSR_derivative_values_opened': False, 'Validation_GSR_derivative_values_may_open': False, 'Confirmation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_may_open': False, 'raw_event_aligned_GSR_PPG_amplitude_opened': False, 'rating_period_physio_tested': False, 'behavioral_outcomes_read_in_dynamic_GSR_branch': False, 'next_phase': 'interpretability_route_specification', 'next_phase_role': 'Final integrated evidence refresh and reproducibility specification incorporating the corrected raw-sync feasibility history and the completed stimulus-period dynamic-GSR Discovery null result.'}
    paths = {'rehash': project / 'outputs/qc/dynamic_gsr_synthesis_discovery_dynamic_gsr_inference_rehash_audit.csv', 'evidence': project / 'outputs/tables/dynamic_gsr_synthesis_dynamic_gsr_evidence_rows.csv', 'decision': project / 'config/dynamic_gsr_synthesis_dynamic_gsr_analysis_conclusion_decision.json', 'summary': project / 'outputs/qc/dynamic_gsr_synthesis_dynamic_gsr_analysis_conclusion_summary.json', 'report': project / 'outputs/reports/dynamic_gsr_synthesis_dynamic_gsr_analysis_conclusion.md', 'contract': project / 'config/dynamic_gsr_synthesis_dynamic_gsr_analysis_conclusion_contract.json'}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    rehash.to_csv(paths['rehash'], index=False)
    pd.DataFrame(evidence_rows).to_csv(paths['evidence'], index=False)
    writej(paths['decision'], decision)
    closure_summary = {'phase': PHASE, 'generated_at': now(), 'status': decision['status'], 'discovery_counts': summary.get('counts', {}), 'tests': {'DG1': {'statistic': float(g['DG1']['observed_statistic']), 'raw_p': float(g['DG1']['raw_p']), 'holm_p': float(g['DG1']['holm_p']), 'bootstrap_ci': [float(boot['DG1']['bootstrap_ci_low']), float(boot['DG1']['bootstrap_ci_high'])], 'primary_supported': False}, 'DG2': {'baseline_balanced_accuracy': float(g['DG2']['baseline_balanced_accuracy']), 'full_balanced_accuracy': float(g['DG2']['full_balanced_accuracy']), 'gain': float(g['DG2']['observed_statistic']), 'raw_p': float(g['DG2']['raw_p']), 'holm_p': float(g['DG2']['holm_p']), 'bootstrap_ci': [float(boot['DG2']['bootstrap_ci_low']), float(boot['DG2']['bootstrap_ci_high'])], 'primary_supported': False}, 'DG3': {'statistic': float(g['DG3']['observed_statistic']), 'raw_p': float(g['DG3']['raw_p']), 'holm_p': float(g['DG3']['holm_p']), 'bootstrap_ci': [float(boot['DG3']['bootstrap_ci_low']), float(boot['DG3']['bootstrap_ci_high'])], 'primary_supported': False}}, 'boundaries': {'Discovery_GSR_derivative_values_opened': True, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'raw_event_aligned_GSR_PPG_amplitude_opened': False, 'rating_period_physio_tested': False, 'new_inference_performed_in_dynamic_gsr_synthesis': False, 'heldout_peeking_performed_in_dynamic_gsr_synthesis': False}, 'interpretation': {'permitted': "Within the authors' released 10x3-s stimulus-period GSR representation, Discovery did not support DG1 family profile differences, DG2 unseen-person family prediction gain, or DG3 coupling to continuous EEG trajectory geometry.", 'not_permitted': ['claiming no autonomic physiology is related to emotion or EEG in general', 'claiming rating-period physiology was tested', 'claiming raw event-aligned GSR/PPG dynamics were analyzed', 'opening held-out GSR derivative values after unsupported Discovery primaries']}, 'next_phase': decision['next_phase']}
    writej(paths['summary'], closure_summary)
    report = f"# dynamic_gsr_synthesis — Dynamic GSR Discovery analysis conclusion\n\nStatus: **{decision['status']}**\n\ndiscovery_dynamic_gsr_inference was independently re-hashed against all 24 final-lock entries with zero mismatches.\n\n## prespecified Discovery results\n\n- DG1 family-profile contrast: statistic={g['DG1']['observed_statistic']:.9g}; raw p={g['DG1']['raw_p']:.6g}; Holm p={g['DG1']['holm_p']:.6g}; unsupported.\n- DG2 unseen-person family prediction: baseline BA={g['DG2']['baseline_balanced_accuracy']:.6f}; full BA={g['DG2']['full_balanced_accuracy']:.6f}; delta BA={g['DG2']['observed_statistic']:.9g}; raw p={g['DG2']['raw_p']:.6g}; Holm p={g['DG2']['holm_p']:.6g}; bootstrap 95% CI [{boot['DG2']['bootstrap_ci_low']:.9g}, {boot['DG2']['bootstrap_ci_high']:.9g}]; unsupported.\n- DG3 GSR-to-EEG-geometry coupling: statistic={g['DG3']['observed_statistic']:.9g}; raw p={g['DG3']['raw_p']:.6g}; Holm p={g['DG3']['holm_p']:.6g}; unsupported.\n\nNo primary was Holm-supported. Therefore no primary could satisfy the prespecified support+robustness gate, no SCR/SCL or iSCR sensitivity analysis could rescue an unsupported primary, and held-out GSR derivative values remain closed.\n\n## Scientific meaning\n\nThis closes the **author-provided stimulus-period dynamic GSR** Discovery branch without Validation or Confirmation opening.\n\nThe result does **not** establish that autonomic physiology is unrelated to EEG or affect. It only says that the prespecified released 10x3-s GSR representation did not support the three prespecified Discovery hypotheses.\n\nRating-period physiology remains untested, and raw event-aligned GSR/PPG amplitude was never analyzed because <=1-s exact raw EEG↔physiology synchronization was not recovered.\n\n## Next\n\nProceed to **interpretability_route_specification** for the final integrated evidence refresh plus reproducibility specification. topographic_analysis must preserve all fixed empirical results and must not open held-out physiology."
    paths['report'].write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'status': decision['status'], 'inputs': {'discovery_dynamic_gsr_inference_final_lock_sha256': sha256(project / J_LOCK), 'discovery_dynamic_gsr_inference_summary_sha256': sha256(project / J_SUM), 'discovery_dynamic_gsr_inference_global_tests_sha256': sha256(project / J_GLOBAL), 'discovery_dynamic_gsr_inference_robustness_sha256': sha256(project / J_ROBUST), 'discovery_dynamic_gsr_inference_bootstrap_sha256': sha256(project / J_BOOT), 'discovery_dynamic_gsr_inference_policy_sha256': sha256(project / J_POLICY)}, 'new_inference_performed': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'raw_event_aligned_GSR_PPG_amplitude_opened': False, 'rating_period_physio_tested': False, 'next_phase': decision['next_phase'], 'outputs': {}}
    lockhash = {}
    for key, p in paths.items():
        if key == 'contract':
            continue
        lockhash[str(p.relative_to(project))] = sha256(p)
    contract['outputs'] = dict(lockhash)
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/dynamic_gsr_synthesis_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': decision['status'], 'hashes': lockhash, 'Discovery_support_plus_robustness_any': False, 'Validation_GSR_derivative_values_opened': False, 'Validation_GSR_derivative_values_may_open': False, 'Confirmation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_may_open': False, 'raw_event_aligned_GSR_PPG_amplitude_opened': False, 'rating_period_physio_tested': False, 'new_inference_performed': False, 'next_phase': decision['next_phase']})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'dynamic_gsr_synthesis_dynamic_gsr_analysis_conclusion_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    h = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{h}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] dynamic_gsr_synthesis PASS', flush=True)
    print('discovery_dynamic_gsr_inference fixed artifacts verified: 24/24', flush=True)
    print(f"DG1 Holm p={g['DG1']['holm_p']:.6g}; supported=False", flush=True)
    print(f"DG2 deltaBA={g['DG2']['observed_statistic']:.9g}; Holm p={g['DG2']['holm_p']:.6g}; supported=False", flush=True)
    print(f"DG3 Holm p={g['DG3']['holm_p']:.6g}; supported=False", flush=True)
    print('Validation/Confirmation GSR derivative values opened: False', flush=True)
    print('Next phase:', decision['next_phase'], flush=True)
    print('', hz, flush=True)
    print('SHA-256:', h, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())