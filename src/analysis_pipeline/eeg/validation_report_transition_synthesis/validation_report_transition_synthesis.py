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
PHASE = 'validation_report_transition_synthesis'
NEXT_PHASE = 'confirmation_report_transition_transfer_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
EXPECTED_TESTED_K = [2, 3]
EXPECTED_PASSING_K = [2, 3]
validation_report_transition_multiresolution_transfer_OUT = Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer')
validation_report_transition_multiresolution_transfer_LOCK = validation_report_transition_multiresolution_transfer_OUT / 'qc/validation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'
validation_report_transition_multiresolution_transfer_LOCK_SHA256 = '8025e9b1ca3357ebcce055554bda3df6b12e1d1bae08737757cb586ba1faf715'
validation_report_transition_multiresolution_transfer_SUMMARY = validation_report_transition_multiresolution_transfer_OUT / 'qc/validation_report_transition_multiresolution_transfer_summary.json'
validation_report_transition_multiresolution_transfer_RESOLUTION = validation_report_transition_multiresolution_transfer_OUT / 'tables/validation_report_transition_multiresolution_transfer_validation_resolution_summary.csv'
validation_report_transition_multiresolution_transfer_GATE = validation_report_transition_multiresolution_transfer_OUT / 'qc/validation_report_transition_multiresolution_transfer_validation_structural_transfer_gate_all_k.csv'
validation_report_transition_multiresolution_transfer_REPORT = validation_report_transition_multiresolution_transfer_OUT / 'reports/validation_report_transition_multiresolution_transfer_report.md'
validation_report_transition_multiresolution_transfer_NEXT_POLICY = Path('analysis_specifications/S085_validation_report_transition_multiresolution_transfer.json')
validation_report_transition_multiresolution_transfer_NEXT_POLICY_SHA256 = 'be89191a5bd408abc8e67ab4e844171bb8ef3db3c7af4127a4b60dffd694cf75'
OUT = Path('outputs/interpretability/validation_report_transition_synthesis_validation_report_transition_synthesis')
NEXT_CONFIG = Path('analysis_specifications/S086_validation_report_transition_synthesis_confirmation.json')

def now() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')
    os.replace(tmp, path)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)

def verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')
    return got

def verify_parent_lock(project: Path) -> list[dict[str, str]]:
    lock_path = project / validation_report_transition_multiresolution_transfer_LOCK
    verify(lock_path, validation_report_transition_multiresolution_transfer_LOCK_SHA256, 'validation_report_transition_multiresolution_transfer final lock')
    lock = readj(lock_path)
    if lock.get('phase') != 'validation_report_transition_multiresolution_transfer':
        raise RuntimeError('validation_report_transition_multiresolution_transfer final-lock phase drift')
    if lock.get('tested_K') != EXPECTED_TESTED_K:
        raise RuntimeError(f"validation_report_transition_multiresolution_transfer tested_K drift: {lock.get('tested_K')}")
    if lock.get('passing_K') != EXPECTED_PASSING_K:
        raise RuntimeError(f"validation_report_transition_multiresolution_transfer passing_K drift: {lock.get('passing_K')}")
    if not bool(lock.get('any_K_replicated')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer says no K replicated')
    for key in ['Validation_behavioral_values_read', 'Confirmation_signal_values_read', 'centroids_refit', 'thresholds_tuned', 'K_reselected', 'semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
        if bool(lock.get(key)):
            raise RuntimeError(f'validation_report_transition_multiresolution_transfer boundary violation: {key}=True')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError('validation_report_transition_multiresolution_transfer final lock has no child hashes')
    rows: list[dict[str, str]] = []
    for rel, expected in sorted(hashes.items()):
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing validation_report_transition_multiresolution_transfer fixed child: {rel}')
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'validation_report_transition_multiresolution_transfer fixed-child SHA drift: {rel}; expected {expected}, got {got}')
        rows.append({'source': rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    return rows

def validate_parent_decision(project: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    verify(project / validation_report_transition_multiresolution_transfer_NEXT_POLICY, validation_report_transition_multiresolution_transfer_NEXT_POLICY_SHA256, 'validation_report_transition_multiresolution_transfer→validation_report_transition_synthesis prespecified policy')
    policy = readj(project / validation_report_transition_multiresolution_transfer_NEXT_POLICY)
    summary = readj(project / validation_report_transition_multiresolution_transfer_SUMMARY)
    if summary.get('phase') != 'validation_report_transition_multiresolution_transfer':
        raise RuntimeError('validation_report_transition_multiresolution_transfer summary phase drift')
    if summary.get('tested_K') != EXPECTED_TESTED_K:
        raise RuntimeError(f"Unexpected validation_report_transition_multiresolution_transfer tested_K: {summary.get('tested_K')}")
    if summary.get('passing_K') != EXPECTED_PASSING_K:
        raise RuntimeError(f"Unexpected validation_report_transition_multiresolution_transfer passing_K: {summary.get('passing_K')}")
    if not bool(summary.get('any_K_replicated')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer summary says no K replicated')
    boundary = summary.get('boundary', {})
    for key in ['Validation_behavioral_values_read', 'Validation_context_values_read', 'Validation_emotion_category_values_read', 'Validation_stimulus_identity_values_read', 'Confirmation_signal_values_read', 'centroids_refit', 'thresholds_tuned', 'K_reselected', 'family_semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
        if bool(boundary.get(key)):
            raise RuntimeError(f'validation_report_transition_multiresolution_transfer summary boundary violation: {key}=True')
    if not bool(boundary.get('Validation_stimulus_late_values_read')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer boundary drift: stimulus_late not marked read')
    if not bool(boundary.get('Validation_rating_early_values_read')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer boundary drift: rating_early not marked read')
    if policy.get('passing_K') != EXPECTED_PASSING_K:
        raise RuntimeError('validation_report_transition_multiresolution_transfer→validation_report_transition_synthesis policy passing_K drift')
    if not bool(policy.get('Confirmation_design_specification_permitted')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer→validation_report_transition_synthesis policy does not permit Confirmation analysis specification')
    if bool(policy.get('Confirmation_signal_opening_in_validation_report_transition_synthesis')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer→validation_report_transition_synthesis policy incorrectly permits Confirmation signal opening')
    if bool(policy.get('semantic_tests_opening_in_validation_report_transition_synthesis')):
        raise RuntimeError('validation_report_transition_multiresolution_transfer→validation_report_transition_synthesis policy incorrectly permits semantic tests in validation_report_transition_synthesis')
    rows = read_csv(project / validation_report_transition_multiresolution_transfer_RESOLUTION)
    by_k = {int(r['k']): r for r in rows}
    if sorted(by_k) != EXPECTED_TESTED_K:
        raise RuntimeError(f'validation_report_transition_multiresolution_transfer resolution table K drift: {sorted(by_k)}')
    for k in EXPECTED_PASSING_K:
        if str(by_k[k]['replicated']).strip().lower() not in {'true', '1', 'yes', 't'}:
            raise RuntimeError(f'validation_report_transition_multiresolution_transfer resolution table says K={k} did not replicate')
    return (summary, rows)

def self_test() -> int:
    assert EXPECTED_TESTED_K == [2, 3]
    assert EXPECTED_PASSING_K == [2, 3]
    assert PHASE == 'validation_report_transition_synthesis'
    assert NEXT_PHASE == 'confirmation_report_transition_transfer_design'
    print('validation_report_transition_synthesis self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    out = project / OUT
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] Verifying validation_report_transition_multiresolution_transfer one-shot Validation transfer and all fixed children', flush=True)
    rehash_rows = verify_parent_lock(project)
    summary_l, res_rows = validate_parent_decision(project)
    decision_rows: list[dict[str, Any]] = []
    for r in sorted(res_rows, key=lambda x: int(x['k'])):
        decision_rows.append({'k': int(r['k']), 'discovery_eligible': True, 'validation_replicated': True, 'advance_unchanged_to_confirmation_design': True, 'validation_eligible_participants': int(float(r['eligible_participants'])), 'validation_paired_report_transition_trials': int(float(r['paired_report_transition_trials_all_participants'])), 'validation_paired_fraction_of_494': float(r['paired_fraction_of_494']), 'validation_participant_balanced_in_support_fraction': float(r['participant_balanced_in_support_fraction']), 'validation_participant_balanced_low_margin_fraction': float(r['participant_balanced_low_margin_fraction']), 'rank_or_select_among_passing_k': False, 'semantic_name_permitted': False, 'distortion_interpretation_permitted': False})
    decision_path = out / 'tables/validation_report_transition_synthesis_confirmation_advancement_decision.csv'
    write_csv(decision_path, decision_rows, ['k', 'discovery_eligible', 'validation_replicated', 'advance_unchanged_to_confirmation_design', 'validation_eligible_participants', 'validation_paired_report_transition_trials', 'validation_paired_fraction_of_494', 'validation_participant_balanced_in_support_fraction', 'validation_participant_balanced_low_margin_fraction', 'rank_or_select_among_passing_k', 'semantic_name_permitted', 'distortion_interpretation_permitted'])
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'specification the one-shot Confirmation stimulus_late-to-rating_early structural transfer for K=2 and K=3 without opening Confirmation signal values', 'advance_K': EXPECTED_PASSING_K, 'advance_rule': 'all and only Validation-replicated report-transition K advance unchanged; no ranking, winner selection, K reselection, centroid refit, threshold tuning, family relabeling, or semantic interpretation', 'Confirmation_signal_opening_in_validation_report_transition_synthesis': False, 'Confirmation_signal_opening_in_confirmation_report_transition_transfer_design': False, 'Confirmation_signal_opening_requires_separate_post_confirmation_report_transition_transfer_design_execution_phase': True, 'semantic_tests_opening_in_validation_report_transition_synthesis': False, 'prespecified_boundaries': {'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'forbidden': ['ranking K=2/3 by Validation performance', 'selecting a preferred transition K after seeing Validation', 'centroid refit', 'threshold tuning', 'K reselection', 'family merging/splitting/renaming', 'semantic naming', 'Validation behavioral/category/context/stimulus target access', 'Confirmation EEG access in validation_report_transition_synthesis', 'Confirmation behavioral/category/context/stimulus target access', 'claims of distortion/dishonesty/suppression/exaggeration/conscious editing']}
    writej(project / NEXT_CONFIG, next_policy)
    rehash_path = out / 'qc/validation_report_transition_synthesis_source_rehash.csv'
    write_csv(rehash_path, rehash_rows, ['source', 'expected_sha256', 'observed_sha256', 'verified'])
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_report_transition_transfer_synthesis_complete', 'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'all_validation_replicated_K_advance': True, 'single_preferred_K_selected': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'counts_from_validation_report_transition_multiresolution_transfer': summary_l.get('counts', {}), 'paired_fraction_of_494_from_validation_report_transition_multiresolution_transfer': summary_l.get('paired_fraction_of_494'), 'next_phase': NEXT_PHASE, 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/validation_report_transition_synthesis_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'synthesis_only_no_new_data_access', 'source_code_sha256': sha256(Path(__file__).resolve()), 'validation_report_transition_multiresolution_transfer_final_lock_sha256': validation_report_transition_multiresolution_transfer_LOCK_SHA256, 'validation_report_transition_multiresolution_transfer_to_validation_report_transition_synthesis_policy_sha256': validation_report_transition_multiresolution_transfer_NEXT_POLICY_SHA256, 'verified_all_validation_report_transition_multiresolution_transfer_fixed_children': True, 'decision': {'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'rank_or_select_among_passing_K': False}, 'boundary': {'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/validation_report_transition_synthesis_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/validation_report_transition_synthesis_report.md'
    report_path.write_text(''.join(['# validation_report_transition_synthesis — Validation report-transition transfer synthesis', '', '**validation_report_transition_synthesis is synthesis-only and opened no new EEG, behavioral, or Confirmation values.**', '', '## prespecified held-out result', '', '- Discovery-eligible report-transition resolutions tested in Validation: **K=2, K=3**.', '- One-shot Validation transfer: **K=2 PASS, K=3 PASS**.', f"- Validation paired `stimulus_late -> rating_early` trials: **{summary_l['counts']['paired_stimulus_late_rating_early_trials']}/494**.", f"- Validation eligible participants: **{summary_l['counts']['participants_with_at_least_16_pairs']}**.", '', 'Both resolutions advance unchanged to a separate Confirmation transfer design-specification.', 'No preferred K is selected or ranked from Validation performance.', '', '## Interpretation boundary', '', 'This supports held-out reproducibility of report/evaluation-stage EEG reconfiguration structure within this dataset.', 'It does not establish emotion semantics, report truthfulness, distortion, dishonesty, suppression, exaggeration, or conscious editing.', '', 'Behavioral semantic tests remain closed.', 'Confirmation remains unopened in validation_report_transition_synthesis.', '', '## Next', '', f'**{NEXT_PHASE}** — specification the one-shot Confirmation report-transition transfer for K=2/3 without opening Confirmation signal values.', '']) + '', encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, contract_path, summary_path, rehash_path, decision_path, report_path]
    final_lock = out / 'qc/validation_report_transition_synthesis_final_output_lock_hashes.json'
    writej(final_lock, {'phase': PHASE, 'generated_at': now(), 'status': 'validation_report_transition_transfer_synthesis_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'single_preferred_K_selected': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE})
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('validation_report_transition_synthesis PASS')
    print(f'Validation passing transition K: {EXPECTED_PASSING_K}')
    print(f'Advance unchanged to Confirmation design: {EXPECTED_PASSING_K}')
    print('Single preferred K selected: False')
    print('New signal values opened: False')
    print('New behavioral values opened: False')
    print('Confirmation opened: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())