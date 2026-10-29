# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
PHASE = 'validation_experience_transfer_synthesis'
NEXT_PHASE = 'confirmation_experience_transfer_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
validation_stimulus_experience_multiresolution_transfer_OUT = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer')
validation_stimulus_experience_multiresolution_transfer_LOCK = validation_stimulus_experience_multiresolution_transfer_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256 = '5c8431957ac2fe3ffdc8886724b7145359062f83580463b527937164daa9b0ca'
validation_stimulus_experience_multiresolution_transfer_SUMMARY = validation_stimulus_experience_multiresolution_transfer_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_summary.json'
validation_stimulus_experience_multiresolution_transfer_RESOLUTION = validation_stimulus_experience_multiresolution_transfer_OUT / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_resolution_summary.csv'
validation_stimulus_experience_multiresolution_transfer_GATE = validation_stimulus_experience_multiresolution_transfer_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_validation_structural_transfer_gate_all_k.csv'
validation_stimulus_experience_multiresolution_transfer_REPORT = validation_stimulus_experience_multiresolution_transfer_OUT / 'reports/validation_stimulus_experience_multiresolution_transfer_report.md'
validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY = Path('analysis_specifications/S091_validation_stimulus_experience_multiresolution_transfer.json')
validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY_SHA256 = '16a40051b9d6854ecf279a206dad47a282ca5a5b11b23f1488e1eb185c27c966'
validation_experience_transfer_design_OUT = Path('outputs/interpretability/validation_experience_transfer_design_validation_experience_transfer_design_specification')
validation_experience_transfer_design_CENTROIDS = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_prespecified_centroids.csv'
validation_experience_transfer_design_FALLBACK = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_discovery_fallback_iqr.csv'
validation_experience_transfer_design_THRESHOLDS = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_support_thresholds.csv'
validation_experience_transfer_design_GATE_SPEC = validation_experience_transfer_design_OUT / 'tables/validation_experience_transfer_design_validation_gate_spec.csv'
OUT = Path('outputs/interpretability/validation_experience_transfer_synthesis_validation_experience_transfer_synthesis')
NEXT_CONFIG = Path('analysis_specifications/S078_validation_experience_transfer_synthesis_confirmation.json')
EXPECTED_TESTED_K = [2, 3, 4]
EXPECTED_PASSING_K = [2, 3, 4]

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

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')
    return got

def verify_parent_lock(project: Path) -> list[dict[str, str]]:
    lock_path = project / validation_stimulus_experience_multiresolution_transfer_LOCK
    verify(lock_path, validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer final lock')
    lock = readj(lock_path)
    if lock.get('phase') != 'validation_stimulus_experience_multiresolution_transfer':
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer final-lock phase drift')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer final lock has no child hashes')
    rows: list[dict[str, str]] = []
    for rel, expected in sorted(hashes.items()):
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing validation_stimulus_experience_multiresolution_transfer fixed child: {rel}')
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer fixed-child SHA drift: {rel}; expected {expected}, got {got}')
        rows.append({'source': rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    return rows

def validate_parent_decision(project: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    verify(project / validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY, validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY_SHA256, 'validation_stimulus_experience_multiresolution_transfer→validation_experience_transfer_synthesis prespecified policy')
    policy = readj(project / validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY)
    summary = readj(project / validation_stimulus_experience_multiresolution_transfer_SUMMARY)
    if summary.get('phase') != 'validation_stimulus_experience_multiresolution_transfer':
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer summary phase drift')
    if summary.get('tested_K') != EXPECTED_TESTED_K:
        raise RuntimeError(f"Unexpected validation_stimulus_experience_multiresolution_transfer tested_K: {summary.get('tested_K')}")
    if summary.get('passing_K') != EXPECTED_PASSING_K:
        raise RuntimeError(f"Unexpected validation_stimulus_experience_multiresolution_transfer passing_K: {summary.get('passing_K')}")
    if not bool(summary.get('any_K_replicated')):
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer says no K replicated; validation_experience_transfer_synthesis advance branch is not permitted')
    boundary = summary.get('boundary', {})
    must_be_false = ['Validation_rating_early_values_read', 'Validation_behavioral_values_read', 'Validation_emotion_category_values_read', 'Validation_context_values_read', 'Validation_stimulus_identity_values_read', 'Confirmation_signal_values_read', 'centroids_refit', 'thresholds_tuned', 'K_reselected', 'family_semantic_relabeling_permitted']
    for key in must_be_false:
        if bool(boundary.get(key)):
            raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer boundary violation: {key}=True')
    if not bool(boundary.get('Validation_stimulus_early_values_read')):
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer boundary drift: stimulus_early not marked read')
    if not bool(boundary.get('Validation_stimulus_late_values_read')):
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer boundary drift: stimulus_late not marked read')
    if policy.get('passing_K') != EXPECTED_PASSING_K:
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer→validation_experience_transfer_synthesis policy passing_K drift')
    if not bool(policy.get('confirmation_design_specification_permitted')):
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer→validation_experience_transfer_synthesis policy does not permit Confirmation analysis specification')
    if bool(policy.get('confirmation_signal_opening_in_validation_experience_transfer_synthesis')):
        raise RuntimeError('validation_stimulus_experience_multiresolution_transfer→validation_experience_transfer_synthesis policy incorrectly permits Confirmation signal opening')
    res_rows = read_csv(project / validation_stimulus_experience_multiresolution_transfer_RESOLUTION)
    by_k = {int(r['k']): r for r in res_rows}
    if sorted(by_k) != EXPECTED_TESTED_K:
        raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer resolution table K drift: {sorted(by_k)}')
    for k in EXPECTED_PASSING_K:
        v = str(by_k[k]['replicated']).strip().lower()
        if v not in {'true', '1', 't', 'yes'}:
            raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer resolution table says K={k} did not replicate')
    return (summary, res_rows)

def self_test() -> int:
    assert EXPECTED_TESTED_K == [2, 3, 4]
    assert EXPECTED_PASSING_K == [2, 3, 4]
    assert PHASE == 'validation_experience_transfer_synthesis'
    assert NEXT_PHASE == 'confirmation_experience_transfer_design'
    print('validation_experience_transfer_synthesis self-test PASS')
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
    out.mkdir(parents=True, exist_ok=True)
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'reports').mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] Verifying prespecified validation_stimulus_experience_multiresolution_transfer parent lock and all fixed children', flush=True)
    rehash_rows = verify_parent_lock(project)
    summary_d, res_rows = validate_parent_decision(project)
    decision_rows: list[dict[str, Any]] = []
    for r in sorted(res_rows, key=lambda x: int(x['k'])):
        k = int(r['k'])
        decision_rows.append({'k': k, 'discovery_eligible': True, 'validation_replicated': True, 'advance_unchanged_to_confirmation_design': True, 'validation_eligible_participants': int(float(r['eligible_participants'])), 'validation_paired_trials': int(float(r['paired_experience_trials_all_participants'])), 'validation_participant_balanced_in_support_fraction': float(r['participant_balanced_in_support_fraction']), 'validation_participant_balanced_low_margin_fraction': float(r['participant_balanced_low_margin_fraction']), 'semantic_name_permitted': False, 'rank_or_select_among_passing_k': False})
    decision_path = out / 'tables/validation_experience_transfer_synthesis_confirmation_advancement_decision.csv'
    write_csv(decision_path, decision_rows, ['k', 'discovery_eligible', 'validation_replicated', 'advance_unchanged_to_confirmation_design', 'validation_eligible_participants', 'validation_paired_trials', 'validation_participant_balanced_in_support_fraction', 'validation_participant_balanced_low_margin_fraction', 'semantic_name_permitted', 'rank_or_select_among_passing_k'])
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'specification the one-shot Confirmation stimulus-only transfer for K=2,3,4 without opening Confirmation signal values', 'advance_K': EXPECTED_PASSING_K, 'advance_rule': 'all and only Validation-replicated K advance unchanged; no ranking, winner selection, K reselection, centroid refit, threshold tuning, or family renaming', 'confirmation_signal_opening_in_validation_experience_transfer_synthesis': False, 'confirmation_signal_opening_in_confirmation_experience_transfer_design': False, 'confirmation_signal_opening_requires_separate_post_confirmation_experience_transfer_design_execution_phase': True, 'reuse_unchanged': [str(validation_experience_transfer_design_CENTROIDS), str(validation_experience_transfer_design_FALLBACK), str(validation_experience_transfer_design_THRESHOLDS), str(validation_experience_transfer_design_GATE_SPEC)], 'prespecified_boundaries': {'Validation_behavioral_values_read': False, 'Validation_rating_early_values_read': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'semantic_relabeling_permitted': False}, 'forbidden': ['ranking K=2/3/4 by Validation performance', 'selecting a single preferred K after seeing Validation', 'centroid refit', 'threshold tuning', 'K reselection', 'family relabeling', 'semantic naming', 'Validation behavioral/category/context/stimulus target access', 'Validation rating_early EEG access', 'Confirmation EEG access in validation_experience_transfer_synthesis', 'Confirmation behavioral/category/context/stimulus target access']}
    writej(project / NEXT_CONFIG, next_policy)
    source_rehash_path = out / 'qc/validation_experience_transfer_synthesis_source_rehash.csv'
    write_csv(source_rehash_path, rehash_rows, ['source', 'expected_sha256', 'observed_sha256', 'verified'])
    summary_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_experience_transfer_synthesis_complete', 'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'all_validation_replicated_K_advance': True, 'single_preferred_K_selected': False, 'new_EEG_feature_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'Validation_rating_early_values_read': False, 'semantic_relabeling_permitted': False, 'counts_from_validation_stimulus_experience_multiresolution_transfer': summary_d.get('counts', {}), 'paired_fraction_of_494_from_validation_stimulus_experience_multiresolution_transfer': summary_d.get('paired_fraction_of_494'), 'next_phase': NEXT_PHASE, 'next_phase_role': next_policy['next_phase_role'], 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/validation_experience_transfer_synthesis_summary.json'
    writej(summary_path, summary_obj)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'synthesis_only_no_new_data_access', 'source_code_sha256': sha256(Path(__file__).resolve()), 'validation_stimulus_experience_multiresolution_transfer_final_lock_sha256': validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer_to_validation_experience_transfer_synthesis_policy_sha256': validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY_SHA256, 'verified_all_validation_stimulus_experience_multiresolution_transfer_fixed_children': True, 'decision': {'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'rank_or_select_among_passing_K': False}, 'boundary': {'new_EEG_feature_values_opened': False, 'new_behavioral_values_opened': False, 'Validation_rating_early_values_read': False, 'Confirmation_signal_values_read': False, 'Confirmation_behavioral_values_read': False, 'semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/validation_experience_transfer_synthesis_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/validation_experience_transfer_synthesis_report.md'
    lines = ['# validation_experience_transfer_synthesis — Validation stimulus-experience transfer synthesis', '', '**This phase is synthesis-only and opened no new EEG, behavioral, rating, or Confirmation values.**', '', '## prespecified result carried forward from validation_stimulus_experience_multiresolution_transfer', '', '- Discovery-eligible resolutions tested in Validation: **K=2, K=3, K=4**', '- Validation-replicated resolutions: **K=2, K=3, K=4**', '- All three advance unchanged to a separate Confirmation transfer design-specification.', '- No single preferred K is selected or ranked from Validation performance.', '', '## Why all three advance', '', 'Each resolution passed its independently prespecified validation_experience_transfer_design structural-transfer gate in the one-shot validation_stimulus_experience_multiresolution_transfer Validation execution.', 'The purpose of Confirmation is therefore to test the same prespecified partitions, not to choose a winner retrospectively.', '', '## Interpretation boundary', '', '- Replication supports only a stimulus-only reconfiguration partition.', '- It does not identify emotion semantics or psychological names for any family.', '- `rating_early` and behavioral/category/context/stimulus targets remain closed.', '- Confirmation remains unopened in validation_experience_transfer_synthesis.', '- No centroids, thresholds, K values, or family identities were changed.', '', f'## Next', '', f'**{NEXT_PHASE}** — specification the one-shot Confirmation transfer for K=2/3/4 without opening Confirmation signal values.', '']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(''.join(lines), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, contract_path, summary_path, source_rehash_path, decision_path, report_path]
    final_lock = out / 'qc/validation_experience_transfer_synthesis_final_output_lock_hashes.json'
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_experience_transfer_synthesis_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'tested_K': EXPECTED_TESTED_K, 'validation_passing_K': EXPECTED_PASSING_K, 'advance_K_to_confirmation_design': EXPECTED_PASSING_K, 'single_preferred_K_selected': False, 'new_EEG_feature_values_opened': False, 'new_behavioral_values_opened': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final_obj)
    script_dir = Path(__file__).resolve().parent
    files = lock_targets + [final_lock]
    for p in sorted(script_dir.iterdir()):
        if p.is_file():
            files.append(p)
    files += [project / validation_stimulus_experience_multiresolution_transfer_LOCK, project / validation_stimulus_experience_multiresolution_transfer_SUMMARY, project / validation_stimulus_experience_multiresolution_transfer_RESOLUTION, project / validation_stimulus_experience_multiresolution_transfer_GATE, project / validation_stimulus_experience_multiresolution_transfer_REPORT, project / validation_stimulus_experience_multiresolution_transfer_NEXT_POLICY]
    print('')
    print('validation_experience_transfer_synthesis PASS')
    print(f'Validation passing K: {EXPECTED_PASSING_K}')
    print(f'Advance unchanged to Confirmation design: {EXPECTED_PASSING_K}')
    print('Single preferred K selected: False')
    print('New EEG feature values opened: False')
    print('New behavioral values opened: False')
    print('Confirmation opened: False')
    print('Semantic relabeling permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())