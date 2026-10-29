# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
PHASE = 'confirmation_report_transition_transfer_design'
NEXT_PHASE = 'confirmation_report_transition_multiresolution_transfer'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
ADVANCE_K = [2, 3]
AUTHORITATIVE_CONFIRMATION_TRIALS = 494
EXPECTED_CONFIRMATION_SUBJECTS = 12
SOURCE_CELLS_PER_EPOCH = 169
validation_report_transition_synthesis_OUT = Path('outputs/interpretability/validation_report_transition_synthesis_validation_report_transition_synthesis')
validation_report_transition_synthesis_LOCK = validation_report_transition_synthesis_OUT / 'qc/validation_report_transition_synthesis_final_output_lock_hashes.json'
validation_report_transition_synthesis_LOCK_SHA256 = '35b63c9f55fd10d290854fc81a2095f9039dc36f0d71fe55ce21a89d5736000e'
validation_report_transition_synthesis_NEXT = Path('analysis_specifications/S086_validation_report_transition_synthesis_confirmation.json')
validation_report_transition_synthesis_NEXT_SHA256 = '47896ee8781f5536d58e13e1e99cb5f44cfe8443a562b6d49147cb1fb0ba75d1'
validation_report_transition_transfer_design_OUT = Path('outputs/interpretability/validation_report_transition_transfer_design_validation_report_transition_transfer_design_specification')
validation_report_transition_transfer_design_LOCK = validation_report_transition_transfer_design_OUT / 'qc/validation_report_transition_transfer_design_final_output_lock_hashes.json'
validation_report_transition_transfer_design_LOCK_SHA256 = 'a2c35c1aa182bbe8c34d8f01d36ec46d0cb413948392cef689ef967cd8c0529d'
validation_report_transition_transfer_design_POLICY = Path('analysis_specifications/S087_validation_report_transition_transfer.json')
validation_report_transition_transfer_design_POLICY_SHA256 = 'c5a800fa20793cc99d14fc3a9fe31cb68aa6bc4063b715c1aa7f33ea24492002'
validation_report_transition_transfer_design_CENTROIDS = validation_report_transition_transfer_design_OUT / 'tables/validation_report_transition_transfer_design_prespecified_centroids.csv'
validation_report_transition_transfer_design_FALLBACK = validation_report_transition_transfer_design_OUT / 'tables/validation_report_transition_transfer_design_discovery_fallback_iqr.csv'
validation_report_transition_transfer_design_SUPPORT = validation_report_transition_transfer_design_OUT / 'tables/validation_report_transition_transfer_design_support_thresholds.csv'
validation_report_transition_transfer_design_GATE = validation_report_transition_transfer_design_OUT / 'tables/validation_report_transition_transfer_design_validation_gate_spec.csv'
validation_report_transition_transfer_design_CENTROIDS_SHA256 = 'e1d6baa3013b2b5b90783d9768d067b774cc91b48299392347646d3df1651066'
validation_report_transition_transfer_design_FALLBACK_SHA256 = '05f77dad2174c8a1fe99543b8696541181480aad9cee3b7387a96ea61f977422'
validation_report_transition_transfer_design_SUPPORT_SHA256 = '696cfb664c571bce153b2090a02eae97ebfd4b978e7c8945454931aba390112d'
validation_report_transition_transfer_design_GATE_SHA256 = '967a5005b99c1e114ce379f7671286f624c0341361864151bbecae8bb70ed191'
confirmation_feature_trajectory_LOCK = Path('outputs/qc/confirmation_feature_trajectory_final_output_lock_hashes.json')
confirmation_feature_trajectory_LOCK_SHA256 = '0aae3cc2f447f8383c9ec5656c9a5ba9b4299316f3a19248f34b43850b8246df'
confirmation_feature_trajectory_SUMMARY = Path('outputs/qc/confirmation_feature_trajectory_confirmation_feature_trajectory_summary.json')
confirmation_feature_trajectory_SUMMARY_SHA256 = 'fa04f35c292881357e3adaa7f01c292e57e70c40de6b26df7e983f091f463408'
confirmation_feature_trajectory_INVENTORY = Path('outputs/tables/confirmation_feature_trajectory_confirmation_compact_store_inventory.csv')
confirmation_feature_trajectory_INVENTORY_SHA256 = 'a60eae0ca4cd586bcb08de32e53674c72960103e3441c4e991c1e89c5bdc156d'
CONFIRMATION_ROOT = Path('outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1')
CONFIRMATION_COMPACT = CONFIRMATION_ROOT / 'global/epoch_compact_float32.npy'
CONFIRMATION_MANIFEST = CONFIRMATION_ROOT / 'global/epoch_manifest.csv'
CONFIRMATION_COMPACT_SHA256 = '734fa3d2581d95adc172ea854d4f1fccc2432eaeb167960fdf1e76da0c1509d3'
CONFIRMATION_MANIFEST_SHA256 = 'ccd83881685da351d79c8de5d199b650d17ed0f6e899c332a7c6d731f7becbd2'
OUT = Path('outputs/interpretability/confirmation_report_transition_transfer_design_confirmation_report_transition_transfer_design_specification')
prespecified_POLICY = Path('analysis_specifications/S006_confirmation_report_transition_transfer.json')

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

def copy_exact(src: Path, dst: Path, expected_sha: str, label: str) -> str:
    verify(src, expected_sha, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    got = sha256(dst)
    if got != expected_sha:
        raise RuntimeError(f'Byte-identical carry-forward failed for {label}')
    return got

def verify_lock_children(project: Path, lock_rel: Path, expected_lock_sha: str, expected_phase: str) -> list[dict[str, str]]:
    lock_path = project / lock_rel
    verify(lock_path, expected_lock_sha, f'{expected_phase} final lock')
    lock = readj(lock_path)
    if lock.get('phase') != expected_phase:
        raise RuntimeError(f'{expected_phase} final-lock phase drift')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{expected_phase} final lock missing child hashes')
    rows: list[dict[str, str]] = []
    for rel, expected in sorted(hashes.items()):
        p = project / rel
        got = verify(p, expected, f'{expected_phase} fixed child {rel}')
        rows.append({'source_phase': expected_phase, 'path': rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    return rows

def validate_validation_report_transition_synthesis(project: Path) -> dict[str, Any]:
    verify(project / validation_report_transition_synthesis_NEXT, validation_report_transition_synthesis_NEXT_SHA256, 'validation_report_transition_synthesis→confirmation_report_transition_transfer_design policy')
    policy = readj(project / validation_report_transition_synthesis_NEXT)
    if policy.get('phase') != 'validation_report_transition_synthesis':
        raise RuntimeError('validation_report_transition_synthesis→confirmation_report_transition_transfer_design policy phase drift')
    if policy.get('next_phase') != PHASE or not bool(policy.get('next_phase_permitted')):
        raise RuntimeError('validation_report_transition_synthesis does not authorize confirmation_report_transition_transfer_design')
    if policy.get('advance_K') != ADVANCE_K:
        raise RuntimeError(f"validation_report_transition_synthesis advance_K drift: {policy.get('advance_K')}")
    if bool(policy.get('Confirmation_signal_opening_in_confirmation_report_transition_transfer_design')):
        raise RuntimeError('validation_report_transition_synthesis policy unexpectedly permits Confirmation signal in confirmation_report_transition_transfer_design')
    return policy

def validate_validation_report_transition_transfer_design_policy(project: Path) -> dict[str, Any]:
    verify(project / validation_report_transition_transfer_design_POLICY, validation_report_transition_transfer_design_POLICY_SHA256, 'validation_report_transition_transfer_design prespecified transfer policy')
    p = readj(project / validation_report_transition_transfer_design_POLICY)
    if p.get('eligible_K') != ADVANCE_K:
        raise RuntimeError(f"validation_report_transition_transfer_design eligible_K drift: {p.get('eligible_K')}")
    if p.get('representation', {}).get('stages') != ['stimulus_late', 'rating_early']:
        raise RuntimeError('validation_report_transition_transfer_design transition stages drift')
    if int(p.get('representation', {}).get('source_cells_per_epoch', -1)) != 169:
        raise RuntimeError('validation_report_transition_transfer_design compact width drift')
    if int(p.get('participant_eligibility', {}).get('minimum_paired_stimulus_late_rating_early_trials', -1)) != 16:
        raise RuntimeError('validation_report_transition_transfer_design participant minimum-pair rule drift')
    g = p.get('structural_transfer_gate', {})
    expected = {'minimum_eligible_participants': 9, 'minimum_paired_report_transition_fraction_of_authoritative_validation_trials': 0.75, 'minimum_participant_balanced_in_support_fraction': 0.8, 'minimum_distinct_participants_per_family': 5, 'minimum_trials_per_family': 30, 'minimum_participant_balanced_occupancy_per_family': 0.05}
    for key, val in expected.items():
        if float(g.get(key, float('nan'))) != float(val):
            raise RuntimeError(f'validation_report_transition_transfer_design gate drift: {key}={g.get(key)}')
    return p

def validate_confirmation_feature_trajectory_metadata(project: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    got = verify(project / confirmation_feature_trajectory_LOCK, confirmation_feature_trajectory_LOCK_SHA256, 'historical confirmation_feature_trajectory final lock')
    rows.append({'source_phase': 'confirmation_feature_trajectory', 'path': str(confirmation_feature_trajectory_LOCK), 'expected_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'observed_sha256': got, 'verified': 'True'})
    lock = readj(project / confirmation_feature_trajectory_LOCK)
    if lock.get('phase') != 'confirmation_feature_trajectory':
        raise RuntimeError('confirmation_feature_trajectory final-lock phase drift')
    if lock.get('hashes', {}).get('summary') != confirmation_feature_trajectory_SUMMARY_SHA256:
        raise RuntimeError('confirmation_feature_trajectory summary hash reference drift')
    if lock.get('hashes', {}).get('global_store_inventory') != confirmation_feature_trajectory_INVENTORY_SHA256:
        raise RuntimeError('confirmation_feature_trajectory inventory hash reference drift')
    if lock.get('hashes', {}).get('epoch_manifest') != CONFIRMATION_MANIFEST_SHA256:
        raise RuntimeError('confirmation_feature_trajectory epoch-manifest hash reference drift')
    for rel, expected, label in [(confirmation_feature_trajectory_SUMMARY, confirmation_feature_trajectory_SUMMARY_SHA256, 'confirmation_feature_trajectory summary'), (confirmation_feature_trajectory_INVENTORY, confirmation_feature_trajectory_INVENTORY_SHA256, 'confirmation_feature_trajectory compact-store inventory')]:
        got = verify(project / rel, expected, label)
        rows.append({'source_phase': 'confirmation_feature_trajectory', 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    summary = readj(project / confirmation_feature_trajectory_SUMMARY)
    counts = summary.get('counts', {})
    if int(counts.get('confirmation_subjects', -1)) != EXPECTED_CONFIRMATION_SUBJECTS:
        raise RuntimeError('confirmation_feature_trajectory Confirmation subject-count drift')
    if int(counts.get('authoritative_trials', -1)) != AUTHORITATIVE_CONFIRMATION_TRIALS:
        raise RuntimeError('confirmation_feature_trajectory authoritative Confirmation trial-count drift')
    if int(counts.get('compact_cells_per_epoch', -1)) != SOURCE_CELLS_PER_EPOCH:
        raise RuntimeError('confirmation_feature_trajectory compact feature-width drift')
    inventory = read_csv(project / confirmation_feature_trajectory_INVENTORY)
    by_rel = {r['relative_path']: r for r in inventory}
    expected_files = {'global/epoch_compact_float32.npy': CONFIRMATION_COMPACT_SHA256, 'global/epoch_manifest.csv': CONFIRMATION_MANIFEST_SHA256}
    for rel, expected in expected_files.items():
        if rel not in by_rel:
            raise RuntimeError(f'confirmation_feature_trajectory inventory missing {rel}')
        if by_rel[rel]['sha256'] != expected:
            raise RuntimeError(f"confirmation_feature_trajectory inventory hash drift for {rel}: {by_rel[rel]['sha256']} != {expected}")
    return rows

def self_test() -> int:
    assert ADVANCE_K == [2, 3]
    assert AUTHORITATIVE_CONFIRMATION_TRIALS == 494
    assert EXPECTED_CONFIRMATION_SUBJECTS == 12
    assert SOURCE_CELLS_PER_EPOCH == 169
    assert PHASE == 'confirmation_report_transition_transfer_design'
    assert NEXT_PHASE == 'confirmation_report_transition_multiresolution_transfer'
    print('confirmation_report_transition_transfer_design self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    out = project / OUT
    final_lock = out / 'qc/confirmation_report_transition_transfer_design_final_output_lock_hashes.json'
    if final_lock.exists() and (not args.force):
        raise RuntimeError(f'confirmation_report_transition_transfer_design already materialized: {final_lock}. Use --force only to rematerialize the identical design.')
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] verifying validation_report_transition_synthesis Validation synthesis', flush=True)
    rehash_rows = verify_lock_children(project, validation_report_transition_synthesis_LOCK, validation_report_transition_synthesis_LOCK_SHA256, 'validation_report_transition_synthesis')
    validation_report_transition_synthesis_policy = validate_validation_report_transition_synthesis(project)
    print(f'[{now()}] verifying unchanged Discovery-calibrated validation_report_transition_transfer_design templates/gates', flush=True)
    rehash_rows += verify_lock_children(project, validation_report_transition_transfer_design_LOCK, validation_report_transition_transfer_design_LOCK_SHA256, 'validation_report_transition_transfer_design')
    validation_report_transition_transfer_design_policy = validate_validation_report_transition_transfer_design_policy(project)
    print(f'[{now()}] verifying historical confirmation_feature_trajectory Confirmation store metadata only', flush=True)
    rehash_rows += validate_confirmation_feature_trajectory_metadata(project)
    prespecified_centroids = out / 'tables/confirmation_report_transition_transfer_design_prespecified_centroids.csv'
    prespecified_fallback = out / 'tables/confirmation_report_transition_transfer_design_discovery_fallback_iqr.csv'
    prespecified_support = out / 'tables/confirmation_report_transition_transfer_design_support_thresholds.csv'
    copy_exact(project / validation_report_transition_transfer_design_CENTROIDS, prespecified_centroids, validation_report_transition_transfer_design_CENTROIDS_SHA256, 'validation_report_transition_transfer_design centroids')
    copy_exact(project / validation_report_transition_transfer_design_FALLBACK, prespecified_fallback, validation_report_transition_transfer_design_FALLBACK_SHA256, 'validation_report_transition_transfer_design fallback IQR')
    copy_exact(project / validation_report_transition_transfer_design_SUPPORT, prespecified_support, validation_report_transition_transfer_design_SUPPORT_SHA256, 'validation_report_transition_transfer_design support thresholds')
    gate_rows = read_csv(project / validation_report_transition_transfer_design_GATE)
    if len(gate_rows) != 16:
        raise RuntimeError(f'Unexpected validation_report_transition_transfer_design gate row count: {len(gate_rows)}')
    confirmation_gate_rows: list[dict[str, Any]] = []
    for r in gate_rows:
        k = int(r['k'])
        if k not in ADVANCE_K:
            raise RuntimeError(f'Unexpected K in validation_report_transition_transfer_design gate spec: {k}')
        metric = r['metric'].replace('Validation', 'Confirmation')
        metric = metric.replace('paired report-transition trials / authoritative Confirmation trials', 'paired report-transition trials / authoritative Confirmation trials')
        confirmation_gate_rows.append({'k': k, 'gate_id': r['gate_id'], 'metric': metric, 'comparison': r['comparison'], 'threshold': r['threshold']})
    confirmation_gate = out / 'tables/confirmation_report_transition_transfer_design_confirmation_gate_spec.csv'
    write_csv(confirmation_gate, confirmation_gate_rows, ['k', 'gate_id', 'metric', 'comparison', 'threshold'])
    store_reference = out / 'tables/confirmation_report_transition_transfer_design_confirmation_store_reference.csv'
    write_csv(store_reference, [{'role': 'compact', 'path': str(CONFIRMATION_COMPACT), 'sha256': CONFIRMATION_COMPACT_SHA256, 'values_opened_in_confirmation_report_transition_transfer_design': False}, {'role': 'manifest', 'path': str(CONFIRMATION_MANIFEST), 'sha256': CONFIRMATION_MANIFEST_SHA256, 'values_opened_in_confirmation_report_transition_transfer_design': False}, {'role': 'historical_final_lock', 'path': str(confirmation_feature_trajectory_LOCK), 'sha256': confirmation_feature_trajectory_LOCK_SHA256, 'values_opened_in_confirmation_report_transition_transfer_design': False}], ['role', 'path', 'sha256', 'values_opened_in_confirmation_report_transition_transfer_design'])
    policy = {'phase': PHASE, 'generated_at': now(), 'scientific_role': 'specification one-shot Confirmation transfer of Validation-replicated stimulus_late->rating_early report-transition families before Confirmation compact EEG values are opened by this branch', 'next_phase': NEXT_PHASE, 'advance_K': ADVANCE_K, 'all_advance_K_tested_independently': True, 'no_resolution_promoted_or_ranked': True, 'fixed_before_confirmation_signal_access_in_this_branch': True, 'representation': {'source_cells_per_epoch': SOURCE_CELLS_PER_EPOCH, 'stages': ['stimulus_late', 'rating_early'], 'signature_cells': 14, 'coordinate_order': validation_report_transition_transfer_design_policy['representation']['coordinate_order'], 'preprocessing': validation_report_transition_transfer_design_policy['representation']['validation_signature_preprocessing'], 'prespecified_discovery_fallback_iqr': validation_report_transition_transfer_design_policy['representation']['prespecified_discovery_fallback_iqr']}, 'fixed_assignment': validation_report_transition_transfer_design_policy['fixed_assignment'], 'participant_eligibility': validation_report_transition_transfer_design_policy['participant_eligibility'], 'structural_transfer_gate': {'evaluated_separately_for_each_K': True, 'K_replication_rule': 'all gates for that K must pass', 'all_source_hash_gates_must_pass': True, 'minimum_eligible_participants': 9, 'minimum_paired_report_transition_fraction_of_authoritative_confirmation_trials': 0.75, 'minimum_participant_balanced_in_support_fraction': 0.8, 'minimum_distinct_participants_per_family': 5, 'minimum_trials_per_family': 30, 'minimum_participant_balanced_occupancy_per_family': 0.05, 'no_behavior_category_context_stimulus_target_mapping': True, 'failure_rule': 'failed K is not retuned, renamed, merged, split, rescued, or reselected'}, 'confirmation_store': {'root': str(CONFIRMATION_ROOT), 'compact': str(CONFIRMATION_COMPACT), 'compact_sha256': CONFIRMATION_COMPACT_SHA256, 'manifest': str(CONFIRMATION_MANIFEST), 'manifest_sha256': CONFIRMATION_MANIFEST_SHA256, 'historical_confirmation_feature_trajectory_final_lock_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'historical_confirmation_feature_trajectory_summary_sha256': confirmation_feature_trajectory_SUMMARY_SHA256, 'historical_confirmation_feature_trajectory_inventory_sha256': confirmation_feature_trajectory_INVENTORY_SHA256, 'authoritative_confirmation_trials': AUTHORITATIVE_CONFIRMATION_TRIALS, 'expected_confirmation_subjects': EXPECTED_CONFIRMATION_SUBJECTS}, 'confirmation_report_transition_transfer_design_data_boundary': {'Confirmation_compact_values_read': False, 'Confirmation_manifest_parsed': False, 'Confirmation_behavioral_values_read': False, 'Confirmation_emotion_category_values_read': False, 'Confirmation_context_target_values_read': False, 'Confirmation_stimulus_identity_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'forbidden': ['Confirmation behavioral reports', 'emotion category as target', 'context as target', 'stimulus identity as target', 'centroid refit', 'threshold tuning', 'K reselection', 'family merging/splitting/renaming', 'ranking K=2/3 by Validation or Confirmation performance', 'semantic naming', 'claims of distortion/dishonesty/suppression/exaggeration/conscious editing'], 'post_confirmation_rule': {'confirmation_report_transition_multiresolution_transfer_role': 'one-shot structural transfer only; open the prespecified Confirmation compact store and manifest, build stimulus_late->rating_early signatures, and apply K=2/3 unchanged', 'semantic_tests_after_confirmation_report_transition_multiresolution_transfer': 'remain closed until the Confirmation structural result is fixed and a separate target-opening execution phase is authorized'}}
    writej(project / prespecified_POLICY, policy)
    source_rehash = out / 'qc/confirmation_report_transition_transfer_design_source_rehash.csv'
    write_csv(source_rehash, rehash_rows, ['source_phase', 'path', 'expected_sha256', 'observed_sha256', 'verified'])
    carry = out / 'tables/confirmation_report_transition_transfer_design_exact_carry_forward_hashes.csv'
    write_csv(carry, [{'artifact': 'prespecified_centroids', 'source_path': str(validation_report_transition_transfer_design_CENTROIDS), 'source_sha256': validation_report_transition_transfer_design_CENTROIDS_SHA256, 'confirmation_report_transition_transfer_design_path': str(prespecified_centroids.relative_to(project)), 'confirmation_report_transition_transfer_design_sha256': sha256(prespecified_centroids), 'byte_identical': True}, {'artifact': 'discovery_fallback_iqr', 'source_path': str(validation_report_transition_transfer_design_FALLBACK), 'source_sha256': validation_report_transition_transfer_design_FALLBACK_SHA256, 'confirmation_report_transition_transfer_design_path': str(prespecified_fallback.relative_to(project)), 'confirmation_report_transition_transfer_design_sha256': sha256(prespecified_fallback), 'byte_identical': True}, {'artifact': 'support_thresholds', 'source_path': str(validation_report_transition_transfer_design_SUPPORT), 'source_sha256': validation_report_transition_transfer_design_SUPPORT_SHA256, 'confirmation_report_transition_transfer_design_path': str(prespecified_support.relative_to(project)), 'confirmation_report_transition_transfer_design_sha256': sha256(prespecified_support), 'byte_identical': True}], ['artifact', 'source_path', 'source_sha256', 'confirmation_report_transition_transfer_design_path', 'confirmation_report_transition_transfer_design_sha256', 'byte_identical'])
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'confirmation_report_transition_transfer_design_prespecified', 'advance_K': ADVANCE_K, 'authoritative_confirmation_trials': AUTHORITATIVE_CONFIRMATION_TRIALS, 'expected_confirmation_subjects': EXPECTED_CONFIRMATION_SUBJECTS, 'Confirmation_compact_values_read': False, 'Confirmation_manifest_parsed': False, 'Confirmation_behavioral_values_read': False, 'Confirmation_emotion_category_values_read': False, 'centroids_refit': False, 'thresholds_tuned': False, 'K_reselected': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_role': 'one-shot Confirmation stimulus_late-to-rating_early structural transfer for K=2/3 using unchanged validation_report_transition_transfer_design templates/support/gates', 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/confirmation_report_transition_transfer_design_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'design_prespecified', 'source_code_sha256': sha256(Path(__file__).resolve()), 'validation_report_transition_synthesis_final_lock_sha256': validation_report_transition_synthesis_LOCK_SHA256, 'validation_report_transition_synthesis_to_confirmation_report_transition_transfer_design_policy_sha256': validation_report_transition_synthesis_NEXT_SHA256, 'validation_report_transition_transfer_design_final_lock_sha256': validation_report_transition_transfer_design_LOCK_SHA256, 'validation_report_transition_transfer_design_policy_sha256': validation_report_transition_transfer_design_POLICY_SHA256, 'confirmation_feature_trajectory_historical_final_lock_sha256': confirmation_feature_trajectory_LOCK_SHA256, 'Confirmation_compact_sha256_prespecified_not_opened': CONFIRMATION_COMPACT_SHA256, 'Confirmation_manifest_sha256_prespecified_not_parsed': CONFIRMATION_MANIFEST_SHA256, 'advance_K': ADVANCE_K, 'boundary': {'Confirmation_compact_values_read': False, 'Confirmation_manifest_parsed': False, 'Confirmation_behavioral_values_read': False, 'Confirmation_emotion_category_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/confirmation_report_transition_transfer_design_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/confirmation_report_transition_transfer_design_report.md'
    support_rows = read_csv(prespecified_support)
    support_text = ''.join((f"- K={r['k']}: distance ceiling={float(r['in_support_squared_distance_max']):.6f}; low-margin threshold={float(r['low_margin_threshold']):.6f}; Discovery LOPO participant-balanced agreement={float(r['lopo_participant_balanced_agreement']):.6f}" for r in support_rows))
    report_path.write_text(''.join(['# confirmation_report_transition_transfer_design — Confirmation report-transition transfer analysis specification', '', '**No Confirmation compact EEG values were opened and the Confirmation manifest was not parsed in confirmation_report_transition_transfer_design.**', '', '## prespecified resolutions', '', '- K=2', '- K=3', '', 'Both replicated in one-shot Validation and both advance unchanged. No winner is selected.', '', '## Exact Discovery-calibrated support carried forward', '', support_text, '', 'The validation_report_transition_transfer_design centroids, Discovery fallback IQR and support thresholds are copied byte-for-byte.', '', '## prespecified Confirmation source', '', f'- authoritative trials: **{AUTHORITATIVE_CONFIRMATION_TRIALS}**', f'- expected participants: **{EXPECTED_CONFIRMATION_SUBJECTS}**', f'- compact store: `{CONFIRMATION_COMPACT}`', f'- compact SHA-256: `{CONFIRMATION_COMPACT_SHA256}`', f'- manifest: `{CONFIRMATION_MANIFEST}`', f'- manifest SHA-256: `{CONFIRMATION_MANIFEST_SHA256}`', '', 'confirmation_report_transition_transfer_design verifies only historical confirmation_feature_trajectory metadata/locks for these files. The actual Confirmation compact values and manifest are first opened by confirmation_report_transition_multiresolution_transfer.', '', '## prespecified structural gate per K', '', '- eligible participants >= 9', '- paired `stimulus_late -> rating_early` fraction >= 0.75 of 494', '- participant-balanced in-support >= 0.80', '- every family present in >= 5 participants', '- every family has >= 30 trials', '- every family participant-balanced occupancy >= 0.05', '- no behavioral/category/context/stimulus target mapping', '', '## Interpretation boundary', '', 'A later Confirmation PASS would support reproducibility of report/evaluation-stage EEG reconfiguration structure only. It would not establish emotion semantics or distortion/dishonesty/suppression/exaggeration/conscious editing.', '', '## Next', '', f'**{NEXT_PHASE}** — one-shot Confirmation report-transition structural transfer for K=2/3.', '']) + '', encoding='utf-8')
    lock_targets = [project / prespecified_POLICY, contract_path, summary_path, source_rehash, carry, prespecified_centroids, prespecified_fallback, prespecified_support, confirmation_gate, store_reference, report_path]
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'confirmation_report_transition_transfer_design_prespecified', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'advance_K': ADVANCE_K, 'Confirmation_compact_values_read': False, 'Confirmation_manifest_parsed': False, 'Confirmation_behavioral_values_read': False, 'Confirmation_emotion_category_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final_obj)
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('confirmation_report_transition_transfer_design PASS')
    print(f'prespecified Confirmation transition K: {ADVANCE_K}')
    print(f'Authoritative Confirmation trials: {AUTHORITATIVE_CONFIRMATION_TRIALS}')
    print(f'Expected Confirmation participants: {EXPECTED_CONFIRMATION_SUBJECTS}')
    print('Confirmation compact values opened: False')
    print('Confirmation manifest parsed: False')
    print('Confirmation behavioral values opened: False')
    print('Centroids refit: False')
    print('Thresholds tuned: False')
    print('K reselected: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())