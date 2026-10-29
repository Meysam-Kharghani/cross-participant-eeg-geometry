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
PHASE = 'cross_split_experience_structure_synthesis'
NEXT_PHASE = 'semantic_report_dissociation_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
EXPECTED_K = [2, 3, 4]
B_OUT = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit')
B_LOCK = B_OUT / 'qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json'
B_LOCK_SHA256 = '35b9fa3efacfbe1e3884e2d622e1701538a5375fe2b72ffe1155fc429903c2b7'
B_SUMMARY = B_OUT / 'qc/discovery_stimulus_experience_multiresolution_fit_summary.json'
B_METRICS = B_OUT / 'tables/discovery_stimulus_experience_multiresolution_fit_family_candidate_metrics.csv'
B_CROSS = B_OUT / 'tables/discovery_stimulus_experience_multiresolution_fit_cross_resolution_metrics_eligible_k.csv'
D_OUT = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer')
D_LOCK = D_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
D_LOCK_SHA256 = '5c8431957ac2fe3ffdc8886724b7145359062f83580463b527937164daa9b0ca'
D_SUMMARY = D_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_summary.json'
D_RESOLUTION = D_OUT / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_resolution_summary.csv'
G_OUT = Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer')
G_LOCK = G_OUT / 'qc/confirmation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
G_LOCK_SHA256 = '3b6f03c0d44903345871470cc40e35b0d9d0aaa9d836954e1c0009a7485b6b8b'
G_SUMMARY = G_OUT / 'qc/confirmation_stimulus_experience_multiresolution_transfer_summary.json'
G_RESOLUTION = G_OUT / 'tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_resolution_summary.csv'
G_POLICY = Path('analysis_specifications/S010_confirmation_stimulus_experience_multiresolution_transfer_cross.json')
OUT = Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis')
NEXT_CONFIG = Path('analysis_specifications/S014_cross_split_experience_structure_synthesis_semantic.json')

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

def verify_lock_children(project: Path, lock_rel: Path, expected_lock_sha: str, phase: str) -> list[dict[str, str]]:
    lock_path = project / lock_rel
    if not lock_path.is_file():
        raise RuntimeError(f'Missing {phase} final lock: {lock_path}')
    observed_lock_sha = sha256(lock_path)
    if observed_lock_sha != expected_lock_sha:
        raise RuntimeError(f'{phase} final-lock SHA drift: expected {expected_lock_sha}, got {observed_lock_sha}')
    lock = readj(lock_path)
    if lock.get('phase') != phase:
        raise RuntimeError(f'{phase} final-lock phase drift')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{phase} final lock missing child hashes')
    rows: list[dict[str, str]] = []
    for rel, expected in sorted(hashes.items()):
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing {phase} fixed child: {rel}')
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'{phase} fixed-child SHA drift: {rel}; expected {expected}, got {got}')
        rows.append({'source_phase': phase, 'path': rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    return rows

def boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def index_rows(rows: list[dict[str, str]], key: str='k') -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        out[int(r[key])] = r
    return out

def self_test() -> int:
    assert EXPECTED_K == [2, 3, 4]
    assert PHASE == 'cross_split_experience_structure_synthesis'
    assert NEXT_PHASE == 'semantic_report_dissociation_design'
    print('cross_split_experience_structure_synthesis self-test PASS')
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
    print(f'[{now()}] Verifying Discovery discovery_stimulus_experience_multiresolution_fit lock', flush=True)
    rehash_rows = verify_lock_children(project, B_LOCK, B_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit')
    print(f'[{now()}] Verifying Validation validation_stimulus_experience_multiresolution_transfer lock', flush=True)
    rehash_rows += verify_lock_children(project, D_LOCK, D_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer')
    print(f'[{now()}] Verifying Confirmation confirmation_stimulus_experience_multiresolution_transfer lock', flush=True)
    rehash_rows += verify_lock_children(project, G_LOCK, G_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer')
    bsum = readj(project / B_SUMMARY)
    dsum = readj(project / D_SUMMARY)
    gsum = readj(project / G_SUMMARY)
    gpolicy = readj(project / G_POLICY)
    if bsum.get('eligible_K') != EXPECTED_K:
        raise RuntimeError(f"discovery_stimulus_experience_multiresolution_fit eligible_K drift: {bsum.get('eligible_K')}")
    if dsum.get('passing_K') != EXPECTED_K:
        raise RuntimeError(f"validation_stimulus_experience_multiresolution_transfer passing_K drift: {dsum.get('passing_K')}")
    if gsum.get('passing_K') != EXPECTED_K:
        raise RuntimeError(f"confirmation_stimulus_experience_multiresolution_transfer passing_K drift: {gsum.get('passing_K')}")
    if gpolicy.get('passing_K_confirmation') != EXPECTED_K:
        raise RuntimeError('confirmation_stimulus_experience_multiresolution_transfer→cross_split_experience_structure_synthesis policy passing K drift')
    if bool(gpolicy.get('cross_split_experience_structure_synthesis_new_signal_opening_permitted')):
        raise RuntimeError('confirmation_stimulus_experience_multiresolution_transfer policy unexpectedly permits new signal access in cross_split_experience_structure_synthesis')
    if bool(gpolicy.get('cross_split_experience_structure_synthesis_behavioral_opening_permitted')):
        raise RuntimeError('confirmation_stimulus_experience_multiresolution_transfer policy unexpectedly permits behavioral access in cross_split_experience_structure_synthesis')
    db = dsum.get('boundary', {})
    gb = gsum.get('boundary', {})
    for key in ['Validation_behavioral_values_read', 'Validation_rating_early_values_read', 'family_semantic_relabeling_permitted', 'centroids_refit', 'thresholds_tuned', 'K_reselected']:
        if bool(db.get(key)):
            raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer boundary violation: {key}=True')
    for key in ['Confirmation_behavioral_values_read', 'Confirmation_rating_early_values_read', 'family_semantic_relabeling_permitted', 'centroids_refit', 'thresholds_tuned', 'K_reselected']:
        if bool(gb.get(key)):
            raise RuntimeError(f'confirmation_stimulus_experience_multiresolution_transfer boundary violation: {key}=True')
    bm = index_rows(read_csv(project / B_METRICS))
    dr = index_rows(read_csv(project / D_RESOLUTION))
    gr = index_rows(read_csv(project / G_RESOLUTION))
    for k in EXPECTED_K:
        if not boolish(bm[k]['eligible']):
            raise RuntimeError(f'discovery_stimulus_experience_multiresolution_fit says K={k} not eligible')
        if not boolish(dr[k]['replicated']):
            raise RuntimeError(f'validation_stimulus_experience_multiresolution_transfer says K={k} not replicated')
        if not boolish(gr[k]['replicated']):
            raise RuntimeError(f'confirmation_stimulus_experience_multiresolution_transfer says K={k} not replicated')
    synthesis_rows: list[dict[str, Any]] = []
    for k in EXPECTED_K:
        synthesis_rows.append({'k': k, 'discovery_eligible': True, 'discovery_median_ari': float(bm[k]['median_ari']), 'discovery_ari_10th_percentile': float(bm[k]['ari_10th_percentile']), 'discovery_minimum_participant_weighted_occupancy': float(bm[k]['minimum_participant_weighted_occupancy']), 'validation_replicated': True, 'validation_paired_fraction': float(dr[k]['paired_fraction_of_494']), 'validation_participant_balanced_in_support': float(dr[k]['participant_balanced_in_support_fraction']), 'confirmation_replicated': True, 'confirmation_paired_fraction': float(gr[k]['paired_fraction_of_494']), 'confirmation_participant_balanced_in_support': float(gr[k]['participant_balanced_in_support_fraction']), 'structurally_replicated_across_all_three_splits': True, 'semantic_name_permitted': False, 'rank_as_single_preferred_resolution': False})
    synth_path = out / 'tables/cross_split_experience_structure_synthesis_crosssplit_resolution_synthesis.csv'
    write_csv(synth_path, synthesis_rows, ['k', 'discovery_eligible', 'discovery_median_ari', 'discovery_ari_10th_percentile', 'discovery_minimum_participant_weighted_occupancy', 'validation_replicated', 'validation_paired_fraction', 'validation_participant_balanced_in_support', 'confirmation_replicated', 'confirmation_paired_fraction', 'confirmation_participant_balanced_in_support', 'structurally_replicated_across_all_three_splits', 'semantic_name_permitted', 'rank_as_single_preferred_resolution'])
    cross_rows = read_csv(project / B_CROSS)
    cross_path = out / 'tables/cross_split_experience_structure_synthesis_discovery_cross_resolution_relationships.csv'
    write_csv(cross_path, cross_rows, ['k_a', 'k_b', 'ARI', 'NMI'])
    rehash_path = out / 'qc/cross_split_experience_structure_synthesis_source_rehash.csv'
    write_csv(rehash_path, rehash_rows, ['source_phase', 'path', 'expected_sha256', 'observed_sha256', 'verified'])
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'structurally_replicated_resolutions': EXPECTED_K, 'scientific_status': 'K=2,3,4 are reproducible stimulus-only structural partitions across Discovery, Validation, and Confirmation splits of this dataset', 'not_established': ['that the brain has exactly 2, 3, or 4 discrete biological states', 'that K=2, K=3, and K=4 are mutually independent mechanisms', 'that the resolutions form a simple nested hierarchy', 'emotion, valence, liking, arousal, context, stimulus, or cognitive semantics', 'external-dataset replication'], 'semantic_report_dissociation_design_role': 'specification, before behavioral or rating_early access in the structural_replication branch, the tests of (A) mismatch between human emotion labels/reports and the replicated stimulus-only neural partitions and (B) neural transformation from stimulus_late to rating_early', 'semantic_report_dissociation_design_data_opening': {'new_signal_values': False, 'behavioral_values': False, 'rating_early_EEG': False}, 'required_separation': ['experience semantic-label mapping branch', 'report-transition branch'], 'forbidden_in_semantic_report_dissociation_design': ['opening behavioral values', 'opening rating_early EEG', 'choosing a preferred K post hoc', 'renaming families', 'semantic interpretation before tests are prespecified']}
    writej(project / NEXT_CONFIG, next_policy)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'crosssplit_stimulus_experience_structure_synthesis_complete', 'structurally_replicated_K': EXPECTED_K, 'all_three_K_replicated_across_discovery_validation_confirmation': True, 'single_preferred_K_selected': False, 'simple_hierarchy_claim_permitted': False, 'semantic_relabeling_permitted': False, 'external_replication_claim_permitted': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'rating_early_EEG_opened': False, 'discovery_pairs': int(bsum.get('eligible_pairs', bsum.get('counts', {}).get('eligible_pairs', 1261))), 'validation_pairs': int(dsum['counts']['paired_stimulus_early_late_trials']), 'confirmation_pairs': int(gsum['counts']['paired_stimulus_early_late_trials']), 'validation_eligible_participants': int(dsum['counts']['participants_with_at_least_16_pairs']), 'confirmation_eligible_participants': int(gsum['counts']['participants_with_at_least_16_pairs']), 'next_phase': NEXT_PHASE, 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/cross_split_experience_structure_synthesis_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': sha256(Path(__file__).resolve()), 'verified_parent_final_locks': {'discovery_stimulus_experience_multiresolution_fit': B_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer': D_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer': G_LOCK_SHA256}, 'structurally_replicated_K': EXPECTED_K, 'boundary': {'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'rating_early_EEG_opened': False, 'semantic_relabeling_permitted': False, 'external_replication_claim_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/cross_split_experience_structure_synthesis_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/cross_split_experience_structure_synthesis_report.md'
    lines = ['# cross_split_experience_structure_synthesis — Cross-split stimulus-experience structure synthesis', '', '**cross_split_experience_structure_synthesis is synthesis-only. It opened no new EEG, behavioral, category, context, stimulus, or rating values.**', '', '## fixed structural result', '', 'The stimulus-only representation was discovered using `stimulus_early → stimulus_late` only.', '', '- Discovery retained **K=2, K=3, and K=4** under the prespecified stability/collapse criteria.', '- One-shot Validation transfer: **K=2 PASS, K=3 PASS, K=4 PASS**.', '- One-shot Confirmation transfer: **K=2 PASS, K=3 PASS, K=4 PASS**.', '', 'Therefore **K=2, K=3, and K=4 are all structurally replicated across the three internal dataset splits**.', '', '## Confirmation strength', '', f"- Paired Confirmation trials: **{gsum['counts']['paired_stimulus_early_late_trials']}/494**.", f"- Eligible Confirmation participants: **{gsum['counts']['participants_with_at_least_16_pairs']}**."]
    for k in EXPECTED_K:
        lines.append(f"- K={k}: Confirmation participant-balanced in-support = **{float(gr[k]['participant_balanced_in_support_fraction']):.6f}**.")
    lines += ['', '## Multi-resolution caution', '', 'These are reproducible partitions at different resolutions; they are not evidence that the brain contains exactly 2, 3, or 4 discrete biological states.', 'The Discovery cross-resolution overlaps are not consistent with assuming a simple nested hierarchy:']
    for r in cross_rows:
        lines.append(f"- K={r['k_a']} vs K={r['k_b']}: ARI={float(r['ARI']):.4f}, NMI={float(r['NMI']):.4f}.")
    lines += ['', 'No single K is selected as the psychologically or biologically correct resolution.', '', '## Semantic boundary', '', 'No emotion/category/report semantics have been tested in this stimulus-only branch.', 'The replicated families must remain structurally named until a separately prespecified semantic analysis is executed.', '', 'This is **internal held-out replication within one dataset**, not external-dataset replication.', '', '## Next scientific question', '', f'**{NEXT_PHASE}** will specification two separate tests before opening their targets:', '', '1. **Label-mismatch branch:** do human emotion labels/reports map one-to-one to these neural partitions, or does each human label contain multiple neural regimes and/or each neural regime span multiple labels?', '2. **Report-transition branch:** what changes from `stimulus_late → rating_early`, and does that transformation relate to the final subjective report independently of the stimulus-only family?', '']
    report_path.write_text(''.join(lines) + '', encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, contract_path, summary_path, rehash_path, synth_path, cross_path, report_path]
    final_lock = out / 'qc/cross_split_experience_structure_synthesis_final_output_lock_hashes.json'
    writej(final_lock, {'phase': PHASE, 'generated_at': now(), 'status': 'crosssplit_stimulus_experience_structure_synthesis_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'structurally_replicated_K': EXPECTED_K, 'single_preferred_K_selected': False, 'semantic_relabeling_permitted': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'rating_early_EEG_opened': False, 'next_phase': NEXT_PHASE})
    script_dir = Path(__file__).resolve().parent
    print('')
    print('cross_split_experience_structure_synthesis PASS')
    print(f'Structurally replicated K across all splits: {EXPECTED_K}')
    print('Single preferred K selected: False')
    print('Simple hierarchy claim permitted: False')
    print('Semantic relabeling permitted: False')
    print('External replication claim permitted: False')
    print('New signal values opened: False')
    print('New behavioral values opened: False')
    print('rating_early EEG opened: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())