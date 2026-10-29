# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'heldout_topography_transfer_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
full_channel_topology_association_LOCK = Path('outputs/interpretability/full_channel_topology_association_full_channel_topology_association/qc/full_channel_topology_association_final_output_lock_hashes.json')
full_channel_topology_association_LOCK_SHA = 'fc66f67febff1bc8876ad0871e4823c3aad495589d0cfc17f8e2c448435dd0f1'
validation_preprocessing_qc_LOCK = Path('outputs/qc/validation_preprocessing_qc_final_output_lock_hashes.json')
validation_preprocessing_qc_LOCK_SHA = '984aba20ce015884184a037dba87b7ed4b4b8228610b479b5a881090a9c6f460'
validation_preprocessing_qc_EPOCH = Path('outputs/tables/validation_preprocessing_qc_validation_epoch_amplitude_qc.csv')
validation_preprocessing_qc_INV = Path('outputs/tables/validation_preprocessing_qc_validation_large_array_inventory.csv')
validation_preprocessing_qc_CH = Path('outputs/qc/validation_preprocessing_qc_validation_recording_channel_qc.csv')
validation_fixed_assignment_LOCK = Path('outputs/qc/validation_fixed_assignment_final_output_lock_hashes.json')
validation_fixed_assignment_LOCK_SHA = 'b6ea679c97243167292bf2e167883b17b067bb4b220d38ed0e1c0e35fd783450'
validation_fixed_assignment_ASSIGN = Path('outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv')
confirmation_preprocessing_qc_LOCK = Path('outputs/qc/confirmation_preprocessing_qc_final_output_lock_hashes.json')
confirmation_preprocessing_qc_LOCK_SHA = 'a7c6c22af5656aee132ab89fddba83f3bba019197f3890bb60866f206833a582'
confirmation_preprocessing_qc_EPOCH = Path('outputs/tables/confirmation_preprocessing_qc_confirmation_epoch_amplitude_qc.csv')
confirmation_preprocessing_qc_INV = Path('outputs/tables/confirmation_preprocessing_qc_confirmation_large_array_inventory.csv')
confirmation_preprocessing_qc_CH = Path('outputs/qc/confirmation_preprocessing_qc_confirmation_recording_channel_qc.csv')
confirmation_fixed_assignment_LOCK = Path('outputs/qc/confirmation_fixed_assignment_final_output_lock_hashes.json')
confirmation_fixed_assignment_LOCK_SHA = '7eb453fcf6822d627d5ea451f183477045eec4d45d8645c218eceb7aa5d7e7a1'
confirmation_fixed_assignment_ASSIGN = Path('outputs/tables/confirmation_fixed_assignment_confirmation_trajectory_assignments.csv')
outcome_free_features_SCRIPT = Path('src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py')
outcome_free_features_SCRIPT_SHA = 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba'
SUPPORTED = {'time_frequency|early_to_late', 'waveform_dct|segment_cross_product', 'covariance_topology|segment_cross_product'}

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def writej(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def truth(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def require(project, rels):
    miss = [str(r) for r in rels if not (project / r).is_file()]
    if miss:
        raise RuntimeError('Missing prerequisite files:\n  - ' + '\n  - '.join(miss))

def verify_lock(project, rel, expected_sha, phase):
    p = project / rel
    got = sha(p)
    if got != expected_sha:
        raise RuntimeError(f'{phase} final-lock SHA drift: {got}')
    x = readj(p)
    if x.get('phase') != phase:
        raise RuntimeError(f'Wrong phase in {rel}')
    return x

def verify_child(project, lock, key, rel):
    exp = lock.get('hashes', {}).get(key)
    if not exp:
        raise RuntimeError(f'Missing lock key {key}')
    got = sha(project / rel)
    if got != exp:
        raise RuntimeError(f'Child hash mismatch: {rel}')

def heldout_audit(project, split, epoch_rel, assign_rel):
    ep = pd.read_csv(project / epoch_rel, usecols=['subject_id', 'raw_behavior_row_id', 'epoch_family', 'raw_source_path', 'source_array_row_0based', 'modeling_eligible'], low_memory=False)
    ass = pd.read_csv(project / assign_rel, usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based'], low_memory=False)
    ep['subject_id'] = ep.subject_id.astype(str)
    ass['subject_id'] = ass.subject_id.astype(str)
    ep['raw_behavior_row_id'] = ep.raw_behavior_row_id.astype(str)
    ass['raw_behavior_row_id'] = ass.raw_behavior_row_id.astype(str)
    ep = ep[ep.modeling_eligible.map(truth)].copy()
    m = ep.merge(ass, on=['subject_id', 'raw_behavior_row_id'], how='inner', validate='many_to_one')
    c = m.groupby(['subject_id', 'raw_behavior_row_id']).epoch_family.nunique()
    if len(c) != len(ass) or not c.eq(3).all():
        raise RuntimeError(f'{split}: 3-epoch coverage failed: {len(c)}/{len(ass)}')
    ps = ass.groupby('subject_id').agg(trajectories=('raw_behavior_row_id', 'size'), family1=('trajectory_family_1based', lambda x: int(np.sum(np.asarray(x) == 1))), family2=('trajectory_family_1based', lambda x: int(np.sum(np.asarray(x) == 2)))).reset_index()
    ps['both_families'] = ps.family1.gt(0) & ps.family2.gt(0)
    return (ass, ps)

def self_test():
    if len(SUPPORTED) != 3 or 2 ** 12 != 4096 or 2 ** 11 != 2048:
        raise RuntimeError('Protocol self-test failed')
    print('heldout_topography_transfer_design self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--bundle-dir', type=Path)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.bundle_dir is None:
        raise RuntimeError('--bundle-dir is required for a real run')
    project = a.project.resolve()
    bdir = a.bundle_dir.resolve()
    template_sha = readj(bdir / 'prespecified_TEMPLATE_HASHES.json')
    for n, exp in template_sha.items():
        p = bdir / n
        if not p.is_file() or sha(p) != exp:
            raise RuntimeError(f'Bundled template integrity failure: {n}')
    require(project, [full_channel_topology_association_LOCK, validation_preprocessing_qc_LOCK, validation_preprocessing_qc_EPOCH, validation_preprocessing_qc_INV, validation_preprocessing_qc_CH, validation_fixed_assignment_LOCK, validation_fixed_assignment_ASSIGN, confirmation_preprocessing_qc_LOCK, confirmation_preprocessing_qc_EPOCH, confirmation_preprocessing_qc_INV, confirmation_preprocessing_qc_CH, confirmation_fixed_assignment_LOCK, confirmation_fixed_assignment_ASSIGN, outcome_free_features_SCRIPT])
    e = verify_lock(project, full_channel_topology_association_LOCK, full_channel_topology_association_LOCK_SHA, 'full_channel_topology_association')
    if e.get('global_parent_tests_Holm_supported') != 3 or e.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('full_channel_topology_association support/boundary drift')
    if e.get('Validation_opened') is not False or e.get('Confirmation_opened') is not False:
        raise RuntimeError('full_channel_topology_association held-out boundary drift')
    for rel, exp in e['hashes'].items():
        p = project / rel
        if not p.is_file() or sha(p) != exp:
            raise RuntimeError(f'full_channel_topology_association artifact mismatch: {rel}')
    es = readj(project / 'outputs/interpretability/full_channel_topology_association_full_channel_topology_association/qc/full_channel_topology_association_summary.json')
    if set(es.get('supported_exploratory_parents', [])) != SUPPORTED:
        raise RuntimeError('full_channel_topology_association supported-parent roster drift')
    k = verify_lock(project, validation_preprocessing_qc_LOCK, validation_preprocessing_qc_LOCK_SHA, 'validation_preprocessing_qc')
    if k.get('reports_outcomes_read') is not False or k.get('confirmation_opened') is not False:
        raise RuntimeError('validation_preprocessing_qc boundary drift')
    for key, rel in [('epoch_qc', validation_preprocessing_qc_EPOCH), ('inventory', validation_preprocessing_qc_INV), ('recording_qc', validation_preprocessing_qc_CH)]:
        verify_child(project, k, key, rel)
    m = verify_lock(project, validation_fixed_assignment_LOCK, validation_fixed_assignment_LOCK_SHA, 'validation_fixed_assignment')
    if m.get('structural_transfer_gate_passed') is not True or m.get('reports_outcomes_read') is not False:
        raise RuntimeError('validation_fixed_assignment boundary drift')
    verify_child(project, m, 'assignments', validation_fixed_assignment_ASSIGN)
    p = verify_lock(project, confirmation_preprocessing_qc_LOCK, confirmation_preprocessing_qc_LOCK_SHA, 'confirmation_preprocessing_qc')
    if p.get('reports_outcomes_read') is not False or p.get('confirmation_reports_outcomes_read') is not False:
        raise RuntimeError('confirmation_preprocessing_qc boundary drift')
    for key, rel in [('epoch_qc', confirmation_preprocessing_qc_EPOCH), ('inventory', confirmation_preprocessing_qc_INV), ('recording_qc', confirmation_preprocessing_qc_CH)]:
        verify_child(project, p, key, rel)
    r = verify_lock(project, confirmation_fixed_assignment_LOCK, confirmation_fixed_assignment_LOCK_SHA, 'confirmation_fixed_assignment')
    if r.get('structural_transfer_gate_passed') is not True or r.get('reports_outcomes_read') is not False:
        raise RuntimeError('confirmation_fixed_assignment boundary drift')
    verify_child(project, r, 'assignments', confirmation_fixed_assignment_ASSIGN)
    if sha(project / outcome_free_features_SCRIPT) != outcome_free_features_SCRIPT_SHA:
        raise RuntimeError('outcome_free_features extractor source hash drift')
    va, vps = heldout_audit(project, 'Validation', validation_preprocessing_qc_EPOCH, validation_fixed_assignment_ASSIGN)
    ca, cps = heldout_audit(project, 'Confirmation', confirmation_preprocessing_qc_EPOCH, confirmation_fixed_assignment_ASSIGN)
    if len(va) != 437 or va.subject_id.nunique() != 12 or (not vps.both_families.all()):
        raise RuntimeError('Validation prespecified roster drift')
    if len(ca) != 446 or ca.subject_id.nunique() != 11 or (not cps.both_families.all()):
        raise RuntimeError('Confirmation prespecified roster drift')
    out = project / 'outputs/interpretability/heldout_topography_transfer_design_heldout_topography_transfer_design_specification'
    final = out / 'qc/heldout_topography_transfer_design_final_output_lock_hashes.json'
    if out.exists():
        if final.is_file():
            if a.force:
                shutil.rmtree(out)
            else:
                raise RuntimeError(f'Completed heldout_topography_transfer_design output exists: {out}')
        else:
            shutil.rmtree(out)
    (out / 'templates').mkdir(parents=True)
    (out / 'tables').mkdir()
    (out / 'qc').mkdir()
    (out / 'reports').mkdir()
    for n in template_sha:
        shutil.copy2(bdir / n, out / 'templates' / n)
    vps.insert(0, 'split', 'Validation')
    cps.insert(0, 'split', 'Confirmation')
    pd.concat([vps, cps], ignore_index=True).to_csv(out / 'tables/heldout_topography_transfer_design_heldout_family_roster.csv', index=False)
    parents = pd.DataFrame([['T1', 'time_frequency|early_to_late', '64ch x 30 TF after per-map spatial median/IQR; late minus early', 'all 1920 Discovery signed channel x TF effects', 0.0028, 'PO8 channel; no TF subfeature passed maxT'], ['T2', 'waveform_dct|segment_cross_product', '64ch x 20 DCT after per-map spatial median/IQR; (late-early)*(rating-late)', 'all 1280 Discovery signed channel x DCT effects', 0.0028, 'FP2 channel and DCT16'], ['T3', 'covariance_topology|segment_cross_product', '2016 Fisher-z correlation edges; (late-early)*(rating-late)', 'top 100 Discovery signed edges', 0.0015, '12 maxT-significant coarse region pairs']], columns=['parent_id', 'discovery_parent', 'full_parent_representation', 'direction_template', 'discovery_Holm_p', 'fixed_local_robustness'])
    parents.to_csv(out / 'tables/heldout_topography_transfer_design_prespecified_parent_transfer_roster.csv', index=False)
    policy = {'phase': PHASE, 'generated_at': now(), 'role': 'design_specification_only_no_heldout_signal_values_opened', 'Discovery_supported_parent_roster': sorted(SUPPORTED), 'heldout_sequence': {'first': 'Validation', 'Confirmation_opening_rule': 'Only if at least 2 of 3 parents replicate in Validation.'}, 'heldout_feature_extraction': {'algorithm': 'exact outcome_free_features extract_epoch_features', 'source_code_path': str(outcome_free_features_SCRIPT), 'source_code_sha256': outcome_free_features_SCRIPT_SHA, 'input': 'prespecified validation_preprocessing_qc/confirmation_preprocessing_qc preprocessed epoch arrays + channel masks', 'behavioral_values_used': False}, 'per_split_primary_tests': {'count': 6, 'tests_per_parent': ['A: full-parent family-association energy, same statistic class as full_channel_topology_association', 'B: directional concordance with prespecified Discovery template using participant-level cosine projections'], 'participant_unit': 'one F2-minus-F1 contrast vector per participant; both families required', 'energy_null': 'all possible participant sign flips (Validation 4096; Confirmation 2048)', 'concordance_null': 'all possible sign flips of participant cosine projections; one-sided mean cosine > 0', 'multiplicity': 'Holm jointly across all 6 primary tests within each split', 'parent_replication_rule': 'BOTH A and B Holm p<=0.05 AND mean cosine>0 AND median participant cosine>0'}, 'Validation_gate': {'pass': 'at least 2 of 3 parents replicate', 'fail': 'Confirmation signal-derived features remain unopened for analysis_stage transfer'}, 'Confirmation_final_gate': {'evaluated_only_if_Validation_gate_passes': True, 'strong_same_dataset_replication': 'at least 2 IDENTICAL parent names replicate in both Validation and Confirmation'}, 'secondary_localization_transfer': {'drives_gate': False, 'T1': 'PO8 multivariate TF channel effect', 'T2': 'FP2 multivariate DCT channel effect and DCT16 spatial effect', 'T3': '12 Discovery-significant covariance region-pair energy localizations with maxT across prespecified 12'}, 'claim_boundaries': {'semantic_relabeling_permitted': False, 'psychological_interpretation_tested': False, 'external_dataset_replication': False, 'behavioral_reports_opened': False, 'family_refit_or_rename': False}, 'next_phase': 'validation_full_channel_transfer'}
    writej(out / 'qc/heldout_topography_transfer_design_prespecified_transfer_policy.json', policy)
    sr = pd.DataFrame([['full_channel_topology_association', 'final_lock', str(full_channel_topology_association_LOCK), full_channel_topology_association_LOCK_SHA, sha(project / full_channel_topology_association_LOCK), True], ['validation_preprocessing_qc', 'final_lock', str(validation_preprocessing_qc_LOCK), validation_preprocessing_qc_LOCK_SHA, sha(project / validation_preprocessing_qc_LOCK), True], ['validation_fixed_assignment', 'final_lock', str(validation_fixed_assignment_LOCK), validation_fixed_assignment_LOCK_SHA, sha(project / validation_fixed_assignment_LOCK), True], ['confirmation_preprocessing_qc', 'final_lock', str(confirmation_preprocessing_qc_LOCK), confirmation_preprocessing_qc_LOCK_SHA, sha(project / confirmation_preprocessing_qc_LOCK), True], ['confirmation_fixed_assignment', 'final_lock', str(confirmation_fixed_assignment_LOCK), confirmation_fixed_assignment_LOCK_SHA, sha(project / confirmation_fixed_assignment_LOCK), True], ['outcome_free_features', 'extractor', str(outcome_free_features_SCRIPT), outcome_free_features_SCRIPT_SHA, sha(project / outcome_free_features_SCRIPT), True]], columns=['source_phase', 'artifact', 'relative_path', 'expected_sha256', 'observed_sha256', 'passed'])
    sr.to_csv(out / 'qc/heldout_topography_transfer_design_source_rehash.csv', index=False)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'heldout_full_channel_transfer_design_prespecified', 'Discovery_supported_parents': sorted(SUPPORTED), 'Validation': {'participants': 12, 'trajectories': 437, 'both_families_participants': int(vps.both_families.sum())}, 'Confirmation': {'participants': 11, 'trajectories': 446, 'both_families_participants': int(cps.both_families.sum())}, 'heldout_signal_values_opened': False, 'behavioral_report_values_opened': False, 'semantic_relabeling_permitted': False, 'Confirmation_transfer_values_may_open_now': False, 'next_phase': 'validation_full_channel_transfer', 'next_phase_role': 'One-shot Validation full-channel structural transfer of the 3 prespecified Discovery parent patterns.'}
    writej(out / 'qc/heldout_topography_transfer_design_summary.json', summary)
    (out / 'reports/heldout_topography_transfer_design_report.md').write_text('# heldout_topography_transfer_design — Held-out full-channel/topology transfer analysis specification\n\nfull_channel_topology_association produced three Holm-supported Discovery parents. heldout_topography_transfer_design specifications a one-shot\nValidation structural transfer without opening held-out signal arrays.\n\nFor each parent, held-out replication requires BOTH:\n1. a family association in the same full parent representation; and\n2. directional concordance with the prespecified Discovery template.\n\nSix primary p-values are corrected jointly by Holm within each split.\nConfirmation opens only if at least 2/3 parents replicate in Validation.\nNo behavioral semantics or family renaming is permitted.', encoding='utf-8')
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    hashes = {str(p.relative_to(project)): sha(p) for p in files}
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': 'heldout_full_channel_transfer_design_prespecified', 'hashes': hashes, 'heldout_signal_values_opened': False, 'behavioral_report_values_opened': False, 'semantic_relabeling_permitted': False, 'Confirmation_transfer_values_may_open_now': False, 'next_phase': 'validation_full_channel_transfer'})
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    hz = project / '' / f'heldout_topography_transfer_design_heldout_topography_transfer_design_specification_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    hs = sha(hz)
    Path(str(hz) + '.sha256').write_text(f'{hs}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] heldout_topography_transfer_design PASS')
    print('Discovery supported parents prespecified: 3')
    print(f'Validation: {len(va)} trajectories / {va.subject_id.nunique()} participants')
    print(f'Confirmation: {len(ca)} trajectories / {ca.subject_id.nunique()} participants')
    print('Held-out signal values opened: False')
    print('Confirmation transfer values may open now: False')
    print('Next phase: validation_full_channel_transfer')
    print('', hz)
    print('SHA-256:', hs)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())