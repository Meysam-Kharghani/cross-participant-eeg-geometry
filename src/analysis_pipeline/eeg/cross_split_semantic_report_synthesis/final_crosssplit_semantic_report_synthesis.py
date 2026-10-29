# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
PHASE = 'cross_split_semantic_report_synthesis'
PHASE_SLUG = 'cross_split_semantic_report_synthesis_final_crosssplit_semantic_report_synthesis'
OUT_REL = Path('outputs/interpretability') / PHASE_SLUG
POLICY_REL = Path('analysis_specifications/S016_cross_split_semantic_report_synthesis_structural.json')
EXPECTED_LOCKS = {'cross_split_experience_structure_synthesis': (Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis/qc/cross_split_experience_structure_synthesis_final_output_lock_hashes.json'), 'c0dc3c2f5930486fd6ab231f0d6b04dc665a3b34da69d5c02af60154148402e5'), 'cross_split_report_transition_structure_synthesis': (Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis/qc/cross_split_report_transition_structure_synthesis_final_output_lock_hashes.json'), '4a2131505761db9a64e888294eb9daf58b30797da42fe486ef9c4ec52d1c5439'), 'discovery_semantic_report_execution': (Path('outputs/interpretability/discovery_semantic_report_execution_discovery_semantic_report_execution/qc/discovery_semantic_report_execution_final_output_lock_hashes.json'), '4ace7a17facf24471572a6a05adc2e6ce55082e1f0890b882c938150f2ed4b59'), 'validation_semantic_report_execution': (Path('outputs/interpretability/validation_semantic_report_execution_validation_semantic_report_execution/qc/validation_semantic_report_execution_final_output_lock_hashes.json'), '51ba5dfa1cd304211cea3093a0a3ab68621852d942920e254adfe34c3410230c'), 'validation_confirmation_semantic_design': (Path('outputs/interpretability/validation_confirmation_semantic_design_validation_synthesis_confirmation_design_specification/qc/validation_confirmation_semantic_design_final_output_lock_hashes.json'), 'abdebbdab3d462c98d70c1771f93908262a0bac9c27799bc5f2db94f549379b6'), 'confirmation_semantic_report_execution': (Path('outputs/interpretability/confirmation_semantic_report_execution_confirmation_semantic_report_execution/qc/confirmation_semantic_report_execution_final_output_lock_hashes.json'), '1f4379632b63bb81a8f99cec3537fdae5af4babeaa654a7cfbeb3c284e1210d0')}
confirmation_semantic_report_execution_POLICY_REL = Path('analysis_specifications/S007_confirmation_semantic_report_cross_split.json')
confirmation_semantic_report_execution_POLICY_SHA = '20e1fbd6bc1ebc5cc1a25f64df87eee28e1be3a77569d107f72eed31d85ab672'
STRUCT_H_SUMMARY = Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis/qc/cross_split_experience_structure_synthesis_summary.json')
STRUCT_H_TABLE = Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis/tables/cross_split_experience_structure_synthesis_crosssplit_resolution_synthesis.csv')
STRUCT_P_SUMMARY = Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis/qc/cross_split_report_transition_structure_synthesis_summary.json')
STRUCT_P_TABLE = Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis/tables/cross_split_report_transition_structure_synthesis_crosssplit_transition_resolution_synthesis.csv')
OVERLAP_TABLE = Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis/tables/cross_split_report_transition_structure_synthesis_experience_transition_crosssplit_metrics.csv')
SPLITS = {'Discovery': {'prefix': 'discovery_semantic_report_execution', 'base': Path('outputs/interpretability/discovery_semantic_report_execution_discovery_semantic_report_execution'), 'registry': 'discovery_semantic_report_execution_discovery_parent_test_registry.csv'}, 'Validation': {'prefix': 'validation_semantic_report_execution', 'base': Path('outputs/interpretability/validation_semantic_report_execution_validation_semantic_report_execution'), 'registry': 'validation_semantic_report_execution_validation_parent_test_registry.csv'}, 'Confirmation': {'prefix': 'confirmation_semantic_report_execution', 'base': Path('outputs/interpretability/confirmation_semantic_report_execution_confirmation_semantic_report_execution'), 'registry': 'confirmation_semantic_report_execution_confirmation_parent_test_registry.csv'}}
EXPECTED_TESTS = ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def verify_lock(project: Path, role: str, rel: Path, expected_sha: str, rehash_rows: list[dict]):
    p = project / rel
    if not p.exists():
        raise RuntimeError(f'missing {role} lock: {rel}')
    obs = sha256_path(p)
    rehash_rows.append({'role': f'{role}_final_lock', 'path': str(rel), 'expected_sha256': expected_sha, 'observed_sha256': obs, 'verified': obs == expected_sha})
    if obs != expected_sha:
        raise RuntimeError(f'{role} final lock drift')
    lock = read_json(p)
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{role} lock has no child hashes')
    for child_rel, child_sha in hashes.items():
        cp = project / child_rel
        if not cp.exists():
            raise RuntimeError(f'{role} fixed child missing: {child_rel}')
        cobs = sha256_path(cp)
        if cobs != child_sha:
            raise RuntimeError(f'{role} child hash drift: {child_rel}')
    return lock

def verify_fixed_child(project: Path, lock: dict, rel: Path):
    hashes = lock.get('hashes', {})
    key = str(rel)
    if key not in hashes:
        raise RuntimeError(f'required child not present in parent lock: {rel}')
    p = project / rel
    obs = sha256_path(p)
    if obs != hashes[key]:
        raise RuntimeError(f'fixed child drift: {rel}')
    return p

def load_split(project: Path, split: str, lock: dict):
    s = SPLITS[split]
    base = s['base']
    prefix = s['prefix']
    reg_rel = base / 'tables' / s['registry']
    reg = pd.read_csv(verify_fixed_child(project, lock, reg_rel))
    if list(reg['test_id']) != EXPECTED_TESTS:
        raise RuntimeError(f'{split} parent-test registry order/content drift')
    if reg['test_id'].duplicated().any():
        raise RuntimeError(f'{split} duplicate parent tests')

    def tbl(suffix):
        rel = base / 'tables' / f'{prefix}_{suffix}'
        return pd.read_csv(verify_fixed_child(project, lock, rel))
    a1 = tbl('A1_parent_metrics.csv')
    a2 = tbl('A2_parent_metrics.csv')
    b2 = tbl('B2_parent_metric.csv')
    b3 = tbl('B3_parent_metrics.csv')
    cohort_rel = base / 'qc' / f'{prefix}_test_cohort_audit.csv'
    cohort = pd.read_csv(verify_fixed_child(project, lock, cohort_rel))
    return (reg, a1, a2, b2, b3, cohort)

def build_effect_table(split, reg, a1, a2, b2, b3, cohort):
    rows = []
    cohort_map = cohort.set_index('test_id').to_dict('index')
    reg_map = reg.set_index('test_id').to_dict('index')
    for _, r in a1.iterrows():
        tid = str(r.test_id)
        rr = reg_map[tid]
        cc = cohort_map[tid]
        rows.append({'split': split, 'family': 'A1', 'test_id': tid, 'k': int(r.k), 'trials': int(cc['trials']), 'participants': int(cc['participants']), 'effect_primary_name': 'NMI', 'effect_primary_value': float(r.NMI), 'effect_secondary_name': 'Cramers_V', 'effect_secondary_value': float(r.Cramers_V), 'p_raw': float(rr['p_raw']), 'p_adjusted': float(rr['p_adjusted']), 'reject_adjusted_0p05': bool(rr['reject_adjusted_0p05'])})
    for _, r in a2.iterrows():
        tid = str(r.test_id)
        rr = reg_map[tid]
        cc = cohort_map[tid]
        rows.append({'split': split, 'family': 'A2', 'test_id': tid, 'k': int(r.k), 'trials': int(cc['trials']), 'participants': int(cc['participants']), 'effect_primary_name': 'Delta_BA', 'effect_primary_value': float(r.Delta_BA), 'effect_secondary_name': 'full_10report_BA_minus_baseline_BA', 'effect_secondary_value': float(r.full_10report_BA - r.baseline_BA), 'p_raw': float(rr['p_raw']), 'p_adjusted': float(rr['p_adjusted']), 'reject_adjusted_0p05': bool(rr['reject_adjusted_0p05'])})
    r = b2.iloc[0]
    tid = 'B2'
    rr = reg_map[tid]
    cc = cohort_map[tid]
    rows.append({'split': split, 'family': 'B2', 'test_id': tid, 'k': np.nan, 'trials': int(cc['trials']), 'participants': int(cc['participants']), 'effect_primary_name': 'Delta_R2_M2_minus_M1', 'effect_primary_value': float(r.Delta_R2_M2_minus_M1), 'effect_secondary_name': 'R2_M2', 'effect_secondary_value': float(r.R2_M2), 'p_raw': float(rr['p_raw']), 'p_adjusted': float(rr['p_adjusted']), 'reject_adjusted_0p05': bool(rr['reject_adjusted_0p05'])})
    for _, r in b3.iterrows():
        tid = str(r.test_id)
        rr = reg_map[tid]
        cc = cohort_map[tid]
        rows.append({'split': split, 'family': 'B3', 'test_id': tid, 'k': int(r.k), 'trials': int(cc['trials']), 'participants': int(cc['participants']), 'effect_primary_name': 'Delta_BA', 'effect_primary_value': float(r.Delta_BA), 'effect_secondary_name': 'full_10report_BA_minus_baseline_BA', 'effect_secondary_value': float(r.full_10report_BA - r.baseline_BA), 'p_raw': float(rr['p_raw']), 'p_adjusted': float(rr['p_adjusted']), 'reject_adjusted_0p05': bool(rr['reject_adjusted_0p05'])})
    out = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(EXPECTED_TESTS)}
    out['_o'] = out.test_id.map(order)
    return out.sort_values('_o').drop(columns='_o').reset_index(drop=True)

def self_test():
    assert EXPECTED_TESTS == ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']
    assert len(EXPECTED_LOCKS) == 6
    assert len(EXPECTED_TESTS) == 9
    print('cross_split_semantic_report_synthesis self-test PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.project:
        raise SystemExit('--project is required')
    project = Path(args.project).resolve()
    out = project / OUT_REL
    if out.exists() and any(out.rglob('*')) and (not args.force):
        raise RuntimeError(f'nonempty cross_split_semantic_report_synthesis output exists: {out}; quarantine it or use --force intentionally')
    if args.force and out.exists():
        shutil.rmtree(out)
    qc = out / 'qc'
    tables = out / 'tables'
    reports = out / 'reports'
    for d in [qc, tables, reports]:
        d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    generated = now_iso()
    rehash = []
    locks = {}
    for role, (rel, exp) in EXPECTED_LOCKS.items():
        locks[role] = verify_lock(project, role, rel, exp, rehash)
    p = project / confirmation_semantic_report_execution_POLICY_REL
    obs = sha256_path(p)
    rehash.append({'role': 'confirmation_semantic_report_execution_to_cross_split_semantic_report_synthesis_policy', 'path': str(confirmation_semantic_report_execution_POLICY_REL), 'expected_sha256': confirmation_semantic_report_execution_POLICY_SHA, 'observed_sha256': obs, 'verified': obs == confirmation_semantic_report_execution_POLICY_SHA})
    if obs != confirmation_semantic_report_execution_POLICY_SHA:
        raise RuntimeError('confirmation_semantic_report_execution->cross_split_semantic_report_synthesis policy drift')
    u_policy = read_json(p)
    if u_policy.get('next_phase') != PHASE or not u_policy.get('next_phase_permitted'):
        raise RuntimeError('confirmation_semantic_report_execution does not authorize cross_split_semantic_report_synthesis')
    if u_policy.get('new_behavioral_target_opening_permitted_in_cross_split_semantic_report_synthesis') is not False or u_policy.get('new_EEG_signal_opening_permitted') is not False:
        raise RuntimeError('cross_split_semantic_report_synthesis value-opening boundary drift')
    h_summary = read_json(verify_fixed_child(project, locks['cross_split_experience_structure_synthesis'], STRUCT_H_SUMMARY))
    h_tbl = pd.read_csv(verify_fixed_child(project, locks['cross_split_experience_structure_synthesis'], STRUCT_H_TABLE))
    p_summary = read_json(verify_fixed_child(project, locks['cross_split_report_transition_structure_synthesis'], STRUCT_P_SUMMARY))
    p_tbl = pd.read_csv(verify_fixed_child(project, locks['cross_split_report_transition_structure_synthesis'], STRUCT_P_TABLE))
    overlap = pd.read_csv(verify_fixed_child(project, locks['cross_split_report_transition_structure_synthesis'], OVERLAP_TABLE))
    if h_summary.get('structurally_replicated_K') != [2, 3, 4] or not h_summary.get('all_three_K_replicated_across_discovery_validation_confirmation'):
        raise RuntimeError('experience structural synthesis drift')
    if p_summary.get('structurally_replicated_report_transition_K') != [2, 3] or not p_summary.get('transition_K_replicated_across_discovery_validation_confirmation'):
        raise RuntimeError('transition structural synthesis drift')
    split_effects = []
    split_regs = {}
    for split, role in [('Discovery', 'discovery_semantic_report_execution'), ('Validation', 'validation_semantic_report_execution'), ('Confirmation', 'confirmation_semantic_report_execution')]:
        reg, a1, a2, b2, b3, cohort = load_split(project, split, locks[role])
        split_regs[split] = reg
        split_effects.append(build_effect_table(split, reg, a1, a2, b2, b3, cohort))
    effects = pd.concat(split_effects, ignore_index=True)
    rec = []
    for tid in EXPECTED_TESTS:
        d = effects[effects.test_id == tid].copy()
        vals = d.effect_primary_value.astype(float).to_numpy()
        if tid.startswith('A1_'):
            direction = 'nonnegative_association_magnitude'
        elif np.all(vals > 0):
            direction = 'positive_all_three'
        elif np.all(vals < 0):
            direction = 'negative_all_three'
        else:
            direction = 'mixed_sign'
        rec.append({'test_id': tid, 'family': d.family.iloc[0], 'n_splits': len(d), 'n_corrected_rejections': int(d.reject_adjusted_0p05.sum()), 'corrected_rejection_in_all_three': bool(d.reject_adjusted_0p05.all()), 'min_raw_p_descriptive_only': float(d.p_raw.min()), 'min_adjusted_p_descriptive_only': float(d.p_adjusted.min()), 'effect_primary_name': d.effect_primary_name.iloc[0], 'effect_direction_pattern': direction, 'crosssplit_p_combination_performed': False})
    recurrence = pd.DataFrame(rec)
    exp_struct = h_tbl[['k', 'structurally_replicated_across_all_three_splits', 'semantic_name_permitted', 'rank_as_single_preferred_resolution']].copy()
    exp_struct.insert(0, 'structure', 'experience')
    trans_struct = p_tbl[['k', 'structurally_replicated_across_all_three_splits', 'semantic_name_permitted', 'rank_as_single_preferred_resolution']].copy()
    trans_struct.insert(0, 'structure', 'report_transition')
    structural = pd.concat([exp_struct, trans_struct], ignore_index=True)
    overlap_summary = overlap.groupby('split', as_index=False).agg(common_trials=('common_trials', 'max'), common_participants=('common_participants', 'max'), min_ARI=('ARI', 'min'), max_ARI=('ARI', 'max'), min_NMI=('NMI', 'min'), max_NMI=('NMI', 'max'))
    total_rej = int(effects.reject_adjusted_0p05.sum())
    all_test_repl = recurrence.corrected_rejection_in_all_three.sum()
    exp_repl = bool(structural.query("structure=='experience'").structurally_replicated_across_all_three_splits.all())
    trans_repl = bool(structural.query("structure=='report_transition'").structurally_replicated_across_all_three_splits.all())
    claims = pd.DataFrame([{'claim_id': 'cross_split_semantic_report_synthesis_outcome_model_integrity', 'claim': 'Experience EEG organization is structurally reproducible at K=2,3,4 across the three internal splits.', 'status': 'SUPPORTED_INTERNAL', 'boundary': 'No single preferred K, simple hierarchy, semantic name, or external-replication claim.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C2', 'claim': 'Report-transition EEG organization is structurally reproducible at K=2,3 across the three internal splits.', 'status': 'SUPPORTED_INTERNAL', 'boundary': 'No semantic name and no claim of a separate biological mechanism.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C3', 'claim': 'The nominal category labels provide a stable mapping to the replicated experience families.', 'status': 'NOT_SUPPORTED', 'boundary': 'Weak/null mapping can support poor alignment or lossy projection; it does not prove labels are wrong.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C4', 'claim': 'The joint ten subjective reports robustly predict replicated experience-family membership beyond the prespecified baseline.', 'status': 'NOT_SUPPORTED', 'boundary': 'Do not mine report subsets after the parent-gate failures.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C5', 'claim': 'Report-transition signatures contain robust incremental information about the ten reports beyond experience signatures.', 'status': 'NOT_SUPPORTED', 'boundary': 'B2 effects are descriptively small; no cross-split p-value combination is permitted.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C6', 'claim': 'Discrete report-transition families robustly map to the ten reports.', 'status': 'NOT_SUPPORTED', 'boundary': 'Validation-only raw tendencies are not replicated in Confirmation and are not corrected discoveries.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C7', 'claim': 'Experience and report-transition partitions have low assignment overlap across all splits.', 'status': 'SUPPORTED_DESCRIPTIVE', 'boundary': 'Low ARI/NMI does not establish statistical independence or distinct biological mechanisms.'}, {'claim_id': 'cross_split_semantic_report_synthesis_C8', 'claim': 'The replicated neural families can be named as specific emotions.', 'status': 'NOT_PERMITTED', 'boundary': 'Semantic identity remains unresolved.'}, {'claim_id': 'cross_split_semantic_report_synthesis_structural_replication', 'claim': 'Report-transition structure demonstrates distortion, dishonesty, suppression, exaggeration, or conscious editing.', 'status': 'NOT_PERMITTED', 'boundary': 'Only neutral report/evaluation-stage neural transformation language is allowed.'}])
    closure = {'phase': PHASE, 'status': 'structural_replication_branch_complete', 'generated_at': generated, 'new_behavioral_values_opened': False, 'new_EEG_signal_values_opened': False, 'new_inferential_tests_run': False, 'crosssplit_p_value_combination_performed': False, 'experience_structurally_replicated_K': [2, 3, 4], 'report_transition_structurally_replicated_K': [2, 3], 'split_level_parent_tests': int(len(effects)), 'split_level_corrected_parent_rejections': total_rej, 'parent_tests_with_corrected_rejection_in_all_three_splits': int(all_test_repl), 'stable_semantic_identity_supported': False, 'stable_report_mapping_supported': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'external_replication_claim_permitted': False, 'branch_decision': 'close_structural_replication_semantic_report_mapping_branch_without_posthoc_rescue', 'recommended_next_roles': ['evidence_ledger_refresh', 'preplanned_physical_neurofeature_characterization_of_replicated_families_using_existing_fixed_structural_features', 'independent_external_replication_if_available']}
    paths = []

    def save_csv(df, name):
        p = tables / name
        df.to_csv(p, index=False)
        paths.append(p)
        return p
    save_csv(effects, 'cross_split_semantic_report_synthesis_crosssplit_parent_test_effect_synthesis.csv')
    save_csv(recurrence, 'cross_split_semantic_report_synthesis_crosssplit_parent_test_recurrence.csv')
    save_csv(structural, 'cross_split_semantic_report_synthesis_structural_replication_summary.csv')
    save_csv(overlap_summary, 'cross_split_semantic_report_synthesis_experience_transition_overlap_summary.csv')
    save_csv(claims, 'cross_split_semantic_report_synthesis_claims_ledger.csv')
    pd.DataFrame([closure]).to_csv(tables / 'cross_split_semantic_report_synthesis_branch_decision.csv', index=False)
    paths.append(tables / 'cross_split_semantic_report_synthesis_branch_decision.csv')
    pd.DataFrame(rehash).to_csv(qc / 'cross_split_semantic_report_synthesis_source_rehash.csv', index=False)
    paths.append(qc / 'cross_split_semantic_report_synthesis_source_rehash.csv')
    contract = {'phase': PHASE, 'generated_at': generated, 'role': 'final cross-split semantic/report synthesis and structural_replication analysis conclusion', 'parent_final_lock_sha256': {k: v[1] for k, v in EXPECTED_LOCKS.items()}, 'confirmation_semantic_report_execution_to_cross_split_semantic_report_synthesis_policy_sha256': confirmation_semantic_report_execution_POLICY_SHA, 'boundary': {'new_behavioral_values_opened': False, 'new_EEG_signal_values_opened': False, 'new_inferential_tests_run': False, 'crosssplit_p_value_combination_performed': False, 'K_reselected': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'source_code_sha256': sha256_path(Path(__file__).resolve())}
    cp = qc / 'cross_split_semantic_report_synthesis_contract.json'
    cp.write_text(json.dumps(contract, indent=2, sort_keys=True) + '', encoding='utf-8')
    paths.append(cp)
    sp = qc / 'cross_split_semantic_report_synthesis_summary.json'
    sp.write_text(json.dumps(closure, indent=2, sort_keys=True) + '', encoding='utf-8')
    paths.append(sp)
    closure_policy = {'phase': PHASE, 'generated_at': generated, 'structural_replication_analysis_concluded': True, 'new_within_structural_replication_posthoc_semantic_tests_permitted': False, 'crosssplit_p_value_combination_permitted': False, 'K_reselection_permitted': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'external_replication_claim_permitted': False, 'allowed_next_roles': closure['recommended_next_roles'], 'core_interpretation': 'Internally replicated multiresolution EEG organization is present, while the prespecified nominal-category and subjective-report mappings do not identify stable semantic content for those families. Report-transition organization is also structurally replicated but is not robustly linked to final report content under the prespecified tests.'}
    pp = project / POLICY_REL
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(closure_policy, indent=2, sort_keys=True) + '', encoding='utf-8')
    paths.append(pp)
    e = effects.set_index(['split', 'test_id'])

    def val(split, tid, col):
        return e.loc[(split, tid), col]
    report = f"# cross_split_semantic_report_synthesis — final cross-split semantic/report synthesis\n\n**structural_replication analysis conclusion. No new targets, EEG values, inferential tests, or cross-split p-value combinations were used.**\n\n## Structural result\n\n- Experience organization: K=2,3,4 replicated structurally across Discovery, Validation, and Confirmation. No single preferred K or simple hierarchy is selected.\n- Report-transition organization: K=2,3 replicated structurally across all three internal splits.\n- Experience vs report-transition assignment overlap remained low: maximum ARI was {overlap_summary.loc[overlap_summary.split == 'Discovery', 'max_ARI'].iloc[0]:.6f} in Discovery, {overlap_summary.loc[overlap_summary.split == 'Validation', 'max_ARI'].iloc[0]:.6f} in Validation, and {overlap_summary.loc[overlap_summary.split == 'Confirmation', 'max_ARI'].iloc[0]:.6f} in Confirmation. This does not establish independent biological mechanisms.\n\n## prespecified semantic/report tests\n\nAcross 3 splits × 9 parent tests = {len(effects)} split-level parent tests, **{total_rej} passed the pre-specified corrected threshold**. No parent test passed correction in all three splits. No post-hoc cross-split combined p-value was calculated.\n\n### A1 — nominal category ↔ experience family\nNo corrected parent test passed in any split. The Validation A1 K=3 raw tendency (raw p={val('Validation', 'A1_K3', 'p_raw'):.6f}; adjusted p={val('Validation', 'A1_K3', 'p_adjusted'):.6f}) was not reproduced in Confirmation (raw p={val('Confirmation', 'A1_K3', 'p_raw'):.6f}; adjusted p={val('Confirmation', 'A1_K3', 'p_adjusted'):.6f}). The allowed interpretation is poor/stable-mapping failure under the prespecified test—not that human emotion labels are intrinsically wrong.\n\n### A2 — all ten reports → experience family\nNo K passed in any split. Confirmation ΔBA values were K2={val('Confirmation', 'A2_K2', 'effect_primary_value'):.6f}, K3={val('Confirmation', 'A2_K3', 'effect_primary_value'):.6f}, K4={val('Confirmation', 'A2_K4', 'effect_primary_value'):.6f}; all parent gates failed. Report-subset fishing is not permitted.\n\n### B2 — transition signature incremental report information\nThe prespecified incremental ΔR² was descriptively positive but tiny in all three splits: Discovery={val('Discovery', 'B2', 'effect_primary_value'):.6f}, Validation={val('Validation', 'B2', 'effect_primary_value'):.6f}, Confirmation={val('Confirmation', 'B2', 'effect_primary_value'):.6f}. The corresponding p-values were {val('Discovery', 'B2', 'p_raw'):.6f}, {val('Validation', 'B2', 'p_raw'):.6f}, {val('Confirmation', 'B2', 'p_raw'):.6f}. This is not sufficient evidence for robust incremental report information.\n\n### B3 — discrete transition family ↔ reports\nNo corrected parent test passed. The Validation K2 raw tendency (raw p={val('Validation', 'B3_K2', 'p_raw'):.6f}; adjusted p={val('Validation', 'B3_K2', 'p_adjusted'):.6f}) was not reproduced in Confirmation (ΔBA={val('Confirmation', 'B3_K2', 'effect_primary_value'):.6f}, raw p={val('Confirmation', 'B3_K2', 'p_raw'):.6f}).\n\n## What the structural_replication branch supports\n\nThe strongest defensible result is a **dissociation between structural reproducibility and semantic identification**: multiresolution EEG organizations replicate internally, but the prespecified nominal-category and ten-report tests do not provide a stable semantic mapping for the replicated families. The report-transition organization is also structurally reproducible, yet its relationship to final report content is not robust under the prespecified B2/B3 tests.\n\nThis is compatible with the possibility that conventional category/report descriptions are a coarse or lossy projection of the measured neural organization, but it does not prove that labels are wrong, nor that the EEG families are true emotion types.\n\n## Family naming\n\nK=2/3/4 experience families and K=2/3 transition families remain structural/feature-space entities. Semantic emotion names are not permitted from structural_replication. Their next defensible characterization, if pursued, is physical/neurofeature characterization using the already-fixed structural representations—not new behavioral outcome mining.\n\n## Branch decision\n\n**Close the structural_replication semantic/report-mapping branch without post-hoc rescue.** Valid next roles are evidence-ledger refresh, preplanned physical characterization of the replicated structural families using existing fixed features, or independent external replication."
    rp = reports / 'cross_split_semantic_report_synthesis_report.md'
    rp.write_text(report, encoding='utf-8')
    paths.append(rp)
    lock_hashes = {}
    for pth in paths:
        rel = str(pth.resolve().relative_to(project.resolve()))
        lock_hashes[rel] = sha256_path(pth)
    final = {'phase': PHASE, 'status': 'structural_replication_branch_complete', 'generated_at': generated, 'hashes': lock_hashes, 'new_behavioral_values_opened': False, 'new_EEG_signal_values_opened': False, 'new_inferential_tests_run': False, 'crosssplit_p_value_combination_performed': False, 'K_reselected': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'external_replication_claim_permitted': False, 'split_level_corrected_parent_rejections': total_rej, 'experience_structurally_replicated_K': [2, 3, 4], 'report_transition_structurally_replicated_K': [2, 3]}
    fp = qc / 'cross_split_semantic_report_synthesis_final_output_lock_hashes.json'
    fp.write_text(json.dumps(final, indent=2, sort_keys=True) + '', encoding='utf-8')
    paths.append(fp)
    parent_refs = [project / rel for rel, _ in EXPECTED_LOCKS.values()] + [project / confirmation_semantic_report_execution_POLICY_REL]
    parent_refs += [project / STRUCT_H_SUMMARY, project / STRUCT_H_TABLE, project / STRUCT_P_SUMMARY, project / STRUCT_P_TABLE, project / OVERLAP_TABLE]
    for split, s in SPLITS.items():
        parent_refs.append(project / s['base'] / 'tables' / s['registry'])
    print('cross_split_semantic_report_synthesis PASS')
    print('structural_replication branch: CLOSED')
    print('Experience structural K: 2,3,4')
    print('Report-transition structural K: 2,3')
    print(f'Corrected parent rejections across 27 split-level tests: {total_rej}')
    print('Stable semantic identity supported: False')
    print('Stable report mapping supported: False')
    print('New behavioral/EEG values opened: False / False')
    print('Cross-split p-value combination performed: False')
    print('Recommended next role: evidence refresh or fixed-feature physical characterization; no structural_replication post-hoc rescue')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())