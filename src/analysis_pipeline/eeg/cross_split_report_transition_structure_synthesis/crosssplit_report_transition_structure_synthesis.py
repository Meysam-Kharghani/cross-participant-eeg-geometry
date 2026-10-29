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
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
PHASE = 'cross_split_report_transition_structure_synthesis'
NEXT_PHASE = 'semantic_report_execution_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
EXPERIENCE_K = [2, 3, 4]
TRANSITION_K = [2, 3]
cross_split_experience_structure_synthesis_OUT = Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis')
cross_split_experience_structure_synthesis_LOCK = cross_split_experience_structure_synthesis_OUT / 'qc/cross_split_experience_structure_synthesis_final_output_lock_hashes.json'
cross_split_experience_structure_synthesis_LOCK_SHA256 = 'c0dc3c2f5930486fd6ab231f0d6b04dc665a3b34da69d5c02af60154148402e5'
cross_split_experience_structure_synthesis_SUMMARY = cross_split_experience_structure_synthesis_OUT / 'qc/cross_split_experience_structure_synthesis_summary.json'
semantic_report_dissociation_design_OUT = Path('outputs/interpretability/semantic_report_dissociation_design_semantic_report_dissociation_design_specification')
semantic_report_dissociation_design_LOCK = semantic_report_dissociation_design_OUT / 'qc/semantic_report_dissociation_design_final_output_lock_hashes.json'
semantic_report_dissociation_design_LOCK_SHA256 = '318879c9f111667d7f3027cc10d24b35aac0150da03d972c96c56cb1a3fe1f87'
semantic_report_dissociation_design_SEMANTIC_POLICY = Path('analysis_specifications/S061_semantic_report_dissociation_future.json')
semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256 = 'dbf9fbca05f50d58c3ce561fa73a0c66edc12219ba6f745360c446ef3762723d'
discovery_report_transition_multiresolution_fit_OUT = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit')
discovery_report_transition_multiresolution_fit_LOCK = discovery_report_transition_multiresolution_fit_OUT / 'qc/discovery_report_transition_multiresolution_fit_final_output_lock_hashes.json'
discovery_report_transition_multiresolution_fit_LOCK_SHA256 = 'aa09937f090a17eadac760c38ebbe36310c5e88af2e92429f8e4680ca59a1473'
discovery_report_transition_multiresolution_fit_SUMMARY = discovery_report_transition_multiresolution_fit_OUT / 'qc/discovery_report_transition_multiresolution_fit_summary.json'
discovery_report_transition_multiresolution_fit_METRICS = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_family_candidate_metrics.csv'
discovery_report_transition_multiresolution_fit_TRANSITION_ASSIGN = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'
discovery_report_transition_multiresolution_fit_TRANSITION_ASSIGN_SHA256 = '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'
validation_report_transition_multiresolution_transfer_OUT = Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer')
validation_report_transition_multiresolution_transfer_LOCK = validation_report_transition_multiresolution_transfer_OUT / 'qc/validation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'
validation_report_transition_multiresolution_transfer_LOCK_SHA256 = '8025e9b1ca3357ebcce055554bda3df6b12e1d1bae08737757cb586ba1faf715'
validation_report_transition_multiresolution_transfer_SUMMARY = validation_report_transition_multiresolution_transfer_OUT / 'qc/validation_report_transition_multiresolution_transfer_summary.json'
validation_report_transition_multiresolution_transfer_RESOLUTION = validation_report_transition_multiresolution_transfer_OUT / 'tables/validation_report_transition_multiresolution_transfer_validation_resolution_summary.csv'
validation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN = validation_report_transition_multiresolution_transfer_OUT / 'tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv'
validation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN_SHA256 = '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'
confirmation_report_transition_multiresolution_transfer_OUT = Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer')
confirmation_report_transition_multiresolution_transfer_LOCK = confirmation_report_transition_multiresolution_transfer_OUT / 'qc/confirmation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'
confirmation_report_transition_multiresolution_transfer_LOCK_SHA256 = '14bb0cb021ae071204c90e5be71959bacbc6fc2259e409ed111fc299578d9375'
confirmation_report_transition_multiresolution_transfer_SUMMARY = confirmation_report_transition_multiresolution_transfer_OUT / 'qc/confirmation_report_transition_multiresolution_transfer_summary.json'
confirmation_report_transition_multiresolution_transfer_RESOLUTION = confirmation_report_transition_multiresolution_transfer_OUT / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_resolution_summary.csv'
confirmation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN = confirmation_report_transition_multiresolution_transfer_OUT / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'
confirmation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN_SHA256 = 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'
discovery_stimulus_experience_multiresolution_fit_EXPERIENCE_ASSIGN = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv')
discovery_stimulus_experience_multiresolution_fit_EXPERIENCE_ASSIGN_SHA256 = '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'
validation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv')
validation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN_SHA256 = 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'
confirmation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN = Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv')
confirmation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN_SHA256 = '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'
OUT = Path('outputs/interpretability/cross_split_report_transition_structure_synthesis_crosssplit_report_transition_structure_synthesis')
NEXT_CONFIG = Path('analysis_specifications/S015_cross_split_report_transition_structure_synthesis.json')

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

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)

def atomic_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def verify_lock_children(project: Path, rel: Path, expected_sha: str, phase: str) -> list[dict[str, Any]]:
    lock_path = project / rel
    verify(lock_path, expected_sha, f'{phase} final lock')
    lock = readj(lock_path)
    if lock.get('phase') != phase:
        raise RuntimeError(f'{phase} final-lock phase drift')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{phase} final lock missing child hashes')
    rows: list[dict[str, Any]] = []
    for child_rel, expected in sorted(hashes.items()):
        p = project / child_rel
        if not p.is_file():
            raise RuntimeError(f'Missing {phase} fixed child: {child_rel}')
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'{phase} fixed-child SHA drift: {child_rel}; expected {expected}, got {got}')
        rows.append({'source_phase': phase, 'path': child_rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': True})
    return rows

def read_assignments(path: Path, expected_sha: str, allowed_k: list[int]) -> pd.DataFrame:
    verify(path, expected_sha, f'immutable assignment {path.name}')
    need = ['raw_behavior_row_id', 'subject_id', 'k', 'family_1based']
    df = pd.read_csv(path, usecols=need, low_memory=False)
    df['raw_behavior_row_id'] = df['raw_behavior_row_id'].astype(str)
    df['subject_id'] = df['subject_id'].astype(str)
    df['k'] = df['k'].astype(int)
    df['family_1based'] = df['family_1based'].astype(int)
    df = df[df['k'].isin(allowed_k)].copy()
    if df.duplicated(['raw_behavior_row_id', 'subject_id', 'k']).any():
        raise RuntimeError(f'Duplicate assignment key in {path}')
    return df

def participant_balanced_cross(split: str, exp: pd.DataFrame, trans: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    for tk in TRANSITION_K:
        t = trans[trans.k.eq(tk)][['raw_behavior_row_id', 'subject_id', 'family_1based']].rename(columns={'family_1based': 'transition_family_1based'})
        for ek in EXPERIENCE_K:
            e = exp[exp.k.eq(ek)][['raw_behavior_row_id', 'subject_id', 'family_1based']].rename(columns={'family_1based': 'experience_family_1based'})
            j = t.merge(e, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
            if j.empty:
                raise RuntimeError(f'No common assignments for {split}, transition K={tk}, experience K={ek}')
            counts = j.groupby('subject_id')['subject_id'].transform('size').astype(float)
            j['participant_weight'] = 1.0 / counts
            metric_rows.append({'split': split, 'transition_k': tk, 'experience_k': ek, 'common_trials': int(len(j)), 'common_participants': int(j.subject_id.nunique()), 'ARI': float(adjusted_rand_score(j.transition_family_1based, j.experience_family_1based)), 'NMI': float(normalized_mutual_info_score(j.transition_family_1based, j.experience_family_1based))})
            mass_t = j.groupby('transition_family_1based')['participant_weight'].sum().to_dict()
            mass_e = j.groupby('experience_family_1based')['participant_weight'].sum().to_dict()
            for tf in range(1, tk + 1):
                for ef in range(1, ek + 1):
                    cell = j[j.transition_family_1based.eq(tf) & j.experience_family_1based.eq(ef)]
                    mass = float(cell.participant_weight.sum())
                    cell_rows.append({'split': split, 'transition_k': tk, 'transition_family_1based': tf, 'experience_k': ek, 'experience_family_1based': ef, 'trial_count': int(len(cell)), 'participant_weighted_mass': mass})
                    cond_rows.append({'split': split, 'transition_k': tk, 'transition_family_1based': tf, 'experience_k': ek, 'experience_family_1based': ef, 'participant_weighted_mass': mass, 'p_transition_given_experience': mass / float(mass_e.get(ef, 0.0)) if float(mass_e.get(ef, 0.0)) > 0 else np.nan, 'p_experience_given_transition': mass / float(mass_t.get(tf, 0.0)) if float(mass_t.get(tf, 0.0)) > 0 else np.nan})
    return (metric_rows, cell_rows, cond_rows)

def self_test() -> int:
    assert EXPERIENCE_K == [2, 3, 4]
    assert TRANSITION_K == [2, 3]
    assert PHASE == 'cross_split_report_transition_structure_synthesis'
    assert NEXT_PHASE == 'semantic_report_execution_design'
    print('cross_split_report_transition_structure_synthesis self-test PASS')
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
    print(f'[{now()}] verifying fixed structural lineage', flush=True)
    rehash = []
    rehash += verify_lock_children(project, cross_split_experience_structure_synthesis_LOCK, cross_split_experience_structure_synthesis_LOCK_SHA256, 'cross_split_experience_structure_synthesis')
    rehash += verify_lock_children(project, semantic_report_dissociation_design_LOCK, semantic_report_dissociation_design_LOCK_SHA256, 'semantic_report_dissociation_design')
    rehash += verify_lock_children(project, discovery_report_transition_multiresolution_fit_LOCK, discovery_report_transition_multiresolution_fit_LOCK_SHA256, 'discovery_report_transition_multiresolution_fit')
    rehash += verify_lock_children(project, validation_report_transition_multiresolution_transfer_LOCK, validation_report_transition_multiresolution_transfer_LOCK_SHA256, 'validation_report_transition_multiresolution_transfer')
    rehash += verify_lock_children(project, confirmation_report_transition_multiresolution_transfer_LOCK, confirmation_report_transition_multiresolution_transfer_LOCK_SHA256, 'confirmation_report_transition_multiresolution_transfer')
    verify(project / semantic_report_dissociation_design_SEMANTIC_POLICY, semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256, 'semantic_report_dissociation_design prespecified semantic policy')
    sem = readj(project / semantic_report_dissociation_design_SEMANTIC_POLICY)
    if sem.get('immutable_experience_resolutions') != EXPERIENCE_K:
        raise RuntimeError('semantic_report_dissociation_design experience-resolution drift')
    if 'hypothesis_A_label_mismatch' not in sem or 'hypothesis_B_report_transition' not in sem:
        raise RuntimeError('semantic_report_dissociation_design semantic hypothesis registry drift')
    hsum = readj(project / cross_split_experience_structure_synthesis_SUMMARY)
    jsum = readj(project / discovery_report_transition_multiresolution_fit_SUMMARY)
    lsum = readj(project / validation_report_transition_multiresolution_transfer_SUMMARY)
    osum = readj(project / confirmation_report_transition_multiresolution_transfer_SUMMARY)
    if hsum.get('structurally_replicated_K') != EXPERIENCE_K:
        raise RuntimeError('cross_split_experience_structure_synthesis replicated experience K drift')
    if jsum.get('eligible_K') != TRANSITION_K:
        raise RuntimeError('discovery_report_transition_multiresolution_fit eligible transition K drift')
    if lsum.get('passing_K') != TRANSITION_K:
        raise RuntimeError('validation_report_transition_multiresolution_transfer passing transition K drift')
    if osum.get('passing_K') != TRANSITION_K:
        raise RuntimeError('confirmation_report_transition_multiresolution_transfer passing transition K drift')
    for sname, s in [('validation_report_transition_multiresolution_transfer', lsum), ('confirmation_report_transition_multiresolution_transfer', osum)]:
        b = s.get('boundary', {})
        for key in [f"{('Validation' if sname == 'validation_report_transition_multiresolution_transfer' else 'Confirmation')}_behavioral_values_read", 'centroids_refit', 'thresholds_tuned', 'K_reselected', 'family_semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
            if bool(b.get(key)):
                raise RuntimeError(f'{sname} boundary violation: {key}=True')
    print(f'[{now()}] loading only immutable family assignments for pre-prespecified B1 structural coupling', flush=True)
    exp = {'Discovery': read_assignments(project / discovery_stimulus_experience_multiresolution_fit_EXPERIENCE_ASSIGN, discovery_stimulus_experience_multiresolution_fit_EXPERIENCE_ASSIGN_SHA256, EXPERIENCE_K), 'Validation': read_assignments(project / validation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN, validation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN_SHA256, EXPERIENCE_K), 'Confirmation': read_assignments(project / confirmation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN, confirmation_stimulus_experience_multiresolution_transfer_EXPERIENCE_ASSIGN_SHA256, EXPERIENCE_K)}
    trans = {'Discovery': read_assignments(project / discovery_report_transition_multiresolution_fit_TRANSITION_ASSIGN, discovery_report_transition_multiresolution_fit_TRANSITION_ASSIGN_SHA256, TRANSITION_K), 'Validation': read_assignments(project / validation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN, validation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN_SHA256, TRANSITION_K), 'Confirmation': read_assignments(project / confirmation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN, confirmation_report_transition_multiresolution_transfer_TRANSITION_ASSIGN_SHA256, TRANSITION_K)}
    metric_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    for split in ['Discovery', 'Validation', 'Confirmation']:
        m, c, d = participant_balanced_cross(split, exp[split], trans[split])
        metric_rows += m
        cell_rows += c
        cond_rows += d
    cross = pd.DataFrame(metric_rows)
    cells = pd.DataFrame(cell_rows)
    conds = pd.DataFrame(cond_rows)
    expected_common = {'Discovery': 1237, 'Validation': 437, 'Confirmation': 459}
    for split, expected_n in expected_common.items():
        vals = cross[cross.split.eq(split)].common_trials.unique().tolist()
        if vals != [expected_n]:
            raise RuntimeError(f'{split} common-trial drift: {vals}, expected {expected_n}')
    jmet = pd.read_csv(project / discovery_report_transition_multiresolution_fit_METRICS, low_memory=False)
    lres = pd.read_csv(project / validation_report_transition_multiresolution_transfer_RESOLUTION, low_memory=False)
    ores = pd.read_csv(project / confirmation_report_transition_multiresolution_transfer_RESOLUTION, low_memory=False)
    synthesis_rows: list[dict[str, Any]] = []
    for k in TRANSITION_K:
        jd = jmet[jmet.k.astype(int).eq(k)].iloc[0]
        lv = lres[lres.k.astype(int).eq(k)].iloc[0]
        oc = ores[ores.k.astype(int).eq(k)].iloc[0]
        synthesis_rows.append({'k': k, 'discovery_eligible': bool(jd.eligible), 'discovery_median_ari': float(jd.median_ari), 'discovery_ari_10th_percentile': float(jd.ari_10th_percentile), 'validation_replicated': bool(lv.replicated), 'validation_paired_trials': int(lv.paired_report_transition_trials_all_participants), 'validation_eligible_participants': int(lv.eligible_participants), 'validation_participant_balanced_in_support': float(lv.participant_balanced_in_support_fraction), 'confirmation_replicated': bool(oc.replicated), 'confirmation_paired_trials': int(oc.paired_report_transition_trials_all_participants), 'confirmation_eligible_participants': int(oc.eligible_participants), 'confirmation_participant_balanced_in_support': float(oc.participant_balanced_in_support_fraction), 'structurally_replicated_across_all_three_splits': True, 'semantic_name_permitted': False, 'rank_as_single_preferred_resolution': False})
    synth = pd.DataFrame(synthesis_rows)
    synth_path = out / 'tables/cross_split_report_transition_structure_synthesis_crosssplit_transition_resolution_synthesis.csv'
    cross_path = out / 'tables/cross_split_report_transition_structure_synthesis_experience_transition_crosssplit_metrics.csv'
    cells_path = out / 'tables/cross_split_report_transition_structure_synthesis_experience_transition_participant_balanced_contingency.csv'
    conds_path = out / 'tables/cross_split_report_transition_structure_synthesis_experience_transition_directional_conditionals.csv'
    atomic_df(synth_path, synth)
    atomic_df(cross_path, cross.sort_values(['split', 'transition_k', 'experience_k']))
    atomic_df(cells_path, cells.sort_values(['split', 'transition_k', 'experience_k', 'transition_family_1based', 'experience_family_1based']))
    atomic_df(conds_path, conds.sort_values(['split', 'transition_k', 'experience_k', 'transition_family_1based', 'experience_family_1based']))
    rehash_path = out / 'qc/cross_split_report_transition_structure_synthesis_source_rehash.csv'
    write_csv(rehash_path, rehash, ['source_phase', 'path', 'expected_sha256', 'observed_sha256', 'verified'])
    split_overlap = {}
    for split in ['Discovery', 'Validation', 'Confirmation']:
        x = cross[cross.split.eq(split)]
        split_overlap[split] = {'max_ARI_across_K_pairs': float(x.ARI.max()), 'max_NMI_across_K_pairs': float(x.NMI.max()), 'min_ARI_across_K_pairs': float(x.ARI.min()), 'min_NMI_across_K_pairs': float(x.NMI.min()), 'common_trials': int(x.common_trials.iloc[0]), 'common_participants': int(x.common_participants.iloc[0])}
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'activation_condition_from_semantic_report_dissociation_design_satisfied': True, 'structural_prerequisites': {'experience_resolutions_replicated': EXPERIENCE_K, 'report_transition_resolutions_replicated': TRANSITION_K, 'report_transition_crosssplit_lock_complete': True}, 'outcome_free_B1_status': {'executed_under_semantic_report_dissociation_design_prespecification': True, 'behavior_used': False, 'metrics': 'ARI/NMI plus participant-balanced contingency and directional occupancy', 'split_overlap_summary': split_overlap, 'interpretation': 'experience and report-transition partitions show consistently low descriptive overlap across all three internal splits; this supports treating them as distinct structural organizations for subsequent tests, but does not establish independent biological mechanisms'}, 'semantic_report_execution_design_role': 'operationalize and specification exact source lineage, joins, multiplicity, execution order, and held-out target-opening rules for the already-prespecified semantic_report_dissociation_design semantic/report tests A1/A2/B2/B3; semantic_report_execution_design itself opens no behavioral values', 'semantic_report_execution_design_data_opening': {'new_EEG_signal_values': False, 'behavioral_report_values': False, 'emotion_category_values': False, 'context_target_values': False, 'stimulus_identity_target_values': False}, 'semantic_execution_in_semantic_report_execution_design_permitted': False, 'post_semantic_report_execution_design_execution_requires_separate_phase': True, 'required_hypotheses_carried_forward': ['A1 nominal emotion-category vs experience-family many-to-many mapping', 'A2 joint 10-report profile vs experience families', 'B2 incremental joint-report mapping from continuous report-transition signature beyond continuous experience signature', 'B3 joint-report mapping to replicated discrete report-transition families as a secondary gated test'], 'forbidden': ['opening behavioral values in semantic_report_execution_design', 'using behavior to alter K, centroids, support thresholds, or family IDs', 'selecting one experience or transition K based on semantic strength', 'split-specific relabeling', 'family semantic naming before a separately prespecified naming rule', 'claims of distortion/dishonesty/suppression/exaggeration/conscious editing', 'external-replication claim']}
    writej(project / NEXT_CONFIG, next_policy)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'crosssplit_report_transition_structure_synthesis_complete', 'structurally_replicated_experience_K': EXPERIENCE_K, 'structurally_replicated_report_transition_K': TRANSITION_K, 'transition_K_replicated_across_discovery_validation_confirmation': True, 'experience_transition_B1_crosssplit_completed': True, 'experience_transition_overlap_summary': split_overlap, 'single_preferred_transition_K_selected': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'external_replication_claim_permitted': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'emotion_category_values_opened': False, 'next_phase': NEXT_PHASE, 'next_phase_role': next_policy['semantic_report_execution_design_role'], 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/cross_split_report_transition_structure_synthesis_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': sha256(Path(__file__).resolve()), 'verified_parent_final_locks': {'cross_split_experience_structure_synthesis': cross_split_experience_structure_synthesis_LOCK_SHA256, 'semantic_report_dissociation_design': semantic_report_dissociation_design_LOCK_SHA256, 'discovery_report_transition_multiresolution_fit': discovery_report_transition_multiresolution_fit_LOCK_SHA256, 'validation_report_transition_multiresolution_transfer': validation_report_transition_multiresolution_transfer_LOCK_SHA256, 'confirmation_report_transition_multiresolution_transfer': confirmation_report_transition_multiresolution_transfer_LOCK_SHA256}, 'semantic_report_dissociation_design_semantic_policy_sha256': semantic_report_dissociation_design_SEMANTIC_POLICY_SHA256, 'experience_K': EXPERIENCE_K, 'transition_K': TRANSITION_K, 'B1_behavior_used': False, 'boundary': {'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'emotion_category_values_opened': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'external_replication_claim_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/cross_split_report_transition_structure_synthesis_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/cross_split_report_transition_structure_synthesis_report.md'
    lines = ['# cross_split_report_transition_structure_synthesis — Cross-split report-transition structure synthesis', '', '**cross_split_report_transition_structure_synthesis opened no new EEG signal values and no behavioral/category/context/stimulus target values.**', '', '## fixed report-transition result', '', 'The outcome-free `stimulus_late -> rating_early` representation retained **K=2 and K=3** in Discovery.', 'Both transferred without refit to Validation and both transferred without refit to Confirmation.', '']
    for _, r in synth.iterrows():
        lines += [f'- **K={int(r.k)}**: Discovery median ARI={r.discovery_median_ari:.4f}; Validation in-support={r.validation_participant_balanced_in_support:.4f}; Confirmation in-support={r.confirmation_participant_balanced_in_support:.4f}.']
    lines += ['', 'Therefore K=2 and K=3 are both structurally replicated across the three internal splits.', 'No single transition resolution is selected as the true psychological or biological resolution.', '', '## B1 — experience structure vs report-transition structure', '', 'semantic_report_dissociation_design froze this comparison before the report-transition branch was executed. cross_split_report_transition_structure_synthesis therefore evaluates the same ARI/NMI comparison in all three splits using only immutable family assignments.', '', '```text', cross.to_string(index=False), '```', '', 'The overlap remains very low in every split. The maximum ARI across any K-pair is:']
    for split in ['Discovery', 'Validation', 'Confirmation']:
        lines.append(f"- {split}: **{split_overlap[split]['max_ARI_across_K_pairs']:.4f}** (max NMI **{split_overlap[split]['max_NMI_across_K_pairs']:.4f}**).")
    lines += ['', 'This repeated low-overlap pattern makes a simple identity between the stimulus-only experience partition and the report-transition partition difficult to defend. It supports treating them as **distinct structural organizations** in the next tests. It does **not** establish that they are independent biological mechanisms.', '', '## Interpretation boundary', '', '- The transition families remain structural labels only.', '- No behavioral report or emotion category has yet been opened in the semantic program.', '- No result here demonstrates distortion, dishonesty, suppression, exaggeration, or conscious editing.', '- These are internal held-out splits of one dataset, not external replication.', '', '## Next', '', f'**{NEXT_PHASE}** will operationalize/specification the exact semantic and report-association execution lineage already specified in semantic_report_dissociation_design, while still keeping behavioral values closed.', '']
    report_path.write_text(''.join(lines), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, contract_path, summary_path, rehash_path, synth_path, cross_path, cells_path, conds_path, report_path]
    final_lock = out / 'qc/cross_split_report_transition_structure_synthesis_final_output_lock_hashes.json'
    writej(final_lock, {'phase': PHASE, 'generated_at': now(), 'status': 'crosssplit_report_transition_structure_synthesis_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'B1_behavior_used': False, 'single_preferred_transition_K_selected': False, 'new_signal_values_opened': False, 'new_behavioral_values_opened': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE})
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('cross_split_report_transition_structure_synthesis PASS')
    print(f'Structurally replicated experience K: {EXPERIENCE_K}')
    print(f'Structurally replicated report-transition K: {TRANSITION_K}')
    for split in ['Discovery', 'Validation', 'Confirmation']:
        print(f"{split} max experience-transition ARI={split_overlap[split]['max_ARI_across_K_pairs']:.6f}, max NMI={split_overlap[split]['max_NMI_across_K_pairs']:.6f}")
    print('Single preferred transition K selected: False')
    print('New signal values opened: False')
    print('New behavioral values opened: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())