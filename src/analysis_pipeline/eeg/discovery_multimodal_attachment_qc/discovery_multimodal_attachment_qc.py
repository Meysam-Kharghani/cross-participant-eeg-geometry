# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase discovery_multimodal_attachment_qc — Discovery-only EEG × physiology attachment and QC.
No M1/M2/M3 inference is performed. Held-out biological feature values remain unopened."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
PHASE = 'discovery_multimodal_attachment_qc'
NEXT = 'discovery_multimodal_inference'
PREFIX = 'discovery_multimodal_attachment_qc'
multimodal_physio_design_LOCK = Path('outputs/qc/multimodal_physio_design_final_output_lock_hashes.json')
multimodal_physio_design_LOCK_SHA = '2e2792d21f49620be426b1a61c70dcd4d2655b03c952c86dd015a58183a25ac2'
multimodal_physio_design_POLICY = Path('analysis_specifications/S044_multimodal_physio.json')
multimodal_physio_design_POLICY_SHA = '2ef19f67a6a5d8f1faf1e6869f5c6accfec0b5c83afa72f65440f62249a8ea8b'
multimodal_physio_design_NEXT = Path('analysis_specifications/S043_multimodal_physio_discovery_attachment_qc.json')
multimodal_physio_design_NEXT_SHA = '5a4d2fe67ad38e7ef8dcb4021603699b93be8ab62fdef96863f15507f3bc6123'
ASSIGN = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv')
ASSIGN_SHA = '666bb5b242cb3acee6d13dfc99c257001bf840a5c941e9b5fed359793ade6bca'
LINEAGE = Path('outputs/tables/mapping_design_trial_mapping_manifest.csv')
LINEAGE_SHA = '6f130a2431e3f0c4fc881c2e8e20ea99290459fc97b6c6695da746ac38a82cab'
PHYS_SHA = '87dfdab02491ace2838b94f7e68eb81203254fb953c75be596a32a4a776b236e'
CANON_SHA = '70eff894db94ed51d65d103392004162c839bfd3f2850dd3fc15ff3685fd28a6'
BIO = ['gsr_scr_mean', 'gsr_scr_sd', 'gsr_scr_median', 'gsr_scr_min', 'gsr_scr_max', 'gsr_scr_sum', 'gsr_scr_iqr', 'gsr_scr_early_mean', 'gsr_scr_late_mean', 'gsr_scr_late_minus_early', 'gsr_scr_slope', 'gsr_scr_auc', 'gsr_scl_mean', 'gsr_scl_sd', 'gsr_scl_median', 'gsr_scl_min', 'gsr_scl_max', 'gsr_scl_sum', 'gsr_scl_iqr', 'gsr_scl_early_mean', 'gsr_scl_late_mean', 'gsr_scl_late_minus_early', 'gsr_scl_slope', 'gsr_scl_auc', 'gsr_iscr_mean', 'gsr_iscr_sd', 'gsr_iscr_median', 'gsr_iscr_min', 'gsr_iscr_max', 'gsr_iscr_sum', 'gsr_iscr_iqr', 'gsr_iscr_early_mean', 'gsr_iscr_late_mean', 'gsr_iscr_late_minus_early', 'gsr_iscr_slope', 'gsr_iscr_auc', 'hr_hr_median', 'hr_hr_iqr', 'hr_hr_skew', 'hr_hr_kurt']

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def jread(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def atom_json(p: Path, v: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    os.replace(t, p)

def normsub(v):
    m = re.fullmatch('(?:sub[-_ ]?)?(\\d+)', str(v).strip(), re.I)
    return f'sub-{int(m.group(1)):02d}' if m else str(v).strip()

def normctx(v):
    s = str(v).strip().lower()
    if s in {'ima', 'imagery', 'image', 'imagination', 'ses-ima'}:
        return 'imagery'
    if s in {'vid', 'video', 'ses-vid'}:
        return 'video'
    return s

def intish(s, label):
    x = pd.to_numeric(s, errors='coerce')
    if x.isna().any() or not np.allclose(x.to_numpy(float), np.round(x.to_numpy(float))):
        raise RuntimeError(f'{label} contains non-integer/missing values')
    return x.astype('int64')

def outputs(project):
    return {'rehash': project / 'outputs/qc/discovery_multimodal_attachment_qc_source_rehash_audit.csv', 'join_audit': project / 'outputs/qc/discovery_multimodal_attachment_qc_one_to_one_join_audit.csv', 'bridge_addendum': project / 'config/discovery_multimodal_attachment_qc_bridge_correction_addendum.json', 'bridge_unmatched': project / 'outputs/qc/discovery_multimodal_attachment_qc_bridge_unmatched_rows.csv', 'manifest': project / 'outputs/tables/discovery_multimodal_attachment_qc_discovery_multimodal_manifest.csv', 'coverage': project / 'outputs/tables/discovery_multimodal_attachment_qc_participant_coverage.csv', 'missing': project / 'outputs/tables/discovery_multimodal_attachment_qc_feature_missingness.csv', 'roster': project / 'outputs/tables/discovery_multimodal_attachment_qc_prespecified_eligible_roster.csv', 'next': project / 'analysis_specifications/S018_discovery_multimodal_attachment_qc.json', 'summary': project / 'outputs/qc/discovery_multimodal_attachment_qc_attachment_summary.json', 'report': project / 'outputs/reports/discovery_multimodal_attachment_qc_discovery_multimodal_attachment_qc.md', 'contract': project / 'config/discovery_multimodal_attachment_qc_attachment_contract.json', 'lock': project / 'outputs/qc/discovery_multimodal_attachment_qc_final_output_lock_hashes.json'}

def verify(rows, label, p, exp=None):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    obs = sha(p)
    ok = exp is None or obs == exp
    rows.append({'artifact': label, 'path': str(p), 'sha256': obs, 'expected_sha256': exp or obs, 'passed': ok})
    if not ok:
        raise RuntimeError(f'{label} SHA mismatch: {obs} != {exp}')
    return obs

def discovery_physio_rows(src: Path, subjects: set[str]):
    header = list(pd.read_csv(src, nrows=0, low_memory=False).columns)
    need = {'subject_id', 'context', 'trial_index', *BIO}
    if not need.issubset(header):
        raise RuntimeError('Physiology source missing prespecified columns: ' + str(sorted(need - set(header))))
    sub_only = pd.read_csv(src, usecols=['subject_id'], dtype={'subject_id': 'string'}, low_memory=False)
    norm = sub_only.subject_id.map(normsub)
    keep0 = np.flatnonzero(norm.isin(subjects).to_numpy())
    keep_lines = set((keep0 + 1).tolist())
    if not keep_lines:
        raise RuntimeError('No Discovery physiology rows found')

    def skipper(i):
        return i > 0 and i not in keep_lines
    d = pd.read_csv(src, usecols=['subject_id', 'context', 'trial_index', *BIO], skiprows=skipper, low_memory=False)
    d['subject_id'] = d.subject_id.map(normsub)
    d['context'] = d.context.map(normctx)
    d['trial_index'] = intish(d.trial_index, 'physiology trial_index')
    if not set(d.subject_id).issubset(subjects):
        raise RuntimeError('Held-out physiology rows entered Discovery table')
    return (d, len(sub_only), len(keep_lines))

def self_test():
    assert len(BIO) == 40 and len(set(BIO)) == 40 and (normsub('7') == 'sub-07') and (normctx('ses-vid') == 'video')
    assert intish(pd.Series([1.0, 2.0, 3.0]), 'x').tolist() == [1, 2, 3]
    a = pd.DataFrame({'subject_id': ['sub-01', 'sub-01'], 'trial_number': [7, 8]})
    lin = pd.DataFrame({'subject_id': ['sub-01', 'sub-01'], 'trial_number': [7, 8], 'context': ['vid', 'ima'], 'stimulus_id': ['joy4', 'sad4']})
    can = pd.DataFrame({'subject_id': ['sub-01', 'sub-01'], 'context': ['video', 'imagery'], 'stimulus_id': ['joy4', 'sad4'], 'context_trial_index': [13, 1]})
    lin['context'] = lin.context.map(normctx)
    can['context'] = can.context.map(normctx)
    x = a.merge(lin, on=['subject_id', 'trial_number'], validate='one_to_one').merge(can, on=['subject_id', 'context', 'stimulus_id'], validate='many_to_one')
    assert x.context_trial_index.tolist() == [13, 1]
    print('discovery_multimodal_attachment_qc self-test: PASS')

def main():
    a = args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    dataset = a.dataset.resolve()
    out = outputs(project)
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not a.force):
        raise RuntimeError('discovery_multimodal_attachment_qc already exists; use --force only for identical rerun')
    rows = []
    verify(rows, 'multimodal_physio_design_final_lock', project / multimodal_physio_design_LOCK, multimodal_physio_design_LOCK_SHA)
    verify(rows, 'multimodal_physio_design_multimodal_policy', project / multimodal_physio_design_POLICY, multimodal_physio_design_POLICY_SHA)
    verify(rows, 'multimodal_physio_design_next_policy', project / multimodal_physio_design_NEXT, multimodal_physio_design_NEXT_SHA)
    nx = jread(project / multimodal_physio_design_NEXT)
    if nx.get('next_phase') != PHASE or not nx.get('next_phase_permitted') or (not nx.get('inference_forbidden')):
        raise RuntimeError('multimodal_physio_design did not authorize inference-free discovery_multimodal_attachment_qc')
    pol = jread(project / multimodal_physio_design_POLICY)
    assign_path = project / ASSIGN
    verify(rows, 'Discovery_fixed_EEG_assignments', assign_path, ASSIGN_SHA)
    A = pd.read_csv(assign_path, low_memory=False)
    req = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'selected_shared_k', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'trajectory_alignment_weight']
    if not set(req).issubset(A.columns):
        raise RuntimeError('Discovery assignment schema changed')
    A = A[req].copy()
    A.subject_id = A.subject_id.map(normsub)
    A['trial_number'] = intish(A.trial_number, 'EEG trial_number')
    A['trajectory_family_1based'] = intish(A.trajectory_family_1based, 'family')
    if len(A) != 1237 or A.subject_id.nunique() != 31 or set(A.trajectory_family_1based) != {1, 2} or (not A.selected_shared_k.eq(2).all()) or A.duplicated(['subject_id', 'trial_number']).any():
        raise RuntimeError('fixed Discovery assignment boundary changed')
    disc = set(A.subject_id.unique())
    phys = Path(pol['physiology_source']['corrected_table'])
    canon = Path(pol['physiology_source']['canonical_table'])
    if not phys.is_file():
        alt = dataset / 'project/physiology_features.csv'
        phys = alt if alt.is_file() else phys
    if not canon.is_file():
        alt = dataset / 'project/canonical_behavioral_table.csv'
        canon = alt if alt.is_file() else canon
    verify(rows, 'corrected_trial_level_physio_source', phys, PHYS_SHA)
    verify(rows, 'canonical_trial_bridge_source', canon, CANON_SHA)
    lineage_path = project / LINEAGE
    verify(rows, 'mapping_design_fixed_trial_lineage', lineage_path, LINEAGE_SHA)
    L = pd.read_csv(lineage_path, usecols=['subject_id', 'trial_number', 'context', 'stimulus_id'], low_memory=False)
    L['subject_id'] = L.subject_id.map(normsub)
    L['trial_number'] = intish(L.trial_number, 'mapping_design trial_number')
    L['context'] = L.context.map(normctx)
    if len(L) != 1237 or L.subject_id.nunique() != 31 or L.duplicated(['subject_id', 'trial_number']).any():
        raise RuntimeError('fixed mapping_design lineage boundary changed')
    if set(zip(A.subject_id, A.trial_number)) != set(zip(L.subject_id, L.trial_number)):
        raise RuntimeError('mapping_design lineage keys do not exactly match fixed Discovery EEG assignment keys')
    AL = A.merge(L, on=['subject_id', 'trial_number'], how='left', validate='one_to_one', indicator=True)
    if not AL._merge.eq('both').all():
        raise RuntimeError('fixed EEG assignments failed exact mapping_design lineage attachment')
    AL = AL.drop(columns=['_merge'])
    C = pd.read_csv(canon, usecols=['subject_id', 'context', 'context_trial_index', 'stimulus_id'], low_memory=False)
    C['subject_id'] = C.subject_id.map(normsub)
    C['context'] = C.context.map(normctx)
    C['context_trial_index'] = intish(C.context_trial_index, 'canonical context_trial_index')
    C['stimulus_id'] = C.stimulus_id.astype('string')
    CS = C.dropna(subset=['stimulus_id']).copy()
    CS['stimulus_id'] = CS.stimulus_id.astype(str).str.strip()
    CS = CS[CS.stimulus_id.ne('')].copy()
    if CS.duplicated(['subject_id', 'context', 'stimulus_id']).any():
        raise RuntimeError('Canonical stimulus bridge has duplicate subject/context/stimulus keys')
    CA = AL.merge(CS[['subject_id', 'context', 'stimulus_id', 'context_trial_index']], on=['subject_id', 'context', 'stimulus_id'], how='left', validate='many_to_one', indicator=True)
    CA['canonical_stimulus_bridge_attached'] = CA._merge.eq('both')
    CA = CA.drop(columns=['_merge']).rename(columns={'context_trial_index': 'trial_index'})
    P, total_phys_rows, disc_phys_rows = discovery_physio_rows(phys, disc)
    phys_subjects = set(P.subject_id.unique())
    unresolved = CA[~CA.canonical_stimulus_bridge_attached].copy()
    unresolved['has_corrected_physio_subject'] = unresolved.subject_id.isin(phys_subjects)
    unresolved[['subject_id', 'trial_number', 'context', 'stimulus_id', 'has_corrected_physio_subject']].to_csv(out['bridge_unmatched'], index=False)
    bad_unresolved = unresolved[unresolved.has_corrected_physio_subject]
    if len(bad_unresolved):
        preview = bad_unresolved[['subject_id', 'trial_number', 'context', 'stimulus_id']].head(20).to_dict('records')
        raise RuntimeError(f'{len(bad_unresolved)} EEG rows from physiology-bearing participants lack corrected stimulus bridge; preview={preview}')
    if P.duplicated(['subject_id', 'context', 'trial_index']).any():
        raise RuntimeError('Physiology source duplicate join keys')
    CA = CA.reset_index(drop=True)
    CA['_eeg_row_id'] = np.arange(len(CA), dtype=np.int64)
    CAB = CA[CA.canonical_stimulus_bridge_attached].copy()
    if CAB.duplicated(['subject_id', 'context', 'trial_index']).any():
        bad = CAB[CAB.duplicated(['subject_id', 'context', 'trial_index'], keep=False)][['subject_id', 'trial_number', 'context', 'stimulus_id', 'trial_index']].head(20).to_dict('records')
        raise RuntimeError(f'Corrected EEG bridge is not one-to-one on physiology keys; preview={bad}')
    JB = CAB.merge(P, on=['subject_id', 'context', 'trial_index'], how='left', validate='one_to_one', indicator=True)
    JB['physiology_row_attached'] = JB._merge.eq('both')
    JB = JB.drop(columns=['_merge'])
    carry = ['_eeg_row_id', 'physiology_row_attached', *BIO]
    J = CA.merge(JB[carry], on='_eeg_row_id', how='left', validate='one_to_one')
    J['physiology_row_attached'] = J['physiology_row_attached'].eq(True)
    joined = J['physiology_row_attached']
    J = J.drop(columns=['_eeg_row_id'])
    J['normalized_trial_order'] = J.groupby('subject_id')['trial_number'].transform(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0)
    base = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'context', 'trial_index', 'normalized_trial_order', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'trajectory_alignment_weight', 'physiology_row_attached']
    J = J[base + BIO].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    J.to_csv(out['manifest'], index=False)
    cov = []
    for sub, g in J.groupby('subject_id', sort=True):
        gj = g[g.physiology_row_attached]
        fam = gj.trajectory_family_1based.value_counts()
        n1 = int(fam.get(1, 0))
        n2 = int(fam.get(2, 0))
        n = len(gj)
        eeg = len(g)
        cov.append({'subject_id': sub, 'fixed_EEG_trajectory_trials': eeg, 'joined_multimodal_trials': n, 'join_coverage': n / eeg if eeg else np.nan, 'family1_joined_trials': n1, 'family2_joined_trials': n2, 'both_families_present': n1 > 0 and n2 > 0, 'eligible_M1_M2': n >= 16 and n1 >= 4 and (n2 >= 4), 'eligible_M3': n >= 16, 'missing_biological_cells': int(gj[BIO].isna().sum().sum()), 'complete_case_trials_40D': int(gj[BIO].notna().all(axis=1).sum())})
    COV = pd.DataFrame(cov)
    COV.to_csv(out['coverage'], index=False)
    JP = J[J.physiology_row_attached]
    miss = []
    for f in BIO:
        arr = pd.to_numeric(JP[f], errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(arr)
        tmp = JP[['subject_id']].copy()
        tmp['_ok'] = ok
        miss.append({'feature': f, 'modality': 'HR' if f.startswith('hr_') else 'GSR', 'attached_trials': len(JP), 'missing_or_nonfinite_trials': int((~ok).sum()), 'missing_or_nonfinite_fraction': float((~ok).mean()) if len(JP) else np.nan, 'participants_with_any_finite': int(tmp.groupby('subject_id')._ok.any().sum())})
    pd.DataFrame(miss).to_csv(out['missing'], index=False)
    R = COV[['subject_id', 'joined_multimodal_trials', 'family1_joined_trials', 'family2_joined_trials', 'eligible_M1_M2', 'eligible_M3']].copy()
    R.to_csv(out['roster'], index=False)
    pd.DataFrame([{'stage': 'fixed_EEG_assignments', 'rows_in': len(A), 'rows_matched': len(A), 'rows_unmatched': 0, 'unique_subjects': A.subject_id.nunique(), 'one_to_one': True}, {'stage': 'EEG_to_mapping_design_fixed_lineage', 'rows_in': len(A), 'rows_matched': len(AL), 'rows_unmatched': 0, 'unique_subjects': AL.subject_id.nunique(), 'one_to_one': True}, {'stage': 'lineage_to_canonical_by_subject_context_stimulus', 'rows_in': len(AL), 'rows_matched': int(CA.canonical_stimulus_bridge_attached.sum()), 'rows_unmatched': int((~CA.canonical_stimulus_bridge_attached).sum()), 'unique_subjects': int(CA.loc[CA.canonical_stimulus_bridge_attached, 'subject_id'].nunique()), 'one_to_one': True}, {'stage': 'canonical_to_Discovery_physio', 'rows_in': len(CA), 'rows_matched': int(joined.sum()), 'rows_unmatched': int((~joined).sum()), 'unique_subjects': int(JP.subject_id.nunique()), 'one_to_one': True}]).to_csv(out['join_audit'], index=False)
    pd.DataFrame(rows).to_csv(out['rehash'], index=False)
    bridge_addendum = {'phase': PHASE, 'generated_at': now(), 'status': 'lineage_correction_before_multimodal_inference', 'supersedes_multimodal_physio_design_bridge_fragment': 'EEG subject_id+trial_number -> canonical subject_id+trial_number_global', 'reason': 'historical canonical trial_number_global is a context-block canonical index and is not the acquisition/behavioral trial identifier used by fixed EEG trajectories', 'corrected_bridge': ['fixed EEG subject_id+trial_number', 'mapping_design fixed lineage carries context+stimulus_id', 'canonical subject_id+context+stimulus_id carries context_trial_index as trial_index', 'physiology subject_id+context+trial_index'], 'hypotheses_changed': False, 'family_assignments_changed': False, 'physiology_features_changed': False, 'heldout_values_opened': False, 'unresolved_bridge_rows_allowed_only_when_subject_has_zero_corrected_physio_rows': True}
    atom_json(out['bridge_addendum'], bridge_addendum)
    nextp = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT, 'next_phase_permitted': bool(R.eligible_M3.any()), 'next_phase_role': 'Discovery-only prespecified M1-M3 inference; held-out physiology remains closed', 'input_manifest': str(out['manifest']), 'input_manifest_sha256': sha(out['manifest']), 'eligible_roster_sha256': sha(out['roster']), 'bridge_addendum_sha256': sha(out['bridge_addendum']), 'heldout_physio_values_opened': False, 'inference_performed_in_discovery_multimodal_attachment_qc': False, 'normalized_trial_order_formula': 'within participant: (trial_number-min)/(max-min), 0 if denominator=0', 'M1_M2_eligibility': 'joined>=16 AND Family1>=4 AND Family2>=4', 'M3_eligibility': 'joined>=16', 'physiology_rows_with_missing_BIO': 'retain; training/analysis-set median imputation only', 'forbidden': ['opening Validation physiology values', 'opening Confirmation physiology values', 'changing EEG family assignments', 'family/outcome-driven feature selection', 'dropping trials solely for partial BIO missingness']}
    atom_json(out['next'], nextp)
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_multimodal_attachment_qc_complete', 'counts': {'fixed_EEG_trials': len(A), 'Discovery_EEG_participants': A.subject_id.nunique(), 'mapping_design_lineage_matches': len(AL), 'canonical_stimulus_bridge_matches': int(CA.canonical_stimulus_bridge_attached.sum()), 'canonical_stimulus_bridge_unmatched_no_physio_subjects': int((~CA.canonical_stimulus_bridge_attached).sum()), 'joined_multimodal_trials': int(joined.sum()), 'unmatched_EEG_trials': int((~joined).sum()), 'joined_participants': int(JP.subject_id.nunique()), 'eligible_M1_M2_participants': int(R.eligible_M1_M2.sum()), 'eligible_M3_participants': int(R.eligible_M3.sum()), 'physiology_source_total_rows_key_scan_only': int(total_phys_rows), 'Discovery_physio_rows_loaded_with_BIO': int(disc_phys_rows), 'biological_features': 40}, 'boundary': {'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'multimodal_inference_performed': False, 'discovery_multimodal_inference_permitted': bool(nextp['next_phase_permitted']), 'multimodal_physio_design_bridge_fragment_superseded_by_lineage_correction': True}}
    atom_json(out['summary'], summary)
    out['report'].write_text(f'# Phase {PHASE} — Discovery EEG × physiology attachment/QC\n\nStatus: **discovery_multimodal_attachment_qc_complete**\n\n## Lineage correction\nThe multimodal_physio_design bridge fragment `EEG trial_number -> canonical trial_number_global` was superseded before any multimodal inference because historical `trial_number_global` is a context-block canonical index, not the behavioral/acquisition trial identifier used by the fixed EEG trajectories. The corrected bridge is `EEG subject+trial -> fixed mapping_design context+stimulus -> canonical context_trial_index -> physiology context+trial_index`.\n\nfixed EEG trajectories: **{len(A)}** across **{A.subject_id.nunique()}** participants.\nCanonical stimulus bridge matches: **{int(CA.canonical_stimulus_bridge_attached.sum())}/{len(CA)}**. Any unresolved bridge rows belong only to participants with zero corrected physiology rows.\nJoined EEG×physiology trials: **{int(joined.sum())}**; unmatched: **{int((~joined).sum())}**.\nEligible M1/M2 participants: **{int(R.eligible_M1_M2.sum())}**; eligible M3 participants: **{int(R.eligible_M3.sum())}**.\n\nNo M1/M2/M3 inference was performed. Only Discovery biological values were materialized; Validation and Confirmation physiology remain closed.', encoding='utf-8')
    keys = ['rehash', 'join_audit', 'bridge_addendum', 'bridge_unmatched', 'manifest', 'coverage', 'missing', 'roster', 'next', 'summary', 'report']
    hashes = {k: sha(out[k]) for k in keys}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_multimodal_attachment_qc_complete', 'source_hashes': {'multimodal_physio_design_final_lock': multimodal_physio_design_LOCK_SHA, 'Discovery_assignments': ASSIGN_SHA, 'mapping_design_fixed_trial_lineage': LINEAGE_SHA, 'corrected_physio_source': PHYS_SHA, 'canonical_bridge_source': CANON_SHA}, 'output_hashes': hashes, 'boundary': summary['boundary']}
    atom_json(out['contract'], contract)
    hashes['contract'] = sha(out['contract'])
    atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': hashes, 'discovery_multimodal_inference_permitted': bool(nextp['next_phase_permitted']), 'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'multimodal_inference_performed': False})
    stamp = tag()
    hroot = project / ''
    hroot.mkdir(parents=True, exist_ok=True)
    z = hroot / f'{PREFIX}_discovery_multimodal_attachment_qc_{stamp}'
    s = sha(z)
    Path(str(z) + '.sha256').write_text(f'{s}  {z.name}', encoding='utf-8')
    print(f'discovery_multimodal_attachment_qc complete: {z}')
    print(f'SHA-256: {s}')
    print(json.dumps(summary['counts'], indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())