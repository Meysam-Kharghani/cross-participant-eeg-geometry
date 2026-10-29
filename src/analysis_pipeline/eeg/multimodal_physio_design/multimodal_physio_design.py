# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase multimodal_physio_design — outcome-free EEG × peripheral-physiology integration analysis specification.

This phase performs no multimodal association/inference and does not inspect
Validation/Confirmation physiology values. It verifies the completed analysis_stage EEG
geometry branch, re-verifies the prespecified P0.2 trial-level physiology source
contract, specifications source/header hashes and the exact EEG→canonical→physiology
bridge, and serializes the Discovery-only multimodal hypotheses, preprocessing,
eligibility, multiplicity, robustness and held-out opening gates."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
PHASE = 'multimodal_physio_design'
NEXT = 'discovery_multimodal_attachment_qc'
PREFIX = 'multimodal_physio_design'
three_cohort_structural_synthesis_LOCK = Path('outputs/qc/three_cohort_structural_synthesis_final_output_lock_hashes.json')
three_cohort_structural_synthesis_LOCK_SHA = '3c0b3b2f0998672a3f6521f7b32489d301b3f06561a7e1855b9fbf8f27b5e216'
three_cohort_structural_synthesis_NEXT = Path('analysis_specifications/S070_three_cohort_structural_synthesis_multimodal_physio.json')
three_cohort_structural_synthesis_NEXT_SHA = 'd8cecf021906888c76a7683a14533be6e182653d56aba7a95c891c9292c956ce'
P02_CONTRACT_REL = Path('outputs/qc/physiology_source_contract.v1.json')
P02_CONTRACT_SHA = '1791ed36e26eaa4a9def67ed4876e0ea866b0d9e031c5af5a770a68dda17e4c7'
CANONICAL_REL = Path('project/canonical_behavioral_table.csv')
ASSIGNMENTS = {'Discovery': (Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv'), '666bb5b242cb3acee6d13dfc99c257001bf840a5c941e9b5fed359793ade6bca'), 'Validation': (Path('outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv'), '4b30434bfdec1bcaaf79887f420a793335a39bc4d42ecd6e8b75674d72c761c9'), 'Confirmation': (Path('outputs/tables/confirmation_fixed_assignment_confirmation_trajectory_assignments.csv'), '0cb63321efa243b2e95e6df15bfa71e21b75a5e8259dd997f46a59c7b8a1b167')}
BIO = ['gsr_scr_mean', 'gsr_scr_sd', 'gsr_scr_median', 'gsr_scr_min', 'gsr_scr_max', 'gsr_scr_sum', 'gsr_scr_iqr', 'gsr_scr_early_mean', 'gsr_scr_late_mean', 'gsr_scr_late_minus_early', 'gsr_scr_slope', 'gsr_scr_auc', 'gsr_scl_mean', 'gsr_scl_sd', 'gsr_scl_median', 'gsr_scl_min', 'gsr_scl_max', 'gsr_scl_sum', 'gsr_scl_iqr', 'gsr_scl_early_mean', 'gsr_scl_late_mean', 'gsr_scl_late_minus_early', 'gsr_scl_slope', 'gsr_scl_auc', 'gsr_iscr_mean', 'gsr_iscr_sd', 'gsr_iscr_median', 'gsr_iscr_min', 'gsr_iscr_max', 'gsr_iscr_sum', 'gsr_iscr_iqr', 'gsr_iscr_early_mean', 'gsr_iscr_late_mean', 'gsr_iscr_late_minus_early', 'gsr_iscr_slope', 'gsr_iscr_auc', 'hr_hr_median', 'hr_hr_iqr', 'hr_hr_skew', 'hr_hr_kurt']
BANNED = ['gsr_source_trial_label', 'gsr_n_windows', 'gsr_expected_10_windows', 'hr_segment', 'hr_n_rows_for_segment']
GEOM = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']

def parse_args():
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

def locate_p02(project: Path):
    direct = project / P02_CONTRACT_REL
    if direct.is_file():
        return (direct, 'project_output')
    hz = sorted((project / '').glob(''))
    for z in reversed(hz):
        pass
    raise FileNotFoundError('prespecified P0.2 physiology source contract not found')

def source_path(contract, dataset):
    p = Path(contract['source']['corrected_table'])
    if p.is_file():
        return p
    alt = dataset / 'project/physiology_features.csv'
    if alt.is_file():
        return alt
    raise FileNotFoundError(f'Corrected physiology source not found: {p} or {alt}')

def outputs(project):
    return {'source': project / 'outputs/qc/multimodal_physio_design_source_lineage_audit.csv', 'features': project / 'outputs/tables/multimodal_physio_design_prespecified_physio_feature_schema.csv', 'hyp': project / 'outputs/tables/multimodal_physio_design_prespecified_multimodal_hypotheses.csv', 'gates': project / 'outputs/tables/multimodal_physio_design_prespecified_multimodal_gate_manifest.csv', 'policy': project / 'analysis_specifications/S044_multimodal_physio.json', 'next': project / 'analysis_specifications/S043_multimodal_physio_discovery_attachment_qc.json', 'summary': project / 'outputs/qc/multimodal_physio_design_multimodal_physio_design_summary.json', 'report': project / 'outputs/reports/multimodal_physio_design_multimodal_physio_design_specification_report.md', 'contract': project / 'config/multimodal_physio_design_multimodal_physio_design_contract.json', 'lock': project / 'outputs/qc/multimodal_physio_design_final_output_lock_hashes.json'}

def self_test():
    assert len(BIO) == 40 and len(set(BIO)) == 40 and (len(GEOM) == 6)
    assert all((x.startswith(('gsr_', 'hr_')) for x in BIO))
    assert normsub('7') == 'sub-07' and normctx('vid') == 'video' and (normctx('ima') == 'imagery')
    print('multimodal_physio_design self-test: PASS')

def main():
    a = parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    dataset = a.dataset.resolve()
    out = outputs(project)
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not a.force):
        raise RuntimeError('multimodal_physio_design already exists; use --force only for identical rerun')
    rows = []

    def verify(label, p, exp=None):
        if not p.is_file():
            raise FileNotFoundError(p)
        obs = sha(p)
        ok = exp is None or obs == exp
        rows.append({'artifact': label, 'path': str(p), 'sha256': obs, 'expected_sha256': exp or obs, 'passed': ok})
        if not ok:
            raise RuntimeError(f'{label} SHA mismatch')
            return obs
        return obs
    verify('three_cohort_structural_synthesis_final_lock', project / three_cohort_structural_synthesis_LOCK, three_cohort_structural_synthesis_LOCK_SHA)
    verify('three_cohort_structural_synthesis_next_policy', project / three_cohort_structural_synthesis_NEXT, three_cohort_structural_synthesis_NEXT_SHA)
    nx = jread(project / three_cohort_structural_synthesis_NEXT)
    if nx.get('next_phase') != PHASE or not nx.get('next_phase_permitted'):
        raise RuntimeError('three_cohort_structural_synthesis did not authorize multimodal_physio_design')
    for split, (rel, exp) in ASSIGNMENTS.items():
        verify(f'{split}_fixed_EEG_assignments', project / rel, exp)
    p02, origin = locate_p02(project)
    verify('P0.2_physio_contract', p02, P02_CONTRACT_SHA)
    pc = jread(p02)
    if pc.get('status') != 'prespecified' or pc['corrected_trial_level']['trials'] != 1827 or pc['corrected_trial_level']['biological_feature_count'] != 40:
        raise RuntimeError('P0.2 physiology contract changed')
    if pc['corrected_trial_level']['biological_features'] != BIO:
        raise RuntimeError('P0.2 biological feature schema changed')
    if pc['analysis_boundary'].get('raw_dynamic_timeseries_model_allowed_now'):
        raise RuntimeError('Dynamic physiology unexpectedly opened')
    src = source_path(pc, dataset)
    src_sha = verify('corrected_trial_level_physio_source', src, None)
    hdr = list(pd.read_csv(src, nrows=0, low_memory=False).columns)
    required = {'subject_id', 'context', 'trial_index', *BIO}
    if not required.issubset(hdr):
        raise RuntimeError('Corrected physiology source lacks prespecified keys/features: ' + str(sorted(required - set(hdr))))
    canonical = dataset / CANONICAL_REL
    can_sha = verify('canonical_trial_bridge_source', canonical, None)
    ch = list(pd.read_csv(canonical, nrows=0, low_memory=False).columns)
    reqc = {'subject_id', 'trial_number_global', 'context', 'context_trial_index'}
    if not reqc.issubset(ch):
        raise RuntimeError('Canonical source lacks required bridge columns')
    feat = []
    for i, f in enumerate(BIO, 1):
        feat.append({'feature_index': i, 'feature': f, 'modality': 'HR' if f.startswith('hr_') else 'GSR', 'primary_allowed': True, 'source_qc_or_leakage': False, 'preprocessing': 'training/analysis-set median impute; robust median/IQR scale; pooled fallback if IQR=0; winsorize [-6,6]'})
    pd.DataFrame(feat).to_csv(out['features'], index=False)
    hyps = pd.DataFrame([{'test_id': 'M1', 'name': 'fixed_family_multivariate_physio_profile', 'primary': True, 'unit': 'participant-balanced trial data', 'statistic': 'squared L2 norm of equally weighted participant Family1-Family2 residualized physiology contrast (40D)', 'nuisance': 'within-participant context + normalized trial order, fit without family', 'null': '10000 within-participant family-label permutations preserving family counts', 'bootstrap': '5000 participant bootstrap', 'multiplicity': 'Holm across M1-M3'}, {'test_id': 'M2', 'name': 'out_of_person_physio_prediction_of_fixed_family', 'primary': True, 'unit': 'eligible multimodal trials', 'statistic': 'participant-balanced outer-LOPO balanced-accuracy gain of B0+physiology over B0', 'nuisance': 'B0=context + normalized trial order; logistic ridge C grid [0.01,0.1,1,10,100] with grouped inner CV', 'null': '1000 within-participant family-label permutations with full nested procedure', 'bootstrap': '5000 participant bootstrap of observed OOF predictions', 'multiplicity': 'Holm across M1-M3'}, {'test_id': 'M3', 'name': 'continuous_physio_EEG_geometry_coupling', 'primary': True, 'unit': 'participant-balanced multimodal trials', 'statistic': 'Frobenius norm of equally weighted participant 40x6 residual cross-correlation matrix', 'nuisance': 'residualize physiology and six EEG geometry features separately on within-participant context + normalized trial order', 'null': '10000 within-participant row permutations of EEG geometry relative to physiology', 'bootstrap': '5000 participant bootstrap', 'multiplicity': 'Holm across M1-M3'}])
    hyps.to_csv(out['hyp'], index=False)
    gates = pd.DataFrame([{'gate': 'discovery_only_values_opened_in_discovery_multimodal_attachment_qc', 'rule': 'Validation and Confirmation physiology values remain unopened through discovery_multimodal_attachment_qc/discovery_multimodal_inference Discovery inference', 'required': True}, {'gate': 'participant_multimodal_minimum', 'rule': 'participant requires >=16 joined multimodal trials', 'required': True}, {'gate': 'family_test_minimum', 'rule': 'M1/M2 additionally require both fixed families and >=4 joined trials in each family', 'required': True}, {'gate': 'feature_schema', 'rule': 'all 40 prespecified biological features retained; no family/outcome-driven feature selection', 'required': True}, {'gate': 'primary_multiplicity', 'rule': 'Holm across M1-M3 at alpha=0.05', 'required': True}, {'gate': 'block_followups', 'rule': 'GSR(36) and HR(4) follow-ups only if corresponding global parent M1 or M3 is Holm-supported; Holm within the two blocks', 'required': True}, {'gate': 'heldout_opening', 'rule': 'Validation physiology may be opened only if >=1 primary M1-M3 is Holm-supported AND its prespecified robustness checks pass; Confirmation remains closed until one-shot Validation transfer passes', 'required': True}, {'gate': 'EEG_immutability', 'rule': 'Family labels/assignments, centroids, geometry features, support thresholds and three_cohort_structural_synthesis interpretation cannot be retuned or renamed', 'required': True}, {'gate': 'dynamic_physio_closed', 'rule': 'raw dynamic GSR/PPG modeling remains forbidden because P0.2 dynamic_alignment_ready_subjects=0', 'required': True}])
    gates.to_csv(out['gates'], index=False)
    policy = {'phase': PHASE, 'generated_at': now(), 'role': 'outcome-free multimodal/peripheral-physiology integration analysis specification', 'EEG_boundary': {'three_cohort_structural_synthesis_final_lock_sha256': three_cohort_structural_synthesis_LOCK_SHA, 'family_labels': ['Family 1', 'Family 2'], 'family_assignments_immutable': True, 'semantic_labels_forbidden': True}, 'physiology_source': {'p02_contract_sha256': P02_CONTRACT_SHA, 'contract_origin': origin, 'corrected_table': str(src), 'corrected_table_sha256': src_sha, 'canonical_table': str(canonical), 'canonical_table_sha256': can_sha, 'merge_key': ['subject_id', 'context', 'trial_index'], 'biological_features': BIO, 'excluded_columns': BANNED, 'raw_dynamic_timeseries_model_allowed': False}, 'bridge': {'EEG_assignment_key': ['subject_id', 'trial_number'], 'canonical_join': ['subject_id', 'trial_number_global'], 'canonical_carry': ['context', 'context_trial_index as trial_index'], 'physiology_join': ['subject_id', 'context', 'trial_index'], 'subject_normalization': 'sub-XX', 'context_aliases': {'ima': 'imagery', 'ses-ima': 'imagery', 'vid': 'video', 'ses-vid': 'video'}, 'metadata_global_trial_as_behavioral_key_forbidden': True}, 'split_policy': {'Discovery': 'development/inference permitted after discovery_multimodal_attachment_qc attachment QC', 'Validation': 'values remain unopened until prespecified Discovery support+robustness gate', 'Confirmation': 'values remain unopened until one-shot Validation transfer passes'}, 'eligibility': {'multimodal_trials_min_per_participant': 16, 'family_tests_min_trials_per_family': 4, 'family_tests_require_both_families': True, 'missing_feature_handling': 'training/analysis-set median imputation only; never drop based on family association'}, 'preprocessing': {'feature_selection': 'none', 'center_scale': 'median/IQR', 'zero_iqr_fallback': 'pooled training/analysis-set IQR then 1.0', 'winsorize': [-6.0, 6.0], 'M1_M3_nuisance': ['context', 'normalized_trial_order'], 'M2_baseline': ['context', 'normalized_trial_order']}, 'primary_tests': ['M1', 'M2', 'M3'], 'primary_alpha': 0.05, 'primary_multiplicity': 'Holm M1-M3', 'robustness_prespecified': ['unadjusted counterpart of M1 and M3', 'leave-one-participant-out influence diagnostics', 'GSR-only and HR-only gated block follow-ups', 'family-label numerical swap invariance', 'complete-case sensitivity where feasible'], 'forbidden': ['Valence/Liking/report-driven feature selection', 'EEG family relabeling', 'EEG centroid refit', 'EEG threshold tuning', 'Validation/Confirmation physiology peeking before gate', 'raw dynamic physiology inference', 'using source QC/leakage columns as biological predictors']}
    atom_json(out['policy'], policy)
    nextp = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT, 'next_phase_permitted': True, 'next_phase_role': 'Discovery-only physiology attachment, key reconciliation, coverage/missingness QC and prespecified multimodal roster construction; no M1-M3 inference', 'allowed_reads': ['fixed Discovery EEG assignments', 'canonical key/context/trial_index columns', 'Discovery rows of prespecified corrected physiology source and the 40 biological features'], 'forbidden_reads': ['Validation physiology values', 'Confirmation physiology values', 'behavioral outcomes/reports for model selection'], 'required_outputs': ['one-to-one EEG-canonical-physiology join audit', 'Discovery multimodal manifest', 'participant coverage', 'feature missingness profile', 'prespecified eligible roster for discovery_multimodal_inference'], 'inference_forbidden': True}
    atom_json(out['next'], nextp)
    pd.DataFrame(rows).to_csv(out['source'], index=False)
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'outcome_free_multimodal_physio_design_prespecified_candidate', 'counts': {'physio_features': 40, 'GSR_features': 36, 'HR_features': 4, 'prespecified_primary_tests': 3, 'prespecified_EEG_assignment_splits': 3, 'P02_corrected_trials': 1827, 'P02_corrected_participants': 46}, 'decision': {'three_cohort_structural_synthesis_interpretation_immutable': True, 'physiology_source_reverified': True, 'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'multimodal_inference_performed': False, 'discovery_multimodal_attachment_qc_permitted': True}}
    atom_json(out['summary'], summary)
    report = f'# Phase {PHASE} — outcome-free EEG × peripheral-physiology integration analysis specification\n\n- Generated: `{now()}`\n- Status: **outcome_free_multimodal_physio_design_prespecified_candidate**\n\n## Immutable EEG result\nThe three_cohort_structural_synthesis two-family EEG trajectory geometry is treated as fixed. Family 1/2 assignments, centroids, thresholds and the non-semantic interpretation boundary cannot be changed by physiology.\n\n## prespecified physiology source\n- P0.2 corrected trial-level source: `{src}`\n- Source SHA-256: `{src_sha}`\n- Canonical merge key: `subject_id + context + trial_index`\n- Biological features: 40 (36 GSR, 4 HR)\n- Raw dynamic physiology remains closed.\n\n## prespecified bridge\n`EEG subject_id + trial_number` → canonical `subject_id + trial_number_global` → carry `context + context_trial_index` → physiology `subject_id + context + trial_index`.\n\n## Primary Discovery tests\n- **M1:** participant-balanced multivariate physiology difference between fixed Family 1/2.\n- **M2:** out-of-person prediction of fixed family from physiology above context/trial-order baseline.\n- **M3:** participant-balanced coupling between 40D physiology and continuous six-dimensional EEG trajectory geometry.\n\nHolm correction is prespecified across M1-M3. Validation physiology remains unopened unless a supported Discovery primary result also passes prespecified robustness. Confirmation remains closed until a one-shot Validation transfer passes.\n\n## Next phase\ndiscovery_multimodal_attachment_qc performs Discovery-only attachment/coverage/missingness QC and specifications the actual multimodal roster. It performs no M1-M3 inference.'
    out['report'].write_text(report, encoding='utf-8')
    lock_names = ['source', 'features', 'hyp', 'gates', 'policy', 'next', 'summary', 'report']
    hashes = {k: sha(out[k]) for k in lock_names}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': 'outcome_free_multimodal_physio_design_prespecified_candidate', 'source_hashes': {'three_cohort_structural_synthesis_final_lock': three_cohort_structural_synthesis_LOCK_SHA, 'P0.2_contract': P02_CONTRACT_SHA, 'corrected_physio_source': src_sha, 'canonical_bridge_source': can_sha}, 'output_hashes': hashes, 'boundary': summary['decision']}
    atom_json(out['contract'], contract)
    hashes['contract'] = sha(out['contract'])
    final = {'generated_at': now(), 'phase': PHASE, 'hashes': hashes, 'discovery_multimodal_attachment_qc_permitted': True, 'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False, 'multimodal_inference_performed': False}
    atom_json(out['lock'], final)
    stamp = tag()
    hroot = project / ''
    hroot.mkdir(parents=True, exist_ok=True)
    zpath = hroot / f'{PREFIX}_multimodal_physio_design_specification_{stamp}'
    rels = [out[k] for k in ['source', 'features', 'hyp', 'gates', 'policy', 'next', 'summary', 'report', 'contract', 'lock']]
    s = sha(zpath)
    Path(str(zpath) + '.sha256').write_text(f'{s}  {zpath.name}', encoding='utf-8')
    print(f'multimodal_physio_design complete: {zpath}')
    print(f'SHA-256: {s}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())