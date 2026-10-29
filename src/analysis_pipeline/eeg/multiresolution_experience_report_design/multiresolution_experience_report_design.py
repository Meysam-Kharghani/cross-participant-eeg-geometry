# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
PHASE = 'multiresolution_experience_report_design'
NEXT_PHASE = 'discovery_stimulus_experience_multiresolution_fit'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
OUT_REL = Path('outputs/interpretability/multiresolution_experience_report_design_multiresolution_experience_report_design_specification')
topographic_transfer_synthesis_LOCK = Path('outputs/interpretability/topographic_transfer_synthesis_topographic_transfer_analysis_conclusion/qc/topographic_transfer_synthesis_final_output_lock_hashes.json')
topographic_transfer_synthesis_LOCK_SHA256 = 'c77d8ede1e309d28588dd3088628d7bfb6b94e6e47ef1e3d995fc7a4c5c19e64'
topographic_transfer_synthesis_SUMMARY = Path('outputs/interpretability/topographic_transfer_synthesis_topographic_transfer_analysis_conclusion/qc/topographic_transfer_synthesis_summary.json')
cross_person_alignment_fit_LOCK = Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json')
cross_person_alignment_fit_LOCK_SHA256 = '64bb8278afd0ddc17b46521180753e5537333e9e55a93741eece62d406864150'
cross_person_alignment_fit_TRAJECTORY_CANDIDATES = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_candidate_metrics.csv')
cross_person_alignment_fit_TRAJECTORY_CANDIDATES_SHA256 = '042fa2ac335b4d7cb33602630795c1733600ca855b732b8eb0e09003a97e0cf0'
DISCOVERY_STORE = Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store')
DISCOVERY_COMPACT = DISCOVERY_STORE / 'epoch_compact_float32.npy'
DISCOVERY_MANIFEST = DISCOVERY_STORE / 'epoch_manifest.csv'
DISCOVERY_COMPACT_SHA256 = '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'
DISCOVERY_MANIFEST_SHA256 = '7c25a3f77a00fcccd0a03bade656d6066c0641a7e34a76f092fcc43171b296e1'
COMPACT_WIDTH = 169
MIN_PAIRED_TRIALS = 30
HELDOUT_STORE_REFERENCES = {'Validation': {'root': 'outputs/large/validation_feature_trajectory_validation_compact_features_v1', 'compact_sha256': '6f8ab1755f8d950a33351d2032ccd190092656923b09b701ab04e78430fec402', 'manifest_sha256': '24643e775d90d30011fbbba7d40defedcbdb699632603c2d838b420adb49c1a8', 'historical_final_lock_sha256': 'ab18935f81020fcc0d8596d63f8a6d3021c3e19285ac075d3c56c2b566cbfabe'}, 'Confirmation': {'root': 'outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1', 'compact_sha256': '734fa3d2581d95adc172ea854d4f1fccc2432eaeb167960fdf1e76da0c1509d3', 'manifest_sha256': 'ccd83881685da351d79c8de5d199b650d17ed0f6e899c332a7c6d731f7becbd2', 'historical_final_lock_sha256': '0aae3cc2f447f8383c9ec5656c9a5ba9b4299316f3a19248f34b43850b8246df'}}
FEATURE_BLOCKS = [{'name': 'time_frequency_channel_median_iqr', 'start': 0, 'stop': 60}, {'name': 'morphology_channel_median_iqr', 'start': 60, 'stop': 72}, {'name': 'plv_channel_median_iqr', 'start': 72, 'stop': 84}, {'name': 'waveform_dct_channel_median_iqr', 'start': 84, 'stop': 124}, {'name': 'covariance_summary', 'start': 124, 'stop': 145}, {'name': 'plv_global', 'start': 145, 'stop': 169}]
discovery_stimulus_experience_multiresolution_fit_POLICY: dict[str, Any] = {'phase': NEXT_PHASE, 'scientific_role': 'Discovery-only outcome-free multi-resolution stimulus-experience reconfiguration discovery', 'primary_question': 'Does stimulus_early to stimulus_late EEG reconfiguration contain stable cross-person structure when rating_early and all human labels are excluded from discovery?', 'discovery_only': True, 'Validation_opened': False, 'Confirmation_opened': False, 'data_values_permitted': {'stimulus_early_compact_EEG': True, 'stimulus_late_compact_EEG': True, 'rating_early_EEG': False, 'behavioral_reports': False, 'emotion_category': False, 'context': False, 'stimulus_identity': False, 'physiology': False}, 'cohort': {'source': 'multiresolution_experience_report_design prespecified Discovery stimulus-early/stimulus-late pair manifest', 'minimum_paired_trials_per_participant': MIN_PAIRED_TRIALS, 'rating_epoch_required_for_experience_discovery': False, 'no_behavior_based_exclusion': True}, 'representation': {'source_cells_per_epoch': COMPACT_WIDTH, 'blocks': FEATURE_BLOCKS, 'participant_stage_preprocessing': ['for each participant and each stimulus epoch separately, median-impute each of the 169 compact cells using that participant/stage only', 'subtract that participant/stage feature median', 'divide by that participant/stage feature IQR', 'if stage IQR <= 1e-8, use participant-pooled early+late IQR for the same feature', 'if pooled IQR also <= 1e-8, use scale 1.0 and record the feature as locally constant', 'winsorize standardized feature values to [-6, 6]'], 'trial_signature': {'per_block': ['RMS displacement between standardized stimulus_early and stimulus_late profiles', 'cosine similarity between standardized stimulus_early and stimulus_late profiles'], 'global': ['RMS displacement across all 169 compact cells', 'cosine similarity across all 169 compact cells'], 'total_signature_cells': 14, 'positive_distance_transform': 'log1p', 'cosine_transform': 'arctanh after clipping to [-1+1e-6, 1-1e-6]', 'reason': 'captures both magnitude and feature-profile reconfiguration while remaining outcome-free and excluding rating-stage activity'}, 'cross_person_standardization': ['within each participant, robust-center every transformed signature cell by the participant median', 'divide by participant IQR; use Discovery participant-balanced pooled IQR only when local IQR <= 1e-8', 'winsorize to [-6, 6]'], 'participant_weight': 'each participant contributes total weight 1 regardless of trial count'}, 'multi_resolution_family_discovery': {'algorithm': 'participant-weighted KMeans', 'candidate_K': [2, 3, 4, 5, 6, 7, 8], 'n_init': 100, 'max_iter': 500, 'tol': 1e-07, 'seed': 540, 'no_single_K_forced': True, 'all_eligible_resolutions_retained': True, 'canonical_label_rule': 'lexicographic ordering of prespecified centroids in the fixed 14-cell signature coordinate order; never behavioral ordering', 'stability': {'repeats': 200, 'participant_fraction_without_replacement': 0.8, 'trial_fraction_within_retained_participant_without_replacement': 0.8, 'comparison': 'adjusted Rand index versus the corresponding full Discovery K solution', 'eligibility_thresholds_reused_from_cross_person_alignment_fit': {'median_ARI_at_least': 0.6, 'ARI_10th_percentile_at_least': 0.3, 'collapse_rate_at_most': 0.1, 'minimum_participants_per_family': 5, 'minimum_trials_per_family': 30, 'minimum_participant_weighted_occupancy': 0.05}}, 'cross_resolution_outputs': ['ARI and NMI for every pair of eligible K resolutions', 'family overlap matrices', 'fine-to-coarse parent purity without assuming a tree'], 'no_eligible_K_rule': 'do not force discrete families; close the discrete experience-family branch and retain continuous reconfiguration signatures only'}, 'historical_K2_boundary': {'old_F1_F2_refit_or_reused_as_targets': False, 'old_three_epoch_geometry_used_in_discovery': False, 'old_family_assignments_may_be_compared_only_after_new_experience_structure_is fixed': True, 'purpose_of_later_comparison': 'quantify how much inclusion of rating_early reorganized the earlier three-epoch family partition; never use the old labels to choose new K or centroids'}, 'next_after_discovery_stimulus_experience_multiresolution_fit': {'if_at_least_one_K_eligible': 'validation_experience_transfer_design specifications Discovery templates/support thresholds and a one-shot Validation stimulus-only structural transfer before Validation signal values are opened', 'if_no_K_eligible': 'validation_experience_transfer_design records discrete-analysis conclusion and specifications a continuous experience/report dissociation route; no family taxonomy is invented', 'Validation_behavior_values_open_before_structural_transfer': False, 'Confirmation_open_before_Validation_gate': False}}
PROGRAM_POLICY: dict[str, Any] = {'phase': PHASE, 'role': 'program-level analysis specification before any new signal-family inference', 'scientific_hypotheses': {'H_label_mismatch': 'human emotion labels may merge multiple reproducible neural regimes or may cut across the natural neural organization', 'H_report_transition': 'the neural transformation from stimulus processing into rating/report generation may be structurally distinct from the experience-period organization and may contribute to the weak EEG-to-report mapping'}, 'separation_principle': {'experience_branch': 'stimulus_early -> stimulus_late only', 'report_transition_branch': 'stimulus_late -> rating_early only, constructed later as a separate outcome-free branch', 'forbidden': 'discovering a family from a representation that mixes experience and report transition and then treating it as a pure experience state'}, 'opening_order': ['multiresolution_experience_report_design: metadata/hash/analysis specification only; no new EEG feature values and no labels', 'discovery_stimulus_experience_multiresolution_fit: Discovery stimulus-only EEG feature values; no rating_early, behavior, category, context, stimulus identity, physiology, Validation, or Confirmation', 'validation_experience_transfer_design: specification templates/support thresholds and held-out structural transfer using only discovery_stimulus_experience_multiresolution_fit structural outputs; still no behavioral labels', 'validation_stimulus_experience_multiresolution_transfer: one-shot Validation stimulus-only structural transfer if authorized by validation_experience_transfer_design; no behavioral labels', 'later: specification and execute a separate Discovery report-transition structural branch without behavior', 'later: only after structural locks, specification exact tests of human-label mixing/fragmentation and report-transition association before opening behavioral values', 'Confirmation: remains closed for the new analysis until a pre-prespecified Validation structural gate explicitly authorizes it'], 'semantic_claim_boundary': ['No new family may be called joy, fear, anger, positive, negative, valence, liking, arousal, or another psychological construct from structural discovery alone.', 'High within-label neural heterogeneity alone does not prove that the human label is wrong; later held-out replication and explicit prespecified mapping tests are required.', 'An association between late-to-rating neural transformation and a report does not prove conscious distortion or dishonesty; the strongest allowed wording is report/evaluation-stage neural transformation unless stronger evidence is independently established.'], 'physiology_boundary': 'raw event-aligned GSR/PPG remains outside the primary structural_replication program because <=1 s EEG-physiology synchronization was not recovered; physiology may only return under a separately prespecified feasible representation.', 'discovery_stimulus_experience_multiresolution_fit_policy': discovery_stimulus_experience_multiresolution_fit_POLICY, 'heldout_store_references_not_opened_in_multiresolution_experience_report_design': HELDOUT_STORE_REFERENCES}

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def readj(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')
    os.replace(tmp, path)

def verify_exact(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: {got} != {expected}')
    return got

def verify_topographic_transfer_synthesis(project: Path) -> dict[str, Any]:
    p = project / topographic_transfer_synthesis_LOCK
    verify_exact(p, topographic_transfer_synthesis_LOCK_SHA256, 'topographic_transfer_synthesis final lock')
    lock = readj(p)
    if lock.get('phase') != 'topographic_transfer_synthesis':
        raise RuntimeError('topographic_transfer_synthesis phase mismatch')
    if lock.get('status') != 'topographic_transfer_analysis_concluded':
        raise RuntimeError('topographic_transfer_synthesis branch is not closed')
    if lock.get('Confirmation_transfer_values_may_open_now') is not False:
        raise RuntimeError('topographic_transfer_synthesis Confirmation boundary drift')
    if lock.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('topographic_transfer_synthesis semantic boundary drift')
    for rel, expected in lock.get('hashes', {}).items():
        child = project / rel
        verify_exact(child, expected, f'topographic_transfer_synthesis fixed child {rel}')
    s = project / topographic_transfer_synthesis_SUMMARY
    if not s.is_file():
        raise RuntimeError(f'Missing topographic_transfer_synthesis summary: {s}')
    summary = readj(s)
    if summary.get('next_phase') != PHASE:
        raise RuntimeError(f"topographic_transfer_synthesis next phase drift: {summary.get('next_phase')}")
    return lock

def verify_historical_context(project: Path) -> pd.DataFrame:
    lock_path = project / cross_person_alignment_fit_LOCK
    verify_exact(lock_path, cross_person_alignment_fit_LOCK_SHA256, 'cross_person_alignment_fit final lock')
    lock = readj(lock_path)
    if lock.get('reports_outcomes_context_qc_read') is not False:
        raise RuntimeError('cross_person_alignment_fit was not outcome-free according to its lock')
    p = project / cross_person_alignment_fit_TRAJECTORY_CANDIDATES
    verify_exact(p, cross_person_alignment_fit_TRAJECTORY_CANDIDATES_SHA256, 'cross_person_alignment_fit trajectory candidate metrics')
    df = pd.read_csv(p)
    req = {'k', 'median_ari', 'ari_10th_percentile', 'participant_weighted_silhouette', 'eligible'}
    if not req.issubset(df.columns):
        raise RuntimeError(f'Historical K table schema drift: missing {sorted(req - set(df.columns))}')
    return df

def verify_discovery_store(project: Path) -> tuple[Path, Path]:
    compact = project / DISCOVERY_COMPACT
    manifest = project / DISCOVERY_MANIFEST
    verify_exact(compact, DISCOVERY_COMPACT_SHA256, 'Discovery compact EEG array')
    verify_exact(manifest, DISCOVERY_MANIFEST_SHA256, 'Discovery epoch manifest')
    return (compact, manifest)

def pair_manifest(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    need = ['raw_behavior_row_id', 'subject_id', 'epoch_family', 'epoch_store_row_0based']
    header = pd.read_csv(manifest_path, nrows=0)
    missing = set(need) - set(header.columns)
    if missing:
        raise RuntimeError(f'Discovery epoch manifest schema drift: missing {sorted(missing)}')
    m = pd.read_csv(manifest_path, usecols=need, low_memory=False)
    m['raw_behavior_row_id'] = m['raw_behavior_row_id'].astype(str)
    m['subject_id'] = m['subject_id'].astype(str)
    m['epoch_family'] = m['epoch_family'].astype(str)
    if m.duplicated(['raw_behavior_row_id', 'epoch_family']).any():
        raise RuntimeError('Duplicate trial/epoch rows in Discovery epoch manifest')

    def one(family: str, row_name: str) -> pd.DataFrame:
        x = m[m.epoch_family.eq(family)][['raw_behavior_row_id', 'subject_id', 'epoch_store_row_0based']].copy()
        x = x.rename(columns={'subject_id': f'subject_id_{family}', 'epoch_store_row_0based': row_name})
        return x
    early = one('stimulus_early', 'stimulus_early_row_0based')
    late = one('stimulus_late', 'stimulus_late_row_0based')
    rating = one('rating_early', 'rating_early_row_0based')
    exp = early.merge(late, on='raw_behavior_row_id', how='inner', validate='one_to_one')
    if not (exp.subject_id_stimulus_early == exp.subject_id_stimulus_late).all():
        raise RuntimeError('Subject mismatch in stimulus pair manifest')
    exp['subject_id'] = exp.subject_id_stimulus_early
    exp = exp[['raw_behavior_row_id', 'subject_id', 'stimulus_early_row_0based', 'stimulus_late_row_0based']]
    exp_counts = exp.groupby('subject_id').size().rename('experience_paired_trials').reset_index()
    exp_counts['experience_eligible'] = exp_counts.experience_paired_trials.ge(MIN_PAIRED_TRIALS)
    exp = exp.merge(exp_counts, on='subject_id', how='left', validate='many_to_one')
    exp = exp.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)
    rpt = late.merge(rating, on='raw_behavior_row_id', how='inner', validate='one_to_one')
    if not (rpt.subject_id_stimulus_late == rpt.subject_id_rating_early).all():
        raise RuntimeError('Subject mismatch in report-transition pair manifest')
    rpt['subject_id'] = rpt.subject_id_stimulus_late
    rpt = rpt[['raw_behavior_row_id', 'subject_id', 'stimulus_late_row_0based', 'rating_early_row_0based']]
    rpt_counts = rpt.groupby('subject_id').size().rename('report_transition_paired_trials').reset_index()
    rpt_counts['report_transition_eligible'] = rpt_counts.report_transition_paired_trials.ge(MIN_PAIRED_TRIALS)
    rpt = rpt.merge(rpt_counts, on='subject_id', how='left', validate='many_to_one')
    rpt = rpt.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)
    tri = exp[['raw_behavior_row_id', 'subject_id', 'stimulus_early_row_0based', 'stimulus_late_row_0based']].merge(rpt[['raw_behavior_row_id', 'subject_id', 'rating_early_row_0based']], on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
    tri = tri.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)
    subj = exp_counts.merge(rpt_counts, on='subject_id', how='outer').fillna({'experience_paired_trials': 0, 'experience_eligible': False, 'report_transition_paired_trials': 0, 'report_transition_eligible': False})
    tri_counts = tri.groupby('subject_id').size().rename('all_three_trials')
    subj = subj.merge(tri_counts, on='subject_id', how='left').fillna({'all_three_trials': 0})
    subj['experience_paired_trials'] = subj.experience_paired_trials.astype(int)
    subj['report_transition_paired_trials'] = subj.report_transition_paired_trials.astype(int)
    subj['all_three_trials'] = subj.all_three_trials.astype(int)
    subj = subj.sort_values('subject_id').reset_index(drop=True)
    return (exp, rpt, tri, subj)

def self_test() -> None:
    assert [b['start'] for b in FEATURE_BLOCKS] == [0, 60, 72, 84, 124, 145]
    assert [b['stop'] for b in FEATURE_BLOCKS] == [60, 72, 84, 124, 145, 169]
    assert discovery_stimulus_experience_multiresolution_fit_POLICY['multi_resolution_family_discovery']['candidate_K'] == [2, 3, 4, 5, 6, 7, 8]
    assert discovery_stimulus_experience_multiresolution_fit_POLICY['data_values_permitted']['rating_early_EEG'] is False
    assert discovery_stimulus_experience_multiresolution_fit_POLICY['data_values_permitted']['behavioral_reports'] is False
    assert discovery_stimulus_experience_multiresolution_fit_POLICY['multi_resolution_family_discovery']['no_single_K_forced'] is True
    assert PROGRAM_POLICY['separation_principle']['experience_branch'] == 'stimulus_early -> stimulus_late only'
    print('multiresolution_experience_report_design self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    print(f'[{now()}] verifying topographic_transfer_synthesis closure', flush=True)
    topographic_transfer_synthesis_lock = verify_topographic_transfer_synthesis(project)
    print(f'[{now()}] verifying historical cross_person_alignment_fit multi-resolution context', flush=True)
    historical = verify_historical_context(project)
    print(f'[{now()}] byte-verifying Discovery compact EEG store (no np.load)', flush=True)
    compact_path, manifest_path = verify_discovery_store(project)
    print(f'[{now()}] freezing stimulus-only and late-to-rating structural trial cohorts from epoch identifiers only', flush=True)
    exp, rpt, tri, subjects = pair_manifest(manifest_path)
    out = project / OUT_REL
    final = out / 'qc/multiresolution_experience_report_design_final_output_lock_hashes.json'
    if out.exists():
        if final.is_file() and (not a.force):
            raise RuntimeError(f'Completed multiresolution_experience_report_design output exists: {out}')
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    config_program = project / 'analysis_specifications/S045_multiresolution_experience_report.json'
    config_discovery_stimulus_experience_multiresolution_fit = project / 'analysis_specifications/S046_multiresolution_experience_report_discovery_stimulus.json'
    writej(config_program, PROGRAM_POLICY)
    writej(config_discovery_stimulus_experience_multiresolution_fit, discovery_stimulus_experience_multiresolution_fit_POLICY)
    exp.to_csv(out / 'tables/multiresolution_experience_report_design_discovery_experience_pair_manifest.csv', index=False)
    rpt.to_csv(out / 'tables/multiresolution_experience_report_design_discovery_report_transition_pair_manifest.csv', index=False)
    tri.to_csv(out / 'tables/multiresolution_experience_report_design_discovery_all_three_epoch_linkage_manifest.csv', index=False)
    subjects.to_csv(out / 'qc/multiresolution_experience_report_design_discovery_pair_eligibility.csv', index=False)
    hcols = [c for c in ['k', 'participant_weighted_silhouette', 'median_ari', 'ari_10th_percentile', 'collapse_rate', 'eligible'] if c in historical.columns]
    historical[hcols].sort_values('k').to_csv(out / 'tables/multiresolution_experience_report_design_historical_three_epoch_multiresolution_context.csv', index=False)
    source_rehash = pd.DataFrame([{'source': 'topographic_transfer_synthesis_final_lock', 'path': str(topographic_transfer_synthesis_LOCK), 'sha256': topographic_transfer_synthesis_LOCK_SHA256, 'verified': True, 'values_opened': False}, {'source': 'cross_person_alignment_fit_final_lock', 'path': str(cross_person_alignment_fit_LOCK), 'sha256': cross_person_alignment_fit_LOCK_SHA256, 'verified': True, 'values_opened': False}, {'source': 'cross_person_alignment_fit_trajectory_candidate_metrics', 'path': str(cross_person_alignment_fit_TRAJECTORY_CANDIDATES), 'sha256': cross_person_alignment_fit_TRAJECTORY_CANDIDATES_SHA256, 'verified': True, 'values_opened': True}, {'source': 'Discovery_compact_EEG_array', 'path': str(DISCOVERY_COMPACT), 'sha256': DISCOVERY_COMPACT_SHA256, 'verified': True, 'values_opened': False}, {'source': 'Discovery_epoch_manifest', 'path': str(DISCOVERY_MANIFEST), 'sha256': DISCOVERY_MANIFEST_SHA256, 'verified': True, 'values_opened': 'identifier_columns_only'}])
    source_rehash.to_csv(out / 'qc/multiresolution_experience_report_design_source_rehash.csv', index=False)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'multiresolution_experience_report_program_and_discovery_stimulus_experience_multiresolution_fit_design_prespecified', 'program_policy_canonical_sha256': canonical_sha(PROGRAM_POLICY), 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': canonical_sha(discovery_stimulus_experience_multiresolution_fit_POLICY), 'counts': {'Discovery_subjects_with_any_experience_pair': int(exp.subject_id.nunique()), 'Discovery_experience_pairs_all': int(len(exp)), 'Discovery_experience_eligible_subjects': int(subjects.experience_eligible.astype(bool).sum()), 'Discovery_experience_pairs_in_eligible_subjects': int(exp.loc[exp.experience_eligible.astype(bool)].shape[0]), 'Discovery_subjects_with_any_report_transition_pair': int(rpt.subject_id.nunique()), 'Discovery_report_transition_pairs_all': int(len(rpt)), 'Discovery_report_transition_eligible_subjects': int(subjects.report_transition_eligible.astype(bool).sum()), 'Discovery_all_three_epoch_trials_for_later_structural_linkage': int(len(tri))}, 'historical_context': {'old_three_epoch_K2_refit': False, 'old_three_epoch_family_assignments_used_as_targets': False, 'old_three_epoch_candidate_K_values_observed': historical.k.astype(int).tolist(), 'old_three_epoch_eligible_K_values': historical.loc[historical.eligible.astype(str).str.lower().isin(['true', '1', 'yes']), 'k'].astype(int).tolist()}, 'scientific_boundaries': {'new_EEG_feature_values_opened_in_multiresolution_experience_report_design': False, 'rating_early_EEG_values_opened_in_multiresolution_experience_report_design': False, 'behavioral_report_values_opened_in_multiresolution_experience_report_design': False, 'emotion_category_values_opened_in_multiresolution_experience_report_design': False, 'context_values_opened_in_multiresolution_experience_report_design': False, 'stimulus_identity_values_opened_in_multiresolution_experience_report_design': False, 'Validation_store_opened_in_multiresolution_experience_report_design': False, 'Confirmation_store_opened_in_multiresolution_experience_report_design': False, 'family_fit_performed_in_multiresolution_experience_report_design': False, 'semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE, 'next_phase_role': 'execute the prespecified Discovery-only stimulus_early-to-stimulus_late multi-resolution EEG reconfiguration family analysis'}
    writej(out / 'qc/multiresolution_experience_report_design_summary.json', summary)
    report = f'# multiresolution_experience_report_design — Multi-resolution experience/report dissociation analysis specification\n\nStatus: **design prespecified**. No new EEG feature values, behavioral reports, category values, context values, Validation signal values, or Confirmation signal values were analyzed.\n\n## Why this is new rather than a repeat of analysis_stage/analysis_stage\nThe earlier F1/F2 solution was built from the three-stage trajectory `stimulus_early -> stimulus_late -> rating_early`. structural_replication explicitly separates the experience-period transition from the later report/evaluation transition.\n\nC9B will use **only** `stimulus_early -> stimulus_late`. `rating_early` is forbidden from the new family-discovery representation.\n\n## Multi-resolution rule\nC9B tests K=2 through K=8. No K is selected merely because it is simplest or resembles the old K=2 result. Every resolution must independently pass the already-established cross_person_alignment_fit stability/collapse criteria. All passing resolutions are retained; if none pass, no discrete family taxonomy is forced.\n\n## Stimulus-only signature\nFor each trial, the 169 compact EEG cells are separated into the same six prespecified feature blocks. After participant/stage-local robust normalization, discovery_stimulus_experience_multiresolution_fit computes blockwise early-to-late RMS displacement and early/late profile cosine, plus global RMS displacement and global cosine: 14 outcome-free reconfiguration cells.\n\n## prespecified Discovery cohort metadata\n- experience pairs (all): **{len(exp)}**\n- experience-eligible participants (>= {MIN_PAIRED_TRIALS} paired trials): **{int(subjects.experience_eligible.astype(bool).sum())}**\n- experience pairs in eligible participants: **{int(exp.loc[exp.experience_eligible.astype(bool)].shape[0])}**\n- late-to-rating pairs reserved only for a later separately prespecified branch: **{len(rpt)}**\n- all-three trials available for later outcome-free structural linkage: **{len(tri)}**\n\n## Claim boundary\nStructural families are not emotions. Later tests may ask whether human labels split across several validated neural families, or whether the late-to-rating transformation relates to reports, but those tests must be separately prespecified before behavioral values are opened. A report-stage association cannot by itself be called distortion, dishonesty, or correction of a true feeling.\n\n## Held-out boundary\nValidation and Confirmation compact stores are not opened in multiresolution_experience_report_design. discovery_stimulus_experience_multiresolution_fit is Discovery-only. If discovery_stimulus_experience_multiresolution_fit yields at least one eligible K, validation_experience_transfer_design must specification centroids, support thresholds, and the one-shot Validation transfer before Validation signal values are opened. Confirmation remains closed until a pre-prespecified Validation gate authorizes it.\n\n## Next phase\n**{NEXT_PHASE}** — Discovery-only stimulus experience reconfiguration fit.'
    (out / 'reports/multiresolution_experience_report_design_report.md').write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'design_prespecified', 'source_code_sha256': sha256(Path(__file__).resolve()), 'verified_input_sha256': {'topographic_transfer_synthesis_final_lock': topographic_transfer_synthesis_LOCK_SHA256, 'cross_person_alignment_fit_final_lock': cross_person_alignment_fit_LOCK_SHA256, 'cross_person_alignment_fit_trajectory_candidate_metrics': cross_person_alignment_fit_TRAJECTORY_CANDIDATES_SHA256, 'Discovery_compact_EEG_array': DISCOVERY_COMPACT_SHA256, 'Discovery_epoch_manifest': DISCOVERY_MANIFEST_SHA256}, 'program_policy_canonical_sha256': canonical_sha(PROGRAM_POLICY), 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': canonical_sha(discovery_stimulus_experience_multiresolution_fit_POLICY), 'heldout_store_reference_hashes_prespecified_but_files_not_opened': HELDOUT_STORE_REFERENCES, 'new_signal_inference_performed': False, 'new_family_fit_performed': False, 'behavioral_values_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/multiresolution_experience_report_design_contract.json', contract)
    fixed = [p for p in sorted(out.rglob('*')) if p.is_file()]
    fixed += [config_program, config_discovery_stimulus_experience_multiresolution_fit]
    hashes = {str(p.relative_to(project)): sha256(p) for p in fixed}
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'multiresolution_experience_report_design_prespecified', 'hashes': hashes, 'program_policy_canonical_sha256': canonical_sha(PROGRAM_POLICY), 'discovery_stimulus_experience_multiresolution_fit_policy_canonical_sha256': canonical_sha(discovery_stimulus_experience_multiresolution_fit_POLICY), 'new_signal_inference_performed': False, 'behavioral_values_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final, final_obj)
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    rels += [str(config_program.relative_to(project)), str(config_discovery_stimulus_experience_multiresolution_fit.relative_to(project))]
    script_dir = project / 'src/analysis_pipeline/eeg/multiresolution_experience_report_design_multiresolution_experience_report_design_specification'
    if script_dir.is_dir():
        rels += [str(p.relative_to(project)) for p in sorted(script_dir.iterdir()) if p.is_file()]
    rels = sorted(dict.fromkeys(rels))
    print('multiresolution_experience_report_design PASS')
    print(f'Experience pairs (eligible subjects): {int(exp.loc[exp.experience_eligible.astype(bool)].shape[0])}')
    print(f'Experience eligible participants: {int(subjects.experience_eligible.astype(bool).sum())}')
    print('New EEG feature values opened: False')
    print('Validation opened: False')
    print('Confirmation opened: False')
    print('Semantic relabeling permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())