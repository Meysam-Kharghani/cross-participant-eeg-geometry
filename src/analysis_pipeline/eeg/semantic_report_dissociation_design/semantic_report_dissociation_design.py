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
PHASE = 'semantic_report_dissociation_design'
NEXT_PHASE = 'discovery_report_transition_multiresolution_fit'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
EXPECTED_K = [2, 3, 4]
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
NOMINAL_EMOTION_CATEGORIES = ['sad', 'dis', 'fear', 'neu', 'joy', 'ten', 'ins']
cross_split_experience_structure_synthesis_OUT = Path('outputs/interpretability/cross_split_experience_structure_synthesis_crosssplit_experience_structure_synthesis')
cross_split_experience_structure_synthesis_LOCK = cross_split_experience_structure_synthesis_OUT / 'qc/cross_split_experience_structure_synthesis_final_output_lock_hashes.json'
cross_split_experience_structure_synthesis_LOCK_SHA256 = 'c0dc3c2f5930486fd6ab231f0d6b04dc665a3b34da69d5c02af60154148402e5'
cross_split_experience_structure_synthesis_NEXT = Path('analysis_specifications/S014_cross_split_experience_structure_synthesis_semantic.json')
cross_split_experience_structure_synthesis_NEXT_SHA256 = '86905a0d76ea3cb9f49ce1e87f243c1c34da9fcb5d254e2500461a85d61f8c89'
multiresolution_experience_report_design_OUT = Path('outputs/interpretability/multiresolution_experience_report_design_multiresolution_experience_report_design_specification')
multiresolution_experience_report_design_LOCK = multiresolution_experience_report_design_OUT / 'qc/multiresolution_experience_report_design_final_output_lock_hashes.json'
multiresolution_experience_report_design_LOCK_SHA256 = 'ae65247138d1aa19cbafa05542702031d273b77503e569146710e2db1d8d467a'
multiresolution_experience_report_design_PROGRAM_POLICY = Path('analysis_specifications/S045_multiresolution_experience_report.json')
multiresolution_experience_report_design_PROGRAM_POLICY_SHA256 = '2642a87c60f5082276e65ef12c7de81037dfbbbb628666e458a366af56e81927'
multiresolution_experience_report_design_RT_MANIFEST = multiresolution_experience_report_design_OUT / 'tables/multiresolution_experience_report_design_discovery_report_transition_pair_manifest.csv'
multiresolution_experience_report_design_RT_MANIFEST_SHA256 = '9deb3788034a3e1d458e2c66b125fb86d0766ada0a6c30ffe9066a5490341983'
discovery_stimulus_experience_multiresolution_fit_OUT = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit')
discovery_stimulus_experience_multiresolution_fit_LOCK = discovery_stimulus_experience_multiresolution_fit_OUT / 'qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json'
discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256 = '35b9fa3efacfbe1e3884e2d622e1701538a5375fe2b72ffe1155fc429903c2b7'
discovery_stimulus_experience_multiresolution_fit_ASSIGN = discovery_stimulus_experience_multiresolution_fit_OUT / 'tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'
discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256 = '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'
discovery_stimulus_experience_multiresolution_fit_SIGNATURE = discovery_stimulus_experience_multiresolution_fit_OUT / 'tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'
discovery_stimulus_experience_multiresolution_fit_SIGNATURE_SHA256 = 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'
validation_stimulus_experience_multiresolution_transfer_OUT = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer')
validation_stimulus_experience_multiresolution_transfer_LOCK = validation_stimulus_experience_multiresolution_transfer_OUT / 'qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256 = '5c8431957ac2fe3ffdc8886724b7145359062f83580463b527937164daa9b0ca'
validation_stimulus_experience_multiresolution_transfer_ASSIGN = validation_stimulus_experience_multiresolution_transfer_OUT / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv'
validation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256 = 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'
validation_stimulus_experience_multiresolution_transfer_SIGNATURE = validation_stimulus_experience_multiresolution_transfer_OUT / 'tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv'
validation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256 = '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'
confirmation_stimulus_experience_multiresolution_transfer_OUT = Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer')
confirmation_stimulus_experience_multiresolution_transfer_LOCK = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'qc/confirmation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256 = '3b6f03c0d44903345871470cc40e35b0d9d0aaa9d836954e1c0009a7485b6b8b'
confirmation_stimulus_experience_multiresolution_transfer_ASSIGN = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'
confirmation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256 = '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'
confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'
confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256 = '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'
primary_analysis_reference_CONTRACT = Path('config/outcome_model_integrity_outcome_model_integrity_contract.json')
primary_analysis_reference_CONTRACT_SHA256 = 'ed328384c7c1eed0bf92e2f1ddbd4daa3009eceb325092b41fadc2d8b182b018'
DISCOVERY_TRIAL_OUTCOMES = Path('outputs/tables/outcome_model_integrity_discovery_trial_outcome_manifest.csv')
DISCOVERY_TRIAL_OUTCOMES_SHA256 = 'a3cf2292d31749b96a0e3729bb5860d69bb642219ce6e478f4f794fb48c832ec'
CORRECTED_DISCOVERY_MASTER = Path('outputs/tables/derivative_closure_corrected_discovery_master_v1.csv')
CORRECTED_DISCOVERY_MASTER_SHA256 = '79aac098573fada34c2dd1247e7ce91e82633130f43cfdea82e2f59b41445039'
DISCOVERY_STORE = Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store')
DISCOVERY_COMPACT = DISCOVERY_STORE / 'epoch_compact_float32.npy'
DISCOVERY_MANIFEST = DISCOVERY_STORE / 'epoch_manifest.csv'
DISCOVERY_COMPACT_SHA256 = '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'
DISCOVERY_MANIFEST_SHA256 = '7c25a3f77a00fcccd0a03bade656d6066c0641a7e34a76f092fcc43171b296e1'
OUT = Path('outputs/interpretability/semantic_report_dissociation_design_semantic_report_dissociation_design_specification')
RT_POLICY = Path('analysis_specifications/S060_semantic_report_dissociation_discovery_transition.json')
SEMANTIC_POLICY = Path('analysis_specifications/S061_semantic_report_dissociation_future.json')
BLOCKS = [{'name': 'time_frequency_channel_median_iqr', 'start': 0, 'stop': 60}, {'name': 'morphology_channel_median_iqr', 'start': 60, 'stop': 72}, {'name': 'plv_channel_median_iqr', 'start': 72, 'stop': 84}, {'name': 'waveform_dct_channel_median_iqr', 'start': 84, 'stop': 124}, {'name': 'covariance_summary', 'start': 124, 'stop': 145}, {'name': 'plv_global', 'start': 145, 'stop': 169}]

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

def verify_lock_children(project: Path, rel: Path, expected_sha: str, phase: str) -> list[dict[str, str]]:
    lock_path = project / rel
    verify(lock_path, expected_sha, f'{phase} final lock')
    lock = readj(lock_path)
    if lock.get('phase') != phase:
        raise RuntimeError(f'{phase} final lock phase drift')
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{phase} final lock missing child hashes')
    rows = []
    for child_rel, expected in sorted(hashes.items()):
        p = project / child_rel
        if not p.is_file():
            raise RuntimeError(f'Missing {phase} fixed child: {child_rel}')
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'{phase} fixed child SHA drift: {child_rel}; expected {expected}, got {got}')
        rows.append({'source_phase': phase, 'path': child_rel, 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    return rows

def self_test() -> int:
    assert EXPECTED_K == [2, 3, 4]
    assert len(REPORTS) == 10
    assert sum((b['stop'] - b['start'] for b in BLOCKS)) == 169
    assert PHASE == 'semantic_report_dissociation_design'
    assert NEXT_PHASE == 'discovery_report_transition_multiresolution_fit'
    print('semantic_report_dissociation_design self-test PASS')
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
    print(f'[{now()}] Verifying cross_split_experience_structure_synthesis structural closure', flush=True)
    rehash_rows = verify_lock_children(project, cross_split_experience_structure_synthesis_LOCK, cross_split_experience_structure_synthesis_LOCK_SHA256, 'cross_split_experience_structure_synthesis')
    verify(project / cross_split_experience_structure_synthesis_NEXT, cross_split_experience_structure_synthesis_NEXT_SHA256, 'cross_split_experience_structure_synthesis→semantic_report_dissociation_design prespecified policy')
    hnext = readj(project / cross_split_experience_structure_synthesis_NEXT)
    if hnext.get('structurally_replicated_resolutions') != EXPECTED_K:
        raise RuntimeError('cross_split_experience_structure_synthesis replicated-resolution set drift')
    if hnext.get('semantic_report_dissociation_design_data_opening') != {'behavioral_values': False, 'new_signal_values': False, 'rating_early_EEG': False}:
        raise RuntimeError('cross_split_experience_structure_synthesis semantic_report_dissociation_design data-opening boundary drift')
    print(f'[{now()}] Verifying original multiresolution_experience_report_design program specification and safe transition identifiers', flush=True)
    rehash_rows += verify_lock_children(project, multiresolution_experience_report_design_LOCK, multiresolution_experience_report_design_LOCK_SHA256, 'multiresolution_experience_report_design')
    verify(project / multiresolution_experience_report_design_PROGRAM_POLICY, multiresolution_experience_report_design_PROGRAM_POLICY_SHA256, 'multiresolution_experience_report_design program policy')
    verify(project / multiresolution_experience_report_design_RT_MANIFEST, multiresolution_experience_report_design_RT_MANIFEST_SHA256, 'multiresolution_experience_report_design report-transition pair manifest')
    multiresolution_experience_report_design_policy = readj(project / multiresolution_experience_report_design_PROGRAM_POLICY)
    if multiresolution_experience_report_design_policy.get('separation_principle', {}).get('report_transition_branch') != 'stimulus_late -> rating_early only, constructed later as a separate outcome-free branch':
        raise RuntimeError('multiresolution_experience_report_design report-transition separation principle drift')
    rt_rows = read_csv(project / multiresolution_experience_report_design_RT_MANIFEST)
    if len(rt_rows) != 1252:
        raise RuntimeError(f'multiresolution_experience_report_design report-transition pair count drift: {len(rt_rows)}')
    required_cols = {'raw_behavior_row_id', 'subject_id', 'stimulus_late_row_0based', 'rating_early_row_0based', 'report_transition_paired_trials', 'report_transition_eligible'}
    if set(rt_rows[0]) != required_cols:
        raise RuntimeError(f'multiresolution_experience_report_design report-transition manifest schema drift: {list(rt_rows[0])}')
    eligible_rows = [r for r in rt_rows if str(r['report_transition_eligible']).lower() == 'true']
    eligible_subjects = sorted({r['subject_id'] for r in eligible_rows})
    if len(eligible_rows) != 1237 or len(eligible_subjects) != 31:
        raise RuntimeError(f'multiresolution_experience_report_design report-transition eligibility drift: rows={len(eligible_rows)}, subjects={len(eligible_subjects)}')
    print(f'[{now()}] Verifying immutable experience-family references', flush=True)
    for rel, expected, label in [(discovery_stimulus_experience_multiresolution_fit_LOCK, discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit final lock'), (validation_stimulus_experience_multiresolution_transfer_LOCK, validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer final lock'), (confirmation_stimulus_experience_multiresolution_transfer_LOCK, confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer final lock'), (discovery_stimulus_experience_multiresolution_fit_ASSIGN, discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256, 'discovery_stimulus_experience_multiresolution_fit experience assignments'), (validation_stimulus_experience_multiresolution_transfer_ASSIGN, validation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256, 'validation_stimulus_experience_multiresolution_transfer experience assignments'), (confirmation_stimulus_experience_multiresolution_transfer_ASSIGN, confirmation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer experience assignments'), (discovery_stimulus_experience_multiresolution_fit_SIGNATURE, discovery_stimulus_experience_multiresolution_fit_SIGNATURE_SHA256, 'discovery_stimulus_experience_multiresolution_fit experience signatures'), (validation_stimulus_experience_multiresolution_transfer_SIGNATURE, validation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256, 'validation_stimulus_experience_multiresolution_transfer experience signatures'), (confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE, confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer experience signatures')]:
        got = verify(project / rel, expected, label)
        rehash_rows.append({'source_phase': 'immutable_experience_reference', 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': 'True'})
    verify(project / primary_analysis_reference_CONTRACT, primary_analysis_reference_CONTRACT_SHA256, 'primary_analysis_reference outcome contract')
    primary_analysis_reference = readj(project / primary_analysis_reference_CONTRACT)
    if primary_analysis_reference.get('output_hashes', {}).get('trial_outcomes') != DISCOVERY_TRIAL_OUTCOMES_SHA256:
        raise RuntimeError('primary_analysis_reference Discovery trial-outcome authority drift')
    if primary_analysis_reference.get('input_hashes', {}).get('canonical_master') != CORRECTED_DISCOVERY_MASTER_SHA256:
        raise RuntimeError('primary_analysis_reference corrected-master authority drift')
    prespecified_rt_manifest = out / 'tables/semantic_report_dissociation_design_prespecified_discovery_report_transition_pair_manifest.csv'
    prespecified_rt_manifest.write_bytes((project / multiresolution_experience_report_design_RT_MANIFEST).read_bytes())
    rt_policy = {'phase': NEXT_PHASE, 'prespecified_by': PHASE, 'generated_at': now(), 'scientific_role': 'Discovery-only outcome-free multi-resolution structural analysis of stimulus_late -> rating_early EEG reconfiguration', 'primary_question': 'Does the transition from late stimulus processing into the rating/report stage contain stable cross-person structure that is distinct from the already fixed stimulus_early -> stimulus_late experience structure?', 'discovery_only': True, 'cohort': {'source_manifest': str(prespecified_rt_manifest.relative_to(project)), 'all_paired_rows': 1252, 'eligible_rows': 1237, 'eligible_subjects': 31, 'minimum_paired_trials_per_participant': 30, 'no_behavior_based_exclusion': True}, 'source_store': {'root': str(DISCOVERY_STORE), 'compact': str(DISCOVERY_COMPACT), 'compact_sha256': DISCOVERY_COMPACT_SHA256, 'manifest': str(DISCOVERY_MANIFEST), 'manifest_sha256': DISCOVERY_MANIFEST_SHA256, 'source_cells_per_epoch': 169, 'values_opened_in_semantic_report_dissociation_design': False}, 'representation': {'stages': ['stimulus_late', 'rating_early'], 'blocks': BLOCKS, 'participant_stage_preprocessing': ['for each participant and each stage separately, median-impute each of the 169 compact cells using that participant/stage only', 'subtract that participant/stage feature median', 'divide by that participant/stage feature IQR', 'if stage IQR <= 1e-8, use participant-pooled stimulus_late+rating_early IQR for the same feature', 'if pooled IQR also <= 1e-8, use scale 1.0 and record local constancy', 'winsorize standardized feature values to [-6, 6]'], 'trial_signature': {'per_block': ['RMS displacement between standardized stimulus_late and rating_early profiles', 'cosine similarity between standardized stimulus_late and rating_early profiles'], 'global': ['RMS displacement across all 169 compact cells', 'cosine similarity across all 169 compact cells'], 'positive_distance_transform': 'log1p', 'cosine_transform': 'arctanh after clipping to [-1+1e-6, 1-1e-6]', 'total_signature_cells': 14, 'fixed_coordinate_order': '[RMS, transformed cosine] for each of the six prespecified blocks in discovery_stimulus_experience_multiresolution_fit order, then [global RMS, global transformed cosine]'}, 'cross_person_standardization': ['within participant, robust-center each transformed signature cell by participant median', 'divide by participant IQR; use Discovery participant-balanced pooled IQR when local IQR <= 1e-8', 'winsorize to [-6, 6]'], 'participant_weight': 'each participant contributes total weight 1'}, 'multi_resolution_discovery': {'algorithm': 'participant-weighted KMeans', 'candidate_K': [2, 3, 4, 5, 6, 7, 8], 'seed': 540, 'n_init': 100, 'max_iter': 500, 'tol': 1e-07, 'no_single_K_forced': True, 'all_eligible_resolutions_retained': True, 'canonical_label_rule': 'lexicographic ordering of centroids in the fixed 14-cell coordinate order; never behavioral ordering', 'stability': {'repeats': 200, 'participant_fraction_without_replacement': 0.8, 'trial_fraction_within_retained_participant_without_replacement': 0.8, 'comparison': 'ARI versus corresponding full Discovery K solution', 'eligibility_thresholds': {'median_ARI_at_least': 0.6, 'ARI_10th_percentile_at_least': 0.3, 'collapse_rate_at_most': 0.1, 'minimum_participant_weighted_occupancy': 0.05, 'minimum_participants_per_family': 5, 'minimum_trials_per_family': 30}}, 'continuous_signature_always_retained': True, 'no_eligible_K_rule': 'do not force discrete report-transition families; retain the continuous 14-cell report-transition signature and close only the discrete sub-branch'}, 'structural_comparison_after_discovery_report_transition_multiresolution_fit_lock_only': {'allowed': True, 'experience_reference_K': EXPECTED_K, 'metrics': ['ARI and NMI between each eligible report-transition K and each experience K on common trials', 'participant-balanced contingency tables', 'directional conditional occupancy matrices'], 'purpose': 'quantify whether experience-period organization and report-stage transition organization are coupled; never use the experience labels to choose report-transition K'}, 'data_values_permitted_in_discovery_report_transition_multiresolution_fit': {'stimulus_late_compact_EEG': True, 'rating_early_compact_EEG': True, 'behavioral_reports': False, 'emotion_category': False, 'context_as_target': False, 'stimulus_identity': False, 'physiology': False, 'Validation': False, 'Confirmation': False}, 'forbidden': ['using any report value to choose K or centroids', 'using emotion category/context/stimulus identity to choose K', 'refitting experience K=2/3/4 families', 'calling a report-transition family an emotion or cognitive state', 'interpreting rating_early activity as distortion, dishonesty, suppression, or exaggeration'], 'next_if_eligible_K': 'specification and execute one-shot Validation structural transfer for all eligible report-transition K', 'next_if_no_eligible_K': 'continue with the fixed continuous report-transition signature; do not invent discrete families'}
    writej(project / RT_POLICY, rt_policy)
    semantic_policy = {'phase': PHASE, 'generated_at': now(), 'status': 'future_semantic_tests_prespecified_but_targets_unopened', 'activation_rule': 'No semantic/report test may execute until the report-transition structural branch has been fixed under the semantic_report_dissociation_design/discovery_report_transition_multiresolution_fit lineage and any held-out structural-transfer decisions required by that branch have been prespecified.', 'immutable_experience_resolutions': EXPECTED_K, 'experience_reference_files': {'Discovery_assignments': {'path': str(discovery_stimulus_experience_multiresolution_fit_ASSIGN), 'sha256': discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256}, 'Validation_assignments': {'path': str(validation_stimulus_experience_multiresolution_transfer_ASSIGN), 'sha256': validation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256}, 'Confirmation_assignments': {'path': str(confirmation_stimulus_experience_multiresolution_transfer_ASSIGN), 'sha256': confirmation_stimulus_experience_multiresolution_transfer_ASSIGN_SHA256}, 'Discovery_continuous_signature': {'path': str(discovery_stimulus_experience_multiresolution_fit_SIGNATURE), 'sha256': discovery_stimulus_experience_multiresolution_fit_SIGNATURE_SHA256}, 'Validation_continuous_signature': {'path': str(validation_stimulus_experience_multiresolution_transfer_SIGNATURE), 'sha256': validation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256}, 'Confirmation_continuous_signature': {'path': str(confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE), 'sha256': confirmation_stimulus_experience_multiresolution_transfer_SIGNATURE_SHA256}}, 'Discovery_behavior_reference_not_opened': {'trial_outcome_manifest': {'path': str(DISCOVERY_TRIAL_OUTCOMES), 'sha256': DISCOVERY_TRIAL_OUTCOMES_SHA256}, 'canonical_master': {'path': str(CORRECTED_DISCOVERY_MASTER), 'sha256': CORRECTED_DISCOVERY_MASTER_SHA256}, 'ordered_reports': REPORTS, 'nominal_emotion_categories': NOMINAL_EMOTION_CATEGORIES, 'join_key': 'raw_behavior_row_id with subject_id/trial_number integrity checks'}, 'heldout_behavior_rule': 'Before Validation or Confirmation behavior is opened, a separate lineage/specification must materialize authoritative held-out behavioral/category/context rows from the same prespecified split and corrected event/stimulus lineage. Discovery-selected report subsets, thresholds, family names, or semantic labels are forbidden.', 'hypothesis_A_label_mismatch': {'question': 'Do human nominal emotion labels and 10-dimensional subjective reports map cleanly to the replicated stimulus-only neural partitions, or are the relations many-to-many?', 'A1_nominal_category_family_mapping': {'tested_K': EXPECTED_K, 'primary_metrics': ['participant-balanced normalized mutual information between emotion_category and family', "participant-balanced Cramer's V between emotion_category and family"], 'mixing_outputs': ['participant-balanced P(family | emotion_category)', 'participant-balanced P(emotion_category | family)', 'conditional entropy H(family | emotion_category)', 'conditional entropy H(emotion_category | family)', 'per-category maximum-family concentration', 'per-family maximum-category concentration'], 'null': "10000 within-participant permutations of family labels preserving each participant's family counts", 'cross_split_requirement': 'the same prespecified family IDs and same metrics are reported separately in Discovery, Validation, and Confirmation; no split-specific relabeling or category regrouping', 'interpretation': 'weak/many-to-many mapping is evidence of poor alignment between the human category taxonomy and the neural partition, but non-significance alone is not proof that a label is wrong'}, 'A2_full_report_profile_family_mapping': {'tested_K': EXPECTED_K, 'predictor': 'all 10 reports jointly; no post-hoc subset selection', 'baseline': 'participant-safe nuisance terms only: context + acquisition_fraction + acquisition_fraction_squared', 'model': 'LOPO multinomial logistic regression with fixed L2 regularization grid selected inside training participants only', 'primary_statistic': 'participant-balanced held-out balanced-accuracy improvement over baseline', 'null': '1000 within-participant permutations of the complete 10-report row vector preserving report covariance', 'dimension_followups': 'individual report effects are secondary and only interpreted after the parent 10D test; Holm within the 10 reports'}, 'multiplicity': 'Holm across the six parent tests A1/A2 x K=2,3,4 within each split; held-out replication is evaluated with the exact same parent definitions', 'semantic_naming_rule': 'even a significant association does not rename a family automatically; a family may only receive a semantic label if the mapping is strong, specific, directionally stable across held-out splits, and a later naming rule has been separately prespecified'}, 'hypothesis_B_report_transition': {'question': 'Does stimulus_late -> rating_early neural transformation carry report-related information beyond the already replicated stimulus-only experience representation?', 'B0_outcome_free_structure_first': 'report-transition structure must be discovered/fixed without behavior under the RT policy above', 'B1_experience_transition_structural_coupling': {'behavior_used': False, 'metrics': ['ARI/NMI between experience and report-transition discrete resolutions when available', 'participant-balanced contingency and conditional occupancy matrices', 'continuous canonical correlation is forbidden unless separately prespecified']}, 'B2_incremental_report_mapping_primary': {'target': 'joint 10-dimensional report vector', 'outer_validation': 'leave-one-participant-out', 'training_only_target_preprocessing': 'for each report dimension, standardize using training participants only; no test-participant outcome centering/calibration', 'M0': 'context + acquisition_fraction + acquisition_fraction_squared', 'M1': 'M0 + prespecified 14-cell continuous experience signature', 'M2': 'M1 + prespecified 14-cell continuous report-transition signature', 'estimator': 'multivariate ridge; alpha grid [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000] selected inside training participants only', 'primary_statistic': 'participant-balanced held-out multivariate SSE reduction / R2 improvement of M2 over M1', 'null': '1000 within-participant permutations of the complete 10-report row vector, preserving covariance among reports', 'secondary': 'dimension-specific Delta-R2 for the 10 reports, gated by the parent global test and Holm-corrected across reports', 'interpretation': 'positive support means rating/report-stage neural transformation contains information about the eventual report beyond the stimulus-only representation. It does NOT establish distortion, dishonesty, conscious editing, suppression, or exaggeration.'}, 'B3_discrete_transition_family_report_mapping': {'activation': 'only if one or more report-transition K are structurally replicated', 'method': 'apply the same 10D family-report mapping framework to each replicated transition K; no K is chosen based on report strength', 'status': 'secondary/gated'}}, 'global_boundaries': ['No emotion/report value may influence any EEG preprocessing, K choice, centroid, support threshold, or structural gate.', 'No family may be renamed from Discovery behavior and then treated as having that meaning in held-out splits.', 'The dataset contains joy/inspiration/tenderness/sadness/fear/disgust/neutral categories; it does not contain anger as a nominal category.', 'The report-transition hypothesis concerns neural transformation around evaluation/report generation, not truthfulness of the participant.', 'Physiology remains outside the primary structural_replication semantic program unless a separately prespecified synchronized representation becomes available.']}
    writej(project / SEMANTIC_POLICY, semantic_policy)
    registry_rows = [{'branch': 'A', 'test_id': 'A1', 'status_in_semantic_report_dissociation_design': 'prespecified_NOT_EXECUTED', 'target': 'nominal emotion_category', 'predictor_or_partition': 'experience families K=2,3,4', 'primary_metric': "participant-balanced NMI + Cramer's V", 'permutations': 10000, 'behavior_opened_in_semantic_report_dissociation_design': False}, {'branch': 'A', 'test_id': 'A2', 'status_in_semantic_report_dissociation_design': 'prespecified_NOT_EXECUTED', 'target': 'joint 10D reports', 'predictor_or_partition': 'experience families K=2,3,4', 'primary_metric': 'LOPO balanced-accuracy increment over nuisance baseline', 'permutations': 1000, 'behavior_opened_in_semantic_report_dissociation_design': False}, {'branch': 'B', 'test_id': 'B1', 'status_in_semantic_report_dissociation_design': 'prespecified_NOT_EXECUTED', 'target': 'none/outcome-free', 'predictor_or_partition': 'experience vs report-transition structure', 'primary_metric': 'ARI/NMI + conditional occupancy', 'permutations': 0, 'behavior_opened_in_semantic_report_dissociation_design': False}, {'branch': 'B', 'test_id': 'B2', 'status_in_semantic_report_dissociation_design': 'prespecified_NOT_EXECUTED', 'target': 'joint 10D reports', 'predictor_or_partition': 'experience signature + incremental report-transition signature', 'primary_metric': 'LOPO multivariate Delta-R2 / SSE reduction', 'permutations': 1000, 'behavior_opened_in_semantic_report_dissociation_design': False}, {'branch': 'B', 'test_id': 'B3', 'status_in_semantic_report_dissociation_design': 'prespecified_NOT_EXECUTED_GATED', 'target': 'joint 10D reports', 'predictor_or_partition': 'replicated report-transition families if any', 'primary_metric': '10D family-report mapping', 'permutations': 1000, 'behavior_opened_in_semantic_report_dissociation_design': False}]
    registry_path = out / 'tables/semantic_report_dissociation_design_prespecified_hypothesis_registry.csv'
    write_csv(registry_path, registry_rows, ['branch', 'test_id', 'status_in_semantic_report_dissociation_design', 'target', 'predictor_or_partition', 'primary_metric', 'permutations', 'behavior_opened_in_semantic_report_dissociation_design'])
    source_ref_rows = [{'role': 'Discovery compact EEG for discovery_report_transition_multiresolution_fit', 'path': str(DISCOVERY_COMPACT), 'sha256': DISCOVERY_COMPACT_SHA256, 'opened_in_semantic_report_dissociation_design': False}, {'role': 'Discovery epoch manifest for discovery_report_transition_multiresolution_fit', 'path': str(DISCOVERY_MANIFEST), 'sha256': DISCOVERY_MANIFEST_SHA256, 'opened_in_semantic_report_dissociation_design': False}, {'role': 'Discovery reports for future semantic phase', 'path': str(DISCOVERY_TRIAL_OUTCOMES), 'sha256': DISCOVERY_TRIAL_OUTCOMES_SHA256, 'opened_in_semantic_report_dissociation_design': False}, {'role': 'Corrected Discovery category/stimulus lineage for future semantic phase', 'path': str(CORRECTED_DISCOVERY_MASTER), 'sha256': CORRECTED_DISCOVERY_MASTER_SHA256, 'opened_in_semantic_report_dissociation_design': False}]
    source_ref_path = out / 'tables/semantic_report_dissociation_design_prespecified_source_references.csv'
    write_csv(source_ref_path, source_ref_rows, ['role', 'path', 'sha256', 'opened_in_semantic_report_dissociation_design'])
    rehash_path = out / 'qc/semantic_report_dissociation_design_source_rehash.csv'
    write_csv(rehash_path, rehash_rows, ['source_phase', 'path', 'expected_sha256', 'observed_sha256', 'verified'])
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'semantic_report_dissociation_program_prespecified', 'structurally_replicated_experience_K': EXPECTED_K, 'report_transition_discovery_pairs_all': 1252, 'report_transition_discovery_pairs_eligible': 1237, 'report_transition_discovery_eligible_subjects': 31, 'report_transition_candidate_K': [2, 3, 4, 5, 6, 7, 8], 'new_signal_values_opened': False, 'rating_early_EEG_values_opened': False, 'behavioral_report_values_opened': False, 'emotion_category_values_opened': False, 'context_target_values_opened': False, 'stimulus_identity_values_opened': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_role': 'execute Discovery-only outcome-free stimulus_late-to-rating_early multi-resolution report-transition structure', 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/semantic_report_dissociation_design_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': sha256(Path(__file__).resolve()), 'verified_parent_locks': {'cross_split_experience_structure_synthesis': cross_split_experience_structure_synthesis_LOCK_SHA256, 'multiresolution_experience_report_design': multiresolution_experience_report_design_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit': discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256, 'validation_stimulus_experience_multiresolution_transfer': validation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer': confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256}, 'prespecified_policy_sha256': {'discovery_report_transition_multiresolution_fit_report_transition_policy': sha256(project / RT_POLICY), 'future_semantic_policy': sha256(project / SEMANTIC_POLICY)}, 'boundary': {'new_signal_values_opened': False, 'rating_early_EEG_values_opened': False, 'behavioral_report_values_opened': False, 'emotion_category_values_opened': False, 'context_target_values_opened': False, 'stimulus_identity_values_opened': False, 'semantic_relabeling_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/semantic_report_dissociation_design_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/semantic_report_dissociation_design_report.md'
    report_path.write_text(''.join(['# semantic_report_dissociation_design — Semantic/report dissociation analysis specification', '', '**semantic_report_dissociation_design opened no new EEG feature values, no `rating_early` EEG values, and no behavioral/category/context/stimulus target values.**', '', '## fixed starting point', '', '- Stimulus-only experience partitions **K=2, K=3, K=4** are structurally replicated across Discovery, Validation, and Confirmation.', '- They remain structural labels only.', '- No single K is declared the true biological resolution.', '', '## Branch B0 — report-transition structure first', '', 'discovery_report_transition_multiresolution_fit is now prespecified to analyze **`stimulus_late -> rating_early` only**, using the same 169-cell compact representation and the same 14-cell magnitude/profile signature logic used for the experience branch.', '', '- Discovery pair rows available: **1252**', '- prespecified eligible rows: **1237**', '- prespecified eligible participants: **31**', '- Candidate K: **2 through 8**', '- Same stability/collapse thresholds as the experience branch', '- No behavioral values may enter discovery_report_transition_multiresolution_fit', '', 'This cleanly separates experience-period organization from the transition into rating/report generation.', '', '## Branch A — human-label mismatch tests', '', 'The future semantic policy is prespecified now, before targets are opened.', '', 'For each replicated experience resolution K=2/3/4, the program will quantify:', '', '- nominal emotion-category ↔ family dependence;', '- `P(family | category)` and `P(category | family)` mixing;', '- conditional entropies and concentration measures;', '- joint 10-report ability to predict family in participant-held-out evaluation.', '', 'A many-to-many or weak mapping may support poor alignment between the human taxonomy and the neural organization, but **a null p-value by itself will never be called proof that the human label is wrong**.', '', '## Branch B — report-stage transformation tests', '', 'After report-transition structure is fixed, the primary behavioral test will ask whether the continuous `stimulus_late -> rating_early` signature improves held-out prediction of the joint 10-report vector beyond the continuous stimulus-only experience signature.', '', 'A positive result would mean report/evaluation-stage neural transformation contains report-related information beyond the experience representation. It would **not** prove that participants distort, hide, exaggerate, or consciously edit their feelings.', '', '## Important dataset boundary', '', 'The nominal categories available the following are joy, inspiration, tenderness, sadness, fear, disgust, and neutral. **Anger is not a nominal category in this dataset.**', '', '## Next', '', f'**{NEXT_PHASE}** — Discovery-only outcome-free multi-resolution `stimulus_late -> rating_early` structural fit.', '']) + '', encoding='utf-8')
    lock_targets = [project / RT_POLICY, project / SEMANTIC_POLICY, contract_path, summary_path, rehash_path, prespecified_rt_manifest, registry_path, source_ref_path, report_path]
    final_lock = out / 'qc/semantic_report_dissociation_design_final_output_lock_hashes.json'
    writej(final_lock, {'phase': PHASE, 'generated_at': now(), 'status': 'semantic_report_dissociation_program_prespecified', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'new_signal_values_opened': False, 'rating_early_EEG_values_opened': False, 'behavioral_report_values_opened': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE})
    script_dir = Path(__file__).resolve().parent
    print('')
    print('semantic_report_dissociation_design PASS')
    print(f'Replicated experience K prespecified: {EXPECTED_K}')
    print('Report-transition candidate K: [2, 3, 4, 5, 6, 7, 8]')
    print('Report-transition eligible Discovery pairs: 1237')
    print('Report-transition eligible Discovery participants: 31')
    print('New EEG signal values opened: False')
    print('rating_early EEG values opened: False')
    print('Behavioral report values opened: False')
    print('Emotion-category values opened: False')
    print('Semantic relabeling permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())