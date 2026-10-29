# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
PHASE = 'validation_report_transition_transfer_design'
NEXT_PHASE = 'validation_report_transition_multiresolution_transfer'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 540
IQR_FLOOR = 1e-08
ELIGIBLE_K = [2, 3]
AUTHORITATIVE_VALIDATION_TRIALS = 494
discovery_report_transition_multiresolution_fit_OUT = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit')
discovery_report_transition_multiresolution_fit_LOCK = discovery_report_transition_multiresolution_fit_OUT / 'qc/discovery_report_transition_multiresolution_fit_final_output_lock_hashes.json'
discovery_report_transition_multiresolution_fit_LOCK_SHA256 = 'aa09937f090a17eadac760c38ebbe36310c5e88af2e92429f8e4680ca59a1473'
discovery_report_transition_multiresolution_fit_SUMMARY = discovery_report_transition_multiresolution_fit_OUT / 'qc/discovery_report_transition_multiresolution_fit_summary.json'
discovery_report_transition_multiresolution_fit_SUMMARY_SHA256 = '28d5ca2a2f4a6309554c58473829411ff554d6066dd2b6915fd0eac780fabf87'
discovery_report_transition_multiresolution_fit_SIGNATURES = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv'
discovery_report_transition_multiresolution_fit_SIGNATURES_SHA256 = '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83'
discovery_report_transition_multiresolution_fit_ASSIGN = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'
discovery_report_transition_multiresolution_fit_ASSIGN_SHA256 = '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'
discovery_report_transition_multiresolution_fit_CENT = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_family_centroids_all_k.csv'
discovery_report_transition_multiresolution_fit_CENT_SHA256 = 'a9484298b3a55bdec096b7e09efb1fb2a770901fd41f9f5c00579c26dc336cff'
discovery_report_transition_multiresolution_fit_METRICS = discovery_report_transition_multiresolution_fit_OUT / 'tables/discovery_report_transition_multiresolution_fit_family_candidate_metrics.csv'
discovery_report_transition_multiresolution_fit_METRICS_SHA256 = '3e4a50f8157a9b8a6a58c15ade3cad901d76d324a0f736dd35b932ee4427d669'
multiresolution_experience_report_design_PROGRAM_POLICY = Path('analysis_specifications/S045_multiresolution_experience_report.json')
multiresolution_experience_report_design_PROGRAM_POLICY_SHA256 = '2642a87c60f5082276e65ef12c7de81037dfbbbb628666e458a366af56e81927'
validation_experience_transfer_design_SOURCE = Path('src/analysis_pipeline/eeg/validation_experience_transfer_design/validation_experience_transfer_design.py')
validation_experience_transfer_design_SOURCE_SHA256 = '01ac4d085aa56c82e6e1536c852363a662e7ce167e801e7cc006c93f5468ebe2'
validation_feature_trajectory_LOCK = Path('outputs/qc/validation_feature_trajectory_final_output_lock_hashes.json')
validation_feature_trajectory_LOCK_SHA256 = 'ab18935f81020fcc0d8596d63f8a6d3021c3e19285ac075d3c56c2b566cbfabe'
VALIDATION_STORE_ROOT = Path('outputs/large/validation_feature_trajectory_validation_compact_features_v1')
VALIDATION_COMPACT = VALIDATION_STORE_ROOT / 'global/epoch_compact_float32.npy'
VALIDATION_MANIFEST = VALIDATION_STORE_ROOT / 'global/epoch_manifest.csv'
VALIDATION_COMPACT_SHA256 = '6f8ab1755f8d950a33351d2032ccd190092656923b09b701ab04e78430fec402'
VALIDATION_MANIFEST_SHA256 = '24643e775d90d30011fbbba7d40defedcbdb699632603c2d838b420adb49c1a8'
OUT = Path('outputs/interpretability/validation_report_transition_transfer_design_validation_report_transition_transfer_design_specification')
CONFIG = Path('analysis_specifications/S087_validation_report_transition_transfer.json')
COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
TR = [f'transformed__{x}' for x in COORDS]
STD = [f'standardized__{x}' for x in COORDS]

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

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def participant_weights(subjects: np.ndarray) -> np.ndarray:
    vc = pd.Series(subjects.astype(str)).value_counts().to_dict()
    return np.array([1.0 / vc[str(s)] for s in subjects], dtype=float)

def import_validation_experience_transfer_design_core(project: Path):
    source = project / validation_experience_transfer_design_SOURCE
    verify(source, validation_experience_transfer_design_SOURCE_SHA256, 'fixed validation_experience_transfer_design numerical helper source')
    spec = importlib.util.spec_from_file_location('validation_experience_transfer_design_fixed_numeric_core', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not construct import spec for validation_experience_transfer_design numerical helper')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod

def verify_discovery_report_transition_multiresolution_fit(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify(project / discovery_report_transition_multiresolution_fit_LOCK, discovery_report_transition_multiresolution_fit_LOCK_SHA256, 'discovery_report_transition_multiresolution_fit final lock')
    lock = readj(project / discovery_report_transition_multiresolution_fit_LOCK)
    if lock.get('phase') != 'discovery_report_transition_multiresolution_fit':
        raise RuntimeError('discovery_report_transition_multiresolution_fit final-lock phase drift')
    if lock.get('next_phase') != PHASE:
        raise RuntimeError(f'discovery_report_transition_multiresolution_fit does not authorize {PHASE}')
    if lock.get('eligible_K') != ELIGIBLE_K:
        raise RuntimeError(f"discovery_report_transition_multiresolution_fit eligible-K drift: {lock.get('eligible_K')}")
    for key in ['behavioral_values_read', 'emotion_category_values_read', 'context_as_target_values_read', 'stimulus_identity_as_target_values_read', 'physiology_values_read', 'Validation_signal_values_read', 'Confirmation_signal_values_read', 'experience_assignments_used_to_choose_transition_K', 'semantic_relabeling_permitted']:
        if bool(lock.get(key)):
            raise RuntimeError(f'discovery_report_transition_multiresolution_fit boundary drift: {key}=True')
    for rel, expected in lock.get('hashes', {}).items():
        verify(project / rel, expected, f'discovery_report_transition_multiresolution_fit fixed child {rel}')
    for rel, expected, label in [(discovery_report_transition_multiresolution_fit_SUMMARY, discovery_report_transition_multiresolution_fit_SUMMARY_SHA256, 'discovery_report_transition_multiresolution_fit summary'), (discovery_report_transition_multiresolution_fit_SIGNATURES, discovery_report_transition_multiresolution_fit_SIGNATURES_SHA256, 'discovery_report_transition_multiresolution_fit transition signatures'), (discovery_report_transition_multiresolution_fit_ASSIGN, discovery_report_transition_multiresolution_fit_ASSIGN_SHA256, 'discovery_report_transition_multiresolution_fit transition assignments'), (discovery_report_transition_multiresolution_fit_CENT, discovery_report_transition_multiresolution_fit_CENT_SHA256, 'discovery_report_transition_multiresolution_fit transition centroids'), (discovery_report_transition_multiresolution_fit_METRICS, discovery_report_transition_multiresolution_fit_METRICS_SHA256, 'discovery_report_transition_multiresolution_fit transition candidate metrics')]:
        verify(project / rel, expected, label)
    summary = readj(project / discovery_report_transition_multiresolution_fit_SUMMARY)
    if summary.get('eligible_K') != ELIGIBLE_K:
        raise RuntimeError('discovery_report_transition_multiresolution_fit summary eligible-K drift')
    return (lock, summary)

def lopo_reference(core, sig: pd.DataFrame, assignments: pd.DataFrame, centroids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    subjects = sig.subject_id.astype(str).to_numpy()
    fullw = participant_weights(subjects)
    full_pool = core.weighted_iqr(sig[TR].to_numpy(float), fullw)
    full_pool = np.where(np.isfinite(full_pool), full_pool, 0.0)
    rows: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for k in ELIGIBLE_K:
        af = assignments[assignments.k.astype(int).eq(k)].copy()
        amap = dict(zip(af.raw_behavior_row_id.astype(str), af.family_1based.astype(int) - 1))
        cdf = centroids[centroids.k.astype(int).eq(k)].sort_values('family_1based')
        fullc = cdf[STD].to_numpy(float)
        if fullc.shape != (k, 14):
            raise RuntimeError(f'discovery_report_transition_multiresolution_fit centroid shape drift for K={k}: {fullc.shape}')
        for si, subject in enumerate(pd.unique(subjects)):
            train = sig[sig.subject_id.astype(str).ne(str(subject))].copy()
            test = sig[sig.subject_id.astype(str).eq(str(subject))].copy()
            tr_sub = train.subject_id.astype(str).to_numpy()
            trw = participant_weights(tr_sub)
            pool = core.weighted_iqr(train[TR].to_numpy(float), trw)
            pool = np.where(np.isfinite(pool), pool, 0.0)
            xtr = core.standardize_from_transformed(train, pool)
            xte = core.standardize_from_transformed(test, pool)
            _, ctr = core.fit_kmeans(xtr, trw, k, SEED + 3000 * k + si)
            cost = np.sum((ctr[:, None, :] - fullc[None, :, :]) ** 2, axis=2)
            rr, cc = linear_sum_assignment(cost)
            aligned = np.empty_like(ctr)
            for r0, outcome_free_features in zip(rr, cc):
                aligned[outcome_free_features] = ctr[r0]
            lab, dist, margin, _ = core.assign(xte, aligned)
            full = np.array([amap[str(x)] for x in test.raw_behavior_row_id.astype(str)], dtype=int)
            agree = lab == full
            for rid, aa, ll, ff, dd, mm in zip(test.raw_behavior_row_id.astype(str), agree, lab, full, dist, margin):
                rows.append({'k': k, 'subject_id': str(subject), 'raw_behavior_row_id': rid, 'lopo_family_1based': int(ll + 1), 'full_fit_family_1based': int(ff + 1), 'agree': bool(aa), 'lopo_assigned_squared_distance': float(dd), 'lopo_assignment_margin': float(mm)})
        frame = pd.DataFrame([x for x in rows if int(x['k']) == k])
        agreeing = frame[frame.agree.astype(bool)]
        if len(agreeing) < 100:
            raise RuntimeError(f'Too few LOPO-agreeing rows for K={k}')
        subj_ag = frame.groupby('subject_id').agree.mean()
        support.append({'k': k, 'lopo_trial_agreement': float(frame.agree.mean()), 'lopo_participant_balanced_agreement': float(subj_ag.mean()), 'lopo_min_participant_agreement': float(subj_ag.min()), 'lopo_agreeing_trials': int(len(agreeing)), 'in_support_squared_distance_max': float(np.quantile(agreeing.lopo_assigned_squared_distance, 0.995, method='linear')), 'low_margin_threshold': float(np.quantile(agreeing.lopo_assignment_margin, 0.01, method='linear')), 'distance_reference_rule': '99.5th percentile of LOPO-agreeing held-out assigned squared distance', 'margin_reference_rule': '1st percentile of LOPO-agreeing held-out assignment margin'})
    return (pd.DataFrame(rows), pd.DataFrame(support), full_pool)

def self_test() -> int:
    assert ELIGIBLE_K == [2, 3]
    assert len(COORDS) == 14
    assert PHASE == 'validation_report_transition_transfer_design'
    assert NEXT_PHASE == 'validation_report_transition_multiresolution_transfer'
    print('validation_report_transition_transfer_design self-test PASS')
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
    print(f'[{now()}] verifying fixed discovery_report_transition_multiresolution_fit transition discovery', flush=True)
    _, discovery_report_transition_multiresolution_fit_summary = verify_discovery_report_transition_multiresolution_fit(project)
    core = import_validation_experience_transfer_design_core(project)
    verify(project / multiresolution_experience_report_design_PROGRAM_POLICY, multiresolution_experience_report_design_PROGRAM_POLICY_SHA256, 'multiresolution_experience_report_design program policy')
    program = readj(project / multiresolution_experience_report_design_PROGRAM_POLICY)
    held = program['heldout_store_references_not_opened_in_multiresolution_experience_report_design']['Validation']
    expected_held = {'root': str(VALIDATION_STORE_ROOT), 'compact_sha256': VALIDATION_COMPACT_SHA256, 'manifest_sha256': VALIDATION_MANIFEST_SHA256, 'historical_final_lock_sha256': validation_feature_trajectory_LOCK_SHA256}
    if held != expected_held:
        raise RuntimeError(f'Historical Validation store reference drift: {held}')
    verify(project / validation_feature_trajectory_LOCK, validation_feature_trajectory_LOCK_SHA256, 'historical validation_feature_trajectory Validation final lock')
    sig = pd.read_csv(project / discovery_report_transition_multiresolution_fit_SIGNATURES, low_memory=False)
    ass = pd.read_csv(project / discovery_report_transition_multiresolution_fit_ASSIGN, low_memory=False)
    cent = pd.read_csv(project / discovery_report_transition_multiresolution_fit_CENT, low_memory=False)
    met = pd.read_csv(project / discovery_report_transition_multiresolution_fit_METRICS, low_memory=False)
    if len(sig) != 1237 or sig.subject_id.astype(str).nunique() != 31:
        raise RuntimeError(f'discovery_report_transition_multiresolution_fit signature cohort drift: n={len(sig)}, subjects={sig.subject_id.nunique()}')
    observed_eligible = met.loc[met.eligible.astype(bool), 'k'].astype(int).sort_values().tolist()
    if observed_eligible != ELIGIBLE_K:
        raise RuntimeError(f'discovery_report_transition_multiresolution_fit metric eligibility drift: {observed_eligible}')
    out = project / OUT
    final = out / 'qc/validation_report_transition_transfer_design_final_output_lock_hashes.json'
    if final.exists() and (not args.force):
        raise RuntimeError(f'validation_report_transition_transfer_design already materialized: {final} (use --force only to rematerialize the same design)')
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] computing Discovery-only LOPO support references for transition K={ELIGIBLE_K}', flush=True)
    lopo, support, pool = lopo_reference(core, sig, ass, cent)
    keep_cent = cent[cent.k.astype(int).isin(ELIGIBLE_K)].copy().sort_values(['k', 'family_1based']).reset_index(drop=True)
    fallback = pd.DataFrame({'coordinate': COORDS, 'discovery_participant_balanced_pooled_iqr': pool})
    gate_rows: list[dict[str, Any]] = []
    for k in ELIGIBLE_K:
        gates = [('source_hashes', 'all source/store hash gates pass', 'boolean', True), ('eligible_participants', 'eligible Validation participants', '>=', 9), ('paired_fraction', 'paired report-transition trials / authoritative Validation trials', '>=', 0.75), ('in_support', 'participant-balanced in-support fraction', '>=', 0.8), ('family_participants', 'minimum distinct participants per family', '>=', 5), ('family_trials', 'minimum trials per family', '>=', 30), ('family_occupancy', 'minimum participant-balanced occupancy per family', '>=', 0.05), ('no_behavior', 'behavior/category/context/stimulus target mapping executed', '==', False)]
        for gid, metric, comp, threshold in gates:
            gate_rows.append({'k': k, 'gate_id': gid, 'metric': metric, 'comparison': comp, 'threshold': threshold})
    gates = pd.DataFrame(gate_rows)
    policy = {'phase': PHASE, 'generated_at': now(), 'scientific_role': 'specification one-shot Validation transfer of Discovery outcome-free stimulus_late->rating_early report-transition families before Validation signal values are opened', 'fixed_before_validation_signal_access': True, 'eligible_K': ELIGIBLE_K, 'all_eligible_K_tested_independently': True, 'no_resolution_promoted_for_simplicity': True, 'representation': {'source_cells_per_epoch': 169, 'stages': ['stimulus_late', 'rating_early'], 'signature_cells': 14, 'coordinate_order': COORDS, 'behavior_forbidden': True, 'validation_signature_preprocessing': 'exact discovery_report_transition_multiresolution_fit/discovery_stimulus_experience_multiresolution_fit participant-stage preprocessing and transformed-signature within-participant robust scaling; use prespecified Discovery participant-balanced pooled transformed-signature IQR only if a Validation participant local signature IQR <= 1e-8', 'prespecified_discovery_fallback_iqr': {c: float(v) for c, v in zip(COORDS, pool)}}, 'fixed_assignment': {str(k): {'centroids': keep_cent[keep_cent.k.astype(int).eq(k)].sort_values('family_1based')[STD].to_numpy(float).tolist(), 'support': support[support.k.astype(int).eq(k)].iloc[0][['in_support_squared_distance_max', 'low_margin_threshold', 'distance_reference_rule', 'margin_reference_rule']].to_dict(), 'distance': 'squared Euclidean in the prespecified 14-cell standardized report-transition signature', 'tie_rule': 'lowest numeric family', 'refit': False, 'rename': False} for k in ELIGIBLE_K}, 'validation_store': held, 'participant_eligibility': {'minimum_paired_stimulus_late_rating_early_trials': 16, 'no_behavior_based_exclusion': True, 'nonfinite_signature_rule': 'exclude trial before assignment and record reason', 'participant_below_minimum_rule': 'exclude participant from transfer gate but retain audit'}, 'structural_transfer_gate': {'evaluated_separately_for_each_K': True, 'minimum_eligible_participants': 9, 'minimum_paired_report_transition_fraction_of_authoritative_validation_trials': 0.75, 'minimum_participant_balanced_in_support_fraction': 0.8, 'minimum_distinct_participants_per_family': 5, 'minimum_trials_per_family': 30, 'minimum_participant_balanced_occupancy_per_family': 0.05, 'all_source_hash_gates_must_pass': True, 'no_behavior_category_context_stimulus_target_mapping': True, 'K_replication_rule': 'all gates for that K must pass', 'failure_rule': 'failed K is not retuned, renamed, merged, split, or advanced'}, 'post_validation_rule': {'if_one_or_more_K_pass': 'only passing transition K may later be considered for an independently prespecified Confirmation structural transfer; Confirmation is not opened in validation_report_transition_transfer_design', 'if_no_K_pass': 'close the discrete report-transition transfer branch; retain the prespecified continuous 14-cell transition signature', 'semantic_tests': 'remain closed until the report-transition structural result is fixed'}, 'forbidden': ['Validation behavioral reports', 'emotion category as target', 'context as target', 'stimulus identity as target', 'centroid refit', 'K reselection', 'threshold tuning', 'family semantic naming', 'claims of distortion/dishonesty/suppression/exaggeration', 'Confirmation signal access'], 'next_phase': NEXT_PHASE}
    writej(project / CONFIG, policy)
    policy_sha = canonical_sha(policy)
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_lopo_support_reference.csv', lopo.sort_values(['k', 'subject_id', 'raw_behavior_row_id']))
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_support_thresholds.csv', support.sort_values('k'))
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_prespecified_centroids.csv', keep_cent)
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_discovery_fallback_iqr.csv', fallback)
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_validation_gate_spec.csv', gates)
    store_ref = pd.DataFrame([{'role': 'compact', 'path': str(VALIDATION_COMPACT), 'sha256': VALIDATION_COMPACT_SHA256, 'values_opened_in_validation_report_transition_transfer_design': False}, {'role': 'manifest', 'path': str(VALIDATION_MANIFEST), 'sha256': VALIDATION_MANIFEST_SHA256, 'values_opened_in_validation_report_transition_transfer_design': False}, {'role': 'historical_final_lock', 'path': str(validation_feature_trajectory_LOCK), 'sha256': validation_feature_trajectory_LOCK_SHA256, 'values_opened_in_validation_report_transition_transfer_design': False}])
    atomic_csv(out / 'tables/validation_report_transition_transfer_design_validation_store_reference.csv', store_ref)
    source_rows = []
    for name, rel, expected in [('discovery_report_transition_multiresolution_fit_final_lock', discovery_report_transition_multiresolution_fit_LOCK, discovery_report_transition_multiresolution_fit_LOCK_SHA256), ('discovery_report_transition_multiresolution_fit_summary', discovery_report_transition_multiresolution_fit_SUMMARY, discovery_report_transition_multiresolution_fit_SUMMARY_SHA256), ('discovery_report_transition_multiresolution_fit_signatures', discovery_report_transition_multiresolution_fit_SIGNATURES, discovery_report_transition_multiresolution_fit_SIGNATURES_SHA256), ('discovery_report_transition_multiresolution_fit_assignments', discovery_report_transition_multiresolution_fit_ASSIGN, discovery_report_transition_multiresolution_fit_ASSIGN_SHA256), ('discovery_report_transition_multiresolution_fit_centroids', discovery_report_transition_multiresolution_fit_CENT, discovery_report_transition_multiresolution_fit_CENT_SHA256), ('discovery_report_transition_multiresolution_fit_metrics', discovery_report_transition_multiresolution_fit_METRICS, discovery_report_transition_multiresolution_fit_METRICS_SHA256), ('multiresolution_experience_report_design_program_policy', multiresolution_experience_report_design_PROGRAM_POLICY, multiresolution_experience_report_design_PROGRAM_POLICY_SHA256), ('validation_experience_transfer_design_numeric_source', validation_experience_transfer_design_SOURCE, validation_experience_transfer_design_SOURCE_SHA256), ('validation_feature_trajectory_historical_lock', validation_feature_trajectory_LOCK, validation_feature_trajectory_LOCK_SHA256)]:
        p = project / rel
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'Unexpected source drift while writing audit: {rel}')
        source_rows.append({'source': name, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': True, 'bytes': p.stat().st_size})
    atomic_csv(out / 'qc/validation_report_transition_transfer_design_source_rehash.csv', pd.DataFrame(source_rows))
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_report_transition_transfer_design_prespecified', 'eligible_transition_K': ELIGIBLE_K, 'discovery_report_transition_multiresolution_fit_status': discovery_report_transition_multiresolution_fit_summary.get('status'), 'validation_report_transition_transfer_design_policy_canonical_sha256': policy_sha, 'lopo_support_reference': support.to_dict('records'), 'Validation_signal_values_read': False, 'Validation_manifest_parsed': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_role': 'one-shot Validation stimulus_late-to-rating_early structural transfer for K=2,3 using prespecified discovery_report_transition_multiresolution_fit templates and validation_report_transition_transfer_design support/gates', 'runtime_seconds': time.time() - started}
    writej(out / 'qc/validation_report_transition_transfer_design_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'design_prespecified', 'source_code_sha256': sha256(Path(__file__).resolve()), 'discovery_report_transition_multiresolution_fit_final_lock_sha256': discovery_report_transition_multiresolution_fit_LOCK_SHA256, 'validation_report_transition_transfer_design_policy_canonical_sha256': policy_sha, 'validation_experience_transfer_design_numeric_source_sha256': validation_experience_transfer_design_SOURCE_SHA256, 'eligible_transition_K': ELIGIBLE_K, 'Validation_signal_values_read': False, 'Validation_manifest_parsed': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/validation_report_transition_transfer_design_contract.json', contract)
    report_lines = ['# validation_report_transition_transfer_design — Validation report-transition transfer analysis specification', '', '**Status: design prespecified; Validation signal values remain unopened.**', '', '## Discovery result carried forward', '', 'Eligible outcome-free `stimulus_late -> rating_early` resolutions: **K=2 and K=3**.', 'K=4..8 remain failed under the prespecified semantic_report_dissociation_design/discovery_report_transition_multiresolution_fit stability criteria and are not rescued.', '', '## Discovery-only LOPO support reference', '', '```text', support.to_string(index=False), '```', '', 'The support-distance ceiling is the 99.5th percentile of LOPO-agreeing held-out Discovery distances; the low-margin threshold is the 1st percentile. Validation does not contribute to either threshold.', '', '## Validation rule', '', 'K=2 and K=3 will be transferred independently with no refit, no relabeling, and no threshold tuning. A K replicates only if every prespecified structural gate passes.', '', 'The prespecified representation uses only `stimulus_late` and `rating_early`; behavioral reports, emotion category, context, and stimulus identity remain forbidden.', '', '## Interpretation boundary', '', "Even a successful held-out transfer would establish a reproducible report/evaluation-stage EEG reconfiguration partition only. It would not prove distortion, dishonesty, suppression, exaggeration, or conscious editing of the participant's report.", '', f'## Next', '', f'**{NEXT_PHASE}** — one-shot Validation report-transition structural transfer for K=2/3.', '']
    report_path = out / 'reports/validation_report_transition_transfer_design_report.md'
    report_path.write_text(''.join(report_lines), encoding='utf-8')
    targets = [project / CONFIG, out / 'qc/validation_report_transition_transfer_design_contract.json', out / 'qc/validation_report_transition_transfer_design_summary.json', out / 'qc/validation_report_transition_transfer_design_source_rehash.csv', out / 'tables/validation_report_transition_transfer_design_lopo_support_reference.csv', out / 'tables/validation_report_transition_transfer_design_support_thresholds.csv', out / 'tables/validation_report_transition_transfer_design_prespecified_centroids.csv', out / 'tables/validation_report_transition_transfer_design_discovery_fallback_iqr.csv', out / 'tables/validation_report_transition_transfer_design_validation_gate_spec.csv', out / 'tables/validation_report_transition_transfer_design_validation_store_reference.csv', report_path]
    hashes = {str(p.relative_to(project)): sha256(p) for p in targets}
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_report_transition_transfer_design_prespecified', 'hashes': hashes, 'eligible_transition_K': ELIGIBLE_K, 'validation_report_transition_transfer_design_policy_canonical_sha256': policy_sha, 'Validation_signal_values_read': False, 'Validation_manifest_parsed': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE}
    final_lock = out / 'qc/validation_report_transition_transfer_design_final_output_lock_hashes.json'
    writej(final_lock, final_obj)
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('validation_report_transition_transfer_design PASS')
    print(f'Eligible transition K prespecified for Validation: {ELIGIBLE_K}')
    for _, row in support.sort_values('k').iterrows():
        print(f'K={int(row.k)} LOPO participant-balanced agreement={row.lopo_participant_balanced_agreement:.6f}, support_distance_max={row.in_support_squared_distance_max:.6f}, low_margin={row.low_margin_threshold:.6f}')
    print('Validation signal values opened: False')
    print('Validation manifest parsed: False')
    print('Validation behavioral values opened: False')
    print('Confirmation opened: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())