# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
PHASE = 'validation_experience_transfer_design'
NEXT_PHASE = 'validation_stimulus_experience_multiresolution_transfer'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 540
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ELIGIBLE_K = [2, 3, 4]
multiresolution_experience_report_design_LOCK = Path('outputs/interpretability/multiresolution_experience_report_design_multiresolution_experience_report_design_specification/qc/multiresolution_experience_report_design_final_output_lock_hashes.json')
multiresolution_experience_report_design_LOCK_SHA256 = 'ae65247138d1aa19cbafa05542702031d273b77503e569146710e2db1d8d467a'
multiresolution_experience_report_design_PROGRAM_POLICY = Path('analysis_specifications/S045_multiresolution_experience_report.json')
multiresolution_experience_report_design_PROGRAM_POLICY_SHA256 = '2642a87c60f5082276e65ef12c7de81037dfbbbb628666e458a366af56e81927'
discovery_stimulus_experience_multiresolution_fit_ROOT = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit')
discovery_stimulus_experience_multiresolution_fit_LOCK = discovery_stimulus_experience_multiresolution_fit_ROOT / 'qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json'
discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256 = '35b9fa3efacfbe1e3884e2d622e1701538a5375fe2b72ffe1155fc429903c2b7'
SIG = discovery_stimulus_experience_multiresolution_fit_ROOT / 'tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'
ASSIGN = discovery_stimulus_experience_multiresolution_fit_ROOT / 'tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'
CENT = discovery_stimulus_experience_multiresolution_fit_ROOT / 'tables/discovery_stimulus_experience_multiresolution_fit_family_centroids_all_k.csv'
METRIC = discovery_stimulus_experience_multiresolution_fit_ROOT / 'tables/discovery_stimulus_experience_multiresolution_fit_family_candidate_metrics.csv'
SUMMARY = discovery_stimulus_experience_multiresolution_fit_ROOT / 'qc/discovery_stimulus_experience_multiresolution_fit_summary.json'
OUT = Path('outputs/interpretability/validation_experience_transfer_design_validation_experience_transfer_design_specification')
CONFIG = Path('analysis_specifications/S077_validation_experience_transfer.json')
COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
STD = ['standardized__' + x for x in COORDS]
TR = ['transformed__' + x for x in COORDS]

def now():
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).isoformat(timespec='seconds')

def stamp():
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime('%Y%m%d_%H%M%S')

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def verify(p: Path, expected: str, label: str):
    if not p.is_file():
        raise RuntimeError(f'Missing {label}: {p}')
    got = sha256(p)
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')

def readj(p: Path):
    return json.loads(p.read_text())

def writej(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '')
    os.replace(tmp, p)

def atomic_csv(p: Path, df: pd.DataFrame):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)

def canonical_sha(obj: Any):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def weighted_quantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    order = np.argsort(x, kind='mergesort')
    x = x[order]
    w = w[order]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, x, left=x[0], right=x[-1]))

def weighted_iqr(x, w):
    return np.array([weighted_quantile(x[:, j], w, 0.75) - weighted_quantile(x[:, j], w, 0.25) for j in range(x.shape[1])])

def participant_weights(subjects):
    vc = pd.Series(subjects.astype(str)).value_counts().to_dict()
    return np.array([1.0 / vc[str(s)] for s in subjects])

def standardize_from_transformed(df: pd.DataFrame, pooled_iqr: np.ndarray):
    subjects = df.subject_id.astype(str).to_numpy()
    x = df[TR].to_numpy(float)
    z = np.empty_like(x)
    for s in pd.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        xx = x[idx]
        med = np.median(xx, axis=0)
        iq = np.quantile(xx, 0.75, axis=0) - np.quantile(xx, 0.25, axis=0)
        fb = ~np.isfinite(iq) | (iq <= IQR_FLOOR)
        scale = np.where(fb, pooled_iqr, iq)
        scale = np.where(~np.isfinite(scale) | (scale <= IQR_FLOOR), 1.0, scale)
        z[idx] = np.clip((xx - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite standardized signatures')
    return z

def canonicalize(labels, centers):
    order = sorted(range(len(centers)), key=lambda i: tuple((float(v) for v in centers[i])))
    rem = {old: new for new, old in enumerate(order)}
    return (np.array([rem[int(v)] for v in labels], int), centers[order].copy())

def fit_kmeans(x, w, k, seed):
    m = KMeans(n_clusters=k, init='k-means++', n_init=100, max_iter=500, tol=1e-07, random_state=seed, algorithm='lloyd')
    lab = m.fit_predict(x, sample_weight=w)
    return canonicalize(lab, m.cluster_centers_)

def assign(x, centers):
    d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    lab = np.argmin(d, axis=1)
    ass = d[np.arange(len(x)), lab]
    alt = d.copy()
    alt[np.arange(len(x)), lab] = np.inf
    mar = np.min(alt, axis=1) - ass
    return (lab, ass, mar, d)

def verify_inputs(project: Path):
    verify(project / multiresolution_experience_report_design_LOCK, multiresolution_experience_report_design_LOCK_SHA256, 'multiresolution_experience_report_design final lock')
    a = readj(project / multiresolution_experience_report_design_LOCK)
    for rel, h in a.get('hashes', {}).items():
        verify(project / rel, h, 'multiresolution_experience_report_design child ' + rel)
    verify(project / multiresolution_experience_report_design_PROGRAM_POLICY, multiresolution_experience_report_design_PROGRAM_POLICY_SHA256, 'multiresolution_experience_report_design program policy')
    verify(project / discovery_stimulus_experience_multiresolution_fit_LOCK, discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit final lock')
    b = readj(project / discovery_stimulus_experience_multiresolution_fit_LOCK)
    if b.get('eligible_K') != ELIGIBLE_K or b.get('next_phase') != PHASE:
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit eligible-K/next-phase drift')
    if any((b.get(k) is not False for k in ['rating_early_EEG_values_read', 'behavioral_values_read', 'Validation_signal_values_read', 'Confirmation_signal_values_read'])):
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit boundary drift')
    for rel, h in b.get('hashes', {}).items():
        verify(project / rel, h, 'discovery_stimulus_experience_multiresolution_fit child ' + rel)
    return (a, b, readj(project / multiresolution_experience_report_design_PROGRAM_POLICY), readj(project / SUMMARY))

def lopo_reference(sig: pd.DataFrame, assignments: pd.DataFrame, centroids: pd.DataFrame):
    subjects = sig.subject_id.astype(str).to_numpy()
    fullw = participant_weights(subjects)
    full_pool = weighted_iqr(sig[TR].to_numpy(float), fullw)
    full_pool = np.where(np.isfinite(full_pool), full_pool, 0.0)
    rows = []
    support = []
    for k in ELIGIBLE_K:
        af = assignments[assignments.k.eq(k)].copy()
        amap = dict(zip(af.raw_behavior_row_id.astype(str), af.family_1based.astype(int) - 1))
        cdf = centroids[centroids.k.eq(k)].sort_values('family_1based')
        fullc = cdf[STD].to_numpy(float)
        agree_all = []
        for si, s in enumerate(pd.unique(subjects)):
            train = sig[sig.subject_id.astype(str).ne(str(s))].copy()
            test = sig[sig.subject_id.astype(str).eq(str(s))].copy()
            tr_sub = train.subject_id.astype(str).to_numpy()
            trw = participant_weights(tr_sub)
            pool = weighted_iqr(train[TR].to_numpy(float), trw)
            pool = np.where(np.isfinite(pool), pool, 0.0)
            xtr = standardize_from_transformed(train, pool)
            xte = standardize_from_transformed(test, pool)
            labtr, ctr = fit_kmeans(xtr, trw, k, SEED + 1000 * k + si)
            cost = np.sum((ctr[:, None, :] - fullc[None, :, :]) ** 2, axis=2)
            r, c = linear_sum_assignment(cost)
            aligned = np.empty_like(ctr)
            for rr, cc in zip(r, c):
                aligned[cc] = ctr[rr]
            lab, dist, margin, _ = assign(xte, aligned)
            full = np.array([amap[str(x)] for x in test.raw_behavior_row_id], int)
            agree = lab == full
            agree_all.extend(agree.tolist())
            for rid, aa, ll, ff, dd, mm in zip(test.raw_behavior_row_id.astype(str), agree, lab, full, dist, margin):
                rows.append({'k': k, 'subject_id': str(s), 'raw_behavior_row_id': rid, 'lopo_family_1based': int(ll + 1), 'full_fit_family_1based': int(ff + 1), 'agree': bool(aa), 'lopo_assigned_squared_distance': float(dd), 'lopo_assignment_margin': float(mm)})
        rr = pd.DataFrame([x for x in rows if x['k'] == k])
        agr = rr[rr.agree]
        if len(agr) < 100:
            raise RuntimeError(f'Too few LOPO-agreeing rows for K={k}')
        subj_ag = rr.groupby('subject_id').agree.mean()
        support.append({'k': k, 'lopo_trial_agreement': float(rr.agree.mean()), 'lopo_participant_balanced_agreement': float(subj_ag.mean()), 'lopo_min_participant_agreement': float(subj_ag.min()), 'lopo_agreeing_trials': int(len(agr)), 'in_support_squared_distance_max': float(np.quantile(agr.lopo_assigned_squared_distance, 0.995, method='linear')), 'low_margin_threshold': float(np.quantile(agr.lopo_assignment_margin, 0.01, method='linear')), 'distance_reference_rule': '99.5th percentile of LOPO-agreeing held-out assigned squared distance', 'margin_reference_rule': '1st percentile of LOPO-agreeing held-out assignment margin'})
    return (pd.DataFrame(rows), pd.DataFrame(support), full_pool)

def self_test():
    rng = np.random.default_rng(3)
    subs = np.repeat([f's{i}' for i in range(8)], 20)
    tr = rng.normal(size=(len(subs), 14))
    df = pd.DataFrame({'subject_id': subs, 'raw_behavior_row_id': [f'r{i}' for i in range(len(subs))]})
    for j, c in enumerate(TR):
        df[c] = tr[:, j]
    w = participant_weights(subs)
    pool = weighted_iqr(tr, w)
    z = standardize_from_transformed(df, pool)
    lab, c = fit_kmeans(z, w, 3, 540)
    lab2, d, m, _ = assign(z, c)
    assert np.array_equal(lab, lab2) and np.isfinite(d).all() and np.isfinite(m).all()
    print('validation_experience_transfer_design self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    _, b, program, summ = verify_inputs(project)
    sig = pd.read_csv(project / SIG, low_memory=False)
    ass = pd.read_csv(project / ASSIGN, low_memory=False)
    cent = pd.read_csv(project / CENT, low_memory=False)
    met = pd.read_csv(project / METRIC, low_memory=False)
    if len(sig) != 1261 or sig.subject_id.nunique() != 31:
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit signature cohort drift')
    if [int(x) for x in met.loc[met.eligible.astype(bool), 'k'].tolist()] != ELIGIBLE_K:
        raise RuntimeError('discovery_stimulus_experience_multiresolution_fit metric eligibility drift')
    out = project / OUT
    final = out / 'qc/validation_experience_transfer_design_final_output_lock_hashes.json'
    if final.exists() and (not args.force):
        raise RuntimeError(f'validation_experience_transfer_design already materialized: {final} (use --force only to reproduce exact design outputs)')
    for d in ['qc', 'tables', 'reports']:
        (out / d).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] computing Discovery-only LOPO support references for K={ELIGIBLE_K}', flush=True)
    lopo, support, pool = lopo_reference(sig, ass, cent)
    ckeep = cent[cent.k.isin(ELIGIBLE_K)].copy().sort_values(['k', 'family_1based'])
    fallback = pd.DataFrame({'coordinate': COORDS, 'discovery_participant_balanced_pooled_iqr': pool})
    gate_rows = []
    for k in ELIGIBLE_K:
        gates = [('source_hashes', 'all source/store hash gates pass', 'boolean', True), ('eligible_participants', 'eligible Validation participants', '>=', 9), ('paired_fraction', 'paired experience trials / authoritative Validation trials', '>=', 0.75), ('in_support', 'participant-balanced in-support fraction', '>=', 0.8), ('family_participants', 'minimum distinct participants per family', '>=', 5), ('family_trials', 'minimum trials per family', '>=', 30), ('family_occupancy', 'minimum participant-balanced occupancy per family', '>=', 0.05), ('no_behavior', 'behavior/category/context/stimulus target mapping executed', '==', False)]
        for gid, metric, comp, thr in gates:
            gate_rows.append({'k': k, 'gate_id': gid, 'metric': metric, 'comparison': comp, 'threshold': thr})
    gates = pd.DataFrame(gate_rows)
    held = program['heldout_store_references_not_opened_in_multiresolution_experience_report_design']['Validation']
    policy = {'phase': PHASE, 'generated_at': now(), 'scientific_role': 'specification one-shot Validation transfer of Discovery stimulus-only multi-resolution experience families before Validation signal values are opened', 'fixed_before_validation_signal_access': True, 'eligible_K': ELIGIBLE_K, 'all_eligible_K_tested_independently': True, 'no_resolution_promoted_for_simplicity': True, 'representation': {'source_cells_per_epoch': 169, 'stages': ['stimulus_early', 'stimulus_late'], 'signature_cells': 14, 'coordinate_order': COORDS, 'rating_early_forbidden': True, 'behavior_forbidden': True, 'validation_signature_preprocessing': 'exact discovery_stimulus_experience_multiresolution_fit participant/stage preprocessing and transformed-signature within-participant robust scaling; use prespecified Discovery participant-balanced pooled transformed-signature IQR only if a Validation participant local signature IQR <=1e-8', 'prespecified_discovery_fallback_iqr': {c: float(v) for c, v in zip(COORDS, pool)}}, 'fixed_assignment': {str(k): {'centroids': ckeep[ckeep.k.eq(k)].sort_values('family_1based')[STD].to_numpy(float).tolist(), 'support': support[support.k.eq(k)].iloc[0][['in_support_squared_distance_max', 'low_margin_threshold', 'distance_reference_rule', 'margin_reference_rule']].to_dict(), 'distance': 'squared Euclidean in the prespecified 14-cell standardized stimulus-only signature', 'tie_rule': 'lowest numeric family', 'refit': False, 'rename': False} for k in ELIGIBLE_K}, 'validation_store': held, 'participant_eligibility': {'minimum_paired_stimulus_early_late_trials': 16, 'no_behavior_based_exclusion': True, 'nonfinite_signature_rule': 'exclude trial before assignment and record reason', 'participant_below_minimum_rule': 'exclude participant from transfer gate but retain audit'}, 'structural_transfer_gate': {'evaluated_separately_for_each_K': True, 'minimum_eligible_participants': 9, 'minimum_paired_experience_fraction_of_authoritative_validation_trials': 0.75, 'minimum_participant_balanced_in_support_fraction': 0.8, 'minimum_distinct_participants_per_family': 5, 'minimum_trials_per_family': 30, 'minimum_participant_balanced_occupancy_per_family': 0.05, 'all_source_hash_gates_must_pass': True, 'no_behavior_category_context_stimulus_target_mapping': True, 'K_replication_rule': 'all gates for that K must pass', 'failure_rule': 'failed K is not retuned, renamed, merged, split, or advanced to Confirmation'}, 'post_validation_rule': {'if_one_or_more_K_pass': 'only the passing K resolutions may later be transferred unchanged to Confirmation; Confirmation is not automatically opened by validation_experience_transfer_design', 'if_no_K_pass': 'close the discrete stimulus-experience family transfer branch; do not invent or retune a taxonomy', 'behavioral_mapping': 'remains forbidden until separately prespecified after structural branches are fixed'}, 'forbidden': ['Validation behavioral reports', 'emotion category as target', 'context as target', 'stimulus identity as target', 'rating_early EEG', 'centroid refit', 'K reselection', 'threshold tuning', 'family semantic naming', 'Confirmation signal access'], 'next_phase': NEXT_PHASE}
    writej(project / CONFIG, policy)
    psha = canonical_sha(policy)
    atomic_csv(out / 'tables/validation_experience_transfer_design_lopo_support_reference.csv', lopo.sort_values(['k', 'subject_id', 'raw_behavior_row_id']))
    atomic_csv(out / 'tables/validation_experience_transfer_design_support_thresholds.csv', support.sort_values('k'))
    atomic_csv(out / 'tables/validation_experience_transfer_design_prespecified_centroids.csv', ckeep)
    atomic_csv(out / 'tables/validation_experience_transfer_design_discovery_fallback_iqr.csv', fallback)
    atomic_csv(out / 'tables/validation_experience_transfer_design_validation_gate_spec.csv', gates)
    source_rows = []
    for name, p in [('multiresolution_experience_report_design_final_lock', project / multiresolution_experience_report_design_LOCK), ('multiresolution_experience_report_design_program_policy', project / multiresolution_experience_report_design_PROGRAM_POLICY), ('discovery_stimulus_experience_multiresolution_fit_final_lock', project / discovery_stimulus_experience_multiresolution_fit_LOCK), ('discovery_stimulus_experience_multiresolution_fit_signatures', project / SIG), ('discovery_stimulus_experience_multiresolution_fit_assignments', project / ASSIGN), ('discovery_stimulus_experience_multiresolution_fit_centroids', project / CENT), ('discovery_stimulus_experience_multiresolution_fit_metrics', project / METRIC)]:
        source_rows.append({'source': name, 'path': str(p.relative_to(project)), 'sha256': sha256(p), 'bytes': p.stat().st_size})
    atomic_csv(out / 'qc/validation_experience_transfer_design_source_rehash.csv', pd.DataFrame(source_rows))
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_experience_transfer_design_prespecified', 'eligible_K': ELIGIBLE_K, 'discovery_stimulus_experience_multiresolution_fit_status': summ.get('status'), 'validation_experience_transfer_design_policy_canonical_sha256': psha, 'lopo_support_reference': support.to_dict('records'), 'Validation_signal_values_read': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_role': 'one-shot Validation stimulus-only structural transfer for K=2,3,4 using prespecified discovery_stimulus_experience_multiresolution_fit templates and validation_experience_transfer_design support/gates'}
    writej(out / 'qc/validation_experience_transfer_design_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'design_prespecified', 'source_code_sha256': sha256(Path(__file__).resolve()), 'multiresolution_experience_report_design_final_lock_sha256': multiresolution_experience_report_design_LOCK_SHA256, 'discovery_stimulus_experience_multiresolution_fit_final_lock_sha256': discovery_stimulus_experience_multiresolution_fit_LOCK_SHA256, 'validation_experience_transfer_design_policy_canonical_sha256': psha, 'eligible_K': ELIGIBLE_K, 'Validation_signal_values_read': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/validation_experience_transfer_design_contract.json', contract)
    rpt = ['# validation_experience_transfer_design — Validation stimulus-experience transfer analysis specification', '', '**Status: design prespecified; Validation signal values remain unopened.**', '', f'Eligible Discovery resolutions retained without reselection: **{ELIGIBLE_K}**.', '', '## Discovery-only LOPO support reference', support.to_string(index=False), '', 'Support distance is prespecified at the 99.5th percentile of LOPO-agreeing held-out Discovery distances; low-margin threshold at the 1st percentile. These diagnostics do not remove an already discovery_stimulus_experience_multiresolution_fit-eligible K.', '', '## Validation rule', 'Each K is transferred independently with no refit. A K replicates only if every prespecified structural gate passes. Failure of one K does not alter another K and cannot be rescued by merging, splitting, threshold tuning, or semantic relabeling.', '', '## Boundaries', 'No Validation EEG amplitudes/compact values or behavioral targets were opened in validation_experience_transfer_design. `rating_early` remains excluded. Confirmation remains closed.', '', f'## Next\n**{NEXT_PHASE}** — one-shot Validation stimulus-only structural transfer.', '']
    (out / 'reports/validation_experience_transfer_design_report.md').write_text(''.join(rpt))
    targets = [project / CONFIG, out / 'qc/validation_experience_transfer_design_contract.json', out / 'qc/validation_experience_transfer_design_summary.json', out / 'qc/validation_experience_transfer_design_source_rehash.csv', out / 'tables/validation_experience_transfer_design_lopo_support_reference.csv', out / 'tables/validation_experience_transfer_design_support_thresholds.csv', out / 'tables/validation_experience_transfer_design_prespecified_centroids.csv', out / 'tables/validation_experience_transfer_design_discovery_fallback_iqr.csv', out / 'tables/validation_experience_transfer_design_validation_gate_spec.csv', out / 'reports/validation_experience_transfer_design_report.md']
    hashes = {str(p.relative_to(project)): sha256(p) for p in targets}
    finalobj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_experience_transfer_design_prespecified', 'hashes': hashes, 'eligible_K': ELIGIBLE_K, 'validation_experience_transfer_design_policy_canonical_sha256': psha, 'Validation_signal_values_read': False, 'Validation_behavioral_values_read': False, 'Confirmation_signal_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final, finalobj)
    hroot = project / ''
    hroot.mkdir(parents=True, exist_ok=True)
    hn = f'validation_experience_transfer_design_validation_experience_transfer_design_specification_{stamp()}'
    hz = hroot / (hn + '')
    pdir = Path(__file__).resolve().parent
    hs = sha256(hz)
    hz.with_suffix('').write_text(f'{hs}  {hz.name}')
    print('')
    print('validation_experience_transfer_design PASS')
    print(f'Eligible K prespecified for Validation: {ELIGIBLE_K}')
    for _, r in support.iterrows():
        print(f'K={int(r.k)} LOPO participant-balanced agreement={r.lopo_participant_balanced_agreement:.6f}, support_distance_max={r.in_support_squared_distance_max:.6f}, low_margin={r.low_margin_threshold:.6f}')
    print('Validation signal values opened: False')
    print('Validation behavioral values opened: False')
    print('Confirmation opened: False')
    print(f'Next phase: {NEXT_PHASE}')
    print(f'{hz}')
    print(f'SHA-256: {hs}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())