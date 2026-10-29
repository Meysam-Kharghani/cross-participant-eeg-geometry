# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase discovery_multimodal_inference — prespecified Discovery-only EEG × physiology inference.

Implements the three hypotheses prespecified in multimodal_physio_design after the corrected discovery_multimodal_attachment_qc
attachment/QC. Validation and Confirmation physiology remain closed."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'discovery_multimodal_inference'
NEXT = 'discovery_multimodal_synthesis'
PREFIX = 'discovery_multimodal_inference'
SEED = 544
PERM_GLOBAL = 10000
PERM_CV = 1000
BOOTSTRAP = 5000
CGRID = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=float)
INNER_FOLDS = 5
BIO = ['gsr_scr_mean', 'gsr_scr_sd', 'gsr_scr_median', 'gsr_scr_min', 'gsr_scr_max', 'gsr_scr_sum', 'gsr_scr_iqr', 'gsr_scr_early_mean', 'gsr_scr_late_mean', 'gsr_scr_late_minus_early', 'gsr_scr_slope', 'gsr_scr_auc', 'gsr_scl_mean', 'gsr_scl_sd', 'gsr_scl_median', 'gsr_scl_min', 'gsr_scl_max', 'gsr_scl_sum', 'gsr_scl_iqr', 'gsr_scl_early_mean', 'gsr_scl_late_mean', 'gsr_scl_late_minus_early', 'gsr_scl_slope', 'gsr_scl_auc', 'gsr_iscr_mean', 'gsr_iscr_sd', 'gsr_iscr_median', 'gsr_iscr_min', 'gsr_iscr_max', 'gsr_iscr_sum', 'gsr_iscr_iqr', 'gsr_iscr_early_mean', 'gsr_iscr_late_mean', 'gsr_iscr_late_minus_early', 'gsr_iscr_slope', 'gsr_iscr_auc', 'hr_hr_median', 'hr_hr_iqr', 'hr_hr_skew', 'hr_hr_kurt']
GSR = BIO[:36]
HR = BIO[36:]
GEOM = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']
raw_eeg_epoch_folds_POLICY = Path('analysis_specifications/S044_multimodal_physio.json')
raw_eeg_epoch_folds_POLICY_SHA = '2ef19f67a6a5d8f1faf1e6869f5c6accfec0b5c83afa72f65440f62249a8ea8b'
raw_eeg_epoch_folds_HYP = Path('outputs/tables/multimodal_physio_design_prespecified_multimodal_hypotheses.csv')
raw_eeg_epoch_folds_HYP_SHA = '7a866ee58c83991e6bbc872d0afb556ad382a1db27a41ab3b479a9d57160d4da'
raw_eeg_epoch_folds_GATES = Path('outputs/tables/multimodal_physio_design_prespecified_multimodal_gate_manifest.csv')
raw_eeg_epoch_folds_GATES_SHA = 'b2329e147c522103d6898bcb08d6c2c5084ad11caafd505d614be32fbe3dc071'
raw_eeg_preprocessing_qc_LOCK = Path('outputs/qc/discovery_multimodal_attachment_qc_final_output_lock_hashes.json')
raw_eeg_preprocessing_qc_LOCK_SHA = '2cc5d2964e130ce6e3afee028ff762399b54b211a86c3a530d5d593b445803f6'
raw_eeg_preprocessing_qc_NEXT = Path('analysis_specifications/S018_discovery_multimodal_attachment_qc.json')
raw_eeg_preprocessing_qc_NEXT_SHA = 'f0d514b5f3830677aa4fc591de98f27702da4ac03dcd0cc6a08032374fc37a39'
raw_eeg_preprocessing_qc_MANIFEST = Path('outputs/tables/discovery_multimodal_attachment_qc_discovery_multimodal_manifest.csv')
raw_eeg_preprocessing_qc_MANIFEST_SHA = '687a2204778a5089f6579ceaee48e565559b02c1e9a62a813b847386fe112bb6'
raw_eeg_preprocessing_qc_ROSTER = Path('outputs/tables/discovery_multimodal_attachment_qc_prespecified_eligible_roster.csv')
raw_eeg_preprocessing_qc_ROSTER_SHA = 'fecbcfe09843e869a36fc560d997b4b185db76b95789dec680093c7c933bb27b'
raw_eeg_preprocessing_qc_ADDENDUM = Path('config/discovery_multimodal_attachment_qc_bridge_correction_addendum.json')
raw_eeg_preprocessing_qc_ADDENDUM_SHA = '1ce1212c57f78ea3c6333a3da4b822914fef386696fcbbe0aacecc88cb3dbeb2'
cross_person_alignment_fit_LOCK = Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json')
cross_person_alignment_fit_LOCK_SHA = '64bb8278afd0ddc17b46521180753e5537333e9e55a93741eece62d406864150'
GEOM_PATH = Path('outputs/tables/cross_person_alignment_fit_trajectory_standardized_geometry.csv')
GEOM_SHA = '289ef948eba726d5644d04d2128a98653b5a6433746ea411f780a7e98f64ef08'
IMPLEMENTATION = {'phase': PHASE, 'seed': SEED, 'status': 'outcome_blind_implementation_resolution_before_M1_M3_statistics', 'primary': {'permutations_M1_M3': PERM_GLOBAL, 'permutations_M2': PERM_CV, 'bootstrap': BOOTSTRAP, 'holm_alpha': 0.05}, 'physiology_preprocessing': {'imputation': 'analysis/train-set feature median', 'center_scale': 'analysis/train-set median/IQR', 'zero_iqr': '1.0', 'winsorize': [-6.0, 6.0]}, 'M1': {'nuisance': 'fit separately within participant: intercept + video indicator + normalized_trial_order', 'statistic': 'squared L2 norm of equal-participant mean 40D Family1-Family2 residual contrast', 'null': 'within-participant family permutation preserving counts'}, 'M2': {'outer': 'leave-one-participant-out', 'inner': '5 deterministic grouped participant folds', 'C_grid': CGRID.tolist(), 'metric': 'mean participant balanced accuracy', 'training_weight': 'equal total weight per participant', 'solver': 'liblinear logistic ridge', 'null': 'within-participant family permutation with full nested C reselection'}, 'M3': {'geometry': 'immutable cross_person_alignment_fit six-dimensional standardized trajectory coordinates', 'nuisance': 'fit separately within participant for physiology and geometry: intercept + video indicator + normalized_trial_order', 'statistic': 'Frobenius norm of equal-participant mean within-person 40x6 residual correlation matrix', 'null': 'within-participant row permutation of residual geometry relative to physiology'}, 'robustness_pass_rules': {'M1': ['unadjusted permutation p<0.05', 'complete-case permutation p<0.05 if feasible', 'minimum leave-one-participant-out contrast-vector cosine >=0.90', 'at least one gated GSR/HR block Holm p<0.05', 'family-label swap statistic invariant'], 'M2': ['participant-bootstrap 95% CI lower bound of BA gain >0', 'complete-case observed BA gain >0 if feasible', 'median outer-participant BA gain >0', 'at least 60% of outer participants have nonnegative BA gain', 'family-label numerical swap invariance'], 'M3': ['unadjusted permutation p<0.05', 'complete-case permutation p<0.05 if feasible', 'minimum leave-one-participant-out correlation-matrix cosine >=0.90', 'at least one gated GSR/HR block Holm p<0.05']}, 'complete_case_feasible': 'at least 16 participants; for M1/M2 also >=4 trials in each family per retained participant', 'heldout_gate': 'Validation physiology may open only for a primary test with Holm p<0.05 AND all applicable prespecified robustness pass rules; Confirmation remains closed.'}

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--test-mode', action='store_true')
    p.add_argument('--perm-global', type=int, default=PERM_GLOBAL)
    p.add_argument('--perm-cv', type=int, default=PERM_CV)
    p.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    p.add_argument('--jobs', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(p: Path):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atom_json(p: Path, v: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    os.replace(t, p)

def verify(rows, label, p, exp):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    obs = sha(p)
    ok = obs == exp
    rows.append({'artifact': label, 'path': str(p), 'sha256': obs, 'expected_sha256': exp, 'passed': ok})
    if not ok:
        raise RuntimeError(f'{label} SHA mismatch: {obs} != {exp}')

def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out

def pp(obs, null):
    return float((1 + np.sum(np.asarray(null) >= obs)) / (len(null) + 1))

def pci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

def cosine(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-15 else float('nan')

def pweights(subjects):
    s = np.asarray(subjects)
    w = np.zeros(len(s), float)
    for x in np.unique(s):
        idx = np.where(s == x)[0]
        w[idx] = 1.0 / len(idx)
    return w

def prep_fit(X):
    X = np.asarray(X, float)
    z = X.copy()
    z[~np.isfinite(z)] = np.nan
    med = np.nanmedian(z, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    z = np.where(np.isfinite(z), z, med)
    q1 = np.quantile(z, 0.25, axis=0)
    q3 = np.quantile(z, 0.75, axis=0)
    scale = q3 - q1
    scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, 1.0)
    out = np.clip((z - med) / scale, -6, 6)
    return (out, med, scale)

def prep_apply(X, med, scale):
    X = np.asarray(X, float)
    z = np.where(np.isfinite(X), X, med)
    return np.clip((z - med) / scale, -6, 6)

def nuisance_residuals(df, Y):
    Y = np.asarray(Y, float)
    out = np.zeros_like(Y, float)
    for sub, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        ctx = (df.loc[idx, 'context'].astype(str).to_numpy() == 'video').astype(float)
        order = df.loc[idx, 'normalized_trial_order'].to_numpy(float)
        X = np.c_[np.ones(len(idx)), ctx, order]
        coef = np.linalg.lstsq(X, Y[idx], rcond=None)[0]
        out[idx] = Y[idx] - X @ coef
    return out

def participant_contrasts(df, R, fam=None):
    f = df.trajectory_family_1based.to_numpy(int) if fam is None else np.asarray(fam, int)
    rows = []
    vec = []
    for sub, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        a = R[idx][f[idx] == 1]
        b = R[idx][f[idx] == 2]
        if len(a) == 0 or len(b) == 0:
            continue
        d = a.mean(axis=0) - b.mean(axis=0)
        vec.append(d)
        rows.append(sub)
    return (rows, np.asarray(vec))

def m1_stat_from_contrasts(C):
    v = np.mean(C, axis=0)
    return (float(np.dot(v, v)), v)

def permute_family(df, rng):
    f = df.trajectory_family_1based.to_numpy(int).copy()
    for _, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        f[idx] = rng.permutation(f[idx])
    return f

def m1_null_stats(df, R, repeats, seed):
    out = np.zeros((repeats, 3), float)
    for i in range(repeats):
        fam = permute_family(df, np.random.default_rng(seed + i))
        _, C = participant_contrasts(df, R, fam)
        v = C.mean(axis=0)
        out[i] = [np.dot(v, v), np.dot(v[:36], v[:36]), np.dot(v[36:], v[36:])]
    return out

def m1_boot(C, repeats, seed):
    rng = np.random.default_rng(seed)
    n = len(C)
    out = np.zeros(repeats)
    for b in range(repeats):
        cb = C[rng.integers(0, n, n)]
        v = cb.mean(axis=0)
        out[b] = np.dot(v, v)
    return out

def participant_corr_matrices(df, P, G):
    mats = []
    subs = []
    for sub, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        A = P[idx]
        B = G[idx]
        A = A - A.mean(0)
        B = B - B.mean(0)
        sa = np.sqrt(np.sum(A * A, axis=0))
        sb = np.sqrt(np.sum(B * B, axis=0))
        den = sa[:, None] * sb[None, :]
        M = np.divide(A.T @ B, den, out=np.zeros((A.shape[1], B.shape[1])), where=den > 1e-12)
        mats.append(M)
        subs.append(sub)
    return (subs, np.asarray(mats))

def m3_stat(mats):
    M = np.mean(mats, axis=0)
    return (float(np.linalg.norm(M, 'fro')), M)

def m3_null(df, P, G, repeats, seed):
    groups = [np.asarray(list(v), int) for v in df.groupby('subject_id', sort=True).groups.values()]
    out = np.zeros((repeats, 3), float)
    for i in range(repeats):
        rng = np.random.default_rng(seed + i)
        Gp = G.copy()
        for idx in groups:
            Gp[idx] = G[idx][rng.permutation(len(idx))]
        _, m = participant_corr_matrices(df, P, Gp)
        M = m.mean(0)
        out[i] = [np.linalg.norm(M, 'fro'), np.linalg.norm(M[:36], 'fro'), np.linalg.norm(M[36:], 'fro')]
    return out

def m3_boot(mats, repeats, seed):
    rng = np.random.default_rng(seed)
    n = len(mats)
    out = np.zeros(repeats)
    for b in range(repeats):
        out[b] = np.linalg.norm(mats[rng.integers(0, n, n)].mean(0), 'fro')
    return out

def inner_subject_folds(subjects, k=INNER_FOLDS):
    us = sorted(np.unique(subjects))
    folds = {s: i % k for i, s in enumerate(us)}
    return np.array([folds[s] for s in subjects], int)

def design_baseline(df):
    return np.c_[(df.context.astype(str).to_numpy() == 'video').astype(float), df.normalized_trial_order.to_numpy(float)]

def fit_logit(X, y, w, C):
    m = LogisticRegression(C=float(C), penalty='l2', solver='liblinear', max_iter=2000, tol=1e-08, random_state=0)
    m.fit(X, y, sample_weight=w)
    return m

def mean_subject_ba(y, p, subs):
    vals = []
    for s in np.unique(subs):
        i = subs == s
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(balanced_accuracy_score(y[i], p[i]))
    return float(np.mean(vals)) if vals else float('nan')

def select_c(df_train, y_train, Xphys_raw, model_kind):
    subs = df_train.subject_id.to_numpy(str)
    folds = inner_subject_folds(subs, INNER_FOLDS)
    scores = []
    for C in CGRID:
        pred = np.zeros(len(df_train), int)
        for f in sorted(np.unique(folds)):
            te = folds == f
            tr = ~te
            w = pweights(subs[tr])
            Xb_tr = design_baseline(df_train.iloc[np.where(tr)[0]])
            Xb_te = design_baseline(df_train.iloc[np.where(te)[0]])
            if model_kind == 'full':
                Ztr, med, sc = prep_fit(Xphys_raw[tr])
                Zte = prep_apply(Xphys_raw[te], med, sc)
                Xtr = np.c_[Xb_tr, Ztr]
                Xte = np.c_[Xb_te, Zte]
            else:
                Xtr, Xte = (Xb_tr, Xb_te)
            pred[te] = fit_logit(Xtr, y_train[tr], w, C).predict(Xte)
        scores.append(mean_subject_ba(y_train, pred, subs))
    mx = np.nanmax(scores)
    idx = int(np.flatnonzero(np.isclose(scores, mx, rtol=0, atol=1e-12))[0])
    return (float(CGRID[idx]), scores)

def nested_oof(df, y, Xphys_raw, return_detail=False):
    subs = df.subject_id.to_numpy(str)
    pred_b = np.zeros(len(df), int)
    pred_f = np.zeros(len(df), int)
    details = []
    for test_sub in sorted(np.unique(subs)):
        te = subs == test_sub
        tr = ~te
        dtr = df.iloc[np.where(tr)[0]]
        dte = df.iloc[np.where(te)[0]]
        ytr = y[tr]
        w = pweights(subs[tr])
        cb, sb = select_c(dtr, ytr, Xphys_raw[tr], 'base')
        cf, sf = select_c(dtr, ytr, Xphys_raw[tr], 'full')
        Xbtr = design_baseline(dtr)
        Xbte = design_baseline(dte)
        Ztr, med, sc = prep_fit(Xphys_raw[tr])
        Zte = prep_apply(Xphys_raw[te], med, sc)
        pred_b[te] = fit_logit(Xbtr, ytr, w, cb).predict(Xbte)
        pred_f[te] = fit_logit(np.c_[Xbtr, Ztr], ytr, w, cf).predict(np.c_[Xbte, Zte])
        if return_detail:
            details.append({'test_subject': test_sub, 'C_baseline': cb, 'C_full': cf, 'inner_score_baseline': float(max(sb)), 'inner_score_full': float(max(sf))})
    ba_b = mean_subject_ba(y, pred_b, subs)
    ba_f = mean_subject_ba(y, pred_f, subs)
    return (ba_b, ba_f, ba_f - ba_b, pred_b, pred_f, details)

def perm_y(df, seed):
    y = (df.trajectory_family_1based.to_numpy(int) == 2).astype(int)
    rng = np.random.default_rng(seed)
    for _, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        y[idx] = rng.permutation(y[idx])
    return y

def m2_perm_one(df, Xraw, seed):
    y = perm_y(df, seed)
    return nested_oof(df, y, Xraw, False)[2]

def checkpoint_m2(df, Xraw, repeats, seed, cache, jobs, note):
    cache.mkdir(parents=True, exist_ok=True)
    key = {'script_sha': sha(Path(__file__)), 'manifest_sha': raw_eeg_preprocessing_qc_MANIFEST_SHA, 'repeats': repeats, 'seed': seed, 'inner_folds': INNER_FOLDS, 'C': CGRID.tolist()}
    kp = cache / 'key.json'
    vp = cache / 'null.npy'
    out = np.full(repeats, np.nan)
    if kp.is_file() and vp.is_file() and (json.loads(kp.read_text()) == key):
        a = np.load(vp)
        if len(a) == repeats:
            out[:] = a
            if np.isfinite(out).all():
                note('M2 permutation checkpoint reused (complete).')
                return out
            note(f'M2 partial checkpoint reused: {int(np.isfinite(out).sum())}/{repeats}')
    else:
        atom_json(kp, key)
        np.save(vp, out)
    chunk = max(1, min(10, repeats))
    missing = np.flatnonzero(~np.isfinite(out))
    for pos in range(0, len(missing), chunk):
        ids = missing[pos:pos + chunk]
        vals = Parallel(n_jobs=jobs, backend='loky')((delayed(m2_perm_one)(df, Xraw, seed + int(i)) for i in ids))
        out[ids] = vals
        np.save(vp, out)
        note(f'M2 permutations {int(np.isfinite(out).sum())}/{repeats}')
    return out

def m2_boot(df, y, pb, pf, repeats, seed):
    subs = df.subject_id.to_numpy(str)
    us = np.unique(subs)
    gains = {}
    b0 = {}
    b1 = {}
    for s in us:
        i = subs == s
        b0[s] = balanced_accuracy_score(y[i], pb[i])
        b1[s] = balanced_accuracy_score(y[i], pf[i])
        gains[s] = b1[s] - b0[s]
    rng = np.random.default_rng(seed)
    out = np.zeros(repeats)
    for b in range(repeats):
        draw = rng.choice(us, size=len(us), replace=True)
        out[b] = np.mean([gains[s] for s in draw])
    return (out, pd.DataFrame([{'subject_id': s, 'baseline_balanced_accuracy': b0[s], 'full_balanced_accuracy': b1[s], 'delta_balanced_accuracy': gains[s]} for s in us]))

def outputs(project):
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    c = project / 'config'
    r = project / 'outputs/reports'
    cache = project / 'outputs/large/discovery_multimodal_inference_discovery_multimodal_inference_cache_v1'
    return {'resolution': c / 'discovery_multimodal_inference_implementation_resolution.json', 'rehash': q / 'discovery_multimodal_inference_source_rehash_audit.csv', 'global': t / 'discovery_multimodal_inference_global_tests.csv', 'm1vec': t / 'discovery_multimodal_inference_m1_contrast_vector.csv', 'm1part': t / 'discovery_multimodal_inference_m1_participant_contrasts.csv', 'm1null': t / 'discovery_multimodal_inference_m1_null_statistics.csv', 'm2pred': t / 'discovery_multimodal_inference_m2_oof_predictions.csv', 'm2sel': t / 'discovery_multimodal_inference_m2_hyperparameter_selection.csv', 'm2null': t / 'discovery_multimodal_inference_m2_null_statistics.csv', 'm3mat': t / 'discovery_multimodal_inference_m3_correlation_matrix.csv', 'm3part': t / 'discovery_multimodal_inference_m3_participant_matrices.csv', 'm3null': t / 'discovery_multimodal_inference_m3_null_statistics.csv', 'boot': t / 'discovery_multimodal_inference_bootstrap_summary.csv', 'bootrep': t / 'discovery_multimodal_inference_bootstrap_replicates.csv', 'robust': t / 'discovery_multimodal_inference_robustness_summary.csv', 'lopo': t / 'discovery_multimodal_inference_lopo_diagnostics.csv', 'blocks': t / 'discovery_multimodal_inference_gated_block_followups.csv', 'cc': t / 'discovery_multimodal_inference_complete_case_sensitivity.csv', 'summary': q / 'discovery_multimodal_inference_inference_summary.json', 'report': r / 'discovery_multimodal_inference_discovery_multimodal_inference.md', 'next': project / 'analysis_specifications/S019_discovery_multimodal_inference_synthesis.json', 'contract': c / 'discovery_multimodal_inference_inference_contract.json', 'lock': q / 'discovery_multimodal_inference_final_output_lock_hashes.json', 'cache': cache}

def self_test():
    assert len(BIO) == 40 and len(GSR) == 36 and (len(HR) == 4) and (len(GEOM) == 6)
    assert np.allclose(holm([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])
    x = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 5.0]])
    z, m, s = prep_fit(x)
    assert np.isfinite(z).all() and np.isfinite(prep_apply(x, m, s)).all()
    d = pd.DataFrame({'subject_id': ['a'] * 8 + ['b'] * 8, 'context': ['imagery', 'video'] * 8, 'normalized_trial_order': np.tile(np.linspace(0, 1, 8), 2), 'trajectory_family_1based': [1, 1, 1, 1, 2, 2, 2, 2] * 2})
    R = np.random.default_rng(1).normal(size=(16, 4))
    rr = nuisance_residuals(d, R)
    assert rr.shape == R.shape
    print('discovery_multimodal_inference self-test: PASS')

def main():
    a = args()
    if a.self_test:
        self_test()
        return 0
    if not a.test_mode and (a.perm_global != PERM_GLOBAL or a.perm_cv != PERM_CV or a.bootstrap != BOOTSTRAP):
        raise RuntimeError('prespecified execution requires 10000 M1/M3 permutations, 1000 M2 permutations, and 5000 bootstraps')
    project = a.project.resolve()
    out = outputs(project)
    for k, p in out.items():
        if k != 'cache':
            Path(p).parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not a.force):
        raise RuntimeError('discovery_multimodal_inference already exists; use --force only for identical rerun')

    def note(x):
        print(f'[{now()}] {x}', flush=True)
    rows = []
    for label, p, e in [('multimodal_physio_design_policy', raw_eeg_epoch_folds_POLICY, raw_eeg_epoch_folds_POLICY_SHA), ('multimodal_physio_design_hypotheses', raw_eeg_epoch_folds_HYP, raw_eeg_epoch_folds_HYP_SHA), ('multimodal_physio_design_gates', raw_eeg_epoch_folds_GATES, raw_eeg_epoch_folds_GATES_SHA), ('discovery_multimodal_attachment_qc_lock', raw_eeg_preprocessing_qc_LOCK, raw_eeg_preprocessing_qc_LOCK_SHA), ('discovery_multimodal_attachment_qc_next', raw_eeg_preprocessing_qc_NEXT, raw_eeg_preprocessing_qc_NEXT_SHA), ('discovery_multimodal_attachment_qc_manifest', raw_eeg_preprocessing_qc_MANIFEST, raw_eeg_preprocessing_qc_MANIFEST_SHA), ('discovery_multimodal_attachment_qc_roster', raw_eeg_preprocessing_qc_ROSTER, raw_eeg_preprocessing_qc_ROSTER_SHA), ('discovery_multimodal_attachment_qc_bridge_addendum', raw_eeg_preprocessing_qc_ADDENDUM, raw_eeg_preprocessing_qc_ADDENDUM_SHA), ('cross_person_alignment_fit_lock', cross_person_alignment_fit_LOCK, cross_person_alignment_fit_LOCK_SHA), ('cross_person_alignment_fit_standardized_geometry', GEOM_PATH, GEOM_SHA)]:
        verify(rows, label, project / p, e)
    nx = json.loads((project / raw_eeg_preprocessing_qc_NEXT).read_text())
    if nx.get('next_phase') != PHASE or not nx.get('next_phase_permitted') or nx.get('heldout_physio_values_opened'):
        raise RuntimeError('discovery_multimodal_attachment_qc did not authorize Discovery-only discovery_multimodal_inference')
    atom_json(out['resolution'], IMPLEMENTATION)
    resolution_sha = sha(out['resolution'])
    note(f'Implementation resolution fixed: {resolution_sha}')
    M = pd.read_csv(project / raw_eeg_preprocessing_qc_MANIFEST, low_memory=False)
    ROST = pd.read_csv(project / raw_eeg_preprocessing_qc_ROSTER)
    G = pd.read_csv(project / GEOM_PATH, usecols=['subject_id', 'trial_number', *GEOM])
    if len(M) != 1237 or M.subject_id.nunique() != 31 or len(BIO) != 40:
        raise RuntimeError('discovery_multimodal_attachment_qc manifest boundary changed')
    if M[BIO].columns.tolist() != BIO:
        raise RuntimeError('prespecified physiology feature order changed')
    if M.duplicated(['subject_id', 'trial_number']).any() or G.duplicated(['subject_id', 'trial_number']).any():
        raise RuntimeError('Duplicate EEG trial key')
    D = M[M.physiology_row_attached.astype(bool)].copy()
    D = D.merge(G, on=['subject_id', 'trial_number'], how='left', validate='one_to_one')
    if D[GEOM].isna().any().any():
        raise RuntimeError('Joined physiology trial lacks prespecified standardized EEG geometry')
    elig12 = set(ROST.loc[ROST.eligible_M1_M2.astype(bool), 'subject_id'])
    elig3 = set(ROST.loc[ROST.eligible_M3.astype(bool), 'subject_id'])
    D12 = D[D.subject_id.isin(elig12)].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    D3 = D[D.subject_id.isin(elig3)].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    if D12.subject_id.nunique() != 25 or D3.subject_id.nunique() != 25:
        raise RuntimeError('prespecified multimodal roster changed')
    X12raw = D12[BIO].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    X3raw = D3[BIO].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    G3 = D3[GEOM].to_numpy(float)
    note('M1 adjusted participant-balanced physiology contrast.')
    X12, _, _ = prep_fit(X12raw)
    R1 = nuisance_residuals(D12, X12)
    subs1, outcome_model_integrity = participant_contrasts(D12, R1)
    m1obs, m1vec = m1_stat_from_contrasts(outcome_model_integrity)
    m1null = m1_null_stats(D12, R1, a.perm_global, SEED + 10000)
    m1p = pp(m1obs, m1null[:, 0])
    m1boot = m1_boot(outcome_model_integrity, a.bootstrap, SEED + 11000)
    _, C1u = participant_contrasts(D12, X12)
    m1u, _ = m1_stat_from_contrasts(C1u)
    m1unull = m1_null_stats(D12, X12, a.perm_global, SEED + 12000)
    m1up = pp(m1u, m1unull[:, 0])
    cc_mask = np.isfinite(X12raw).all(1)
    ccdf = D12.loc[cc_mask].copy().reset_index(drop=True)
    ccXraw = X12raw[cc_mask]
    keep = []
    for s, g in ccdf.groupby('subject_id'):
        vc = g.trajectory_family_1based.value_counts()
        if len(g) >= 16 and vc.get(1, 0) >= 4 and (vc.get(2, 0) >= 4):
            keep.append(s)
    ccdf = ccdf[ccdf.subject_id.isin(keep)].reset_index(drop=True)
    ccXraw = D12.loc[cc_mask].loc[D12.loc[cc_mask].subject_id.isin(keep), BIO].to_numpy(float) if keep else np.empty((0, 40))
    m1ccp = np.nan
    m1ccobs = np.nan
    m1ccnull = np.full(a.perm_global, np.nan)
    if len(keep) >= 16:
        ccX, _, _ = prep_fit(ccXraw)
        ccR = nuisance_residuals(ccdf, ccX)
        _, ccC = participant_contrasts(ccdf, ccR)
        m1ccobs, _ = m1_stat_from_contrasts(ccC)
        m1ccnull = m1_null_stats(ccdf, ccR, a.perm_global, SEED + 13000)[:, 0]
        m1ccp = pp(m1ccobs, m1ccnull)
    lopo = []
    for i, s in enumerate(subs1):
        v = np.delete(outcome_model_integrity, i, axis=0).mean(0)
        lopo.append({'test': 'M1', 'subject_id': s, 'cosine_to_full': cosine(m1vec, v), 'statistic_without_subject': float(np.dot(v, v))})
    m1mincos = float(np.nanmin([x['cosine_to_full'] for x in lopo]))
    m1swaperr = abs(m1obs - m1obs)
    note('M2 observed nested LOPO logistic ridge.')
    y12 = (D12.trajectory_family_1based.to_numpy(int) == 2).astype(int)
    b0, b1, m2obs, pb, pf, sel = nested_oof(D12, y12, X12raw, True)
    m2boot, m2part = m2_boot(D12, y12, pb, pf, a.bootstrap, SEED + 21000)
    note('M2 prespecified permutation null (checkpointed).')
    m2null = checkpoint_m2(D12, X12raw, a.perm_cv, SEED + 22000, out['cache'], a.jobs, note)
    m2p = pp(m2obs, m2null)
    m2cc = np.nan
    if len(keep) >= 16:
        ccm2 = D12.loc[cc_mask].copy()
        ccm2 = ccm2[ccm2.subject_id.isin(keep)].reset_index(drop=True)
        ccy = (ccm2.trajectory_family_1based.to_numpy(int) == 2).astype(int)
        ccxr = ccm2[BIO].to_numpy(float)
        m2cc = nested_oof(ccm2, ccy, ccxr, False)[2]
    m2ci = pci(m2boot)
    m2med = float(m2part.delta_balanced_accuracy.median())
    m2frac = float((m2part.delta_balanced_accuracy >= 0).mean())
    m2swaperr = 0.0
    for _, rr in m2part.iterrows():
        lopo.append({'test': 'M2', 'subject_id': rr.subject_id, 'cosine_to_full': np.nan, 'statistic_without_subject': np.nan, 'outer_delta_balanced_accuracy': float(rr.delta_balanced_accuracy), 'outer_baseline_balanced_accuracy': float(rr.baseline_balanced_accuracy), 'outer_full_balanced_accuracy': float(rr.full_balanced_accuracy)})
    note('M3 participant-balanced residual cross-correlation coupling.')
    within_trial_geometry_continuity, _, _ = prep_fit(X3raw)
    P3 = nuisance_residuals(D3, within_trial_geometry_continuity)
    GR3 = nuisance_residuals(D3, G3)
    subs3, mats3 = participant_corr_matrices(D3, P3, GR3)
    m3obs, m3M = m3_stat(mats3)
    m3null = m3_null(D3, P3, GR3, a.perm_global, SEED + 30000)
    m3p = pp(m3obs, m3null[:, 0])
    m3boot = m3_boot(mats3, a.bootstrap, SEED + 31000)
    subs3u, mats3u = participant_corr_matrices(D3, within_trial_geometry_continuity, G3)
    m3u, m3Mu = m3_stat(mats3u)
    m3unull = m3_null(D3, within_trial_geometry_continuity, G3, a.perm_global, SEED + 32000)
    m3up = pp(m3u, m3unull[:, 0])
    cc3mask = np.isfinite(X3raw).all(1)
    cc3df = D3.loc[cc3mask].copy().reset_index(drop=True)
    keep3 = [s for s, g in cc3df.groupby('subject_id') if len(g) >= 16]
    cc3df = cc3df[cc3df.subject_id.isin(keep3)].reset_index(drop=True)
    m3ccp = np.nan
    m3ccobs = np.nan
    m3ccnull = np.full(a.perm_global, np.nan)
    if len(keep3) >= 16:
        cc3Xraw = cc3df[BIO].to_numpy(float)
        cc3X, _, _ = prep_fit(cc3Xraw)
        cc3P = nuisance_residuals(cc3df, cc3X)
        cc3G = cc3df[GEOM].to_numpy(float)
        cc3GR = nuisance_residuals(cc3df, cc3G)
        _, cc3m = participant_corr_matrices(cc3df, cc3P, cc3GR)
        m3ccobs, _ = m3_stat(cc3m)
        m3ccnull = m3_null(cc3df, cc3P, cc3GR, a.perm_global, SEED + 33000)[:, 0]
        m3ccp = pp(m3ccobs, m3ccnull)
    for i, s in enumerate(subs3):
        Mlo = np.delete(mats3, i, axis=0).mean(0)
        lopo.append({'test': 'M3', 'subject_id': s, 'cosine_to_full': cosine(m3M, Mlo), 'statistic_without_subject': float(np.linalg.norm(Mlo, 'fro'))})
    m3mincos = float(np.nanmin([x['cosine_to_full'] for x in lopo if x['test'] == 'M3']))
    raw = np.array([m1p, m2p, m3p])
    adj = holm(raw)
    primary = adj < 0.05
    blocks = []
    m1block = False
    m3block = False
    if primary[0]:
        obs = [float(np.dot(m1vec[:36], m1vec[:36])), float(np.dot(m1vec[36:], m1vec[36:]))]
        ps = [pp(obs[0], m1null[:, 1]), pp(obs[1], m1null[:, 2])]
        ha = holm(ps)
        m1block = bool(np.any(ha < 0.05))
        for name, o, pv, hv in zip(['GSR', 'HR'], obs, ps, ha):
            blocks.append({'parent': 'M1', 'block': name, 'observed_statistic': o, 'raw_p': pv, 'holm_within_parent': hv, 'supported': bool(hv < 0.05)})
    if primary[2]:
        obs = [float(np.linalg.norm(m3M[:36], 'fro')), float(np.linalg.norm(m3M[36:], 'fro'))]
        ps = [pp(obs[0], m3null[:, 1]), pp(obs[1], m3null[:, 2])]
        ha = holm(ps)
        m3block = bool(np.any(ha < 0.05))
        for name, o, pv, hv in zip(['GSR', 'HR'], obs, ps, ha):
            blocks.append({'parent': 'M3', 'block': name, 'observed_statistic': o, 'raw_p': pv, 'holm_within_parent': hv, 'supported': bool(hv < 0.05)})
    m1rob = bool(m1up < 0.05 and (np.isnan(m1ccp) or m1ccp < 0.05) and (m1mincos >= 0.9) and m1block and (m1swaperr <= 1e-12))
    m2rob = bool(m2ci[0] > 0 and (np.isnan(m2cc) or m2cc > 0) and (m2med > 0) and (m2frac >= 0.6) and (m2swaperr <= 1e-12))
    m3rob = bool(m3up < 0.05 and (np.isnan(m3ccp) or m3ccp < 0.05) and (m3mincos >= 0.9) and m3block)
    robust = np.array([m1rob, m2rob, m3rob])
    validated_candidate = primary & robust
    any_gate = bool(np.any(validated_candidate))
    pd.DataFrame(rows).to_csv(out['rehash'], index=False)
    pd.DataFrame({'feature': BIO, 'mean_adjusted_family1_minus_family2': m1vec}).to_csv(out['m1vec'], index=False)
    pd.DataFrame(outcome_model_integrity, columns=BIO).assign(subject_id=subs1).loc[:, ['subject_id', *BIO]].to_csv(out['m1part'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_global), 'adjusted_global_statistic': m1null[:, 0], 'adjusted_GSR_statistic': m1null[:, 1], 'adjusted_HR_statistic': m1null[:, 2], 'unadjusted_global_statistic': m1unull[:, 0], 'complete_case_adjusted_global_statistic': m1ccnull}).to_csv(out['m1null'], index=False)
    pred = D12[['subject_id', 'trial_number', 'context', 'normalized_trial_order', 'trajectory_family_1based']].copy()
    pred['observed_binary_family2'] = y12
    pred['baseline_prediction'] = pb
    pred['full_prediction'] = pf
    pred.to_csv(out['m2pred'], index=False)
    pd.DataFrame(sel).to_csv(out['m2sel'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_cv), 'delta_balanced_accuracy': m2null}).to_csv(out['m2null'], index=False)
    rowsm = []
    for i, f in enumerate(BIO):
        for j, g in enumerate(GEOM):
            rowsm.append({'physiology_feature': f, 'geometry_feature': g, 'mean_residual_correlation': m3M[i, j]})
    pd.DataFrame(rowsm).to_csv(out['m3mat'], index=False)
    pm = []
    for si, s in enumerate(subs3):
        for i, f in enumerate(BIO):
            for j, g in enumerate(GEOM):
                pm.append({'subject_id': s, 'physiology_feature': f, 'geometry_feature': g, 'residual_correlation': mats3[si, i, j]})
    pd.DataFrame(pm).to_csv(out['m3part'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_global), 'adjusted_global_statistic': m3null[:, 0], 'adjusted_GSR_statistic': m3null[:, 1], 'adjusted_HR_statistic': m3null[:, 2], 'unadjusted_global_statistic': m3unull[:, 0], 'complete_case_adjusted_global_statistic': m3ccnull}).to_csv(out['m3null'], index=False)
    pd.DataFrame([{'test': 'M1', 'observed_statistic': m1obs, 'bootstrap_ci_low': pci(m1boot)[0], 'bootstrap_ci_high': pci(m1boot)[1]}, {'test': 'M2', 'observed_statistic': m2obs, 'bootstrap_ci_low': m2ci[0], 'bootstrap_ci_high': m2ci[1]}, {'test': 'M3', 'observed_statistic': m3obs, 'bootstrap_ci_low': pci(m3boot)[0], 'bootstrap_ci_high': pci(m3boot)[1]}]).to_csv(out['boot'], index=False)
    pd.concat([pd.DataFrame({'test': 'M1', 'replicate': np.arange(a.bootstrap), 'statistic': m1boot}), pd.DataFrame({'test': 'M2', 'replicate': np.arange(a.bootstrap), 'statistic': m2boot}), pd.DataFrame({'test': 'M3', 'replicate': np.arange(a.bootstrap), 'statistic': m3boot})], ignore_index=True).to_csv(out['bootrep'], index=False)
    pd.DataFrame(lopo).to_csv(out['lopo'], index=False)
    pd.DataFrame(blocks, columns=['parent', 'block', 'observed_statistic', 'raw_p', 'holm_within_parent', 'supported']).to_csv(out['blocks'], index=False)
    pd.DataFrame([{'test': 'M1', 'participants': len(keep), 'observed_statistic': m1ccobs, 'permutation_p': m1ccp, 'feasible': len(keep) >= 16}, {'test': 'M2', 'participants': len(keep), 'observed_statistic': m2cc, 'permutation_p': np.nan, 'feasible': len(keep) >= 16}, {'test': 'M3', 'participants': len(keep3), 'observed_statistic': m3ccobs, 'permutation_p': m3ccp, 'feasible': len(keep3) >= 16}]).to_csv(out['cc'], index=False)
    robust_rows = [{'test': 'M1', 'unadjusted_p': m1up, 'complete_case_p_or_gain': m1ccp, 'min_lopo_cosine': m1mincos, 'bootstrap_ci_low': pci(m1boot)[0], 'participant_gain_median': np.nan, 'participant_gain_nonnegative_fraction': np.nan, 'block_followup_any_supported': m1block, 'label_swap_error': m1swaperr, 'robustness_pass': m1rob}, {'test': 'M2', 'unadjusted_p': np.nan, 'complete_case_p_or_gain': m2cc, 'min_lopo_cosine': np.nan, 'bootstrap_ci_low': m2ci[0], 'participant_gain_median': m2med, 'participant_gain_nonnegative_fraction': m2frac, 'block_followup_any_supported': np.nan, 'label_swap_error': m2swaperr, 'robustness_pass': m2rob}, {'test': 'M3', 'unadjusted_p': m3up, 'complete_case_p_or_gain': m3ccp, 'min_lopo_cosine': m3mincos, 'bootstrap_ci_low': pci(m3boot)[0], 'participant_gain_median': np.nan, 'participant_gain_nonnegative_fraction': np.nan, 'block_followup_any_supported': m3block, 'label_swap_error': np.nan, 'robustness_pass': m3rob}]
    pd.DataFrame(robust_rows).to_csv(out['robust'], index=False)
    glob = pd.DataFrame([{'test': 'M1', 'statistic_name': 'squared_L2_family_contrast', 'observed_statistic': m1obs, 'raw_p': m1p, 'holm_p': adj[0], 'primary_supported': bool(primary[0]), 'robustness_pass': m1rob, 'validation_gate_pass': bool(validated_candidate[0])}, {'test': 'M2', 'statistic_name': 'LOPO_balanced_accuracy_gain', 'observed_statistic': m2obs, 'raw_p': m2p, 'holm_p': adj[1], 'primary_supported': bool(primary[1]), 'robustness_pass': m2rob, 'validation_gate_pass': bool(validated_candidate[1]), 'baseline_balanced_accuracy': b0, 'full_balanced_accuracy': b1}, {'test': 'M3', 'statistic_name': 'frobenius_residual_cross_correlation', 'observed_statistic': m3obs, 'raw_p': m3p, 'holm_p': adj[2], 'primary_supported': bool(primary[2]), 'robustness_pass': m3rob, 'validation_gate_pass': bool(validated_candidate[2])}])
    glob.to_csv(out['global'], index=False)
    nextp = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT, 'next_phase_permitted': True, 'discovery_primary_support_any': bool(np.any(primary)), 'discovery_support_plus_robustness_any': any_gate, 'Validation_physio_may_open_next_phase': any_gate, 'Confirmation_physio_may_open': False, 'next_phase_role': 'specification one-shot Validation physiology transfer for supported multimodal result(s)' if any_gate else 'Discovery multimodal synthesis and analysis conclusion without opening held-out physiology', 'supported_tests_for_validation': glob.loc[glob.validation_gate_pass, 'test'].tolist(), 'forbidden': ['retuning Family 1/2', 'opening Confirmation physiology', 'testing unsupported Discovery hypotheses in held-out data', 'outcome-driven physiology feature selection']}
    atom_json(out['next'], nextp)
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_multimodal_inference_complete', 'counts': {'joined_trials': len(D), 'eligible_M1_M2_participants': D12.subject_id.nunique(), 'eligible_M3_participants': D3.subject_id.nunique(), 'complete_case_M1_M2_participants': len(keep), 'complete_case_M3_participants': len(keep3), 'biological_features': 40, 'M1_M3_permutations': a.perm_global, 'M2_permutations': a.perm_cv, 'bootstraps': a.bootstrap}, 'tests': glob.to_dict('records'), 'decision': nextp, 'boundary': {'Validation_physio_values_opened': False, 'Confirmation_physio_values_opened': False}}
    atom_json(out['summary'], summary)
    out['report'].write_text('# Phase ' + PHASE + ' — Discovery EEG × physiology inference\n\nStatus: **discovery_multimodal_inference_complete**\n' + ''.join([f'- **{r.test}**: {r.statistic_name}={r.observed_statistic:.6g}; raw p={r.raw_p:.6g}; Holm p={r.holm_p:.6g}; primary={bool(r.primary_supported)}; robustness={bool(r.robustness_pass)}; Validation gate={bool(r.validation_gate_pass)}.' for _, r in glob.iterrows()]) + f'\nValidation physiology may open next phase: **{any_gate}**. Confirmation physiology remains closed. Family 1/2 remain non-semantic fixed EEG geometry labels.', encoding='utf-8')
    keys = ['resolution', 'rehash', 'global', 'm1vec', 'm1part', 'm1null', 'm2pred', 'm2sel', 'm2null', 'm3mat', 'm3part', 'm3null', 'boot', 'robust', 'lopo', 'blocks', 'cc', 'summary', 'report', 'next']
    hashes = {k: sha(out[k]) for k in keys}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_multimodal_inference_complete', 'source_hashes': {'discovery_multimodal_inference_script': sha(Path(__file__)), 'multimodal_physio_design_policy': raw_eeg_epoch_folds_POLICY_SHA, 'multimodal_physio_design_hypotheses': raw_eeg_epoch_folds_HYP_SHA, 'discovery_multimodal_attachment_qc_lock': raw_eeg_preprocessing_qc_LOCK_SHA, 'discovery_multimodal_attachment_qc_manifest': raw_eeg_preprocessing_qc_MANIFEST_SHA, 'discovery_multimodal_attachment_qc_bridge_addendum': raw_eeg_preprocessing_qc_ADDENDUM_SHA, 'cross_person_alignment_fit_standardized_geometry': GEOM_SHA}, 'implementation_resolution_sha256': resolution_sha, 'output_hashes': hashes, 'boundary': summary['boundary'], 'decision': nextp}
    atom_json(out['contract'], contract)
    hashes['contract'] = sha(out['contract'])
    atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': hashes, 'Validation_physio_may_open_next_phase': any_gate, 'Confirmation_physio_may_open': False})
    print(glob.to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())