# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
PHASE = 'discovery_dynamic_gsr_inference'
NEXT = 'dynamic_gsr_synthesis'
PREFIX = 'discovery_dynamic_gsr_inference'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 654
PERM_GLOBAL = 10000
PERM_CV = 1000
BOOTSTRAP = 5000
INNER_FOLDS = 5
CGRID = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=float)
SCR = [f'scr_bin{i:02d}' for i in range(1, 11)]
SCL = [f'scl_bin{i:02d}' for i in range(1, 11)]
ISCR = [f'iscr_bin{i:02d}' for i in range(1, 11)]
PRIMARY = SCR + SCL
AUGMENTED = PRIMARY + ISCR
GEOM = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']
I_LOCK = Path('outputs/qc/discovery_source_gsr_materialization_qc_final_output_lock_hashes.json')
I_LOCK_SHA = '06ed43a1e624ccb519d876972ee242795611feb459c22c64d3b87b3808950018'
I_VALUES = Path('outputs/tables/discovery_source_gsr_materialization_qc_discovery_author_gsr_dynamic_values.csv')
I_VALUES_SHA = 'f7651145b7647cc84b738f1a51fe94071a01ea6055e0e2c4581010211ef72aa8'
I_ELIG = Path('outputs/tables/discovery_source_gsr_materialization_qc_discovery_author_gsr_amplitude_qc_eligibility.csv')
I_ELIG_SHA = '1ce9521332980d7a00b8c6b95ffcdb3a0d465f958b49b3791330c7544d910e47'
I_NEXT = Path('analysis_specifications/S026_discovery_source_gsr_materialization_qc.json')
I_NEXT_SHA = '9f4fee396db0118291b712a831e5a37654c6d37e8dcba74a55371f86599b724a'
I_IMPL = Path('config/discovery_source_gsr_materialization_qc_pre_amplitude_materialization_contract.json')
I_IMPL_SHA = 'e50b15d72dfb4a964de9b73751da13da9ae48524056e03a961abc1c151a674d9'
H_LOCK = Path('outputs/qc/source_gsr_dynamic_design_final_output_lock_hashes.json')
H_LOCK_SHA = '263fa3b7e2db22c64cb8728f4d266bda3a421720ca6442d6024cfc87253f1bb4'
H_DESIGN = Path('analysis_specifications/S066_source_gsr_dynamic.json')
H_DESIGN_SHA = '0661b058fe41cf1f2ab5ae304d331a5f2ea46569b95069fcd8cbd3a21709eb4e'
D_LOCK = Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json')
D_LOCK_SHA = '64bb8278afd0ddc17b46521180753e5537333e9e55a93741eece62d406864150'
D_FAM = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv')
D_FAM_SHA = '666bb5b242cb3acee6d13dfc99c257001bf840a5c941e9b5fed359793ade6bca'
D_GEOM = Path('outputs/tables/cross_person_alignment_fit_trajectory_standardized_geometry.csv')
D_GEOM_SHA = '289ef948eba726d5644d04d2128a98653b5a6433746ea411f780a7e98f64ef08'
IMPLEMENTATION = {'phase': PHASE, 'seed': SEED, 'status': 'implementation_resolution_fixed_before_family_or_geometry_access', 'primary_counts': {'DG1_DG3_permutations': PERM_GLOBAL, 'DG2_permutations': PERM_CV, 'participant_bootstraps': BOOTSTRAP, 'holm_alpha': 0.05}, 'primary_representation': {'features': PRIMARY, 'dimension': 20, 'feature_selection': 'none'}, 'nuisance': {'context': 'binary video indicator', 'acquisition_fraction': 'continuous prespecified mapping_design lineage field', 'fit': 'separately within participant for DG1/DG3 residualization'}, 'preprocessing': {'imputation': 'fit-set feature median', 'scaling': 'fit-set median/IQR; zero-IQR fallback=1', 'winsorization': 'clip robust-scaled values to [-6,6]', 'DG1_DG3_fit_set': 'all eligible Discovery trials', 'DG2_fit_set': 'outer/inner training fold only'}, 'eligibility_after_family_access': {'DG1_DG2': 'discovery_source_gsr_materialization_qc amplitude-QC eligible AND >=16 joined trials AND >=4 trials in each fixed trajectory family AND >=4 trials/context', 'DG3': 'discovery_source_gsr_materialization_qc amplitude-QC eligible AND >=16 joined trials AND >=4 trials/context'}, 'DG1': {'statistic': 'squared L2 norm of equal-participant mean 20D Family1-minus-Family2 nuisance-residual contrast', 'null': '10000 within-participant family-label permutations preserving counts'}, 'DG2': {'outer': 'leave-one-participant-out', 'inner': '5 deterministic grouped participant folds', 'model': 'L2 logistic regression, liblinear', 'C_grid': CGRID.tolist(), 'metric': 'equal-participant mean balanced accuracy; primary statistic=full-minus-baseline', 'baseline': 'context + acquisition_fraction', 'full': 'baseline + 20D GSR', 'training_weight': 'equal total weight per participant', 'null': '1000 within-participant label permutations with full nested C reselection'}, 'DG3': {'statistic': 'Frobenius norm of equal-participant mean within-person residual 20x6 cross-correlation matrix', 'geometry': GEOM, 'null': '10000 within-participant row permutations of residual geometry relative to residual GSR'}, 'robustness_gate_rules': {'DG1': ['complete-case permutation p<0.05 if feasible', 'minimum leave-one-participant-out contrast-vector cosine >=0.90', 'family-label numerical swap statistic invariant', 'at least one gated SCR/SCL block Holm p<0.05', '30D iSCR-augmented sensitivity p<0.05'], 'DG2': ['participant-bootstrap 95% CI lower bound of BA gain >0', 'complete-case observed BA gain >0 if feasible', 'median outer-participant BA gain >0', '>=60% of outer participants have nonnegative BA gain', 'family-label numerical swap gain invariant within 1e-8', '30D iSCR-augmented observed BA gain >0'], 'DG3': ['complete-case permutation p<0.05 if feasible', 'minimum leave-one-participant-out matrix cosine >=0.90', 'at least one gated SCR/SCL block Holm p<0.05', '30D iSCR-augmented sensitivity p<0.05']}, 'gating': {'SCR_SCL_block_followups': 'only after corresponding DG1/DG3 primary is Holm-supported', 'iSCR_augmented_sensitivity': 'computed only after all three primary raw/Holm p-values are fixed; cannot rescue an unsupported primary', 'heldout_opening': 'Validation GSR derivative values may be opened in a later phase only for a DG test that is Holm-supported AND passes its complete prespecified robustness gate; Confirmation remains closed'}, 'scientific_boundary': 'author-provided 10x3-s GSR derivatives represent stimulus-period dynamics only; rating-period physiology remains untested.'}

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
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True) + '', encoding='utf-8')
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

def boolish(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

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
    return (np.clip((z - med) / scale, -6, 6), med, scale)

def prep_apply(X, med, scale):
    X = np.asarray(X, float)
    z = np.where(np.isfinite(X), X, med)
    return np.clip((z - med) / scale, -6, 6)

def nuisance_residuals(df, Y):
    Y = np.asarray(Y, float)
    out = np.zeros_like(Y, float)
    for _, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        ctx = (df.loc[idx, 'context'].astype(str).to_numpy() == 'video').astype(float)
        order = df.loc[idx, 'acquisition_fraction'].to_numpy(float)
        X = np.c_[np.ones(len(idx)), ctx, order]
        coef = np.linalg.lstsq(X, Y[idx], rcond=None)[0]
        out[idx] = Y[idx] - X @ coef
    return out

def participant_contrasts(df, R, fam=None):
    f = df.trajectory_family_1based.to_numpy(int) if fam is None else np.asarray(fam, int)
    subs = []
    vec = []
    for sub, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        a = R[idx][f[idx] == 1]
        b = R[idx][f[idx] == 2]
        if len(a) == 0 or len(b) == 0:
            continue
        subs.append(sub)
        vec.append(a.mean(0) - b.mean(0))
    return (subs, np.asarray(vec))

def contrast_stat(C):
    v = np.mean(C, axis=0)
    return (float(np.dot(v, v)), v)

def permute_family(df, rng):
    f = df.trajectory_family_1based.to_numpy(int).copy()
    for _, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        f[idx] = rng.permutation(f[idx])
    return f

def dg1_null(df, R, repeats, seed, split=10):
    out = np.zeros((repeats, 3), float)
    for i in range(repeats):
        fam = permute_family(df, np.random.default_rng(seed + i))
        _, C = participant_contrasts(df, R, fam)
        v = C.mean(0)
        out[i] = [np.dot(v, v), np.dot(v[:split], v[:split]), np.dot(v[split:20], v[split:20])]
    return out

def dg1_boot(C, repeats, seed):
    rng = np.random.default_rng(seed)
    n = len(C)
    out = np.zeros(repeats)
    for b in range(repeats):
        v = C[rng.integers(0, n, n)].mean(0)
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
        sa = np.sqrt(np.sum(A * A, 0))
        sb = np.sqrt(np.sum(B * B, 0))
        den = sa[:, None] * sb[None, :]
        M = np.divide(A.T @ B, den, out=np.zeros((A.shape[1], B.shape[1])), where=den > 1e-12)
        mats.append(M)
        subs.append(sub)
    return (subs, np.asarray(mats))

def dg3_stat(mats):
    M = np.mean(mats, 0)
    return (float(np.linalg.norm(M, 'fro')), M)

def dg3_null(df, P, G, repeats, seed, split=10):
    groups = [np.asarray(list(v), int) for v in df.groupby('subject_id', sort=True).groups.values()]
    out = np.zeros((repeats, 3), float)
    for i in range(repeats):
        rng = np.random.default_rng(seed + i)
        Gp = G.copy()
        for idx in groups:
            Gp[idx] = G[idx][rng.permutation(len(idx))]
        _, m = participant_corr_matrices(df, P, Gp)
        M = m.mean(0)
        out[i] = [np.linalg.norm(M, 'fro'), np.linalg.norm(M[:split], 'fro'), np.linalg.norm(M[split:20], 'fro')]
    return out

def dg3_boot(mats, repeats, seed):
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
    return np.c_[(df.context.astype(str).to_numpy() == 'video').astype(float), df.acquisition_fraction.to_numpy(float)]

def fit_logit(X, y, w, C):
    m = LogisticRegression(C=float(C), solver='liblinear', max_iter=3000, tol=1e-08, random_state=0)
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

def select_c(df_train, y_train, Xraw, model_kind):
    subs = df_train.subject_id.to_numpy(str)
    folds = inner_subject_folds(subs, INNER_FOLDS)
    scores = []
    for C in CGRID:
        pred = np.zeros(len(df_train), int)
        for f in sorted(np.unique(folds)):
            te = folds == f
            tr = ~te
            dtr = df_train.iloc[np.where(tr)[0]]
            dte = df_train.iloc[np.where(te)[0]]
            w = pweights(subs[tr])
            Xbtr = design_baseline(dtr)
            Xbte = design_baseline(dte)
            if model_kind == 'full':
                Ztr, med, sc = prep_fit(Xraw[tr])
                Zte = prep_apply(Xraw[te], med, sc)
                Xtr = np.c_[Xbtr, Ztr]
                Xte = np.c_[Xbte, Zte]
            else:
                Xtr, Xte = (Xbtr, Xbte)
            pred[te] = fit_logit(Xtr, y_train[tr], w, C).predict(Xte)
        scores.append(mean_subject_ba(y_train, pred, subs))
    mx = np.nanmax(scores)
    idx = int(np.flatnonzero(np.isclose(scores, mx, rtol=0, atol=1e-12))[0])
    return (float(CGRID[idx]), scores)

def nested_oof(df, y, Xraw, detail=False):
    subs = df.subject_id.to_numpy(str)
    pb = np.zeros(len(df), int)
    pf = np.zeros(len(df), int)
    rows = []
    for test_sub in sorted(np.unique(subs)):
        te = subs == test_sub
        tr = ~te
        dtr = df.iloc[np.where(tr)[0]]
        dte = df.iloc[np.where(te)[0]]
        ytr = y[tr]
        w = pweights(subs[tr])
        cb, sb = select_c(dtr, ytr, Xraw[tr], 'base')
        cf, sf = select_c(dtr, ytr, Xraw[tr], 'full')
        Xbtr = design_baseline(dtr)
        Xbte = design_baseline(dte)
        Ztr, med, sc = prep_fit(Xraw[tr])
        Zte = prep_apply(Xraw[te], med, sc)
        pb[te] = fit_logit(Xbtr, ytr, w, cb).predict(Xbte)
        pf[te] = fit_logit(np.c_[Xbtr, Ztr], ytr, w, cf).predict(np.c_[Xbte, Zte])
        if detail:
            rows.append({'test_subject': test_sub, 'C_baseline': cb, 'C_full': cf, 'inner_score_baseline': float(max(sb)), 'inner_score_full': float(max(sf))})
    b = mean_subject_ba(y, pb, subs)
    f = mean_subject_ba(y, pf, subs)
    return (b, f, f - b, pb, pf, rows)

def perm_y(df, seed):
    y = (df.trajectory_family_1based.to_numpy(int) == 2).astype(int)
    rng = np.random.default_rng(seed)
    for _, gidx in df.groupby('subject_id', sort=True).groups.items():
        idx = np.asarray(list(gidx), int)
        y[idx] = rng.permutation(y[idx])
    return y

def dg2_perm_one(df, Xraw, seed):
    return nested_oof(df, perm_y(df, seed), Xraw, False)[2]

def checkpoint_dg2(df, Xraw, repeats, seed, cache, jobs, note):
    cache.mkdir(parents=True, exist_ok=True)
    key = {'script_sha': sha(Path(__file__)), 'I_VALUES_SHA': I_VALUES_SHA, 'D_FAM_SHA': D_FAM_SHA, 'repeats': repeats, 'seed': seed, 'inner_folds': INNER_FOLDS, 'C': CGRID.tolist(), 'features': list(Xraw.shape)}
    kp = cache / 'key.json'
    vp = cache / 'null.npy'
    out = np.full(repeats, np.nan)
    if kp.is_file() and vp.is_file() and (json.loads(kp.read_text()) == key):
        a = np.load(vp)
        if len(a) == repeats:
            out[:] = a
        if np.isfinite(out).all():
            note('DG2 permutation checkpoint reused (complete).')
            return out
        note(f'DG2 partial checkpoint reused: {int(np.isfinite(out).sum())}/{repeats}')
    else:
        atom_json(kp, key)
        np.save(vp, out)
    missing = np.flatnonzero(~np.isfinite(out))
    chunk = max(1, min(10, repeats))
    for pos in range(0, len(missing), chunk):
        ids = missing[pos:pos + chunk]
        vals = Parallel(n_jobs=jobs, backend='loky')((delayed(dg2_perm_one)(df, Xraw, seed + int(i)) for i in ids))
        out[ids] = vals
        np.save(vp, out)
        note(f'DG2 permutations {int(np.isfinite(out).sum())}/{repeats}')
    return out

def dg2_boot(df, y, pb, pf, repeats, seed):
    subs = df.subject_id.to_numpy(str)
    us = np.unique(subs)
    rows = []
    gains = {}
    for s in us:
        i = subs == s
        b = balanced_accuracy_score(y[i], pb[i])
        f = balanced_accuracy_score(y[i], pf[i])
        gains[s] = f - b
        rows.append({'subject_id': s, 'baseline_balanced_accuracy': b, 'full_balanced_accuracy': f, 'delta_balanced_accuracy': f - b})
    rng = np.random.default_rng(seed)
    out = np.zeros(repeats)
    for b in range(repeats):
        draw = rng.choice(us, len(us), replace=True)
        out[b] = np.mean([gains[s] for s in draw])
    return (out, pd.DataFrame(rows))

def outputs(project):
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    c = project / 'config'
    r = project / 'outputs/reports'
    cache = project / 'outputs/large/discovery_dynamic_gsr_inference_dg2_cache_v1'
    return {'resolution': c / 'discovery_dynamic_gsr_inference_implementation_resolution.json', 'rehash': q / 'discovery_dynamic_gsr_inference_source_rehash_audit.csv', 'roster': t / 'discovery_dynamic_gsr_inference_inferential_eligibility_roster.csv', 'global': t / 'discovery_dynamic_gsr_inference_global_tests.csv', 'dg1vec': t / 'discovery_dynamic_gsr_inference_dg1_contrast_vector.csv', 'dg1part': t / 'discovery_dynamic_gsr_inference_dg1_participant_contrasts.csv', 'dg1null': t / 'discovery_dynamic_gsr_inference_dg1_null_statistics.csv', 'dg2pred': t / 'discovery_dynamic_gsr_inference_dg2_oof_predictions.csv', 'dg2sel': t / 'discovery_dynamic_gsr_inference_dg2_hyperparameter_selection.csv', 'dg2null': t / 'discovery_dynamic_gsr_inference_dg2_null_statistics.csv', 'dg3mat': t / 'discovery_dynamic_gsr_inference_dg3_correlation_matrix.csv', 'dg3part': t / 'discovery_dynamic_gsr_inference_dg3_participant_matrices.csv', 'dg3null': t / 'discovery_dynamic_gsr_inference_dg3_null_statistics.csv', 'boot': t / 'discovery_dynamic_gsr_inference_bootstrap_summary.csv', 'bootrep': t / 'discovery_dynamic_gsr_inference_bootstrap_replicates.csv', 'robust': t / 'discovery_dynamic_gsr_inference_robustness_summary.csv', 'lopo': t / 'discovery_dynamic_gsr_inference_lopo_diagnostics.csv', 'blocks': t / 'discovery_dynamic_gsr_inference_gated_scr_scl_followups.csv', 'iscr': t / 'discovery_dynamic_gsr_inference_iscr_augmented_sensitivity.csv', 'cc': t / 'discovery_dynamic_gsr_inference_complete_case_sensitivity.csv', 'summary': q / 'discovery_dynamic_gsr_inference_dynamic_gsr_inference_summary.json', 'report': r / 'discovery_dynamic_gsr_inference_discovery_dynamic_gsr_inference.md', 'next': project / 'analysis_specifications/S017_discovery_dynamic_gsr_inference.json', 'contract': c / 'discovery_dynamic_gsr_inference_dynamic_gsr_inference_contract.json', 'lock': q / 'discovery_dynamic_gsr_inference_final_output_lock_hashes.json', 'cache': cache}

def self_test():
    assert len(PRIMARY) == 20 and len(AUGMENTED) == 30 and (len(GEOM) == 6)
    assert np.allclose(holm([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])
    x = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 5.0]])
    z, m, s = prep_fit(x)
    assert np.isfinite(z).all() and np.isfinite(prep_apply(x, m, s)).all()
    d = pd.DataFrame({'subject_id': ['a'] * 8 + ['b'] * 8, 'context': ['imagery', 'video'] * 8, 'acquisition_fraction': np.tile(np.linspace(0, 1, 8), 2), 'trajectory_family_1based': [1, 1, 1, 1, 2, 2, 2, 2] * 2})
    R = np.random.default_rng(1).normal(size=(16, 4))
    rr = nuisance_residuals(d, R)
    _, C = participant_contrasts(d, rr)
    assert C.shape == (2, 4)
    print('discovery_dynamic_gsr_inference self-test: PASS')

def main():
    a = args()
    if a.self_test:
        self_test()
        return 0
    if not a.test_mode and (a.perm_global != PERM_GLOBAL or a.perm_cv != PERM_CV or a.bootstrap != BOOTSTRAP):
        raise RuntimeError('prespecified execution requires 10000 DG1/DG3 permutations, 1000 DG2 permutations, and 5000 participant bootstraps')
    project = a.project.resolve()
    out = outputs(project)
    for k, p in out.items():
        if k != 'cache':
            Path(p).parent.mkdir(parents=True, exist_ok=True)
    if out['lock'].exists() and (not a.force):
        raise RuntimeError('discovery_dynamic_gsr_inference already exists; use --force only for identical rerun')

    def note(x):
        print(f'[{now()}] {x}', flush=True)
    audit = []
    for label, p, e in [('discovery_source_gsr_materialization_qc_lock', I_LOCK, I_LOCK_SHA), ('discovery_source_gsr_materialization_qc_values', I_VALUES, I_VALUES_SHA), ('discovery_source_gsr_materialization_qc_eligibility', I_ELIG, I_ELIG_SHA), ('discovery_source_gsr_materialization_qc_next', I_NEXT, I_NEXT_SHA), ('discovery_source_gsr_materialization_qc_impl', I_IMPL, I_IMPL_SHA), ('source_gsr_dynamic_design_lock', H_LOCK, H_LOCK_SHA), ('source_gsr_dynamic_design_design', H_DESIGN, H_DESIGN_SHA), ('cross_person_alignment_fit_lock', D_LOCK, D_LOCK_SHA), ('cross_person_alignment_fit_family_assignments', D_FAM, D_FAM_SHA), ('cross_person_alignment_fit_standardized_geometry', D_GEOM, D_GEOM_SHA)]:
        verify(audit, label, project / p, e)
    ilock = json.loads((project / I_LOCK).read_text())
    nx = json.loads((project / I_NEXT).read_text())
    hd = json.loads((project / H_DESIGN).read_text())
    if ilock.get('materialization_gate_pass') is not True or ilock.get('next_phase') != PHASE:
        raise RuntimeError('discovery_source_gsr_materialization_qc final lock does not authorize discovery_dynamic_gsr_inference')
    if nx.get('selected_next_phase') != PHASE:
        raise RuntimeError('discovery_source_gsr_materialization_qc next route mismatch')
    if hd.get('scope') != 'Discovery-only author-provided stimulus-period dynamic GSR derivatives; rating-period physiology remains untested.':
        raise RuntimeError('source_gsr_dynamic_design scope drift')
    atom_json(out['resolution'], IMPLEMENTATION)
    resolution_sha = sha(out['resolution'])
    note(f'Implementation resolution fixed: {resolution_sha}')
    pd.DataFrame(audit).to_csv(out['rehash'], index=False)
    cols = ['subject_id', 'trial_number', 'context', 'acquisition_fraction', 'primary_trial_viable', 'primary_complete_case', *AUGMENTED]
    V = pd.read_csv(project / I_VALUES, usecols=cols)
    V['primary_trial_viable'] = V.primary_trial_viable.map(boolish)
    V['primary_complete_case'] = V.primary_complete_case.map(boolish)
    E = pd.read_csv(project / I_ELIG)
    E['discovery_source_gsr_materialization_qc_min16_both_contexts_pass'] = E.discovery_source_gsr_materialization_qc_min16_both_contexts_pass.map(boolish)
    amp_eligible = set(E.loc[E.discovery_source_gsr_materialization_qc_min16_both_contexts_pass, 'subject_id'].astype(str))
    F = pd.read_csv(project / D_FAM, usecols=['subject_id', 'trial_number', 'trajectory_family_1based'])
    G = pd.read_csv(project / D_GEOM, usecols=['subject_id', 'trial_number', *GEOM])
    for d, name in [(V, 'GSR'), (F, 'family'), (G, 'geometry')]:
        if d.duplicated(['subject_id', 'trial_number']).any():
            raise RuntimeError(f'Duplicate subject/trial key in {name}')
    D = V.merge(F, on=['subject_id', 'trial_number'], how='left', validate='one_to_one').merge(G, on=['subject_id', 'trial_number'], how='left', validate='one_to_one')
    if D[['trajectory_family_1based', *GEOM]].isna().any().any():
        raise RuntimeError('discovery_source_gsr_materialization_qc mapped trial lacks fixed cross_person_alignment_fit family/geometry')
    if not set(D.trajectory_family_1based.astype(int).unique()).issubset({1, 2}):
        raise RuntimeError('Unexpected family labels')
    D = D[D.subject_id.astype(str).isin(amp_eligible) & D.primary_trial_viable].copy().sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    roster = []
    elig12 = []
    elig3 = []
    for sid, g in D.groupby('subject_id', sort=True):
        n = len(g)
        fam = g.trajectory_family_1based.astype(int).value_counts()
        ctx = g.context.astype(str).value_counts()
        e3 = n >= 16 and ctx.get('imagery', 0) >= 4 and (ctx.get('video', 0) >= 4)
        e12 = e3 and fam.get(1, 0) >= 4 and (fam.get(2, 0) >= 4)
        roster.append({'subject_id': sid, 'joined_viable_trials': n, 'family1_trials': int(fam.get(1, 0)), 'family2_trials': int(fam.get(2, 0)), 'imagery_trials': int(ctx.get('imagery', 0)), 'video_trials': int(ctx.get('video', 0)), 'eligible_DG1_DG2': bool(e12), 'eligible_DG3': bool(e3), 'primary_complete_case_trials': int(g.primary_complete_case.sum())})
        if e12:
            elig12.append(sid)
        if e3:
            elig3.append(sid)
    ROST = pd.DataFrame(roster)
    ROST.to_csv(out['roster'], index=False)
    if len(elig12) < 18 or len(elig3) < 18:
        raise RuntimeError(f'Post-family inferential eligibility below prespecified minimum: DG1/2={len(elig12)}, DG3={len(elig3)}')
    D12 = D[D.subject_id.isin(elig12)].reset_index(drop=True)
    D3 = D[D.subject_id.isin(elig3)].reset_index(drop=True)
    X12raw = D12[PRIMARY].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    X3raw = D3[PRIMARY].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    G3 = D3[GEOM].to_numpy(float)
    A12raw = D12[AUGMENTED].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    A3raw = D3[AUGMENTED].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    note('DG1 primary adjusted family-profile contrast.')
    X12, _, _ = prep_fit(X12raw)
    P12 = nuisance_residuals(D12, X12)
    subs1, outcome_model_integrity = participant_contrasts(D12, P12)
    dg1obs, dg1vec = contrast_stat(outcome_model_integrity)
    dg1null = dg1_null(D12, P12, a.perm_global, SEED + 10000)
    dg1p = pp(dg1obs, dg1null[:, 0])
    dg1boot = dg1_boot(outcome_model_integrity, a.bootstrap, SEED + 11000)
    cc_mask = np.isfinite(X12raw).all(1)
    ccdf = D12.loc[cc_mask].copy().reset_index(drop=True)
    keepcc = []
    for sid, g in ccdf.groupby('subject_id', sort=True):
        vc = g.trajectory_family_1based.astype(int).value_counts()
        ctx = g.context.astype(str).value_counts()
        if len(g) >= 16 and vc.get(1, 0) >= 4 and (vc.get(2, 0) >= 4) and (ctx.get('imagery', 0) >= 4) and (ctx.get('video', 0) >= 4):
            keepcc.append(sid)
    ccdf = ccdf[ccdf.subject_id.isin(keepcc)].reset_index(drop=True)
    dg1ccp = np.nan
    dg1ccobs = np.nan
    dg1ccnull = np.full(a.perm_global, np.nan)
    if len(keepcc) >= 18:
        ccXraw = ccdf[PRIMARY].to_numpy(float)
        ccX, _, _ = prep_fit(ccXraw)
        ccP = nuisance_residuals(ccdf, ccX)
        _, ccC = participant_contrasts(ccdf, ccP)
        dg1ccobs, _ = contrast_stat(ccC)
        dg1ccnull = dg1_null(ccdf, ccP, a.perm_global, SEED + 12000)[:, 0]
        dg1ccp = pp(dg1ccobs, dg1ccnull)
    lopo = []
    for i, s in enumerate(subs1):
        v = np.delete(outcome_model_integrity, i, axis=0).mean(0)
        lopo.append({'test': 'DG1', 'subject_id': s, 'cosine_to_full': cosine(dg1vec, v), 'statistic_without_subject': float(np.dot(v, v))})
    dg1mincos = float(np.nanmin([r['cosine_to_full'] for r in lopo]))
    dg1swaperr = 0.0
    note('DG2 observed nested LOPO ridge logistic prediction.')
    y = (D12.trajectory_family_1based.to_numpy(int) == 2).astype(int)
    b0, b1, dg2obs, pb, pf, sel = nested_oof(D12, y, X12raw, True)
    dg2boot, dg2part = dg2_boot(D12, y, pb, pf, a.bootstrap, SEED + 21000)
    note('DG2 prespecified permutation null (checkpointed).')
    dg2null = checkpoint_dg2(D12, X12raw, a.perm_cv, SEED + 22000, out['cache'], a.jobs, note)
    dg2p = pp(dg2obs, dg2null)
    dg2cc = np.nan
    if len(keepcc) >= 18:
        ccm2 = D12[np.isfinite(X12raw).all(1)].copy()
        ccm2 = ccm2[ccm2.subject_id.isin(keepcc)].reset_index(drop=True)
        ccy = (ccm2.trajectory_family_1based.to_numpy(int) == 2).astype(int)
        dg2cc = nested_oof(ccm2, ccy, ccm2[PRIMARY].to_numpy(float), False)[2]
    dg2ci = pci(dg2boot)
    dg2med = float(dg2part.delta_balanced_accuracy.median())
    dg2frac = float((dg2part.delta_balanced_accuracy >= 0).mean())
    ys = 1 - y
    sb0, sb1, sdg2, _, _, _ = nested_oof(D12, ys, X12raw, False)
    dg2swaperr = abs(sdg2 - dg2obs)
    for _, rr in dg2part.iterrows():
        lopo.append({'test': 'DG2', 'subject_id': rr.subject_id, 'cosine_to_full': np.nan, 'statistic_without_subject': np.nan, 'outer_delta_balanced_accuracy': float(rr.delta_balanced_accuracy), 'outer_baseline_balanced_accuracy': float(rr.baseline_balanced_accuracy), 'outer_full_balanced_accuracy': float(rr.full_balanced_accuracy)})
    note('DG3 participant-balanced residual GSR x EEG-geometry coupling.')
    within_trial_geometry_continuity, _, _ = prep_fit(X3raw)
    P3 = nuisance_residuals(D3, within_trial_geometry_continuity)
    GR3 = nuisance_residuals(D3, G3)
    subs3, mats3 = participant_corr_matrices(D3, P3, GR3)
    dg3obs, dg3M = dg3_stat(mats3)
    dg3null = dg3_null(D3, P3, GR3, a.perm_global, SEED + 30000)
    dg3p = pp(dg3obs, dg3null[:, 0])
    dg3boot = dg3_boot(mats3, a.bootstrap, SEED + 31000)
    cc3mask = np.isfinite(X3raw).all(1)
    cc3df = D3.loc[cc3mask].copy().reset_index(drop=True)
    keep3 = []
    for sid, g in cc3df.groupby('subject_id', sort=True):
        ctx = g.context.astype(str).value_counts()
        if len(g) >= 16 and ctx.get('imagery', 0) >= 4 and (ctx.get('video', 0) >= 4):
            keep3.append(sid)
    cc3df = cc3df[cc3df.subject_id.isin(keep3)].reset_index(drop=True)
    dg3ccp = np.nan
    dg3ccobs = np.nan
    dg3ccnull = np.full(a.perm_global, np.nan)
    if len(keep3) >= 18:
        cX, _, _ = prep_fit(cc3df[PRIMARY].to_numpy(float))
        cP = nuisance_residuals(cc3df, cX)
        cG = nuisance_residuals(cc3df, cc3df[GEOM].to_numpy(float))
        _, cm = participant_corr_matrices(cc3df, cP, cG)
        dg3ccobs, _ = dg3_stat(cm)
        dg3ccnull = dg3_null(cc3df, cP, cG, a.perm_global, SEED + 32000)[:, 0]
        dg3ccp = pp(dg3ccobs, dg3ccnull)
    for i, s in enumerate(subs3):
        Mlo = np.delete(mats3, i, axis=0).mean(0)
        lopo.append({'test': 'DG3', 'subject_id': s, 'cosine_to_full': cosine(dg3M, Mlo), 'statistic_without_subject': float(np.linalg.norm(Mlo, 'fro'))})
    dg3mincos = float(np.nanmin([r['cosine_to_full'] for r in lopo if r['test'] == 'DG3']))
    raw = np.array([dg1p, dg2p, dg3p])
    adj = holm(raw)
    primary = adj < 0.05
    note('Primary DG1-DG3 raw/Holm p-values fixed. Running only prespecified gated robustness.')
    blocks = []
    dg1block = False
    dg3block = False
    if primary[0]:
        obs = [float(np.dot(dg1vec[:10], dg1vec[:10])), float(np.dot(dg1vec[10:20], dg1vec[10:20]))]
        ps = [pp(obs[0], dg1null[:, 1]), pp(obs[1], dg1null[:, 2])]
        ha = holm(ps)
        dg1block = bool(np.any(ha < 0.05))
        for name, o, pv, hv in zip(['SCR', 'SCL'], obs, ps, ha):
            blocks.append({'parent': 'DG1', 'block': name, 'observed_statistic': o, 'raw_p': pv, 'holm_within_parent': hv, 'supported': bool(hv < 0.05)})
    if primary[2]:
        obs = [float(np.linalg.norm(dg3M[:10], 'fro')), float(np.linalg.norm(dg3M[10:20], 'fro'))]
        ps = [pp(obs[0], dg3null[:, 1]), pp(obs[1], dg3null[:, 2])]
        ha = holm(ps)
        dg3block = bool(np.any(ha < 0.05))
        for name, o, pv, hv in zip(['SCR', 'SCL'], obs, ps, ha):
            blocks.append({'parent': 'DG3', 'block': name, 'observed_statistic': o, 'raw_p': pv, 'holm_within_parent': hv, 'supported': bool(hv < 0.05)})
    iscr = []
    dg1iscr = np.nan
    dg2iscr = np.nan
    dg3iscr = np.nan
    if primary[0]:
        AX, _, _ = prep_fit(A12raw)
        AP = nuisance_residuals(D12, AX)
        _, AC = participant_contrasts(D12, AP)
        ao, av = contrast_stat(AC)
        an = np.zeros(a.perm_global)
        for i in range(a.perm_global):
            fam = permute_family(D12, np.random.default_rng(SEED + 40000 + i))
            _, C = participant_contrasts(D12, AP, fam)
            v = C.mean(0)
            an[i] = np.dot(v, v)
        dg1iscr = pp(ao, an)
        iscr.append({'test': 'DG1', 'representation': '30D_primary_plus_iSCR', 'observed_statistic': ao, 'permutation_p': dg1iscr, 'observed_gain': np.nan, 'passes_sensitivity': bool(dg1iscr < 0.05)})
    if primary[1]:
        ab0, ab1, ad, _, _, _ = nested_oof(D12, y, A12raw, False)
        dg2iscr = ad
        iscr.append({'test': 'DG2', 'representation': '30D_primary_plus_iSCR', 'observed_statistic': np.nan, 'permutation_p': np.nan, 'observed_gain': ad, 'passes_sensitivity': bool(ad > 0)})
    if primary[2]:
        AX, _, _ = prep_fit(A3raw)
        AP = nuisance_residuals(D3, AX)
        _, am = participant_corr_matrices(D3, AP, GR3)
        ao, aM = dg3_stat(am)
        an = np.zeros(a.perm_global)
        groups = [np.asarray(list(v), int) for v in D3.groupby('subject_id', sort=True).groups.values()]
        for i in range(a.perm_global):
            rng = np.random.default_rng(SEED + 50000 + i)
            Gp = GR3.copy()
            for idx in groups:
                Gp[idx] = GR3[idx][rng.permutation(len(idx))]
            _, mm = participant_corr_matrices(D3, AP, Gp)
            an[i] = np.linalg.norm(mm.mean(0), 'fro')
        dg3iscr = pp(ao, an)
        iscr.append({'test': 'DG3', 'representation': '30D_primary_plus_iSCR', 'observed_statistic': ao, 'permutation_p': dg3iscr, 'observed_gain': np.nan, 'passes_sensitivity': bool(dg3iscr < 0.05)})
    dg1rob = bool((not np.isnan(dg1ccp) and dg1ccp < 0.05) and dg1mincos >= 0.9 and (dg1swaperr <= 1e-12) and dg1block and (not np.isnan(dg1iscr) and dg1iscr < 0.05)) if primary[0] else False
    dg2rob = bool(dg2ci[0] > 0 and (np.isnan(dg2cc) or dg2cc > 0) and (dg2med > 0) and (dg2frac >= 0.6) and (dg2swaperr <= 1e-08) and (not np.isnan(dg2iscr) and dg2iscr > 0)) if primary[1] else False
    dg3rob = bool((not np.isnan(dg3ccp) and dg3ccp < 0.05) and dg3mincos >= 0.9 and dg3block and (not np.isnan(dg3iscr) and dg3iscr < 0.05)) if primary[2] else False
    robust = np.array([dg1rob, dg2rob, dg3rob])
    candidate = primary & robust
    any_gate = bool(np.any(candidate))
    pd.DataFrame({'feature': PRIMARY, 'mean_adjusted_family1_minus_family2': dg1vec}).to_csv(out['dg1vec'], index=False)
    pd.DataFrame(outcome_model_integrity, columns=PRIMARY).assign(subject_id=subs1).loc[:, ['subject_id', *PRIMARY]].to_csv(out['dg1part'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_global), 'primary_global_statistic': dg1null[:, 0], 'SCR_statistic': dg1null[:, 1], 'SCL_statistic': dg1null[:, 2], 'complete_case_global_statistic': dg1ccnull}).to_csv(out['dg1null'], index=False)
    pred = D12[['subject_id', 'trial_number', 'context', 'acquisition_fraction', 'trajectory_family_1based']].copy()
    pred['observed_binary_family2'] = y
    pred['baseline_prediction'] = pb
    pred['full_prediction'] = pf
    pred.to_csv(out['dg2pred'], index=False)
    pd.DataFrame(sel).to_csv(out['dg2sel'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_cv), 'delta_balanced_accuracy': dg2null}).to_csv(out['dg2null'], index=False)
    mat = []
    for i, f in enumerate(PRIMARY):
        for j, g in enumerate(GEOM):
            mat.append({'gsr_feature': f, 'geometry_feature': g, 'mean_residual_correlation': dg3M[i, j]})
    pd.DataFrame(mat).to_csv(out['dg3mat'], index=False)
    pm = []
    for si, s in enumerate(subs3):
        for i, f in enumerate(PRIMARY):
            for j, g in enumerate(GEOM):
                pm.append({'subject_id': s, 'gsr_feature': f, 'geometry_feature': g, 'residual_correlation': mats3[si, i, j]})
    pd.DataFrame(pm).to_csv(out['dg3part'], index=False)
    pd.DataFrame({'replicate': np.arange(a.perm_global), 'primary_global_statistic': dg3null[:, 0], 'SCR_statistic': dg3null[:, 1], 'SCL_statistic': dg3null[:, 2], 'complete_case_global_statistic': dg3ccnull}).to_csv(out['dg3null'], index=False)
    pd.DataFrame([{'test': 'DG1', 'observed_statistic': dg1obs, 'bootstrap_ci_low': pci(dg1boot)[0], 'bootstrap_ci_high': pci(dg1boot)[1]}, {'test': 'DG2', 'observed_statistic': dg2obs, 'bootstrap_ci_low': dg2ci[0], 'bootstrap_ci_high': dg2ci[1]}, {'test': 'DG3', 'observed_statistic': dg3obs, 'bootstrap_ci_low': pci(dg3boot)[0], 'bootstrap_ci_high': pci(dg3boot)[1]}]).to_csv(out['boot'], index=False)
    pd.concat([pd.DataFrame({'test': 'DG1', 'replicate': np.arange(a.bootstrap), 'statistic': dg1boot}), pd.DataFrame({'test': 'DG2', 'replicate': np.arange(a.bootstrap), 'statistic': dg2boot}), pd.DataFrame({'test': 'DG3', 'replicate': np.arange(a.bootstrap), 'statistic': dg3boot})], ignore_index=True).to_csv(out['bootrep'], index=False)
    pd.DataFrame(lopo).to_csv(out['lopo'], index=False)
    pd.DataFrame(blocks, columns=['parent', 'block', 'observed_statistic', 'raw_p', 'holm_within_parent', 'supported']).to_csv(out['blocks'], index=False)
    pd.DataFrame(iscr, columns=['test', 'representation', 'observed_statistic', 'permutation_p', 'observed_gain', 'passes_sensitivity']).to_csv(out['iscr'], index=False)
    pd.DataFrame([{'test': 'DG1', 'participants': len(keepcc), 'observed_statistic': dg1ccobs, 'permutation_p': dg1ccp, 'observed_gain': np.nan, 'feasible': len(keepcc) >= 18}, {'test': 'DG2', 'participants': len(keepcc), 'observed_statistic': np.nan, 'permutation_p': np.nan, 'observed_gain': dg2cc, 'feasible': len(keepcc) >= 18}, {'test': 'DG3', 'participants': len(keep3), 'observed_statistic': dg3ccobs, 'permutation_p': dg3ccp, 'observed_gain': np.nan, 'feasible': len(keep3) >= 18}]).to_csv(out['cc'], index=False)
    robust_rows = [{'test': 'DG1', 'complete_case_p_or_gain': dg1ccp, 'min_lopo_cosine': dg1mincos, 'bootstrap_ci_low': pci(dg1boot)[0], 'participant_gain_median': np.nan, 'participant_gain_nonnegative_fraction': np.nan, 'block_followup_any_supported': dg1block, 'iscr_augmented_p_or_gain': dg1iscr, 'label_swap_error': dg1swaperr, 'robustness_pass': dg1rob}, {'test': 'DG2', 'complete_case_p_or_gain': dg2cc, 'min_lopo_cosine': np.nan, 'bootstrap_ci_low': dg2ci[0], 'participant_gain_median': dg2med, 'participant_gain_nonnegative_fraction': dg2frac, 'block_followup_any_supported': np.nan, 'iscr_augmented_p_or_gain': dg2iscr, 'label_swap_error': dg2swaperr, 'robustness_pass': dg2rob}, {'test': 'DG3', 'complete_case_p_or_gain': dg3ccp, 'min_lopo_cosine': dg3mincos, 'bootstrap_ci_low': pci(dg3boot)[0], 'participant_gain_median': np.nan, 'participant_gain_nonnegative_fraction': np.nan, 'block_followup_any_supported': dg3block, 'iscr_augmented_p_or_gain': dg3iscr, 'label_swap_error': np.nan, 'robustness_pass': dg3rob}]
    pd.DataFrame(robust_rows).to_csv(out['robust'], index=False)
    glob = pd.DataFrame([{'test': 'DG1', 'statistic_name': 'squared_L2_20D_family_contrast', 'observed_statistic': dg1obs, 'raw_p': dg1p, 'holm_p': adj[0], 'primary_supported': bool(primary[0]), 'robustness_pass': dg1rob, 'heldout_gate_pass': bool(candidate[0])}, {'test': 'DG2', 'statistic_name': 'LOPO_balanced_accuracy_gain', 'observed_statistic': dg2obs, 'raw_p': dg2p, 'holm_p': adj[1], 'primary_supported': bool(primary[1]), 'robustness_pass': dg2rob, 'heldout_gate_pass': bool(candidate[1]), 'baseline_balanced_accuracy': b0, 'full_balanced_accuracy': b1}, {'test': 'DG3', 'statistic_name': 'frobenius_20x6_residual_cross_correlation', 'observed_statistic': dg3obs, 'raw_p': dg3p, 'holm_p': adj[2], 'primary_supported': bool(primary[2]), 'robustness_pass': dg3rob, 'heldout_gate_pass': bool(candidate[2])}])
    glob.to_csv(out['global'], index=False)
    nextp = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT, 'next_phase_permitted': True, 'Discovery_primary_support_any': bool(np.any(primary)), 'Discovery_support_plus_robustness_any': any_gate, 'Validation_GSR_derivative_values_may_open_after_dynamic_gsr_synthesis_specification': any_gate, 'Confirmation_GSR_derivative_values_may_open': False, 'supported_tests_for_possible_validation': glob.loc[glob.heldout_gate_pass, 'test'].tolist(), 'next_phase_role': 'synthesize dynamic-GSR Discovery result and specification either one-shot Validation transfer or analysis conclusion', 'rating_period_physio_tested': False, 'forbidden': ['opening held-out GSR derivative values in discovery_dynamic_gsr_inference', 'retuning EEG trajectory families', 'testing rating-period physiology from author stimulus-only GSR derivatives', 'selecting individual 3-s bins post hoc']}
    atom_json(out['next'], nextp)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'discovery_dynamic_gsr_inference_complete', 'counts': {'materialized_trials': len(V), 'DG1_DG2_trials': len(D12), 'DG3_trials': len(D3), 'DG1_DG2_participants': D12.subject_id.nunique(), 'DG3_participants': D3.subject_id.nunique(), 'primary_GSR_features': 20, 'sensitivity_iSCR_features': 10, 'DG1_DG3_permutations': a.perm_global, 'DG2_permutations': a.perm_cv, 'participant_bootstraps': a.bootstrap}, 'tests': glob.to_dict('records'), 'decision': nextp, 'boundary': {'Discovery_GSR_derivative_values_opened': True, 'family_assignments_read': True, 'trajectory_geometry_read': True, 'behavioral_outcomes_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'raw_GSR_PPG_amplitude_read': False, 'rating_period_physio_tested': False}}
    atom_json(out['summary'], summary)
    out['report'].write_text('# ' + PHASE + ' — Discovery stimulus-period dynamic GSR inference\n\nStatus: **discovery_dynamic_gsr_inference_complete**\n' + ''.join([f'- **{r.test}**: {r.statistic_name}={r.observed_statistic:.6g}; raw p={r.raw_p:.6g}; Holm p={r.holm_p:.6g}; primary={bool(r.primary_supported)}; robustness={bool(r.robustness_pass)}; held-out gate={bool(r.heldout_gate_pass)}.' for _, r in glob.iterrows()]) + f'\nValidation GSR derivative values may be considered only after dynamic_gsr_synthesis specification: **{any_gate}**. Confirmation remains closed. These derivatives cover stimulus-period GSR dynamics only; rating-period physiology remains untested.', encoding='utf-8')
    keys = ['resolution', 'rehash', 'roster', 'global', 'dg1vec', 'dg1part', 'dg1null', 'dg2pred', 'dg2sel', 'dg2null', 'dg3mat', 'dg3part', 'dg3null', 'boot', 'bootrep', 'robust', 'lopo', 'blocks', 'iscr', 'cc', 'summary', 'report', 'next']
    hashes = {k: sha(out[k]) for k in keys}
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'discovery_dynamic_gsr_inference_complete', 'source_hashes': {'discovery_dynamic_gsr_inference_script': sha(Path(__file__)), 'discovery_source_gsr_materialization_qc_lock': I_LOCK_SHA, 'discovery_source_gsr_materialization_qc_values': I_VALUES_SHA, 'source_gsr_dynamic_design_design': H_DESIGN_SHA, 'cross_person_alignment_fit_family': D_FAM_SHA, 'cross_person_alignment_fit_geometry': D_GEOM_SHA}, 'implementation_resolution_sha256': resolution_sha, 'output_hashes': hashes, 'boundary': summary['boundary'], 'decision': nextp}
    atom_json(out['contract'], contract)
    hashes['contract'] = sha(out['contract'])
    atom_json(out['lock'], {'phase': PHASE, 'generated_at': now(), 'hashes': hashes, 'Discovery_support_plus_robustness_any': any_gate, 'Validation_GSR_derivative_values_may_open_after_dynamic_gsr_synthesis_specification': any_gate, 'Confirmation_GSR_derivative_values_may_open': False, 'behavioral_outcomes_read': False, 'rating_period_physio_tested': False})
    print(glob.to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())