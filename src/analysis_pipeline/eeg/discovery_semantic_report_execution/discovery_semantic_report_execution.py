# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from joblib import Parallel, delayed
PHASE = 'discovery_semantic_report_execution'
NEXT_PHASE = 'validation_semantic_report_execution'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 540
EXPERIENCE_K = [2, 3, 4]
TRANSITION_K = [2, 3]
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
CATEGORIES = ['sadness', 'disgust', 'fear', 'neutral', 'joy', 'tenderness', 'inspiration']
SIG_COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
STD_SIG = [f'standardized__{x}' for x in SIG_COORDS]
semantic_report_execution_design_OUT = Path('outputs/interpretability/semantic_report_execution_design_semantic_report_execution_design_specification')
semantic_report_execution_design_LOCK = semantic_report_execution_design_OUT / 'qc/semantic_report_execution_design_final_output_lock_hashes.json'
semantic_report_execution_design_LOCK_SHA256 = '63b1a0f3ea555b0e03067a6ad324d8ec733baa9a34b700b05305ba72967fce25'
semantic_report_execution_design_PROGRAM = Path('analysis_specifications/S063_semantic_report.json')
semantic_report_execution_design_PROGRAM_SHA256 = 'b9b6800c4ee6bf22033b81c41031d5d805dc57c5020b6eda21d6d931cc5b28eb'
semantic_report_execution_design_NEXT = Path('analysis_specifications/S062_semantic_report_discovery.json')
semantic_report_execution_design_NEXT_SHA256 = 'b7f75ee516a7aaac983b752d721f472ec9233d53e261643f4713cab112cadb88'
DISCOVERY_OUTCOME = Path('outputs/tables/outcome_model_integrity_discovery_trial_outcome_manifest.csv')
DISCOVERY_OUTCOME_SHA256 = 'a3cf2292d31749b96a0e3729bb5860d69bb642219ce6e478f4f794fb48c832ec'
DISCOVERY_MASTER = Path('outputs/tables/derivative_closure_corrected_discovery_master_v1.csv')
DISCOVERY_MASTER_SHA256 = '79aac098573fada34c2dd1247e7ce91e82633130f43cfdea82e2f59b41445039'
EXP_ASSIGN = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv')
EXP_ASSIGN_SHA256 = '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'
EXP_SIG = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv')
EXP_SIG_SHA256 = 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'
TR_ASSIGN = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv')
TR_ASSIGN_SHA256 = '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'
TR_SIG = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv')
TR_SIG_SHA256 = '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83'
OUT = Path('outputs/interpretability/discovery_semantic_report_execution_discovery_semantic_report_execution')
CACHE = Path('outputs/large/discovery_semantic_report_execution_discovery_semantic_report_execution_cache_v1')
NEXT_CONFIG = Path('analysis_specifications/S025_discovery_semantic_report_validation.json')
A1_PERMUTATIONS = 10000
MODEL_PERMUTATIONS = 1000
A2_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
B2_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ALPHA = 0.05

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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv.tmp', dir=path.parent, delete=False, encoding='utf-8') as f:
        tmp = Path(f.name)
        df.to_csv(f, index=False)
    os.replace(tmp, path)

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def stable_seed(*parts: Any) -> int:
    s = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(s).digest()[:8], 'big') % (2 ** 32 - 1)

def participant_weights(subjects: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects).astype(str)
    u, c = np.unique(subjects, return_counts=True)
    counts = dict(zip(u.tolist(), c.tolist()))
    return np.array([1.0 / counts[s] for s in subjects], dtype=float)

def deterministic_group_folds(subjects: Iterable[str], n_folds: int, salt: str) -> dict[str, int]:
    uniq = sorted(set(map(str, subjects)))
    if len(uniq) < n_folds:
        n_folds = len(uniq)
    ordered = sorted(uniq, key=lambda s: hashlib.sha256(f'{SEED}|{salt}|{s}'.encode()).hexdigest())
    return {s: i % n_folds for i, s in enumerate(ordered)}

def weighted_mean_std(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    w = w / np.sum(w)
    mu = np.sum(X * w[:, None], axis=0)
    var = np.sum((X - mu) ** 2 * w[:, None], axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (mu, sd)

def unweighted_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return (mu, sd)

def subject_balanced_accuracy(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    subjects = np.asarray(subjects).astype(str)
    vals = []
    for s in sorted(set(subjects.tolist())):
        idx = subjects == s
        ys, ps = (y[idx], pred[idx])
        recalls = []
        for cls in sorted(set(ys.tolist())):
            m = ys == cls
            recalls.append(float(np.mean(ps[m] == cls)))
        if recalls:
            vals.append(float(np.mean(recalls)))
    return float(np.mean(vals)) if vals else float('nan')

def class_balanced_participant_weights(subjects: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = participant_weights(subjects)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    total = float(np.sum(base))
    mass = {c: float(np.sum(base[y == c])) for c in classes}
    factors = {c: total / (len(classes) * mass[c]) for c in classes}
    w = np.array([bw * factors[yy] for bw, yy in zip(base, y)], dtype=float)
    return w

def permute_rows_within_subject(Y: np.ndarray, subjects: np.ndarray, perm_index: int, salt: str) -> np.ndarray:
    Y = np.asarray(Y)
    subjects = np.asarray(subjects).astype(str)
    out = np.empty_like(Y)
    for s in sorted(set(subjects.tolist())):
        idx = np.flatnonzero(subjects == s)
        rng = np.random.default_rng(stable_seed(PHASE, salt, perm_index, s))
        out[idx] = Y[idx[rng.permutation(len(idx))]]
    return out

def holm_table(ids: list[str], pvals: list[float], alpha: float=ALPHA) -> pd.DataFrame:
    m = len(ids)
    order = np.argsort(np.asarray(pvals, float), kind='mergesort')
    adj = np.empty(m, float)
    running = 0.0
    for rank, oi in enumerate(order):
        raw = float(pvals[oi])
        val = min(1.0, (m - rank) * raw)
        running = max(running, val)
        adj[oi] = running
    return pd.DataFrame({'test_id': ids, 'p_raw': pvals, 'p_holm': adj, 'reject_holm_0p05': adj <= alpha})

def weighted_contingency(categories: np.ndarray, families: np.ndarray, subjects: np.ndarray, category_levels: list[str], family_levels: list[int]) -> np.ndarray:
    w = participant_weights(subjects)
    tab = np.zeros((len(category_levels), len(family_levels)), dtype=float)
    ci = {c: i for i, c in enumerate(category_levels)}
    fi = {f: i for i, f in enumerate(family_levels)}
    for c, f, ww in zip(categories, families, w):
        if c in ci and int(f) in fi:
            tab[ci[c], fi[int(f)]] += float(ww)
    return tab

def contingency_metrics(tab: np.ndarray) -> dict[str, float]:
    tab = np.asarray(tab, float)
    total = float(tab.sum())
    if not np.isfinite(total) or total <= 0:
        return {'NMI': 0.0, 'Cramers_V': 0.0, 'H_family_given_category': float('nan'), 'H_category_given_family': float('nan'), 'H_category': float('nan'), 'H_family': float('nan'), 'norm_H_family_given_category': float('nan'), 'norm_H_category_given_family': float('nan')}
    p = tab / total
    pc = p.sum(axis=1)
    pf = p.sum(axis=0)
    nz = p > 0
    denom = pc[:, None] * pf[None, :]
    mi = float(np.sum(p[nz] * np.log(p[nz] / denom[nz])))
    hc = -float(np.sum(pc[pc > 0] * np.log(pc[pc > 0])))
    hf = -float(np.sum(pf[pf > 0] * np.log(pf[pf > 0])))
    nmi = mi / math.sqrt(hc * hf) if hc > 0 and hf > 0 else 0.0
    expected = np.outer(pc, pf) * total
    valid = expected > 0
    chi2 = float(np.sum((tab[valid] - expected[valid]) ** 2 / expected[valid]))
    d = min(tab.shape[0] - 1, tab.shape[1] - 1)
    v = math.sqrt(chi2 / (total * d)) if d > 0 and total > 0 else 0.0
    hfgc = 0.0
    for i, pcat in enumerate(pc):
        if pcat <= 0:
            continue
        q = p[i] / pcat
        hfgc += pcat * -float(np.sum(q[q > 0] * np.log(q[q > 0])))
    hcgf = 0.0
    for j, pfam in enumerate(pf):
        if pfam <= 0:
            continue
        q = p[:, j] / pfam
        hcgf += pfam * -float(np.sum(q[q > 0] * np.log(q[q > 0])))
    return {'NMI': nmi, 'Cramers_V': v, 'H_family_given_category': hfgc, 'H_category_given_family': hcgf, 'H_category': hc, 'H_family': hf, 'norm_H_family_given_category': hfgc / hf if hf > 0 else float('nan'), 'norm_H_category_given_family': hcgf / hc if hc > 0 else float('nan')}

def a1_observed_and_mixing(df: pd.DataFrame, k: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cats = df['emotion_category'].astype(str).to_numpy()
    fam = df['family_1based'].astype(int).to_numpy()
    subj = df['subject_id'].astype(str).to_numpy()
    family_levels = list(range(1, k + 1))
    tab = weighted_contingency(cats, fam, subj, CATEGORIES, family_levels)
    metrics = contingency_metrics(tab)
    metrics.update({'test_id': f'A1_K{k}', 'k': k, 'trials': int(len(df)), 'participants': int(df.subject_id.nunique())})
    total = tab.sum()
    pc = tab.sum(axis=1)
    pf = tab.sum(axis=0)
    cell_rows = []
    for i, c in enumerate(CATEGORIES):
        for j, f in enumerate(family_levels):
            mass = float(tab[i, j])
            cell_rows.append({'k': k, 'emotion_category': c, 'family_1based': f, 'participant_weighted_mass': mass, 'p_family_given_category': mass / pc[i] if pc[i] > 0 else np.nan, 'p_category_given_family': mass / pf[j] if pf[j] > 0 else np.nan})
    conc_rows = []
    for i, c in enumerate(CATEGORIES):
        conc_rows.append({'k': k, 'axis': 'category', 'level': c, 'maximum_concentration': float(np.max(tab[i] / pc[i])) if pc[i] > 0 else np.nan})
    for j, f in enumerate(family_levels):
        conc_rows.append({'k': k, 'axis': 'family', 'level': str(f), 'maximum_concentration': float(np.max(tab[:, j] / pf[j])) if pf[j] > 0 else np.nan})
    return (metrics, pd.DataFrame(cell_rows), pd.DataFrame(conc_rows))

def run_a1_permutations(df: pd.DataFrame, k: int, n_perm: int) -> pd.DataFrame:
    cats = df['emotion_category'].astype(str).to_numpy()
    fam = df['family_1based'].astype(int).to_numpy()
    subj = df['subject_id'].astype(str).to_numpy()
    rows = []
    for p in range(n_perm):
        fp = permute_rows_within_subject(fam[:, None], subj, p, 'A1_family').ravel()
        tab = weighted_contingency(cats, fp, subj, CATEGORIES, list(range(1, k + 1)))
        met = contingency_metrics(tab)
        rows.append({'permutation': p, 'NMI': met['NMI'], 'Cramers_V': met['Cramers_V']})
    return pd.DataFrame(rows)

def _fit_logit(X_train: np.ndarray, y_train: np.ndarray, subj_train: np.ndarray, X_test: np.ndarray, C: float) -> np.ndarray:
    mu, sd = unweighted_mean_std(X_train)
    xtr = (X_train - mu) / sd
    xte = (X_test - mu) / sd
    w = class_balanced_participant_weights(subj_train, y_train)
    clf = LogisticRegression(C=float(C), penalty='l2', solver='lbfgs', fit_intercept=True, max_iter=4000, tol=1e-08, random_state=SEED)
    clf.fit(xtr, y_train, sample_weight=w)
    return clf.predict(xte)

def select_logit_C(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, C_grid: list[float], salt: str) -> tuple[float, pd.DataFrame]:
    fold_map = deterministic_group_folds(subjects, 5, salt)
    folds = np.array([fold_map[str(s)] for s in subjects], dtype=int)
    rows = []
    for C in C_grid:
        pred = np.empty_like(y)
        valid = np.zeros(len(y), dtype=bool)
        for fold in sorted(set(folds.tolist())):
            te = folds == fold
            tr = ~te
            if len(set(y[tr].tolist())) < 2:
                continue
            pred[te] = _fit_logit(X[tr], y[tr], subjects[tr], X[te], C)
            valid[te] = True
        score = subject_balanced_accuracy(y[valid], pred[valid], subjects[valid]) if valid.any() else -np.inf
        rows.append({'C': float(C), 'participant_balanced_BA': float(score)})
    scores = pd.DataFrame(rows)
    best_score = scores['participant_balanced_BA'].max()
    best = float(scores.loc[np.isclose(scores.participant_balanced_BA, best_score), 'C'].min())
    return (best, scores)

@dataclass
class LogisticCVResult:
    statistic: float
    baseline_ba: float
    full_ba: float
    fold_rows: list[dict[str, Any]]

def nested_lopo_logistic(df: pd.DataFrame, y_col: str, report_columns: list[str], salt: str) -> LogisticCVResult:
    subjects = df['subject_id'].astype(str).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    nuisance = np.column_stack([(df['context'].astype(str).to_numpy() == 'video').astype(float), df['acquisition_fraction'].astype(float).to_numpy(), np.square(df['acquisition_fraction'].astype(float).to_numpy())])
    reports = df[report_columns].to_numpy(float) if report_columns else np.empty((len(df), 0))
    full = np.column_stack([nuisance, reports])
    pred0 = np.empty_like(y)
    pred1 = np.empty_like(y)
    fold_rows = []
    for outer_subject in sorted(set(subjects.tolist())):
        te = subjects == outer_subject
        tr = ~te
        outcome_free_features, _ = select_logit_C(nuisance[tr], y[tr], subjects[tr], A2_C_GRID, f'{salt}|outer={outer_subject}|baseline')
        outcome_model_integrity, _ = select_logit_C(full[tr], y[tr], subjects[tr], A2_C_GRID, f'{salt}|outer={outer_subject}|full')
        pred0[te] = _fit_logit(nuisance[tr], y[tr], subjects[tr], nuisance[te], outcome_free_features)
        pred1[te] = _fit_logit(full[tr], y[tr], subjects[tr], full[te], outcome_model_integrity)
        fold_rows.append({'outer_subject': outer_subject, 'n_test_trials': int(te.sum()), 'selected_C_baseline': outcome_free_features, 'selected_C_full': outcome_model_integrity, 'baseline_BA_subject': subject_balanced_accuracy(y[te], pred0[te], subjects[te]), 'full_BA_subject': subject_balanced_accuracy(y[te], pred1[te], subjects[te])})
    b0 = subject_balanced_accuracy(y, pred0, subjects)
    b1 = subject_balanced_accuracy(y, pred1, subjects)
    return LogisticCVResult(float(b1 - b0), float(b0), float(b1), fold_rows)

def nested_lopo_logistic_full_ba(df: pd.DataFrame, y_col: str, report_columns: list[str], salt: str) -> float:
    subjects = df['subject_id'].astype(str).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    nuisance = np.column_stack([(df['context'].astype(str).to_numpy() == 'video').astype(float), df['acquisition_fraction'].astype(float).to_numpy(), np.square(df['acquisition_fraction'].astype(float).to_numpy())])
    reports = df[report_columns].to_numpy(float) if report_columns else np.empty((len(df), 0))
    full = np.column_stack([nuisance, reports])
    pred = np.empty_like(y)
    for outer_subject in sorted(set(subjects.tolist())):
        te = subjects == outer_subject
        tr = ~te
        outcome_model_integrity, _ = select_logit_C(full[tr], y[tr], subjects[tr], A2_C_GRID, f'{salt}|outer={outer_subject}|full')
        pred[te] = _fit_logit(full[tr], y[tr], subjects[tr], full[te], outcome_model_integrity)
    return subject_balanced_accuracy(y, pred, subjects)

def permutation_logistic_stat(df: pd.DataFrame, y_col: str, report_columns: list[str], perm_index: int, salt: str, baseline_ba: float) -> float:
    work = df.copy()
    Y = work[REPORTS].to_numpy(float)
    subjects = work.subject_id.astype(str).to_numpy()
    perm_salt = 'A2_reports' if salt.startswith('A2_') else 'B3_reports' if salt.startswith('B3_') else f'{salt}|reports'
    Yp = permute_rows_within_subject(Y, subjects, perm_index, perm_salt)
    pos = {name: j for j, name in enumerate(REPORTS)}
    for c in report_columns:
        work[c] = Yp[:, pos[c]]
    return nested_lopo_logistic_full_ba(work, y_col, report_columns, salt=salt) - baseline_ba

def _fit_ridge_predict(Xtr: np.ndarray, Ytr: np.ndarray, Str: np.ndarray, Xte: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = participant_weights(Str)
    xmu, xsd = weighted_mean_std(Xtr, w)
    ymu, ysd = weighted_mean_std(Ytr, w)
    xtr = (Xtr - xmu) / xsd
    xte = (Xte - xmu) / xsd
    ytr = (Ytr - ymu) / ysd
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver='cholesky')
    model.fit(xtr, ytr, sample_weight=w)
    pred = model.predict(xte)
    return (pred, ymu, ysd)

def subject_multivariate_mse(Ystd: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    subjects = np.asarray(subjects).astype(str)
    for s in sorted(set(subjects.tolist())):
        idx = subjects == s
        vals.append(float(np.mean((Ystd[idx] - pred[idx]) ** 2)))
    return float(np.mean(vals))

def select_ridge_alpha(X: np.ndarray, Y: np.ndarray, subjects: np.ndarray, alpha_grid: list[float], salt: str) -> tuple[float, pd.DataFrame]:
    fold_map = deterministic_group_folds(subjects, 5, salt)
    folds = np.array([fold_map[str(s)] for s in subjects], dtype=int)
    rows = []
    for alpha in alpha_grid:
        fold_subject_mse = []
        for fold in sorted(set(folds.tolist())):
            te = folds == fold
            tr = ~te
            pred, ymu, ysd = _fit_ridge_predict(X[tr], Y[tr], subjects[tr], X[te], alpha)
            yte_std = (Y[te] - ymu) / ysd
            te_subjects = subjects[te]
            for s in sorted(set(te_subjects.tolist())):
                m = te_subjects == s
                fold_subject_mse.append(float(np.mean((yte_std[m] - pred[m]) ** 2)))
        score = float(np.mean(fold_subject_mse))
        rows.append({'alpha': float(alpha), 'participant_balanced_multivariate_MSE': score})
    scores = pd.DataFrame(rows)
    best_score = scores.participant_balanced_multivariate_MSE.min()
    best = float(scores.loc[np.isclose(scores.participant_balanced_multivariate_MSE, best_score), 'alpha'].max())
    return (best, scores)

@dataclass
class RidgeCVResult:
    primary_delta_r2: float
    dimension_delta_r2: np.ndarray
    m0_r2: float
    m1_r2: float
    m2_r2: float
    fold_rows: list[dict[str, Any]]

def nested_lopo_ridge_b2(df: pd.DataFrame) -> RidgeCVResult:
    subjects = df.subject_id.astype(str).to_numpy()
    Y = df[REPORTS].to_numpy(float)
    nuisance = np.column_stack([(df['context'].astype(str).to_numpy() == 'video').astype(float), df['acquisition_fraction'].astype(float).to_numpy(), np.square(df['acquisition_fraction'].astype(float).to_numpy())])
    xe = df[[f'exp__{c}' for c in STD_SIG]].to_numpy(float)
    xt = df[[f'tr__{c}' for c in STD_SIG]].to_numpy(float)
    Xs = [nuisance, np.column_stack([nuisance, xe]), np.column_stack([nuisance, xe, xt])]
    per_subject = []
    fold_rows = []
    for outer_subject in sorted(set(subjects.tolist())):
        te = subjects == outer_subject
        tr = ~te
        r2_models = []
        dim_models = []
        selected = []
        for mi, X in enumerate(Xs):
            alpha, _ = select_ridge_alpha(X[tr], Y[tr], subjects[tr], B2_ALPHA_GRID, f'B2|outer={outer_subject}|M{mi}')
            pred, ymu, ysd = _fit_ridge_predict(X[tr], Y[tr], subjects[tr], X[te], alpha)
            ystd = (Y[te] - ymu) / ysd
            sse = float(np.sum((ystd - pred) ** 2))
            sse0 = float(np.sum(ystd ** 2))
            r2 = 1.0 - sse / sse0 if sse0 > 0 else np.nan
            dim_sse = np.sum((ystd - pred) ** 2, axis=0)
            dim_sse0 = np.sum(ystd ** 2, axis=0)
            dim_r2 = np.where(dim_sse0 > 0, 1.0 - dim_sse / dim_sse0, np.nan)
            r2_models.append(float(r2))
            dim_models.append(dim_r2)
            selected.append(float(alpha))
        delta = r2_models[2] - r2_models[1]
        dim_delta = dim_models[2] - dim_models[1]
        per_subject.append((outer_subject, r2_models, dim_delta))
        fold_rows.append({'outer_subject': outer_subject, 'n_test_trials': int(te.sum()), 'alpha_M0': selected[0], 'alpha_M1': selected[1], 'alpha_M2': selected[2], 'R2_M0': r2_models[0], 'R2_M1': r2_models[1], 'R2_M2': r2_models[2], 'Delta_R2_M2_minus_M1': delta, **{f'Delta_R2_{REPORTS[j]}': float(dim_delta[j]) for j in range(len(REPORTS))}})
    m0 = float(np.nanmean([x[1][0] for x in per_subject]))
    m1 = float(np.nanmean([x[1][1] for x in per_subject]))
    m2 = float(np.nanmean([x[1][2] for x in per_subject]))
    dims = np.nanmean(np.vstack([x[2] for x in per_subject]), axis=0)
    return RidgeCVResult(float(m2 - m1), dims, m0, m1, m2, fold_rows)

def permutation_b2(df: pd.DataFrame, perm_index: int) -> RidgeCVResult:
    work = df.copy()
    subjects = work.subject_id.astype(str).to_numpy()
    Yp = permute_rows_within_subject(work[REPORTS].to_numpy(float), subjects, perm_index, 'B2_reports')
    work.loc[:, REPORTS] = Yp
    return nested_lopo_ridge_b2(work)

def verify_semantic_report_execution_design(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verify(project / semantic_report_execution_design_LOCK, semantic_report_execution_design_LOCK_SHA256, 'semantic_report_execution_design final lock')
    lock = readj(project / semantic_report_execution_design_LOCK)
    if lock.get('phase') != 'semantic_report_execution_design' or lock.get('next_phase') != PHASE:
        raise RuntimeError('semantic_report_execution_design authorization drift')
    if lock.get('experience_K') != EXPERIENCE_K or lock.get('report_transition_K') != TRANSITION_K:
        raise RuntimeError('semantic_report_execution_design structural K drift')
    for key in ['new_EEG_signal_values_opened', 'behavioral_report_values_opened', 'emotion_category_values_opened', 'context_target_values_opened', 'canonical_context_category_use_permitted', 'semantic_relabeling_permitted']:
        if bool(lock.get(key)):
            raise RuntimeError(f'semantic_report_execution_design boundary drift: {key}=True')
    for rel, expected in lock.get('hashes', {}).items():
        verify(project / rel, expected, f'semantic_report_execution_design fixed child {rel}')
    verify(project / semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256, 'semantic_report_execution_design program policy')
    verify(project / semantic_report_execution_design_NEXT, semantic_report_execution_design_NEXT_SHA256, 'semantic_report_execution_design→discovery_semantic_report_execution policy')
    program = readj(project / semantic_report_execution_design_PROGRAM)
    nxt = readj(project / semantic_report_execution_design_NEXT)
    if nxt.get('Discovery_behavior_opening_permitted') is not True:
        raise RuntimeError('semantic_report_execution_design does not permit Discovery behavior opening')
    if nxt.get('Validation_behavior_opening_permitted') is not False:
        raise RuntimeError('semantic_report_execution_design unexpectedly permits Validation behavior')
    if nxt.get('Confirmation_behavior_opening_permitted') is not False:
        raise RuntimeError('semantic_report_execution_design unexpectedly permits Confirmation behavior')
    if nxt.get('new_EEG_signal_opening_permitted') is not False:
        raise RuntimeError('semantic_report_execution_design unexpectedly permits new EEG')
    if nxt.get('tests') != ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']:
        raise RuntimeError('semantic_report_execution_design test registry drift')
    return (lock, program, nxt)

def load_discovery(project: Path) -> dict[str, pd.DataFrame]:
    refs = [(DISCOVERY_OUTCOME, DISCOVERY_OUTCOME_SHA256, 'Discovery outcome manifest'), (DISCOVERY_MASTER, DISCOVERY_MASTER_SHA256, 'Discovery corrected master'), (EXP_ASSIGN, EXP_ASSIGN_SHA256, 'Discovery experience assignments'), (EXP_SIG, EXP_SIG_SHA256, 'Discovery experience signatures'), (TR_ASSIGN, TR_ASSIGN_SHA256, 'Discovery transition assignments'), (TR_SIG, TR_SIG_SHA256, 'Discovery transition signatures')]
    for rel, h, label in refs:
        verify(project / rel, h, label)
    outcomes = pd.read_csv(project / DISCOVERY_OUTCOME, low_memory=False)
    master = pd.read_csv(project / DISCOVERY_MASTER, low_memory=False)
    exp_assign = pd.read_csv(project / EXP_ASSIGN, low_memory=False)
    exp_sig = pd.read_csv(project / EXP_SIG, low_memory=False)
    tr_assign = pd.read_csv(project / TR_ASSIGN, low_memory=False)
    tr_sig = pd.read_csv(project / TR_SIG, low_memory=False)
    for df in [outcomes, master, exp_assign, exp_sig, tr_assign, tr_sig]:
        df['raw_behavior_row_id'] = df['raw_behavior_row_id'].astype(str)
        df['subject_id'] = df['subject_id'].astype(str)
    if outcomes.duplicated(['raw_behavior_row_id', 'subject_id']).any():
        raise RuntimeError('Discovery outcome manifest key duplication')
    if master.duplicated(['raw_behavior_row_id', 'subject_id']).any():
        raise RuntimeError('Discovery corrected-master key duplication')
    if exp_sig.duplicated(['raw_behavior_row_id', 'subject_id']).any():
        raise RuntimeError('Experience signature key duplication')
    if tr_sig.duplicated(['raw_behavior_row_id', 'subject_id']).any():
        raise RuntimeError('Transition signature key duplication')
    needed_out = ['raw_behavior_row_id', 'subject_id', 'trial_number', 'acquisition_fraction'] + REPORTS
    missing = set(needed_out) - set(outcomes.columns)
    if missing:
        raise RuntimeError(f'Discovery outcome columns missing: {sorted(missing)}')
    needed_master = ['raw_behavior_row_id', 'subject_id', 'context', 'emotion_category', 'context_authoritative_for_analysis', 'stimulus_authoritative_for_analysis']
    missing = set(needed_master) - set(master.columns)
    if missing:
        raise RuntimeError(f'Discovery corrected-master columns missing: {sorted(missing)}')
    for c in STD_SIG:
        if c not in exp_sig.columns or c not in tr_sig.columns:
            raise RuntimeError(f'Missing prespecified signature coordinate: {c}')
    targets = outcomes[needed_out].merge(master[needed_master], on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    return {'targets': targets, 'exp_assign': exp_assign, 'exp_sig': exp_sig, 'tr_assign': tr_assign, 'tr_sig': tr_sig}

def finite_rows(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].to_numpy(float)
    return np.isfinite(arr).all(axis=1)

def prepare_a1(data: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    a = data['exp_assign']
    a = a[a['k'].astype(int).eq(k)][['raw_behavior_row_id', 'subject_id', 'family_1based']].copy()
    d = a.merge(data['targets'], on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    mask = d['stimulus_authoritative_for_analysis'].map(bval)
    mask &= d['emotion_category'].astype(str).isin(CATEGORIES)
    d = d[mask].copy()
    if d.subject_id.nunique() < 20 or len(d) < 500:
        raise RuntimeError(f'A1 K={k} unexpectedly small cohort')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def prepare_logistic(data: dict[str, pd.DataFrame], role: str, k: int) -> pd.DataFrame:
    if role == 'experience':
        a = data['exp_assign']
    elif role == 'transition':
        a = data['tr_assign']
    else:
        raise ValueError(role)
    a = a[a['k'].astype(int).eq(k)][['raw_behavior_row_id', 'subject_id', 'family_1based']].copy()
    d = a.merge(data['targets'], on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    mask = d['context_authoritative_for_analysis'].map(bval)
    mask &= d['context'].astype(str).isin(['video', 'imagery'])
    mask &= np.isfinite(pd.to_numeric(d['acquisition_fraction'], errors='coerce'))
    mask &= finite_rows(d, REPORTS)
    d = d[mask].copy()
    if d.subject_id.nunique() < 20 or len(d) < 500:
        raise RuntimeError(f'{role} K={k} logistic cohort unexpectedly small')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def prepare_b2(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    es = data['exp_sig'][['raw_behavior_row_id', 'subject_id'] + STD_SIG].copy()
    ts = data['tr_sig'][['raw_behavior_row_id', 'subject_id'] + STD_SIG].copy()
    es = es.rename(columns={c: f'exp__{c}' for c in STD_SIG})
    ts = ts.rename(columns={c: f'tr__{c}' for c in STD_SIG})
    d = es.merge(ts, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
    d = d.merge(data['targets'], on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    mask = d['context_authoritative_for_analysis'].map(bval)
    mask &= d['context'].astype(str).isin(['video', 'imagery'])
    mask &= np.isfinite(pd.to_numeric(d['acquisition_fraction'], errors='coerce'))
    mask &= finite_rows(d, REPORTS)
    mask &= finite_rows(d, [f'exp__{c}' for c in STD_SIG])
    mask &= finite_rows(d, [f'tr__{c}' for c in STD_SIG])
    d = d[mask].copy()
    if d.subject_id.nunique() < 20 or len(d) < 500:
        raise RuntimeError('B2 cohort unexpectedly small')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def ensure_cache(cache: Path, meta: dict[str, Any]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    mp = cache / 'cache_meta.json'
    if mp.is_file():
        old = readj(mp)
        if old != meta:
            raise RuntimeError(f'discovery_semantic_report_execution cache metadata mismatch at {cache}. Delete this discovery_semantic_report_execution cache directory before rerunning changed code/input.')
    else:
        writej(mp, meta)

def append_checkpoint(path: Path, rows: list[dict[str, Any]], sort_by: str='permutation') -> None:
    df = pd.DataFrame(rows)
    if len(df) and sort_by in df.columns:
        df = df.sort_values(sort_by).drop_duplicates(sort_by, keep='last')
    atomic_csv(path, df)

def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict('records') if path.is_file() else []

def permutation_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))

def parallel_map(func, indices: list[int], workers: int) -> list[Any]:
    if workers <= 1:
        return [func(i) for i in indices]
    return Parallel(n_jobs=workers, backend='threading', batch_size=1)((delayed(func)(i) for i in indices))

def single_report_permutation_row(df: pd.DataFrame, y_col: str, perm_index: int, model_salt: str, permutation_salt: str, baseline_ba: float) -> dict[str, Any]:
    work = df.copy()
    subjects = work.subject_id.astype(str).to_numpy()
    Yp = permute_rows_within_subject(work[REPORTS].to_numpy(float), subjects, perm_index, permutation_salt)
    work.loc[:, REPORTS] = Yp
    row: dict[str, Any] = {'permutation': int(perm_index)}
    for report in REPORTS:
        row[report] = nested_lopo_logistic_full_ba(work, y_col, [report], salt=model_salt) - baseline_ba
    return row

def self_test() -> int:
    assert EXPERIENCE_K == [2, 3, 4]
    assert TRANSITION_K == [2, 3]
    assert len(REPORTS) == 10
    assert len(STD_SIG) == 14
    t = holm_table(['a', 'b', 'c'], [0.01, 0.03, 0.2])
    assert np.allclose(t.p_holm, [0.03, 0.06, 0.2])
    x = np.array([[10], [11], [20], [21]])
    s = np.array(['a', 'a', 'b', 'b'])
    p = permute_rows_within_subject(x, s, 0, 'selftest')
    assert set(p[:2, 0]) == {10, 11} and set(p[2:, 0]) == {20, 21}
    tab = np.array([[1.0, 0.0], [0.0, 1.0]])
    met = contingency_metrics(tab)
    assert met['NMI'] > 0.99 and met['Cramers_V'] > 0.99
    print('discovery_semantic_report_execution self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, int(os.environ.get('discovery_semantic_report_execution_WORKERS', '1'))), help='parallel permutation workers; deterministic results are independent of worker count')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    project = args.project.resolve()
    workers = max(1, int(args.workers))
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    print(f'[{now()}] discovery_semantic_report_execution permutation workers: {workers}', flush=True)
    print(f'[{now()}] verifying semantic_report_execution_design analysis specification and Discovery-only authorization', flush=True)
    _, program, nxt = verify_semantic_report_execution_design(project)
    print(f'[{now()}] opening only prespecified Discovery target sources and immutable structural outputs', flush=True)
    data = load_discovery(project)
    out = project / OUT
    final_lock = out / 'qc/discovery_semantic_report_execution_final_output_lock_hashes.json'
    if final_lock.is_file() and (not args.force):
        raise RuntimeError(f'Completed discovery_semantic_report_execution output already exists: {final_lock}')
    if out.exists() and args.force:
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    script_sha = sha256(Path(__file__).resolve())
    cache = project / CACHE
    cache_meta = {'phase': PHASE, 'source_code_sha256': script_sha, 'semantic_report_execution_design_final_lock_sha256': semantic_report_execution_design_LOCK_SHA256, 'semantic_report_execution_design_program_sha256': semantic_report_execution_design_PROGRAM_SHA256, 'Discovery_outcome_sha256': DISCOVERY_OUTCOME_SHA256, 'Discovery_master_sha256': DISCOVERY_MASTER_SHA256, 'experience_assign_sha256': EXP_ASSIGN_SHA256, 'experience_signature_sha256': EXP_SIG_SHA256, 'transition_assign_sha256': TR_ASSIGN_SHA256, 'transition_signature_sha256': TR_SIG_SHA256, 'A1_permutations': A1_PERMUTATIONS, 'model_permutations': MODEL_PERMUTATIONS}
    ensure_cache(cache, cache_meta)
    print(f'[{now()}] A1 category ↔ experience-family mapping/mixing', flush=True)
    a1_metrics_rows = []
    a1_cells = []
    a1_conc = []
    a1_perm_summaries = []
    a1_perm_values = []
    for k in EXPERIENCE_K:
        df = prepare_a1(data, k)
        obs, cells, conc = a1_observed_and_mixing(df, k)
        a1_cells.append(cells)
        a1_conc.append(conc)
        cp = cache / f'A1_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        for p in range(A1_PERMUTATIONS):
            if p in done:
                continue
            fp = permute_rows_within_subject(df['family_1based'].astype(int).to_numpy()[:, None], df.subject_id.astype(str).to_numpy(), p, 'A1_family').ravel()
            tab = weighted_contingency(df.emotion_category.astype(str).to_numpy(), fp, df.subject_id.astype(str).to_numpy(), CATEGORIES, list(range(1, k + 1)))
            m = contingency_metrics(tab)
            rows.append({'permutation': p, 'NMI': m['NMI'], 'Cramers_V': m['Cramers_V']})
            if (p + 1) % 100 == 0:
                append_checkpoint(cp, rows)
                print(f'  A1 K={k}: {p + 1}/{A1_PERMUTATIONS}', flush=True)
        append_checkpoint(cp, rows)
        perm = pd.DataFrame(rows).sort_values('permutation')
        perm_export = perm.copy()
        perm_export['k'] = k
        a1_perm_values.append(perm_export)
        p_nmi = permutation_p(obs['NMI'], perm['NMI'].to_numpy(float))
        p_v = permutation_p(obs['Cramers_V'], perm['Cramers_V'].to_numpy(float))
        parent_p = min(1.0, 2.0 * min(p_nmi, p_v))
        obs.update({'p_NMI': p_nmi, 'p_Cramers_V': p_v, 'parent_p_raw': parent_p})
        a1_metrics_rows.append(obs)
        a1_perm_summaries.append({'test_id': f'A1_K{k}', 'permutations': A1_PERMUTATIONS, 'null_NMI_mean': float(perm.NMI.mean()), 'null_NMI_sd': float(perm.NMI.std(ddof=1)), 'null_V_mean': float(perm.Cramers_V.mean()), 'null_V_sd': float(perm.Cramers_V.std(ddof=1))})
    a1_metrics = pd.DataFrame(a1_metrics_rows)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A1_parent_metrics.csv', a1_metrics)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A1_mixing_cells.csv', pd.concat(a1_cells, ignore_index=True))
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A1_concentration.csv', pd.concat(a1_conc, ignore_index=True))
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A1_null_summary.csv', pd.DataFrame(a1_perm_summaries))
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A1_permutation_values.csv', pd.concat(a1_perm_values, ignore_index=True).sort_values(['k', 'permutation']))
    print(f'[{now()}] A2 joint 10-report prediction of experience family', flush=True)
    a2_rows = []
    a2_fold_tables = []
    a2_perm_values = []
    for k in EXPERIENCE_K:
        df = prepare_logistic(data, 'experience', k)
        obs = nested_lopo_logistic(df, 'family_1based', REPORTS, salt=f'A2_K{k}')
        a2_fold = pd.DataFrame(obs.fold_rows)
        a2_fold['k'] = k
        a2_fold_tables.append(a2_fold)
        cp = cache / f'A2_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        batch_size = max(1, workers * 2)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            def _a2_job(pp: int) -> dict[str, Any]:
                return {'permutation': int(pp), 'Delta_BA': permutation_logistic_stat(df, 'family_1based', REPORTS, pp, f'A2_K{k}', obs.baseline_ba)}
            rows.extend(parallel_map(_a2_job, batch, workers))
            append_checkpoint(cp, rows)
            completed = len({int(r['permutation']) for r in rows})
            print(f'  A2 K={k}: {completed}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        perm_export = perm.copy()
        perm_export['k'] = k
        a2_perm_values.append(perm_export)
        p_raw = permutation_p(obs.statistic, perm.Delta_BA.to_numpy(float))
        a2_rows.append({'test_id': f'A2_K{k}', 'k': k, 'trials': len(df), 'participants': df.subject_id.nunique(), 'baseline_BA': obs.baseline_ba, 'full_10report_BA': obs.full_ba, 'Delta_BA': obs.statistic, 'parent_p_raw': p_raw, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(perm.Delta_BA.mean()), 'null_sd': float(perm.Delta_BA.std(ddof=1))})
    a2_metrics = pd.DataFrame(a2_rows)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A2_parent_metrics.csv', a2_metrics)
    atomic_csv(out / 'qc/discovery_semantic_report_execution_A2_outer_fold_audit.csv', pd.concat(a2_fold_tables, ignore_index=True))
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A2_permutation_values.csv', pd.concat(a2_perm_values, ignore_index=True).sort_values(['k', 'permutation']))
    parent_ids, parent_p = ([], [])
    for k in EXPERIENCE_K:
        parent_ids.append(f'A1_K{k}')
        parent_p.append(float(a1_metrics.loc[a1_metrics.k.eq(k), 'parent_p_raw'].iloc[0]))
    for k in EXPERIENCE_K:
        parent_ids.append(f'A2_K{k}')
        parent_p.append(float(a2_metrics.loc[a2_metrics.k.eq(k), 'parent_p_raw'].iloc[0]))
    a_holm = holm_table(parent_ids, parent_p)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A_family_holm.csv', a_holm)
    a2_child_rows = []
    a2_child_perm_values = []
    for k in EXPERIENCE_K:
        parent = a_holm[a_holm.test_id.eq(f'A2_K{k}')].iloc[0]
        if not bool(parent.reject_holm_0p05):
            for report in REPORTS:
                a2_child_rows.append({'k': k, 'report': report, 'status': 'NOT_RUN_PARENT_GATE', 'Delta_BA': np.nan, 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
            continue
        df = prepare_logistic(data, 'experience', k)
        baseline_ba = float(a2_metrics.loc[a2_metrics.k.eq(k), 'baseline_BA'].iloc[0])
        child_obs = {}
        for report in REPORTS:
            child_obs[report] = nested_lopo_logistic_full_ba(df, 'family_1based', [report], salt=f'A2_K{k}') - baseline_ba
        cp = cache / f'A2_K{k}_single_report_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        batch_size = max(1, workers)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            def _a2_child_job(pp: int) -> dict[str, Any]:
                return single_report_permutation_row(df, 'family_1based', pp, f'A2_K{k}', 'A2_reports', baseline_ba)
            rows.extend(parallel_map(_a2_child_job, batch, workers))
            append_checkpoint(cp, rows)
            completed = len({int(r['permutation']) for r in rows})
            print(f'  A2 K={k} child reports: {completed}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        perm_export = perm.copy()
        perm_export['k'] = k
        a2_child_perm_values.append(perm_export)
        rawp = [permutation_p(child_obs[r], perm[r].to_numpy(float)) for r in REPORTS]
        h = holm_table(REPORTS, rawp)
        for report, p0 in zip(REPORTS, rawp):
            hr = h[h.test_id.eq(report)].iloc[0]
            a2_child_rows.append({'k': k, 'report': report, 'status': 'RUN_PARENT_PASSED', 'Delta_BA': child_obs[report], 'p_raw': p0, 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A2_single_report_followups.csv', pd.DataFrame(a2_child_rows))
    if a2_child_perm_values:
        a2_child_perm = pd.concat(a2_child_perm_values, ignore_index=True).sort_values(['k', 'permutation'])
    else:
        a2_child_perm = pd.DataFrame(columns=['permutation'] + REPORTS + ['k'])
    atomic_csv(out / 'tables/discovery_semantic_report_execution_A2_single_report_permutation_values.csv', a2_child_perm)
    print(f'[{now()}] B2 incremental report information from report-transition signature', flush=True)
    b2df = prepare_b2(data)
    b2obs = nested_lopo_ridge_b2(b2df)
    atomic_csv(out / 'qc/discovery_semantic_report_execution_B2_outer_fold_audit.csv', pd.DataFrame(b2obs.fold_rows))
    cp = cache / 'B2_permutations.csv'
    rows = load_checkpoint(cp)
    done = {int(r['permutation']) for r in rows}
    pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
    batch_size = max(1, workers * 2)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]

        def _b2_job(pp: int) -> dict[str, Any]:
            rr = permutation_b2(b2df, pp)
            row = {'permutation': int(pp), 'Delta_R2': rr.primary_delta_r2}
            for j, report in enumerate(REPORTS):
                row[f'Delta_R2_{report}'] = float(rr.dimension_delta_r2[j])
            return row
        rows.extend(parallel_map(_b2_job, batch, workers))
        append_checkpoint(cp, rows)
        completed = len({int(r['permutation']) for r in rows})
        print(f'  B2: {completed}/{MODEL_PERMUTATIONS}', flush=True)
    b2perm = pd.DataFrame(rows).sort_values('permutation')
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B2_permutation_values.csv', b2perm)
    b2p = permutation_p(b2obs.primary_delta_r2, b2perm.Delta_R2.to_numpy(float))
    b2parent = pd.DataFrame([{'test_id': 'B2', 'trials': len(b2df), 'participants': b2df.subject_id.nunique(), 'R2_M0': b2obs.m0_r2, 'R2_M1': b2obs.m1_r2, 'R2_M2': b2obs.m2_r2, 'Delta_R2_M2_minus_M1': b2obs.primary_delta_r2, 'p_raw': b2p, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(b2perm.Delta_R2.mean()), 'null_sd': float(b2perm.Delta_R2.std(ddof=1))}])
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B2_parent_metric.csv', b2parent)
    b2_children = []
    if b2p < ALPHA:
        rawp = [permutation_p(float(b2obs.dimension_delta_r2[j]), b2perm[f'Delta_R2_{r}'].to_numpy(float)) for j, r in enumerate(REPORTS)]
        h = holm_table(REPORTS, rawp)
        for j, r in enumerate(REPORTS):
            hr = h[h.test_id.eq(r)].iloc[0]
            b2_children.append({'report': r, 'status': 'RUN_PARENT_PASSED', 'Delta_R2': float(b2obs.dimension_delta_r2[j]), 'p_raw': rawp[j], 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
    else:
        for j, r in enumerate(REPORTS):
            b2_children.append({'report': r, 'status': 'NOT_INTERPRETED_PARENT_GATE', 'Delta_R2': float(b2obs.dimension_delta_r2[j]), 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B2_dimension_followups.csv', pd.DataFrame(b2_children))
    print(f'[{now()}] B3 joint 10-report prediction of report-transition family', flush=True)
    b3_rows = []
    b3_fold_tables = []
    b3_perm_values = []
    for k in TRANSITION_K:
        df = prepare_logistic(data, 'transition', k)
        obs = nested_lopo_logistic(df, 'family_1based', REPORTS, salt=f'B3_K{k}')
        ff = pd.DataFrame(obs.fold_rows)
        ff['k'] = k
        b3_fold_tables.append(ff)
        cp = cache / f'B3_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        batch_size = max(1, workers * 2)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            def _b3_job(pp: int) -> dict[str, Any]:
                return {'permutation': int(pp), 'Delta_BA': permutation_logistic_stat(df, 'family_1based', REPORTS, pp, f'B3_K{k}', obs.baseline_ba)}
            rows.extend(parallel_map(_b3_job, batch, workers))
            append_checkpoint(cp, rows)
            completed = len({int(r['permutation']) for r in rows})
            print(f'  B3 K={k}: {completed}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        perm_export = perm.copy()
        perm_export['k'] = k
        b3_perm_values.append(perm_export)
        p_raw = permutation_p(obs.statistic, perm.Delta_BA.to_numpy(float))
        b3_rows.append({'test_id': f'B3_K{k}', 'k': k, 'trials': len(df), 'participants': df.subject_id.nunique(), 'baseline_BA': obs.baseline_ba, 'full_10report_BA': obs.full_ba, 'Delta_BA': obs.statistic, 'parent_p_raw': p_raw, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(perm.Delta_BA.mean()), 'null_sd': float(perm.Delta_BA.std(ddof=1))})
    b3_metrics = pd.DataFrame(b3_rows)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B3_parent_metrics.csv', b3_metrics)
    atomic_csv(out / 'qc/discovery_semantic_report_execution_B3_outer_fold_audit.csv', pd.concat(b3_fold_tables, ignore_index=True))
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B3_permutation_values.csv', pd.concat(b3_perm_values, ignore_index=True).sort_values(['k', 'permutation']))
    b3_holm = holm_table([f'B3_K{k}' for k in TRANSITION_K], [float(b3_metrics.loc[b3_metrics.k.eq(k), 'parent_p_raw'].iloc[0]) for k in TRANSITION_K])
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B3_holm.csv', b3_holm)
    b3_child_rows = []
    b3_child_perm_values = []
    for k in TRANSITION_K:
        parent = b3_holm[b3_holm.test_id.eq(f'B3_K{k}')].iloc[0]
        if not bool(parent.reject_holm_0p05):
            for report in REPORTS:
                b3_child_rows.append({'k': k, 'report': report, 'status': 'NOT_RUN_PARENT_GATE', 'Delta_BA': np.nan, 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
            continue
        df = prepare_logistic(data, 'transition', k)
        baseline_ba = float(b3_metrics.loc[b3_metrics.k.eq(k), 'baseline_BA'].iloc[0])
        child_obs = {r: nested_lopo_logistic_full_ba(df, 'family_1based', [r], salt=f'B3_K{k}') - baseline_ba for r in REPORTS}
        cp = cache / f'B3_K{k}_single_report_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        batch_size = max(1, workers)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            def _b3_child_job(pp: int) -> dict[str, Any]:
                return single_report_permutation_row(df, 'family_1based', pp, f'B3_K{k}', 'B3_reports', baseline_ba)
            rows.extend(parallel_map(_b3_child_job, batch, workers))
            append_checkpoint(cp, rows)
            completed = len({int(r['permutation']) for r in rows})
            print(f'  B3 K={k} child reports: {completed}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        perm_export = perm.copy()
        perm_export['k'] = k
        b3_child_perm_values.append(perm_export)
        rawp = [permutation_p(child_obs[r], perm[r].to_numpy(float)) for r in REPORTS]
        h = holm_table(REPORTS, rawp)
        for r, p0 in zip(REPORTS, rawp):
            hr = h[h.test_id.eq(r)].iloc[0]
            b3_child_rows.append({'k': k, 'report': r, 'status': 'RUN_PARENT_PASSED', 'Delta_BA': child_obs[r], 'p_raw': p0, 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B3_single_report_followups.csv', pd.DataFrame(b3_child_rows))
    if b3_child_perm_values:
        b3_child_perm = pd.concat(b3_child_perm_values, ignore_index=True).sort_values(['k', 'permutation'])
    else:
        b3_child_perm = pd.DataFrame(columns=['permutation'] + REPORTS + ['k'])
    atomic_csv(out / 'tables/discovery_semantic_report_execution_B3_single_report_permutation_values.csv', b3_child_perm)
    discovery_parent_rows = []
    for _, r in a_holm.iterrows():
        discovery_parent_rows.append({'family': 'A', 'test_id': r.test_id, 'p_raw': float(r.p_raw), 'p_adjusted': float(r.p_holm), 'reject_adjusted_0p05': bool(r.reject_holm_0p05)})
    discovery_parent_rows.append({'family': 'B2', 'test_id': 'B2', 'p_raw': float(b2p), 'p_adjusted': float(b2p), 'reject_adjusted_0p05': bool(b2p < ALPHA)})
    for _, r in b3_holm.iterrows():
        discovery_parent_rows.append({'family': 'B3', 'test_id': r.test_id, 'p_raw': float(r.p_raw), 'p_adjusted': float(r.p_holm), 'reject_adjusted_0p05': bool(r.reject_holm_0p05)})
    parent_table = pd.DataFrame(discovery_parent_rows)
    atomic_csv(out / 'tables/discovery_semantic_report_execution_discovery_parent_test_registry.csv', parent_table)
    numeric_specification = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': script_sha, 'inner_group_folds': "For each outer-training participant set, sort participant IDs by SHA256('540|<test salt>|<subject_id>') and assign round-robin to 5 folds. Fold assignment depends only on participant ID, outer subject, and pre-prespecified test ID; it is fixed across permutations and report subsets for a given K/test.", 'logistic': {'solver': 'sklearn LogisticRegression lbfgs, L2, intercept, max_iter=4000, tol=1e-8', 'C_grid': A2_C_GRID, 'predictor_scaling': 'ordinary training-row mean/std only; zero/nonfinite std -> 1', 'sample_weight': 'start with 1/n_trials_within_training_participant, then multiply by total_participant_mass/(n_classes*participant_weighted_class_mass); no further renormalization', 'inner_score': 'mean across validation participants of balanced accuracy; within participant balanced accuracy is mean recall over classes actually present for that participant', 'tie_rule': 'smaller C', 'outer_score': 'mean participant-level balanced accuracy'}, 'ridge_B2': {'solver': 'sklearn Ridge cholesky, intercept, participant sample weights', 'alpha_grid': B2_ALPHA_GRID, 'X_scaling': 'participant-weighted training-only mean/std; zero/nonfinite std -> 1', 'Y_scaling': 'participant-weighted training-only mean/std per report; zero/nonfinite std -> 1', 'sample_weight': '1/n_trials_within_training_participant; each participant total weight 1', 'inner_score': 'mean participant multivariate MSE in training-standardized target space', 'tie_rule': 'larger alpha', 'outer_R2': 'within held-out participant, 1-SSE_model/SSE_zero in training-standardized target space; SSE_zero predicts zero; primary is mean participant Delta-R2 M2-M1'}, 'permutation_rng': {'seed_rule': 'SHA256(PHASE|permutation_salt|permutation_index|subject_id) -> uint32', 'A1': 'within-participant family-label permutation', 'A2': 'complete 10-report row-vector permutation within participant; common draw across K=2/3/4 and children', 'B2': 'complete 10-report row-vector permutation within participant', 'B3': 'complete 10-report row-vector permutation within participant; common draw across K=2/3 and children'}, 'multiplicity': {'A1_parent': 'min(1, 2*min(p_NMI,p_CramersV))', 'A_family': 'Holm across A1/A2 x K=2/3/4', 'B2': 'single parent; children only if parent p<0.05 then Holm across 10 reports', 'B3': 'Holm across K=2/3; children only after parent survives then Holm across 10 reports'}}
    numeric_specification_path = out / 'qc/discovery_semantic_report_execution_numeric_implementation_specification.json'
    writej(numeric_specification_path, numeric_specification)
    numeric_specification_sha = sha256(numeric_specification_path)
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'role': 'one-shot Validation target materialization and execution of the entire semantic_report_execution_design-prespecified A1/A2/B2/B3 program', 'Discovery_results_select_Validation_tests': False, 'Validation_tests_required': ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3'], 'Validation_behavior_opening_permitted_in_validation_semantic_report_execution': True, 'Confirmation_behavior_opening_permitted_in_validation_semantic_report_execution': False, 'new_EEG_signal_opening_permitted': False, 'canonical_context_stimulus_category_permitted': False, 'required_program_policy': {'path': str(semantic_report_execution_design_PROGRAM), 'sha256': semantic_report_execution_design_PROGRAM_SHA256}, 'discovery_semantic_report_execution_numeric_source_sha256': script_sha, 'discovery_semantic_report_execution_numeric_implementation_specification': {'path': str(numeric_specification_path.relative_to(project)), 'sha256': numeric_specification_sha}, 'validation_semantic_report_execution_must_reuse_discovery_semantic_report_execution_modeling_helpers': True, 'Discovery_parent_results_for_audit_only_not_selection': parent_table.to_dict('records'), 'Confirmation_remains_closed_until_post_Validation_design_specification': True, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}
    writej(project / NEXT_CONFIG, next_policy)
    source_rows = []
    for role, rel, expected in [('semantic_report_execution_design_final_lock', semantic_report_execution_design_LOCK, semantic_report_execution_design_LOCK_SHA256), ('semantic_report_execution_design_program', semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256), ('semantic_report_execution_design_next', semantic_report_execution_design_NEXT, semantic_report_execution_design_NEXT_SHA256), ('Discovery_outcome', DISCOVERY_OUTCOME, DISCOVERY_OUTCOME_SHA256), ('Discovery_canonical_master', DISCOVERY_MASTER, DISCOVERY_MASTER_SHA256), ('experience_assignments', EXP_ASSIGN, EXP_ASSIGN_SHA256), ('experience_signature', EXP_SIG, EXP_SIG_SHA256), ('transition_assignments', TR_ASSIGN, TR_ASSIGN_SHA256), ('transition_signature', TR_SIG, TR_SIG_SHA256)]:
        p = project / rel
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f'Unexpected rehash drift: {rel}')
        source_rows.append({'role': role, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': True})
    atomic_csv(out / 'qc/discovery_semantic_report_execution_source_rehash.csv', pd.DataFrame(source_rows))
    cohort_rows = []
    for k in EXPERIENCE_K:
        d1 = prepare_a1(data, k)
        d2 = prepare_logistic(data, 'experience', k)
        cohort_rows += [{'test_id': f'A1_K{k}', 'trials': len(d1), 'participants': d1.subject_id.nunique()}, {'test_id': f'A2_K{k}', 'trials': len(d2), 'participants': d2.subject_id.nunique()}]
    cohort_rows.append({'test_id': 'B2', 'trials': len(b2df), 'participants': b2df.subject_id.nunique()})
    for k in TRANSITION_K:
        d = prepare_logistic(data, 'transition', k)
        cohort_rows.append({'test_id': f'B3_K{k}', 'trials': len(d), 'participants': d.subject_id.nunique()})
    atomic_csv(out / 'qc/discovery_semantic_report_execution_test_cohort_audit.csv', pd.DataFrame(cohort_rows))
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'Discovery_semantic_report_execution_complete', 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'A_family_parent_results': parent_table[parent_table.family.eq('A')].to_dict('records'), 'B2_parent_result': parent_table[parent_table.test_id.eq('B2')].to_dict('records')[0], 'B3_parent_results': parent_table[parent_table.family.eq('B3')].to_dict('records'), 'Discovery_behavioral_report_values_read': True, 'Discovery_emotion_category_values_read': True, 'Discovery_context_values_read': True, 'new_EEG_signal_values_opened': False, 'Validation_behavioral_values_read': False, 'Confirmation_behavioral_values_read': False, 'canonical_context_category_used': False, 'behavior_changed_structural_K_or_family_ids': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_runs_all_prespecified_tests_regardless_Discovery_p': True, 'runtime_seconds': time.time() - started}
    writej(out / 'qc/discovery_semantic_report_execution_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': script_sha, 'semantic_report_execution_design_final_lock_sha256': semantic_report_execution_design_LOCK_SHA256, 'semantic_report_execution_design_program_sha256': semantic_report_execution_design_PROGRAM_SHA256, 'semantic_report_execution_design_next_sha256': semantic_report_execution_design_NEXT_SHA256, 'discovery_semantic_report_execution_numeric_implementation_specification_sha256': numeric_specification_sha, 'boundary': {'Discovery_behavioral_report_values_read': True, 'Discovery_emotion_category_values_read': True, 'Discovery_context_values_read': True, 'new_EEG_signal_values_opened': False, 'Validation_behavioral_values_read': False, 'Confirmation_behavioral_values_read': False, 'canonical_context_category_used': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/discovery_semantic_report_execution_contract.json', contract)
    report = ['# discovery_semantic_report_execution — Discovery semantic/report execution', '', '**This is the first structural_replication phase to open Discovery behavioral/report targets. No Validation or Confirmation behavioral values and no new EEG signal values were opened.**', '', '## Parent results', '', '```text', parent_table.to_string(index=False), '```', '', '## Interpretation discipline', '', '- A1 tests whether nominal human emotion categories align with the replicated experience partitions; a null result alone is not proof that a category is invalid.', '- A2 tests whether all ten reports jointly predict experience-family membership beyond context/acquisition nuisance.', '- B2 tests whether the continuous `stimulus_late -> rating_early` signature carries report information beyond the continuous stimulus-only experience signature.', '- B3 is the secondary joint-report mapping to replicated report-transition families.', '', 'No family is renamed from these Discovery results. Any positive report-transition association is evidence only of report/evaluation-stage neural transformation related to the eventual report; it is not evidence of distortion, dishonesty, suppression, exaggeration, or conscious editing.', '', '## Held-out lockbox', '', 'All A1/A2/B2/B3 tests now proceed unchanged to Validation regardless of Discovery p-values. Discovery cannot select a K, report subset, family name, or held-out test.', '', f'## Next', '', f'**{NEXT_PHASE}** — one-shot Validation target materialization + execution of every semantic_report_execution_design-prespecified test.', '']
    (out / 'reports/discovery_semantic_report_execution_report.md').write_text(''.join(report), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, out / 'qc/discovery_semantic_report_execution_contract.json', out / 'qc/discovery_semantic_report_execution_summary.json', out / 'qc/discovery_semantic_report_execution_source_rehash.csv', out / 'qc/discovery_semantic_report_execution_test_cohort_audit.csv', numeric_specification_path, out / 'qc/discovery_semantic_report_execution_A2_outer_fold_audit.csv', out / 'qc/discovery_semantic_report_execution_B2_outer_fold_audit.csv', out / 'qc/discovery_semantic_report_execution_B3_outer_fold_audit.csv', out / 'tables/discovery_semantic_report_execution_A1_parent_metrics.csv', out / 'tables/discovery_semantic_report_execution_A1_mixing_cells.csv', out / 'tables/discovery_semantic_report_execution_A1_concentration.csv', out / 'tables/discovery_semantic_report_execution_A1_null_summary.csv', out / 'tables/discovery_semantic_report_execution_A1_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_A2_parent_metrics.csv', out / 'tables/discovery_semantic_report_execution_A2_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_A_family_holm.csv', out / 'tables/discovery_semantic_report_execution_A2_single_report_followups.csv', out / 'tables/discovery_semantic_report_execution_A2_single_report_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_B2_parent_metric.csv', out / 'tables/discovery_semantic_report_execution_B2_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_B2_dimension_followups.csv', out / 'tables/discovery_semantic_report_execution_B3_parent_metrics.csv', out / 'tables/discovery_semantic_report_execution_B3_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_B3_holm.csv', out / 'tables/discovery_semantic_report_execution_B3_single_report_followups.csv', out / 'tables/discovery_semantic_report_execution_B3_single_report_permutation_values.csv', out / 'tables/discovery_semantic_report_execution_discovery_parent_test_registry.csv', out / 'reports/discovery_semantic_report_execution_report.md']
    final = {'phase': PHASE, 'generated_at': now(), 'status': 'Discovery_semantic_report_execution_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'Discovery_behavioral_values_read': True, 'Validation_behavioral_values_read': False, 'Confirmation_behavioral_values_read': False, 'new_EEG_signal_values_opened': False, 'canonical_context_category_used': False, 'behavior_changed_structural_K_or_family_ids': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_runs_all_prespecified_tests_regardless_Discovery_p': True}
    writej(final_lock, final)
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('discovery_semantic_report_execution PASS')
    for _, r in parent_table.iterrows():
        print(f'{r.test_id}: p_raw={r.p_raw:.6g}, p_adjusted={r.p_adjusted:.6g}, reject={bool(r.reject_adjusted_0p05)}')
    print('Discovery behavioral values opened: True')
    print('Validation behavioral values opened: False')
    print('Confirmation behavioral values opened: False')
    print('New EEG signal values opened: False')
    print('Canonical context/category used: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print('Next phase runs all prespecified tests regardless Discovery p: True')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())