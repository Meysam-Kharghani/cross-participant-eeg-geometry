# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, log_loss, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
PHASE = 'family_interpretability_extension'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 20260812
ALPHA = 0.05
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
POS_REPORTS = ['joy', 'inspiration', 'tenderness']
NEG_REPORTS = ['sadness', 'fear', 'disgust']
EMOTION_REPORTS = POS_REPORTS + NEG_REPORTS
POS_CATEGORIES = {'joy', 'inspiration', 'tenderness'}
NEG_CATEGORIES = {'sadness', 'fear', 'disgust'}
NEUTRAL_CATEGORIES = {'neutral'}
GEOM = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']
GEOM_SHORT = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
GEOM_SHORT_TO_STD = dict(zip(GEOM_SHORT, GEOM))
cross_person_alignment_fit_LOCK = 'outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json'
cross_person_alignment_fit_CANDIDATE_METRICS = 'outputs/tables/cross_person_alignment_fit_trajectory_family_candidate_metrics.csv'
cross_person_alignment_fit_CANDIDATE_ASSIGN = 'outputs/tables/cross_person_alignment_fit_trajectory_family_candidate_assignments.csv'
cross_person_alignment_fit_CANDIDATE_CENTROIDS = 'outputs/tables/cross_person_alignment_fit_trajectory_family_candidate_centroids.csv'
cross_person_alignment_fit_SELECTED_ASSIGN = 'outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv'
cross_person_alignment_fit_SELECTED_CENTROIDS = 'outputs/tables/cross_person_alignment_fit_trajectory_family_centroids.csv'
cross_person_alignment_fit_STD_GEOM = 'outputs/tables/cross_person_alignment_fit_trajectory_standardized_geometry.csv'
mapping_design_LOCK = 'outputs/qc/mapping_design_final_output_lock_hashes.json'
mapping_design_TRIAL = 'outputs/tables/mapping_design_trial_mapping_manifest.csv'
mapping_inference_LOCK = 'outputs/qc/mapping_inference_final_output_lock_hashes.json'
mapping_inference_GLOBAL = 'outputs/tables/mapping_inference_global_hypothesis_tests.csv'
baseline_evidence_refresh_LOCK = '<BASELINE_EVIDENCE_CHECKPOINT>'
cross_person_alignment_fit_CRITICAL = {cross_person_alignment_fit_CANDIDATE_METRICS: 'trajectory_candidates', cross_person_alignment_fit_CANDIDATE_ASSIGN: 'trajectory_candidate_assignments', cross_person_alignment_fit_CANDIDATE_CENTROIDS: 'trajectory_candidate_centroids', cross_person_alignment_fit_SELECTED_ASSIGN: 'trajectory_selected_assignments', cross_person_alignment_fit_SELECTED_CENTROIDS: 'trajectory_selected_centroids', cross_person_alignment_fit_STD_GEOM: 'trajectory_standardized'}

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def write_csv(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        pd.DataFrame(obj).to_csv(path, index=False)

def boolish(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj_sorted = []
    running = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * float(p))
        running = max(running, a)
        adj_sorted.append((k, running))
    return dict(adj_sorted)

def t_abs_from_contrasts(c: np.ndarray) -> float:
    c = np.asarray(c, float)
    c = c[np.isfinite(c)]
    if len(c) < 3:
        return math.nan
    sd = float(np.std(c, ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(float(np.mean(c))) <= 1e-12 else float('inf')
    return abs(float(np.mean(c))) / (sd / math.sqrt(len(c)))

def signed_mean_from_contrasts(c: np.ndarray) -> float:
    c = np.asarray(c, float)
    c = c[np.isfinite(c)]
    return float(np.mean(c)) if len(c) else math.nan

def joint_signflip_maxt(C: np.ndarray, B: int, seed: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """C: participants x children. Joint sign flip preserves child covariance."""
    C = np.asarray(C, float)
    obs = np.array([t_abs_from_contrasts(C[:, j]) for j in range(C.shape[1])], float)
    rng = np.random.default_rng(seed)
    null = np.full((B, C.shape[1]), np.nan, float)
    for b in range(B):
        signs = rng.choice(np.array([-1.0, 1.0]), size=C.shape[0])
        Cp = C * signs[:, None]
        for j in range(C.shape[1]):
            null[b, j] = t_abs_from_contrasts(Cp[:, j])
    maxnull = np.nanmax(null, axis=1)
    raw = np.array([(1 + np.sum(null[:, j] >= obs[j])) / (B + 1) for j in range(C.shape[1])])
    maxt = np.array([(1 + np.sum(maxnull >= obs[j])) / (B + 1) for j in range(C.shape[1])])
    global_p = float((1 + np.sum(maxnull >= np.nanmax(obs))) / (B + 1))
    return (obs, raw, global_p, maxt)

def participant_contrast(d: pd.DataFrame, value: str, group: str, level_a: Any, level_b: Any, subjects: list[str]) -> np.ndarray:
    out = np.full(len(subjects), np.nan)
    for i, sid in enumerate(subjects):
        q = d[d.subject_id == sid]
        a = pd.to_numeric(q.loc[q[group] == level_a, value], errors='coerce').dropna().to_numpy(float)
        b = pd.to_numeric(q.loc[q[group] == level_b, value], errors='coerce').dropna().to_numpy(float)
        if len(a) and len(b):
            out[i] = float(np.mean(b) - np.mean(a))
    return out

def participant_did(d: pd.DataFrame, value: str, g1: str, a1: Any, b1: Any, g2: str, a2: Any, b2: Any, subjects: list[str]) -> np.ndarray:
    out = np.full(len(subjects), np.nan)
    for i, sid in enumerate(subjects):
        q = d[d.subject_id == sid]
        cells = {}
        for x1 in [a1, b1]:
            for x2 in [a2, b2]:
                v = pd.to_numeric(q.loc[(q[g1] == x1) & (q[g2] == x2), value], errors='coerce').dropna().to_numpy(float)
                if len(v):
                    cells[x1, x2] = float(np.mean(v))
        if len(cells) == 4:
            out[i] = cells[b1, b2] - cells[a1, b2] - (cells[b1, a2] - cells[a1, a2])
    return out

def robust_z_by_subject(d: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(index=d.index, dtype=float)
    for sid, idx in d.groupby('subject_id').groups.items():
        x = pd.to_numeric(d.loc[idx, col], errors='coerce').astype(float)
        med = float(np.nanmedian(x))
        q25, q75 = np.nanpercentile(x, [25, 75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(x, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        out.loc[idx] = (x - med) / scale
    return out

def add_derived_behavior(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['positive_mean'] = d[POS_REPORTS].mean(axis=1)
    d['negative_mean'] = d[NEG_REPORTS].mean(axis=1)
    d['component_balance'] = d['positive_mean'] - d['negative_mean']
    d['mixed_affect'] = np.minimum(d['positive_mean'], d['negative_mean'])
    d['overall_emotion_intensity'] = d[EMOTION_REPORTS].mean(axis=1)
    arr = d[EMOTION_REPORTS].to_numpy(float)
    s = np.nansum(arr, axis=1)
    ent = np.zeros(len(d), float)
    for i in range(len(d)):
        if np.isfinite(s[i]) and s[i] > 0:
            p = np.clip(arr[i], 0, None)
            ps = p.sum()
            if ps > 0:
                p = p / ps
                nz = p[p > 0]
                ent[i] = -float(np.sum(nz * np.log(nz))) / math.log(len(EMOTION_REPORTS))
    d['emotion_entropy'] = ent
    sort6 = np.sort(np.nan_to_num(arr, nan=-np.inf), axis=1)
    d['top2_margin'] = sort6[:, -1] - sort6[:, -2]
    d['rating_extremity'] = np.nanmean(np.abs(d[REPORTS].to_numpy(float) - 3.5), axis=1)
    d['liking_valence_gap'] = np.abs(d['liking'] - d['valence'])
    d['valence_robust_z'] = robust_z_by_subject(d, 'valence')
    d['component_balance_robust_z'] = robust_z_by_subject(d, 'component_balance')
    d['valence_component_discrepancy'] = np.abs(d['valence_robust_z'] - d['component_balance_robust_z'])
    pol_sign = d['stimulus_polarity'].map({'positive': 1.0, 'negative': -1.0})
    d['stimulus_valence_congruence'] = pol_sign * (d['valence'] - 3.5)
    target_dom = np.full(len(d), np.nan)
    for i, row in d.iterrows():
        cat = row.get('emotion_category')
        if cat in EMOTION_REPORTS:
            others = [c for c in EMOTION_REPORTS if c != cat]
            target_dom[d.index.get_loc(i)] = float(row[cat] - np.nanmax(row[others].to_numpy(float)))
    d['target_emotion_dominance'] = target_dom
    return d

def report_family_contrast_matrix(d: pd.DataFrame, cols: list[str], subjects: list[str]) -> np.ndarray:
    C = np.full((len(subjects), len(cols)), np.nan)
    for i, sid in enumerate(subjects):
        q = d[d.subject_id == sid]
        for j, c in enumerate(cols):
            a = pd.to_numeric(q.loc[q.family2 == 0, c], errors='coerce').dropna().to_numpy(float)
            b = pd.to_numeric(q.loc[q.family2 == 1, c], errors='coerce').dropna().to_numpy(float)
            if len(a) and len(b):
                C[i, j] = float(np.mean(b) - np.mean(a))
    return C

def extreme_family_contrasts(d: pd.DataFrame, cols: list[str], subjects: list[str]) -> np.ndarray:
    C = np.full((len(subjects), len(cols)), np.nan)
    for j, c in enumerate(cols):
        pct = d.groupby('subject_id')[c].rank(method='average', pct=True)
        for i, sid in enumerate(subjects):
            qidx = d.index[d.subject_id == sid]
            low = qidx[pct.loc[qidx] <= 1 / 3]
            high = qidx[pct.loc[qidx] >= 2 / 3]
            if len(low) and len(high):
                C[i, j] = float(d.loc[high, 'family2'].mean() - d.loc[low, 'family2'].mean())
    return C

def participant_balanced_ba(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    for sid in np.unique(subjects):
        m = subjects == sid
        yt = y[m]
        pr = pred[m]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(balanced_accuracy_score(yt, pr))
    return float(np.mean(vals)) if vals else math.nan

def participant_balanced_logloss(y: np.ndarray, prob: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    for sid in np.unique(subjects):
        m = subjects == sid
        yt = y[m]
        pp = np.clip(prob[m], 1e-06, 1 - 1e-06)
        if len(np.unique(yt)) < 2:
            continue
        vals.append(log_loss(yt, np.c_[1 - pp, pp], labels=[0, 1]))
    return float(np.mean(vals)) if vals else math.nan

def build_X(d: pd.DataFrame, report_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    base = pd.DataFrame(index=d.index)
    base['context_video'] = (d['context'].astype(str) == 'video').astype(float)
    base['acquisition_fraction'] = pd.to_numeric(d['acquisition_fraction'], errors='coerce').astype(float)
    base['acquisition_fraction_sq'] = base['acquisition_fraction'] ** 2
    for c in report_cols:
        base[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    if base.isna().any().any():
        raise RuntimeError(f'Nonfinite predictor in model columns: {base.columns[base.isna().any()].tolist()}')
    return (base.to_numpy(float), base.columns.tolist())

def lopo_fixed_logistic(d: pd.DataFrame, report_cols: list[str], y_override: np.ndarray | None=None) -> dict[str, Any]:
    X, names = build_X(d, report_cols)
    y = d['family2'].to_numpy(int) if y_override is None else np.asarray(y_override, int)
    subjects = d['subject_id'].astype(str).to_numpy()
    probs = np.full(len(d), np.nan)
    preds = np.full(len(d), -1, int)
    coefs = []
    unique_subjects = sorted(np.unique(subjects))
    for sid in unique_subjects:
        te = subjects == sid
        tr = ~te
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        model = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', max_iter=2000, random_state=SEED)
        model.fit(Xtr, y[tr])
        probs[te] = model.predict_proba(Xte)[:, 1]
        preds[te] = (probs[te] >= 0.5).astype(int)
        coefs.append({'heldout_subject': sid, **{n: float(v) for n, v in zip(names, model.coef_[0])}})
    return {'ba': participant_balanced_ba(y, preds, subjects), 'logloss': participant_balanced_logloss(y, probs, subjects), 'probs': probs, 'preds': preds, 'coefs': pd.DataFrame(coefs)}

def permute_y_within_subject(y: np.ndarray, subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yp = np.asarray(y).copy()
    for sid in np.unique(subjects):
        idx = np.flatnonzero(subjects == sid)
        yp[idx] = rng.permutation(yp[idx])
    return yp

def model_perm_once(d: pd.DataFrame, seed: int) -> float:
    subjects = d['subject_id'].astype(str).to_numpy()
    y = d['family2'].to_numpy(int)
    yp = permute_y_within_subject(y, subjects, seed)
    b = lopo_fixed_logistic(d, [], y_override=yp)
    f = lopo_fixed_logistic(d, REPORTS, y_override=yp)
    return float(f['ba'] - b['ba'])

def subset_score(d: pd.DataFrame, subset: tuple[str, ...], baseline_ba: float) -> dict[str, Any]:
    r = lopo_fixed_logistic(d, list(subset))
    return {'subset_size': len(subset), 'report_subset': '|'.join(subset), 'participant_balanced_ba': r['ba'], 'delta_ba_vs_nuisance_baseline': float(r['ba'] - baseline_ba), 'participant_balanced_logloss': r['logloss']}

def exhaustive_subset_atlas_ridge(d: pd.DataFrame, baseline_ba: float, ridge_lambda: float=1.0) -> pd.DataFrame:
    """Fast descriptive 1023-subset atlas using fixed linear ridge classification.

    This is intentionally NOT the inferential N2 model. It is a hypothesis-generating
    screen only, so no p-value is attached to the winning subset.
    """
    base = pd.DataFrame(index=d.index)
    base['context_video'] = (d['context'].astype(str) == 'video').astype(float)
    base['acquisition_fraction'] = pd.to_numeric(d['acquisition_fraction'], errors='coerce').astype(float)
    base['acquisition_fraction_sq'] = base['acquisition_fraction'] ** 2
    for c in REPORTS:
        base[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    if base.isna().any().any():
        raise RuntimeError('Nonfinite value in subset-atlas predictors')
    Xall = base.to_numpy(float)
    y = d['family2'].to_numpy(int)
    ypm = 2.0 * y - 1.0
    subjects = d['subject_id'].astype(str).to_numpy()
    n_masks = (1 << len(REPORTS)) - 1
    preds = np.empty((n_masks, len(d)), dtype=np.int8)
    subset_cols = []
    subset_names = []
    for mask in range(1, n_masks + 1):
        reps = [REPORTS[j] for j in range(len(REPORTS)) if mask & 1 << j]
        cols = [0, 1, 2] + [3 + j for j in range(len(REPORTS)) if mask & 1 << j]
        subset_cols.append(cols)
        subset_names.append(reps)
    for sid in sorted(np.unique(subjects)):
        te = subjects == sid
        tr = ~te
        mu = Xall[tr].mean(axis=0)
        sd = Xall[tr].std(axis=0, ddof=0)
        sd[sd <= 1e-12] = 1.0
        Xtr0 = (Xall[tr] - mu) / sd
        Xte0 = (Xall[te] - mu) / sd
        yt = ypm[tr]
        Ztr_full = np.column_stack([np.ones(len(Xtr0)), Xtr0])
        Zte_full = np.column_stack([np.ones(len(Xte0)), Xte0])
        Gfull = Ztr_full.T @ Ztr_full
        bfull = Ztr_full.T @ yt
        for mi, cols in enumerate(subset_cols):
            zcols = [0] + [1 + c for c in cols]
            A = Gfull[np.ix_(zcols, zcols)].copy()
            rhs = bfull[zcols]
            pen = np.eye(A.shape[0]) * ridge_lambda
            pen[0, 0] = 0.0
            beta = np.linalg.solve(A + pen, rhs)
            score = Zte_full[:, zcols] @ beta
            preds[mi, te] = (score >= 0.0).astype(np.int8)
    rows = []
    for mi, reps in enumerate(subset_names):
        ba = participant_balanced_ba(y, preds[mi], subjects)
        rows.append({'subset_size': len(reps), 'report_subset': '|'.join(reps), 'participant_balanced_ba': ba, 'delta_ba_vs_nuisance_logistic_baseline': float(ba - baseline_ba), 'atlas_model': 'fixed_linear_ridge_lambda1_LOPO_descriptive'})
    return pd.DataFrame(rows)

def sequential_persistence(d: pd.DataFrame, B: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    subjects = sorted(d.subject_id.unique())
    seqs = []
    rows = []
    obs_vals = []
    for sid in subjects:
        q = d[d.subject_id == sid].sort_values('trial_number')
        y = q.family2.to_numpy(int)
        if len(y) < 2:
            continue
        stay = float(np.mean(y[1:] == y[:-1]))
        obs_vals.append(stay)
        seqs.append((sid, y))
        n00 = int(np.sum((y[:-1] == 0) & (y[1:] == 0)))
        n01 = int(np.sum((y[:-1] == 0) & (y[1:] == 1)))
        n10 = int(np.sum((y[:-1] == 1) & (y[1:] == 0)))
        n11 = int(np.sum((y[:-1] == 1) & (y[1:] == 1)))
        rows.append({'subject_id': sid, 'n_trials': len(y), 'family2_fraction': float(np.mean(y)), 'stay_fraction': stay, 'n_F1_to_F1': n00, 'n_F1_to_F2': n01, 'n_F2_to_F1': n10, 'n_F2_to_F2': n11, 'p_F2_given_F1': n01 / max(1, n00 + n01), 'p_F2_given_F2': n11 / max(1, n10 + n11)})
    obs = float(np.mean(obs_vals))
    rng = np.random.default_rng(seed)
    null = np.empty(B, float)
    for b in range(B):
        vals = []
        for sid, y in seqs:
            yp = rng.permutation(y)
            vals.append(float(np.mean(yp[1:] == yp[:-1])))
        null[b] = float(np.mean(vals))
    center = float(np.mean(null))
    dev = abs(obs - center)
    p = float((1 + np.sum(np.abs(null - center) >= dev)) / (B + 1))
    return ({'observed_participant_balanced_stay_fraction': obs, 'null_mean': center, 'effect_vs_null': obs - center, 'two_sided_p': p, 'participants': len(obs_vals)}, pd.DataFrame(rows), null)

def occupancy_reliability(d: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rows = []
    subj = []
    for sid, q in d.groupby('subject_id'):
        a = q[q.acquisition_fraction <= 0.5].family2.mean()
        b = q[q.acquisition_fraction > 0.5].family2.mean()
        im = q[q.context == 'imagery'].family2.mean()
        vi = q[q.context == 'video'].family2.mean()
        subj.append([sid, a, b, im, vi])
    s = pd.DataFrame(subj, columns=['subject_id', 'f2_first_half', 'f2_second_half', 'f2_imagery', 'f2_video'])
    rng = np.random.default_rng(seed)
    for name, a, b in [('first_half_vs_second_half', 'f2_first_half', 'f2_second_half'), ('imagery_vs_video', 'f2_imagery', 'f2_video')]:
        ok = s[[a, b]].dropna()
        rho = float(spearmanr(ok[a], ok[b]).statistic) if len(ok) >= 3 else math.nan
        null = np.empty(B, float)
        bv = ok[b].to_numpy(float)
        av = ok[a].to_numpy(float)
        for i in range(B):
            null[i] = float(spearmanr(av, rng.permutation(bv)).statistic)
        p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (B + 1))
        rows.append({'reliability': name, 'n_participants': len(ok), 'spearman_rho': rho, 'permutation_p': p})
    return (pd.DataFrame(rows), s)

def verify_inputs(project: Path) -> dict[str, Any]:
    required = [cross_person_alignment_fit_LOCK, *cross_person_alignment_fit_CRITICAL.keys(), mapping_design_LOCK, mapping_design_TRIAL, baseline_evidence_refresh_LOCK]
    missing = [r for r in required if not (project / r).is_file()]
    if missing:
        raise RuntimeError('Missing prerequisites:\n  ' + '\n  '.join(missing))
    dlock = json.loads((project / cross_person_alignment_fit_LOCK).read_text())
    if dlock.get('phase') != 'cross_person_alignment_fit' or boolish(dlock.get('reports_outcomes_context_qc_read')):
        raise RuntimeError('cross_person_alignment_fit outcome-free boundary mismatch')
    for rel, key in cross_person_alignment_fit_CRITICAL.items():
        exp = dlock.get('hashes', {}).get(key)
        got = sha256(project / rel)
        if not exp or got != exp:
            raise RuntimeError(f'cross_person_alignment_fit critical hash mismatch: {rel}')
    elock = json.loads((project / mapping_design_LOCK).read_text())
    if elock.get('phase') != 'mapping_design':
        raise RuntimeError('mapping_design phase mismatch')
    if boolish(elock.get('associations_calculated')) or boolish(elock.get('validation_opened')) or boolish(elock.get('confirmation_opened')):
        raise RuntimeError('mapping_design boundary mismatch')
    if sha256(project / mapping_design_TRIAL) != elock.get('hashes', {}).get('trial_manifest'):
        raise RuntimeError('mapping_design trial manifest hash mismatch')
    b = json.loads((project / baseline_evidence_refresh_LOCK).read_text())
    if b.get('phase') != 'interpretability_route_specification':
        raise RuntimeError('baseline_evidence_refresh phase mismatch')
    if b.get('external_replication_claim_permitted') is not False or b.get('family_labels_remain_geometric') is not True:
        raise RuntimeError('baseline_evidence_refresh claim boundary mismatch')
    historical = None
    if (project / mapping_inference_LOCK).is_file() and (project / mapping_inference_GLOBAL).is_file():
        flock = json.loads((project / mapping_inference_LOCK).read_text())
        if flock.get('phase') != 'mapping_inference':
            raise RuntimeError('mapping_inference phase mismatch')
        exp = flock.get('hashes', {}).get('global_tests')
        if exp and sha256(project / mapping_inference_GLOBAL) != exp:
            raise RuntimeError('mapping_inference global-test hash mismatch')
        historical = pd.read_csv(project / mapping_inference_GLOBAL)
    return {'cross_person_alignment_fit_lock': dlock, 'mapping_design_lock': elock, 'baseline_evidence_refresh_lock': b, 'historical': historical}
POLICY = {'phase': PHASE, 'seed': SEED, 'scientific_role': 'post-specification Discovery-only family-interpretability extension; no semantic relabeling without new held-out replication', 'why_K2_is_a_clue_not_a_binary_truth': 'K=2 is the selected parsimonious stable coarse partition. Binary explanations are prioritized but continuous, nonlinear, sequential and hierarchical alternatives are tested too.', 'historical_tests_not_rebranded_as_new': {'context': 'already formally tested as mapping_inference H4 and unsupported', '10D_report_mean_profile': 'already formally tested as mapping_inference H1 and unsupported', '7_level_emotion_category': 'already formally tested as mapping_inference H5 and unsupported', 'stimulus_identity': 'already formally tested as mapping_inference H6 and unsupported'}, 'new_primary_domains': {'N1_binary_task_stimulus_structure': {'children': ['nominal positive-vs-negative stimulus polarity (neutral excluded)', 'context x polarity difference-in-differences', 'late-vs-early acquisition half', 'context-switch-vs-repeat'], 'test': 'participant contrast vector; joint 10000 sign-flip maxT family'}, 'N2_full_10D_report_predictive_information': {'question': 'can same-trial 10D report vector predict fixed family in an unseen participant beyond context + acquisition order?', 'model': 'fixed L2 logistic C=1, leave-one-participant-out; nuisance baseline vs nuisance+all 10 reports', 'null': 'within-participant family-label permutations; full pipeline refit', 'subset_atlas': 'all 2^10-1=1023 nonempty report subsets scored with a fixed LOPO linear-ridge screen for speed; descriptive only, no best-subset p-value'}, 'N3_nonlinear_report_structure': {'children': ['component_balance', 'mixed_affect', 'overall_emotion_intensity', 'emotion_entropy', 'top2_margin', 'rating_extremity', 'liking_valence_gap', 'valence_component_discrepancy', 'stimulus_valence_congruence', 'target_emotion_dominance'], 'test': 'participant family2-minus-family1 contrasts; joint sign-flip maxT'}, 'N4_family_sequence_persistence': {'question': "do adjacent trials stay in the same family more or less often than expected from each participant's family occupancy?", 'null': 'within-participant sequence permutation preserving family counts'}}, 'secondary_exploratory_but_multiplicity_controlled': {'low_vs_high_original_reports': 'within-participant bottom vs top tertile, middle omitted; joint maxT across 10 reports', 'occupancy_reliability': 'family2 occupancy first-half vs second-half and imagery vs video', 'K_resolution_hierarchy': 'outcome-free K2-vs-K3/K4/K6 nestedness diagnostics', 'family_axis': 'continuous signed centroid-distance axis and assignment margin; descriptive only'}, 'global_multiplicity': 'Holm across N1, N2, N3, N4 global p-values; child semantic claims require both parent global Holm<0.05 and child maxT<0.05.', 'validation_confirmation': 'remain unopened in family_interpretability_extension; any supported interpretation requires a new authoritative held-out behavior-to-EEG event lineage before replication', 'forbidden': ['renaming Family 1/2 as positive/negative, imagery/video, pleasure/pain, appraisal/reporting, or any semantic state based on Discovery alone', 'choosing a report subset after inspecting its score and then treating it as confirmatory', 'opening Validation/Confirmation behavioral semantics in this phase', 'changing K, family assignments, EEG geometry, or original mapping_inference results']}

def self_test() -> None:
    subsets = sum((math.comb(10, k) for k in range(1, 11)))
    assert subsets == 1023
    h = holm({'a': 0.01, 'b': 0.03, 'c': 0.2})
    assert h['a'] <= h['b'] <= h['c']
    C = np.array([[1, 2], [2, 1], [-1, 0.5], [0.5, -1]], float)
    obs, raw, g, maxt = joint_signflip_maxt(C, 100, 1)
    assert len(obs) == 2 and len(raw) == 2 and (len(maxt) == 2) and (0 <= g <= 1)
    print('family_interpretability_extension self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--contrast-permutations', type=int, default=10000)
    ap.add_argument('--model-permutations', type=int, default=1000)
    ap.add_argument('--sequence-permutations', type=int, default=10000)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--subset-atlas', choices=['full', 'skip'], default='full')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    upstream = verify_inputs(project)
    outroot = project / 'outputs/interpretability/family_interpretability_extension_family_interpretability_extension'
    if outroot.exists():
        if a.force:
            shutil.rmtree(outroot)
        else:
            raise RuntimeError(f'Output exists: {outroot}; use --force to reproduce')
    for dname in ['tables', 'qc', 'reports', 'figures']:
        (outroot / dname).mkdir(parents=True, exist_ok=True)
    policy_path = project / 'analysis_specifications/S032_family_interpretability_extension.json'
    write_json(policy_path, POLICY)
    policy_sha = canonical_sha(POLICY)
    policy_lock = project / 'outputs/qc/family_interpretability_extension_policy_lock.json'
    write_json(policy_lock, {'phase': PHASE, 'generated_at': now(), 'policy_canonical_sha256': policy_sha, 'behavior_values_opened_after_this_lock': True})
    metrics = pd.read_csv(project / cross_person_alignment_fit_CANDIDATE_METRICS)
    cand = pd.read_csv(project / cross_person_alignment_fit_CANDIDATE_ASSIGN)
    cents = pd.read_csv(project / cross_person_alignment_fit_SELECTED_CENTROIDS)
    geom = pd.read_csv(project / cross_person_alignment_fit_STD_GEOM)
    sel = pd.read_csv(project / cross_person_alignment_fit_SELECTED_ASSIGN)
    write_csv(outroot / 'tables/family_interpretability_extension_k_candidate_metrics.csv', metrics)
    wide = cand.pivot_table(index=['subject_id', 'raw_behavior_row_id', 'trial_number'], columns='candidate_shared_k', values='trajectory_family_1based', aggfunc='first').reset_index()
    resolution = []
    for k in [3, 4, 6]:
        if 2 in wide.columns and k in wide.columns:
            x = wide[2].astype(int).to_numpy()
            y = wide[k].astype(int).to_numpy()
            resolution.append({'coarse_k': 2, 'comparison_k': k, 'n': len(x), 'adjusted_rand_index': adjusted_rand_score(x, y), 'normalized_mutual_information': normalized_mutual_info_score(x, y)})
            ct = pd.crosstab(wide[2], wide[k], margins=True)
            ct.to_csv(outroot / f'tables/family_interpretability_extension_k2_vs_k{k}_crosstab.csv')
    write_csv(outroot / 'tables/family_interpretability_extension_resolution_nestedness.csv', resolution)
    Cshort = cents.pivot(index='trajectory_family_1based', columns='standardized_geometry_feature', values='centroid_value').loc[[1, 2], GEOM_SHORT]
    Cmat = Cshort.rename(columns=GEOM_SHORT_TO_STD)
    centroid_summary = []
    for g in GEOM:
        centroid_summary.append({'geometry_feature': g, 'family1_centroid': float(Cmat.loc[1, g]), 'family2_centroid': float(Cmat.loc[2, g]), 'family2_minus_family1': float(Cmat.loc[2, g] - Cmat.loc[1, g])})
    write_csv(outroot / 'tables/family_interpretability_extension_family_centroid_geometry.csv', centroid_summary)
    gg = geom.merge(sel[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assignment_margin']], on=['subject_id', 'raw_behavior_row_id', 'trial_number'], validate='one_to_one')
    Xg = gg[GEOM].to_numpy(float)
    outcome_model_integrity = Cmat.loc[1].to_numpy(float)
    c2 = Cmat.loc[2].to_numpy(float)
    d1 = np.sum((Xg - outcome_model_integrity) ** 2, axis=1)
    d2 = np.sum((Xg - c2) ** 2, axis=1)
    gg['distance_sq_family1'] = d1
    gg['distance_sq_family2'] = d2
    gg['signed_family_axis_d1_minus_d2'] = d1 - d2
    gg['normalized_family_axis'] = (d1 - d2) / (d1 + d2 + 1e-12)
    gg['nearest_centroid_family'] = np.where(d1 <= d2, 1, 2)
    if not np.all(gg.nearest_centroid_family.to_numpy() == gg.trajectory_family_1based.to_numpy()):
        raise RuntimeError('Centroid distance reconstruction does not reproduce prespecified family assignments')
    write_csv(outroot / 'tables/family_interpretability_extension_continuous_family_axis.csv', gg[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'assignment_margin', 'distance_sq_family1', 'distance_sq_family2', 'signed_family_axis_d1_minus_d2', 'normalized_family_axis']])
    d = pd.read_csv(project / mapping_design_TRIAL)
    needed = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'raw_run', 'within_subject_block_fold', 'acquisition_fraction', 'context', 'stimulus_id', 'emotion_category', 'trajectory_family_1based', 'assignment_margin', *REPORTS, *GEOM]
    miss = [c for c in needed if c not in d.columns]
    if miss:
        raise RuntimeError(f'mapping_design manifest missing columns: {miss}')
    d = d[needed].copy()
    if len(d) != 1237 or d.subject_id.nunique() != 31:
        raise RuntimeError('Discovery cohort drift')
    d['family2'] = (d.trajectory_family_1based.astype(int) == 2).astype(int)
    d['stimulus_polarity'] = d.emotion_category.map(lambda x: 'positive' if x in POS_CATEGORIES else 'negative' if x in NEG_CATEGORIES else 'neutral' if x in NEUTRAL_CATEGORIES else np.nan)
    d['acquisition_half'] = np.where(d.acquisition_fraction <= 0.5, 'early', 'late')
    d = d.sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    d['prev_context'] = d.groupby('subject_id').context.shift(1)
    d['context_switch'] = pd.Series(np.where(d.context != d.prev_context, 'switch', 'repeat'), index=d.index, dtype='object')
    d.loc[d.prev_context.isna(), 'context_switch'] = np.nan
    d['prev_polarity'] = d.groupby('subject_id').stimulus_polarity.shift(1)
    d['polarity_switch'] = pd.Series(np.where(d.stimulus_polarity != d.prev_polarity, 'switch', 'repeat'), index=d.index, dtype='object')
    _bad = d.prev_polarity.isna() | d.stimulus_polarity.isna() | d.prev_polarity.eq('neutral') | d.stimulus_polarity.eq('neutral')
    d.loc[_bad, 'polarity_switch'] = np.nan
    d = add_derived_behavior(d)
    subjects = sorted(d.subject_id.unique())
    if upstream['historical'] is not None:
        write_csv(outroot / 'tables/family_interpretability_extension_historical_mapping_inference_global_tests.csv', upstream['historical'])
    n1_names = ['stimulus_negative_minus_positive', 'context_x_polarity_DID', 'late_minus_early_half', 'context_switch_minus_repeat']
    outcome_model_integrity = np.full((len(subjects), len(n1_names)), np.nan)
    outcome_model_integrity[:, 0] = participant_contrast(d, 'family2', 'stimulus_polarity', 'positive', 'negative', subjects)
    outcome_model_integrity[:, 1] = participant_did(d[d.stimulus_polarity.isin(['positive', 'negative'])], 'family2', 'stimulus_polarity', 'positive', 'negative', 'context', 'imagery', 'video', subjects)
    outcome_model_integrity[:, 2] = participant_contrast(d, 'family2', 'acquisition_half', 'early', 'late', subjects)
    outcome_model_integrity[:, 3] = participant_contrast(d.dropna(subset=['context_switch']), 'family2', 'context_switch', 'repeat', 'switch', subjects)
    n1_obs, n1_raw, n1_global, n1_maxt = joint_signflip_maxt(outcome_model_integrity, a.contrast_permutations, SEED + 11)
    n1_rows = []
    for j, n in enumerate(n1_names):
        n1_rows.append({'test': n, 'n_participants': int(np.isfinite(outcome_model_integrity[:, j]).sum()), 'participant_mean_contrast': signed_mean_from_contrasts(outcome_model_integrity[:, j]), 't_abs': n1_obs[j], 'raw_signflip_p': n1_raw[j], 'maxT_p': n1_maxt[j]})
    write_csv(outroot / 'tables/family_interpretability_extension_N1_binary_task_stimulus_tests.csv', n1_rows)
    contextC = participant_contrast(d, 'family2', 'context', 'imagery', 'video', subjects)
    write_csv(outroot / 'tables/family_interpretability_extension_historical_context_descriptive.csv', [{'contrast': 'video_minus_imagery_family2_fraction', 'n_participants': int(np.isfinite(contextC).sum()), 'participant_mean_contrast': signed_mean_from_contrasts(contextC), 't_abs': t_abs_from_contrasts(contextC), 'status': 'historical_H4_already_formally_tested'}])
    base = lopo_fixed_logistic(d, [])
    full = lopo_fixed_logistic(d, REPORTS)
    delta = float(full['ba'] - base['ba'])
    seeds = (SEED + 1000 + np.arange(a.model_permutations)).tolist()
    if a.model_permutations > 0:
        null = np.array(Parallel(n_jobs=a.jobs, prefer='threads', verbose=0)((delayed(model_perm_once)(d, int(s)) for s in seeds)), float)
        n2_p = float((1 + np.sum(null >= delta)) / (len(null) + 1))
    else:
        null = np.array([], float)
        n2_p = math.nan
    n2_summary = pd.DataFrame([{'baseline_participant_balanced_BA': base['ba'], 'full10_participant_balanced_BA': full['ba'], 'delta_BA': delta, 'baseline_logloss': base['logloss'], 'full10_logloss': full['logloss'], 'model_permutations': len(null), 'permutation_p': n2_p}])
    write_csv(outroot / 'tables/family_interpretability_extension_N2_full10_report_prediction.csv', n2_summary)
    write_csv(outroot / 'tables/family_interpretability_extension_N2_full10_outer_coefficients.csv', full['coefs'])
    write_csv(outroot / 'tables/family_interpretability_extension_N2_null_deltaBA.csv', pd.DataFrame({'replicate': np.arange(len(null)), 'delta_BA': null}))
    atlas = pd.DataFrame()
    if a.subset_atlas == 'full':
        atlas = exhaustive_subset_atlas_ridge(d, base['ba'], ridge_lambda=1.0)
        if len(atlas) != 1023:
            raise RuntimeError('Subset roster != 1023')
        atlas = atlas.sort_values(['delta_ba_vs_nuisance_logistic_baseline', 'subset_size'], ascending=[False, True]).reset_index(drop=True)
        atlas.insert(0, 'rank', np.arange(1, len(atlas) + 1))
        write_csv(outroot / 'tables/family_interpretability_extension_all_1023_report_subset_atlas.csv', atlas)
        topn = min(50, len(atlas))
        top = atlas.head(topn)
        incl = []
        for r in REPORTS:
            flags = top.report_subset.str.split('|').map(lambda xs: r in xs)
            incl.append({'report': r, 'included_in_top50': int(flags.sum()), 'top50_fraction': float(flags.mean())})
        write_csv(outroot / 'tables/family_interpretability_extension_report_subset_top50_inclusion.csv', incl)
    CR = report_family_contrast_matrix(d, REPORTS, subjects)
    r_obs, r_raw, r_glob, r_maxt = joint_signflip_maxt(CR, a.contrast_permutations, SEED + 21)
    report_rows = []
    for j, c in enumerate(REPORTS):
        report_rows.append({'report': c, 'family2_minus_family1_mean': signed_mean_from_contrasts(CR[:, j]), 't_abs': r_obs[j], 'raw_p': r_raw[j], 'maxT_p': r_maxt[j], 'claim_role': 'exploratory_re-expression_only; historical H1 parent was unsupported'})
    write_csv(outroot / 'tables/family_interpretability_extension_original_report_family_contrasts.csv', report_rows)
    CE = extreme_family_contrasts(d, REPORTS, subjects)
    e_obs, e_raw, e_glob, e_maxt = joint_signflip_maxt(CE, a.contrast_permutations, SEED + 22)
    ext_rows = []
    for j, c in enumerate(REPORTS):
        ext_rows.append({'report': c, 'family2_fraction_high_minus_low_tertile': signed_mean_from_contrasts(CE[:, j]), 't_abs': e_obs[j], 'raw_p': e_raw[j], 'maxT_p': e_maxt[j], 'middle_tertile_omitted': True})
    write_csv(outroot / 'tables/family_interpretability_extension_low_high_report_extremes.csv', ext_rows)
    derived = ['component_balance', 'mixed_affect', 'overall_emotion_intensity', 'emotion_entropy', 'top2_margin', 'rating_extremity', 'liking_valence_gap', 'valence_component_discrepancy', 'stimulus_valence_congruence', 'target_emotion_dominance']
    CD = report_family_contrast_matrix(d, derived, subjects)
    d_obs, d_raw, d_global, d_maxt = joint_signflip_maxt(CD, a.contrast_permutations, SEED + 31)
    d_rows = []
    for j, c in enumerate(derived):
        d_rows.append({'derived_metric': c, 'family2_minus_family1_mean': signed_mean_from_contrasts(CD[:, j]), 'n_participants': int(np.isfinite(CD[:, j]).sum()), 't_abs': d_obs[j], 'raw_p': d_raw[j], 'maxT_p': d_maxt[j]})
    write_csv(outroot / 'tables/family_interpretability_extension_N3_nonlinear_report_structure.csv', d_rows)
    n4, trans, nullstay = sequential_persistence(d, a.sequence_permutations, SEED + 41)
    write_csv(outroot / 'tables/family_interpretability_extension_family_sequence_participant_transitions.csv', trans)
    write_csv(outroot / 'tables/family_interpretability_extension_N4_sequence_persistence_null.csv', pd.DataFrame({'replicate': np.arange(len(nullstay)), 'participant_balanced_stay_fraction': nullstay}))
    rel, occ = occupancy_reliability(d, a.sequence_permutations, SEED + 42)
    write_csv(outroot / 'tables/family_interpretability_extension_family_occupancy_by_participant.csv', occ)
    write_csv(outroot / 'tables/family_interpretability_extension_family_occupancy_reliability.csv', rel)
    pc = participant_contrast(d.dropna(subset=['polarity_switch']), 'family2', 'polarity_switch', 'repeat', 'switch', subjects)
    cs = participant_contrast(d.dropna(subset=['context_switch']), 'family2', 'context_switch', 'repeat', 'switch', subjects)
    write_csv(outroot / 'tables/family_interpretability_extension_sequence_switch_descriptives.csv', [{'contrast': 'family2_on_context_switch_minus_repeat', 'mean': signed_mean_from_contrasts(cs), 't_abs': t_abs_from_contrasts(cs), 'n_participants': int(np.isfinite(cs).sum())}, {'contrast': 'family2_on_polarity_switch_minus_repeat', 'mean': signed_mean_from_contrasts(pc), 't_abs': t_abs_from_contrasts(pc), 'n_participants': int(np.isfinite(pc).sum())}])
    globals_raw = {'N1_binary_task_stimulus_structure': float(n1_global), 'N2_full10_report_prediction': float(n2_p), 'N3_nonlinear_report_structure': float(d_global), 'N4_sequence_persistence': float(n4['two_sided_p'])}
    globals_holm = holm(globals_raw)
    global_rows = []
    for k, p in globals_raw.items():
        global_rows.append({'domain': k, 'raw_global_p': p, 'holm_p': globals_holm[k], 'supported_after_Holm': bool(globals_holm[k] < ALPHA)})
    gdf = pd.DataFrame(global_rows)
    write_csv(outroot / 'tables/family_interpretability_extension_global_interpretability_tests.csv', gdf)
    any_supported = bool(gdf.supported_after_Holm.any())
    supported_domains = gdf.loc[gdf.supported_after_Holm, 'domain'].tolist()
    next_phase = 'family_source_decomposition_design' if any_supported else 'outcome_free_family_source_decomposition'
    next_role = 'specification an authoritative Validation/Confirmation behavior-to-EEG event-lineage mapping and replicate only supported family_interpretability_extension interpretations without changing family assignments.' if any_supported else 'specification outcome-free neurophysiological source decomposition of the two trajectory families (feature groups/frequency/topography/epoch contribution), because tested Discovery semantic/behavior explanations remain unsupported.'
    score = []
    N1_parent = globals_holm['N1_binary_task_stimulus_structure'] < ALPHA
    for r in n1_rows:
        score.append({'candidate': r['test'], 'domain': 'N1', 'effect': r['participant_mean_contrast'], 'child_adjusted_p': r['maxT_p'], 'parent_Holm_p': globals_holm['N1_binary_task_stimulus_structure'], 'claim_supported': bool(N1_parent and r['maxT_p'] < ALPHA)})
    score.append({'candidate': 'all_10_reports_combined_unseen_person_prediction', 'domain': 'N2', 'effect': delta, 'child_adjusted_p': n2_p, 'parent_Holm_p': globals_holm['N2_full10_report_prediction'], 'claim_supported': bool(globals_holm['N2_full10_report_prediction'] < ALPHA)})
    N3_parent = globals_holm['N3_nonlinear_report_structure'] < ALPHA
    for r in d_rows:
        score.append({'candidate': r['derived_metric'], 'domain': 'N3', 'effect': r['family2_minus_family1_mean'], 'child_adjusted_p': r['maxT_p'], 'parent_Holm_p': globals_holm['N3_nonlinear_report_structure'], 'claim_supported': bool(N3_parent and r['maxT_p'] < ALPHA)})
    score.append({'candidate': 'adjacent_trial_family_persistence_or_alternation', 'domain': 'N4', 'effect': n4['effect_vs_null'], 'child_adjusted_p': n4['two_sided_p'], 'parent_Holm_p': globals_holm['N4_sequence_persistence'], 'claim_supported': bool(globals_holm['N4_sequence_persistence'] < ALPHA)})
    write_csv(outroot / 'tables/family_interpretability_extension_interpretation_scorecard.csv', score)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'interpretability_extension_complete', 'counts': {'trials': len(d), 'participants': d.subject_id.nunique(), 'family1': int((d.family2 == 0).sum()), 'family2': int((d.family2 == 1).sum()), 'report_subsets_scored': int(len(atlas))}, 'K_selection_context': metrics[['k', 'participant_weighted_silhouette', 'median_ari', 'ari_10th_percentile', 'eligible']].to_dict('records'), 'primary_domains': {row['domain']: {'raw_p': row['raw_global_p'], 'holm_p': row['holm_p'], 'supported': bool(row['supported_after_Holm'])} for _, row in gdf.iterrows()}, 'N2': {'baseline_BA': base['ba'], 'full10_BA': full['ba'], 'delta_BA': delta, 'permutation_p': n2_p}, 'N4': n4, 'supported_domains': supported_domains, 'semantic_relabeling_permitted': False, 'reason_semantic_relabeling_still_forbidden': 'family_interpretability_extension is Discovery-only post-hoc interpretation. Even supported candidates require held-out replication after a new authoritative behavior-event lineage specification.', 'Validation_behavior_semantics_opened': False, 'Confirmation_behavior_semantics_opened': False, 'next_phase': next_phase, 'next_phase_role': next_role}
    write_json(outroot / 'qc/family_interpretability_extension_summary.json', summary)
    write_json(project / 'analysis_specifications/S033_family_interpretability_extension.json', {'phase': PHASE, 'generated_at': now(), 'any_primary_domain_supported_after_Holm': any_supported, 'supported_domains': supported_domains, 'next_phase': next_phase, 'role': next_role, 'semantic_relabeling_permitted': False})
    report = f"# family_interpretability_extension — Family interpretability extension\n\nStatus: **complete**.\n\nThis phase reopens **Discovery-only interpretation**, not family discovery. The two-family EEG solution, all family assignments, and the six-dimensional trajectory geometry remain immutable. Validation and Confirmation behavioral semantics were not opened.\n\n## Why binary hypotheses were tested but not assumed\nThe prespecified K scan showed that K=2 is the selected parsimonious coarse solution, while finer stable candidate resolutions also existed. Therefore binary explanations are useful hypotheses, not proof of an intrinsically binary biological mechanism.\n\n## Primary global domains (Holm across four){gdf.to_markdown(index=False)}\n\n## N2 all-ten-report model\n- nuisance-only participant-balanced BA: **{base['ba']:.6f}**\n- nuisance + all 10 reports BA: **{full['ba']:.6f}**\n- delta BA: **{delta:.6f}**\n- permutation p: **{n2_p:.6g}**\n\n## N4 family-to-family temporal relation\n- observed adjacent-trial stay fraction: **{n4['observed_participant_balanced_stay_fraction']:.6f}**\n- occupancy-preserving null mean: **{n4['null_mean']:.6f}**\n- effect versus null: **{n4['effect_vs_null']:.6f}**\n- two-sided p: **{n4['two_sided_p']:.6g}**\n\n## Exhaustive report combinations\nThe question 'which combination of ten columns?' has **1023** nonempty subsets, not 10! permutations. family_interpretability_extension scores all 1023 with fixed leave-one-participant-out prediction when the full atlas is enabled. These rankings are **descriptive/hypothesis-generating only**; the inferential combination test is the pre-specified all-10D N2 model.\n\n## Claim boundary\nNo Family 1/2 semantic name is licensed by family_interpretability_extension alone. A supported Discovery candidate must first be replicated against the already fixed held-out EEG family assignments using a new authoritative behavior-to-event lineage.\n\n## Next phase\n**{next_phase}** — {next_role}"
    (outroot / 'reports/family_interpretability_extension_report.md').write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'policy_canonical_sha256': policy_sha, 'inputs': {'cross_person_alignment_fit_final_lock_sha256': sha256(project / cross_person_alignment_fit_LOCK), 'mapping_design_final_lock_sha256': sha256(project / mapping_design_LOCK), 'mapping_design_trial_manifest_sha256': sha256(project / mapping_design_TRIAL), 'baseline_evidence_refresh_final_lock_sha256': sha256(project / baseline_evidence_refresh_LOCK)}, 'analysis_role': 'post-specification Discovery-only interpretability extension', 'family_assignments_refit': False, 'K_reselected': False, 'EEG_geometry_refit': False, 'Validation_behavior_semantics_opened': False, 'Confirmation_behavior_semantics_opened': False, 'semantic_relabeling_permitted': False, 'next_phase': next_phase}
    write_json(outroot / 'qc/family_interpretability_extension_contract.json', contract)
    files = [p for p in sorted(outroot.rglob('*')) if p.is_file()]
    external = [policy_path, policy_lock, project / 'analysis_specifications/S033_family_interpretability_extension.json']
    hashes = {str(p.relative_to(project)): sha256(p) for p in files + external}
    final = outroot / 'qc/family_interpretability_extension_final_output_lock_hashes.json'
    write_json(final, {'phase': PHASE, 'generated_at': now(), 'status': 'interpretability_extension_complete', 'hashes': hashes, 'any_primary_domain_supported_after_Holm': any_supported, 'supported_domains': supported_domains, 'semantic_relabeling_permitted': False, 'Validation_behavior_semantics_opened': False, 'Confirmation_behavior_semantics_opened': False, 'next_phase': next_phase})
    rels = [str(p.relative_to(project)) for p in sorted(outroot.rglob('*')) if p.is_file()] + [str(x.relative_to(project)) for x in external]
    hz = project / '' / f'family_interpretability_extension_family_interpretability_extension_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    h = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{h}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] family_interpretability_extension COMPLETE')
    print('Primary global tests:')
    for _, r in gdf.iterrows():
        print(f'  {r.domain}: raw p={r.raw_global_p:.6g}; Holm p={r.holm_p:.6g}; supported={bool(r.supported_after_Holm)}')
    print(f"N2 baseline BA={base['ba']:.6f}; full10 BA={full['ba']:.6f}; delta={delta:.6f}; p={n2_p:.6g}")
    print(f"N4 stay={n4['observed_participant_balanced_stay_fraction']:.6f}; null={n4['null_mean']:.6f}; p={n4['two_sided_p']:.6g}")
    print('Report subsets scored:', len(atlas))
    print('Semantic relabeling permitted: False')
    print('Next phase:', next_phase)
    print('', hz)
    print('SHA-256:', h)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())