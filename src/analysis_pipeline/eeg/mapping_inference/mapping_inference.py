# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase mapping_inference: execute the prespecified Discovery-only mapping inference.

This phase verifies every mapping_design fixed input, writes an outcome-blind
implementation-resolution lock for mathematical details that mapping_design named but
did not serialize fully, then executes all six prespecified global hypotheses H1-H6.
It never refits cross_person_alignment_fit alignment, changes family/state assignments, renames
families, or opens Validation/Confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'mapping_inference'
SOURCE_PHASE = 'mapping_design'
NEXT_PHASE = 'discovery_synthesis'
SEED = 540
EXPECTED_MAPPING_POLICY_SHA = '9e277b8127e38268480b56ee1e89e8e7907076aea7b479e28aa203e714ed97fe'
PERM_GLOBAL = 10000
PERM_CV = 1000
BOOTSTRAP_REPEATS = 5000
ALPHA_LEVEL = 0.05
RIDGE_GRID = np.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
H3_FIXED_ALPHA = 1.0
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
GEOMETRY = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']
STATE_PCS = [f'aligned_state_pc{i}' for i in range(1, 9)]

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def require(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Missing prerequisite(s):\n  ' + '\n  '.join(missing))

def bval(x: Any) -> bool:
    return bool(x) and str(x).strip().lower() not in {'false', '0', 'none', 'nan', ''}

def participant_weights(subjects: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects).astype(str)
    unique, counts = np.unique(subjects, return_counts=True)
    lookup = dict(zip(unique, counts))
    return np.array([1.0 / lookup[s] for s in subjects], dtype=float)

def weighted_mean(x: np.ndarray, w: np.ndarray, axis: int=0) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return np.sum(x * np.expand_dims(w, tuple(range(1, x.ndim))), axis=axis) / np.sum(w)

def weighted_lstsq(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.asarray(w, dtype=float))[:, None]
    return np.linalg.lstsq(X * sw, Y * sw, rcond=None)[0]

def fit_predict_ridge(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, wtr: np.ndarray, alpha: float) -> np.ndarray:
    Xtr = np.asarray(Xtr, float)
    Ytr = np.asarray(Ytr, float)
    Xte = np.asarray(Xte, float)
    wtr = np.asarray(wtr, float)
    wsum = wtr.sum()
    xmu = (wtr[:, None] * Xtr).sum(0) / wsum
    xvar = (wtr[:, None] * (Xtr - xmu) ** 2).sum(0) / wsum
    xsd = np.sqrt(np.maximum(xvar, 1e-12))
    Ztr = (Xtr - xmu) / xsd
    Zte = (Xte - xmu) / xsd
    ymu = (wtr[:, None] * Ytr).sum(0) / wsum
    Yc = Ytr - ymu
    G = Ztr.T @ (wtr[:, None] * Ztr) + float(alpha) * np.eye(Ztr.shape[1])
    rhs = Ztr.T @ (wtr[:, None] * Yc)
    coef = np.linalg.solve(G, rhs)
    return Zte @ coef + ymu

def metric_delta_r2(Y: np.ndarray, Pb: np.ndarray, Pf: np.ndarray, subjects: np.ndarray, sample_weights: np.ndarray | None=None) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    Y = np.asarray(Y, float)
    Pb = np.asarray(Pb, float)
    Pf = np.asarray(Pf, float)
    if sample_weights is None:
        w = participant_weights(subjects)
    else:
        w = np.asarray(sample_weights, float)
    mu = (w[:, None] * Y).sum(0) / w.sum()
    sst = (w[:, None] * (Y - mu) ** 2).sum(0)
    sseb = (w[:, None] * (Y - Pb) ** 2).sum(0)
    ssef = (w[:, None] * (Y - Pf) ** 2).sum(0)
    d = (sseb - ssef) / np.maximum(sst, 1e-15)
    global_d = float((sseb.sum() - ssef.sum()) / max(sst.sum(), 1e-15))
    return (global_d, d, {'sst': sst, 'sse_baseline': sseb, 'sse_full': ssef})

def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out

def percentile_ci(x: np.ndarray) -> tuple[float, float]:
    return (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))

def perm_p(obs: float, null: np.ndarray, two_sided: bool=False) -> float:
    arr = np.asarray(null, float)
    if two_sided:
        return float((1 + np.sum(np.abs(arr) >= abs(obs))) / (len(arr) + 1))
    return float((1 + np.sum(arr >= obs)) / (len(arr) + 1))

def nuisance_design(df: pd.DataFrame, training_subjects: list[str] | None=None, include_subject_fe: bool=True) -> tuple[np.ndarray, list[str]]:
    n = len(df)
    cols = [np.ones(n)]
    names = ['intercept']
    contexts = sorted(df['context'].astype(str).unique().tolist())
    for cat in contexts[1:]:
        cols.append((df['context'].astype(str).to_numpy() == cat).astype(float))
        names.append(f'context={cat}')
    a = df['acquisition_fraction'].to_numpy(float)
    cols.extend([a, a * a])
    names.extend(['acquisition_fraction', 'acquisition_fraction_squared'])
    if include_subject_fe:
        if training_subjects is None:
            subs = sorted(df['subject_id'].astype(str).unique().tolist())
        else:
            subs = sorted(training_subjects)
        for sub in subs[1:]:
            cols.append((df['subject_id'].astype(str).to_numpy() == sub).astype(float))
            names.append(f'subject={sub}')
    return (np.column_stack(cols), names)

def residualize_h1(trial: pd.DataFrame) -> np.ndarray:
    X, _ = nuisance_design(trial, include_subject_fe=True)
    Y = trial[REPORTS].to_numpy(float)
    w = participant_weights(trial['subject_id'].to_numpy())
    beta = weighted_lstsq(X, Y, w)
    return Y - X @ beta

def h1_stat(D: np.ndarray) -> float:
    D = np.asarray(D, float)
    n = len(D)
    m = D.mean(0)
    cov = LedoitWolf(assume_centered=False).fit(D).covariance_
    return float(n * (m @ np.linalg.pinv(cov, rcond=1e-12) @ m))

def h1_observed(trial: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, pd.DataFrame]:
    adj = residualize_h1(trial)
    work = trial[['subject_id', 'trajectory_family_1based']].copy()
    raw = trial[REPORTS].to_numpy(float)
    rows = []
    D = []
    Draw = []
    for sub in sorted(trial.subject_id.unique()):
        idx = np.where(trial.subject_id.to_numpy() == sub)[0]
        fam = trial.iloc[idx]['trajectory_family_1based'].to_numpy(int)
        da = adj[idx][fam == 2].mean(0) - adj[idx][fam == 1].mean(0)
        dr = raw[idx][fam == 2].mean(0) - raw[idx][fam == 1].mean(0)
        D.append(da)
        Draw.append(dr)
        row = {'subject_id': sub, 'n_family1': int((fam == 1).sum()), 'n_family2': int((fam == 2).sum())}
        row.update({f'adjusted_contrast_{r}': float(v) for r, v in zip(REPORTS, da)})
        row.update({f'raw_contrast_{r}': float(v) for r, v in zip(REPORTS, dr)})
        rows.append(row)
    D = np.asarray(D)
    Draw = np.asarray(Draw)
    stat = h1_stat(D)
    desc = []
    for fam in [1, 2]:
        for r in REPORTS:
            vals = []
            for sub in sorted(trial.subject_id.unique()):
                z = trial[(trial.subject_id == sub) & (trial.trajectory_family_1based == fam)][r].mean()
                vals.append(z)
            desc.append({'trajectory_family_1based': fam, 'report': r, 'participant_balanced_mean': float(np.mean(vals)), 'participant_sd': float(np.std(vals, ddof=1))})
    return (pd.DataFrame(rows), D, Draw, stat, pd.DataFrame(desc))

def make_context_design(trial: pd.DataFrame) -> np.ndarray:
    ctx = trial['context'].astype(str).to_numpy()
    cats = sorted(np.unique(ctx).tolist())
    dummies = [(ctx == c).astype(float) for c in cats[1:]]
    a = trial['acquisition_fraction'].to_numpy(float)
    return np.column_stack(dummies + [a, a * a])

def cv_predict_h2(Y: np.ndarray, Xb: np.ndarray, Xf: np.ndarray, subjects: np.ndarray, folds: np.ndarray, return_selection: bool=False) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(Y)
    pb = np.zeros_like(Y)
    pf = np.zeros_like(Y)
    selections = []
    for outer in sorted(np.unique(folds)):
        te = np.where(folds == outer)[0]
        tr = np.where(folds != outer)[0]
        inner_folds = sorted(np.unique(folds[tr]))
        chosen = []
        for label, X in [('baseline', Xb), ('full', Xf)]:
            scores = []
            for alpha in RIDGE_GRID:
                pred = np.zeros((len(tr), Y.shape[1]))
                filled = np.zeros(len(tr), bool)
                for inn in inner_folds:
                    iv = np.where(folds[tr] == inn)[0]
                    it = np.where(folds[tr] != inn)[0]
                    trtrain = tr[it]
                    trval = tr[iv]
                    w = participant_weights(subjects[trtrain])
                    pred[iv] = fit_predict_ridge(X[trtrain], Y[trtrain], X[trval], w, float(alpha))
                    filled[iv] = True
                if not filled.all():
                    raise RuntimeError('H2 inner predictions incomplete')
                wv = participant_weights(subjects[tr])
                sse = float((wv[:, None] * (Y[tr] - pred) ** 2).sum())
                scores.append(sse)
            ai = int(np.argmin(scores))
            alpha = float(RIDGE_GRID[ai])
            chosen.append(alpha)
            if return_selection:
                selections.append({'outer_cross_person_fold': int(outer), 'model': label, 'selected_alpha': alpha, 'inner_participant_balanced_sse': float(scores[ai])})
        w = participant_weights(subjects[tr])
        pb[te] = fit_predict_ridge(Xb[tr], Y[tr], Xb[te], w, chosen[0])
        pf[te] = fit_predict_ridge(Xf[tr], Y[tr], Xf[te], w, chosen[1])
    return (pb, pf, selections)

def permute_rows_within_subject(Y: np.ndarray, subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = Y.copy()
    for sub in np.unique(subjects):
        idx = np.where(subjects == sub)[0]
        out[idx] = Y[rng.permutation(idx)]
    return out

def bootstrap_prediction_metrics(Y: np.ndarray, Pb: np.ndarray, Pf: np.ndarray, subjects: np.ndarray, repeats: int, seed: int, sample_weights: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    subs = sorted(np.unique(subjects).tolist())
    blocks = []
    for s in subs:
        idx = np.where(subjects == s)[0]
        if sample_weights is None:
            w = np.ones(len(idx)) / len(idx)
        else:
            w = np.asarray(sample_weights)[idx]
            w = w / w.sum()
        y = Y[idx]
        b = Pb[idx]
        f = Pf[idx]
        blocks.append({'mean_y': (w[:, None] * y).sum(0), 'mean_y2': (w[:, None] * y * y).sum(0), 'mse_b': (w[:, None] * (y - b) ** 2).sum(0), 'mse_f': (w[:, None] * (y - f) ** 2).sum(0)})
    rng = np.random.default_rng(seed)
    G = np.zeros(repeats)
    D = np.zeros((repeats, Y.shape[1]))
    for r in range(repeats):
        pick = rng.integers(0, len(blocks), len(blocks))
        my = np.array([blocks[i]['mean_y'] for i in pick])
        my2 = np.array([blocks[i]['mean_y2'] for i in pick])
        mb = np.array([blocks[i]['mse_b'] for i in pick])
        mf = np.array([blocks[i]['mse_f'] for i in pick])
        mu = my.mean(0)
        sst = np.sum(my2 - 2 * mu * my + mu * mu, axis=0)
        sb = mb.sum(0)
        sf = mf.sum(0)
        D[r] = (sb - sf) / np.maximum(sst, 1e-15)
        G[r] = (sb.sum() - sf.sum()) / max(sst.sum(), 1e-15)
    return (G, D)

def build_state_profiles_for_outer(state: pd.DataFrame, test_subject: str) -> pd.DataFrame:
    train_subjects = sorted([s for s in state.subject_id.unique() if s != test_subject])
    train_mask = state.subject_id.isin(train_subjects).to_numpy()
    X, _ = nuisance_design(state, training_subjects=train_subjects, include_subject_fe=True)
    Y = state[REPORTS].to_numpy(float)
    base = state['trial_epoch_weight'].to_numpy(float)
    denom = state.groupby('subject_id')['trial_epoch_weight'].transform('sum').to_numpy(float)
    w = base / denom
    beta = weighted_lstsq(X[train_mask], Y[train_mask], w[train_mask])
    resid = Y - X @ beta
    rows = []
    for obj, g in state.groupby('state_object_id', sort=True):
        idx = g.index.to_numpy()
        ww = state.loc[idx, 'trial_epoch_weight'].to_numpy(float)
        ww = ww / ww.sum()
        row = {'state_object_id': obj, 'subject_id': g.subject_id.iloc[0], 'state_alignment_weight': float(g.state_alignment_weight.iloc[0])}
        for pc in STATE_PCS:
            row[pc] = float(g[pc].iloc[0])
        prof = (ww[:, None] * resid[idx]).sum(0)
        for r, v in zip(REPORTS, prof):
            row[r] = float(v)
        rows.append(row)
    prof = pd.DataFrame(rows)
    for sub, idx in prof.groupby('subject_id').groups.items():
        ii = np.array(list(idx))
        ww = prof.loc[ii, 'state_alignment_weight'].to_numpy(float)
        ww = ww / ww.sum()
        vals = prof.loc[ii, REPORTS].to_numpy(float)
        cen = (ww[:, None] * vals).sum(0)
        prof.loc[ii, REPORTS] = vals - cen
    return prof.sort_values(['subject_id', 'state_object_id']).reset_index(drop=True)

def h3_observed_and_profiles(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(state.subject_id.unique().tolist())
    pred_rows = []
    profile_rows = []
    for test in subjects:
        prof = build_state_profiles_for_outer(state, test)
        profile_rows.append(prof.assign(outer_test_subject=test))
        tr = prof.subject_id.to_numpy() != test
        te = ~tr
        X = prof[STATE_PCS].to_numpy(float)
        Y = prof[REPORTS].to_numpy(float)
        w = prof.loc[tr, 'state_alignment_weight'].to_numpy(float)
        P = fit_predict_ridge(X[tr], Y[tr], X[te], w, H3_FIXED_ALPHA)
        t = prof.loc[te, ['state_object_id', 'subject_id', 'state_alignment_weight', *STATE_PCS, *REPORTS]].copy()
        for j, r in enumerate(REPORTS):
            t[f'predicted_{r}'] = P[:, j]
        pred_rows.append(t)
    return (pd.concat(pred_rows, ignore_index=True), pd.concat(profile_rows, ignore_index=True))

def h3_null_once(profiles_by_outer: dict[str, pd.DataFrame], perm_maps: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    obs = []
    pred = []
    subs = []
    weights = []
    for test, prof0 in profiles_by_outer.items():
        prof = prof0.copy()
        Y = prof[REPORTS].to_numpy(float)
        for sub in prof.subject_id.unique():
            idx = np.where(prof.subject_id.to_numpy() == sub)[0]
            order = perm_maps[sub]
            Y[idx] = Y[idx[order]]
        tr = prof.subject_id.to_numpy() != test
        te = ~tr
        X = prof[STATE_PCS].to_numpy(float)
        P = fit_predict_ridge(X[tr], Y[tr], X[te], prof.loc[tr, 'state_alignment_weight'].to_numpy(float), H3_FIXED_ALPHA)
        obs.append(Y[te])
        pred.append(P)
        subs.extend(prof.loc[te, 'subject_id'].tolist())
        weights.extend(prof.loc[te, 'state_alignment_weight'].tolist())
    Y = np.vstack(obs)
    P = np.vstack(pred)
    B = np.zeros_like(Y)
    return metric_delta_r2(Y, B, P, np.array(subs), np.array(weights))[:2]

def categorical_stats(df: pd.DataFrame, target: str) -> tuple[float, float, pd.DataFrame]:
    d = df[['subject_id', 'trajectory_family_1based', target]].dropna().copy()
    fam = d.trajectory_family_1based.to_numpy(int) - 1
    cats = sorted(d[target].astype(str).unique().tolist())
    cmap = {c: i for i, c in enumerate(cats)}
    tar = np.array([cmap[x] for x in d[target].astype(str)], int)
    w = participant_weights(d.subject_id.to_numpy())
    counts = np.bincount(fam * len(cats) + tar, weights=w, minlength=2 * len(cats)).reshape(2, len(cats))
    total = counts.sum()
    p = counts / total
    px = p.sum(1)
    py = p.sum(0)
    mi = 0.0
    for i in range(2):
        for j in range(len(cats)):
            if p[i, j] > 0:
                mi += p[i, j] * math.log(p[i, j] / (px[i] * py[j]))
    hx = -sum((v * math.log(v) for v in px if v > 0))
    hy = -sum((v * math.log(v) for v in py if v > 0))
    nmi = float(mi / math.sqrt(hx * hy)) if hx > 0 and hy > 0 else 0.0
    expected = np.outer(counts.sum(1), counts.sum(0)) / total
    chi = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-15)))
    v = float(math.sqrt(chi / (total * min(1, len(cats) - 1)))) if len(cats) > 1 else 0.0
    rows = []
    for i in range(2):
        for j, c in enumerate(cats):
            rows.append({'trajectory_family_1based': i + 1, 'target_level': c, 'participant_balanced_count': float(counts[i, j]), 'participant_balanced_proportion_within_family': float(counts[i, j] / counts[i].sum())})
    return (nmi, v, pd.DataFrame(rows))

def categorical_null_once(df: pd.DataFrame, target: str, rng: np.random.Generator) -> tuple[float, float]:
    d = df[['subject_id', 'trajectory_family_1based', target]].dropna().copy()
    fam = d.trajectory_family_1based.to_numpy(int).copy()
    subs = d.subject_id.to_numpy()
    for s in np.unique(subs):
        idx = np.where(subs == s)[0]
        fam[idx] = fam[rng.permutation(idx)]
    d['trajectory_family_1based'] = fam
    a, b, _ = categorical_stats(d, target)
    return (a, b)

def run_checkpointed(name: str, repeats: int, width: int, cache_dir: Path, key: dict[str, Any], fn: Callable[[int], np.ndarray], note: Callable[[str], None], chunk: int=100) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f'{name}.npz'
    meta = cache_dir / f'{name}.json'
    arr = np.full((repeats, width), np.nan)
    completed = 0
    if npz.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old.get('key') == key and old.get('repeats') == repeats and (old.get('width') == width):
            z = np.load(npz)
            prior = z['stats']
            completed = int(z['completed'])
            if prior.shape == arr.shape and 0 <= completed <= repeats:
                arr[:completed] = prior[:completed]
                note(f'Resuming {name} at replicate {completed}/{repeats}.')
    for i in range(completed, repeats):
        arr[i] = fn(i)
        if (i + 1) % chunk == 0 or i + 1 == repeats:
            tmp = npz.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, stats=arr, completed=i + 1)
            os.replace(tmp, npz)
            atomic_json(meta, {'key': key, 'repeats': repeats, 'width': width, 'completed': i + 1})
            note(f'{name}: {i + 1}/{repeats}')
    return arr

def input_paths(project: Path) -> dict[str, Path]:
    return {'contract': project / 'config/mapping_design_mapping_design_contract.json', 'final_lock': project / 'outputs/qc/mapping_design_final_output_lock_hashes.json', 'policy': project / 'analysis_specifications/S041_mapping.json', 'next_policy': project / 'analysis_specifications/S040_mapping_inference.json', 'policy_lock': project / 'outputs/qc/mapping_design_policy_lock_hashes.json', 'cross_person_alignment_fit_rehash': project / 'outputs/qc/mapping_design_cross_person_alignment_fit_final_rehash.csv', 'source_audit': project / 'outputs/qc/mapping_design_source_attachment_audit.csv', 'variable_schema': project / 'outputs/tables/mapping_design_mapping_variable_schema.csv', 'trial': project / 'outputs/tables/mapping_design_trial_mapping_manifest.csv', 'state': project / 'outputs/tables/mapping_design_state_mapping_manifest.csv', 'summary': project / 'outputs/qc/mapping_design_mapping_design_specification_summary.json', 'report': project / 'outputs/reports/mapping_design_mapping_design_specification_report.md'}

def output_paths(project: Path) -> dict[str, Path]:
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    c = project / 'config'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'log': l / 'mapping_inference_mapping_inference.log', 'input_rehash': q / 'mapping_inference_input_rehash.csv', 'implementation': c / 'mapping_inference_fixed_implementation_resolution.json', 'implementation_lock': q / 'mapping_inference_implementation_lock.json', 'h1_contrasts': t / 'mapping_inference_h1_participant_family_contrasts.csv', 'h1_descriptives': t / 'mapping_inference_h1_raw_family_descriptives.csv', 'h1_null': t / 'mapping_inference_h1_null_replicates.csv', 'h1_boot': t / 'mapping_inference_h1_bootstrap_summary.csv', 'h2_predictions': t / 'mapping_inference_h2_oof_predictions.csv', 'h2_selection': t / 'mapping_inference_h2_outer_alpha_selection.csv', 'h2_null': t / 'mapping_inference_h2_null_replicates.csv', 'h2_boot': t / 'mapping_inference_h2_bootstrap_summary.csv', 'h3_profiles': t / 'mapping_inference_h3_oof_state_profiles_predictions.csv', 'h3_null': t / 'mapping_inference_h3_null_replicates.csv', 'h3_boot': t / 'mapping_inference_h3_bootstrap_summary.csv', 'categorical_tests': t / 'mapping_inference_h4_h6_categorical_tests.csv', 'categorical_crosstabs': t / 'mapping_inference_h4_h6_descriptive_crosstabs.csv', 'categorical_null': t / 'mapping_inference_h4_h6_null_replicates.csv', 'global_tests': t / 'mapping_inference_global_hypothesis_tests.csv', 'report_followups': t / 'mapping_inference_report_dimension_followups.csv', 'summary': q / 'mapping_inference_mapping_inference_summary.json', 'contract': c / 'mapping_inference_mapping_inference_contract.json', 'next_policy': project / 'analysis_specifications/S042_mapping_inference_discovery_synthesis.json', 'final_lock': q / 'mapping_inference_final_output_lock_hashes.json', 'report': r / 'mapping_inference_mapping_inference_report.md'}
IMPLEMENTATION_SPEC = {'phase': PHASE, 'fixed_before_association_calculation': True, 'h1_covariance': 'Ledoit-Wolf shrinkage covariance across the 31 participant contrast vectors; T=n*m^T*Sigma^-1*m', 'h1_child_statistic': 'absolute mean participant contrast with the same joint sign-flip replicates', 'h2_ridge_grid': RIDGE_GRID.tolist(), 'h2_alpha_selection': 'separate baseline/full alpha selected by participant-balanced inner cross-person-fold SSE inside each outer training set', 'h2_global_delta_r2': 'sum(SSE_baseline-SSE_full) across reports divided by summed participant-balanced SST', 'h3_ridge_alpha_resolution': {'alpha': H3_FIXED_ALPHA, 'reason': 'mapping_design froze ridge and its grid but did not serialize an H3 tuning rule; alpha=1.0 is the central neutral grid value fixed outcome-blind before H3 calculation'}, 'h3_null': 'complete 10D state profile vectors permuted among local states within participant; fixed alpha; full LOPO refit each replicate', 'participant_balance': 'trial analyses give each participant total weight one; state analyses use fixed state_alignment_weight summing to one per participant', 'nmi': 'MI/sqrt(H_family*H_target) from participant-balanced weighted contingency tables using natural logarithms', 'cramers_v': 'weighted chi-square with total weight equal to number of represented participants', 'permutation_p': 'plus-one upper-tail p=(1+count(null>=observed))/(B+1); child H1 uses absolute statistic', 'bootstrap': 'percentile 95% participant bootstrap with participant resampling and equal weight per sampled participant copy', 'multiplicity': 'Holm across H1-H6; Holm across ten children within each H1/H2/H3 parent, with parent gate'}
IMPLEMENTATION_SHA = canonical_sha(IMPLEMENTATION_SPEC)

def verify_inputs(inputs: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    require(inputs.values())
    contract = json.loads(inputs['contract'].read_text())
    lock = json.loads(inputs['final_lock'].read_text())
    policy = json.loads(inputs['policy'].read_text())
    nxt = json.loads(inputs['next_policy'].read_text())
    summary = json.loads(inputs['summary'].read_text())
    if contract.get('phase') != SOURCE_PHASE or nxt.get('next_phase') != PHASE or (not bval(nxt.get('next_phase_permitted'))):
        raise RuntimeError('mapping_design did not authorize mapping_inference')
    if canonical_sha(policy) != EXPECTED_MAPPING_POLICY_SHA or contract.get('mapping_policy_sha256') != EXPECTED_MAPPING_POLICY_SHA:
        raise RuntimeError('prespecified mapping-policy canonical SHA mismatch')
    if bval(lock.get('associations_calculated')) or bval(lock.get('validation_opened')) or bval(lock.get('confirmation_opened')):
        raise RuntimeError('mapping_design boundary violation')
    mapkeys = {'mapping_policy': 'policy', 'policy_lock': 'policy_lock', 'cross_person_alignment_fit_rehash': 'cross_person_alignment_fit_rehash', 'source_audit': 'source_audit', 'variable_schema': 'variable_schema', 'trial_manifest': 'trial', 'state_manifest': 'state', 'summary': 'summary', 'contract': 'contract', 'next_policy': 'next_policy', 'report': 'report'}
    rows = []
    observed = {}
    for k, pk in mapkeys.items():
        exp = lock.get('hashes', {}).get(k)
        obs = sha256(inputs[pk])
        observed[pk] = obs
        rows.append({'object': k, 'path': str(inputs[pk]), 'expected_sha256': exp, 'observed_sha256': obs, 'match': exp == obs})
        if not exp or exp != obs:
            raise RuntimeError(f'mapping_design hash mismatch: {k}')
    if summary.get('counts', {}).get('trial_rows') != 1237 or summary.get('counts', {}).get('local_state_objects') != 40:
        raise RuntimeError('mapping_design count boundary mismatch')
    return (pd.DataFrame(rows), observed, summary)

def validate_manifests(trial: pd.DataFrame, state: pd.DataFrame) -> None:
    reqt = ['subject_id', 'raw_behavior_row_id', 'cross_person_fold', 'acquisition_fraction', 'context', 'stimulus_id', 'emotion_category', 'trajectory_family_1based', *REPORTS, *GEOMETRY]
    reqs = ['state_object_id', 'subject_id', 'state_alignment_weight', 'trial_epoch_weight', 'acquisition_fraction', 'context', *REPORTS, *STATE_PCS]
    for c in reqt:
        if c not in trial:
            raise RuntimeError(f'Missing trial column {c}')
    for c in reqs:
        if c not in state:
            raise RuntimeError(f'Missing state column {c}')
    if len(trial) != 1237 or trial.subject_id.nunique() != 31 or set(trial.trajectory_family_1based.unique()) != {1, 2}:
        raise RuntimeError('Trial manifest boundary mismatch')
    if trial[REPORTS + GEOMETRY + ['acquisition_fraction']].isna().any().any():
        raise RuntimeError('Unexpected numeric missingness in trial manifest')
    if len(state) != 2319 or state.subject_id.nunique() != 19 or state.state_object_id.nunique() != 40:
        raise RuntimeError('State manifest boundary mismatch')
    if state[REPORTS + STATE_PCS + ['state_alignment_weight', 'trial_epoch_weight', 'acquisition_fraction']].isna().any().any():
        raise RuntimeError('Unexpected numeric missingness in state manifest')
    if int(trial.stimulus_id.isna().sum()) != 21 or int(trial.emotion_category.isna().sum()) != 21:
        raise RuntimeError('prespecified stimulus missingness mismatch')
    if not (trial.groupby('subject_id').trajectory_family_1based.nunique() == 2).all():
        raise RuntimeError('A participant lacks one trajectory family')
    sums = state.groupby('subject_id').state_alignment_weight.first() if False else state[['subject_id', 'state_object_id', 'state_alignment_weight']].drop_duplicates().groupby('subject_id').state_alignment_weight.sum()
    if not np.allclose(sums.to_numpy(), 1.0):
        raise RuntimeError('State alignment weights do not sum to one per participant')

def write_report(path: Path, summary: dict[str, Any], global_df: pd.DataFrame) -> None:
    lines = [f'# Phase {PHASE} — prespecified Discovery mapping inference', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f'- Mapping-policy SHA-256: `{EXPECTED_MAPPING_POLICY_SHA}`', f'- Implementation-resolution SHA-256: `{IMPLEMENTATION_SHA}`', '- Validation/Confirmation opened: **no**', '', '## Global results', '']
    for _, r in global_df.iterrows():
        lines.append(f'- {r.hypothesis}: {r.statistic_name} = {r.observed_statistic:.6g}; raw p = {r.raw_p:.6g}; Holm p = {r.holm_p:.6g}; support = {bool(r.global_supported)}.')
    lines += ['', '## Guardrails', '', '- Family labels remain geometric family 1 and family 2.', '- No cross_person_alignment_fit alignment, family count, state count, or assignment was refit.', '- These are post-hoc Discovery mappings, not independent validation.', '- Validation and Confirmation remain closed.', '']
    atomic_text(path, ''.join(lines))

def self_test() -> int:
    assert len(REPORTS) == 10 and len(GEOMETRY) == 6 and (len(STATE_PCS) == 8)
    assert IMPLEMENTATION_SHA == canonical_sha(IMPLEMENTATION_SPEC)
    p = np.array([0.01, 0.04, 0.03])
    a = holm_adjust(p)
    assert np.all((a >= p) & (a <= 1))
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.column_stack([np.arange(6), np.arange(6) ** 2])
    w = np.ones(6)
    P = fit_predict_ridge(X, Y, X, w, 1.0)
    assert P.shape == Y.shape and np.isfinite(P).all()
    d = pd.DataFrame({'subject_id': ['a'] * 4 + ['b'] * 4, 'trajectory_family_1based': [1, 1, 2, 2] * 2, 'context': ['x', 'y', 'x', 'y'] * 2})
    nmi, v, _ = categorical_stats(d, 'context')
    assert 0 <= nmi <= 1 + 1e-09 and 0 <= v <= 1 + 1e-09
    print('mapping_inference self-test: PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--perm-global', type=int, default=PERM_GLOBAL)
    ap.add_argument('--perm-cv', type=int, default=PERM_CV)
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP_REPEATS)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.test_mode and (args.perm_global != PERM_GLOBAL or args.perm_cv != PERM_CV or args.bootstrap != BOOTSTRAP_REPEATS):
        raise RuntimeError('prespecified execution requires 10000 global permutations, 1000 CV permutations, and 5000 bootstraps')
    project = args.project.resolve()
    inputs = input_paths(project)
    outputs = output_paths(project)
    if outputs['contract'].exists() and (not args.force):
        raise RuntimeError(f"Output exists; rerun with --force: {outputs['contract']}")
    log = []

    def note(msg: str) -> None:
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        log.append(line)
        atomic_text(outputs['log'], ''.join(log) + '')
    note('Verifying mapping_design authorization and every fixed output hash before analysis.')
    rehash, input_hashes, source_summary = verify_inputs(inputs)
    atomic_csv(outputs['input_rehash'], rehash)
    note('Writing and locking outcome-blind mathematical implementation resolutions before calculating associations.')
    atomic_json(outputs['implementation'], IMPLEMENTATION_SPEC)
    atomic_json(outputs['implementation_lock'], {'generated_at': now(), 'phase': PHASE, 'canonical_sha256': IMPLEMENTATION_SHA, 'file_sha256': sha256(outputs['implementation']), 'associations_calculated_before_lock': False})
    trial = pd.read_csv(inputs['trial'])
    state = pd.read_csv(inputs['state'])
    validate_manifests(trial, state)
    cache = project / 'outputs/cache/mapping_inference_mapping_inference'
    script_hash = sha256(Path(__file__).resolve())
    keybase = {'script_sha256': script_hash, 'mapping_policy_sha256': EXPECTED_MAPPING_POLICY_SHA, 'implementation_sha256': IMPLEMENTATION_SHA, 'trial_sha256': input_hashes['trial'], 'state_sha256': input_hashes['state']}
    note('H1: calculating participant-specific family contrasts and the prespecified multivariate sign-flip test.')
    h1contr, D, Draw, h1obs, h1desc = h1_observed(trial)
    atomic_csv(outputs['h1_contrasts'], h1contr)
    atomic_csv(outputs['h1_descriptives'], h1desc)

    def h1rep(i: int) -> np.ndarray:
        rng = np.random.default_rng(SEED + 100000 + i)
        signs = rng.choice([-1.0, 1.0], size=len(D))
        Z = D * signs[:, None]
        return np.r_[h1_stat(Z), Z.mean(0)]
    h1null = run_checkpointed('h1_null', args.perm_global, 11, cache, {**keybase, 'test': 'H1'}, h1rep, note, 250)
    h1p = perm_p(h1obs, h1null[:, 0])
    h1child = np.array([perm_p(D[:, j].mean(), h1null[:, j + 1], two_sided=True) for j in range(10)])
    rng = np.random.default_rng(SEED + 110000)
    h1boot = np.zeros((args.bootstrap, 21))
    for b in range(args.bootstrap):
        pick = rng.integers(0, len(D), len(D))
        Z = D[pick]
        m = Z.mean(0)
        sd = Z.std(0, ddof=1)
        z = m / np.where(sd > 1e-12, sd, np.nan)
        h1boot[b, :10] = m
        h1boot[b, 10:20] = z
        h1boot[b, 20] = np.sqrt(np.nansum(z * z))
    h1bootrows = []
    for j, r in enumerate(REPORTS):
        lo, hi = percentile_ci(h1boot[:, j])
        zlo, zhi = percentile_ci(h1boot[:, 10 + j])
        h1bootrows.append({'quantity': 'report_contrast', 'report': r, 'observed': float(D[:, j].mean()), 'ci_low': lo, 'ci_high': hi, 'standardized_observed': float(D[:, j].mean() / D[:, j].std(ddof=1)), 'standardized_ci_low': zlo, 'standardized_ci_high': zhi})
    lo, hi = percentile_ci(h1boot[:, 20])
    h1bootrows.append({'quantity': 'global_standardized_norm', 'report': 'ALL', 'observed': float(np.sqrt(np.nansum((D.mean(0) / D.std(0, ddof=1)) ** 2))), 'ci_low': lo, 'ci_high': hi, 'standardized_observed': np.nan, 'standardized_ci_low': np.nan, 'standardized_ci_high': np.nan})
    atomic_csv(outputs['h1_boot'], pd.DataFrame(h1bootrows))
    atomic_csv(outputs['h1_null'], pd.DataFrame(h1null, columns=['global_T', *[f'mean_contrast_{r}' for r in REPORTS]]).assign(replicate=np.arange(1, args.perm_global + 1)))
    note('H2: nested cross-person ridge mapping and within-participant report-vector permutations.')
    Y = trial[REPORTS].to_numpy(float)
    subjects = trial.subject_id.to_numpy(str)
    folds = trial.cross_person_fold.to_numpy(int)
    Xb = make_context_design(trial)
    Xf = np.column_stack([Xb, trial[GEOMETRY].to_numpy(float)])
    h2pb, h2pf, h2sel = cv_predict_h2(Y, Xb, Xf, subjects, folds, True)
    h2obs, h2dim, _ = metric_delta_r2(Y, h2pb, h2pf, subjects)
    pred = trial[['subject_id', 'raw_behavior_row_id', 'cross_person_fold']].copy()
    for j, r in enumerate(REPORTS):
        pred[f'observed_{r}'] = Y[:, j]
        pred[f'baseline_predicted_{r}'] = h2pb[:, j]
        pred[f'full_predicted_{r}'] = h2pf[:, j]
    atomic_csv(outputs['h2_predictions'], pred)
    atomic_csv(outputs['h2_selection'], pd.DataFrame(h2sel))

    def h2rep(i: int) -> np.ndarray:
        rng = np.random.default_rng(SEED + 200000 + i)
        yp = permute_rows_within_subject(Y, subjects, rng)
        b, f, _ = cv_predict_h2(yp, Xb, Xf, subjects, folds, False)
        g, d, _ = metric_delta_r2(yp, b, f, subjects)
        return np.r_[g, d]
    h2null = run_checkpointed('h2_null', args.perm_cv, 11, cache, {**keybase, 'test': 'H2'}, h2rep, note, 25)
    h2p = perm_p(h2obs, h2null[:, 0])
    h2child = np.array([perm_p(h2dim[j], h2null[:, j + 1]) for j in range(10)])
    h2bg, h2bd = bootstrap_prediction_metrics(Y, h2pb, h2pf, subjects, args.bootstrap, SEED + 210000)
    h2brows = [{'quantity': 'global_delta_r2', 'report': 'ALL', 'observed': h2obs, 'ci_low': percentile_ci(h2bg)[0], 'ci_high': percentile_ci(h2bg)[1]}]
    for j, r in enumerate(REPORTS):
        lo, hi = percentile_ci(h2bd[:, j])
        h2brows.append({'quantity': 'dimension_delta_r2', 'report': r, 'observed': float(h2dim[j]), 'ci_low': lo, 'ci_high': hi})
    atomic_csv(outputs['h2_boot'], pd.DataFrame(h2brows))
    atomic_csv(outputs['h2_null'], pd.DataFrame(h2null, columns=['global_delta_r2', *[f'delta_r2_{r}' for r in REPORTS]]).assign(replicate=np.arange(1, args.perm_cv + 1)))
    note('H3: building fold-contained state report profiles and executing fixed-alpha LOPO mapping.')
    h3pred, allprofiles = h3_observed_and_profiles(state)
    atomic_csv(outputs['h3_profiles'], h3pred)
    h3Y = h3pred[REPORTS].to_numpy(float)
    h3P = h3pred[[f'predicted_{r}' for r in REPORTS]].to_numpy(float)
    h3subs = h3pred.subject_id.to_numpy(str)
    h3w = h3pred.state_alignment_weight.to_numpy(float)
    h3obs, h3dim, _ = metric_delta_r2(h3Y, np.zeros_like(h3Y), h3P, h3subs, h3w)
    profiles_by_outer = {s: g.drop(columns='outer_test_subject').reset_index(drop=True) for s, g in allprofiles.groupby('outer_test_subject', sort=True)}
    state_orders = {s: np.arange(len(g)) for s, g in next(iter(profiles_by_outer.values())).groupby('subject_id', sort=True)}

    def h3rep(i: int) -> np.ndarray:
        rng = np.random.default_rng(SEED + 300000 + i)
        maps = {s: rng.permutation(len(o)) for s, o in state_orders.items()}
        g, d = h3_null_once(profiles_by_outer, maps)
        return np.r_[g, d]
    h3null = run_checkpointed('h3_null', args.perm_global, 11, cache, {**keybase, 'test': 'H3', 'alpha': H3_FIXED_ALPHA}, h3rep, note, 100)
    h3p = perm_p(h3obs, h3null[:, 0])
    h3child = np.array([perm_p(h3dim[j], h3null[:, j + 1]) for j in range(10)])
    h3bg, h3bd = bootstrap_prediction_metrics(h3Y, np.zeros_like(h3Y), h3P, h3subs, args.bootstrap, SEED + 310000, h3w)
    h3brows = [{'quantity': 'global_delta_r2', 'report': 'ALL', 'observed': h3obs, 'ci_low': percentile_ci(h3bg)[0], 'ci_high': percentile_ci(h3bg)[1]}]
    for j, r in enumerate(REPORTS):
        lo, hi = percentile_ci(h3bd[:, j])
        h3brows.append({'quantity': 'dimension_delta_r2', 'report': r, 'observed': float(h3dim[j]), 'ci_low': lo, 'ci_high': hi})
    atomic_csv(outputs['h3_boot'], pd.DataFrame(h3brows))
    atomic_csv(outputs['h3_null'], pd.DataFrame(h3null, columns=['global_delta_r2', *[f'delta_r2_{r}' for r in REPORTS]]).assign(replicate=np.arange(1, args.perm_global + 1)))
    note('H4-H6: participant-balanced categorical mapping and prespecified within-participant permutations.')
    cat_specs = [('H4', 'context'), ('H5', 'emotion_category'), ('H6', 'stimulus_id')]
    catrows = []
    cross = []
    nullframes = []
    cat_p = {}
    for hi, target in cat_specs:
        obs_nmi, obs_v, ct = categorical_stats(trial, target)
        ct.insert(0, 'hypothesis', hi)
        ct.insert(1, 'target', target)
        cross.append(ct)
        width = 2

        def makefn(target=target, hi=hi):

            def f(i: int) -> np.ndarray:
                rng = np.random.default_rng(SEED + {'H4': 400000, 'H5': 500000, 'H6': 600000}[hi] + i)
                return np.array(categorical_null_once(trial, target, rng))
            return f
        nul = run_checkpointed(f'{hi.lower()}_null', args.perm_global, width, cache, {**keybase, 'test': hi}, makefn(), note, 250)
        p = perm_p(obs_nmi, nul[:, 0])
        cat_p[hi] = p
        catrows.append({'hypothesis': hi, 'target': target, 'n_rows': int(trial[target].notna().sum()), 'n_participants': int(trial.loc[trial[target].notna(), 'subject_id'].nunique()), 'observed_nmi': obs_nmi, 'observed_cramers_v': obs_v, 'raw_p_nmi': p, 'null_mean_nmi': float(nul[:, 0].mean()), 'null_sd_nmi': float(nul[:, 0].std(ddof=1))})
        nullframes.append(pd.DataFrame({'hypothesis': hi, 'target': target, 'replicate': np.arange(1, args.perm_global + 1), 'null_nmi': nul[:, 0], 'null_cramers_v': nul[:, 1]}))
    atomic_csv(outputs['categorical_tests'], pd.DataFrame(catrows))
    atomic_csv(outputs['categorical_crosstabs'], pd.concat(cross, ignore_index=True))
    atomic_csv(outputs['categorical_null'], pd.concat(nullframes, ignore_index=True))
    note('Applying prespecified Holm multiplicity control across H1-H6 and gated report follow-ups.')
    gp = np.array([h1p, h2p, h3p, cat_p['H4'], cat_p['H5'], cat_p['H6']])
    gadj = holm_adjust(gp)
    global_df = pd.DataFrame([{'hypothesis': 'H1', 'description': 'trajectory family and adjusted 10D report profile', 'statistic_name': 'shrinkage_Hotelling_T', 'observed_statistic': h1obs, 'raw_p': h1p}, {'hypothesis': 'H2', 'description': 'continuous trajectory geometry incremental report mapping', 'statistic_name': 'variance_weighted_delta_r2', 'observed_statistic': h2obs, 'raw_p': h2p}, {'hypothesis': 'H3', 'description': 'continuous aligned-state embedding report mapping', 'statistic_name': 'variance_weighted_delta_r2', 'observed_statistic': h3obs, 'raw_p': h3p}, {'hypothesis': 'H4', 'description': 'trajectory family and context', 'statistic_name': 'participant_balanced_nmi', 'observed_statistic': float(pd.DataFrame(catrows).set_index('hypothesis').loc['H4', 'observed_nmi']), 'raw_p': cat_p['H4']}, {'hypothesis': 'H5', 'description': 'trajectory family and emotion category', 'statistic_name': 'participant_balanced_nmi', 'observed_statistic': float(pd.DataFrame(catrows).set_index('hypothesis').loc['H5', 'observed_nmi']), 'raw_p': cat_p['H5']}, {'hypothesis': 'H6', 'description': 'trajectory family and stimulus identity', 'statistic_name': 'participant_balanced_nmi', 'observed_statistic': float(pd.DataFrame(catrows).set_index('hypothesis').loc['H6', 'observed_nmi']), 'raw_p': cat_p['H6']}])
    global_df['holm_p'] = gadj
    global_df['global_supported'] = global_df.holm_p < ALPHA_LEVEL
    atomic_csv(outputs['global_tests'], global_df)
    follow = []
    for parent, obsdim, childp, bootdf in [('H1', D.mean(0), h1child, pd.DataFrame(h1bootrows)), ('H2', h2dim, h2child, pd.DataFrame(h2brows)), ('H3', h3dim, h3child, pd.DataFrame(h3brows))]:
        cadj = holm_adjust(childp)
        parent_hp = float(global_df.set_index('hypothesis').loc[parent, 'holm_p'])
        for j, r in enumerate(REPORTS):
            br = bootdf[bootdf.report == r].iloc[0]
            follow.append({'parent_hypothesis': parent, 'report': r, 'effect_or_delta_r2': float(obsdim[j]), 'raw_p': float(childp[j]), 'within_parent_holm_p': float(cadj[j]), 'parent_global_holm_p': parent_hp, 'gated_supported': bool(parent_hp < ALPHA_LEVEL and cadj[j] < ALPHA_LEVEL), 'ci_low': float(br.ci_low), 'ci_high': float(br.ci_high)})
    followdf = pd.DataFrame(follow)
    atomic_csv(outputs['report_followups'], followdf)
    decision = {'all_six_global_hypotheses_executed': True, 'holm_across_h1_h6_applied': True, 'gated_report_holm_applied': True, 'alignment_refit': False, 'family_count_or_assignment_revised': False, 'family_labels_renamed': False, 'continuous_only_participants_retained': True, 'validation_opened': False, 'confirmation_opened': False, 'global_supported_hypotheses': global_df.loc[global_df.global_supported, 'hypothesis'].tolist(), 'n_global_supported': int(global_df.global_supported.sum())}
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'posthoc_discovery_mapping_inference_complete_candidate' if not args.test_mode else 'test_mode_complete', 'mapping_policy_sha256': EXPECTED_MAPPING_POLICY_SHA, 'implementation_resolution_sha256': IMPLEMENTATION_SHA, 'counts': {'trial_rows': len(trial), 'trial_participants': trial.subject_id.nunique(), 'state_objects': h3pred.state_object_id.nunique(), 'state_participants': h3pred.subject_id.nunique(), 'permutations_global': args.perm_global, 'permutations_cv': args.perm_cv, 'bootstrap_repeats': args.bootstrap}, 'decision': decision}
    atomic_json(outputs['summary'], summary)
    write_report(outputs['report'], summary, global_df)
    primary_keys = [k for k in outputs if k not in {'contract', 'next_policy', 'final_lock', 'log'}]
    hashes = {k: sha256(outputs[k]) for k in primary_keys}
    contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'source_phase': SOURCE_PHASE, 'mapping_policy_sha256': EXPECTED_MAPPING_POLICY_SHA, 'implementation_resolution_sha256': IMPLEMENTATION_SHA, 'source_mapping_design_final_lock_sha256': sha256(inputs['final_lock']), 'execution_counts': summary['counts'], 'decision': decision, 'output_hashes': hashes, 'source_dataset_modified': False, 'cross_person_alignment_fit_alignment_modified': False}
    atomic_json(outputs['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': not args.test_mode, 'next_phase_role': 'integrate the fixed Discovery mapping results, specification interpretation/robustness checks, and define the next Validation/Confirmation gate without opening either cohort', 'required_boundary': ['verify every mapping_inference hash', 'do not rename geometric families', 'do not refit alignment or mapping models to improve interpretation', 'Validation and Confirmation remain closed until a new explicit specification'], 'mapping_policy_sha256': EXPECTED_MAPPING_POLICY_SHA, 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['next_policy'], next_policy)
    note('mapping_inference mapping inference complete; no alignment or family assignment was changed. Locking final outputs.')
    hashes['contract'] = sha256(outputs['contract'])
    hashes['next_policy'] = sha256(outputs['next_policy'])
    hashes['log'] = sha256(outputs['log'])
    atomic_json(outputs['final_lock'], {'generated_at': now(), 'phase': PHASE, 'mapping_policy_sha256': EXPECTED_MAPPING_POLICY_SHA, 'implementation_resolution_sha256': IMPLEMENTATION_SHA, 'validation_opened': False, 'confirmation_opened': False, 'hashes': hashes})
    if not args.test_mode:
        pass
    print(global_df[['hypothesis', 'observed_statistic', 'raw_p', 'holm_p', 'global_supported']].to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())