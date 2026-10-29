# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Post-hoc K4 semantic quadrant hypothesis battery.

IMPORTANT:
- This branch is separate from the prespecified primary analysis.
- It does not modify structural_replication/signal_space_characterization/primary_analysis_reference outputs.
- It tests a small, pre-specified post-hoc battery only:
    H1 PRIMARY:   Valence x Arousal
    H2 SECONDARY: Liking  x Arousal
    H3 SECONDARY: Valence x Liking
- Family-to-quadrant mapping is learned in Discovery only and transferred
  unchanged to Validation and Confirmation.
- High/low is defined person-relatively within each split using the sign of
  within-participant z-scored reports (equivalently, report above/below the
  participant mean; zero-SD participants are excluded for that hypothesis).
- No causal, source-localized, biological-independence, or 'true emotion'
  interpretation is permitted."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
PHASE = 'quadrant_semantic_evaluation_k4_quadrant_semantic_battery'
SEED = 20260826
N_PERM = 5000
N_BOOT = 5000
MIN_QUADRANT_PROP = 0.05
CHANCE_BA = 0.25
cross_split_semantic_report_synthesis_FINAL_LOCK_SHA = '967c5cc35877e0e788a672d621c7825300eba7ab88ca2508a49030b081d2bd25'
HYPOTHESES = [{'id': 'H1', 'name': 'valence_x_arousal', 'x': 'Valence', 'y': 'Arousal', 'tier': 'PRIMARY'}, {'id': 'H2', 'name': 'liking_x_arousal', 'x': 'Liking', 'y': 'Arousal', 'tier': 'SECONDARY'}, {'id': 'H3', 'name': 'valence_x_liking', 'x': 'Valence', 'y': 'Liking', 'tier': 'SECONDARY'}]
SPLITS = ['Discovery', 'Validation', 'Confirmation']
DEFAULT_ASSIGNMENTS = {'Discovery': 'outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv', 'Validation': 'outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv', 'Confirmation': 'outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'}
DEFAULT_CANONICAL = '<CANONICAL_BEHAVIORAL_TABLE>'

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)

def norm_col(x: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(x).strip().lower())

def resolve_column(df: pd.DataFrame, candidates: Sequence[str], contains: Sequence[str]=()) -> str | None:
    by_norm = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        if norm_col(cand) in by_norm:
            return by_norm[norm_col(cand)]
    if contains:
        hits = []
        for c in df.columns:
            nc = norm_col(c)
            if all((norm_col(tok) in nc for tok in contains)):
                hits.append(c)
        if len(hits) == 1:
            return hits[0]
    return None

def resolve_subject_col(df: pd.DataFrame) -> str:
    c = resolve_column(df, ['participant_id_x1', 'participant_id', 'subject_id', 'subject', 'participant', 'sub', 'sub_id', 'participant'])
    if c is None:
        raise RuntimeError(f'Could not resolve participant column. Columns={list(df.columns)}')
    return c

def resolve_k_col(df: pd.DataFrame) -> str:
    c = resolve_column(df, ['k', 'K', 'resolution_k', 'n_clusters', 'nclusters', 'k_value', 'kvalue'])
    if c is None:
        hits = []
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().unique()
            if len(vals) and len(vals) <= 10 and np.all(np.isclose(vals, np.round(vals))) and set(np.round(vals).astype(int)).issubset(set(range(2, 10))):
                if norm_col(col) in {'clusterk', 'modelk', 'resolution'} or norm_col(col).endswith('k'):
                    hits.append(col)
        if len(hits) == 1:
            c = hits[0]
    if c is None:
        raise RuntimeError(f'Could not resolve K/resolution column. Columns={list(df.columns)}')
    return c

def resolve_family_col(df: pd.DataFrame) -> str:
    preferred = ['family_id', 'family', 'family_label', 'assigned_family', 'family_assignment', 'cluster_id', 'cluster', 'label', 'assignment']
    c = resolve_column(df, preferred)
    if c is not None:
        return c
    hits = [c for c in df.columns if 'family' in norm_col(c) and (not any((bad in norm_col(c) for bad in ['prob', 'count', 'size', 'distance', 'summary'])))]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f'Could not resolve family column unambiguously. Candidates={hits}; columns={list(df.columns)}')

def resolve_report_col(df: pd.DataFrame, report: str) -> str | None:
    candidates = [report, report.lower(), report.upper(), f'report_{report}', f'{report}_report', f'rating_{report}']
    c = resolve_column(df, candidates)
    if c is not None:
        return c
    hits = []
    target = norm_col(report)
    for col in df.columns:
        nc = norm_col(col)
        if nc == target or nc.endswith(target) or nc.startswith(target):
            if not any((bad in nc for bad in ['pred', 'prob', 'mean', 'sd', 'z', 'center', 'model'])):
                hits.append(col)
    return hits[0] if len(hits) == 1 else None

def _stringify_key(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace('^(-?\\d+)\\.0$', '\\1', regex=True)
    return out

def discover_join_keys(assign: pd.DataFrame, master: pd.DataFrame) -> Tuple[List[str], Dict[str, str], Dict[str, object]]:
    """Conservatively discover a unique high-coverage join from assignment rows to canonical master.

    The authoritative structural_replication assignment lineage uses ``raw_behavior_row_id`` values
    such as ``sub-02|trial1``, ``validation|sub-07|trial-001``, and
    ``confirmation|sub-03|trial-001``.  The numeric trial suffix is the
    canonical ``trial_number_global`` within subject.  This explicit lineage
    adapter is attempted first and is itself audited; only if it is unavailable
    is shared-name key discovery used.

    Returns assignment columns, mapping assignment_col -> master_col, and audit.
    Fails if no unique >=95% coverage key is found.
    """
    a_sub = resolve_subject_col(assign)
    m_sub = resolve_subject_col(master)
    raw_col = resolve_column(assign, ['raw_behavior_row_id'])
    m_trial = resolve_column(master, ['trial_number_global'])
    if raw_col is not None and m_trial is not None:
        raw = assign[raw_col].astype(str).str.strip()
        parsed_trial = pd.to_numeric(raw.str.extract('(?:^|\\|)trial-?0*(\\d+)$', expand=False), errors='coerce')
        parse_rate = float(parsed_trial.notna().mean())
        embedded_subject = raw.str.extract('(sub-\\d+)', expand=False)
        a_subject_str = _stringify_key(assign[a_sub])
        subject_check = embedded_subject.notna()
        subject_agreement = float((embedded_subject[subject_check].astype(str) == a_subject_str[subject_check].astype(str)).mean()) if bool(subject_check.any()) else float('nan')
        if bool(subject_check.any()) and subject_agreement < 1.0:
            bad = assign.loc[subject_check & (embedded_subject.astype(str) != a_subject_str.astype(str)), [raw_col, a_sub]].head(10)
            raise RuntimeError(f"raw_behavior_row_id embedded subject disagrees with assignment subject; examples={bad.to_dict('records')}")
        if parse_rate >= 0.95:
            derived = '__trial_number_global_from_raw_behavior_row_id'
            assign[derived] = parsed_trial.astype('Int64')
            aa = assign[[a_sub, derived]].dropna().copy()
            mm = master[[m_sub, m_trial]].dropna().copy()
            aa.columns = ['subject', 'trial']
            mm.columns = ['subject', 'trial']
            for c in aa.columns:
                aa[c] = _stringify_key(aa[c])
                mm[c] = _stringify_key(mm[c])
            if aa.duplicated().any():
                raise RuntimeError('Derived raw_behavior_row_id join key is not unique in assignment K4 rows')
            if mm.duplicated().any():
                raise RuntimeError('Canonical subject_id + trial_number_global is not unique')
            aset = set(map(tuple, aa.to_numpy()))
            mset = set(map(tuple, mm.to_numpy()))
            matched = aset & mset
            coverage = len(matched) / max(1, len(aset))
            if coverage >= 0.95:
                unmatched = sorted(aset - mset)[:20]
                return ([a_sub, derived], {a_sub: m_sub, derived: m_trial}, {'assignment_join_columns': [a_sub, derived], 'master_join_columns': [m_sub, m_trial], 'coverage': float(coverage), 'priority_score': 1000, 'candidate_count': 1, 'join_strategy': 'explicit_raw_behavior_row_id_trial_number_global', 'raw_behavior_row_id_column': raw_col, 'raw_behavior_row_id_parse_rate': parse_rate, 'embedded_subject_agreement': subject_agreement, 'unmatched_assignment_key_examples': [list(x) for x in unmatched]})
    master_by_norm = {norm_col(c): c for c in master.columns}
    id_tokens = ('trial', 'stim', 'event', 'pair', 'run', 'block', 'item', 'epoch', 'presentation', 'canonical', 'index', 'row')
    candidates = []
    for ac in assign.columns:
        nac = norm_col(ac)
        if ac == a_sub or nac not in master_by_norm:
            continue
        if any((tok in nac for tok in id_tokens)):
            mc = master_by_norm[nac]
            if mc != m_sub:
                candidates.append((ac, mc))

    def priority(pair):
        n = norm_col(pair[0])
        scores = [('canonicaltrial', 100), ('trialuid', 95), ('trialid', 90), ('trialkey', 85), ('pairid', 80), ('eventid', 75), ('stimulusid', 60), ('run', 30), ('block', 20), ('index', 10)]
        return max([v for tok, v in scores if tok in n] or [0])
    candidates = sorted(candidates, key=priority, reverse=True)[:8]
    trials = []
    for r in range(1, min(3, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, r):
            a_cols = [a_sub] + [x[0] for x in combo]
            m_cols = [m_sub] + [x[1] for x in combo]
            aa = assign[a_cols].copy()
            mm = master[m_cols].copy()
            aa.columns = [f'k{i}' for i in range(len(a_cols))]
            mm.columns = aa.columns
            for c in aa.columns:
                aa[c] = _stringify_key(aa[c])
                mm[c] = _stringify_key(mm[c])
            a_unique = not aa.duplicated().any()
            m_unique = not mm.duplicated().any()
            if not (a_unique and m_unique):
                continue
            aset = set(map(tuple, aa.to_numpy()))
            mset = set(map(tuple, mm.to_numpy()))
            coverage = len(aset & mset) / max(1, len(aset))
            if coverage >= 0.95:
                trials.append((coverage, sum((priority(x) for x in combo)), a_cols, m_cols))
    if not trials:
        raise RuntimeError(f'Could not discover a unique >=95% coverage join between K4 assignment rows and canonical master. Assignment columns={list(assign.columns)}; master ID-like common candidates={candidates}')
    trials.sort(key=lambda x: (x[0], x[1], -len(x[2])), reverse=True)
    cov, score, a_cols, m_cols = trials[0]
    mapping = dict(zip(a_cols, m_cols))
    return (a_cols, mapping, {'assignment_join_columns': a_cols, 'master_join_columns': m_cols, 'coverage': cov, 'priority_score': score, 'candidate_count': len(trials)})

def attach_reports(assign_k4: pd.DataFrame, master: pd.DataFrame, reports: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    direct = {r: resolve_report_col(assign_k4, r) for r in reports}
    if all((v is not None for v in direct.values())):
        out = assign_k4.copy()
        for r, c in direct.items():
            if c != r:
                out[r] = pd.to_numeric(out[c], errors='coerce')
            else:
                out[r] = pd.to_numeric(out[r], errors='coerce')
        return (out, {'mode': 'reports_already_in_assignment', 'report_columns': direct})
    a_cols, mapping, audit = discover_join_keys(assign_k4, master)
    m_cols = [mapping[c] for c in a_cols]
    report_cols = {}
    for r in reports:
        c = resolve_report_col(master, r)
        if c is None:
            raise RuntimeError(f'Canonical master missing resolvable report {r}. Columns={list(master.columns)}')
        report_cols[r] = c
    left = assign_k4.copy()
    right = master[m_cols + list(dict.fromkeys(report_cols.values()))].copy()
    for ac in a_cols:
        left[ac] = _stringify_key(left[ac])
    right = right.rename(columns={mc: ac for ac, mc in mapping.items()})
    for ac in a_cols:
        right[ac] = _stringify_key(right[ac])
    for r, mc in report_cols.items():
        rc = {v: k for k, v in mapping.items()}.get(mc, mc)
        if rc in right.columns and rc in a_cols:
            raise RuntimeError(f'Report column collided with join key: {r} -> {mc}')
        if mc in right.columns:
            right = right.rename(columns={mc: r})
        elif r not in right.columns:
            src = report_cols[r]
            if src in right.columns:
                right = right.rename(columns={src: r})
    merged = left.merge(right, on=a_cols, how='left', validate='one_to_one', indicator=True)
    coverage = float((merged['_merge'] == 'both').mean())
    if coverage < 0.95:
        raise RuntimeError(f'Join coverage after merge dropped below 95%: {coverage:.4f}')
    merged = merged.drop(columns=['_merge'])
    for r in reports:
        merged[r] = pd.to_numeric(merged[r], errors='coerce')
    audit.update({'mode': 'joined_to_canonical_master', 'postmerge_coverage': coverage, 'report_columns': report_cols})
    return (merged, audit)

def filter_k4(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    kcol = resolve_k_col(df)
    kval = pd.to_numeric(df[kcol], errors='coerce')
    out = df.loc[kval == 4].copy()
    if out.empty:
        raise RuntimeError(f'No K=4 rows found in {kcol}; unique={sorted(pd.Series(kval).dropna().unique().tolist())}')
    fam = resolve_family_col(out)
    sub = resolve_subject_col(out)
    if out[fam].nunique(dropna=True) != 4:
        raise RuntimeError(f'K4 must have exactly four family labels; got {out[fam].dropna().unique().tolist()}')
    out = out.rename(columns={fam: 'family_k4', sub: 'participant_id_x1'})
    return (out, {'k_column': kcol, 'family_column': fam, 'subject_column': sub, 'n_rows_k4': int(len(out)), 'families': sorted(map(str, out['family_k4'].dropna().unique()))})

def person_relative_z(df: pd.DataFrame, col: str, participant_col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors='coerce')
    means = x.groupby(df[participant_col]).transform('mean')
    sds = x.groupby(df[participant_col]).transform('std')
    z = (x - means) / sds.replace(0, np.nan)
    return z

def quadrant_labels(df: pd.DataFrame, xcol: str, ycol: str, participant_col: str) -> Tuple[pd.Series, Dict[str, object]]:
    zx = person_relative_z(df, xcol, participant_col)
    zy = person_relative_z(df, ycol, participant_col)
    q = pd.Series(index=df.index, dtype='object')
    ok = zx.notna() & zy.notna() & (zx != 0) & (zy != 0)
    q.loc[ok & (zx < 0) & (zy < 0)] = 'LOW_LOW'
    q.loc[ok & (zx < 0) & (zy > 0)] = 'LOW_HIGH'
    q.loc[ok & (zx > 0) & (zy < 0)] = 'HIGH_LOW'
    q.loc[ok & (zx > 0) & (zy > 0)] = 'HIGH_HIGH'
    counts = q.value_counts(dropna=True).to_dict()
    n = int(q.notna().sum())
    props = {k: float(v) / n if n else 0.0 for k, v in counts.items()}
    for lab in ['LOW_LOW', 'LOW_HIGH', 'HIGH_LOW', 'HIGH_HIGH']:
        counts.setdefault(lab, 0)
        props.setdefault(lab, 0.0)
    return (q, {'n_eligible': n, 'n_excluded_missing_zero_or_zero_sd': int(len(df) - n), 'counts': counts, 'proportions': props, 'min_quadrant_prop': min(props.values()) if props else 0.0, 'occupancy_gate': bool(n > 0 and min(props.values()) >= MIN_QUADRANT_PROP), 'coding': 'within-participant z-score sign; exact zero excluded'})

def contingency(family: Sequence, quadrant: Sequence, families: Sequence[str], quadrants: Sequence[str]) -> np.ndarray:
    fam_idx = {str(f): i for i, f in enumerate(families)}
    q_idx = {str(q): i for i, q in enumerate(quadrants)}
    mat = np.zeros((len(families), len(quadrants)), dtype=np.int64)
    for f, q in zip(family, quadrant):
        fs, qs = (str(f), str(q))
        if fs in fam_idx and qs in q_idx:
            mat[fam_idx[fs], q_idx[qs]] += 1
    return mat

def ba_from_mapping_matrix(mat: np.ndarray, perm: Tuple[int, ...]) -> float:
    class_totals = mat.sum(axis=0)
    recalls = []
    for q in range(mat.shape[1]):
        if class_totals[q] <= 0:
            continue
        fam_rows = [i for i, qq in enumerate(perm) if qq == q]
        if len(fam_rows) != 1:
            return float('nan')
        recalls.append(mat[fam_rows[0], q] / class_totals[q])
    return float(np.mean(recalls)) if recalls else float('nan')

def best_bijection(family: Sequence, quadrant: Sequence) -> Tuple[Dict[str, str], float, np.ndarray, List[str], List[str]]:
    families = sorted(map(str, pd.Series(family).dropna().unique()))
    quadrants = ['LOW_LOW', 'LOW_HIGH', 'HIGH_LOW', 'HIGH_HIGH']
    if len(families) != 4:
        raise RuntimeError(f'Expected 4 family labels, got {families}')
    mat = contingency(family, quadrant, families, quadrants)
    best = None
    for perm in itertools.permutations(range(4)):
        ba = ba_from_mapping_matrix(mat, perm)
        key = (ba, tuple((-p for p in perm)))
        if best is None or key > best[0]:
            best = (key, perm, ba)
    assert best is not None
    perm = best[1]
    mapping = {families[i]: quadrants[perm[i]] for i in range(4)}
    return (mapping, float(best[2]), mat, families, quadrants)

def predict_with_mapping(family: Sequence, mapping: Dict[str, str]) -> np.ndarray:
    return np.asarray([mapping.get(str(f), '__UNMAPPED__') for f in family], dtype=object)

def metrics_for(y_true: Sequence, y_pred: Sequence, family: Sequence) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    family = np.asarray([str(x) for x in family], dtype=object)
    return {'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'ari_family_quadrant': float(adjusted_rand_score(y_true, family)), 'nmi_family_quadrant': float(normalized_mutual_info_score(y_true, family))}

def participant_bootstrap_ba(df: pd.DataFrame, true_col: str, pred_col: str, participant_col: str, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, np.ndarray]:
    participants = np.asarray(sorted(df[participant_col].astype(str).unique()))
    grouped = {p: df.index[df[participant_col].astype(str) == p].to_numpy() for p in participants}
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        idx = np.concatenate([grouped[p] for p in sampled])
        sub = df.loc[idx]
        vals[b] = balanced_accuracy_score(sub[true_col], sub[pred_col])
    lo, hi = np.quantile(vals[np.isfinite(vals)], [0.025, 0.975])
    return (float(lo), float(hi), vals)

def shuffle_within_participant(values: np.ndarray, participant: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for p in np.unique(participant):
        idx = np.flatnonzero(participant == p)
        out[idx] = rng.permutation(out[idx])
    return out

def discovery_selection_perm_p(df: pd.DataFrame, qcol: str, participant_col: str, n_perm: int, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    y = df[qcol].to_numpy(dtype=object)
    fam = df['family_k4'].astype(str).to_numpy(dtype=object)
    part = df[participant_col].astype(str).to_numpy(dtype=object)
    _, obs, _, _, _ = best_bijection(fam, y)
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = shuffle_within_participant(y, part, rng)
        _, vals[i], _, _, _ = best_bijection(fam, yp)
    p = (1.0 + float(np.sum(vals >= obs - 1e-15))) / (n_perm + 1.0)
    return (p, vals)

def heldout_fixed_mapping_perm_p(df: pd.DataFrame, qcol: str, pred_col: str, participant_col: str, n_perm: int, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    y = df[qcol].to_numpy(dtype=object)
    pred = df[pred_col].to_numpy(dtype=object)
    part = df[participant_col].astype(str).to_numpy(dtype=object)
    obs = balanced_accuracy_score(y, pred)
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = shuffle_within_participant(y, part, rng)
        vals[i] = balanced_accuracy_score(yp, pred)
    p = (1.0 + float(np.sum(vals >= obs - 1e-15))) / (n_perm + 1.0)
    return (p, vals)

def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()

def build_confusion_rows(hid: str, hname: str, split: str, true: Sequence, pred: Sequence) -> List[Dict[str, object]]:
    labs = ['LOW_LOW', 'LOW_HIGH', 'HIGH_LOW', 'HIGH_HIGH']
    rows = []
    for t in labs:
        for p in labs:
            rows.append({'hypothesis_id': hid, 'hypothesis': hname, 'split': split, 'true_quadrant': t, 'predicted_quadrant': p, 'n': int(np.sum((np.asarray(true) == t) & (np.asarray(pred) == p)))})
    return rows

def verify_historical_boundaries(project: Path) -> Dict[str, object]:
    cands = list(project.glob('outputs/interpretability/cross_split_semantic_report_synthesis_final_crosssplit_semantic_report_synthesis/qc/*final_output_lock_hashes.json'))
    if len(cands) != 1:
        raise RuntimeError(f'Expected exactly one cross_split_semantic_report_synthesis final lock file, found {cands}')
    cross_split_semantic_report_synthesis = cands[0]
    structural_synthesis_sha = sha256_file(cross_split_semantic_report_synthesis)
    if structural_synthesis_sha != cross_split_semantic_report_synthesis_FINAL_LOCK_SHA:
        raise RuntimeError(f'cross_split_semantic_report_synthesis final lock SHA mismatch: expected {cross_split_semantic_report_synthesis_FINAL_LOCK_SHA}, observed {structural_synthesis_sha}')
    return {'cross_split_semantic_report_synthesis_final_lock_path': str(cross_split_semantic_report_synthesis), 'cross_split_semantic_report_synthesis_final_lock_sha256': structural_synthesis_sha, 'cross_split_semantic_report_synthesis_verified': True, 'branch_status': 'POST_HOC_SEPARATE_FROM_PRESPECIFIED_CORE'}

def run(project: Path, canonical: Path, n_perm: int, n_boot: int) -> Path:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if not canonical.exists():
        raise FileNotFoundError(canonical)
    boundaries = verify_historical_boundaries(project)
    outroot = project / 'outputs' / 'supplementary_analyses' / PHASE
    tables = outroot / 'tables'
    qc = outroot / 'qc'
    reports = outroot / 'reports'
    for d in [tables, qc, reports]:
        d.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(canonical, low_memory=False)
    required_reports = sorted(set([h['x'] for h in HYPOTHESES] + [h['y'] for h in HYPOTHESES]))
    source_rows = [{'role': 'canonical_master', 'path': str(canonical), 'sha256': sha256_file(canonical), 'size_bytes': canonical.stat().st_size}]
    split_frames = {}
    schema_audits = {}
    for split in SPLITS:
        p = project / DEFAULT_ASSIGNMENTS[split]
        if not p.exists():
            raise FileNotFoundError(p)
        source_rows.append({'role': f'{split.lower()}_k_family_assignments', 'path': str(p), 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size})
        raw = pd.read_csv(p, low_memory=False)
        k4, k_audit = filter_k4(raw)
        merged, join_audit = attach_reports(k4, master, required_reports)
        if 'participant_id_x1' not in merged.columns:
            raise RuntimeError('Internal participant standardization failed')
        merged['participant_id_x1'] = merged['participant_id_x1'].astype(str)
        merged['family_k4'] = merged['family_k4'].astype(str)
        split_frames[split] = merged
        schema_audits[split] = {'assignment_path': str(p), 'assignment_sha256': sha256_file(p), 'raw_rows': int(len(raw)), **k_audit, 'report_attachment': join_audit, 'final_rows': int(len(merged)), 'participants': int(merged['participant_id_x1'].nunique())}
    pd.DataFrame(source_rows).to_csv(qc / 'source_registry.csv', index=False)
    atomic_json(qc / 'schema_and_join_audit.json', schema_audits)
    atomic_json(qc / 'historical_boundary_audit.json', boundaries)
    metrics_rows = []
    mapping_rows = []
    occupancy_rows = []
    confusion_rows = []
    null_rows = []
    heldout_test_rows = []
    heldout_lookup = {}
    for h in HYPOTHESES:
        hid, hname, x, y, tier = (h['id'], h['name'], h['x'], h['y'], h['tier'])
        prepared = {}
        for split in SPLITS:
            df = split_frames[split].copy()
            q, occ = quadrant_labels(df, x, y, 'participant_id_x1')
            qcol = f'quadrant_{hname}'
            df[qcol] = q
            df = df[df[qcol].notna() & df['family_k4'].notna()].copy()
            if df.empty:
                raise RuntimeError(f'No eligible rows for {hname} in {split}')
            prepared[split] = (df, qcol, occ)
            occupancy_rows.append({'hypothesis_id': hid, 'hypothesis': hname, 'tier': tier, 'split': split, 'x_report': x, 'y_report': y, **{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in occ.items()}})
        ddf, dqcol, docc = prepared['Discovery']
        mapping, dba, dmat, families, quadrants = best_bijection(ddf['family_k4'], ddf[dqcol])
        for fam in families:
            mapping_rows.append({'hypothesis_id': hid, 'hypothesis': hname, 'family_k4': fam, 'quadrant': mapping[fam], 'learned_in': 'Discovery'})
        dp, dnull = discovery_selection_perm_p(ddf, dqcol, 'participant_id_x1', n_perm, rng)
        dpred_col = f'pred_{hname}'
        ddf[dpred_col] = predict_with_mapping(ddf['family_k4'], mapping)
        dm = metrics_for(ddf[dqcol], ddf[dpred_col], ddf['family_k4'])
        dlo, dhi, _ = participant_bootstrap_ba(ddf, dqcol, dpred_col, 'participant_id_x1', n_boot, rng)
        metrics_rows.append({'hypothesis_id': hid, 'hypothesis': hname, 'tier': tier, 'split': 'Discovery', 'n': len(ddf), 'participants': ddf['participant_id_x1'].nunique(), **dm, 'ba_ci_low': dlo, 'ba_ci_high': dhi, 'permutation_p': dp, 'permutation_mode': 'selection_corrected_reoptimize_24_bijections_within_participant_shuffle', 'occupancy_gate': bool(docc['occupancy_gate'])})
        confusion_rows.extend(build_confusion_rows(hid, hname, 'Discovery', ddf[dqcol], ddf[dpred_col]))
        null_rows.extend(({'hypothesis_id': hid, 'hypothesis': hname, 'split': 'Discovery', 'iteration': i + 1, 'null_balanced_accuracy': float(v)} for i, v in enumerate(dnull)))
        heldout_lookup[hname] = {}
        for split in ['Validation', 'Confirmation']:
            df, qcol, occ = prepared[split]
            pred_col = f'pred_{hname}'
            df[pred_col] = predict_with_mapping(df['family_k4'], mapping)
            mm = metrics_for(df[qcol], df[pred_col], df['family_k4'])
            lo, hi, _ = participant_bootstrap_ba(df, qcol, pred_col, 'participant_id_x1', n_boot, rng)
            pp, pnull = heldout_fixed_mapping_perm_p(df, qcol, pred_col, 'participant_id_x1', n_perm, rng)
            row = {'hypothesis_id': hid, 'hypothesis': hname, 'tier': tier, 'split': split, 'n': len(df), 'participants': df['participant_id_x1'].nunique(), **mm, 'ba_ci_low': lo, 'ba_ci_high': hi, 'permutation_p': pp, 'permutation_mode': 'fixed_discovery_mapping_within_participant_quadrant_shuffle', 'occupancy_gate': bool(occ['occupancy_gate'])}
            metrics_rows.append(row)
            heldout_lookup[hname][split] = row
            heldout_test_rows.append({'hypothesis_id': hid, 'hypothesis': hname, 'tier': tier, 'split': split, 'raw_p': pp})
            confusion_rows.extend(build_confusion_rows(hid, hname, split, df[qcol], df[pred_col]))
            null_rows.extend(({'hypothesis_id': hid, 'hypothesis': hname, 'split': split, 'iteration': i + 1, 'null_balanced_accuracy': float(v)} for i, v in enumerate(pnull)))
    held = pd.DataFrame(heldout_test_rows)
    held['holm_family'] = ''
    held['p_holm'] = np.nan
    mask_primary = held['tier'] == 'PRIMARY'
    held.loc[mask_primary, 'p_holm'] = holm_adjust(held.loc[mask_primary, 'raw_p'].tolist())
    held.loc[mask_primary, 'holm_family'] = 'H1_validation_confirmation'
    mask_secondary = held['tier'] == 'SECONDARY'
    held.loc[mask_secondary, 'p_holm'] = holm_adjust(held.loc[mask_secondary, 'raw_p'].tolist())
    held.loc[mask_secondary, 'holm_family'] = 'H2_H3_x_validation_confirmation'
    held.to_csv(tables / 'held_out_permutation_holm.csv', index=False)
    gate_rows = []
    for h in HYPOTHESES:
        hname = h['name']
        details = []
        for split in ['Validation', 'Confirmation']:
            mrow = heldout_lookup[hname][split]
            prow = held[(held['hypothesis'] == hname) & (held['split'] == split)].iloc[0]
            split_pass = bool(mrow['occupancy_gate'] and mrow['balanced_accuracy'] > CHANCE_BA and (mrow['ba_ci_low'] > CHANCE_BA) and (float(prow['p_holm']) < 0.05))
            details.append({'split': split, 'balanced_accuracy': mrow['balanced_accuracy'], 'ba_ci_low': mrow['ba_ci_low'], 'ba_ci_high': mrow['ba_ci_high'], 'occupancy_gate': mrow['occupancy_gate'], 'raw_p': float(prow['raw_p']), 'p_holm': float(prow['p_holm']), 'pass': split_pass})
        gate_rows.append({'hypothesis_id': h['id'], 'hypothesis': hname, 'tier': h['tier'], 'heldout_replication_gate_pass': bool(all((x['pass'] for x in details))), 'validation_pass': details[0]['pass'], 'confirmation_pass': details[1]['pass'], 'details_json': json.dumps(details, sort_keys=True), 'allowed_interpretation_if_pass': 'K4 assignments show internally replicated statistical alignment with the pre-specified person-relative report quadrant; this does not establish true emotion identities, biological independence, causality, or external validity.', 'interpretation_if_fail': 'No internally replicated K4-to-quadrant semantic mapping under this prespecified test.'})
    pd.DataFrame(metrics_rows).to_csv(tables / 'quadrant_mapping_metrics.csv', index=False)
    pd.DataFrame(mapping_rows).to_csv(tables / 'discovery_prespecified_family_to_quadrant_mapping.csv', index=False)
    pd.DataFrame(occupancy_rows).to_csv(tables / 'quadrant_occupancy_audit.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(tables / 'confusion_cells.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(tables / 'hypothesis_gate_summary.csv', index=False)
    pd.DataFrame(null_rows).to_csv(tables / 'permutation_null_balanced_accuracy.csv.gz', index=False, compression='gzip')
    design_lock = {'phase': PHASE, 'status': 'POST_HOC_HYPOTHESIS_BATTERY_EXECUTED_SEPARATE_FROM_PRESPECIFIED_CORE', 'seed': SEED, 'n_permutations': n_perm, 'n_participant_bootstrap': n_boot, 'k': 4, 'family_mapping_rule': 'learn optimal bijection in Discovery only by balanced accuracy; deterministic tie-break; specification unchanged for Validation and Confirmation', 'quadrant_coding': 'within-participant z-score sign separately within each split; exact zeros/zero-SD unavailable rows excluded', 'minimum_quadrant_proportion_gate': MIN_QUADRANT_PROP, 'chance_balanced_accuracy': CHANCE_BA, 'hypotheses': HYPOTHESES, 'primary_multiple_testing': 'Holm across H1 Validation + Confirmation permutation p-values', 'secondary_multiple_testing': 'Holm across H2/H3 x Validation/Confirmation (4 p-values)', 'replication_gate': 'both heldout splits: occupancy >=5% each quadrant, BA>0.25, participant-bootstrap 95% lower CI>0.25, Holm p<0.05', 'discovery_p_value': 'selection-corrected permutation p-value by re-optimizing all 24 bijections per within-participant shuffled null', 'heldout_p_value': 'fixed Discovery mapping; within-participant quadrant-label shuffle', 'branch_boundary': 'does not modify structural_replication/signal_space_characterization/primary_analysis_reference or prespecified primary analysis', 'claim_boundary': ['No true emotion identity claim.', 'No biological independence claim.', 'No causal claim.', 'No source-localized generator claim.', 'No external-validation claim.', 'No claim that conventional labels are wrong.']}
    atomic_json(qc / 'posthoc_design_lock.json', design_lock)
    gdf = pd.DataFrame(gate_rows)
    lines = ['# Post-hoc K4 report-quadrant semantic battery', '', '**Status:** separate post-hoc branch; prespecified primary analysis not modified.', '', '## prespecified hypotheses', '', '- H1 PRIMARY: Valence × Arousal', '- H2 SECONDARY: Liking × Arousal', '- H3 SECONDARY: Valence × Liking', '', 'High/low is person-relative within each split (sign of within-participant z score).', 'Discovery alone determines the four-family ↔ four-quadrant bijection; the same mapping is transferred unchanged to Validation and Confirmation.', '', '## Held-out replication gates', '']
    for _, r in gdf.iterrows():
        lines.append(f"- {r['hypothesis_id']} {r['hypothesis']}: {('PASS' if r['heldout_replication_gate_pass'] else 'NOT SUPPORTED')} (Validation={r['validation_pass']}, Confirmation={r['confirmation_pass']})")
    lines += ['', '## Interpretation boundary', '', 'A PASS would support only an internally replicated statistical alignment between K4 assignments and that pre-specified person-relative report quadrant. It would not identify true emotions, causal generators, biological independence, or external validity.', '', 'A failure means that specific quadrant interpretation is not internally replicated under this prespecified test.']
    (reports / 'quadrant_semantic_evaluation_report.md').write_text(''.join(lines) + '', encoding='utf-8')
    summary = {'phase': PHASE, 'canonical_master_sha256': sha256_file(canonical), 'historical_boundaries': boundaries, 'source_files': source_rows, 'hypothesis_results': gate_rows, 'prespecified_primary_analysis_modified': False, 'structural_replication_signal_space_characterization_primary_analysis_reference_reopened': False, 'new_eeg_feature_extraction': False, 'posthoc_status_disclosed': True}
    atomic_json(qc / 'summary.json', summary)
    child_files = sorted([p for p in outroot.rglob('*') if p.is_file() and p.name not in {'final_output_lock_hashes.json'}])
    hashes = {str(p.relative_to(outroot)): sha256_file(p) for p in child_files}
    atomic_json(qc / 'final_output_lock_hashes.json', hashes)
    final_lock_sha = sha256_file(qc / 'final_output_lock_hashes.json')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    print('quadrant_semantic_evaluation PASS')
    print(f"cross_split_semantic_report_synthesis final lock verified: {boundaries['cross_split_semantic_report_synthesis_verified']}")
    print('prespecified hypotheses: H1 Valence×Arousal; H2 Liking×Arousal; H3 Valence×Liking')
    for r in gate_rows:
        print(f"{r['hypothesis_id']} {r['hypothesis']}: {('PASS' if r['heldout_replication_gate_pass'] else 'NOT_SUPPORTED')}")
    print(f'Final output lock SHA-256: {final_lock_sha}')
    return None

def self_test() -> None:
    fam = np.array(['0', '1', '2', '3'] * 10, dtype=object)
    q = np.array(['LOW_LOW', 'LOW_HIGH', 'HIGH_LOW', 'HIGH_HIGH'] * 10, dtype=object)
    mapping, ba, mat, families, quadrants = best_bijection(fam, q)
    assert math.isclose(ba, 1.0)
    pred = predict_with_mapping(fam, mapping)
    assert np.all(pred == q)
    m = metrics_for(q, pred, fam)
    assert math.isclose(m['balanced_accuracy'], 1.0)
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert len(adj) == 3 and all((0 <= x <= 1 for x in adj))
    aj = pd.DataFrame({'participant_id_x1': ['sub-02', 'sub-07', 'sub-03'], 'raw_behavior_row_id': ['sub-02|trial1', 'validation|sub-07|trial-001', 'confirmation|sub-03|trial-001']})
    mj = pd.DataFrame({'subject_id': ['sub-02', 'sub-07', 'sub-03'], 'trial_number_global': [1, 1, 1], 'valence': [1.0, 2.0, 3.0]})
    ac, mp, ja = discover_join_keys(aj, mj)
    assert ja['join_strategy'] == 'explicit_raw_behavior_row_id_trial_number_global'
    assert math.isclose(ja['coverage'], 1.0)
    assert mp[ac[0]] == 'subject_id' and mp[ac[1]] == 'trial_number_global'
    df = pd.DataFrame({'p': np.repeat(['a', 'b', 'c', 'd'], 8), 'Valence': np.tile([1, 2, 3, 4, 6, 7, 8, 9], 4), 'Arousal': np.tile([1, 8, 2, 7, 3, 6, 4, 5], 4)})
    qq, audit = quadrant_labels(df, 'Valence', 'Arousal', 'p')
    assert qq.notna().all()
    assert audit['n_eligible'] == len(df)
    print('quadrant_semantic_evaluation self-test PASS')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='<PROJECT_ROOT>')
    ap.add_argument('--canonical-master', default=DEFAULT_CANONICAL)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.n_perm < 1000 or args.n_boot < 1000:
        raise SystemExit('Refusing execution with n_perm or n_boot < 1000; prespecified default is 5000.')
    run(Path(args.project), Path(args.canonical_master), args.n_perm, args.n_boot)
if __name__ == '__main__':
    main()