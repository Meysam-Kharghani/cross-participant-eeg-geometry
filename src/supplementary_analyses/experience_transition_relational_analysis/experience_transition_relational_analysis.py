# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Post-hoc experience↔transition relational architecture.

Separate post-hoc branch. Does NOT modify prespecified primary analysis, structural_replication, signal_space_characterization, or primary_analysis_reference.

prespecified questions:
H1: Is transition K2 a Discovery-selected coarsening of experience K4?
H2: Is there participant-balanced conditional association/specialization between
    experience K4 and transition K2?
H3: Descriptive 14D geometry correspondence under the fixed H1 mapping.

Discovery selects the K4→K2 map. Validation and Confirmation never re-optimize it."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
SEED = 20260826
DEFAULT_N_PERM = 5000
DEFAULT_N_BOOT = 5000
cross_split_semantic_report_synthesis_FINAL_LOCK_SHA = '967c5cc35877e0e788a672d621c7825300eba7ab88ca2508a49030b081d2bd25'
OUT_REL = Path('outputs/supplementary_analyses/experience_transition_relational_analysis_experience_transition_relational_architecture')
INPUTS = {'Discovery': {'exp_assign': ('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv', '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'), 'trans_assign': ('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv', '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'), 'exp_sig': ('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv', 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'), 'trans_sig': ('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv', '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83')}, 'Validation': {'exp_assign': ('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv', 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'), 'trans_assign': ('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv', '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'), 'exp_sig': ('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv', '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'), 'trans_sig': ('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv', 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067')}, 'Confirmation': {'exp_assign': ('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv', '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'), 'trans_assign': ('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv', 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'), 'exp_sig': ('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv', '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'), 'trans_sig': ('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv', '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6')}}
EXPECTED_COMMON = {'Discovery': (1237, 31), 'Validation': (437, 12), 'Confirmation': (459, 12)}
KEYS = ['raw_behavior_row_id', 'subject_id']

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0 or (not np.isfinite(na + nb)):
        return float('nan')
    return float(np.dot(a, b) / (na * nb))

def all_surjective_maps_4_to_2() -> List[Tuple[int, int, int, int]]:
    out = []
    for x in itertools.product([1, 2], repeat=4):
        if len(set(x)) == 2:
            out.append(tuple((int(v) for v in x)))
    return out
MAPS = all_surjective_maps_4_to_2()
assert len(MAPS) == 14

def map_label_tuple(mapping: Tuple[int, int, int, int]) -> str:
    return ';'.join((f'E{i + 1}->T{mapping[i]}' for i in range(4)))

def table_from_arrays(E: np.ndarray, T: np.ndarray, weight: np.ndarray) -> np.ndarray:
    tab = np.zeros((4, 2), dtype=float)
    np.add.at(tab, (E.astype(int) - 1, T.astype(int) - 1), weight.astype(float))
    return tab

def per_subject_tables(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    subjects = sorted(df['subject_id'].astype(str).unique().tolist())
    arr = np.zeros((len(subjects), 4, 2), dtype=float)
    for i, s in enumerate(subjects):
        g = df[df['subject_id'].astype(str) == s]
        n = len(g)
        if n <= 0:
            raise RuntimeError(f'Empty participant block: {s}')
        w = np.full(n, 1.0 / n, dtype=float)
        arr[i] = table_from_arrays(g['E'].to_numpy(), g['T'].to_numpy(), w)
    return (subjects, arr)

def ba_from_table(tab: np.ndarray, mapping: Tuple[int, int, int, int]) -> float:
    recalls = []
    for t in (1, 2):
        denom = float(tab[:, t - 1].sum())
        if denom <= 0:
            return float('nan')
        idx = [e - 1 for e in (1, 2, 3, 4) if mapping[e - 1] == t]
        num = float(tab[idx, t - 1].sum())
        recalls.append(num / denom)
    return float(np.mean(recalls))

def weighted_nmi_from_table(tab: np.ndarray) -> Tuple[float, float, float, float, float]:
    total = float(tab.sum())
    if total <= 0:
        return (float('nan'),) * 5
    p = tab / total
    pe = p.sum(axis=1)
    pt = p.sum(axis=0)
    mi = 0.0
    for i in range(4):
        for j in range(2):
            if p[i, j] > 0 and pe[i] > 0 and (pt[j] > 0):
                mi += p[i, j] * math.log(p[i, j] / (pe[i] * pt[j]))
    he = -sum((x * math.log(x) for x in pe if x > 0))
    ht = -sum((x * math.log(x) for x in pt if x > 0))
    nmi = mi / math.sqrt(he * ht) if he > 0 and ht > 0 else float('nan')
    h_t_given_e = max(0.0, ht - mi) if np.isfinite(ht) else float('nan')
    reduction = mi / ht if ht > 0 else float('nan')
    return (float(nmi), float(mi), float(he), float(ht), float(reduction))

def fixed_mapping_ari_nmi(df: pd.DataFrame, mapping: Tuple[int, int, int, int]) -> Tuple[float, float]:
    pred = np.array([mapping[int(e) - 1] for e in df['E'].to_numpy()], dtype=int)
    true = df['T'].to_numpy(dtype=int)
    return (float(adjusted_rand_score(true, pred)), float(normalized_mutual_info_score(true, pred)))

def shuffled_subject_table(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    tab = np.zeros((4, 2), dtype=float)
    for _, g in df.groupby('subject_id', sort=False):
        E = g['E'].to_numpy(dtype=int)
        T = g['T'].to_numpy(dtype=int).copy()
        rng.shuffle(T)
        w = np.full(len(g), 1.0 / len(g), dtype=float)
        tab += table_from_arrays(E, T, w)
    return tab

def percentile_ci(x: np.ndarray, alpha: float=0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float('nan'), float('nan'))
    return tuple((float(v) for v in np.quantile(x, [alpha / 2, 1 - alpha / 2])))

def bootstrap_tables(subj_tabs: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = subj_tabs.shape[0]
    out = np.empty((n_boot, 4, 2), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = subj_tabs[idx].sum(axis=0)
    return out

def permutation_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))

def holm_adjust(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * float(p))
        running = max(running, adj)
        out[name] = running
    return out

def load_assign_pair(project: Path, split: str, kE: int, kT: int) -> Tuple[pd.DataFrame, dict]:
    ep = project / INPUTS[split]['exp_assign'][0]
    tp = project / INPUTS[split]['trans_assign'][0]
    e = pd.read_csv(ep)
    t = pd.read_csv(tp)
    need = set(KEYS + ['k', 'family_1based'])
    if not need.issubset(e.columns) or not need.issubset(t.columns):
        raise RuntimeError(f'Assignment schema mismatch in {split}')
    e = e[e['k'].astype(int) == int(kE)][KEYS + ['family_1based']].copy()
    t = t[t['k'].astype(int) == int(kT)][KEYS + ['family_1based']].copy()
    e = e.rename(columns={'family_1based': 'E'})
    t = t.rename(columns={'family_1based': 'T'})
    if e.duplicated(KEYS).any() or t.duplicated(KEYS).any():
        raise RuntimeError(f'Non-unique assignment keys in {split} K{int(kE)}/K{int(kT)}')
    m = e.merge(t, on=KEYS, how='inner', validate='one_to_one')
    audit = {'split': split, 'experience_k': int(kE), 'transition_k': int(kT), 'n_experience_rows': int(len(e)), 'n_transition_rows': int(len(t)), 'n_common_rows': int(len(m)), 'n_common_participants': int(m['subject_id'].nunique()), 'coverage_vs_experience': float(len(m) / len(e)) if len(e) else float('nan'), 'coverage_vs_transition': float(len(m) / len(t)) if len(t) else float('nan')}
    return (m, audit)

def load_signature_pair(project: Path, split: str, common: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], dict]:
    ep = project / INPUTS[split]['exp_sig'][0]
    tp = project / INPUTS[split]['trans_sig'][0]
    e = pd.read_csv(ep)
    t = pd.read_csv(tp)
    for frame, role in [(e, 'experience'), (t, 'transition')]:
        if not set(KEYS).issubset(frame.columns):
            raise RuntimeError(f'Missing signature keys: {split} {role}')
        if frame.duplicated(KEYS).any():
            raise RuntimeError(f'Duplicate signature keys: {split} {role}')
    std_e = sorted([c for c in e.columns if c.startswith('standardized__')])
    std_t = sorted([c for c in t.columns if c.startswith('standardized__')])
    tr_e = sorted([c for c in e.columns if c.startswith('transformed__') and (not c.startswith('transformed__participant'))])
    tr_t = sorted([c for c in t.columns if c.startswith('transformed__') and (not c.startswith('transformed__participant'))])
    if std_e != std_t or len(std_e) != 14:
        raise RuntimeError(f'14D standardized coordinate schema mismatch in {split}: {len(std_e)} vs {len(std_t)}')
    if tr_e != tr_t or len(tr_e) != 14:
        raise RuntimeError(f'14D transformed coordinate schema mismatch in {split}: {len(tr_e)} vs {len(tr_t)}')
    keep_e = KEYS + std_e + tr_e
    keep_t = KEYS + std_t + tr_t
    e = e[keep_e].rename(columns={c: f'E::{c}' for c in std_e + tr_e})
    t = t[keep_t].rename(columns={c: f'T::{c}' for c in std_t + tr_t})
    g = common.merge(e, on=KEYS, how='inner', validate='one_to_one')
    g = g.merge(t, on=KEYS, how='inner', validate='one_to_one')
    audit = {'split': split, 'common_assignment_rows': int(len(common)), 'geometry_rows': int(len(g)), 'geometry_participants': int(g['subject_id'].nunique()), 'geometry_coverage': float(len(g) / len(common)) if len(common) else float('nan'), 'n_standardized_coordinates': len(std_e), 'n_transformed_coordinates': len(tr_e), 'standardized_schema_identical': bool(std_e == std_t), 'transformed_schema_identical': bool(tr_e == tr_t)}
    return (g, std_e, tr_e, audit)

def participant_balanced_centroid(df: pd.DataFrame, value_cols: List[str], mask: np.ndarray) -> np.ndarray:
    sub = df.loc[mask, ['subject_id'] + value_cols].copy()
    if len(sub) == 0:
        return np.full(len(value_cols), np.nan)
    pm = sub.groupby('subject_id', sort=False)[value_cols].mean()
    return pm.mean(axis=0).to_numpy(dtype=float)

def geometry_metrics_for_mapping(df: pd.DataFrame, cols: List[str], mapping: Tuple[int, int, int, int], prefix: str) -> dict:
    ecols = [f'E::{c}' for c in cols]
    tcols = [f'T::{c}' for c in cols]
    E = df['E'].to_numpy(dtype=int)
    T = df['T'].to_numpy(dtype=int)
    mapped = np.array([mapping[e - 1] for e in E], dtype=int)
    g1 = participant_balanced_centroid(df, ecols, mapped == 1)
    g2 = participant_balanced_centroid(df, ecols, mapped == 2)
    t1 = participant_balanced_centroid(df, tcols, T == 1)
    t2 = participant_balanced_centroid(df, tcols, T == 2)
    return {f'{prefix}_coarsened_E1_cos_T1': cosine(g1, t1), f'{prefix}_coarsened_E2_cos_T2': cosine(g2, t2), f'{prefix}_contrast_cosine': cosine(g2 - g1, t2 - t1)}

def pairwise_geometry_rows(split: str, df: pd.DataFrame, cols: List[str], layer: str, mapping: Tuple[int, int, int, int]) -> List[dict]:
    rows = []
    ecols = [f'E::{c}' for c in cols]
    tcols = [f'T::{c}' for c in cols]
    E = df['E'].to_numpy(dtype=int)
    T = df['T'].to_numpy(dtype=int)
    ecent = {e: participant_balanced_centroid(df, ecols, E == e) for e in range(1, 5)}
    tcent = {t: participant_balanced_centroid(df, tcols, T == t) for t in (1, 2)}
    for e in range(1, 5):
        sims = {t: cosine(ecent[e], tcent[t]) for t in (1, 2)}
        mapped_t = mapping[e - 1]
        rows.append({'split': split, 'layer': layer, 'experience_family': e, 'transition_family': 1, 'cosine': sims[1], 'is_fixed_mapping_target': mapped_t == 1, 'mapped_minus_other_cosine': sims[mapped_t] - sims[3 - mapped_t]})
        rows.append({'split': split, 'layer': layer, 'experience_family': e, 'transition_family': 2, 'cosine': sims[2], 'is_fixed_mapping_target': mapped_t == 2, 'mapped_minus_other_cosine': sims[mapped_t] - sims[3 - mapped_t]})
    return rows

def geometry_bootstrap_contrast(df: pd.DataFrame, cols: List[str], mapping: Tuple[int, int, int, int], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    subjects = sorted(df['subject_id'].astype(str).unique())
    ecols = [f'E::{c}' for c in cols]
    tcols = [f'T::{c}' for c in cols]
    cache = {}
    for s in subjects:
        g = df[df['subject_id'].astype(str) == s]
        E = g['E'].to_numpy(dtype=int)
        T = g['T'].to_numpy(dtype=int)
        mapped = np.array([mapping[e - 1] for e in E], dtype=int)
        rec = {}
        for lab in (1, 2):
            x = g.loc[mapped == lab, ecols]
            rec[f'E{lab}'] = x.mean(axis=0).to_numpy(dtype=float) if len(x) else None
            y = g.loc[T == lab, tcols]
            rec[f'T{lab}'] = y.mean(axis=0).to_numpy(dtype=float) if len(y) else None
        cache[s] = rec
    out = np.full(n_boot, np.nan, dtype=float)
    n = len(subjects)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        vec = {}
        valid = True
        for key in ('E1', 'E2', 'T1', 'T2'):
            vals = [cache[subjects[i]][key] for i in draw if cache[subjects[i]][key] is not None]
            if not vals:
                valid = False
                break
            vec[key] = np.mean(np.stack(vals), axis=0)
        if valid:
            out[b] = cosine(vec['E2'] - vec['E1'], vec['T2'] - vec['T1'])
    return out

def identity_control_k2(project: Path, split: str) -> dict:
    m, _ = load_assign_pair(project, split, 2, 2)
    ari = float(adjusted_rand_score(m['E'], m['T']))
    nmi = float(normalized_mutual_info_score(m['E'], m['T']))
    subjects = sorted(m['subject_id'].unique())
    tabs = []
    for s in subjects:
        g = m[m['subject_id'] == s]
        tab = np.zeros((2, 2), float)
        w = np.full(len(g), 1 / len(g), float)
        np.add.at(tab, (g['E'].to_numpy(int) - 1, g['T'].to_numpy(int) - 1), w)
        tabs.append(tab)
    tab = np.sum(tabs, axis=0)
    acc_identity = float((tab[0, 0] + tab[1, 1]) / tab.sum())
    acc_swap = float((tab[0, 1] + tab[1, 0]) / tab.sum())
    return {'split': split, 'common_trials': int(len(m)), 'common_participants': int(m['subject_id'].nunique()), 'ARI': ari, 'NMI': nmi, 'best_participant_balanced_label_accuracy': max(acc_identity, acc_swap), 'best_alignment': 'E1->T1;E2->T2' if acc_identity >= acc_swap else 'E1->T2;E2->T1'}

def verify_sources(project: Path) -> List[dict]:
    rows = []
    for split in ('Discovery', 'Validation', 'Confirmation'):
        for role, (rel, expected) in INPUTS[split].items():
            p = project / rel
            if not p.is_file():
                raise RuntimeError(f'Missing fixed source: {p}')
            obs = sha256_file(p)
            if obs != expected:
                raise RuntimeError(f'fixed source SHA drift: {split} {role}\nexpected {expected}\nobserved {obs}{p}')
            rows.append({'split': split, 'role': role, 'path': str(p), 'sha256': obs, 'size_bytes': p.stat().st_size})
    lock_candidates = list(project.glob('outputs/interpretability/cross_split_semantic_report_synthesis_final_crosssplit_semantic_report_synthesis/qc/*final_output_lock_hashes.json'))
    if len(lock_candidates) != 1:
        raise RuntimeError(f'Expected exactly one cross_split_semantic_report_synthesis final lock, found {len(lock_candidates)}')
    lock_sha = sha256_file(lock_candidates[0])
    if lock_sha != cross_split_semantic_report_synthesis_FINAL_LOCK_SHA:
        raise RuntimeError(f'cross_split_semantic_report_synthesis final lock SHA drift: expected {cross_split_semantic_report_synthesis_FINAL_LOCK_SHA}, observed {lock_sha}')
    rows.append({'split': 'BOUNDARY', 'role': 'cross_split_semantic_report_synthesis_final_lock', 'path': str(lock_candidates[0]), 'sha256': lock_sha, 'size_bytes': lock_candidates[0].stat().st_size})
    return rows

def write_lock(outroot: Path) -> Path:
    records = []
    for p in sorted(outroot.rglob('*')):
        if p.is_file() and p.name != 'experience_transition_relational_analysis_final_output_lock_hashes.json':
            records.append({'relative_path': str(p.relative_to(outroot)), 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size})
    target = outroot / 'qc' / 'experience_transition_relational_analysis_final_output_lock_hashes.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'files': records, 'n_files': len(records)}, indent=2), encoding='utf-8')
    return target

def run(project: Path, n_perm: int, n_boot: int, source_path: Path) -> None:
    if n_perm < 99 or n_boot < 99:
        if os.environ.get('experience_transition_relational_analysis_ALLOW_SMALL_TEST') != '1':
            raise RuntimeError('n_perm and n_boot must each be >=99 outside development smoke tests')
    source_rows = verify_sources(project)
    outroot = project / OUT_REL
    if outroot.exists():
        shutil.rmtree(outroot)
    (outroot / 'tables').mkdir(parents=True, exist_ok=True)
    (outroot / 'qc').mkdir(parents=True, exist_ok=True)
    (outroot / 'reports').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(outroot / 'qc' / 'experience_transition_relational_analysis_source_rehash.csv', index=False)
    rng_perm = np.random.default_rng(SEED)
    rng_boot = np.random.default_rng(SEED + 1)
    rng_geom = np.random.default_rng(SEED + 2)
    data = {}
    join_rows = []
    geom_audits = []
    for split in ('Discovery', 'Validation', 'Confirmation'):
        m, audit = load_assign_pair(project, split, 4, 2)
        expected_n, expected_p = EXPECTED_COMMON[split]
        if len(m) != expected_n or m['subject_id'].nunique() != expected_p:
            raise RuntimeError(f"Historical common-trial identity drift in {split}: got {len(m)}/{m['subject_id'].nunique()}, expected {expected_n}/{expected_p}")
        if set(m['E'].astype(int).unique()) != {1, 2, 3, 4}:
            raise RuntimeError(f'Missing experience K4 family in {split}')
        if set(m['T'].astype(int).unique()) != {1, 2}:
            raise RuntimeError(f'Missing transition K2 family in {split}')
        join_rows.append(audit)
        g, std_cols, tr_cols, ga = load_signature_pair(project, split, m)
        if ga['geometry_coverage'] < 0.95:
            raise RuntimeError(f"Geometry coverage <95% in {split}: {ga['geometry_coverage']}")
        geom_audits.append(ga)
        subs, subtabs = per_subject_tables(m)
        data[split] = {'m': m, 'subjects': subs, 'subtabs': subtabs, 'g': g, 'std_cols': std_cols, 'tr_cols': tr_cols}
    pd.DataFrame(join_rows).to_csv(outroot / 'qc' / 'experience_transition_relational_analysis_assignment_join_audit.csv', index=False)
    pd.DataFrame(geom_audits).to_csv(outroot / 'qc' / 'experience_transition_relational_analysis_geometry_schema_audit.csv', index=False)
    Dtab = data['Discovery']['subtabs'].sum(axis=0)
    discovery_map_rows = []
    for mp in MAPS:
        discovery_map_rows.append({'mapping': map_label_tuple(mp), 'balanced_accuracy': ba_from_table(Dtab, mp), 'mapping_tuple': ''.join(map(str, mp))})
    discovery_map_rows = sorted(discovery_map_rows, key=lambda r: (-r['balanced_accuracy'], r['mapping_tuple']))
    best_tuple_str = discovery_map_rows[0]['mapping_tuple']
    best_map = tuple((int(x) for x in best_tuple_str))
    pd.DataFrame(discovery_map_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_discovery_all_k4_to_k2_maps.csv', index=False)
    d_null_best = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        tabp = shuffled_subject_table(data['Discovery']['m'], rng_perm)
        d_null_best[b] = max((ba_from_table(tabp, mp) for mp in MAPS))
    d_obs = ba_from_table(Dtab, best_map)
    d_sel_p = permutation_p(d_obs, d_null_best)
    h1_rows = []
    h2_rows = []
    directional_rows = []
    perm_summary_rows = []
    boot_cache = {}
    for split in ('Discovery', 'Validation', 'Confirmation'):
        m = data[split]['m']
        subtabs = data[split]['subtabs']
        tab = subtabs.sum(axis=0)
        boots = bootstrap_tables(subtabs, n_boot, rng_boot)
        boot_cache[split] = boots
        ba = ba_from_table(tab, best_map)
        ari, nmi_unw = fixed_mapping_ari_nmi(m, best_map)
        ba_boot = np.array([ba_from_table(x, best_map) for x in boots], dtype=float)
        ba_lo, ba_hi = percentile_ci(ba_boot)
        null_ba = np.empty(n_perm, dtype=float)
        null_nmi = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            tp = shuffled_subject_table(m, rng_perm)
            null_ba[b] = ba_from_table(tp, best_map)
            null_nmi[b] = weighted_nmi_from_table(tp)[0]
        p_ba_fixed = permutation_p(ba, null_ba)
        wnmi, mi, he, ht, reduction = weighted_nmi_from_table(tab)
        nmi_boot = np.array([weighted_nmi_from_table(x)[0] for x in boots], dtype=float)
        nmi_lo, nmi_hi = percentile_ci(nmi_boot)
        red_boot = np.array([weighted_nmi_from_table(x)[4] for x in boots], dtype=float)
        red_lo, red_hi = percentile_ci(red_boot)
        p_nmi = permutation_p(wnmi, null_nmi)
        h1_rows.append({'split': split, 'mapping': map_label_tuple(best_map), 'common_trials': len(m), 'common_participants': m['subject_id'].nunique(), 'participant_balanced_BA': ba, 'BA_boot95_lo': ba_lo, 'BA_boot95_hi': ba_hi, 'fixed_mapping_permutation_p': p_ba_fixed, 'selection_corrected_permutation_p': d_sel_p if split == 'Discovery' else np.nan, 'ARI_fixed_coarsening_vs_transition': ari, 'NMI_unweighted_fixed_coarsening_vs_transition': nmi_unw})
        h2_rows.append({'split': split, 'common_trials': len(m), 'common_participants': m['subject_id'].nunique(), 'participant_balanced_weighted_NMI': wnmi, 'NMI_boot95_lo': nmi_lo, 'NMI_boot95_hi': nmi_hi, 'mutual_information_nats': mi, 'H_experience_nats': he, 'H_transition_nats': ht, 'transition_uncertainty_reduction_MI_over_HT': reduction, 'uncertainty_reduction_boot95_lo': red_lo, 'uncertainty_reduction_boot95_hi': red_hi, 'within_participant_permutation_p': p_nmi})
        perm_summary_rows.append({'split': split, 'metric': 'H1_BA_fixed_map', 'observed': ba, 'null_mean': float(np.mean(null_ba)), 'null_sd': float(np.std(null_ba, ddof=1)), 'null_95': float(np.quantile(null_ba, 0.95)), 'p': p_ba_fixed})
        perm_summary_rows.append({'split': split, 'metric': 'H2_weighted_NMI', 'observed': wnmi, 'null_mean': float(np.mean(null_nmi)), 'null_sd': float(np.std(null_nmi, ddof=1)), 'null_95': float(np.quantile(null_nmi, 0.95)), 'p': p_nmi})
        rowden = tab.sum(axis=1)
        colden = tab.sum(axis=0)
        for e in range(1, 5):
            for t in (1, 2):
                pte = tab[e - 1, t - 1] / rowden[e - 1] if rowden[e - 1] > 0 else np.nan
                pet = tab[e - 1, t - 1] / colden[t - 1] if colden[t - 1] > 0 else np.nan
                b_pte = np.array([x[e - 1, t - 1] / x[e - 1, :].sum() if x[e - 1, :].sum() > 0 else np.nan for x in boots])
                b_pet = np.array([x[e - 1, t - 1] / x[:, t - 1].sum() if x[:, t - 1].sum() > 0 else np.nan for x in boots])
                pte_lo, pte_hi = percentile_ci(b_pte)
                pet_lo, pet_hi = percentile_ci(b_pet)
                directional_rows.append({'split': split, 'experience_family': e, 'transition_family': t, 'participant_weighted_mass': tab[e - 1, t - 1], 'p_transition_given_experience': pte, 'pte_boot95_lo': pte_lo, 'pte_boot95_hi': pte_hi, 'p_experience_given_transition': pet, 'pet_boot95_lo': pet_lo, 'pet_boot95_hi': pet_hi, 'fixed_mapping_target': best_map[e - 1] == t})
    perm_summary_rows.append({'split': 'Discovery', 'metric': 'H1_BA_best_of_14_selection_corrected', 'observed': d_obs, 'null_mean': float(np.mean(d_null_best)), 'null_sd': float(np.std(d_null_best, ddof=1)), 'null_95': float(np.quantile(d_null_best, 0.95)), 'p': d_sel_p})
    h1 = pd.DataFrame(h1_rows)
    h2 = pd.DataFrame(h2_rows)
    pset = {'H1_Validation': float(h1.loc[h1.split == 'Validation', 'fixed_mapping_permutation_p'].iloc[0]), 'H1_Confirmation': float(h1.loc[h1.split == 'Confirmation', 'fixed_mapping_permutation_p'].iloc[0]), 'H2_Validation': float(h2.loc[h2.split == 'Validation', 'within_participant_permutation_p'].iloc[0]), 'H2_Confirmation': float(h2.loc[h2.split == 'Confirmation', 'within_participant_permutation_p'].iloc[0])}
    adj = holm_adjust(pset)
    h1['heldout_parent_holm_p'] = np.nan
    h2['heldout_parent_holm_p'] = np.nan
    for split in ('Validation', 'Confirmation'):
        h1.loc[h1.split == split, 'heldout_parent_holm_p'] = adj[f'H1_{split}']
        h2.loc[h2.split == split, 'heldout_parent_holm_p'] = adj[f'H2_{split}']
    h1.to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_H1_coarsening_metrics.csv', index=False)
    h2.to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_H2_conditional_association_metrics.csv', index=False)
    pd.DataFrame(directional_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_directional_conditionals.csv', index=False)
    pd.DataFrame(perm_summary_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_permutation_summary.csv', index=False)
    identity_rows = [identity_control_k2(project, s) for s in ('Discovery', 'Validation', 'Confirmation')]
    pd.DataFrame(identity_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_K2_identity_control.csv', index=False)
    geom_rows = []
    pair_rows = []
    rank_rows = []
    for split in ('Discovery', 'Validation', 'Confirmation'):
        g = data[split]['g']
        for layer, cols in [('standardized_14D', data[split]['std_cols']), ('transformed_14D_sensitivity', data[split]['tr_cols'])]:
            gm = geometry_metrics_for_mapping(g, cols, best_map, layer)
            boot = geometry_bootstrap_contrast(g, cols, best_map, n_boot, rng_geom)
            lo, hi = percentile_ci(boot)
            fixed_cos = gm[f'{layer}_contrast_cosine']
            allmaps = []
            for mp in MAPS:
                x = geometry_metrics_for_mapping(g, cols, mp, layer)[f'{layer}_contrast_cosine']
                allmaps.append((mp, x))
            finite = sorted([x for x in allmaps if np.isfinite(x[1])], key=lambda a: a[1], reverse=True)
            rank = next((i + 1 for i, (mp, x) in enumerate(finite) if mp == best_map), np.nan)
            percentile = 1.0 - (rank - 1) / max(1, len(finite) - 1) if np.isfinite(rank) and len(finite) > 1 else np.nan
            geom_rows.append({'split': split, 'layer': layer, 'fixed_mapping': map_label_tuple(best_map), 'coarsened_E_group1_cos_transition_T1': gm[f'{layer}_coarsened_E1_cos_T1'], 'coarsened_E_group2_cos_transition_T2': gm[f'{layer}_coarsened_E2_cos_T2'], 'fixed_mapping_contrast_cosine': fixed_cos, 'contrast_boot95_lo': lo, 'contrast_boot95_hi': hi, 'fixed_mapping_geometry_rank_among_14': rank, 'fixed_mapping_geometry_percentile_among_14': percentile, 'inferential_status': 'DESCRIPTIVE_ONLY'})
            for i, (mp, x) in enumerate(finite, start=1):
                rank_rows.append({'split': split, 'layer': layer, 'mapping': map_label_tuple(mp), 'contrast_cosine': x, 'rank': i, 'is_fixed_discovery_mapping': mp == best_map})
            pair_rows.extend(pairwise_geometry_rows(split, g, cols, layer, best_map))
    pd.DataFrame(geom_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_H3_geometry_correspondence.csv', index=False)
    pd.DataFrame(rank_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_H3_all_mapping_geometry_ranks.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_H3_pairwise_family_cosines.csv', index=False)

    def row(df, split):
        return df[df.split == split].iloc[0]
    v1, outcome_model_integrity = (row(h1, 'Validation'), row(h1, 'Confirmation'))
    v2, c2 = (row(h2, 'Validation'), row(h2, 'Confirmation'))
    h1_pass = bool(v1.participant_balanced_BA > 0.5 and outcome_model_integrity.participant_balanced_BA > 0.5 and (v1.BA_boot95_lo > 0.5) and (outcome_model_integrity.BA_boot95_lo > 0.5) and (v1.heldout_parent_holm_p < 0.05) and (outcome_model_integrity.heldout_parent_holm_p < 0.05))
    h2_pass = bool(v2.heldout_parent_holm_p < 0.05 and c2.heldout_parent_holm_p < 0.05)
    decisions = pd.DataFrame([{'hypothesis': 'H1_K4_experience_to_K2_transition_coarsening', 'status': 'PASS' if h1_pass else 'NOT_REPLICATED', 'gate': 'Both heldout: BA>0.5, bootstrap lower95>0.5, Holm p<0.05', 'max_claim_if_pass': 'Internally replicated statistical coarsening alignment under the fixed Discovery map; not family identity or mechanism.'}, {'hypothesis': 'H2_experience_transition_conditional_association', 'status': 'PASS' if h2_pass else 'NOT_REPLICATED', 'gate': 'Both heldout: Holm-adjusted within-participant permutation p<0.05; effect size reported as weighted NMI/uncertainty reduction.', 'max_claim_if_pass': 'Internally replicated conditional association; not determinism, independence, or biological mechanism.'}, {'hypothesis': 'H3_shared_14D_geometry_correspondence', 'status': 'DESCRIPTIVE_ONLY', 'gate': 'No inferential pass gate. Fixed-map contrast cosine, bootstrap CI, and rank among 14 mappings are descriptive.', 'max_claim_if_pass': 'Shared-coordinate profile resemblance only; not metric identity, source identity, or mechanism.'}])
    decisions.to_csv(outroot / 'tables' / 'experience_transition_relational_analysis_replication_decisions.csv', index=False)
    summary = {'phase': 'experience_transition_relational_analysis_experience_transition_relational_architecture', 'scientific_status': 'POST_HOC_SEPARATE_FROM_PRESPECIFIED_CORE', 'timestamp': datetime.now().isoformat(), 'seed': SEED, 'n_permutations': int(n_perm), 'n_bootstrap': int(n_boot), 'cross_split_semantic_report_synthesis_final_lock_verified': True, 'prespecified_primary_analysis_modified': False, 'behavioral_or_semantic_values_opened': False, 'discovery_selected_mapping': map_label_tuple(best_map), 'H1_status': 'PASS' if h1_pass else 'NOT_REPLICATED', 'H2_status': 'PASS' if h2_pass else 'NOT_REPLICATED', 'H3_status': 'DESCRIPTIVE_ONLY', 'historical_note': 'cross_split_report_transition_structure_synthesis already reported low ARI/NMI plus descriptive directional conditionals. experience_transition_relational_analysis adds formal fixed-map K4→K2 coarsening inference and descriptive geometry correspondence.', 'claim_boundary': ['No biological independence claim', 'No family identity claim from coarsening alone', 'No causal or source-localized interpretation', 'No external replication claim', 'H3 geometry is descriptive because experience and transition are distinct stage-to-stage representations']}
    (outroot / 'qc' / 'experience_transition_relational_analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = []
    report.append('# Experience↔transition relational architecture')
    report.append('**Post-hoc branch; prespecified primary analysis/structural_replication/signal_space_characterization/primary_analysis_reference were not modified.**')
    report.append(f'Discovery-selected K4→K2 mapping: **{map_label_tuple(best_map)}**')
    report.append('## H1 — fixed coarsening')
    report.append(h1.to_markdown(index=False))
    report.append('\n\n## H2 — conditional association')
    report.append(h2.to_markdown(index=False))
    report.append('\n\n## Decisions')
    report.append(decisions.to_markdown(index=False))
    report.append('\n\n## Interpretation boundary')
    report.append('A positive result can establish only internally replicated statistical alignment/association under the prespecified representations. It does not establish that experience and transition families are the same biological states, independent mechanisms, causal stages, or source-localized generators. H3 is descriptive only.')
    (outroot / 'reports' / 'experience_transition_relational_analysis_report.md').write_text(''.join(report), encoding='utf-8')
    shutil.copy2(source_path, outroot / 'qc' / source_path.name)
    lock_path = write_lock(outroot)
    lock_sha = sha256_file(lock_path)
    print('experience_transition_relational_analysis PASS')
    print(f'cross_split_semantic_report_synthesis final lock verified: {cross_split_semantic_report_synthesis_FINAL_LOCK_SHA}')
    print(f"Common trial identity: Discovery={EXPECTED_COMMON['Discovery'][0]}, Validation={EXPECTED_COMMON['Validation'][0]}, Confirmation={EXPECTED_COMMON['Confirmation'][0]}")
    print(f'Discovery-selected mapping: {map_label_tuple(best_map)}')
    print(f"H1 coarsening: {('PASS' if h1_pass else 'NOT_REPLICATED')}")
    print(f"H2 conditional association: {('PASS' if h2_pass else 'NOT_REPLICATED')}")
    print('H3 geometry: DESCRIPTIVE_ONLY')
    print(f'Final lock SHA-256: {lock_sha}')

def self_test() -> None:
    rng = np.random.default_rng(1)
    rows = []
    true_map = (1, 1, 2, 2)
    for s in range(10):
        for i in range(48):
            e = i % 4 + 1
            t = true_map[e - 1]
            if rng.random() < 0.1:
                t = 3 - t
            rows.append({'subject_id': f'sub-{s:02d}', 'raw_behavior_row_id': f's{s}|{i}', 'E': e, 'T': t})
    df = pd.DataFrame(rows)
    _, tabs = per_subject_tables(df)
    tab = tabs.sum(axis=0)
    vals = [(ba_from_table(tab, mp), mp) for mp in MAPS]
    best = sorted(vals, key=lambda x: (-x[0], x[1]))[0]
    assert best[1] == true_map, best
    assert best[0] > 0.8
    nmi = weighted_nmi_from_table(tab)[0]
    assert nmi > 0.3
    p = holm_adjust({'a': 0.01, 'b': 0.02, 'c': 0.04, 'd': 0.2})
    assert all((0 <= x <= 1 for x in p.values()))
    assert len(MAPS) == 14
    print('experience_transition_relational_analysis self-test PASS')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('<PROJECT_ROOT>'))
    ap.add_argument('--n-perm', type=int, default=DEFAULT_N_PERM)
    ap.add_argument('--n-boot', type=int, default=DEFAULT_N_BOOT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run(args.project, args.n_perm, args.n_boot, Path(__file__).resolve())
if __name__ == '__main__':
    main()