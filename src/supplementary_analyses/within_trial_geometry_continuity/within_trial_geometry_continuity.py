# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Post-hoc within-trial experience→transition 14D geometry continuity.

Separate post-hoc branch. Does NOT modify prespecified primary analysis, structural_replication, signal_space_characterization, or primary_analysis_reference.

Primary question:
Does the continuous 14D experience signature of a trial align with that same
trial's transition signature more strongly than with other transition
signatures from the same participant and context?

Primary representation: standardized 14D coordinates.
Sensitivity representation: transformed 14D coordinates."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
SEED = 20260826
DEFAULT_N_PERM = 5000
DEFAULT_N_BOOT = 5000
cross_split_semantic_report_synthesis_FINAL_LOCK_SHA = '967c5cc35877e0e788a672d621c7825300eba7ab88ca2508a49030b081d2bd25'
CANONICAL_SHA = '70eff894db94ed51d65d103392004162c839bfd3f2850dd3fc15ff3685fd28a6'
CANONICAL_DEFAULT = Path('<CANONICAL_BEHAVIORAL_TABLE>')
OUT_REL = Path('outputs/supplementary_analyses/within_trial_geometry_continuity_within_trial_geometry_continuity')
KEYS = ['raw_behavior_row_id', 'subject_id']
INPUTS = {'Discovery': {'exp_assign': ('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv', '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'), 'trans_assign': ('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv', '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'), 'exp_sig': ('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv', 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'), 'trans_sig': ('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv', '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83')}, 'Validation': {'exp_assign': ('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv', 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'), 'trans_assign': ('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv', '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'), 'exp_sig': ('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv', '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'), 'trans_sig': ('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv', 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067')}, 'Confirmation': {'exp_assign': ('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv', '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'), 'trans_assign': ('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv', 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'), 'exp_sig': ('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv', '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'), 'trans_sig': ('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv', '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6')}}
EXPECTED_COMMON = {'Discovery': (1237, 31), 'Validation': (437, 12), 'Confirmation': (459, 12)}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def percentile_ci(x: np.ndarray, alpha: float=0.05) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (float('nan'), float('nan'))
    return (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2)))

def holm_adjust(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * float(p))
        running = max(running, adj)
        out[name] = running
    return out

def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    n = np.linalg.norm(x, axis=1)
    out = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(n) & (n > 0) & np.isfinite(x).all(axis=1)
    out[good] = x[good] / n[good, None]
    return out

def verify_sources(project: Path, canonical: Path) -> List[dict]:
    rows = []
    for split in ('Discovery', 'Validation', 'Confirmation'):
        for role, (rel, expected) in INPUTS[split].items():
            p = project / rel
            if not p.is_file():
                raise RuntimeError(f'Missing fixed source: {p}')
            obs = sha256_file(p)
            if obs != expected:
                raise RuntimeError(f'fixed source SHA drift: {split} {role}: expected {expected}, observed {obs}')
            rows.append({'split': split, 'role': role, 'path': str(p), 'sha256': obs, 'size_bytes': p.stat().st_size})
    if not canonical.is_file():
        raise RuntimeError(f'Missing canonical master: {canonical}')
    csha = sha256_file(canonical)
    if csha != CANONICAL_SHA:
        raise RuntimeError(f'Canonical master SHA drift: expected {CANONICAL_SHA}, observed {csha}')
    rows.append({'split': 'BOUNDARY', 'role': 'canonical_master_context_only', 'path': str(canonical), 'sha256': csha, 'size_bytes': canonical.stat().st_size})
    locks = list(project.glob('outputs/interpretability/cross_split_semantic_report_synthesis_final_crosssplit_semantic_report_synthesis/qc/*final_output_lock_hashes.json'))
    if len(locks) != 1:
        raise RuntimeError(f'Expected exactly one cross_split_semantic_report_synthesis final lock, found {len(locks)}')
    lsha = sha256_file(locks[0])
    if lsha != cross_split_semantic_report_synthesis_FINAL_LOCK_SHA:
        raise RuntimeError(f'cross_split_semantic_report_synthesis final lock SHA drift: expected {cross_split_semantic_report_synthesis_FINAL_LOCK_SHA}, observed {lsha}')
    rows.append({'split': 'BOUNDARY', 'role': 'cross_split_semantic_report_synthesis_final_lock', 'path': str(locks[0]), 'sha256': lsha, 'size_bytes': locks[0].stat().st_size})
    return rows

def load_common_and_signatures(project: Path, split: str) -> Tuple[pd.DataFrame, List[str], List[str], dict]:
    ea = pd.read_csv(project / INPUTS[split]['exp_assign'][0])
    ta = pd.read_csv(project / INPUTS[split]['trans_assign'][0])
    need = set(KEYS + ['k', 'family_1based'])
    if not need.issubset(ea.columns) or not need.issubset(ta.columns):
        raise RuntimeError(f'Assignment schema mismatch: {split}')
    ea4 = ea[ea['k'].astype(int) == 4][KEYS + ['family_1based']].rename(columns={'family_1based': 'E4'})
    ta2 = ta[ta['k'].astype(int) == 2][KEYS + ['family_1based']].rename(columns={'family_1based': 'T2'})
    if ea4.duplicated(KEYS).any() or ta2.duplicated(KEYS).any():
        raise RuntimeError(f'Non-unique K4/K2 assignment keys: {split}')
    common = ea4.merge(ta2, on=KEYS, how='inner', validate='one_to_one')
    exp_n, exp_p = EXPECTED_COMMON[split]
    if len(common) != exp_n or common['subject_id'].nunique() != exp_p:
        raise RuntimeError(f"Historical common-trial identity mismatch {split}: {len(common)}/{common['subject_id'].nunique()} expected {exp_n}/{exp_p}")
    es = pd.read_csv(project / INPUTS[split]['exp_sig'][0])
    ts = pd.read_csv(project / INPUTS[split]['trans_sig'][0])
    for frame, role in ((es, 'experience'), (ts, 'transition')):
        if not set(KEYS).issubset(frame.columns) or frame.duplicated(KEYS).any():
            raise RuntimeError(f'Signature key schema/nonuniqueness: {split} {role}')
    std_e = sorted((c for c in es.columns if c.startswith('standardized__')))
    std_t = sorted((c for c in ts.columns if c.startswith('standardized__')))
    tr_e = sorted((c for c in es.columns if c.startswith('transformed__') and (not c.startswith('transformed__participant'))))
    tr_t = sorted((c for c in ts.columns if c.startswith('transformed__') and (not c.startswith('transformed__participant'))))
    if std_e != std_t or len(std_e) != 14:
        raise RuntimeError(f'Standardized 14D schema mismatch: {split}')
    if tr_e != tr_t or len(tr_e) != 14:
        raise RuntimeError(f'Transformed 14D schema mismatch: {split}')
    es = es[KEYS + std_e + tr_e].rename(columns={c: f'E::{c}' for c in std_e + tr_e})
    ts = ts[KEYS + std_t + tr_t].rename(columns={c: f'T::{c}' for c in std_t + tr_t})
    g = common.merge(es, on=KEYS, how='inner', validate='one_to_one').merge(ts, on=KEYS, how='inner', validate='one_to_one')
    if len(g) / len(common) < 0.99:
        raise RuntimeError(f'Signature coverage <99% in {split}: {len(g)}/{len(common)}')
    audit = {'split': split, 'common_trials': len(common), 'common_participants': common.subject_id.nunique(), 'geometry_rows': len(g), 'geometry_coverage': len(g) / len(common), 'n_standardized': len(std_e), 'n_transformed': len(tr_e)}
    return (g, std_e, tr_e, audit)

def attach_context(g: pd.DataFrame, master: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, dict]:
    raw = g['raw_behavior_row_id'].astype(str).str.strip()
    trial = pd.to_numeric(raw.str.extract('(?:^|\\|)trial-?0*(\\d+)$', expand=False), errors='coerce')
    embedded = raw.str.extract('(sub-\\d+)', expand=False)
    agree_mask = embedded.notna()
    agree = float((embedded[agree_mask].astype(str).to_numpy() == g.loc[agree_mask, 'subject_id'].astype(str).to_numpy()).mean()) if agree_mask.any() else float('nan')
    if agree_mask.any() and agree < 1.0:
        raise RuntimeError(f'Embedded subject mismatch in {split}')
    gg = g.copy()
    gg['trial_number_global'] = trial.astype('Int64')
    mm = master[['subject_id', 'trial_number_global', 'context']].copy()
    if mm.duplicated(['subject_id', 'trial_number_global']).any():
        raise RuntimeError('Canonical context key is not unique')
    before = len(gg)
    gg = gg.merge(mm, on=['subject_id', 'trial_number_global'], how='inner', validate='one_to_one')
    coverage = len(gg) / before if before else float('nan')
    if coverage < 0.95:
        raise RuntimeError(f'Canonical context lineage <95% in {split}: {coverage}')
    audit = {'split': split, 'pre_context_rows': before, 'context_rows': len(gg), 'context_coverage': coverage, 'context_participants': gg.subject_id.nunique(), 'raw_id_parse_rate': float(trial.notna().mean()), 'embedded_subject_agreement': agree, 'contexts': ';'.join(sorted(gg.context.astype(str).unique()))}
    return (gg, audit)

def layer_arrays(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    e = df[[f'E::{c}' for c in cols]].to_numpy(float)
    t = df[[f'T::{c}' for c in cols]].to_numpy(float)
    return (normalize_rows(e), normalize_rows(t))

def group_indices(df: pd.DataFrame) -> List[np.ndarray]:
    return [x.to_numpy(int) for _, x in pd.Series(np.arange(len(df)), index=df.index).groupby([df['subject_id'].astype(str), df['context'].astype(str)])]

def continuity_components(df: pd.DataFrame, E: np.ndarray, T: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
    n = len(df)
    same = np.einsum('ij,ij->i', E, T)
    mismatch = np.full(n, np.nan, float)
    groups = []
    for _, idxs in df.groupby(['subject_id', 'context'], sort=False).indices.items():
        idx = np.asarray(idxs, int)
        groups.append(idx)
        valid = np.isfinite(E[idx]).all(axis=1) & np.isfinite(T[idx]).all(axis=1)
        ii = idx[valid]
        if len(ii) < 2:
            continue
        tsum = np.sum(T[ii], axis=0)
        ref = (tsum[None, :] - T[ii]) / (len(ii) - 1)
        mismatch[ii] = np.einsum('ij,ij->i', E[ii], ref)
    excess = same - mismatch
    trial = df[['raw_behavior_row_id', 'subject_id', 'context', 'E4', 'T2']].copy()
    trial['same_trial_cosine'] = same
    trial['within_participant_context_mismatch_cosine'] = mismatch
    trial['excess_cosine'] = excess
    subjects = sorted(df.subject_id.astype(str).unique())
    subj_idx = np.array([subjects.index(str(s)) for s in df.subject_id.astype(str)], int)
    return (trial, same, mismatch, excess, groups, subjects)

def participant_metrics(df: pd.DataFrame, trial: pd.DataFrame, split: str, layer: str) -> pd.DataFrame:
    tmp = trial.copy()
    tmp['subject_id'] = tmp['subject_id'].astype(str)
    rows = []
    for s, g in tmp.groupby('subject_id', sort=True):
        v = g['excess_cosine'].to_numpy(float)
        same = g['same_trial_cosine'].to_numpy(float)
        mis = g['within_participant_context_mismatch_cosine'].to_numpy(float)
        good = np.isfinite(v)
        rows.append({'split': split, 'layer': layer, 'subject_id': s, 'n_trials': len(g), 'n_primary_trials': int(good.sum()), 'mean_same_trial_cosine': float(np.nanmean(same)), 'mean_mismatch_cosine': float(np.nanmean(mis)), 'mean_excess_cosine': float(np.nanmean(v)) if good.any() else np.nan})
    return pd.DataFrame(rows)

def permutation_null(df: pd.DataFrame, E: np.ndarray, T: np.ndarray, mismatch: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Exact randomization null for the same-minus-mismatch statistic.

    Transition rows are permuted only within participant × context.  For every
    permutation, BOTH the putative same-trial cosine and its leave-one-out
    mismatched-transition reference are recomputed from that permuted
    assignment.  Keeping the observed mismatch reference fixed would shift the
    randomization null upward whenever observed same-trial alignment is
    positive, so that implementation is intentionally forbidden here.

    The ``mismatch`` argument is retained only for call-signature compatibility
    with the prespecified first implementation; it is not used in the null.
    """
    groups = [np.asarray(v, int) for _, v in df.groupby(['subject_id', 'context'], sort=False).indices.items()]
    subjects = sorted(df.subject_id.astype(str).unique())
    subj_rows = [np.flatnonzero(df.subject_id.astype(str).to_numpy() == s) for s in subjects]
    null = np.full(n_perm, np.nan, float)
    base = np.arange(len(df))
    for b in range(n_perm):
        perm = base.copy()
        for idx in groups:
            if len(idx) > 1:
                perm[idx] = rng.permutation(idx)
        Tp = T[perm]
        same_p = np.einsum('ij,ij->i', E, Tp)
        mismatch_p = np.full(len(df), np.nan, float)
        for idx in groups:
            valid = np.isfinite(E[idx]).all(axis=1) & np.isfinite(Tp[idx]).all(axis=1)
            ii = idx[valid]
            if len(ii) < 2:
                continue
            tsum = np.sum(Tp[ii], axis=0)
            ref = (tsum[None, :] - Tp[ii]) / (len(ii) - 1)
            mismatch_p[ii] = np.einsum('ij,ij->i', E[ii], ref)
        excess_p = same_p - mismatch_p
        vals = []
        for idx in subj_rows:
            v = excess_p[idx]
            if np.isfinite(v).any():
                vals.append(float(np.nanmean(v)))
        if vals:
            null[b] = float(np.mean(vals))
    return null

def bootstrap_subject_effect(pm: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    vals = pm['mean_excess_cosine'].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if not n:
        return np.full(n_boot, np.nan)
    out = np.empty(n_boot, float)
    for b in range(n_boot):
        out[b] = float(np.mean(vals[rng.integers(0, n, n)]))
    return out

def permutation_p(obs: float, null: np.ndarray) -> float:
    z = np.asarray(null, float)
    z = z[np.isfinite(z)]
    return float((1 + np.sum(z >= obs)) / (1 + len(z)))

def family_context_summary(df: pd.DataFrame, trial: pd.DataFrame, split: str, layer: str) -> pd.DataFrame:
    x = trial.copy()
    x['E4'] = df.E4.to_numpy(int)
    x['T2'] = df.T2.to_numpy(int)
    x['context'] = df.context.astype(str).to_numpy()
    rows = []
    for keys, g in x.groupby(['context', 'E4', 'T2'], sort=True):
        rows.append({'split': split, 'layer': layer, 'context': keys[0], 'experience_family_k4': keys[1], 'transition_family_k2': keys[2], 'n_trials': len(g), 'mean_same_trial_cosine': float(np.nanmean(g.same_trial_cosine)), 'mean_mismatch_cosine': float(np.nanmean(g.within_participant_context_mismatch_cosine)), 'mean_excess_cosine': float(np.nanmean(g.excess_cosine))})
    return pd.DataFrame(rows)

def write_lock(outroot: Path) -> Path:
    records = []
    for p in sorted(outroot.rglob('*')):
        if p.is_file() and p.name != 'within_trial_geometry_continuity_final_output_lock_hashes.json':
            records.append({'relative_path': str(p.relative_to(outroot)), 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size})
    target = outroot / 'qc' / 'within_trial_geometry_continuity_final_output_lock_hashes.json'
    target.write_text(json.dumps({'files': records, 'n_files': len(records)}, indent=2), encoding='utf-8')
    return target

def run(project: Path, canonical: Path, n_perm: int, n_boot: int, source: Path) -> None:
    if min(n_perm, n_boot) < 99 and os.environ.get('within_trial_geometry_continuity_ALLOW_SMALL_TEST') != '1':
        raise RuntimeError('n_perm and n_boot must each be >=99 outside development smoke tests')
    source_rows = verify_sources(project, canonical)
    boundary = verify_publication_boundary(project)
    master = pd.read_csv(canonical, usecols=['subject_id', 'trial_number_global', 'context'])
    outroot = project / OUT_REL
    if outroot.exists():
        shutil.rmtree(outroot)
    for d in ('tables', 'qc', 'reports'):
        (outroot / d).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(outroot / 'qc' / 'within_trial_geometry_continuity_source_rehash.csv', index=False)
    (outroot / 'qc' / 'within_trial_geometry_continuity_publication_boundary.json').write_text(json.dumps(boundary, indent=2), encoding='utf-8')
    join_rows = []
    context_rows = []
    schema_rows = []
    trial_outputs = []
    pm_outputs = []
    split_rows = []
    fam_outputs = []
    perm_rows = []
    rng = np.random.default_rng(SEED)
    layer_data = {}
    for split in ('Discovery', 'Validation', 'Confirmation'):
        g, std_cols, tr_cols, ja = load_common_and_signatures(project, split)
        join_rows.append(ja)
        g, ca = attach_context(g, master, split)
        context_rows.append(ca)
        schema_rows.append({'split': split, 'standardized_schema': '|'.join(std_cols), 'transformed_schema': '|'.join(tr_cols)})
        layer_data[split] = {}
        for layer, cols in (('standardized', std_cols), ('transformed', tr_cols)):
            E, T = layer_arrays(g, cols)
            trial, same, mis, exc, groups, subjects = continuity_components(g, E, T)
            trial.insert(0, 'split', split)
            trial.insert(1, 'layer', layer)
            trial_outputs.append(trial)
            pm = participant_metrics(g, trial, split, layer)
            pm_outputs.append(pm)
            obs = float(pm.mean_excess_cosine.mean())
            boots = bootstrap_subject_effect(pm, n_boot, rng)
            ci = percentile_ci(boots)
            null = permutation_null(g, E, T, mis, n_perm, rng)
            p = permutation_p(obs, null)
            row = {'split': split, 'layer': layer, 'n_rows': len(g), 'n_participants': g.subject_id.nunique(), 'participant_balanced_same_trial_cosine': float(pm.mean_same_trial_cosine.mean()), 'participant_balanced_mismatch_cosine': float(pm.mean_mismatch_cosine.mean()), 'participant_balanced_excess_cosine': obs, 'bootstrap_ci_low': ci[0], 'bootstrap_ci_high': ci[1], 'permutation_p_raw': p, 'null_mean': float(np.nanmean(null)), 'null_sd': float(np.nanstd(null, ddof=1))}
            split_rows.append(row)
            perm_rows.append({'split': split, 'layer': layer, 'n_perm': n_perm, 'observed_excess_cosine': obs, 'null_mean': row['null_mean'], 'null_sd': row['null_sd'], 'p_raw': p})
            fam_outputs.append(family_context_summary(g, trial, split, layer))
            layer_data[split][layer] = row
    pd.DataFrame(join_rows).to_csv(outroot / 'qc' / 'within_trial_geometry_continuity_assignment_geometry_join_audit.csv', index=False)
    pd.DataFrame(context_rows).to_csv(outroot / 'qc' / 'within_trial_geometry_continuity_context_lineage_audit.csv', index=False)
    pd.DataFrame(schema_rows).to_csv(outroot / 'qc' / 'within_trial_geometry_continuity_geometry_schema_audit.csv', index=False)
    pd.concat(trial_outputs, ignore_index=True).to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_trial_continuity_metrics.csv.gz', index=False, compression='gzip')
    pm_all = pd.concat(pm_outputs, ignore_index=True)
    pm_all.to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_participant_continuity_metrics.csv', index=False)
    split_df = pd.DataFrame(split_rows)
    primary = split_df[split_df.layer == 'standardized'].copy()
    held = primary[primary.split.isin(['Validation', 'Confirmation'])].copy()
    adj = holm_adjust({r.split: float(r.permutation_p_raw) for r in held.itertuples(index=False)})
    split_df['permutation_p_holm_primary_heldout'] = np.nan
    for split, padj in adj.items():
        split_df.loc[(split_df.layer == 'standardized') & (split_df.split == split), 'permutation_p_holm_primary_heldout'] = padj
    split_df.to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_split_continuity_summary.csv', index=False)
    pd.DataFrame(perm_rows).to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_permutation_summary.csv', index=False)
    pd.concat(fam_outputs, ignore_index=True).to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_family_context_descriptive.csv', index=False)
    decisions = []
    pass_primary = True
    for split in ('Validation', 'Confirmation'):
        r = split_df[(split_df.layer == 'standardized') & (split_df.split == split)].iloc[0]
        sens = split_df[(split_df.layer == 'transformed') & (split_df.split == split)].iloc[0]
        gate = bool(r.participant_balanced_excess_cosine > 0 and r.bootstrap_ci_low > 0 and (r.permutation_p_holm_primary_heldout < 0.05) and (sens.participant_balanced_excess_cosine > 0))
        pass_primary &= gate
        decisions.append({'hypothesis': 'H1_within_trial_continuous_geometry_continuity', 'split': split, 'pass': gate, 'primary_excess_cosine': r.participant_balanced_excess_cosine, 'primary_ci_low': r.bootstrap_ci_low, 'primary_holm_p': r.permutation_p_holm_primary_heldout, 'transformed_sensitivity_excess_cosine': sens.participant_balanced_excess_cosine})
    decisions.append({'hypothesis': 'H1_within_trial_continuous_geometry_continuity', 'split': 'CROSS_HELDOUT', 'pass': bool(pass_primary), 'primary_excess_cosine': np.nan, 'primary_ci_low': np.nan, 'primary_holm_p': np.nan, 'transformed_sensitivity_excess_cosine': np.nan})
    dec = pd.DataFrame(decisions)
    dec.to_csv(outroot / 'tables' / 'within_trial_geometry_continuity_replication_decisions.csv', index=False)
    summary = {'phase': 'within_trial_geometry_continuity_within_trial_geometry_continuity', 'status': 'PASS' if pass_primary else 'NOT_SUPPORTED', 'scientific_status': 'POST_HOC_SEPARATE_FROM_PRESPECIFIED_CORE', 'primary_replication_gate': bool(pass_primary), 'primary_definition': 'participant-balanced same-trial cosine minus within-participant×context mismatched cosine in standardized 14D coordinates', 'sensitivity': 'transformed 14D excess cosine must have same positive direction in both held-out splits', 'n_perm': n_perm, 'n_boot': n_boot, 'seed': SEED, 'prespecified_primary_analysis_modified': False, 'claim_boundary': 'A PASS supports within-trial continuity of continuous signal-space geometry; it does not establish family identity, semantic identity, causal mechanism, biological independence, or source localization.'}
    (outroot / 'qc' / 'within_trial_geometry_continuity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = ['# Within-trial experience→transition geometry continuity', '', f"Status: **{summary['status']}**", '', 'Primary effect: same-trial 14D cosine minus the expected mismatched cosine from other transition trials within the same participant and context.', '', 'This post-hoc branch does not modify the prespecified primary analysis. A positive result would support continuity of continuous geometry, not identity of discrete families or biological mechanisms.']
    (outroot / 'reports' / 'within_trial_geometry_continuity_report.md').write_text(''.join(report) + '', encoding='utf-8')
    lock = write_lock(outroot)
    lock_sha = sha256_file(lock)
    print('within_trial_geometry_continuity PASS' if pass_primary else 'within_trial_geometry_continuity COMPLETE — primary gate NOT SUPPORTED')
    print(f'Final lock SHA-256: {lock_sha}')

def self_test() -> None:
    rng = np.random.default_rng(7)
    nsub = 6
    rows = []
    for s in range(nsub):
        for c in ('video', 'imagery'):
            for j in range(8):
                base = rng.normal(size=14)
                t = base + 0.2 * rng.normal(size=14)
                row = {'subject_id': f'sub-{s + 1:02d}', 'context': c, 'E4': 1 + j % 4, 'T2': 1 + j % 2, 'raw_behavior_row_id': f'sub-{s + 1:02d}|trial{j + 1}'}
                for k in range(14):
                    row[f'E::x{k}'] = base[k]
                    row[f'T::x{k}'] = t[k]
                rows.append(row)
    df = pd.DataFrame(rows)
    cols = [f'x{k}' for k in range(14)]
    E = normalize_rows(df[[f'E::{c}' for c in cols]].to_numpy(float))
    T = normalize_rows(df[[f'T::{c}' for c in cols]].to_numpy(float))
    trial, _, mis, _, _, _ = continuity_components(df, E, T)
    pm = participant_metrics(df, trial, 'S', 'standardized')
    assert pm.mean_excess_cosine.mean() > 0.5
    null = permutation_null(df, E, T, mis, 199, rng)
    assert permutation_p(float(pm.mean_excess_cosine.mean()), null) < 0.05
    assert holm_adjust({'a': 0.01, 'b': 0.04})['a'] <= 0.02 + 1e-12
    print('within_trial_geometry_continuity self-test PASS')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('<PROJECT_ROOT>'))
    ap.add_argument('--canonical-master', type=Path, default=CANONICAL_DEFAULT)
    ap.add_argument('--n-perm', type=int, default=DEFAULT_N_PERM)
    ap.add_argument('--n-boot', type=int, default=DEFAULT_N_BOOT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run(args.project, args.canonical_master, args.n_perm, args.n_boot, Path(__file__).resolve())
if __name__ == '__main__':
    main()