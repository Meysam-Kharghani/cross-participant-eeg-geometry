# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase discovery_robustness_audit: execute the prespecified fixed-output Discovery robustness audit.

This phase verifies the discovery_synthesis specification and every fixed mapping_inference object, reproduces
all prespecified p-values/Holm decisions, executes only the nine pre-registered
fixed-output robustness checks, and evaluates the structural/integrity gate for
a later validation_geometry_transfer_design analysis specification. It never refits cross-person alignment, changes
family assignments, retunes mapping models, renames families, or reads
Validation/Confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'discovery_robustness_audit'
SOURCE_PHASE = 'discovery_synthesis'
NEXT_PHASE = 'validation_geometry_transfer_design'
SEED = 540
PERM_LOPO = 10000
ALPHA_LEVEL = 0.05
EXPECTED_ROBUSTNESS_POLICY_SHA = '424061bad979c7bcbf549d655cd27a479ea933572da975ae8c4aded68a4e9a91'
EXPECTED_VALIDATION_GATE_POLICY_SHA = 'edf280b8a49e94c9eeda443d8d7c4d3a2aa64ae2ce23244b842e8f2e0042bd4d'
EXPECTED_INTERPRETATION_POLICY_SHA = '5f91f9a291e02eeb3b820a12866a2d7c46690159531a387bddbd471b0cf0b209'
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

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

def metric_delta_r2(Y: np.ndarray, Pb: np.ndarray, Pf: np.ndarray, subjects: np.ndarray, sample_weights: np.ndarray | None=None) -> tuple[float, np.ndarray]:
    Y = np.asarray(Y, float)
    Pb = np.asarray(Pb, float)
    Pf = np.asarray(Pf, float)
    w = participant_weights(subjects) if sample_weights is None else np.asarray(sample_weights, float)
    mu = (w[:, None] * Y).sum(0) / w.sum()
    sst = (w[:, None] * (Y - mu) ** 2).sum(0)
    sseb = (w[:, None] * (Y - Pb) ** 2).sum(0)
    ssef = (w[:, None] * (Y - Pf) ** 2).sum(0)
    dim = (sseb - ssef) / np.maximum(sst, 1e-15)
    glob = float((sseb.sum() - ssef.sum()) / max(sst.sum(), 1e-15))
    return (glob, dim)

def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(running, 1.0)
    return out

def perm_p(obs: float, null: np.ndarray, two_sided: bool=False) -> float:
    a = np.asarray(null, float)
    if two_sided:
        return float((1 + np.sum(np.abs(a) >= abs(obs))) / (len(a) + 1))
    return float((1 + np.sum(a >= obs)) / (len(a) + 1))

def h1_stat(D: np.ndarray) -> float:
    D = np.asarray(D, float)
    n = len(D)
    m = D.mean(0)
    cov = LedoitWolf(assume_centered=False).fit(D).covariance_
    return float(n * (m @ np.linalg.pinv(cov, rcond=1e-12) @ m))

def weighted_lstsq(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.asarray(w, float))[:, None]
    return np.linalg.lstsq(X * sw, Y * sw, rcond=None)[0]

def nuisance_design(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    cols = [np.ones(n)]
    ctx = df['context'].astype(str).to_numpy()
    cats = sorted(np.unique(ctx).tolist())
    for c in cats[1:]:
        cols.append((ctx == c).astype(float))
    a = df['acquisition_fraction'].to_numpy(float)
    cols.extend([a, a * a])
    subs = df['subject_id'].astype(str).to_numpy()
    su = sorted(np.unique(subs).tolist())
    for s in su[1:]:
        cols.append((subs == s).astype(float))
    return np.column_stack(cols)

def h1_contrast_matrix(trial: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X = nuisance_design(trial)
    Y = trial[REPORTS].to_numpy(float)
    w = participant_weights(trial.subject_id.to_numpy())
    adj = Y - X @ weighted_lstsq(X, Y, w)
    subjects = sorted(trial.subject_id.astype(str).unique().tolist())
    rows = []
    ss = trial.subject_id.astype(str).to_numpy()
    ff = trial.trajectory_family_1based.to_numpy(int)
    for s in subjects:
        idx = np.where(ss == s)[0]
        fam = ff[idx]
        if not (np.any(fam == 1) and np.any(fam == 2)):
            raise RuntimeError(f'H1 requires both families for {s}')
        rows.append(adj[idx][fam == 2].mean(0) - adj[idx][fam == 1].mean(0))
    return (np.asarray(rows, float), subjects)

def categorical_stats(df: pd.DataFrame, target: str) -> tuple[float, float]:
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
    return (nmi, v)

def categorical_arrays(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    d = df[['subject_id', 'trajectory_family_1based', target]].dropna().copy()
    subs = d.subject_id.astype(str).to_numpy()
    fam = d.trajectory_family_1based.to_numpy(int) - 1
    cats = sorted(d[target].astype(str).unique().tolist())
    cmap = {c: i for i, c in enumerate(cats)}
    tar = np.array([cmap[x] for x in d[target].astype(str)], int)
    w = participant_weights(subs)
    _, inv = np.unique(subs, return_inverse=True)
    return (fam, tar, w, inv, len(cats))

def categorical_from_arrays(fam: np.ndarray, tar: np.ndarray, w: np.ndarray, ncat: int) -> tuple[float, float]:
    counts = np.bincount(fam * ncat + tar, weights=w, minlength=2 * ncat).reshape(2, ncat)
    total = counts.sum()
    p = counts / total
    px = p.sum(1)
    py = p.sum(0)
    nz = p > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(p[nz] * np.log(p[nz] / denom[nz])))
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    nmi = float(mi / math.sqrt(hx * hy)) if hx > 0 and hy > 0 else 0.0
    expected = np.outer(counts.sum(1), counts.sum(0)) / total
    chi = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-15)))
    v = float(math.sqrt(chi / (total * min(1, ncat - 1)))) if ncat > 1 else 0.0
    return (nmi, v)

def input_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'g_contract': c / 'discovery_synthesis_discovery_synthesis_contract.json', 'g_next': project / 'analysis_specifications/S028_discovery_synthesis_robustness_audit.json', 'g_robust': project / 'analysis_specifications/S029_discovery_synthesis_robustness_audit.json', 'g_validation': project / 'analysis_specifications/S030_discovery_synthesis_validation_gate.json', 'g_interpretation': project / 'analysis_specifications/S027_discovery_synthesis_fixed_interpretation.json', 'g_lock': q / 'discovery_synthesis_final_output_lock_hashes.json', 'g_mapping_inference_rehash': q / 'discovery_synthesis_mapping_inference_final_rehash.csv', 'g_upstream_rehash': q / 'discovery_synthesis_upstream_evidence_rehash.csv', 'g_evidence': t / 'discovery_synthesis_cross_phase_evidence_matrix.csv', 'g_claims': t / 'discovery_synthesis_fixed_interpretation_claims.csv', 'g_robust_manifest': t / 'discovery_synthesis_prespecified_robustness_check_manifest.csv', 'g_validation_manifest': t / 'discovery_synthesis_prespecified_validation_gate_manifest.csv', 'g_summary': q / 'discovery_synthesis_discovery_synthesis_summary.json', 'g_report': r / 'discovery_synthesis_discovery_synthesis_report.md', 'g_log': l / 'discovery_synthesis_discovery_synthesis.log', 'f_lock': q / 'mapping_inference_final_output_lock_hashes.json', 'f_contract': c / 'mapping_inference_mapping_inference_contract.json', 'f_next': project / 'analysis_specifications/S042_mapping_inference_discovery_synthesis.json', 'f_implementation': c / 'mapping_inference_fixed_implementation_resolution.json', 'f_implementation_lock': q / 'mapping_inference_implementation_lock.json', 'f_input_rehash': q / 'mapping_inference_input_rehash.csv', 'f_summary': q / 'mapping_inference_mapping_inference_summary.json', 'f_global': t / 'mapping_inference_global_hypothesis_tests.csv', 'f_followups': t / 'mapping_inference_report_dimension_followups.csv', 'f_h1_contrasts': t / 'mapping_inference_h1_participant_family_contrasts.csv', 'f_h1_descriptives': t / 'mapping_inference_h1_raw_family_descriptives.csv', 'f_h1_null': t / 'mapping_inference_h1_null_replicates.csv', 'f_h1_boot': t / 'mapping_inference_h1_bootstrap_summary.csv', 'f_h2_predictions': t / 'mapping_inference_h2_oof_predictions.csv', 'f_h2_selection': t / 'mapping_inference_h2_outer_alpha_selection.csv', 'f_h2_null': t / 'mapping_inference_h2_null_replicates.csv', 'f_h2_boot': t / 'mapping_inference_h2_bootstrap_summary.csv', 'f_h3_predictions': t / 'mapping_inference_h3_oof_state_profiles_predictions.csv', 'f_h3_null': t / 'mapping_inference_h3_null_replicates.csv', 'f_h3_boot': t / 'mapping_inference_h3_bootstrap_summary.csv', 'f_categorical': t / 'mapping_inference_h4_h6_categorical_tests.csv', 'f_crosstabs': t / 'mapping_inference_h4_h6_descriptive_crosstabs.csv', 'f_cat_null': t / 'mapping_inference_h4_h6_null_replicates.csv', 'f_report': r / 'mapping_inference_mapping_inference_report.md', 'f_log': l / 'mapping_inference_mapping_inference.log', 'd_lock': q / 'cross_person_alignment_fit_final_output_lock_hashes.json', 'd_assign': t / 'cross_person_alignment_fit_trajectory_family_assignments.csv', 'd_lopo': t / 'cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv', 'e_lock': q / 'mapping_design_final_output_lock_hashes.json', 'e_trial': t / 'mapping_design_trial_mapping_manifest.csv'}

def output_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'log': l / 'discovery_robustness_audit_discovery_robustness_audit.log', 'source_rehash': q / 'discovery_robustness_audit_source_rehash.csv', 'r1': t / 'discovery_robustness_audit_r1_integrity_reproduction.csv', 'r2': t / 'discovery_robustness_audit_r2_h1_fixed_contrast_jackknife.csv', 'r3': t / 'discovery_robustness_audit_r3_h2_fixed_oof_jackknife.csv', 'r4': t / 'discovery_robustness_audit_r4_h3_fixed_oof_jackknife.csv', 'r5': t / 'discovery_robustness_audit_r5_categorical_fixed_assignment_jackknife.csv', 'r6_tests': t / 'discovery_robustness_audit_r6_lopo_subset_sensitivity_tests.csv', 'r6_null': t / 'discovery_robustness_audit_r6_lopo_subset_null_replicates.csv', 'r7': t / 'discovery_robustness_audit_r7_null_calibration_audit.csv', 'r8_subjects': t / 'discovery_robustness_audit_r8_trajectory_structural_gate_subjects.csv', 'r8_summary': q / 'discovery_robustness_audit_r8_trajectory_structural_gate_summary.json', 'r9': t / 'discovery_robustness_audit_r9_label_swap_invariance.csv', 'check_summary': t / 'discovery_robustness_audit_robustness_check_summary.csv', 'summary': q / 'discovery_robustness_audit_discovery_robustness_summary.json', 'report': r / 'discovery_robustness_audit_discovery_robustness_report.md', 'contract': c / 'discovery_robustness_audit_discovery_robustness_contract.json', 'next_policy': project / 'analysis_specifications/S024_discovery_robustness_audit_validation_geometry_transfer.json', 'final_lock': q / 'discovery_robustness_audit_final_output_lock_hashes.json'}
G_LOCK_MAP = {'g_mapping_inference_rehash': 'mapping_inference_rehash', 'g_upstream_rehash': 'upstream_rehash', 'g_evidence': 'evidence_matrix', 'g_claims': 'claims', 'g_robust_manifest': 'robustness_manifest', 'g_validation_manifest': 'validation_manifest', 'g_interpretation': 'interpretation_policy', 'g_robust': 'robustness_policy', 'g_validation': 'validation_policy', 'g_summary': 'summary', 'g_report': 'report', 'g_contract': 'contract', 'g_next': 'next_policy', 'g_log': 'log'}
F_LOCK_MAP = {'f_input_rehash': 'input_rehash', 'f_implementation': 'implementation', 'f_implementation_lock': 'implementation_lock', 'f_h1_contrasts': 'h1_contrasts', 'f_h1_descriptives': 'h1_descriptives', 'f_h1_null': 'h1_null', 'f_h1_boot': 'h1_boot', 'f_h2_predictions': 'h2_predictions', 'f_h2_selection': 'h2_selection', 'f_h2_null': 'h2_null', 'f_h2_boot': 'h2_boot', 'f_h3_predictions': 'h3_profiles', 'f_h3_null': 'h3_null', 'f_h3_boot': 'h3_boot', 'f_categorical': 'categorical_tests', 'f_crosstabs': 'categorical_crosstabs', 'f_cat_null': 'categorical_null', 'f_global': 'global_tests', 'f_followups': 'report_followups', 'f_summary': 'summary', 'f_report': 'report', 'f_contract': 'contract', 'f_next': 'next_policy', 'f_log': 'log'}

def verify_sources(inputs: dict[str, Path]) -> pd.DataFrame:
    require(inputs.values())
    glock = json.loads(inputs['g_lock'].read_text())
    gnext = json.loads(inputs['g_next'].read_text())
    robust = json.loads(inputs['g_robust'].read_text())
    validation = json.loads(inputs['g_validation'].read_text())
    interp = json.loads(inputs['g_interpretation'].read_text())
    if glock.get('phase') != SOURCE_PHASE or gnext.get('next_phase') != PHASE or (not bval(gnext.get('next_phase_permitted'))):
        raise RuntimeError('discovery_synthesis authorization mismatch')
    if canonical_sha(robust) != EXPECTED_ROBUSTNESS_POLICY_SHA or canonical_sha(validation) != EXPECTED_VALIDATION_GATE_POLICY_SHA or canonical_sha(interp) != EXPECTED_INTERPRETATION_POLICY_SHA:
        raise RuntimeError('discovery_synthesis canonical policy SHA mismatch')
    if bval(glock.get('validation_opened')) or bval(glock.get('confirmation_opened')):
        raise RuntimeError('Later-cohort boundary violation in discovery_synthesis')
    rows = []
    for key, logical in G_LOCK_MAP.items():
        exp = glock['hashes'].get(logical, '')
        obs = sha256(inputs[key])
        ok = exp == obs
        rows.append({'source_phase': 'discovery_synthesis', 'object': logical, 'path': str(inputs[key]), 'expected_sha256': exp, 'observed_sha256': obs, 'match': ok})
        if not ok:
            raise RuntimeError(f'discovery_synthesis hash mismatch: {logical}')
    flock = json.loads(inputs['f_lock'].read_text())
    for key, logical in F_LOCK_MAP.items():
        exp = flock['hashes'].get(logical, '')
        obs = sha256(inputs[key])
        ok = exp == obs
        rows.append({'source_phase': 'mapping_inference', 'object': logical, 'path': str(inputs[key]), 'expected_sha256': exp, 'observed_sha256': obs, 'match': ok})
        if not ok:
            raise RuntimeError(f'mapping_inference hash mismatch: {logical}')
    dlock = json.loads(inputs['d_lock'].read_text())
    elock = json.loads(inputs['e_lock'].read_text())
    for key, logical, lock in [('d_assign', 'trajectory_selected_assignments', dlock), ('d_lopo', 'trajectory_lopo', dlock), ('e_trial', 'trial_manifest', elock)]:
        exp = lock['hashes'].get(logical, '')
        obs = sha256(inputs[key])
        ok = exp == obs
        rows.append({'source_phase': 'cross_person_alignment_fit' if key.startswith('d') else 'mapping_design', 'object': logical, 'path': str(inputs[key]), 'expected_sha256': exp, 'observed_sha256': obs, 'match': ok})
        if not ok:
            raise RuntimeError(f'Upstream hash mismatch: {logical}')
    if sha256(inputs['f_lock']) != robust['source_hashes']['mapping_inference_final_lock'] or sha256(inputs['d_lock']) != robust['source_hashes']['cross_person_alignment_fit_final_lock'] or sha256(inputs['e_lock']) != robust['source_hashes']['mapping_design_final_lock']:
        raise RuntimeError('discovery_synthesis robustness source-lock SHA mismatch')
    if sha256(inputs['e_trial']) != robust['source_hashes']['trial_manifest'] or sha256(inputs['d_lopo']) != robust['source_hashes']['trajectory_lopo']:
        raise RuntimeError('discovery_synthesis robustness source-object SHA mismatch')
    return pd.DataFrame(rows)

def reproduce_r1(inputs: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    global_df = pd.read_csv(inputs['f_global'])
    follow = pd.read_csv(inputs['f_followups'])
    h1c = pd.read_csv(inputs['f_h1_contrasts'])
    h1n = pd.read_csv(inputs['f_h1_null'])
    h2p = pd.read_csv(inputs['f_h2_predictions'])
    h2n = pd.read_csv(inputs['f_h2_null'])
    h3p = pd.read_csv(inputs['f_h3_predictions'])
    h3n = pd.read_csv(inputs['f_h3_null'])
    cat = pd.read_csv(inputs['f_categorical'])
    catn = pd.read_csv(inputs['f_cat_null'])
    trial = pd.read_csv(inputs['e_trial'])
    rows = []
    ok_all = True
    expected_counts = {'global_rows': 6, 'followup_rows': 30, 'h1_participants': 31, 'h1_null': 10000, 'h2_predictions': 1237, 'h2_null': 1000, 'h3_states': 40, 'h3_null': 10000, 'categorical_null': 30000}
    observed_counts = {'global_rows': len(global_df), 'followup_rows': len(follow), 'h1_participants': len(h1c), 'h1_null': len(h1n), 'h2_predictions': len(h2p), 'h2_null': len(h2n), 'h3_states': len(h3p), 'h3_null': len(h3n), 'categorical_null': len(catn)}
    for name, exp in expected_counts.items():
        obs = observed_counts[name]
        good = obs == exp
        ok_all &= good
        rows.append({'component': 'row_count', 'item': name, 'observed': obs, 'reported_or_expected': exp, 'tolerance': 0, 'pass': good})
    D = h1c[[f'adjusted_contrast_{r}' for r in REPORTS]].to_numpy(float)
    h1obs = h1_stat(D)
    Y = h2p[[f'observed_{r}' for r in REPORTS]].to_numpy(float)
    Pb = h2p[[f'baseline_predicted_{r}' for r in REPORTS]].to_numpy(float)
    Pf = h2p[[f'full_predicted_{r}' for r in REPORTS]].to_numpy(float)
    h2obs, h2dim = metric_delta_r2(Y, Pb, Pf, h2p.subject_id.to_numpy(str))
    Y3 = h3p[REPORTS].to_numpy(float)
    P3 = h3p[[f'predicted_{r}' for r in REPORTS]].to_numpy(float)
    w3 = h3p.state_alignment_weight.to_numpy(float)
    h3obs, h3dim = metric_delta_r2(Y3, np.zeros_like(Y3), P3, h3p.subject_id.to_numpy(str), w3)
    observed = {'H1': h1obs, 'H2': h2obs, 'H3': h3obs}
    nulls = {'H1': h1n.global_T.to_numpy(float), 'H2': h2n.global_delta_r2.to_numpy(float), 'H3': h3n.global_delta_r2.to_numpy(float)}
    for h, target in [('H4', 'context'), ('H5', 'emotion_category'), ('H6', 'stimulus_id')]:
        observed[h] = categorical_stats(trial, target)[0]
        nulls[h] = catn.loc[catn.hypothesis == h, 'null_nmi'].to_numpy(float)
    pvals = np.array([perm_p(observed[h], nulls[h]) for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']])
    holm = holm_adjust(pvals)
    g = global_df.set_index('hypothesis')
    for i, h in enumerate(['H1', 'H2', 'H3', 'H4', 'H5', 'H6']):
        B = len(nulls[h])
        unit = 1 / (B + 1)
        vals = [('observed_statistic', observed[h], float(g.loc[h, 'observed_statistic']), 1e-10), ('raw_p', pvals[i], float(g.loc[h, 'raw_p']), unit + 1e-12), ('holm_p', holm[i], float(g.loc[h, 'holm_p']), unit * 6 + 1e-12)]
        for item, obs, rep, tol in vals:
            good = abs(obs - rep) <= tol
            ok_all &= good
            rows.append({'component': h, 'item': item, 'observed': obs, 'reported_or_expected': rep, 'tolerance': tol, 'pass': good})
        support = bool(holm[i] < ALPHA_LEVEL)
        good = support == bval(g.loc[h, 'global_supported'])
        ok_all &= good
        rows.append({'component': h, 'item': 'global_supported', 'observed': support, 'reported_or_expected': bval(g.loc[h, 'global_supported']), 'tolerance': 0, 'pass': good})
    child_obs = {'H1': D.mean(0), 'H2': h2dim, 'H3': h3dim}
    child_null = {'H1': h1n[[f'mean_contrast_{r}' for r in REPORTS]].to_numpy(float), 'H2': h2n[[f'delta_r2_{r}' for r in REPORTS]].to_numpy(float), 'H3': h3n[[f'delta_r2_{r}' for r in REPORTS]].to_numpy(float)}
    for parent in ['H1', 'H2', 'H3']:
        cp = np.array([perm_p(child_obs[parent][j], child_null[parent][:, j], two_sided=parent == 'H1') for j in range(10)])
        ca = holm_adjust(cp)
        f = follow[follow.parent_hypothesis == parent].set_index('report')
        parent_hp = float(g.loc[parent, 'holm_p'])
        for j, r in enumerate(REPORTS):
            for item, obs, rep, tol in [('child_raw_p', cp[j], float(f.loc[r, 'raw_p']), 1 / (len(child_null[parent]) + 1) + 1e-12), ('child_holm_p', ca[j], float(f.loc[r, 'within_parent_holm_p']), 10 / (len(child_null[parent]) + 1) + 1e-12)]:
                good = abs(obs - rep) <= tol
                ok_all &= good
                rows.append({'component': f'{parent}:{r}', 'item': item, 'observed': obs, 'reported_or_expected': rep, 'tolerance': tol, 'pass': good})
            gated = bool(parent_hp < ALPHA_LEVEL and ca[j] < ALPHA_LEVEL)
            good = gated == bval(f.loc[r, 'gated_supported'])
            ok_all &= good
            rows.append({'component': f'{parent}:{r}', 'item': 'gated_supported', 'observed': gated, 'reported_or_expected': bval(f.loc[r, 'gated_supported']), 'tolerance': 0, 'pass': good})
    rows.append({'component': 'R1', 'item': 'overall_pass', 'observed': ok_all, 'reported_or_expected': True, 'tolerance': 0, 'pass': ok_all})
    return (pd.DataFrame(rows), observed, nulls)

def jackknives(inputs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h1 = pd.read_csv(inputs['f_h1_contrasts'])
    D = h1[[f'adjusted_contrast_{r}' for r in REPORTS]].to_numpy(float)
    subs = h1.subject_id.astype(str).tolist()
    r2 = []
    for i, s in enumerate(subs):
        x = np.delete(D, i, axis=0)
        row = {'held_out_subject_id': s, 'n_participants': len(x), 'shrinkage_Hotelling_T': h1_stat(x)}
        row.update({f'mean_contrast_{r}': float(v) for r, v in zip(REPORTS, x.mean(0))})
        r2.append(row)
    h2 = pd.read_csv(inputs['f_h2_predictions'])
    r3 = []
    for s in sorted(h2.subject_id.astype(str).unique()):
        d = h2[h2.subject_id.astype(str) != s]
        Y = d[[f'observed_{r}' for r in REPORTS]].to_numpy(float)
        Pb = d[[f'baseline_predicted_{r}' for r in REPORTS]].to_numpy(float)
        Pf = d[[f'full_predicted_{r}' for r in REPORTS]].to_numpy(float)
        g, dd = metric_delta_r2(Y, Pb, Pf, d.subject_id.to_numpy(str))
        row = {'held_out_subject_id': s, 'n_rows': len(d), 'global_delta_r2': g}
        row.update({f'delta_r2_{r}': float(v) for r, v in zip(REPORTS, dd)})
        r3.append(row)
    h3 = pd.read_csv(inputs['f_h3_predictions'])
    r4 = []
    for s in sorted(h3.subject_id.astype(str).unique()):
        d = h3[h3.subject_id.astype(str) != s]
        Y = d[REPORTS].to_numpy(float)
        P = d[[f'predicted_{r}' for r in REPORTS]].to_numpy(float)
        g, dd = metric_delta_r2(Y, np.zeros_like(Y), P, d.subject_id.to_numpy(str), d.state_alignment_weight.to_numpy(float))
        row = {'held_out_subject_id': s, 'n_states': len(d), 'global_delta_r2': g}
        row.update({f'delta_r2_{r}': float(v) for r, v in zip(REPORTS, dd)})
        r4.append(row)
    trial = pd.read_csv(inputs['e_trial'])
    r5 = []
    for h, target in [('H4', 'context'), ('H5', 'emotion_category'), ('H6', 'stimulus_id')]:
        for s in sorted(trial.subject_id.astype(str).unique()):
            d = trial[trial.subject_id.astype(str) != s]
            nmi, v = categorical_stats(d, target)
            r5.append({'hypothesis': h, 'target': target, 'held_out_subject_id': s, 'n_rows': int(d[target].notna().sum()), 'n_participants': int(d.loc[d[target].notna(), 'subject_id'].nunique()), 'participant_balanced_nmi': nmi, 'cramers_v': v})
    return (pd.DataFrame(r2), pd.DataFrame(r3), pd.DataFrame(r4), pd.DataFrame(r5))

def checkpoint_array(cache: Path, name: str, key: dict[str, Any], repeats: int, width: int, fn, chunk: int=250) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / f'{name}.npz'
    meta = cache / f'{name}.json'
    kh = canonical_sha(key)
    arr = np.full((repeats, width), np.nan, float)
    done = 0
    if npz.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if m.get('key_sha256') == kh and int(m.get('repeats', -1)) == repeats and (int(m.get('width', -1)) == width):
            old = np.load(npz)['arr']
            if old.shape == arr.shape:
                arr = old
                done = int(np.sum(np.all(np.isfinite(arr), axis=1)))
    for i in range(done, repeats):
        arr[i] = fn(i)
        if (i + 1) % chunk == 0 or i + 1 == repeats:
            tmp = npz.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, arr=arr)
            os.replace(tmp, npz)
            atomic_json(meta, {'key_sha256': kh, 'repeats': repeats, 'width': width, 'completed': i + 1})
    return arr

def run_r6(inputs: dict[str, Path], perms: int, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trial = pd.read_csv(inputs['e_trial'])
    lopo = pd.read_csv(inputs['d_lopo'])
    m = trial.merge(lopo[['held_out_subject_id', 'raw_behavior_row_id', 'assignment_agrees_with_full_fit']], left_on=['subject_id', 'raw_behavior_row_id'], right_on=['held_out_subject_id', 'raw_behavior_row_id'], how='inner', validate='one_to_one')
    if len(m) != len(trial):
        raise RuntimeError('LOPO/trial merge did not preserve all rows')
    sub = m[m.assignment_agrees_with_full_fit.map(bval)].copy()
    D, subjects = h1_contrast_matrix(sub)
    h1obs = h1_stat(D)

    def h1fn(i: int) -> np.ndarray:
        rng = np.random.default_rng(SEED + 700000 + i)
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(D))
        Ds = D * signs[:, None]
        return np.r_[h1_stat(Ds), Ds.mean(0)]
    keybase = {'phase': PHASE, 'source_trial_sha': sha256(inputs['e_trial']), 'lopo_sha': sha256(inputs['d_lopo']), 'permutations': perms, 'seed': SEED}
    h1null = checkpoint_array(cache, 'r6_h1', {**keybase, 'test': 'H1'}, perms, 11, h1fn, 100)
    tests = [{'hypothesis': 'H1', 'target': 'adjusted_10D_report_profile', 'n_rows': len(sub), 'n_participants': len(subjects), 'observed_statistic': h1obs, 'statistic_name': 'shrinkage_Hotelling_T', 'raw_p': perm_p(h1obs, h1null[:, 0]), 'sensitivity_only': True}]
    nullframes = [pd.DataFrame(h1null, columns=['null_statistic', *[f'null_mean_contrast_{r}' for r in REPORTS]]).assign(hypothesis='H1', target='adjusted_10D_report_profile', replicate=np.arange(1, perms + 1))]
    offsets = {'H4': 800000, 'H5': 900000, 'H6': 1000000}
    for h, target in [('H4', 'context'), ('H5', 'emotion_category'), ('H6', 'stimulus_id')]:
        fam, tar, w, inv, ncat = categorical_arrays(sub, target)
        obs, v = categorical_from_arrays(fam, tar, w, ncat)
        groups = [np.where(inv == g)[0] for g in np.unique(inv)]

        def fn(i: int, h=h, fam=fam, tar=tar, w=w, ncat=ncat, groups=groups):
            rng = np.random.default_rng(SEED + offsets[h] + i)
            fp = fam.copy()
            for idx in groups:
                fp[idx] = fam[idx[rng.permutation(len(idx))]]
            return np.array(categorical_from_arrays(fp, tar, w, ncat))
        nul = checkpoint_array(cache, f'r6_{h.lower()}', {**keybase, 'test': h, 'target': target}, perms, 2, fn, 250)
        tests.append({'hypothesis': h, 'target': target, 'n_rows': int(sub[target].notna().sum()), 'n_participants': int(sub.loc[sub[target].notna(), 'subject_id'].nunique()), 'observed_statistic': obs, 'statistic_name': 'participant_balanced_nmi', 'raw_p': perm_p(obs, nul[:, 0]), 'sensitivity_only': True})
        nullframes.append(pd.DataFrame({'hypothesis': h, 'target': target, 'replicate': np.arange(1, perms + 1), 'null_statistic': nul[:, 0], 'null_cramers_v': nul[:, 1]}))
    meta = {'full_rows': len(trial), 'agreeing_rows': len(sub), 'excluded_rows': len(trial) - len(sub), 'participants': sub.subject_id.nunique(), 'all_participants_have_both_families': bool((sub.groupby('subject_id').trajectory_family_1based.nunique() == 2).all())}
    return (pd.DataFrame(tests), pd.concat(nullframes, ignore_index=True, sort=False), meta)

def r7_audit(inputs: dict[str, Path], observed: dict[str, float], nulls: dict[str, np.ndarray]) -> pd.DataFrame:
    g = pd.read_csv(inputs['f_global']).set_index('hypothesis')
    rows = []
    for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
        a = np.asarray(nulls[h], float)
        p = perm_p(observed[h], a)
        reported = float(g.loc[h, 'raw_p'])
        B = len(a)
        resolution = 1 / (B + 1)
        finite = bool(np.isfinite(a).all())
        sd = float(np.std(a, ddof=1))
        mcse = float(math.sqrt(max(p * (1 - p), 0) / B))
        percentile = float(np.mean(a <= observed[h]))
        reproduced = abs(p - reported) <= resolution + 1e-12
        rows.append({'hypothesis': h, 'n_null': B, 'all_finite': finite, 'null_mean': float(np.mean(a)), 'null_sd': sd, 'observed_statistic': observed[h], 'observed_percentile_leq': percentile, 'plus_one_p_recomputed': p, 'reported_raw_p': reported, 'absolute_p_difference': abs(p - reported), 'monte_carlo_se': mcse, 'minimum_p_resolution': resolution, 'pass': bool(finite and sd > 0 and reproduced)})
    return pd.DataFrame(rows)

def r8_audit(inputs: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    l = pd.read_csv(inputs['d_lopo'])
    a = pd.read_csv(inputs['d_assign'])
    agree = l.assignment_agrees_with_full_fit.map(bval)
    rows = []
    for s, g in l.assign(_agree=agree).groupby('held_out_subject_id', sort=True):
        ag = float(g._agree.mean())
        af = a[a.subject_id.astype(str) == str(s)]
        rows.append({'subject_id': s, 'n_trajectories': len(g), 'lopo_agreement': ag, 'at_or_above_0_90': ag >= 0.9, 'at_or_above_0_95': ag >= 0.95, 'n_full_fit_families': int(af.trajectory_family_1based.nunique()), 'both_full_fit_families': int(af.trajectory_family_1based.nunique()) == 2})
    df = pd.DataFrame(rows)
    overall = float(agree.mean())
    minimum = float(df.lopo_agreement.min())
    n95 = int(df.at_or_above_0_95.sum())
    both = int(df.both_full_fit_families.sum())
    converged = bool(l.training_fit_converged.map(bval).all())
    passes = {'overall_lopo_agreement': overall >= 0.99, 'minimum_subject_lopo_agreement': minimum >= 0.9, 'subjects_at_or_above_0_95': n95 >= 30, 'participants_with_both_full_fit_families': both >= 31, 'all_lopo_training_fits_converged': converged}
    summary = {'overall_lopo_agreement': overall, 'minimum_subject_lopo_agreement': minimum, 'subjects_at_or_above_0_95': n95, 'participants_with_both_full_fit_families': both, 'participants': len(df), 'all_lopo_training_fits_converged': converged, 'thresholds': {'overall_min': 0.99, 'minimum_subject_min': 0.9, 'subjects_at_or_above_0_95_min': 30, 'both_families_participants_min': 31}, 'component_pass': passes, 'R8_pass': bool(all(passes.values()))}
    return (df, summary)

def r9_audit(inputs: dict[str, Path]) -> pd.DataFrame:
    trial = pd.read_csv(inputs['e_trial'])
    swapped = trial.copy()
    swapped['trajectory_family_1based'] = 3 - swapped.trajectory_family_1based.astype(int)
    D, _ = h1_contrast_matrix(trial)
    Ds, _ = h1_contrast_matrix(swapped)
    rows = []
    vals = [('H1_global_T_invariance', h1_stat(D), h1_stat(Ds), 'equal')]
    for r, x, y in zip(REPORTS, D.mean(0), Ds.mean(0)):
        vals.append((f'H1_{r}_contrast_sign_reversal', x, y, 'negative'))
    for h, target in [('H4', 'context'), ('H5', 'emotion_category'), ('H6', 'stimulus_id')]:
        n, v = categorical_stats(trial, target)
        ns, vs = categorical_stats(swapped, target)
        vals.extend([(f'{h}_NMI_invariance', n, ns, 'equal'), (f'{h}_CramersV_invariance', v, vs, 'equal')])
    for name, x, y, rule in vals:
        err = abs(x - y) if rule == 'equal' else abs(x + y)
        rows.append({'check': name, 'original': x, 'label_swapped': y, 'required_relation': rule, 'absolute_relation_error': err, 'pass': err <= 1e-10})
    return pd.DataFrame(rows)

def write_report(path: Path, summary: dict[str, Any], checks: pd.DataFrame, r6: pd.DataFrame, r8: dict[str, Any]) -> None:
    lines = ['# Phase discovery_robustness_audit — prespecified Discovery robustness audit', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Validation opened: **no**', '- Confirmation opened: **no**', '', '## Robustness checks', '']
    for _, x in checks.iterrows():
        lines.append(f'- **{x.check_id} — {x.check_name}:** {x.status}')
    lines += ['', '## Fixed LOPO-agreeing sensitivity (non-primary)', '']
    for _, x in r6.iterrows():
        lines.append(f'- {x.hypothesis}: statistic={x.observed_statistic:.6g}, p={x.raw_p:.6g}; sensitivity-only.')
    lines += ['', '## Structural gate', '', f"- Overall LOPO agreement: {r8['overall_lopo_agreement']:.6f}", f"- Minimum subject agreement: {r8['minimum_subject_lopo_agreement']:.6f}", f"- Subjects at or above 0.95: {r8['subjects_at_or_above_0_95']}/31", f"- Participants containing both full-fit families: {r8['participants_with_both_full_fit_families']}/31", f"- R8 passed: {r8['R8_pass']}", '', '## Boundary', '', '- Sensitivity findings cannot override the prespecified mapping_inference primary null decisions.', '- Family 1 and Family 2 remain geometric labels.', '- validation_geometry_transfer_design may specification outcome-free Validation geometry transfer only if the integrity/null/structural gates passed.', '- This phase did not read Validation or Confirmation.']
    atomic_text(path, ''.join(lines) + '')

def self_test() -> None:
    assert canonical_sha({'b': 2, 'a': 1}) == canonical_sha({'a': 1, 'b': 2})
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    D = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0], [0.0, -1.0]])
    assert np.isfinite(h1_stat(D))
    d = pd.DataFrame({'subject_id': ['a', 'a', 'b', 'b'], 'trajectory_family_1based': [1, 2, 1, 2], 'x': ['u', 'v', 'u', 'v']})
    n, v = categorical_stats(d, 'x')
    ds = d.copy()
    ds.trajectory_family_1based = 3 - ds.trajectory_family_1based
    ns, vs = categorical_stats(ds, 'x')
    assert abs(n - ns) < 1e-12 and abs(v - vs) < 1e-12
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x'
        atomic_text(p, 'abc')
        assert sha256(p) == hashlib.sha256(b'abc').hexdigest()
    print('discovery_robustness_audit self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--permutations', type=int, default=PERM_LOPO)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    inputs = input_paths(project)
    outputs = output_paths(project)
    if outputs['final_lock'].exists() and (not args.force):
        raise RuntimeError('discovery_robustness_audit final lock already exists; use --force to reproduce')
    outputs['log'].parent.mkdir(parents=True, exist_ok=True)
    atomic_text(outputs['log'], '')

    def note(msg: str):
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        with_open = outputs['log'].open('a', encoding='utf-8')
        with_open.write(line + '')
        with_open.close()
    note('Verifying discovery_synthesis specification, all mapping_inference locks, and fixed cross_person_alignment_fit/mapping_design sources.')
    rehash = verify_sources(inputs)
    atomic_csv(outputs['source_rehash'], rehash)
    note('R1: reproducing all prespecified mapping_inference row counts, statistics, p-values, Holm decisions, and gated follow-ups.')
    r1, observed, nulls = reproduce_r1(inputs)
    atomic_csv(outputs['r1'], r1)
    r1pass = bool(r1['pass'].all())
    note('R2-R5: executing fixed-output leave-one-participant influence analyses without refitting fixed models.')
    r2, r3, r4, r5 = jackknives(inputs)
    atomic_csv(outputs['r2'], r2)
    atomic_csv(outputs['r3'], r3)
    atomic_csv(outputs['r4'], r4)
    atomic_csv(outputs['r5'], r5)
    note(f'R6: executing {args.permutations} prespecified permutations for H1 and H4-H6 on the LOPO-agreeing subset.')
    cache = project / 'outputs/cache/discovery_robustness_audit'
    r6, r6null, r6meta = run_r6(inputs, args.permutations, cache)
    atomic_csv(outputs['r6_tests'], r6)
    atomic_csv(outputs['r6_null'], r6null)
    note('R7: auditing null finiteness, variance, empirical p-value reproduction, Monte-Carlo SE, and resolution.')
    r7 = r7_audit(inputs, observed, nulls)
    atomic_csv(outputs['r7'], r7)
    r7pass = bool(r7['pass'].all())
    note('R8: recomputing the prespecified trajectory structural gate from fixed full-fit and LOPO assignments.')
    r8df, r8sum = r8_audit(inputs)
    atomic_csv(outputs['r8_subjects'], r8df)
    atomic_json(outputs['r8_summary'], r8sum)
    note('R9: verifying exact label-swap invariance/sign reversal.')
    r9 = r9_audit(inputs)
    atomic_csv(outputs['r9'], r9)
    r9pass = bool(r9['pass'].all())
    check_rows = [{'check_id': 'R1', 'check_name': 'integrity_and_reproduction', 'status': 'PASS' if r1pass else 'FAIL', 'scientific_role': 'gate'}, {'check_id': 'R2', 'check_name': 'h1_fixed_contrast_jackknife', 'status': 'COMPLETE', 'scientific_role': 'descriptive_influence_only'}, {'check_id': 'R3', 'check_name': 'h2_fixed_oof_jackknife', 'status': 'COMPLETE', 'scientific_role': 'descriptive_influence_only'}, {'check_id': 'R4', 'check_name': 'h3_fixed_oof_jackknife', 'status': 'COMPLETE', 'scientific_role': 'descriptive_influence_only'}, {'check_id': 'R5', 'check_name': 'categorical_fixed_assignment_jackknife', 'status': 'COMPLETE', 'scientific_role': 'descriptive_influence_only'}, {'check_id': 'R6', 'check_name': 'lopo_agreeing_subset_sensitivity', 'status': 'COMPLETE', 'scientific_role': 'sensitivity_only_cannot_override_primary'}, {'check_id': 'R7', 'check_name': 'null_calibration_audit', 'status': 'PASS' if r7pass else 'FAIL', 'scientific_role': 'gate'}, {'check_id': 'R8', 'check_name': 'trajectory_structural_gate_audit', 'status': 'PASS' if r8sum['R8_pass'] else 'FAIL', 'scientific_role': 'gate'}, {'check_id': 'R9', 'check_name': 'label_swap_invariance', 'status': 'PASS' if r9pass else 'FAIL', 'scientific_role': 'integrity'}]
    checks = pd.DataFrame(check_rows)
    atomic_csv(outputs['check_summary'], checks)
    validation_geometry_transfer_design_permitted = bool(r1pass and r7pass and r8sum['R8_pass'] and (not args.test_mode))
    decision = {'R1_pass': r1pass, 'R7_pass': r7pass, 'R8_pass': bool(r8sum['R8_pass']), 'R9_pass': r9pass, 'all_prespecified_checks_executed': True, 'sensitivity_can_override_primary': False, 'family_labels_renamed': False, 'alignment_refit': False, 'family_assignment_changed': False, 'validation_opened': False, 'confirmation_opened': False, 'validation_geometry_transfer_design_specification_permitted': validation_geometry_transfer_design_permitted, 'future_scope': 'outcome_free_trajectory_geometry_transfer_only' if validation_geometry_transfer_design_permitted else 'none_until_gate_failure_resolved'}
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_robustness_complete_candidate' if not args.test_mode else 'test_mode_complete', 'counts': {'source_hashes_verified': len(rehash), 'r2_runs': len(r2), 'r3_runs': len(r3), 'r4_runs': len(r4), 'r5_runs': len(r5), 'r6_permutations_per_test': args.permutations, 'r6_tests': len(r6), 'r7_tests': len(r7), 'r8_subjects': len(r8df), 'r9_relations': len(r9)}, 'r6_subset': r6meta, 'decision': decision}
    atomic_json(outputs['summary'], summary)
    write_report(outputs['report'], summary, checks, r6, r8sum)
    primary = [k for k in outputs if k not in {'contract', 'next_policy', 'final_lock'}]
    hashes = {k: sha256(outputs[k]) for k in primary}
    contract = {'generated_at': now(), 'phase': PHASE, 'source_phase': SOURCE_PHASE, 'status': summary['status'], 'robustness_policy_canonical_sha256': EXPECTED_ROBUSTNESS_POLICY_SHA, 'validation_gate_policy_canonical_sha256': EXPECTED_VALIDATION_GATE_POLICY_SHA, 'interpretation_policy_canonical_sha256': EXPECTED_INTERPRETATION_POLICY_SHA, 'source_discovery_synthesis_final_lock_sha256': sha256(inputs['g_lock']), 'execution_counts': summary['counts'], 'decision': decision, 'output_hashes': hashes, 'source_dataset_modified': False, 'alignment_or_mapping_refit': False, 'family_count_or_assignment_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': validation_geometry_transfer_design_permitted, 'next_phase_role': 'specification every detail of outcome-free transfer of the fixed Discovery trajectory preprocessing and two centroids to Validation before any Validation file is read', 'required_boundary': ['verify every discovery_robustness_audit and source hash', 'do not read any Validation file while creating validation_geometry_transfer_design specification', 'specification preprocessing, centroids, distance rule, coverage metrics, exclusions, failure rules, and output schema', 'do not test H1-H6 in Validation', 'do not transfer a discrete local-state taxonomy', 'do not rename Family 1 or Family 2', 'Confirmation remains closed'], 'permitted_scope': 'outcome_free_trajectory_geometry_transfer_only' if validation_geometry_transfer_design_permitted else 'none', 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['next_policy'], next_policy)
    note('discovery_robustness_audit complete. Sensitivity outputs remain subordinate to prespecified mapping_inference primary decisions; later cohorts remain unopened.')
    hashes['contract'] = sha256(outputs['contract'])
    hashes['next_policy'] = sha256(outputs['next_policy'])
    hashes['log'] = sha256(outputs['log'])
    atomic_json(outputs['final_lock'], {'generated_at': now(), 'phase': PHASE, 'robustness_policy_canonical_sha256': EXPECTED_ROBUSTNESS_POLICY_SHA, 'validation_gate_policy_canonical_sha256': EXPECTED_VALIDATION_GATE_POLICY_SHA, 'validation_opened': False, 'confirmation_opened': False, 'hashes': hashes})
    if not args.test_mode:
        pass
    print(checks.to_string(index=False))
    print('\nR6 sensitivity-only tests:' + r6.to_string(index=False))
    print('\nR8 structural gate:' + json.dumps(r8sum, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())