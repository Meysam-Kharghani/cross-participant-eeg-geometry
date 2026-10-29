# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase cross_person_alignment_fit: execute prespecified outcome-free cross-person alignment.

This phase implements, without reading subjective outcomes, reports, context, QC
values, validation participants, or confirmation participants, the two branches
prespecified in cross_person_alignment_design:

1. participant-balanced constrained alignment of stable local-state centroids;
2. participant-balanced alignment of invariant three-epoch trajectory geometry.

Both branches are SHA-256 fixed before the outcome-free cross-branch linkage is
computed.  No post-hoc report/context mapping occurs in this phase."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'cross_person_alignment_fit'
SOURCE_PHASE = 'cross_person_alignment_design'
NEXT_PHASE = 'mapping_design'
SEED = 540
POLICY_SHA = 'bfc10e1e646406359b989e03b36d50058c51e311c1df23cdb28c97f0712ee7e1'
PER_EPOCH_COMPACT = 169
STATE_ALIGNMENT_PCS = 8
STATE_FAMILY_COUNTS = [3, 4, 5, 6]
TRAJECTORY_FEATURES = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
TRAJECTORY_FAMILY_COUNTS = [2, 3, 4, 5, 6]
EPOCH_ORDER = ['stimulus_early', 'stimulus_late', 'rating_early']
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
N_INIT = 100
MAX_ITER = 500
TOL = 1e-07
STABILITY_REPEATS = 200
PARTICIPANT_FRACTION = 0.8
TRIAL_FRACTION = 0.8
MEDIAN_ARI_MIN = 0.6
P10_ARI_MIN = 0.3
COLLAPSE_RATE_MAX = 0.1
SILHOUETTE_DELTA = 0.05
STATE_MIN_PARTICIPANTS = 3
STATE_MIN_OCCUPANCY = 0.08
TRAJ_MIN_PARTICIPANTS = 5
TRAJ_MIN_TRIALS = 30
TRAJ_MIN_OCCUPANCY = 0.05
LINKAGE_NULL_REPEATS = 1000
REL = {'cross_person_alignment_design_contract': Path('config/cross_person_alignment_design_alignment_design_contract.json'), 'alignment_policy': Path('analysis_specifications/S012_cross_person_alignment.json'), 'execution_policy': Path('analysis_specifications/S011_cross_person_alignment.json'), 'state_manifest': Path('outputs/tables/cross_person_alignment_design_stable_local_state_manifest.csv'), 'trajectory_manifest': Path('outputs/tables/cross_person_alignment_design_continuous_geometry_manifest.csv'), 'feature_schema': Path('outputs/tables/cross_person_alignment_design_alignment_feature_schema.csv'), 'latent_state_fit_lock': Path('outputs/qc/latent_state_fit_primary_lock_hashes.json'), 'centroids_common': Path('outputs/tables/latent_state_fit_state_centroids_common_axes.csv'), 'continuous_trajectories': Path('outputs/tables/latent_state_fit_continuous_trajectories.csv'), 'state_assignments': Path('outputs/tables/latent_state_fit_state_assignments.csv')}
latent_state_fit_PRIMARY_OUTPUT_MAP = {'preprocessing': Path('outputs/tables/latent_state_fit_participant_preprocessing_summary.csv'), 'candidate_metrics': Path('outputs/tables/latent_state_fit_state_candidate_metrics.csv'), 'stability': Path('outputs/tables/latent_state_fit_stability_replicates.csv'), 'selection': Path('outputs/tables/latent_state_fit_participant_state_selection.csv'), 'pca_scores': Path('outputs/tables/latent_state_fit_epoch_pca_scores.csv'), 'pca_loadings': Path('outputs/tables/latent_state_fit_pca_loadings.csv'), 'assignments': Path('outputs/tables/latent_state_fit_state_assignments.csv'), 'continuous_trajectories': Path('outputs/tables/latent_state_fit_continuous_trajectories.csv'), 'discrete_trajectories': Path('outputs/tables/latent_state_fit_discrete_trajectories.csv'), 'occupancy': Path('outputs/tables/latent_state_fit_state_occupancy.csv'), 'transitions': Path('outputs/tables/latent_state_fit_transition_matrices.csv'), 'transition_summary': Path('outputs/tables/latent_state_fit_participant_transition_summaries.csv'), 'null_replicates': Path('outputs/tables/latent_state_fit_trajectory_null_replicates.csv'), 'null_tests': Path('outputs/tables/latent_state_fit_trajectory_null_tests.csv'), 'centroids_pca': Path('outputs/tables/latent_state_fit_state_centroids_pca.csv'), 'centroids_common': Path('outputs/tables/latent_state_fit_state_centroids_common_axes.csv')}
BANNED_EXACT_COLUMNS = {'valence', 'liking', 'arousal', 'emotion', 'emotion_label', 'emotion_category', 'context', 'condition', 'stimulus_category', 'stimulus_type', 'pleasantness', 'unpleasantness', 'wanting', 'pain_intensity', 'pain_unpleasantness', 'subjective_factor_1', 'subjective_factor_2', 'report_factor_1', 'report_factor_2'}

@dataclass
class StatePreprocess:
    feature_names: list[str]
    medians: np.ndarray
    iqrs: np.ndarray
    retained: np.ndarray
    weighted_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    scores: np.ndarray

@dataclass
class TrajectoryPreprocess:
    transformed: np.ndarray
    standardized: np.ndarray
    pooled_iqr: np.ndarray
    participant_medians: dict[str, np.ndarray]
    participant_iqrs: dict[str, np.ndarray]
    participant_scales: dict[str, np.ndarray]
    fallback_used: dict[str, np.ndarray]

@dataclass
class FitResult:
    labels: np.ndarray
    centers: np.ndarray
    objective: float
    iterations: int
    converged: bool
    distances: np.ndarray
    margins: np.ndarray

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--recompute-cache', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def csha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def array_sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            arr = np.ascontiguousarray(part)
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        elif isinstance(part, (list, tuple, pd.Series)):
            h.update('\x1f'.join(map(str, list(part))).encode('utf-8'))
        else:
            h.update(str(part).encode('utf-8'))
        h.update(b'\x1e')
    return h.hexdigest()

def jread(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def bval(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}

def required(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(map(str, missing)))

def atom_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '', encoding='utf-8')
    os.replace(tmp, path)

def atom_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def ensure_no_banned_columns(df: pd.DataFrame, source: str) -> None:
    bad = sorted({str(c).strip().lower() for c in df.columns} & BANNED_EXACT_COLUMNS)
    if bad:
        raise RuntimeError(f'Forbidden outcome/context columns loaded from {source}: {bad}')

def stable_seed(branch: str, k: int=0, repeat: int=0, stream: int=0) -> int:
    token = f'{SEED}|{branch}|{k}|{repeat}|{stream}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little') & 2147483647

def output_paths(project: Path) -> dict[str, Path]:
    t = project / 'outputs/tables'
    q = project / 'outputs/qc'
    c = project / 'config'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'log': l / 'cross_person_alignment_fit_cross_person_alignment_fit.log', 'latent_state_fit_rehash': q / 'cross_person_alignment_fit_latent_state_fit_primary_rehash.csv', 'cross_person_alignment_design_rehash': q / 'cross_person_alignment_fit_cross_person_alignment_design_specification_rehash.csv', 'state_preprocessing': t / 'cross_person_alignment_fit_state_alignment_preprocessing.csv', 'state_pca_loadings': t / 'cross_person_alignment_fit_state_alignment_pca_loadings.csv', 'state_embedding': t / 'cross_person_alignment_fit_state_alignment_embedding.csv', 'state_pairwise': t / 'cross_person_alignment_fit_state_pairwise_distances.csv', 'state_candidates': t / 'cross_person_alignment_fit_state_family_candidate_metrics.csv', 'state_candidate_assignments': t / 'cross_person_alignment_fit_state_family_candidate_assignments.csv', 'state_candidate_prototypes': t / 'cross_person_alignment_fit_state_family_candidate_prototypes.csv', 'state_stability': t / 'cross_person_alignment_fit_state_family_stability_replicates.csv', 'state_selection': t / 'cross_person_alignment_fit_state_family_selection.csv', 'state_selected_assignments': t / 'cross_person_alignment_fit_state_family_assignments.csv', 'state_selected_prototypes': t / 'cross_person_alignment_fit_state_family_prototypes.csv', 'state_lopo': t / 'cross_person_alignment_fit_state_family_lopo_diagnostics.csv', 'trajectory_preprocessing': t / 'cross_person_alignment_fit_trajectory_preprocessing.csv', 'trajectory_standardized': t / 'cross_person_alignment_fit_trajectory_standardized_geometry.csv', 'trajectory_candidates': t / 'cross_person_alignment_fit_trajectory_family_candidate_metrics.csv', 'trajectory_candidate_assignments': t / 'cross_person_alignment_fit_trajectory_family_candidate_assignments.csv', 'trajectory_candidate_centroids': t / 'cross_person_alignment_fit_trajectory_family_candidate_centroids.csv', 'trajectory_stability': t / 'cross_person_alignment_fit_trajectory_family_stability_replicates.csv', 'trajectory_selection': t / 'cross_person_alignment_fit_trajectory_family_selection.csv', 'trajectory_selected_assignments': t / 'cross_person_alignment_fit_trajectory_family_assignments.csv', 'trajectory_selected_centroids': t / 'cross_person_alignment_fit_trajectory_family_centroids.csv', 'trajectory_lopo': t / 'cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv', 'branch_lock': q / 'cross_person_alignment_fit_alignment_branch_primary_lock_hashes.json', 'linkage_rows': t / 'cross_person_alignment_fit_outcome_free_cross_branch_linkage.csv', 'linkage_crosstab': t / 'cross_person_alignment_fit_outcome_free_cross_branch_crosstab.csv', 'linkage_nulls': t / 'cross_person_alignment_fit_outcome_free_cross_branch_null_replicates.csv', 'linkage_tests': t / 'cross_person_alignment_fit_outcome_free_cross_branch_tests.csv', 'final_lock': q / 'cross_person_alignment_fit_final_output_lock_hashes.json', 'summary': q / 'cross_person_alignment_fit_cross_person_alignment_summary.json', 'report': r / 'cross_person_alignment_fit_cross_person_alignment_report.md', 'contract': c / 'cross_person_alignment_fit_cross_person_alignment_contract.json', 'next_policy': project / 'analysis_specifications/S013_cross_person_alignment_fit_mapping.json'}

def verify_inputs(project: Path, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required(paths.values())
    contract = jread(paths['cross_person_alignment_design_contract'])
    policy = jread(paths['alignment_policy'])
    execution = jread(paths['execution_policy'])
    lock = jread(paths['latent_state_fit_lock'])
    if contract.get('phase') != SOURCE_PHASE or contract.get('status') != 'outcome_free_cross_person_alignment_design_prespecified_candidate':
        raise RuntimeError('cross_person_alignment_design contract/status mismatch')
    if execution.get('next_phase') != PHASE or not bval(execution.get('next_phase_permitted')):
        raise RuntimeError('cross_person_alignment_design did not authorize cross_person_alignment_fit')
    if contract.get('alignment_policy_sha256') != POLICY_SHA or execution.get('alignment_policy_sha256') != POLICY_SHA:
        raise RuntimeError('prespecified alignment-policy SHA reference mismatch')
    if csha(policy) != POLICY_SHA:
        raise RuntimeError('prespecified alignment-policy canonical SHA mismatch')
    if bval(contract.get('validation_opened')) or bval(contract.get('confirmation_opened')):
        raise RuntimeError('Validation or confirmation was opened before cross_person_alignment_fit')
    decision = contract.get('decision', {})
    for key in ['latent_state_fit_verified_complete', 'all_latent_state_fit_primary_hashes_verified', 'continuous_only_participants_represented_without_pseudo_states', 'cross_person_alignment_fit_execution_design_prespecified']:
        if not bval(decision.get(key)):
            raise RuntimeError(f'cross_person_alignment_design required decision flag false: {key}')
    for key in ['reports_outcomes_context_columns_read', 'qc_values_read', 'cross_person_alignment_models_fitted', 'shared_family_count_selected', 'validation_opened', 'confirmation_opened']:
        if bval(decision.get(key)):
            raise RuntimeError(f'cross_person_alignment_design forbidden decision flag true: {key}')
    cross_person_alignment_design_files = {'state_manifest': paths['state_manifest'], 'trajectory_manifest': paths['trajectory_manifest'], 'feature_schema': paths['feature_schema'], 'alignment_policy': paths['alignment_policy']}
    expected_c = contract.get('output_hashes', {})
    c_rows = []
    for key, p in cross_person_alignment_design_files.items():
        observed = sha(p)
        expected = expected_c.get(key)
        c_rows.append({'output_key': key, 'relative_path': str(p.relative_to(project)), 'size_bytes': p.stat().st_size, 'expected_sha256': expected, 'observed_sha256': observed, 'sha256_match': observed == expected})
        if observed != expected:
            raise RuntimeError(f'cross_person_alignment_design prespecified output hash mismatch: {key}')
    expected_b = lock.get('hashes', {})
    if set(expected_b) != set(latent_state_fit_PRIMARY_OUTPUT_MAP):
        raise RuntimeError('latent_state_fit primary-lock keys mismatch')
    b_rows = []
    for key, rel in latent_state_fit_PRIMARY_OUTPUT_MAP.items():
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing latent_state_fit fixed output: {p}')
        observed = sha(p)
        expected = expected_b[key]
        b_rows.append({'output_key': key, 'relative_path': str(rel), 'size_bytes': p.stat().st_size, 'expected_sha256': expected, 'observed_sha256': observed, 'sha256_match': observed == expected})
        if observed != expected:
            raise RuntimeError(f'latent_state_fit fixed output hash mismatch: {key}')
    return (contract, policy, execution, pd.DataFrame(c_rows), pd.DataFrame(b_rows))

def weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = (v[mask], w[mask])
    if len(v) == 0:
        return float('nan')
    order = np.argsort(v, kind='mergesort')
    v, w = (v[order], w[order])
    midpoint_cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, midpoint_cdf, v, left=v[0], right=v[-1]))

def weighted_location_iqr(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.array([weighted_quantile_1d(x[:, j], w, 0.5) for j in range(x.shape[1])])
    q1 = np.array([weighted_quantile_1d(x[:, j], w, 0.25) for j in range(x.shape[1])])
    q3 = np.array([weighted_quantile_1d(x[:, j], w, 0.75) for j in range(x.shape[1])])
    return (med, q3 - q1)

def weighted_pca_state(x: np.ndarray, w: np.ndarray, feature_names: Sequence[str]) -> StatePreprocess:
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise RuntimeError('Nonfinite/nonpositive state alignment input')
    med, iq = weighted_location_iqr(x, w)
    retained = iq > IQR_FLOOR
    if retained.sum() < 2:
        raise RuntimeError('Fewer than two state-alignment features survive weighted-IQR rule')
    z = np.clip((x[:, retained] - med[retained]) / iq[retained], -WINSOR_LIMIT, WINSOR_LIMIT)
    wn = w / w.sum()
    mean = np.sum(z * wn[:, None], axis=0)
    centered = z - mean
    weighted = centered * np.sqrt(wn[:, None])
    _, s, vt = np.linalg.svd(weighted, full_matrices=False)
    ncomp = min(STATE_ALIGNMENT_PCS, centered.shape[0] - 1, centered.shape[1])
    components = vt[:ncomp].copy()
    for j in range(ncomp):
        pivot = int(np.argmax(np.abs(components[j])))
        if components[j, pivot] < 0:
            components[j] *= -1
    scores = centered @ components.T
    eig = s[:ncomp] ** 2
    total = float(np.sum(s ** 2))
    evr = eig / total if total > 0 else np.zeros(ncomp)
    return StatePreprocess(list(feature_names), med, iq, retained, mean, components, evr, scores)

def transform_state(x: np.ndarray, prep: StatePreprocess) -> np.ndarray:
    z = np.clip((x[:, prep.retained] - prep.medians[prep.retained]) / prep.iqrs[prep.retained], -WINSOR_LIMIT, WINSOR_LIMIT)
    return (z - prep.weighted_mean) @ prep.components.T

def canonicalize(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    order = sorted(range(len(centers)), key=lambda i: tuple((float(v) for v in centers[i])))
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[int(v)] for v in labels], dtype=int)
    return (new_labels, centers[order].copy(), remap)

def assigned_distances_and_margins(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assigned = d[np.arange(len(x)), labels]
    if centers.shape[0] <= 1:
        margin = np.full(len(x), np.nan)
    else:
        alt = d.copy()
        alt[np.arange(len(x)), labels] = np.inf
        margin = np.min(alt, axis=1) - assigned
    return (assigned, margin)

def weighted_kmeanspp(x: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    p = w / w.sum()
    first = int(rng.choice(n, p=p))
    chosen = [first]
    closest = np.sum((x - x[first]) ** 2, axis=1)
    while len(chosen) < k:
        score = w * closest
        score[np.array(chosen, int)] = 0
        if score.sum() <= 0:
            remaining = np.setdiff1d(np.arange(n), np.array(chosen), assume_unique=False)
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(n, p=score / score.sum()))
        chosen.append(nxt)
        closest = np.minimum(closest, np.sum((x - x[nxt]) ** 2, axis=1))
    return x[np.array(chosen)].copy()

def constrained_assign(x: np.ndarray, centers: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    labels = np.full(len(x), -1, dtype=int)
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        if len(idx) > len(centers):
            raise RuntimeError(f'Participant {subject} has more local states than shared families')
        cost = np.sum((x[idx, None, :] - centers[None, :, :]) ** 2, axis=2)
        rr, cc = linear_sum_assignment(cost)
        labels[idx[rr]] = cc
    if np.any(labels < 0):
        raise RuntimeError('Constrained assignment left unassigned state objects')
    return labels

def fit_constrained_kmeans(x: np.ndarray, w: np.ndarray, subjects: np.ndarray, k: int, seed: int, n_init: int=N_INIT) -> FitResult:
    best: FitResult | None = None
    for init in range(n_init):
        rng = np.random.default_rng(stable_seed(f'state-init-{seed}', k, init, 0))
        centers = weighted_kmeanspp(x, w, k, rng)
        old_obj = math.inf
        converged = False
        labels = np.zeros(len(x), dtype=int)
        iterations = 0
        for iterations in range(1, MAX_ITER + 1):
            labels = constrained_assign(x, centers, subjects)
            new_centers = centers.copy()
            empty = []
            for j in range(k):
                idx = labels == j
                if not idx.any():
                    empty.append(j)
                else:
                    new_centers[j] = np.average(x[idx], axis=0, weights=w[idx])
            if empty:
                dist, _ = assigned_distances_and_margins(x, labels, centers)
                candidates = np.argsort(-(w * dist))
                used: set[int] = set()
                for j in empty:
                    pick = next((int(i) for i in candidates if int(i) not in used))
                    used.add(pick)
                    new_centers[j] = x[pick]
            labels2 = constrained_assign(x, new_centers, subjects)
            obj = float(np.sum(w * np.sum((x - new_centers[labels2]) ** 2, axis=1)))
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            if abs(old_obj - obj) <= TOL * max(1.0, abs(old_obj) if np.isfinite(old_obj) else 1.0) and shift <= math.sqrt(TOL):
                centers, labels, old_obj, converged = (new_centers, labels2, obj, True)
                break
            centers, labels, old_obj = (new_centers, labels2, obj)
        labels, centers, _ = canonicalize(labels, centers)
        dist, margin = assigned_distances_and_margins(x, labels, centers)
        result = FitResult(labels, centers, float(np.sum(w * dist)), iterations, converged, dist, margin)
        if best is None or result.objective < best.objective - 1e-12:
            best = result
    if best is None:
        raise RuntimeError('No constrained k-means initialization completed')
    return best

def fit_weighted_kmeans(x: np.ndarray, w: np.ndarray, k: int, seed: int, n_init: int=N_INIT) -> FitResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = KMeans(n_clusters=k, init='k-means++', n_init=n_init, max_iter=MAX_ITER, tol=TOL, random_state=seed, algorithm='lloyd')
        labels = model.fit_predict(x, sample_weight=w)
    labels, centers, _ = canonicalize(labels, model.cluster_centers_)
    dist, margin = assigned_distances_and_margins(x, labels, centers)
    return FitResult(labels, centers, float(np.sum(w * dist)), int(model.n_iter_), bool(model.n_iter_ < MAX_ITER), dist, margin)

def weighted_silhouette(x: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    counts = np.bincount(labels)
    if len(counts) < 2 or np.any(counts == 0) or len(x) <= len(counts):
        return float('nan')
    s = silhouette_samples(x, labels, metric='euclidean')
    return float(np.average(s, weights=w))

def state_collapse(labels: np.ndarray, subjects: np.ndarray, weights: np.ndarray, k: int) -> tuple[bool, int, float]:
    total = weights.sum()
    min_participants = min((len(set(subjects[labels == j])) if np.any(labels == j) else 0 for j in range(k)))
    min_occ = min((float(weights[labels == j].sum() / total) if np.any(labels == j) else 0.0 for j in range(k)))
    collapsed = min_participants < STATE_MIN_PARTICIPANTS or min_occ < STATE_MIN_OCCUPANCY
    return (collapsed, int(min_participants), float(min_occ))

def trajectory_collapse(labels: np.ndarray, subjects: np.ndarray, weights: np.ndarray, k: int) -> tuple[bool, int, int, float]:
    total = weights.sum()
    pcounts, ncounts, occs = ([], [], [])
    for j in range(k):
        idx = labels == j
        pcounts.append(len(set(subjects[idx])) if idx.any() else 0)
        ncounts.append(int(idx.sum()))
        occs.append(float(weights[idx].sum() / total) if idx.any() else 0.0)
    collapsed = min(pcounts) < TRAJ_MIN_PARTICIPANTS or min(ncounts) < TRAJ_MIN_TRIALS or min(occs) < TRAJ_MIN_OCCUPANCY
    return (collapsed, int(min(pcounts)), int(min(ncounts)), float(min(occs)))

def transform_geometry_raw(df: pd.DataFrame) -> np.ndarray:
    x = df[TRAJECTORY_FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite trajectory geometry')
    out = np.empty_like(x)
    out[:, 0:4] = np.log1p(x[:, 0:4])
    p = np.clip(x[:, 4], 1e-06, 1 - 1e-06)
    out[:, 4] = np.log(p / (1 - p))
    c = np.clip(x[:, 5], -1 + 1e-06, 1 - 1e-06)
    out[:, 5] = np.arctanh(c)
    if not np.isfinite(out).all():
        raise RuntimeError('Nonfinite transformed trajectory geometry')
    return out

def preprocess_trajectories(df: pd.DataFrame, weights: np.ndarray | None=None) -> TrajectoryPreprocess:
    transformed = transform_geometry_raw(df)
    subjects = df.subject_id.astype(str).to_numpy()
    if weights is None:
        counts = pd.Series(subjects).value_counts().to_dict()
        weights = np.array([1.0 / counts[s] for s in subjects], float)
    _, pooled_iqr = weighted_location_iqr(transformed, weights)
    pooled_iqr = np.where(pooled_iqr > IQR_FLOOR, pooled_iqr, 1.0)
    standardized = np.empty_like(transformed)
    medians: dict[str, np.ndarray] = {}
    iqrs: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    fallback: dict[str, np.ndarray] = {}
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        med = np.median(transformed[idx], axis=0)
        q1 = np.quantile(transformed[idx], 0.25, axis=0)
        q3 = np.quantile(transformed[idx], 0.75, axis=0)
        iq = q3 - q1
        fb = iq <= IQR_FLOOR
        scale = np.where(fb, pooled_iqr, iq)
        scale = np.where(scale > IQR_FLOOR, scale, 1.0)
        standardized[idx] = np.clip((transformed[idx] - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
        medians[str(subject)] = med
        iqrs[str(subject)] = iq
        scales[str(subject)] = scale
        fallback[str(subject)] = fb
    return TrajectoryPreprocess(transformed, standardized, pooled_iqr, medians, iqrs, scales, fallback)

def transform_heldout_trajectories(df: pd.DataFrame, pooled_iqr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformed = transform_geometry_raw(df)
    med = np.median(transformed, axis=0)
    iq = np.quantile(transformed, 0.75, axis=0) - np.quantile(transformed, 0.25, axis=0)
    fb = iq <= IQR_FLOOR
    scale = np.where(fb, pooled_iqr, iq)
    scale = np.where(scale > IQR_FLOOR, scale, 1.0)
    z = np.clip((transformed - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
    return (z, med, iq, fb)

def cache_meta(folder: Path, branch: str, k: int, input_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'policy_sha256': POLICY_SHA, 'script_sha256': sha(Path(__file__).resolve()), 'branch': branch, 'k': k, 'input_sha256': input_sha, 'n_init': N_INIT, 'stability_repeats': STABILITY_REPEATS}

def prepare_cache(folder: Path, expected: dict[str, Any], recompute: bool) -> None:
    meta_path = folder / 'meta.json'
    if recompute and folder.exists():
        shutil.rmtree(folder)
    if folder.exists():
        try:
            if jread(meta_path) != expected:
                shutil.rmtree(folder)
        except Exception:
            shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        atom_json(meta_path, expected)

def save_fit_npz(path: Path, fit: FitResult) -> None:
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, labels=fit.labels, centers=fit.centers, objective=np.array([fit.objective]), iterations=np.array([fit.iterations]), converged=np.array([int(fit.converged)]), distances=fit.distances, margins=fit.margins)
    os.replace(tmp, path)

def load_fit_npz(path: Path) -> FitResult:
    d = np.load(path, allow_pickle=False)
    return FitResult(d['labels'].astype(int), d['centers'].astype(float), float(d['objective'][0]), int(d['iterations'][0]), bool(d['converged'][0]), d['distances'].astype(float), d['margins'].astype(float))

def fit_state_candidates(state_ids: np.ndarray, subjects: np.ndarray, x: np.ndarray, weights: np.ndarray, feature_names: list[str], cache_root: Path, recompute: bool, note) -> tuple[StatePreprocess, dict[int, FitResult], pd.DataFrame, pd.DataFrame]:
    prep = weighted_pca_state(x, weights, feature_names)
    input_hash = array_sha(state_ids, subjects, x, weights)
    full: dict[int, FitResult] = {}
    stability_parts = []
    for k in STATE_FAMILY_COUNTS:
        note(f'State branch K={k}: fitting/loading full constrained alignment and {STABILITY_REPEATS} participant resamples.')
        folder = cache_root / f'state_k{k}'
        prepare_cache(folder, cache_meta(folder, 'state', k, input_hash), recompute)
        fit_path = folder / 'full_fit.npz'
        if fit_path.exists():
            fit = load_fit_npz(fit_path)
        else:
            fit = fit_constrained_kmeans(prep.scores, weights, subjects, k, stable_seed('state-full', k))
            save_fit_npz(fit_path, fit)
        full[k] = fit
        reps_path = folder / 'stability.csv'
        if reps_path.exists():
            reps = pd.read_csv(reps_path)
        else:
            reps = pd.DataFrame()
        done = set(reps.repeat.astype(int)) if len(reps) else set()
        participants = np.array(sorted(pd.unique(subjects)), dtype=object)
        n_keep = max(k, int(round(PARTICIPANT_FRACTION * len(participants))))
        rows = reps.to_dict('records') if len(reps) else []
        for r in range(STABILITY_REPEATS):
            if r in done:
                continue
            rng = np.random.default_rng(stable_seed('state-resample', k, r))
            kept = rng.choice(participants, size=n_keep, replace=False)
            idx = np.flatnonzero(np.isin(subjects, kept))
            try:
                rp = weighted_pca_state(x[idx], weights[idx], feature_names)
                rf = fit_constrained_kmeans(rp.scores, weights[idx], subjects[idx], k, stable_seed('state-resample-fit', k, r))
                ari = float(adjusted_rand_score(fit.labels[idx], rf.labels))
                collapsed, minp, minocc = state_collapse(rf.labels, subjects[idx], weights[idx], k)
                row = {'branch': 'state', 'k': k, 'repeat': r, 'retained_participants': len(kept), 'retained_objects': len(idx), 'ari_vs_full': ari, 'collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_weighted_occupancy': minocc, 'objective': rf.objective, 'converged': rf.converged, 'failure': ''}
            except Exception as e:
                row = {'branch': 'state', 'k': k, 'repeat': r, 'retained_participants': len(kept), 'retained_objects': len(idx), 'ari_vs_full': np.nan, 'collapsed': True, 'minimum_participants_per_family': 0, 'minimum_weighted_occupancy': 0.0, 'objective': np.nan, 'converged': False, 'failure': f'{type(e).__name__}: {e}'}
            rows.append(row)
            if (r + 1) % 5 == 0 or r + 1 == STABILITY_REPEATS:
                atom_csv(reps_path, pd.DataFrame(rows).sort_values('repeat'))
        stability_parts.append(pd.DataFrame(rows).sort_values('repeat'))
    stability = pd.concat(stability_parts, ignore_index=True)
    metrics = []
    for k, fit in full.items():
        reps = stability[stability.k == k]
        valid = reps[np.isfinite(reps.ari_vs_full)]
        collapsed, minp, minocc = state_collapse(fit.labels, subjects, weights, k)
        median_ari = float(valid.ari_vs_full.median()) if len(valid) else np.nan
        p10 = float(valid.ari_vs_full.quantile(0.1)) if len(valid) else np.nan
        collapse_rate = float(reps.collapsed.map(bval).mean())
        sil = weighted_silhouette(prep.scores, fit.labels, weights)
        eligible = bool(fit.converged and (not collapsed) and (len(valid) == STABILITY_REPEATS) and (median_ari >= MEDIAN_ARI_MIN) and (p10 >= P10_ARI_MIN) and (collapse_rate <= COLLAPSE_RATE_MAX))
        metrics.append({'branch': 'state', 'k': k, 'full_fit_converged': fit.converged, 'full_fit_iterations': fit.iterations, 'objective': fit.objective, 'participant_weighted_silhouette': sil, 'full_fit_collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_weighted_occupancy': minocc, 'stability_valid_repeats': len(valid), 'stability_failed_repeats': STABILITY_REPEATS - len(valid), 'median_ari': median_ari, 'ari_10th_percentile': p10, 'collapse_rate': collapse_rate, 'eligible': eligible})
    return (prep, full, stability, pd.DataFrame(metrics))

def fit_trajectory_candidates(df: pd.DataFrame, cache_root: Path, recompute: bool, note) -> tuple[TrajectoryPreprocess, np.ndarray, dict[int, FitResult], pd.DataFrame, pd.DataFrame]:
    prep = preprocess_trajectories(df)
    subjects = df.subject_id.astype(str).to_numpy()
    counts = pd.Series(subjects).value_counts().to_dict()
    weights = np.array([1.0 / counts[s] for s in subjects], float)
    ids = df.raw_behavior_row_id.astype(str).to_numpy()
    raw = df[TRAJECTORY_FEATURES].to_numpy(float)
    input_hash = array_sha(ids, subjects, raw, weights)
    full: dict[int, FitResult] = {}
    stability_parts = []
    participants = np.array(sorted(pd.unique(subjects)), dtype=object)
    n_keep_p = max(max(TRAJECTORY_FAMILY_COUNTS), int(round(PARTICIPANT_FRACTION * len(participants))))
    for k in TRAJECTORY_FAMILY_COUNTS:
        note(f'Trajectory branch K={k}: fitting/loading full weighted alignment and {STABILITY_REPEATS} two-level resamples.')
        folder = cache_root / f'trajectory_k{k}'
        prepare_cache(folder, cache_meta(folder, 'trajectory', k, input_hash), recompute)
        fit_path = folder / 'full_fit.npz'
        if fit_path.exists():
            fit = load_fit_npz(fit_path)
        else:
            fit = fit_weighted_kmeans(prep.standardized, weights, k, stable_seed('trajectory-full', k))
            save_fit_npz(fit_path, fit)
        full[k] = fit
        reps_path = folder / 'stability.csv'
        reps = pd.read_csv(reps_path) if reps_path.exists() else pd.DataFrame()
        done = set(reps.repeat.astype(int)) if len(reps) else set()
        rows = reps.to_dict('records') if len(reps) else []
        for r in range(STABILITY_REPEATS):
            if r in done:
                continue
            rng = np.random.default_rng(stable_seed('trajectory-resample', k, r))
            kept_p = rng.choice(participants, size=n_keep_p, replace=False)
            idx_parts = []
            for subject in kept_p:
                idx_s = np.flatnonzero(subjects == subject)
                n_keep_t = max(2, int(round(TRIAL_FRACTION * len(idx_s))))
                idx_parts.append(rng.choice(idx_s, size=n_keep_t, replace=False))
            idx = np.sort(np.concatenate(idx_parts))
            sub_df = df.iloc[idx].reset_index(drop=True)
            sub_subjects = sub_df.subject_id.astype(str).to_numpy()
            sub_counts = pd.Series(sub_subjects).value_counts().to_dict()
            sub_weights = np.array([1.0 / sub_counts[s] for s in sub_subjects], float)
            try:
                rp = preprocess_trajectories(sub_df, sub_weights)
                rf = fit_weighted_kmeans(rp.standardized, sub_weights, k, stable_seed('trajectory-resample-fit', k, r))
                ari = float(adjusted_rand_score(fit.labels[idx], rf.labels))
                collapsed, minp, minn, minocc = trajectory_collapse(rf.labels, sub_subjects, sub_weights, k)
                row = {'branch': 'trajectory', 'k': k, 'repeat': r, 'retained_participants': len(kept_p), 'retained_trajectories': len(idx), 'ari_vs_full': ari, 'collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_weighted_occupancy': minocc, 'objective': rf.objective, 'converged': rf.converged, 'failure': ''}
            except Exception as e:
                row = {'branch': 'trajectory', 'k': k, 'repeat': r, 'retained_participants': len(kept_p), 'retained_trajectories': len(idx), 'ari_vs_full': np.nan, 'collapsed': True, 'minimum_participants_per_family': 0, 'minimum_trials_per_family': 0, 'minimum_weighted_occupancy': 0.0, 'objective': np.nan, 'converged': False, 'failure': f'{type(e).__name__}: {e}'}
            rows.append(row)
            if (r + 1) % 5 == 0 or r + 1 == STABILITY_REPEATS:
                atom_csv(reps_path, pd.DataFrame(rows).sort_values('repeat'))
        stability_parts.append(pd.DataFrame(rows).sort_values('repeat'))
    stability = pd.concat(stability_parts, ignore_index=True)
    metrics = []
    for k, fit in full.items():
        reps = stability[stability.k == k]
        valid = reps[np.isfinite(reps.ari_vs_full)]
        collapsed, minp, minn, minocc = trajectory_collapse(fit.labels, subjects, weights, k)
        median_ari = float(valid.ari_vs_full.median()) if len(valid) else np.nan
        p10 = float(valid.ari_vs_full.quantile(0.1)) if len(valid) else np.nan
        collapse_rate = float(reps.collapsed.map(bval).mean())
        sil = weighted_silhouette(prep.standardized, fit.labels, weights)
        eligible = bool(fit.converged and (not collapsed) and (len(valid) == STABILITY_REPEATS) and (median_ari >= MEDIAN_ARI_MIN) and (p10 >= P10_ARI_MIN) and (collapse_rate <= COLLAPSE_RATE_MAX))
        metrics.append({'branch': 'trajectory', 'k': k, 'full_fit_converged': fit.converged, 'full_fit_iterations': fit.iterations, 'objective': fit.objective, 'participant_weighted_silhouette': sil, 'full_fit_collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_weighted_occupancy': minocc, 'stability_valid_repeats': len(valid), 'stability_failed_repeats': STABILITY_REPEATS - len(valid), 'median_ari': median_ari, 'ari_10th_percentile': p10, 'collapse_rate': collapse_rate, 'eligible': eligible})
    return (prep, weights, full, stability, pd.DataFrame(metrics))

def select_k(metrics: pd.DataFrame) -> tuple[int | None, str]:
    eligible = metrics[metrics.eligible.map(bval) & np.isfinite(metrics.participant_weighted_silhouette)].copy()
    if len(eligible) == 0:
        return (None, 'no_eligible_family_count_continuous_representation_only')
    best = float(eligible.participant_weighted_silhouette.max())
    within = eligible[eligible.participant_weighted_silhouette >= best - SILHOUETTE_DELTA]
    return (int(within.k.min()), 'smallest_eligible_k_within_silhouette_delta_0.05')

def map_labels_to_reference(train_labels: np.ndarray, reference_labels: np.ndarray, k: int, weights: np.ndarray | None=None) -> dict[int, int]:
    table = np.zeros((k, k), float)
    if weights is None:
        weights = np.ones(len(train_labels))
    for a, b, w in zip(train_labels, reference_labels, weights):
        table[int(a), int(b)] += float(w)
    rr, cc = linear_sum_assignment(-table)
    return {int(a): int(b) for a, b in zip(rr, cc)}

def state_lopo(x: np.ndarray, weights: np.ndarray, subjects: np.ndarray, feature_names: list[str], selected_k: int | None, full_fit: FitResult | None, state_manifest: pd.DataFrame, note) -> pd.DataFrame:
    columns = ['held_out_subject_id', 'state_object_id', 'local_state_1based', 'selected_shared_k', 'training_fit_converged', 'training_to_full_label_map', 'assigned_training_family_1based', 'assigned_full_mapped_family_1based', 'full_fit_family_1based', 'assignment_agrees_with_full_fit', 'assigned_squared_distance', 'assignment_margin', 'failure']
    if selected_k is None or full_fit is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for subject in sorted(pd.unique(subjects)):
        note(f'State LOPO: held out {subject}.')
        train = subjects != subject
        test = ~train
        try:
            prep = weighted_pca_state(x[train], weights[train], feature_names)
            fit = fit_constrained_kmeans(prep.scores, weights[train], subjects[train], selected_k, stable_seed(f'state-lopo-{subject}', selected_k))
            mapping = map_labels_to_reference(fit.labels, full_fit.labels[train], selected_k, weights[train])
            test_scores = transform_state(x[test], prep)
            raw_test_labels = constrained_assign(test_scores, fit.centers, subjects[test])
            dist, margin = assigned_distances_and_margins(test_scores, raw_test_labels, fit.centers)
            mapped = np.array([mapping[int(v)] for v in raw_test_labels], int)
            idxs = np.flatnonzero(test)
            for ii, idx in enumerate(idxs):
                rows.append({'held_out_subject_id': subject, 'state_object_id': state_manifest.iloc[idx].state_object_id, 'local_state_1based': int(state_manifest.iloc[idx].state_1based), 'selected_shared_k': selected_k, 'training_fit_converged': fit.converged, 'training_to_full_label_map': json.dumps({str(a + 1): b + 1 for a, b in mapping.items()}, sort_keys=True), 'assigned_training_family_1based': int(raw_test_labels[ii] + 1), 'assigned_full_mapped_family_1based': int(mapped[ii] + 1), 'full_fit_family_1based': int(full_fit.labels[idx] + 1), 'assignment_agrees_with_full_fit': bool(mapped[ii] == full_fit.labels[idx]), 'assigned_squared_distance': float(dist[ii]), 'assignment_margin': float(margin[ii]), 'failure': ''})
        except Exception as e:
            for idx in np.flatnonzero(test):
                rows.append({'held_out_subject_id': subject, 'state_object_id': state_manifest.iloc[idx].state_object_id, 'local_state_1based': int(state_manifest.iloc[idx].state_1based), 'selected_shared_k': selected_k, 'training_fit_converged': False, 'training_to_full_label_map': '', 'assigned_training_family_1based': np.nan, 'assigned_full_mapped_family_1based': np.nan, 'full_fit_family_1based': int(full_fit.labels[idx] + 1), 'assignment_agrees_with_full_fit': False, 'assigned_squared_distance': np.nan, 'assignment_margin': np.nan, 'failure': f'{type(e).__name__}: {e}'})
    return pd.DataFrame(rows, columns=columns)

def trajectory_lopo(df: pd.DataFrame, selected_k: int | None, full_fit: FitResult | None, note) -> pd.DataFrame:
    columns = ['held_out_subject_id', 'raw_behavior_row_id', 'trial_number', 'selected_shared_k', 'training_fit_converged', 'assigned_training_family_1based', 'assigned_full_mapped_family_1based', 'full_fit_family_1based', 'assignment_agrees_with_full_fit', 'assigned_squared_distance', 'assignment_margin', 'held_out_family_occupancy', 'failure']
    if selected_k is None or full_fit is None:
        return pd.DataFrame(columns=columns)
    subjects = df.subject_id.astype(str).to_numpy()
    rows = []
    for subject in sorted(pd.unique(subjects)):
        note(f'Trajectory LOPO: held out {subject}.')
        train_idx = np.flatnonzero(subjects != subject)
        test_idx = np.flatnonzero(subjects == subject)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_sub = train_df.subject_id.astype(str).to_numpy()
        counts = pd.Series(train_sub).value_counts().to_dict()
        w = np.array([1.0 / counts[s] for s in train_sub], float)
        try:
            prep = preprocess_trajectories(train_df, w)
            fit = fit_weighted_kmeans(prep.standardized, w, selected_k, stable_seed(f'trajectory-lopo-{subject}', selected_k))
            mapping = map_labels_to_reference(fit.labels, full_fit.labels[train_idx], selected_k, w)
            ztest, _, _, _ = transform_heldout_trajectories(test_df, prep.pooled_iqr)
            dmat = np.sum((ztest[:, None, :] - fit.centers[None, :, :]) ** 2, axis=2)
            raw_labels = np.argmin(dmat, axis=1)
            dist, margin = assigned_distances_and_margins(ztest, raw_labels, fit.centers)
            mapped = np.array([mapping[int(v)] for v in raw_labels], int)
            occupancy = pd.Series(mapped).value_counts(normalize=True).to_dict()
            for i, idx in enumerate(test_idx):
                rows.append({'held_out_subject_id': subject, 'raw_behavior_row_id': df.iloc[idx].raw_behavior_row_id, 'trial_number': int(df.iloc[idx].trial_number), 'selected_shared_k': selected_k, 'training_fit_converged': fit.converged, 'assigned_training_family_1based': int(raw_labels[i] + 1), 'assigned_full_mapped_family_1based': int(mapped[i] + 1), 'full_fit_family_1based': int(full_fit.labels[idx] + 1), 'assignment_agrees_with_full_fit': bool(mapped[i] == full_fit.labels[idx]), 'assigned_squared_distance': float(dist[i]), 'assignment_margin': float(margin[i]), 'held_out_family_occupancy': float(occupancy.get(int(mapped[i]), 0.0)), 'failure': ''})
        except Exception as e:
            for idx in test_idx:
                rows.append({'held_out_subject_id': subject, 'raw_behavior_row_id': df.iloc[idx].raw_behavior_row_id, 'trial_number': int(df.iloc[idx].trial_number), 'selected_shared_k': selected_k, 'training_fit_converged': False, 'assigned_training_family_1based': np.nan, 'assigned_full_mapped_family_1based': np.nan, 'full_fit_family_1based': int(full_fit.labels[idx] + 1), 'assignment_agrees_with_full_fit': False, 'assigned_squared_distance': np.nan, 'assignment_margin': np.nan, 'held_out_family_occupancy': np.nan, 'failure': f'{type(e).__name__}: {e}'})
    return pd.DataFrame(rows, columns=columns)

def state_output_tables(state_manifest: pd.DataFrame, feature_names: list[str], prep: StatePreprocess, fits: dict[int, FitResult], metrics: pd.DataFrame, stability: pd.DataFrame, selected_k: int | None, lopo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pre_rows = []
    for i, name in enumerate(feature_names):
        pre_rows.append({'feature_order_0based': i, 'feature_name': name, 'weighted_median': prep.medians[i], 'weighted_iqr': prep.iqrs[i], 'retained_for_weighted_pca': bool(prep.retained[i])})
    load_rows = []
    retained_idx = np.flatnonzero(prep.retained)
    for pc in range(prep.components.shape[0]):
        for jj, original in enumerate(retained_idx):
            load_rows.append({'aligned_state_pc_1based': pc + 1, 'feature_order_0based': int(original), 'feature_name': feature_names[original], 'loading': prep.components[pc, jj], 'explained_variance_ratio': prep.explained_variance_ratio[pc]})
    emb = state_manifest[['state_object_id', 'subject_id', 'state_1based', 'state_alignment_weight']].copy()
    for pc in range(prep.scores.shape[1]):
        emb[f'aligned_state_pc{pc + 1}'] = prep.scores[:, pc]
    pair_rows = []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            pair_rows.append({'state_object_id_a': emb.iloc[i].state_object_id, 'state_object_id_b': emb.iloc[j].state_object_id, 'same_participant': bool(emb.iloc[i].subject_id == emb.iloc[j].subject_id), 'euclidean_distance': float(np.linalg.norm(prep.scores[i] - prep.scores[j]))})
    assign_rows, proto_rows = ([], [])
    for k, fit in fits.items():
        for i in range(len(state_manifest)):
            assign_rows.append({'state_object_id': state_manifest.iloc[i].state_object_id, 'subject_id': state_manifest.iloc[i].subject_id, 'local_state_1based': int(state_manifest.iloc[i].state_1based), 'candidate_shared_k': k, 'shared_family_1based': int(fit.labels[i] + 1), 'assigned_squared_distance': fit.distances[i], 'assignment_margin': fit.margins[i], 'state_alignment_weight': state_manifest.iloc[i].state_alignment_weight})
        for family in range(k):
            for pc, value in enumerate(fit.centers[family]):
                proto_rows.append({'candidate_shared_k': k, 'shared_family_1based': family + 1, 'aligned_state_pc_1based': pc + 1, 'prototype_value': value})
    selection = pd.DataFrame([{'branch': 'stable_discrete_state_centroids', 'selected_shared_k': selected_k if selected_k is not None else np.nan, 'stable_discrete_family_solution': selected_k is not None, 'selection_reason': 'smallest_eligible_k_within_silhouette_delta_0.05' if selected_k is not None else 'no_eligible_family_count_continuous_representation_only'}])
    if selected_k is None:
        selected_assign = pd.DataFrame(columns=['state_object_id', 'subject_id', 'local_state_1based', 'selected_shared_k', 'shared_state_family_1based', 'assigned_squared_distance', 'assignment_margin', 'state_alignment_weight'])
        selected_proto = pd.DataFrame(columns=['selected_shared_k', 'shared_state_family_1based', 'aligned_state_pc_1based', 'prototype_value'])
    else:
        fit = fits[selected_k]
        selected_assign = pd.DataFrame({'state_object_id': state_manifest.state_object_id, 'subject_id': state_manifest.subject_id, 'local_state_1based': state_manifest.state_1based.astype(int), 'selected_shared_k': selected_k, 'shared_state_family_1based': fit.labels + 1, 'assigned_squared_distance': fit.distances, 'assignment_margin': fit.margins, 'state_alignment_weight': state_manifest.state_alignment_weight})
        selected_proto = pd.DataFrame([{'selected_shared_k': selected_k, 'shared_state_family_1based': family + 1, 'aligned_state_pc_1based': pc + 1, 'prototype_value': fit.centers[family, pc]} for family in range(selected_k) for pc in range(fit.centers.shape[1])])
    return {'state_preprocessing': pd.DataFrame(pre_rows), 'state_pca_loadings': pd.DataFrame(load_rows), 'state_embedding': emb, 'state_pairwise': pd.DataFrame(pair_rows), 'state_candidates': metrics, 'state_candidate_assignments': pd.DataFrame(assign_rows), 'state_candidate_prototypes': pd.DataFrame(proto_rows), 'state_stability': stability, 'state_selection': selection, 'state_selected_assignments': selected_assign, 'state_selected_prototypes': selected_proto, 'state_lopo': lopo}

def trajectory_output_tables(df: pd.DataFrame, prep: TrajectoryPreprocess, weights: np.ndarray, fits: dict[int, FitResult], metrics: pd.DataFrame, stability: pd.DataFrame, selected_k: int | None, lopo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pre_rows = []
    for subject in sorted(prep.participant_medians):
        for j, feature in enumerate(TRAJECTORY_FEATURES):
            pre_rows.append({'subject_id': subject, 'feature_order_0based': j, 'feature_name': feature, 'within_participant_median': prep.participant_medians[subject][j], 'within_participant_iqr': prep.participant_iqrs[subject][j], 'scale_used': prep.participant_scales[subject][j], 'pooled_participant_balanced_iqr': prep.pooled_iqr[j], 'pooled_fallback_used': bool(prep.fallback_used[subject][j])})
    std = df[['subject_id', 'raw_behavior_row_id', 'trial_number']].copy()
    std['trajectory_alignment_weight'] = weights
    for j, feature in enumerate(TRAJECTORY_FEATURES):
        std[f'standardized_{feature}'] = prep.standardized[:, j]
    assign_rows, centroid_rows = ([], [])
    for k, fit in fits.items():
        for i in range(len(df)):
            assign_rows.append({'subject_id': df.iloc[i].subject_id, 'raw_behavior_row_id': df.iloc[i].raw_behavior_row_id, 'trial_number': int(df.iloc[i].trial_number), 'candidate_shared_k': k, 'trajectory_family_1based': int(fit.labels[i] + 1), 'assigned_squared_distance': fit.distances[i], 'assignment_margin': fit.margins[i], 'trajectory_alignment_weight': weights[i]})
        for family in range(k):
            for j, feature in enumerate(TRAJECTORY_FEATURES):
                centroid_rows.append({'candidate_shared_k': k, 'trajectory_family_1based': family + 1, 'feature_order_0based': j, 'standardized_geometry_feature': feature, 'centroid_value': fit.centers[family, j]})
    selection = pd.DataFrame([{'branch': 'continuous_trajectory_geometry', 'selected_shared_k': selected_k if selected_k is not None else np.nan, 'stable_discrete_family_solution': selected_k is not None, 'selection_reason': 'smallest_eligible_k_within_silhouette_delta_0.05' if selected_k is not None else 'no_eligible_family_count_continuous_representation_only'}])
    if selected_k is None:
        selected_assign = pd.DataFrame(columns=['subject_id', 'raw_behavior_row_id', 'trial_number', 'selected_shared_k', 'trajectory_family_1based', 'assigned_squared_distance', 'assignment_margin', 'trajectory_alignment_weight'])
        selected_cent = pd.DataFrame(columns=['selected_shared_k', 'trajectory_family_1based', 'feature_order_0based', 'standardized_geometry_feature', 'centroid_value'])
    else:
        fit = fits[selected_k]
        selected_assign = pd.DataFrame({'subject_id': df.subject_id, 'raw_behavior_row_id': df.raw_behavior_row_id, 'trial_number': df.trial_number.astype(int), 'selected_shared_k': selected_k, 'trajectory_family_1based': fit.labels + 1, 'assigned_squared_distance': fit.distances, 'assignment_margin': fit.margins, 'trajectory_alignment_weight': weights})
        selected_cent = pd.DataFrame([{'selected_shared_k': selected_k, 'trajectory_family_1based': family + 1, 'feature_order_0based': j, 'standardized_geometry_feature': feature, 'centroid_value': fits[selected_k].centers[family, j]} for family in range(selected_k) for j, feature in enumerate(TRAJECTORY_FEATURES)])
    return {'trajectory_preprocessing': pd.DataFrame(pre_rows), 'trajectory_standardized': std, 'trajectory_candidates': metrics, 'trajectory_candidate_assignments': pd.DataFrame(assign_rows), 'trajectory_candidate_centroids': pd.DataFrame(centroid_rows), 'trajectory_stability': stability, 'trajectory_selection': selection, 'trajectory_selected_assignments': selected_assign, 'trajectory_selected_centroids': selected_cent, 'trajectory_lopo': lopo}

def write_branch_lock(out: dict[str, Path], branch_keys: Sequence[str]) -> dict[str, Any]:
    hashes = {key: sha(out[key]) for key in branch_keys}
    lock = {'generated_at': now(), 'phase': PHASE, 'alignment_policy_sha256': POLICY_SHA, 'branch_outputs_fixed_before_cross_branch_linkage': True, 'reports_outcomes_context_qc_read_before_lock': False, 'cross_branch_linkage_started_before_lock': False, 'hashes': hashes}
    atom_json(out['branch_lock'], lock)
    return lock

def weighted_information_metrics(path_labels: np.ndarray, family_labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    paths = sorted(pd.unique(path_labels))
    fams = sorted(pd.unique(family_labels))
    pi = {v: i for i, v in enumerate(paths)}
    fj = {v: j for j, v in enumerate(fams)}
    tab = np.zeros((len(paths), len(fams)), float)
    for a, b, w in zip(path_labels, family_labels, weights):
        tab[pi[a], fj[b]] += float(w)
    total = tab.sum()
    pxy = tab / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    denom = px @ py
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    nmi = float(2 * mi / (hx + hy)) if hx + hy > 0 else 0.0
    expected = px @ py * total
    chi = float(np.sum((tab[expected > 0] - expected[expected > 0]) ** 2 / expected[expected > 0]))
    denom_v = total * max(1, min(tab.shape[0] - 1, tab.shape[1] - 1))
    v = float(math.sqrt(max(0.0, chi / denom_v))) if denom_v > 0 else 0.0
    return {'mutual_information': mi, 'normalized_mutual_information': nmi, 'path_by_geometry_family_cramers_v': v}

def cross_branch_linkage(project: Path, paths: dict[str, Path], state_selected: pd.DataFrame, trajectory_selected: pd.DataFrame, note) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_cols = ['subject_id', 'raw_behavior_row_id', 'trial_number', 'global_state_family_path', 'trajectory_family_1based', 'participant_balanced_weight']
    cross_cols = ['global_state_family_path', 'trajectory_family_1based', 'trial_count', 'participant_balanced_weight']
    null_cols = ['repeat', 'mutual_information', 'normalized_mutual_information', 'path_by_geometry_family_cramers_v']
    test_cols = ['statistic', 'observed', 'null_mean', 'null_sd', 'one_sided_permutation_p', 'null_repeats']
    if len(state_selected) == 0 or len(trajectory_selected) == 0:
        return (pd.DataFrame(columns=row_cols), pd.DataFrame(columns=cross_cols), pd.DataFrame(columns=null_cols), pd.DataFrame([{'statistic': 'cross_branch_linkage_omitted_no_stable_solution', 'observed': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'one_sided_permutation_p': np.nan, 'null_repeats': 0}], columns=test_cols))
    note('Both branch locks exist; loading only latent_state_fit state lineage for prespecified outcome-free cross-branch linkage.')
    assn = pd.read_csv(paths['state_assignments'], usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', 'epoch_family', 'state_1based'], low_memory=False)
    ensure_no_banned_columns(assn, 'state_assignments lineage')
    mapping = state_selected.set_index(['subject_id', 'local_state_1based']).shared_state_family_1based.to_dict()
    assn['shared_family_1based'] = [mapping.get((str(s), int(k)), np.nan) for s, k in zip(assn.subject_id, assn.state_1based)]
    assn = assn[np.isfinite(assn.shared_family_1based)].copy()
    if len(assn) == 0:
        return (pd.DataFrame(columns=row_cols), pd.DataFrame(columns=cross_cols), pd.DataFrame(columns=null_cols), pd.DataFrame([{'statistic': 'cross_branch_linkage_omitted_no_eligible_rows', 'observed': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'one_sided_permutation_p': np.nan, 'null_repeats': 0}], columns=test_cols))
    pivot = assn.pivot(index=['subject_id', 'raw_behavior_row_id', 'trial_number'], columns='epoch_family', values='shared_family_1based').reset_index()
    if not set(EPOCH_ORDER).issubset(pivot.columns):
        raise RuntimeError('Cross-branch state lineage lacks one or more prespecified epoch families')
    pivot['global_state_family_path'] = pivot.apply(lambda r: '>'.join((f'F{int(r[e])}' for e in EPOCH_ORDER)), axis=1)
    geom = trajectory_selected[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based', 'trajectory_alignment_weight']]
    rows = pivot.merge(geom, on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='one_to_one')
    rows = rows.rename(columns={'trajectory_alignment_weight': 'participant_balanced_weight'})
    rows = rows[row_cols].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    if len(rows) == 0:
        raise RuntimeError('No rows remain for cross-branch linkage despite two stable branch solutions')
    grouped = rows.groupby(['global_state_family_path', 'trajectory_family_1based'], as_index=False).agg(trial_count=('raw_behavior_row_id', 'size'), participant_balanced_weight=('participant_balanced_weight', 'sum'))
    observed = weighted_information_metrics(rows.global_state_family_path.to_numpy(object), rows.trajectory_family_1based.to_numpy(int), rows.participant_balanced_weight.to_numpy(float))
    null_rows = []
    subjects = rows.subject_id.astype(str).to_numpy()
    original_family = rows.trajectory_family_1based.to_numpy(int)
    for rep in range(LINKAGE_NULL_REPEATS):
        rng = np.random.default_rng(stable_seed('cross-linkage', 0, rep))
        perm = original_family.copy()
        for subject in pd.unique(subjects):
            idx = np.flatnonzero(subjects == subject)
            perm[idx] = rng.permutation(perm[idx])
        stats = weighted_information_metrics(rows.global_state_family_path.to_numpy(object), perm, rows.participant_balanced_weight.to_numpy(float))
        null_rows.append({'repeat': rep, **stats})
    nulls = pd.DataFrame(null_rows)
    tests = []
    for name, obs in observed.items():
        vals = nulls[name].to_numpy(float)
        tests.append({'statistic': name, 'observed': obs, 'null_mean': float(vals.mean()), 'null_sd': float(vals.std(ddof=1)), 'one_sided_permutation_p': float((1 + np.sum(vals >= obs)) / (LINKAGE_NULL_REPEATS + 1)), 'null_repeats': LINKAGE_NULL_REPEATS})
    return (rows, grouped, nulls, pd.DataFrame(tests))

def write_report(path: Path, summary: dict[str, Any], state_metrics: pd.DataFrame, traj_metrics: pd.DataFrame, linkage_tests: pd.DataFrame) -> None:
    ssel = summary['state_branch']['selected_shared_k']
    tsel = summary['trajectory_branch']['selected_shared_k']
    lines = ['# Phase cross_person_alignment_fit — prespecified outcome-free cross-person alignment', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f'- prespecified alignment-policy SHA-256: `{POLICY_SHA}`', '- Reports/outcomes/context/QC values read: **no**', '- Validation/confirmation opened: **no**', '', '## Stable local-state centroid branch', '', f"- Local state objects: **{summary['counts']['stable_local_state_objects']}** from **{summary['counts']['stable_discrete_participants']}** participants.", f"- Selected shared family count: **{(ssel if ssel is not None else 'none forced')}**.", f"- LOPO object-level agreement with the full fit: **{summary['state_branch']['lopo_agreement']:.3f}**." if summary['state_branch']['lopo_rows'] else '- LOPO omitted because no stable shared family solution was selected.', '', '## Continuous trajectory-geometry branch', '', f"- Trajectories: **{summary['counts']['continuous_trajectories']:,}** from all **{summary['counts']['all_participants']}** participants.", f"- Selected shared family count: **{(tsel if tsel is not None else 'none forced')}**.", f"- LOPO trajectory-level agreement with the full fit: **{summary['trajectory_branch']['lopo_agreement']:.3f}**." if summary['trajectory_branch']['lopo_rows'] else '- LOPO omitted because no stable shared family solution was selected.', '', '## Outcome-free cross-branch linkage', '']
    if len(linkage_tests) and linkage_tests.statistic.astype(str).str.startswith('cross_branch_linkage_omitted').any():
        lines += ['- Linkage was omitted because at least one branch lacked a stable discrete family solution.', '']
    else:
        for _, row in linkage_tests.iterrows():
            lines.append(f'- `{row.statistic}` = **{row.observed:.6f}**, permutation p = **{row.one_sided_permutation_p:.6f}**.')
        lines.append('')
    lines += ['## Guardrails', '', '- These families are reproducible alignments of the prespecified representation, not universal emotion or valence categories.', '- Family counts and assignments were fixed before any report/context mapping.', '- Continuous-only participants were represented without fabricated local states.', '- mapping_design may specification a post-hoc mapping design, but it may not revise any cross_person_alignment_fit alignment result.', '', '### Candidate metrics: state branch', '', '```text', state_metrics.to_string(index=False), '```', '', '### Candidate metrics: trajectory branch', '', '```text', traj_metrics.to_string(index=False), '```', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')

def self_test() -> None:
    x = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [4.0, 6.0, 1.0]])
    w = np.array([0.5, 0.5, 0.5, 0.5])
    prep = weighted_pca_state(x, w, ['a', 'b', 'constant'])
    assert prep.retained.tolist() == [True, True, False]
    subjects = np.array(['s1', 's1', 's2', 's2'], object)
    fit = fit_constrained_kmeans(prep.scores, w, subjects, 2, 1, n_init=5)
    for s in pd.unique(subjects):
        assert len(set(fit.labels[subjects == s])) == 2
    df = pd.DataFrame({'subject_id': ['s1'] * 6 + ['s2'] * 6, 'raw_behavior_row_id': [f'r{i}' for i in range(12)], 'trial_number': list(range(6)) * 2, 'early_to_late_norm': np.linspace(0.1, 2, 12), 'late_to_rating_norm': np.linspace(0.2, 3, 12), 'early_to_rating_direct_norm': np.linspace(0.1, 1.5, 12), 'total_path_length': np.linspace(0.3, 5, 12), 'directness_ratio': np.linspace(0.05, 0.95, 12), 'segment_cosine': np.linspace(-0.9, 0.9, 12)})
    tp = preprocess_trajectories(df)
    assert np.isfinite(tp.standardized).all()
    tw = np.array([1 / 6] * 12)
    tf = fit_weighted_kmeans(tp.standardized, tw, 2, 1, n_init=5)
    assert len(tf.labels) == 12
    stats = weighted_information_metrics(np.array(['a', 'a', 'b', 'b']), np.array([1, 1, 2, 2]), np.ones(4))
    assert stats['normalized_mutual_information'] > 0.99
    print('cross_person_alignment_fit self-test: PASS')

def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    paths = {k: project / v for k, v in REL.items()}
    out = output_paths(project)
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"Output already exists; review or use --force: {out['contract']}")
    log_lines: list[str] = []

    def note(message: str) -> None:
        line = f'[{now()}] {message}'
        log_lines.append(line)
        print(line, flush=True)
        out['log'].write_text(''.join(log_lines) + '', encoding='utf-8')
    note('Verifying cross_person_alignment_design specification, execution authorization, and all latent_state_fit/cross_person_alignment_design SHA-256 locks.')
    contract_c, policy, execution, c_rehash, b_rehash = verify_inputs(project, paths)
    atom_csv(out['cross_person_alignment_design_rehash'], c_rehash)
    atom_csv(out['latent_state_fit_rehash'], b_rehash)
    note('Loading only prespecified outcome-free state-centroid and invariant trajectory-geometry inputs.')
    state_manifest = pd.read_csv(paths['state_manifest'], low_memory=False)
    trajectory_manifest = pd.read_csv(paths['trajectory_manifest'], low_memory=False)
    schema = pd.read_csv(paths['feature_schema'], low_memory=False)
    cent = pd.read_csv(paths['centroids_common'], usecols=['subject_id', 'state_1based', 'compact_cell_0based', 'feature_name', 'within_person_stage_standardized_centroid'], low_memory=False)
    continuous = pd.read_csv(paths['continuous_trajectories'], usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', *TRAJECTORY_FEATURES], low_memory=False)
    for df, name in [(state_manifest, 'state_manifest'), (trajectory_manifest, 'trajectory_manifest'), (schema, 'feature_schema'), (cent, 'centroids_common restricted'), (continuous, 'continuous_trajectories restricted')]:
        ensure_no_banned_columns(df, name)
    if len(state_manifest) != 40 or state_manifest.subject_id.nunique() != 19:
        raise RuntimeError('Unexpected stable-state manifest size')
    if len(trajectory_manifest) != 1237 or trajectory_manifest.subject_id.nunique() != 31:
        raise RuntimeError('Unexpected trajectory manifest size')
    state_order = state_manifest[['state_object_id', 'subject_id', 'state_1based']].copy()
    wide = cent.pivot(index=['subject_id', 'state_1based'], columns='compact_cell_0based', values='within_person_stage_standardized_centroid')
    wide = wide.reindex(pd.MultiIndex.from_frame(state_order[['subject_id', 'state_1based']]))
    if wide.shape != (40, PER_EPOCH_COMPACT) or wide.isna().any().any():
        raise RuntimeError(f'Unexpected state centroid matrix: {wide.shape}')
    feature_lookup = cent[['compact_cell_0based', 'feature_name']].drop_duplicates().sort_values('compact_cell_0based')
    if len(feature_lookup) != PER_EPOCH_COMPACT:
        raise RuntimeError('Unexpected common-axis feature schema')
    state_x = wide.to_numpy(float)
    state_weights = state_manifest.state_alignment_weight.to_numpy(float)
    state_subjects = state_manifest.subject_id.astype(str).to_numpy()
    if not np.allclose(state_manifest.groupby('subject_id').state_alignment_weight.sum().to_numpy(float), 1.0):
        raise RuntimeError('State weights do not sum to one per participant')
    manifest_geom = trajectory_manifest[['subject_id', 'raw_behavior_row_id', 'trial_number', *TRAJECTORY_FEATURES]].copy()
    continuous = manifest_geom.merge(continuous, on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='inner', validate='one_to_one', suffixes=('_manifest', '_source'))
    if len(continuous) != 1237:
        raise RuntimeError('Trajectory manifest and latent_state_fit trajectory identifiers disagree')
    for feature in TRAJECTORY_FEATURES:
        a = continuous[f'{feature}_manifest'].to_numpy(float)
        b = continuous[f'{feature}_source'].to_numpy(float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-12):
            raise RuntimeError(f'Trajectory manifest and latent_state_fit geometry disagree: {feature}')
        continuous[feature] = a
    continuous = continuous[['subject_id', 'raw_behavior_row_id', 'trial_number', *TRAJECTORY_FEATURES]]
    cache_root = project / 'outputs/cache/cross_person_alignment_fit_cross_person_alignment'
    cache_root.mkdir(parents=True, exist_ok=True)
    note('Executing stable local-state centroid alignment candidates and participant-level stability.')
    state_prep, state_fits, state_stability, state_metrics = fit_state_candidates(state_manifest.state_object_id.astype(str).to_numpy(), state_subjects, state_x, state_weights, feature_lookup.feature_name.astype(str).tolist(), cache_root, args.recompute_cache, note)
    state_selected_k, state_reason = select_k(state_metrics)
    state_full_selected = state_fits.get(state_selected_k) if state_selected_k is not None else None
    note(f"State branch prespecified selection: {(state_selected_k if state_selected_k is not None else 'no forced family solution')}.")
    state_lopo_df = state_lopo(state_x, state_weights, state_subjects, feature_lookup.feature_name.astype(str).tolist(), state_selected_k, state_full_selected, state_manifest, note)
    state_tables = state_output_tables(state_manifest, feature_lookup.feature_name.astype(str).tolist(), state_prep, state_fits, state_metrics, state_stability, state_selected_k, state_lopo_df)
    for key, df in state_tables.items():
        atom_csv(out[key], df)
    note('Executing continuous trajectory-geometry alignment candidates and two-level stability.')
    traj_prep, traj_weights, traj_fits, traj_stability, traj_metrics = fit_trajectory_candidates(continuous, cache_root, args.recompute_cache, note)
    traj_selected_k, traj_reason = select_k(traj_metrics)
    traj_full_selected = traj_fits.get(traj_selected_k) if traj_selected_k is not None else None
    note(f"Trajectory branch prespecified selection: {(traj_selected_k if traj_selected_k is not None else 'no forced family solution')}.")
    traj_lopo_df = trajectory_lopo(continuous, traj_selected_k, traj_full_selected, note)
    traj_tables = trajectory_output_tables(continuous, traj_prep, traj_weights, traj_fits, traj_metrics, traj_stability, traj_selected_k, traj_lopo_df)
    for key, df in traj_tables.items():
        atom_csv(out[key], df)
    branch_keys = ['state_preprocessing', 'state_pca_loadings', 'state_embedding', 'state_pairwise', 'state_candidates', 'state_candidate_assignments', 'state_candidate_prototypes', 'state_stability', 'state_selection', 'state_selected_assignments', 'state_selected_prototypes', 'state_lopo', 'trajectory_preprocessing', 'trajectory_standardized', 'trajectory_candidates', 'trajectory_candidate_assignments', 'trajectory_candidate_centroids', 'trajectory_stability', 'trajectory_selection', 'trajectory_selected_assignments', 'trajectory_selected_centroids', 'trajectory_lopo']
    note('Writing SHA-256 primary lock for both alignment branches before cross-branch linkage.')
    branch_lock = write_branch_lock(out, branch_keys)
    note('Executing only the prespecified outcome-free cross-branch linkage after branch lock.')
    linkage_rows, linkage_crosstab, linkage_nulls, linkage_tests = cross_branch_linkage(project, paths, state_tables['state_selected_assignments'], traj_tables['trajectory_selected_assignments'], note)
    atom_csv(out['linkage_rows'], linkage_rows)
    atom_csv(out['linkage_crosstab'], linkage_crosstab)
    atom_csv(out['linkage_nulls'], linkage_nulls)
    atom_csv(out['linkage_tests'], linkage_tests)
    all_fixed_keys = branch_keys + ['linkage_rows', 'linkage_crosstab', 'linkage_nulls', 'linkage_tests']
    final_lock = {'generated_at': now(), 'phase': PHASE, 'alignment_policy_sha256': POLICY_SHA, 'branch_primary_lock_sha256': sha(out['branch_lock']), 'reports_outcomes_context_qc_read': False, 'validation_opened': False, 'confirmation_opened': False, 'hashes': {key: sha(out[key]) for key in all_fixed_keys}}
    atom_json(out['final_lock'], final_lock)
    state_lopo_valid = state_lopo_df[state_lopo_df.failure.fillna('') == ''] if len(state_lopo_df) else state_lopo_df
    traj_lopo_valid = traj_lopo_df[traj_lopo_df.failure.fillna('') == ''] if len(traj_lopo_df) else traj_lopo_df
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'prespecified_outcome_free_cross_person_alignment_complete_candidate', 'alignment_policy_sha256': POLICY_SHA, 'counts': {'all_participants': 31, 'stable_discrete_participants': 19, 'continuous_only_participants': 12, 'stable_local_state_objects': 40, 'continuous_trajectories': 1237, 'state_candidate_family_counts': len(STATE_FAMILY_COUNTS), 'trajectory_candidate_family_counts': len(TRAJECTORY_FAMILY_COUNTS), 'state_stability_replicates': len(state_stability), 'trajectory_stability_replicates': len(traj_stability), 'linkage_null_replicates': len(linkage_nulls)}, 'state_branch': {'selected_shared_k': state_selected_k, 'selection_reason': state_reason, 'stable_discrete_family_solution': state_selected_k is not None, 'eligible_candidate_counts': state_metrics.loc[state_metrics.eligible.map(bval), 'k'].astype(int).tolist(), 'lopo_rows': len(state_lopo_valid), 'lopo_agreement': float(state_lopo_valid.assignment_agrees_with_full_fit.map(bval).mean()) if len(state_lopo_valid) else None}, 'trajectory_branch': {'selected_shared_k': traj_selected_k, 'selection_reason': traj_reason, 'stable_discrete_family_solution': traj_selected_k is not None, 'eligible_candidate_counts': traj_metrics.loc[traj_metrics.eligible.map(bval), 'k'].astype(int).tolist(), 'lopo_rows': len(traj_lopo_valid), 'lopo_agreement': float(traj_lopo_valid.assignment_agrees_with_full_fit.map(bval).mean()) if len(traj_lopo_valid) else None}, 'cross_branch_linkage_executed': bool(len(linkage_nulls) == LINKAGE_NULL_REPEATS), 'decision': {'all_prespecified_candidate_counts_fitted': True, 'all_prespecified_stability_resamples_executed_or_failure_recorded': len(state_stability) == len(STATE_FAMILY_COUNTS) * STABILITY_REPEATS and len(traj_stability) == len(TRAJECTORY_FAMILY_COUNTS) * STABILITY_REPEATS, 'silhouette_parsimony_and_no_force_rules_applied': True, 'lopo_diagnostics_completed_without_revising_selection': True, 'alignment_branches_fixed_before_cross_branch_linkage': True, 'reports_outcomes_context_qc_read': False, 'posthoc_report_mapping_performed': False, 'validation_opened': False, 'confirmation_opened': False}}
    atom_json(out['summary'], summary)
    write_report(out['report'], summary, state_metrics, traj_metrics, linkage_tests)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'next_phase_role': 'specification a strictly post-hoc Discovery-only mapping design from fixed cross_person_alignment_fit state/trajectory families and continuous embeddings to reports, context, and stimulus variables without revising alignment', 'required_inputs': {'cross_person_alignment_fit_contract': str(out['contract']), 'cross_person_alignment_fit_final_lock': str(out['final_lock']), 'state_family_assignments': str(out['state_selected_assignments']), 'state_embedding': str(out['state_embedding']), 'trajectory_family_assignments': str(out['trajectory_selected_assignments']), 'trajectory_standardized_geometry': str(out['trajectory_standardized']), 'cross_branch_linkage': str(out['linkage_rows'])}, 'required_boundary': ['all cross_person_alignment_fit family counts, preprocessing, embeddings, assignments, and locks are immutable', 'mapping_design specifications mapping hypotheses and multiplicity control before any subjective report/context association is calculated', 'absence of a stable discrete family in either branch must be respected; continuous representations remain analyzable', 'Discovery only; Validation and Confirmation remain closed'], 'forbidden': ['refitting alignment after opening reports or context', 'selecting or relabeling families from valence, liking, arousal, emotion labels, context, or stimulus identity', 'dropping continuous-only participants to improve an association', 'opening validation or confirmation participants'], 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['next_policy'], next_policy)
    contract = {**summary, 'source_cross_person_alignment_design_contract_sha256': sha(paths['cross_person_alignment_design_contract']), 'prespecified_alignment_policy_file_sha256': sha(paths['alignment_policy']), 'cross_person_alignment_design_execution_policy_sha256': sha(paths['execution_policy']), 'latent_state_fit_primary_lock_sha256': sha(paths['latent_state_fit_lock']), 'branch_primary_lock_sha256': sha(out['branch_lock']), 'final_output_lock_sha256': sha(out['final_lock']), 'branch_primary_output_hashes': branch_lock['hashes'], 'final_output_hashes': final_lock['hashes'], 'implementation_details': {'weighted_quantile_definition': 'linear interpolation on midpoint weighted empirical CDF', 'state_weighted_pca_centering': 'participant-balanced weighted arithmetic mean after weighted-median/IQR scaling', 'canonical_label_rule_applied': True, 'state_constraint_solver': 'per-participant Hungarian assignment inside Lloyd updates', 'trajectory_solver': 'sample-weighted sklearn KMeans with prespecified n_init/max_iter/tol', 'linkage_statistics': 'participant-balanced weighted contingency statistics'}, 'outcomes_modified': False, 'latent_state_fit_outputs_modified': False, 'cross_person_alignment_design_outputs_modified': False, 'feature_store_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atom_json(out['contract'], contract)
    files = [*paths.values(), *out.values()]
    return 0
if __name__ == '__main__':
    raise SystemExit(main())