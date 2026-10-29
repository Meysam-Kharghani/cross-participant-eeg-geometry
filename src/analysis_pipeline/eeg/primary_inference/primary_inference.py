# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase primary_inference: prespecified primary bootstrap and permutation inference.

This phase consumes the fully prespecified primary_models outer predictions and outcome-free
split matrices. It performs the inference committed before the primary
outer-test results were inspected:

- 5,000 participant-cluster bootstrap replicates;
- 1,000 within-participant/context circular-shift permutations;
- complete inner PCA/ridge retuning for every permutation;
- Holm correction across valence and liking for the primary P1-minus-B0 test;
- the prespecified signal-specific support gate requiring positive bootstrap lower
  bounds for P1-B0 and P1-B1, plus Holm-adjusted permutation p < .05.

The primary metric is the outcome_model_integrity.1-prespecified pooled person-relative R² on raw-scale
outer predictions:

    1 - sum_i,t (y_it - pred_it)^2 / sum_i,t (y_it - mean_i)^2

where mean_i is computed over all prespecified primary rows of participant i.

Permutation plan and technical feasibility correction
-----------------------------------------------------
The final inference implementation conditions complete circular-shift proposals on feasibility across every prespecified split and both outcomes.
All outcome-free design transformations remain exactly those created by
primary_models. primary_inference builds deterministic SVD decompositions of those fixed matrices;
then every permuted target receives complete inner hyperparameter retuning.
Before permutation inference, the SVD engine must exactly reproduce primary_models's
observed selected hyperparameters and outer predictions.

No secondary model, validation participant, or confirmation participant is
opened. Failure of the primary support gate is not described as EEG failure;
it means that this prespecified full-signal representation did not demonstrate
signal-specific incremental prediction under the predeclared test."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
REL = {'summary': Path('outputs/qc/primary_models_primary_nested_model_summary.json'), 'contract': Path('config/primary_models_primary_nested_model_contract.json'), 'policy': Path('analysis_specifications/S051_primary_models_inference.json'), 'predictions': Path('outputs/tables/primary_models_primary_outer_predictions.csv'), 'selected': Path('outputs/tables/primary_models_selected_hyperparameters.csv'), 'inner_scores': Path('outputs/tables/primary_models_inner_candidate_scores.csv'), 'cache_inventory': Path('outputs/tables/primary_models_cache_inventory.csv')}
CACHE_REL = Path('outputs/large/primary_models_primary_nested_cache_v1')
INFERENCE_CACHE_REL = Path('outputs/large/primary_inference_primary_inference_cache_v1')
SEED = 540
BOOTSTRAP_REPLICATES = 5000
PERMUTATION_REPLICATES = 1000
OUTCOMES = ['valence', 'liking']
MODELS = ['B0_seen_covariate_baseline', 'B1_seen_qc_only', 'P1_seen_all3_full_signal']
ALPHAS = np.asarray([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=float)
COMPONENTS = [8, 16, 32, 64, 128]
FOLDS = [1, 2, 3, 4, 5]
SD_FLOOR = 1e-08
TIE_TOLERANCE = 1e-12
EXPECTED = {'cache_files': 254, 'cache_bytes': 113418187, 'prediction_rows': 7422, 'primary_rows': 1237, 'subjects': 31, 'selected_rows': 30, 'inner_score_rows': 630}
INFERENCE_POLICY = {'phase': 'primary_inference', 'seed': SEED, 'primary_metric': 'pooled person-relative R2 = 1 - raw-scale outer SSE / raw-scale participant-centered SST', 'bootstrap': {'unit': 'participant', 'replicates': BOOTSTRAP_REPLICATES, 'rng': 'numpy.random.default_rng(540)', 'draw': '31 participant indices independently with replacement per replicate', 'interval': 'percentile 95 percent', 'same_draws_all_models_outcomes': True}, 'permutation': {'replicates': PERMUTATION_REPLICATES, 'rng': 'numpy.random.default_rng(540), independent of bootstrap RNG', 'group': 'participant x context', 'ordering': 'subject_acquisition_order_1based', 'allowed_offsets': 'min(offset, group_size-offset) >= 3', 'sampling': 'uniform over allowed integer offsets', 'same_row_map_valence_liking': True, 'same_row_map_all_epoch_families': True, 'proposal_distribution': 'independent uniform allowed offset for every participant-context group', 'feasibility_condition': 'for both outcomes and every prespecified outer/final and inner training split, each evaluated participant has at least two training rows and sample SD > 1e-8', 'sampling_rule': 'reject an infeasible complete proposal and continue until 1000 jointly feasible maps', 'conditional_randomization_distribution': True, 'rejections_fully_recorded': True, 'complete_inner_retuning': True}, 'primary_permutation_statistic': 'Delta pooled person-relative R2: P1 minus B0', 'ancillary_permutation_statistic': 'Delta pooled person-relative R2: P1 minus B1', 'permutation_p': '(1 + count(null statistic >= observed statistic)) / 1001', 'multiplicity': 'Holm across valence and liking for primary P1-minus-B0 p-values', 'support_gate_per_outcome': ['bootstrap lower 95 percent CI for P1-minus-B0 R2 > 0', 'bootstrap lower 95 percent CI for P1-minus-B1 R2 > 0', 'Holm-adjusted primary permutation p < 0.05'], 'observed_engine_reproduction_required': True, 'secondary_models_fitted': False, 'secondary_models_remain_mandatory': True, 'validation_opened': False, 'confirmation_opened': False}
POLICY_SHA = hashlib.sha256(json.dumps(INFERENCE_POLICY, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
DESIGNS = [('B0', MODELS[0], None), ('B1', MODELS[1], None), *[(f'P1_c{component}', MODELS[2], component) for component in COMPONENTS]]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the prespecified participant bootstrap and fully retuned circular-shift permutation inference.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--force', action='store_true', help='Rewrite phase-level outputs; valid inference caches are reused.')
    parser.add_argument('--recompute-cache', action='store_true', help='Rebuild valid SVD, plan, and permutation checkpoints.')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Required files are missing:\n  - ' + '\n  - '.join((str(path) for path in missing)))

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + '', encoding='utf-8')
    os.replace(temporary, path)

def stored_manifest(root: Path, exclude: set[str]) -> list[dict[str, Any]]:
    return [{'relative_path': str(path.relative_to(root)), 'size_bytes': int(path.stat().st_size), 'sha256': sha256_file(path)} for path in sorted(root.rglob('*')) if path.is_file() and str(path.relative_to(root)) not in exclude]

def checkpoint_ok(root: Path, summary_name: str, input_signature: str) -> bool:
    summary_path = root / summary_name
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except Exception:
        return False
    if summary.get('complete') is not True:
        return False
    if summary.get('policy_sha256') != POLICY_SHA:
        return False
    if summary.get('input_signature_sha256') != input_signature:
        return False
    for item in summary.get('stored_files', []):
        path = root / item['relative_path']
        if not path.is_file():
            return False
        if int(path.stat().st_size) != int(item['size_bytes']):
            return False
        if sha256_file(path) != item['sha256']:
            return False
    return True

def load_inputs(project: Path):
    paths = {name: project / relative for name, relative in REL.items()}
    require_files(paths.values())
    summary = read_json(paths['summary'])
    contract = read_json(paths['contract'])
    policy = read_json(paths['policy'])
    if summary.get('phase') != 'primary_models':
        raise RuntimeError('Unexpected primary_models prerequisite phase.')
    required = ['complete_outer_prediction_coverage', 'all_transforms_training_only', 'complete_inner_hyperparameter_selection', 'primary_inference_inference_permitted']
    for key in required:
        if not bool(summary['decision'].get(key)):
            raise RuntimeError(f'primary_models prerequisite failed: {key}.')
    if summary['decision'].get('scientific_signal_support_decided'):
        raise RuntimeError('primary_models unexpectedly made the scientific support decision.')
    if summary['decision'].get('validation_opened') or summary['decision'].get('confirmation_opened'):
        raise RuntimeError('Validation or confirmation was opened.')
    if policy.get('next_phase') != 'primary_inference' or not policy.get('next_phase_permitted'):
        raise RuntimeError('primary_models did not permit the prespecified primary_inference inference phase.')
    hash_map = {'summary': 'summary', 'predictions': 'predictions', 'selected': 'selected', 'inner_scores': 'inner_scores', 'cache_inventory': 'cache_inventory'}
    for logical_name, hash_key in hash_map.items():
        observed = sha256_file(paths[logical_name])
        expected = contract['output_hashes'][hash_key]
        if observed != expected:
            raise RuntimeError(f'{logical_name}: hash changed after primary_models.')
    predictions = pd.read_csv(paths['predictions'], low_memory=False)
    selected = pd.read_csv(paths['selected'], low_memory=False)
    inner_scores = pd.read_csv(paths['inner_scores'], low_memory=False)
    cache_inventory = pd.read_csv(paths['cache_inventory'], low_memory=False)
    if len(predictions) != EXPECTED['prediction_rows']:
        raise RuntimeError('primary_models prediction count changed.')
    if len(selected) != EXPECTED['selected_rows'] or len(inner_scores) != EXPECTED['inner_score_rows']:
        raise RuntimeError('primary_models tuning count changed.')
    if len(cache_inventory) != EXPECTED['cache_files']:
        raise RuntimeError('primary_models cache inventory count changed.')
    if int(cache_inventory['size_bytes'].sum()) != EXPECTED['cache_bytes']:
        raise RuntimeError('primary_models cache byte count changed.')
    return (paths, summary, contract, policy, predictions, selected, inner_scores, cache_inventory)

def rehash_cache(cache_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(cache_inventory)
    for position, row in cache_inventory.iterrows():
        path = Path(str(row['stored_path']))
        if not path.is_file():
            raise RuntimeError(f'primary_models cache file is missing: {path}')
        size = int(path.stat().st_size)
        if size != int(row['size_bytes']):
            raise RuntimeError(f'primary_models cache size mismatch: {path}')
        observed_hash = sha256_file(path)
        if observed_hash != row['sha256']:
            raise RuntimeError(f'primary_models cache SHA-256 mismatch: {path}')
        rows.append({'inventory_row_1based': int(position + 1), 'stored_path': str(path), 'size_bytes': size, 'sha256': observed_hash, 'passed': True})
        if (position + 1) % 50 == 0 or position + 1 == total:
            print(f'[{now_iso()}] Rehashed {position + 1}/{total} primary_models cache files.', flush=True)
    return pd.DataFrame(rows)

def load_primary_metadata(cache_root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    metadata_path = cache_root / 'assembled_primary' / 'metadata.csv'
    if not metadata_path.is_file():
        raise RuntimeError('primary_models assembled metadata is missing.')
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if len(metadata) != EXPECTED['primary_rows'] or metadata['subject_id'].nunique() != EXPECTED['subjects']:
        raise RuntimeError('primary_models primary metadata dimensions changed.')
    metadata = metadata.sort_values('primary_row_0based').reset_index(drop=True)
    if metadata['primary_row_0based'].astype(int).tolist() != list(range(len(metadata))):
        raise RuntimeError('Primary row indices are not contiguous.')
    observed = predictions[['raw_behavior_row_id', 'outcome', 'observed_raw']].drop_duplicates().pivot(index='raw_behavior_row_id', columns='outcome', values='observed_raw')
    check = metadata.set_index('raw_behavior_row_id')[OUTCOMES].sort_index()
    observed = observed.sort_index()
    if not check.index.equals(observed.index):
        raise RuntimeError('Prediction and metadata trial sets differ.')
    np.testing.assert_allclose(check[OUTCOMES].to_numpy(float), observed[OUTCOMES].to_numpy(float), rtol=0, atol=1e-12)
    metadata['subject_code'] = pd.Categorical(metadata['subject_id'], categories=sorted(metadata['subject_id'].unique()), ordered=True).codes.astype(int)
    if (metadata['subject_code'] < 0).any():
        raise RuntimeError('Subject coding failed.')
    return metadata

def load_split(cache_root: Path, split_id: str) -> dict[str, np.ndarray]:
    directory = cache_root / 'fold_transforms' / split_id
    mapping = {'train': 'train_indices.npy', 'eval': 'eval_indices.npy', 'cov_train': 'cov_train.npy', 'cov_eval': 'cov_eval.npy', 'qc_train': 'qc_train.npy', 'qc_eval': 'qc_eval.npy', 'pca_train': 'pca_train.npy', 'pca_eval': 'pca_eval.npy'}
    require_files((directory / name for name in mapping.values()))
    return {key: np.load(directory / file_name, mmap_mode='r', allow_pickle=False) for key, file_name in mapping.items()}

def design_matrices(split: dict[str, np.ndarray], model_id: str, component: int | None):
    cov_train = np.asarray(split['cov_train'], dtype=np.float64)
    cov_eval = np.asarray(split['cov_eval'], dtype=np.float64)
    if model_id == MODELS[0]:
        return (cov_train, cov_eval)
    if model_id == MODELS[1]:
        return (np.concatenate([cov_train, np.asarray(split['qc_train'], dtype=np.float64)], axis=1), np.concatenate([cov_eval, np.asarray(split['qc_eval'], dtype=np.float64)], axis=1))
    if model_id == MODELS[2] and component is not None:
        if component > split['pca_train'].shape[1]:
            raise RuntimeError('PCA component request exceeds primary_models cache.')
        return (np.concatenate([cov_train, np.asarray(split['pca_train'][:, :component], dtype=np.float64)], axis=1), np.concatenate([cov_eval, np.asarray(split['pca_eval'][:, :component], dtype=np.float64)], axis=1))
    raise RuntimeError(f'Unknown design: {model_id}, {component}.')

def split_ids() -> list[str]:
    values = []
    for outer in FOLDS:
        values.append(f'outer-{outer}')
        for inner in FOLDS:
            if inner != outer:
                values.append(f'outer-{outer}_inner-{inner}')
    return values

def build_svd_split(primary_models_cache_root: Path, svd_root: Path, split_id: str, base_signature: str, recompute: bool) -> Path:
    final = svd_root / split_id
    summary_name = 'svd_summary.json'
    input_signature = canonical_sha256({'base': base_signature, 'split_id': split_id})
    if not recompute and checkpoint_ok(final, summary_name, input_signature):
        return final
    temporary = svd_root / f'.{split_id}.tmp'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        split = load_split(primary_models_cache_root, split_id)
        np.save(temporary / 'train_indices_int64.npy', np.asarray(split['train'], dtype=np.int64), allow_pickle=False)
        np.save(temporary / 'eval_indices_int64.npy', np.asarray(split['eval'], dtype=np.int64), allow_pickle=False)
        design_rows = []
        for design_id, model_id, component in DESIGNS:
            x_train, x_eval = design_matrices(split, model_id, component)
            x_mean = x_train.mean(axis=0)
            centered_train = x_train - x_mean
            centered_eval = x_eval - x_mean
            u, singular, vt = np.linalg.svd(centered_train, full_matrices=False)
            eval_projection = centered_eval @ vt.T
            np.save(temporary / f'{design_id}_u_float64.npy', u.astype(np.float64), allow_pickle=False)
            np.save(temporary / f'{design_id}_s_float64.npy', singular.astype(np.float64), allow_pickle=False)
            np.save(temporary / f'{design_id}_eval_projection_float64.npy', eval_projection.astype(np.float64), allow_pickle=False)
            design_rows.append({'design_id': design_id, 'model_id': model_id, 'pca_components': component, 'train_rows': int(x_train.shape[0]), 'eval_rows': int(x_eval.shape[0]), 'design_columns': int(x_train.shape[1]), 'svd_rank_dimension': int(len(singular)), 'minimum_singular_value': float(singular.min()), 'maximum_singular_value': float(singular.max())})
        pd.DataFrame(design_rows).to_csv(temporary / 'design_inventory.csv', index=False)
        summary = {'generated_at': now_iso(), 'complete': True, 'phase': 'primary_inference', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': input_signature, 'split_id': split_id, 'design_count': len(DESIGNS), 'stored_files': stored_manifest(temporary, {summary_name})}
        atomic_json(temporary / summary_name, summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(temporary, final)
        return final
    except Exception:
        if temporary.exists():
            (temporary / 'FAILED.txt').write_text(f'Failed at {now_iso()}.', encoding='utf-8')
        raise

def load_svd_design(svd_root: Path, split_id: str, design_id: str):
    directory = svd_root / split_id
    return (np.load(directory / f'{design_id}_u_float64.npy', mmap_mode='r', allow_pickle=False), np.load(directory / f'{design_id}_s_float64.npy', mmap_mode='r', allow_pickle=False), np.load(directory / f'{design_id}_eval_projection_float64.npy', mmap_mode='r', allow_pickle=False))

def svd_predict_grid(u: np.ndarray, singular: np.ndarray, eval_projection: np.ndarray, y_train: np.ndarray, alphas: np.ndarray) -> list[np.ndarray]:
    target = np.asarray(y_train, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    target_mean = target.mean(axis=0)
    projected_target = np.asarray(u, dtype=np.float64).T @ (target - target_mean[None, :])
    output = []
    singular = np.asarray(singular, dtype=np.float64)
    eval_projection = np.asarray(eval_projection, dtype=np.float64)
    for alpha in np.asarray(alphas, dtype=float):
        weights = (singular / (singular * singular + float(alpha)))[:, None] * projected_target
        output.append(target_mean[None, :] + eval_projection @ weights)
    return output

def target_scale_batch(metadata: pd.DataFrame, y_matrix: np.ndarray, train_indices: np.ndarray, eval_indices: np.ndarray):
    y = np.asarray(y_matrix, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    train_codes = metadata.iloc[train_indices]['subject_code'].to_numpy(dtype=int)
    eval_codes = metadata.iloc[eval_indices]['subject_code'].to_numpy(dtype=int)
    y_train_raw = y[train_indices]
    y_eval_raw = y[eval_indices]
    y_train_z = np.empty_like(y_train_raw)
    y_eval_z = np.empty_like(y_eval_raw)
    eval_mean = np.empty_like(y_eval_raw)
    eval_sd = np.empty_like(y_eval_raw)
    for code in np.unique(eval_codes):
        train_rows = np.flatnonzero(train_codes == code)
        eval_rows = np.flatnonzero(eval_codes == code)
        if len(train_rows) < 2:
            raise RuntimeError(f'Subject code {code} has fewer than two training rows.')
        mean = y_train_raw[train_rows].mean(axis=0)
        sd = y_train_raw[train_rows].std(axis=0, ddof=1)
        if not np.isfinite(sd).all() or (sd <= SD_FLOOR).any():
            raise RuntimeError(f'Fatal permuted target SD for subject code {code}.')
        y_train_z[train_rows] = (y_train_raw[train_rows] - mean[None, :]) / sd[None, :]
        y_eval_z[eval_rows] = (y_eval_raw[eval_rows] - mean[None, :]) / sd[None, :]
        eval_mean[eval_rows] = mean[None, :]
        eval_sd[eval_rows] = sd[None, :]
    return (y_train_z, y_eval_z, eval_mean, eval_sd)

def candidate_metadata(model_id: str) -> pd.DataFrame:
    rows = []
    if model_id == MODELS[2]:
        for component in COMPONENTS:
            for alpha in ALPHAS:
                rows.append({'pca_components': component, 'ridge_alpha': float(alpha)})
    else:
        for alpha in ALPHAS:
            rows.append({'pca_components': None, 'ridge_alpha': float(alpha)})
    frame = pd.DataFrame(rows)
    components = frame['pca_components'].fillna(0).astype(int).to_numpy()
    alpha = frame['ridge_alpha'].to_numpy(float)
    preference_order = np.lexsort((-alpha, components))
    preference_rank = np.empty(len(frame), dtype=int)
    preference_rank[preference_order] = np.arange(len(frame), dtype=int)
    frame['preference_rank'] = preference_rank
    frame['candidate_index'] = np.arange(len(frame), dtype=int)
    return frame

def design_id_for(model_id: str, component: int | None) -> str:
    if model_id == MODELS[0]:
        return 'B0'
    if model_id == MODELS[1]:
        return 'B1'
    if model_id == MODELS[2] and component is not None:
        return f'P1_c{int(component)}'
    raise RuntimeError('Invalid design request.')

def choose_candidates(score: np.ndarray, candidates: pd.DataFrame) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    minimum = values.min(axis=1)
    eligible = values <= minimum[:, None] + TIE_TOLERANCE
    rank = candidates['preference_rank'].to_numpy(dtype=int)
    ranked = np.where(eligible, rank[None, :], np.iinfo(np.int32).max)
    return np.argmin(ranked, axis=1).astype(int)

def accumulate_subject_sse(destination: np.ndarray, candidate_index: int, error_squared: np.ndarray, eval_subject_codes: np.ndarray) -> None:
    for code in np.unique(eval_subject_codes):
        rows = eval_subject_codes == code
        destination[code, :, candidate_index] += error_squared[rows].sum(axis=0)

def fit_outer_batch(metadata: pd.DataFrame, y_matrix: np.ndarray, outer_fold: int, primary_models_cache_root: Path, svd_root: Path, *, return_predictions: bool):
    replicate_count = np.asarray(y_matrix).shape[1] if np.asarray(y_matrix).ndim == 2 else 1
    outer_split_id = f'outer-{outer_fold}'
    outer_split = load_split(primary_models_cache_root, outer_split_id)
    outer_train = np.asarray(outer_split['train'], dtype=int)
    outer_eval = np.asarray(outer_split['eval'], dtype=int)
    subject_count = int(metadata['subject_code'].max()) + 1
    results: dict[str, dict[str, np.ndarray]] = {}
    for model_id in MODELS:
        candidates = candidate_metadata(model_id)
        candidate_lookup = {(None if pd.isna(row.pca_components) else int(row.pca_components), float(row.ridge_alpha)): int(row.candidate_index) for row in candidates.itertuples(index=False)}
        subject_sse = np.zeros((subject_count, replicate_count, len(candidates)), dtype=np.float64)
        subject_counts = np.zeros(subject_count, dtype=int)
        for inner_fold in FOLDS:
            if inner_fold == outer_fold:
                continue
            split_id = f'outer-{outer_fold}_inner-{inner_fold}'
            split = load_split(primary_models_cache_root, split_id)
            train_indices = np.asarray(split['train'], dtype=int)
            eval_indices = np.asarray(split['eval'], dtype=int)
            y_train_z, y_eval_z, _, _ = target_scale_batch(metadata, y_matrix, train_indices, eval_indices)
            eval_codes = metadata.iloc[eval_indices]['subject_code'].to_numpy(dtype=int)
            subject_counts += np.bincount(eval_codes, minlength=subject_count)
            component_values = COMPONENTS if model_id == MODELS[2] else [None]
            for component in component_values:
                design_id = design_id_for(model_id, component)
                u, singular, eval_projection = load_svd_design(svd_root, split_id, design_id)
                predictions = svd_predict_grid(u, singular, eval_projection, y_train_z, ALPHAS)
                for alpha, prediction in zip(ALPHAS, predictions):
                    candidate_index = candidate_lookup[component, float(alpha)]
                    accumulate_subject_sse(subject_sse, candidate_index, (y_eval_z - prediction) ** 2, eval_codes)
        if (subject_counts <= 0).any():
            raise RuntimeError(f'Outer {outer_fold}: incomplete inner OOF subject coverage.')
        score = np.mean(subject_sse / subject_counts[:, None, None], axis=0)
        selected_index = choose_candidates(score, candidates)
        selected_component = candidates.iloc[selected_index]['pca_components'].fillna(-1).to_numpy(dtype=int)
        selected_alpha = candidates.iloc[selected_index]['ridge_alpha'].to_numpy(dtype=float)
        selected_inner_mse = score[np.arange(replicate_count), selected_index]
        y_outer_train_z, y_outer_eval_z, eval_mean, eval_sd = target_scale_batch(metadata, y_matrix, outer_train, outer_eval)
        prediction_z = np.empty((len(outer_eval), replicate_count), dtype=np.float64)
        component_values = COMPONENTS if model_id == MODELS[2] else [None]
        for component in component_values:
            design_id = design_id_for(model_id, component)
            u, singular, eval_projection = load_svd_design(svd_root, outer_split_id, design_id)
            predictions = svd_predict_grid(u, singular, eval_projection, y_outer_train_z, ALPHAS)
            for alpha, prediction in zip(ALPHAS, predictions):
                candidate_index = candidate_lookup[component, float(alpha)]
                columns = np.flatnonzero(selected_index == candidate_index)
                if len(columns):
                    prediction_z[:, columns] = prediction[:, columns]
        prediction_raw = prediction_z * eval_sd + eval_mean
        y_eval_raw = np.asarray(y_matrix, dtype=np.float64)
        if y_eval_raw.ndim == 1:
            y_eval_raw = y_eval_raw[:, None]
        y_eval_raw = y_eval_raw[outer_eval]
        sse = np.sum((y_eval_raw - prediction_raw) ** 2, axis=0)
        result = {'selected_index': selected_index, 'selected_component': selected_component, 'selected_alpha': selected_alpha, 'selected_inner_mse': selected_inner_mse, 'sse': sse, 'eval_indices': outer_eval}
        if return_predictions:
            result['prediction_raw'] = prediction_raw
            result['prediction_z'] = prediction_z
            result['observed_z'] = y_outer_eval_z
        results[model_id] = result
    return results

def build_svd_cache(primary_models_cache_root: Path, inference_cache_root: Path, base_signature: str, recompute: bool) -> Path:
    svd_root = inference_cache_root / 'svd_transforms'
    svd_root.mkdir(parents=True, exist_ok=True)
    for position, split_id in enumerate(split_ids(), start=1):
        print(f'[{now_iso()}] Building or reusing SVD split {position}/25: {split_id}.', flush=True)
        build_svd_split(primary_models_cache_root, svd_root, split_id, base_signature, recompute)
    return svd_root

def reproduce_observed_engine(metadata: pd.DataFrame, predictions: pd.DataFrame, selected: pd.DataFrame, primary_models_cache_root: Path, svd_root: Path) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        y = metadata[outcome].to_numpy(dtype=float)[:, None]
        for outer in FOLDS:
            result = fit_outer_batch(metadata, y, outer, primary_models_cache_root, svd_root, return_predictions=True)
            for model_id in MODELS:
                expected_selected = selected[(selected['outcome'] == outcome) & (selected['outer_fold'].astype(int) == outer) & (selected['model_id'] == model_id)]
                if len(expected_selected) != 1:
                    raise RuntimeError('Observed selected-hyperparameter row is not unique.')
                expected_component = expected_selected.iloc[0]['pca_components']
                expected_component = -1 if pd.isna(expected_component) else int(expected_component)
                expected_alpha = float(expected_selected.iloc[0]['ridge_alpha'])
                observed_component = int(result[model_id]['selected_component'][0])
                observed_alpha = float(result[model_id]['selected_alpha'][0])
                if observed_component != expected_component or observed_alpha != expected_alpha:
                    raise RuntimeError(f'Observed engine selection mismatch: {outcome}, outer {outer}, {model_id}.')
                eval_indices = result[model_id]['eval_indices']
                expected_predictions = predictions[(predictions['outcome'] == outcome) & (predictions['outer_fold'].astype(int) == outer) & (predictions['model_id'] == model_id)].set_index('primary_row_0based')['predicted_raw']
                expected_vector = expected_predictions.loc[eval_indices].to_numpy(dtype=float)
                observed_vector = result[model_id]['prediction_raw'][:, 0]
                maximum_difference = float(np.max(np.abs(expected_vector - observed_vector)))
                if maximum_difference > 1e-08:
                    raise RuntimeError(f'Observed prediction reproduction mismatch {maximum_difference}: {outcome}, outer {outer}, {model_id}.')
                rows.append({'outcome': outcome, 'outer_fold': outer, 'model_id': model_id, 'selected_component_expected': expected_component, 'selected_component_reproduced': observed_component, 'selected_alpha_expected': expected_alpha, 'selected_alpha_reproduced': observed_alpha, 'maximum_absolute_prediction_difference': maximum_difference, 'passed': True})
    return pd.DataFrame(rows)

def point_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    subject_rows = []
    for (outcome, model_id), group in predictions.groupby(['outcome', 'model_id'], sort=True):
        group = group.sort_values('primary_row_0based')
        y = group['observed_raw'].to_numpy(dtype=float)
        prediction = group['predicted_raw'].to_numpy(dtype=float)
        participant_mean = group.groupby('subject_id')['observed_raw'].transform('mean').to_numpy(dtype=float)
        sse = float(np.sum((y - prediction) ** 2))
        sst = float(np.sum((y - participant_mean) ** 2))
        pooled_r2 = float(1.0 - sse / sst)
        for subject_id, subject in group.groupby('subject_id', sort=True):
            sy = subject['observed_raw'].to_numpy(dtype=float)
            sp = subject['predicted_raw'].to_numpy(dtype=float)
            denominator = float(np.sum((sy - sy.mean()) ** 2))
            subject_rows.append({'outcome': outcome, 'model_id': model_id, 'subject_id': subject_id, 'rows': int(len(subject)), 'sse': float(np.sum((sy - sp) ** 2)), 'sst': denominator, 'person_relative_r2': float(1.0 - np.sum((sy - sp) ** 2) / denominator), 'mae': float(np.mean(np.abs(sy - sp))), 'absolute_error_sum': float(np.sum(np.abs(sy - sp)))})
        current_subject = [row for row in subject_rows if row['outcome'] == outcome and row['model_id'] == model_id]
        metric_rows.append({'outcome': outcome, 'model_id': model_id, 'pooled_person_relative_r2': pooled_r2, 'participant_median_r2': float(np.median([row['person_relative_r2'] for row in current_subject])), 'participant_median_mae': float(np.median([row['mae'] for row in current_subject])), 'pooled_raw_mae': float(np.mean(np.abs(y - prediction))), 'pooled_pearson_r': float(np.corrcoef(y, prediction)[0, 1]), 'pooled_spearman_r': float(pd.Series(y).corr(pd.Series(prediction), method='spearman')), 'sse': sse, 'sst': sst, 'trial_rows': int(len(group)), 'subjects': int(group['subject_id'].nunique())})
    metrics = pd.DataFrame(metric_rows)
    subjects = pd.DataFrame(subject_rows)
    contrast_rows = []
    for outcome in OUTCOMES:
        lookup = metrics[metrics['outcome'] == outcome].set_index('model_id')
        contrast_rows.extend([{'outcome': outcome, 'comparison': 'P1_minus_B0', 'delta_pooled_person_relative_r2': float(lookup.loc[MODELS[2], 'pooled_person_relative_r2'] - lookup.loc[MODELS[0], 'pooled_person_relative_r2'])}, {'outcome': outcome, 'comparison': 'P1_minus_B1', 'delta_pooled_person_relative_r2': float(lookup.loc[MODELS[2], 'pooled_person_relative_r2'] - lookup.loc[MODELS[1], 'pooled_person_relative_r2'])}])
    return (metrics, subjects, pd.DataFrame(contrast_rows))

def bootstrap_plan(subject_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, subject_count, size=(BOOTSTRAP_REPLICATES, subject_count), endpoint=False)
    counts = np.zeros((BOOTSTRAP_REPLICATES, subject_count), dtype=np.int16)
    for replicate in range(BOOTSTRAP_REPLICATES):
        counts[replicate] = np.bincount(draws[replicate], minlength=subject_count)
    return (draws.astype(np.int16), counts)

def bootstrap_inference(subject_metrics: pd.DataFrame, point_metrics_frame: pd.DataFrame):
    subject_levels = sorted(subject_metrics['subject_id'].unique())
    subject_count = len(subject_levels)
    if subject_count != EXPECTED['subjects']:
        raise RuntimeError('Bootstrap subject count changed.')
    subject_lookup = {subject: index for index, subject in enumerate(subject_levels)}
    draws, counts = bootstrap_plan(subject_count)
    long_rows = []
    interval_rows = []
    replicate_values: dict[tuple[str, str], np.ndarray] = {}
    for outcome in OUTCOMES:
        for model_id in MODELS:
            frame = subject_metrics[(subject_metrics['outcome'] == outcome) & (subject_metrics['model_id'] == model_id)].copy()
            frame['subject_code'] = frame['subject_id'].map(subject_lookup)
            frame = frame.sort_values('subject_code')
            if frame['subject_code'].tolist() != list(range(subject_count)):
                raise RuntimeError('Bootstrap subject metric coverage is incomplete.')
            sse = frame['sse'].to_numpy(dtype=float)
            sst = frame['sst'].to_numpy(dtype=float)
            r2 = frame['person_relative_r2'].to_numpy(dtype=float)
            mae = frame['mae'].to_numpy(dtype=float)
            ae_sum = frame['absolute_error_sum'].to_numpy(dtype=float)
            n_rows = frame['rows'].to_numpy(dtype=float)
            pooled_r2 = 1.0 - counts @ sse / (counts @ sst)
            median_r2 = np.median(r2[draws], axis=1)
            pooled_mae = counts @ ae_sum / (counts @ n_rows)
            median_mae = np.median(mae[draws], axis=1)
            replicate_values[outcome, model_id] = pooled_r2
            for metric_name, values in [('pooled_person_relative_r2', pooled_r2), ('participant_median_r2', median_r2), ('pooled_raw_mae', pooled_mae), ('participant_median_mae', median_mae)]:
                point = float(point_metrics_frame[(point_metrics_frame['outcome'] == outcome) & (point_metrics_frame['model_id'] == model_id)][metric_name].iloc[0])
                lower, upper = np.quantile(values, [0.025, 0.975])
                interval_rows.append({'outcome': outcome, 'model_id': model_id, 'comparison': '', 'metric': metric_name, 'point_estimate': point, 'bootstrap_mean': float(values.mean()), 'ci_lower_95': float(lower), 'ci_upper_95': float(upper), 'replicates': BOOTSTRAP_REPLICATES})
            for replicate, value in enumerate(pooled_r2, start=1):
                long_rows.append({'replicate': replicate, 'outcome': outcome, 'model_id': model_id, 'pooled_person_relative_r2': float(value)})
        for comparison, left_model in [('P1_minus_B0', MODELS[0]), ('P1_minus_B1', MODELS[1])]:
            delta = replicate_values[outcome, MODELS[2]] - replicate_values[outcome, left_model]
            point_lookup = point_metrics_frame[point_metrics_frame['outcome'] == outcome].set_index('model_id')
            point = float(point_lookup.loc[MODELS[2], 'pooled_person_relative_r2'] - point_lookup.loc[left_model, 'pooled_person_relative_r2'])
            lower, upper = np.quantile(delta, [0.025, 0.975])
            interval_rows.append({'outcome': outcome, 'model_id': '', 'comparison': comparison, 'metric': 'delta_pooled_person_relative_r2', 'point_estimate': point, 'bootstrap_mean': float(delta.mean()), 'ci_lower_95': float(lower), 'ci_upper_95': float(upper), 'replicates': BOOTSTRAP_REPLICATES})
    return (draws, counts, pd.DataFrame(long_rows), pd.DataFrame(interval_rows))

def prespecified_training_split_rows(metadata: pd.DataFrame):
    fold = metadata['within_subject_block_fold'].to_numpy(dtype=int)
    codes = metadata['subject_code'].to_numpy(dtype=int)
    subject_count = int(codes.max()) + 1
    splits = []
    for outer_fold in FOLDS:
        outer_train = np.flatnonzero(fold != outer_fold)
        outer_eval = np.flatnonzero(fold == outer_fold)
        splits.append({'split_id': f'outer-{outer_fold}', 'outer_fold': int(outer_fold), 'inner_validation_fold': 0, 'stage': 'outer_final', 'train_indices': outer_train, 'train_codes': codes[outer_train], 'eval_subject_codes': np.unique(codes[outer_eval]), 'subject_count': subject_count})
        for inner_validation_fold in FOLDS:
            if inner_validation_fold == outer_fold:
                continue
            inner_train = np.flatnonzero((fold != outer_fold) & (fold != inner_validation_fold))
            inner_eval = np.flatnonzero(fold == inner_validation_fold)
            splits.append({'split_id': f'outer-{outer_fold}_inner-{inner_validation_fold}', 'outer_fold': int(outer_fold), 'inner_validation_fold': int(inner_validation_fold), 'stage': 'inner_selection', 'train_indices': inner_train, 'train_codes': codes[inner_train], 'eval_subject_codes': np.unique(codes[inner_eval]), 'subject_count': subject_count})
    if len(splits) != 25:
        raise RuntimeError('prespecified training-split count changed.')
    return splits

def permutation_feasibility_failures(metadata: pd.DataFrame, candidate_map: np.ndarray, split_rows=None) -> list[dict[str, Any]]:
    mapping = np.asarray(candidate_map, dtype=np.int32)
    if mapping.shape != (len(metadata),):
        raise RuntimeError('A permutation candidate has the wrong shape.')
    if sorted(mapping.tolist()) != list(range(len(metadata))):
        raise RuntimeError('A permutation candidate is not bijective.')
    if np.any(mapping == np.arange(len(metadata), dtype=np.int32)):
        raise RuntimeError('A permutation candidate contains a fixed row.')
    if split_rows is None:
        split_rows = prespecified_training_split_rows(metadata)
    outcomes = metadata[OUTCOMES].to_numpy(dtype=float)
    permuted = outcomes[mapping]
    subject_ids = metadata[['subject_code', 'subject_id']].drop_duplicates().sort_values('subject_code')['subject_id'].astype(str).tolist()
    failures: list[dict[str, Any]] = []
    for split in split_rows:
        train_indices = np.asarray(split['train_indices'], dtype=int)
        train_codes = np.asarray(split['train_codes'], dtype=int)
        subject_count = int(split['subject_count'])
        counts = np.bincount(train_codes, minlength=subject_count).astype(float)
        train_values = permuted[train_indices]
        sums = np.column_stack([np.bincount(train_codes, weights=train_values[:, outcome_index], minlength=subject_count) for outcome_index in range(len(OUTCOMES))])
        sums_squared = np.column_stack([np.bincount(train_codes, weights=train_values[:, outcome_index] ** 2, minlength=subject_count) for outcome_index in range(len(OUTCOMES))])
        with np.errstate(divide='ignore', invalid='ignore'):
            centered_ss = sums_squared - sums * sums / counts[:, None]
            variance = centered_ss / (counts[:, None] - 1.0)
        variance = np.maximum(variance, 0.0)
        sd = np.sqrt(variance)
        for code in np.asarray(split['eval_subject_codes'], dtype=int):
            if counts[code] < 2:
                for outcome in OUTCOMES:
                    failures.append({'split_id': split['split_id'], 'stage': split['stage'], 'outer_fold': split['outer_fold'], 'inner_validation_fold': split['inner_validation_fold'], 'subject_code': int(code), 'subject_id': subject_ids[int(code)], 'outcome': outcome, 'training_rows': int(counts[code]), 'training_sd': np.nan, 'failure_reason': 'fewer_than_two_training_rows'})
                continue
            for outcome_index, outcome in enumerate(OUTCOMES):
                if not np.isfinite(sd[code, outcome_index]) or sd[code, outcome_index] <= SD_FLOOR:
                    failures.append({'split_id': split['split_id'], 'stage': split['stage'], 'outer_fold': split['outer_fold'], 'inner_validation_fold': split['inner_validation_fold'], 'subject_code': int(code), 'subject_id': subject_ids[int(code)], 'outcome': outcome, 'training_rows': int(counts[code]), 'training_sd': float(sd[code, outcome_index]), 'failure_reason': 'nonfinite_or_zero_training_sd'})
    return failures

def make_permutation_plan(metadata: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    identity = np.arange(len(metadata), dtype=np.int32)
    group_specs = []
    grouped = metadata.sort_values(['subject_id', 'context', 'subject_acquisition_order_1based']).groupby(['subject_id', 'context'], sort=True)
    for (subject_id, context), group in grouped:
        indices = group['primary_row_0based'].to_numpy(dtype=np.int32)
        size = len(indices)
        allowed = np.asarray([offset for offset in range(1, size) if min(offset, size - offset) >= 3], dtype=np.int16)
        if len(allowed) == 0:
            raise RuntimeError(f'No allowed permutation offset: {subject_id}, {context}, n={size}.')
        group_specs.append((str(subject_id), str(context), indices, allowed))
    split_rows = prespecified_training_split_rows(metadata)
    accepted_maps = []
    offset_rows = []
    rejection_rows = []
    candidate_draw = 0
    maximum_candidate_draws = 100000
    while len(accepted_maps) < PERMUTATION_REPLICATES:
        candidate_draw += 1
        if candidate_draw > maximum_candidate_draws:
            raise RuntimeError('Permutation rejection sampling exceeded 100,000 proposals.')
        candidate_map = identity.copy()
        candidate_offsets = []
        for subject_id, context, indices, allowed in group_specs:
            offset = int(rng.choice(allowed))
            candidate_map[indices] = np.roll(indices, offset)
            candidate_offsets.append({'candidate_draw': int(candidate_draw), 'subject_id': subject_id, 'context': context, 'group_rows': int(len(indices)), 'offset': offset, 'circular_distance': int(min(offset, len(indices) - offset))})
        failures = permutation_feasibility_failures(metadata, candidate_map, split_rows=split_rows)
        if failures:
            for failure in failures:
                rejection_rows.append({'candidate_draw': int(candidate_draw), 'accepted': False, **failure})
            continue
        replicate = len(accepted_maps) + 1
        accepted_maps.append(candidate_map)
        for row in candidate_offsets:
            offset_rows.append({'replicate': int(replicate), 'accepted': True, **row})
    row_map = np.vstack(accepted_maps).astype(np.int32, copy=False)
    if row_map.shape != (PERMUTATION_REPLICATES, len(metadata)):
        raise RuntimeError('Accepted permutation-plan shape changed.')
    for replicate in range(PERMUTATION_REPLICATES):
        failures = permutation_feasibility_failures(metadata, row_map[replicate], split_rows=split_rows)
        if failures:
            raise RuntimeError(f'Accepted permutation {replicate + 1} failed the feasibility audit.')
    rejection_columns = ['candidate_draw', 'accepted', 'split_id', 'stage', 'outer_fold', 'inner_validation_fold', 'subject_code', 'subject_id', 'outcome', 'training_rows', 'training_sd', 'failure_reason']
    rejections = pd.DataFrame(rejection_rows, columns=rejection_columns)
    return (row_map, pd.DataFrame(offset_rows), rejections)

def permutation_plan_checkpoint(inference_cache_root: Path, metadata: pd.DataFrame, base_signature: str, recompute: bool):
    final = inference_cache_root / 'permutation_plan'
    summary_name = 'plan_summary.json'
    target_hash = hashlib.sha256(np.ascontiguousarray(metadata[OUTCOMES].to_numpy(dtype=float)).tobytes()).hexdigest()
    input_signature = canonical_sha256({'base': base_signature, 'metadata_ids': metadata['raw_behavior_row_id'].astype(str).tolist(), 'target_values_sha256': target_hash, 'seed': SEED, 'replicates': PERMUTATION_REPLICATES, 'minimum_circular_distance': 3, 'feasibility': 'all evaluated participants have at least two training rows and sample target SD > 1e-8 for both outcomes in every prespecified outer and inner training split', 'sampling': 'uniform complete proposals with rejection until 1000 accepted'})
    if not recompute and checkpoint_ok(final, summary_name, input_signature):
        return final
    temporary = inference_cache_root / '.permutation_plan.tmp'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        row_map, offsets, rejections = make_permutation_plan(metadata)
        np.save(temporary / 'row_map_int32.npy', row_map, allow_pickle=False)
        offsets.to_csv(temporary / 'offsets.csv', index=False)
        rejections.to_csv(temporary / 'rejections.csv', index=False)
        candidate_draws = int(offsets['candidate_draw'].max())
        rejected_candidates = int(rejections['candidate_draw'].nunique()) if len(rejections) else 0
        summary = {'generated_at': now_iso(), 'complete': True, 'phase': 'primary_inference', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': input_signature, 'replicates': PERMUTATION_REPLICATES, 'candidate_draws': candidate_draws, 'rejected_candidate_draws': rejected_candidates, 'acceptance_fraction': float(PERMUTATION_REPLICATES / candidate_draws), 'primary_rows': int(len(metadata)), 'groups': int(offsets[['subject_id', 'context']].drop_duplicates().shape[0]), 'minimum_circular_distance': int(offsets['circular_distance'].min()), 'joint_feasibility_outcomes': OUTCOMES, 'all_accepted_maps_feasible': True, 'stored_files': stored_manifest(temporary, {summary_name})}
        atomic_json(temporary / summary_name, summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(temporary, final)
        return final
    except Exception:
        if temporary.exists():
            (temporary / 'FAILED.txt').write_text(f'Failed at {now_iso()}.', encoding='utf-8')
        raise

def permutation_outer_checkpoint(outcome: str, outer_fold: int, metadata: pd.DataFrame, y_permuted: np.ndarray, primary_models_cache_root: Path, svd_root: Path, permutation_root: Path, base_signature: str, recompute: bool) -> Path:
    final = permutation_root / outcome / f'outer-{outer_fold}'
    summary_name = 'permutation_outer_summary.json'
    input_signature = canonical_sha256({'base': base_signature, 'outcome': outcome, 'outer_fold': outer_fold, 'permuted_target_sha256': hashlib.sha256(np.ascontiguousarray(y_permuted).tobytes()).hexdigest()})
    if not recompute and checkpoint_ok(final, summary_name, input_signature):
        return final
    temporary = permutation_root / outcome / f'.outer-{outer_fold}.tmp'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        results = fit_outer_batch(metadata, y_permuted, outer_fold, primary_models_cache_root, svd_root, return_predictions=False)
        sse = np.vstack([results[model]['sse'] for model in MODELS])
        component = np.vstack([results[model]['selected_component'] for model in MODELS]).astype(np.int16)
        alpha = np.vstack([results[model]['selected_alpha'] for model in MODELS]).astype(np.float64)
        inner_mse = np.vstack([results[model]['selected_inner_mse'] for model in MODELS]).astype(np.float64)
        np.save(temporary / 'sse_float64.npy', sse, allow_pickle=False)
        np.save(temporary / 'selected_component_int16.npy', component, allow_pickle=False)
        np.save(temporary / 'selected_alpha_float64.npy', alpha, allow_pickle=False)
        np.save(temporary / 'selected_inner_mse_float64.npy', inner_mse, allow_pickle=False)
        summary = {'generated_at': now_iso(), 'complete': True, 'phase': 'primary_inference', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': input_signature, 'outcome': outcome, 'outer_fold': outer_fold, 'models': MODELS, 'replicates': PERMUTATION_REPLICATES, 'stored_files': stored_manifest(temporary, {summary_name})}
        atomic_json(temporary / summary_name, summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(temporary, final)
        return final
    except Exception:
        if temporary.exists():
            (temporary / 'FAILED.txt').write_text(f'Failed at {now_iso()}.', encoding='utf-8')
        raise

def participant_sst(metadata: pd.DataFrame, outcome: str) -> float:
    y = metadata[outcome].to_numpy(dtype=float)
    means = metadata.groupby('subject_id')[outcome].transform('mean').to_numpy(dtype=float)
    return float(np.sum((y - means) ** 2))

def collect_permutation_results(metadata: pd.DataFrame, point_contrasts: pd.DataFrame, permutation_root: Path):
    metric_rows = []
    selection_rows = []
    p_rows = []
    raw_p_primary: dict[str, float] = {}
    for outcome in OUTCOMES:
        total_sse = np.zeros((len(MODELS), PERMUTATION_REPLICATES), dtype=np.float64)
        for outer in FOLDS:
            directory = permutation_root / outcome / f'outer-{outer}'
            sse = np.load(directory / 'sse_float64.npy', allow_pickle=False)
            components = np.load(directory / 'selected_component_int16.npy', allow_pickle=False)
            alphas = np.load(directory / 'selected_alpha_float64.npy', allow_pickle=False)
            inner_mse = np.load(directory / 'selected_inner_mse_float64.npy', allow_pickle=False)
            if sse.shape != (len(MODELS), PERMUTATION_REPLICATES):
                raise RuntimeError('Permutation SSE checkpoint shape changed.')
            total_sse += sse
            for model_index, model_id in enumerate(MODELS):
                for replicate in range(PERMUTATION_REPLICATES):
                    selection_rows.append({'replicate': replicate + 1, 'outcome': outcome, 'outer_fold': outer, 'model_id': model_id, 'pca_components': np.nan if int(components[model_index, replicate]) < 0 else int(components[model_index, replicate]), 'ridge_alpha': float(alphas[model_index, replicate]), 'selected_inner_mse': float(inner_mse[model_index, replicate])})
        sst = participant_sst(metadata, outcome)
        r2 = 1.0 - total_sse / sst
        delta_b0 = r2[2] - r2[0]
        delta_b1 = r2[2] - r2[1]
        for replicate in range(PERMUTATION_REPLICATES):
            metric_rows.append({'replicate': replicate + 1, 'outcome': outcome, 'B0_pooled_person_relative_r2': float(r2[0, replicate]), 'B1_pooled_person_relative_r2': float(r2[1, replicate]), 'P1_pooled_person_relative_r2': float(r2[2, replicate]), 'delta_P1_minus_B0': float(delta_b0[replicate]), 'delta_P1_minus_B1': float(delta_b1[replicate])})
        for comparison, values in [('P1_minus_B0', delta_b0), ('P1_minus_B1', delta_b1)]:
            observed = float(point_contrasts[(point_contrasts['outcome'] == outcome) & (point_contrasts['comparison'] == comparison)]['delta_pooled_person_relative_r2'].iloc[0])
            exceedances = int(np.sum(values >= observed))
            p_raw = float((1 + exceedances) / (PERMUTATION_REPLICATES + 1))
            if comparison == 'P1_minus_B0':
                raw_p_primary[outcome] = p_raw
            p_rows.append({'outcome': outcome, 'comparison': comparison, 'role': 'primary_gate' if comparison == 'P1_minus_B0' else 'ancillary_not_gate', 'observed_delta_r2': observed, 'null_mean': float(values.mean()), 'null_sd': float(values.std(ddof=1)), 'null_q025': float(np.quantile(values, 0.025)), 'null_q975': float(np.quantile(values, 0.975)), 'exceedances_greater_or_equal': exceedances, 'permutations': PERMUTATION_REPLICATES, 'p_raw_one_sided': p_raw, 'p_holm_primary': np.nan})
    adjusted = holm_adjust(raw_p_primary)
    for row in p_rows:
        if row['comparison'] == 'P1_minus_B0':
            row['p_holm_primary'] = adjusted[row['outcome']]
    return (pd.DataFrame(metric_rows), pd.DataFrame(selection_rows), pd.DataFrame(p_rows))

def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda item: item[1])
    count = len(items)
    adjusted_sorted = []
    running = 0.0
    for position, (name, value) in enumerate(items):
        adjusted_value = min(1.0, (count - position) * float(value))
        running = max(running, adjusted_value)
        adjusted_sorted.append((name, running))
    return dict(adjusted_sorted)

def support_decisions(point_metrics_frame: pd.DataFrame, bootstrap_intervals: pd.DataFrame, permutation_p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        lookup = point_metrics_frame[point_metrics_frame['outcome'] == outcome].set_index('model_id')
        delta_b0 = bootstrap_intervals[(bootstrap_intervals['outcome'] == outcome) & (bootstrap_intervals['comparison'] == 'P1_minus_B0')].iloc[0]
        delta_b1 = bootstrap_intervals[(bootstrap_intervals['outcome'] == outcome) & (bootstrap_intervals['comparison'] == 'P1_minus_B1')].iloc[0]
        permutation = permutation_p[(permutation_p['outcome'] == outcome) & (permutation_p['comparison'] == 'P1_minus_B0')].iloc[0]
        gate_b0 = float(delta_b0['ci_lower_95']) > 0
        gate_b1 = float(delta_b1['ci_lower_95']) > 0
        gate_permutation = float(permutation['p_holm_primary']) < 0.05
        rows.append({'outcome': outcome, 'B0_pooled_person_relative_r2': float(lookup.loc[MODELS[0], 'pooled_person_relative_r2']), 'B1_pooled_person_relative_r2': float(lookup.loc[MODELS[1], 'pooled_person_relative_r2']), 'P1_pooled_person_relative_r2': float(lookup.loc[MODELS[2], 'pooled_person_relative_r2']), 'delta_P1_minus_B0': float(delta_b0['point_estimate']), 'delta_P1_minus_B0_ci_lower': float(delta_b0['ci_lower_95']), 'delta_P1_minus_B0_ci_upper': float(delta_b0['ci_upper_95']), 'delta_P1_minus_B1': float(delta_b1['point_estimate']), 'delta_P1_minus_B1_ci_lower': float(delta_b1['ci_lower_95']), 'delta_P1_minus_B1_ci_upper': float(delta_b1['ci_upper_95']), 'primary_permutation_p_raw': float(permutation['p_raw_one_sided']), 'primary_permutation_p_holm': float(permutation['p_holm_primary']), 'gate_bootstrap_P1_minus_B0': bool(gate_b0), 'gate_bootstrap_P1_minus_B1': bool(gate_b1), 'gate_permutation_holm': bool(gate_permutation), 'primary_signal_specific_support': bool(gate_b0 and gate_b1 and gate_permutation)})
    return pd.DataFrame(rows)

def collect_cache_inventory(root: Path) -> pd.DataFrame:
    return pd.DataFrame([{'stored_path': str(path), 'relative_path': str(path.relative_to(root)), 'size_bytes': int(path.stat().st_size), 'sha256': sha256_file(path)} for path in sorted(root.rglob('*')) if path.is_file()])

def write_report(path: Path, summary: dict[str, Any], point_metrics_frame: pd.DataFrame, bootstrap_intervals: pd.DataFrame, permutation_p: pd.DataFrame, decisions: pd.DataFrame) -> None:
    lines = ['# Phase primary_inference — prespecified primary inference', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Participant bootstrap replicates: **5,000**', '- Circular-shift permutations with complete inner retuning: **1,000**', '- Holm correction: **valence and liking primary P1-minus-B0 tests**', '- Validation/confirmation opened: **no**', '', '## Technical permutation correction', '', f"- Complete circular-shift proposals drawn: **{summary['counts']['permutation_candidate_draws']}**", f"- Infeasible proposals rejected: **{summary['counts']['permutation_rejected_candidate_draws']}**", '- Accepted jointly feasible maps: **1,000**', '', 'A proposal was rejected only when the prespecified participant-specific target standardization was undefined in at least one outer or inner training split for valence or liking. The accepted maps are rejection samples from the original uniform circular-shift proposal conditional on a defined test statistic. Observed fits and support gates were not changed.', '', '## Observed primary metrics', '', '| Outcome | Model | Pooled person-relative R² | Median participant R² | Pooled raw MAE |', '|---|---|---:|---:|---:|']
    for _, row in point_metrics_frame.sort_values(['outcome', 'model_id']).iterrows():
        lines.append(f"| `{row['outcome']}` | `{row['model_id']}` | {row['pooled_person_relative_r2']:.6f} | {row['participant_median_r2']:.6f} | {row['pooled_raw_mae']:.6f} |")
    lines += ['', '## prespecified support comparisons', '', '| Outcome | Delta P1-B0 [95% CI] | Delta P1-B1 [95% CI] | Permutation p | Holm p | Support gate |', '|---|---:|---:|---:|---:|---|']
    for _, row in decisions.iterrows():
        lines.append(f"| `{row['outcome']}` | {row['delta_P1_minus_B0']:.6f} [{row['delta_P1_minus_B0_ci_lower']:.6f}, {row['delta_P1_minus_B0_ci_upper']:.6f}] | {row['delta_P1_minus_B1']:.6f} [{row['delta_P1_minus_B1_ci_lower']:.6f}, {row['delta_P1_minus_B1_ci_upper']:.6f}] | {row['primary_permutation_p_raw']:.6f} | {row['primary_permutation_p_holm']:.6f} | {bool(row['primary_signal_specific_support'])} |")
    supported = decisions[decisions['primary_signal_specific_support']]
    lines += ['', '## Interpretation boundary', '']
    if len(supported):
        lines.append('At least one outcome met the complete prespecified within-person signal-specific support gate. This is evidence of incremental prediction for this representation and design; it is not a universal biosignal claim and does not make EEG ground truth for experience.')
    else:
        lines.append('Neither outcome met the complete prespecified signal-specific support gate. This means the predeclared all-three full-signal representation did not demonstrate incremental within-person prediction beyond both baselines under the prespecified inference. It does not show that EEG lacks physiological implementation, and it does not invalidate the report-only subjective geometry.')
    lines += ['', 'All prespecified secondary and diagnostic analyses remain mandatory and must be reported independently of this primary result.', '']
    path.write_text(''.join(lines), encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, primary_models_summary, primary_models_contract, primary_models_policy, predictions, selected, inner_scores, primary_models_cache_inventory = load_inputs(project)
    primary_models_cache_root = project / CACHE_REL
    inference_cache_root = project / INFERENCE_CACHE_REL
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'cache_rehash': qc / 'primary_inference_primary_models_cache_rehash.csv', 'observed_reproduction': qc / 'primary_inference_observed_engine_reproduction.csv', 'point_metrics': tables / 'primary_inference_primary_point_metrics.csv', 'subject_metrics': tables / 'primary_inference_participant_metrics.csv', 'point_contrasts': tables / 'primary_inference_primary_point_contrasts.csv', 'bootstrap_replicates': tables / 'primary_inference_bootstrap_primary_r2_replicates.csv', 'bootstrap_intervals': tables / 'primary_inference_bootstrap_intervals.csv', 'permutation_offsets': tables / 'primary_inference_permutation_offsets.csv', 'permutation_rejections': tables / 'primary_inference_permutation_rejections.csv', 'permutation_metrics': tables / 'primary_inference_permutation_metrics.csv', 'permutation_selection': tables / 'primary_inference_permutation_selected_hyperparameters.csv', 'permutation_p': tables / 'primary_inference_permutation_pvalues.csv', 'support_decisions': tables / 'primary_inference_primary_support_decisions.csv', 'cache_inventory': tables / 'primary_inference_inference_cache_inventory.csv', 'summary': qc / 'primary_inference_primary_inference_summary.json', 'contract': config / 'primary_inference_primary_inference_contract.json', 'next_policy': config / 'S050_primary_inference_secondary.json', 'report': reports / 'primary_inference_primary_inference_report.md', 'log': logs / 'primary_inference_primary_inference.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Phase outputs already exist. Use --force after review:\n  - ' + '\n  - '.join(map(str, existing)))
    with outputs['log'].open('w', encoding='utf-8') as log:

        def note(message: str) -> None:
            line = f'[{now_iso()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Rehashing every primary_models cache file before inference.')
        cache_rehash = rehash_cache(primary_models_cache_inventory)
        cache_rehash.to_csv(outputs['cache_rehash'], index=False)
        metadata = load_primary_metadata(primary_models_cache_root, predictions)
        base_signature = canonical_sha256({'primary_models_contract_sha256': sha256_file(paths['contract']), 'primary_models_predictions_sha256': sha256_file(paths['predictions']), 'primary_models_selected_sha256': sha256_file(paths['selected']), 'primary_models_cache_inventory_sha256': sha256_file(paths['cache_inventory']), 'inference_policy_sha256': POLICY_SHA})
        note('Building or verifying 25 outcome-free SVD transform checkpoints.')
        svd_root = build_svd_cache(primary_models_cache_root, inference_cache_root, base_signature, args.recompute_cache)
        note('Reproducing all 30 observed outer fits before permutation inference.')
        reproduction = reproduce_observed_engine(metadata, predictions, selected, primary_models_cache_root, svd_root)
        reproduction.to_csv(outputs['observed_reproduction'], index=False)
        note('Computing observed prespecified metrics and 5,000 participant bootstraps.')
        point, subject, contrasts = point_metrics(predictions)
        point.to_csv(outputs['point_metrics'], index=False)
        subject.to_csv(outputs['subject_metrics'], index=False)
        contrasts.to_csv(outputs['point_contrasts'], index=False)
        draws, counts, bootstrap_replicates, bootstrap_intervals = bootstrap_inference(subject, point)
        bootstrap_replicates.to_csv(outputs['bootstrap_replicates'], index=False)
        bootstrap_intervals.to_csv(outputs['bootstrap_intervals'], index=False)
        bootstrap_dir = inference_cache_root / 'bootstrap_plan'
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        np.save(bootstrap_dir / 'draws_int16.npy', draws, allow_pickle=False)
        np.save(bootstrap_dir / 'counts_int16.npy', counts, allow_pickle=False)
        note('Creating or verifying the shared 1,000-replicate circular-shift plan.')
        plan_root = permutation_plan_checkpoint(inference_cache_root, metadata, base_signature, args.recompute_cache)
        row_map = np.load(plan_root / 'row_map_int32.npy', mmap_mode='r', allow_pickle=False)
        offsets = pd.read_csv(plan_root / 'offsets.csv', low_memory=False)
        rejections = pd.read_csv(plan_root / 'rejections.csv', low_memory=False)
        offsets.to_csv(outputs['permutation_offsets'], index=False)
        rejections.to_csv(outputs['permutation_rejections'], index=False)
        permutation_root = inference_cache_root / 'permutation_outer'
        permutation_root.mkdir(parents=True, exist_ok=True)
        for outcome in OUTCOMES:
            y = metadata[outcome].to_numpy(dtype=float)
            y_permuted = y[np.asarray(row_map, dtype=int).T]
            observed_sst = participant_sst(metadata, outcome)
            for replicate in [0, PERMUTATION_REPLICATES // 2, PERMUTATION_REPLICATES - 1]:
                check = metadata.copy()
                check[outcome] = y_permuted[:, replicate]
                if not math.isclose(participant_sst(check, outcome), observed_sst, rel_tol=0, abs_tol=1e-09):
                    raise RuntimeError('Permutation plan did not preserve participant SST.')
            for outer in FOLDS:
                note(f'Permutation inference: {outcome}, outer fold {outer}/5.')
                permutation_outer_checkpoint(outcome, outer, metadata, y_permuted, primary_models_cache_root, svd_root, permutation_root, base_signature, args.recompute_cache)
        note('Aggregating permutation nulls, Holm p-values, and the prespecified support gate.')
        permutation_metrics, permutation_selection, permutation_p = collect_permutation_results(metadata, contrasts, permutation_root)
        permutation_metrics.to_csv(outputs['permutation_metrics'], index=False)
        permutation_selection.to_csv(outputs['permutation_selection'], index=False)
        permutation_p.to_csv(outputs['permutation_p'], index=False)
        decisions = support_decisions(point, bootstrap_intervals, permutation_p)
        decisions.to_csv(outputs['support_decisions'], index=False)
        inference_cache_inventory = collect_cache_inventory(inference_cache_root)
        inference_cache_inventory.to_csv(outputs['cache_inventory'], index=False)
        any_support = bool(decisions['primary_signal_specific_support'].any())
        decision = {'observed_engine_reproduced_exactly': True, 'participant_bootstrap_complete': True, 'circular_shift_permutation_complete': True, 'permutation_feasibility_condition_applied': True, 'all_accepted_permutations_defined_in_every_prespecified_split': True, 'complete_inner_retuning_every_permutation': True, 'holm_correction_complete': True, 'scientific_signal_support_decided': True, 'any_primary_outcome_met_signal_specific_support_gate': any_support, 'secondary_models_fitted': False, 'secondary_models_remain_mandatory': True, 'validation_opened': False, 'confirmation_opened': False}
        summary = {'generated_at': now_iso(), 'phase': 'primary_inference', 'inference_feasibility_rule': {'proposal_distribution': 'uniform circular-shift proposals conditioned on defined target standardization in every prespecified split for both outcomes', 'observed_models_or_predictions_changed': False, 'support_gate_changed': False}, 'status': 'conditional_feasible_primary_bootstrap_permutation_inference_complete', 'inference_policy_sha256': POLICY_SHA, 'counts': {'primary_models_cache_files_rehashed': int(len(cache_rehash)), 'primary_models_cache_bytes_rehashed': int(cache_rehash['size_bytes'].sum()), 'svd_split_checkpoints': 25, 'svd_designs_per_split': len(DESIGNS), 'observed_outer_fits_reproduced': int(len(reproduction)), 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'permutation_replicates': PERMUTATION_REPLICATES, 'permutation_group_offset_rows': int(len(offsets)), 'permutation_candidate_draws': int(offsets['candidate_draw'].max()), 'permutation_rejected_candidate_draws': int(rejections['candidate_draw'].nunique()) if len(rejections) else 0, 'permutation_rejection_failure_rows': int(len(rejections)), 'permutation_outer_checkpoints': len(OUTCOMES) * len(FOLDS), 'permutation_selected_hyperparameter_rows': int(len(permutation_selection)), 'inference_cache_files': int(len(inference_cache_inventory)), 'inference_cache_bytes': int(inference_cache_inventory['size_bytes'].sum())}, 'primary_support_decisions': decisions.to_dict(orient='records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        contract = {**summary, 'inference_policy': INFERENCE_POLICY, 'primary_models_cache_root': str(primary_models_cache_root), 'inference_cache_root': str(inference_cache_root), 'large_cache_embedded_in_release': False, 'input_hashes': {name: sha256_file(path) for name, path in paths.items()}, 'output_hashes': {name: sha256_file(path) for name, path in outputs.items() if name not in {'contract', 'next_policy', 'report', 'log'} and path.is_file()}, 'feature_store_modified': False, 'primary_models_cache_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
        outputs['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        next_policy = {'generated_at': now_iso(), 'phase': 'primary_inference', 'next_phase': 'secondary_models', 'next_phase_permitted': True, 'next_phase_role': 'execute every prespecified secondary and diagnostic discovery model regardless of the primary support result', 'mandatory_models': ['S1_seen_single_epoch_full_signal', 'S2_seen_pair_epoch_full_signal', 'P2_person_specific_compact_all3', 'U1_unseen_person_universal_all3', 'D1_subject_identity_probe_all3', 'D2_context_probe', 'E1_liking_given_valence'], 'primary_results_may_not_select_or_cancel_secondary_models': True, 'primary_permutation_distribution': 'original uniform complete circular-shift proposal conditioned on defined participant target standardization in every prespecified split', 'claim_guardrails': ['EEG is not ground truth for latent experience', 'primary support failure is not called EEG failure', 'stimulus and rating branches remain separately reported', 'identity is separability, not emotion decoding', 'unseen-person failure is nonseparability or misalignment, not absent physiology'], 'validation_opened': False, 'confirmation_opened': False}
        outputs['next_policy'].write_text(json.dumps(next_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(outputs['report'], summary, point, bootstrap_intervals, permutation_p, decisions)
        note('primary_inference completed. The primary inference and support decision are now prespecified.')
    print('' + '=' * 100)
    print('PHASE primary_inference PRIMARY INFERENCE COMPLETED')
    print('=' * 100)
    print(f'Bootstrap replicates:            {BOOTSTRAP_REPLICATES}')
    print(f'Permutation replicates:          {PERMUTATION_REPLICATES}')
    print(f"Permutation candidate draws:     {int(offsets['candidate_draw'].max())}")
    print(f"Rejected candidate draws:       {(int(rejections['candidate_draw'].nunique()) if len(rejections) else 0)}")
    print(f'Observed fits reproduced:        {len(reproduction)}')
    print(f'Any primary outcome supported:   {any_support}')
    print(f'Inference cache:                 {inference_cache_root}')
    print(f"Report:                          {outputs['report']}")
    print(f'Runtime:                         {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    raise SystemExit(main())