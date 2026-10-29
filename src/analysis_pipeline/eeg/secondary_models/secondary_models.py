# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase secondary_models: execute all prespecified secondary and diagnostic models.

Mandatory prespecified families:
- S1 single-epoch full-signal models;
- S2 pair-epoch full-signal models;
- P2 person-specific compact all-three models;
- U1 unseen-person universal all-three models;
- D1 subject-identity probes;
- D2 context probes;
- E1 liking-given-valence conditional model.

This phase is discovery-only. It does not open validation or confirmation and
it does not use outer-test results to select or cancel any family or epoch.
Large outcome-free matrices and split transforms are checkpointed outside the"""
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
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score, top_k_accuracy_score
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
CACHE_REL = Path('outputs/large/secondary_models_secondary_models_cache_v1')
primary_models_CACHE_REL = Path('outputs/large/primary_models_primary_nested_cache_v1')
REL = {'primary_analysis_reference_summary': Path('outputs/qc/outcome_model_integrity_outcome_model_integrity_summary.json'), 'primary_analysis_reference_contract': Path('config/outcome_model_integrity_outcome_model_integrity_contract.json'), 'feature_integrity': Path('outputs/tables/outcome_model_integrity_feature_store_file_integrity.csv'), 'feature_rows': Path('outputs/tables/outcome_model_integrity_unified_feature_row_manifest.csv'), 'trial_outcomes': Path('outputs/tables/outcome_model_integrity_discovery_trial_outcome_manifest.csv'), 'analysis_rows': Path('outputs/tables/outcome_model_integrity_analysis_set_row_manifest.csv'), 'model_spec': Path('config/outcome_model_integrity_prespecified_model_specification.csv'), 'target_policy': Path('config/outcome_model_integrity_prespecified_target_policy.json'), 'nested_policy': Path('config/outcome_model_integrity_prespecified_nested_model_policy.json'), 'primary_models_summary': Path('outputs/qc/primary_models_primary_nested_model_summary.json'), 'primary_models_contract': Path('config/primary_models_primary_nested_model_contract.json'), 'primary_models_cache_inventory': Path('outputs/tables/primary_models_cache_inventory.csv'), 'primary_inference_summary': Path('outputs/qc/primary_inference_primary_inference_summary.json'), 'primary_inference_contract': Path('config/primary_inference_primary_inference_contract.json'), 'primary_inference_support': Path('outputs/tables/primary_inference_primary_support_decisions.csv'), 'primary_inference_next': Path('analysis_specifications/S050_primary_inference_secondary.json')}
SEED = 540
FOLDS = [1, 2, 3, 4, 5]
OUTCOMES = ['valence', 'liking']
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
SINGLE_SETS = {'stimulus_early': ['stimulus_early'], 'stimulus_late': ['stimulus_late'], 'rating_early': ['rating_early']}
PAIR_SETS = {'stimulus_early_vs_late': ['stimulus_early', 'stimulus_late'], 'stimulus_early_vs_rating': ['stimulus_early', 'rating_early'], 'stimulus_late_vs_rating': ['stimulus_late', 'rating_early']}
MANDATORY_MODELS = ['S1_seen_single_epoch_full_signal', 'S2_seen_pair_epoch_full_signal', 'P2_person_specific_compact_all3', 'U1_unseen_person_universal_all3', 'D1_subject_identity_probe_all3', 'D2_context_probe', 'E1_liking_given_valence']
ALPHAS = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000.0], dtype=float)
POOLED_COMPONENTS = [8, 16, 32, 64, 128]
PERSON_COMPONENTS = [2, 4, 8]
LOGISTIC_C = [0.01, 0.1, 1.0, 10.0]
SD_FLOOR = 1e-08
TIE = 1e-12
BOOTSTRAPS = 5000
UPPER = np.triu_indices(64)
PER_EPOCH_FULL = 6093
PER_EPOCH_COMPACT = 169
PER_EPOCH_QC = 65
FULL_BLOCKS = [('time_frequency_path', 'tf', 1920), ('morphology_path', 'morph', 384), ('covariance_path', 'cov', 2080), ('covariance_summary_path', 'covsum', 21), ('plv_channel_path', 'plvch', 384), ('plv_global_path', 'plvgl', 24), ('waveform_dct_path', 'dct', 1280)]
POLICY = {'phase': 'secondary_models', 'seed': SEED, 'mandatory_models': MANDATORY_MODELS, 'no_outer_test_selection': True, 'discovery_only': True, 'validation_opened': False, 'confirmation_opened': False, 'analysis_sets': {'S1': SINGLE_SETS, 'S2': PAIR_SETS, 'P2': 'all_three_epoch_families; within_person_eligible', 'U1': 'all_three_epoch_families; cross_person_eligible', 'D1': 'all_three_epoch_families; within_person_eligible', 'D2': SINGLE_SETS, 'E1': 'all_three_epoch_families; within_person_eligible'}, 'training_only_preprocessing': {'full_signal': 'drop all-training-missing; median impute; zero-variance drop; z-score; randomized PCA', 'compact_or_qc': 'drop all-training-missing; median impute; zero-variance drop; z-score', 'covariates': 'training-only continuous z-scores; subject fixed effects only in seen-person pooled regression'}, 'regression_tuning': {'ridge_alpha': ALPHAS.tolist(), 'pooled_pca_components': POOLED_COMPONENTS, 'person_pca_components': PERSON_COMPONENTS, 'metric': 'participant-equal mean squared standardized target error', 'tie_break': 'fewer PCA components then larger ridge alpha within 1e-12'}, 'classification_completion_rule': {'C_grid': LOGISTIC_C, 'inner_metric': 'participant-equal negative log likelihood', 'tie_break': 'smaller C within 1e-12', 'outer_metrics': 'accuracy, balanced accuracy, macro F1, log loss; plus top-5 identity accuracy or ROC AUC for context'}, 'matched_reference_models': {'purpose': 'descriptive within-analysis-set covariate-only references required to interpret incremental signal; not additional prespecified claim families', 'S1_S2_E1': 'same split, target transform, and covariates without signal PCA scores', 'P2': 'same person-specific split and covariates without compact signal', 'U1': 'same unseen-person split and covariates without full signal'}, 'formal_secondary_decisions': {'P2': 'median subject delta-R2 > 0 and two-sided exact sign test Holm-adjusted across valence/liking; primary gate is also required for implementation support', 'U1': 'lower 95% participant-bootstrap conventional raw-scale R2 > 0', 'S1_S2_D1_D2_E1': 'descriptive/diagnostic; no family or epoch is selected from outer-test performance'}, 'claim_guardrails': ['EEG is not ground truth for latent experience', 'primary support failure is not called EEG failure', 'stimulus and rating branches remain separately reported', 'identity is separability, not emotion decoding', 'unseen-person failure is nonseparability or misalignment, not absent physiology', 'E1 is conditional association, not deployable liking prediction']}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

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
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def csha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()

def jread(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def bval(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {'true', '1', 'yes'}:
        return True
    if s in {'false', '0', 'no'}:
        return False
    raise RuntimeError(f'Invalid boolean value: {value!r}')

def required(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(map(str, missing)))

def atom_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '', encoding='utf-8')
    os.replace(tmp, path)

def stored_manifest(root: Path, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel in exclude:
            continue
        rows.append({'relative_path': rel, 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    return rows

def checkpoint_ok(root: Path, summary_name: str, input_signature: str) -> bool:
    p = root / summary_name
    if not p.is_file():
        return False
    try:
        summary = jread(p)
    except Exception:
        return False
    if summary.get('complete') is not True:
        return False
    if summary.get('policy_sha256') != POLICY_SHA:
        return False
    if summary.get('input_signature_sha256') != input_signature:
        return False
    for row in summary.get('stored_files', []):
        f = root / row['relative_path']
        if not f.is_file() or f.stat().st_size != int(row['size_bytes']) or sha(f) != row['sha256']:
            return False
    return True

def load_inputs(project: Path) -> tuple[dict[str, Path], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {k: project / v for k, v in REL.items()}
    required(paths.values())
    primary_analysis_reference_summary = jread(paths['primary_analysis_reference_summary'])
    primary_analysis_reference_contract = jread(paths['primary_analysis_reference_contract'])
    if primary_analysis_reference_summary.get('phase') != 'outcome_model_integrity':
        raise RuntimeError('Unexpected outcome_model_integrity.1 phase')
    for key in ['feature_store_integrity_passed', 'discovery_outcome_attachment_exact', 'primary_analysis_rows_prespecified', 'nested_model_design_prespecified', 'primary_models_nested_model_fitting_permitted']:
        if not primary_analysis_reference_summary['decision'].get(key):
            raise RuntimeError(f'outcome_model_integrity.1 prerequisite failed: {key}')
    primary_analysis_reference_hash_map = {'primary_analysis_reference_summary': 'summary', 'feature_integrity': 'file_integrity', 'feature_rows': 'feature_row_manifest', 'trial_outcomes': 'trial_outcomes', 'analysis_rows': 'analysis_rows', 'model_spec': 'model_table', 'target_policy': 'target_policy', 'nested_policy': 'model_policy'}
    for key, contract_key in primary_analysis_reference_hash_map.items():
        if sha(paths[key]) != primary_analysis_reference_contract['output_hashes'][contract_key]:
            raise RuntimeError(f'prespecified outcome_model_integrity.1 file changed: {key}')
    spec = pd.read_csv(paths['model_spec'], low_memory=False)
    observed_mandatory = spec.loc[~spec['primary_comparison'].map(bval), 'model_id'].tolist()
    if observed_mandatory != MANDATORY_MODELS:
        raise RuntimeError(f'prespecified mandatory model order changed: {observed_mandatory}')
    primary_models_summary = jread(paths['primary_models_summary'])
    primary_models_contract = jread(paths['primary_models_contract'])
    if primary_models_summary.get('phase') != 'primary_models' or not primary_models_summary['decision'].get('complete_outer_prediction_coverage'):
        raise RuntimeError('primary_models prerequisite incomplete')
    if sha(paths['primary_models_summary']) != primary_models_contract['output_hashes']['summary']:
        raise RuntimeError('primary_models summary hash changed')
    if sha(paths['primary_models_cache_inventory']) != primary_models_contract['output_hashes']['cache_inventory']:
        raise RuntimeError('primary_models cache inventory hash changed')
    b1_summary = jread(paths['primary_inference_summary'])
    b1_contract = jread(paths['primary_inference_contract'])
    b1_next = jread(paths['primary_inference_next'])
    if b1_summary.get('phase') != 'primary_inference':
        raise RuntimeError('Unexpected primary_inference phase')
    decision = b1_summary.get('decision', {})
    for key in ['observed_engine_reproduced_exactly', 'participant_bootstrap_complete', 'circular_shift_permutation_complete', 'complete_inner_retuning_every_permutation', 'holm_correction_complete', 'scientific_signal_support_decided', 'secondary_models_remain_mandatory']:
        if not decision.get(key):
            raise RuntimeError(f'primary_inference prerequisite failed: {key}')
    if decision.get('validation_opened') or decision.get('confirmation_opened'):
        raise RuntimeError('Validation or confirmation was unexpectedly opened')
    if b1_next.get('next_phase') != 'secondary_models' or not b1_next.get('next_phase_permitted'):
        raise RuntimeError('primary_inference did not authorize secondary_models')
    if b1_next.get('mandatory_models') != MANDATORY_MODELS:
        raise RuntimeError('primary_inference mandatory model list changed')
    b1_hash_map = {'primary_inference_summary': 'summary', 'primary_inference_support': 'support_decisions'}
    for key, contract_key in b1_hash_map.items():
        if sha(paths[key]) != b1_contract['output_hashes'][contract_key]:
            raise RuntimeError(f'primary_inference output changed: {key}')
    feature_integrity = pd.read_csv(paths['feature_integrity'], low_memory=False)
    feature_rows = pd.read_csv(paths['feature_rows'], low_memory=False)
    trial_outcomes = pd.read_csv(paths['trial_outcomes'], low_memory=False)
    analysis_rows = pd.read_csv(paths['analysis_rows'], low_memory=False)
    if len(feature_integrity) != 1425 or len(feature_rows) != 3853 or len(trial_outcomes) != 1301:
        raise RuntimeError('prespecified source counts changed')
    contracts = {'primary_analysis_reference': primary_analysis_reference_contract, 'primary_models': primary_models_contract, 'primary_inference': b1_contract, 'primary_support': pd.read_csv(paths['primary_inference_support']).to_dict(orient='records')}
    return (paths, contracts, feature_integrity, feature_rows, trial_outcomes, analysis_rows)

def rehash_feature_store(fi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in fi.iterrows():
        p = Path(str(r.stored_path))
        if not p.is_file():
            raise RuntimeError(f'Missing feature file: {p}')
        observed_sha = sha(p)
        passed = p.stat().st_size == int(r.expected_size_bytes) and observed_sha == str(r.expected_sha256)
        if not passed:
            raise RuntimeError(f'Feature-store integrity failure: {p}')
        rows.append({'inventory_row_1based': i + 1, 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': observed_sha, 'passed': True})
        if (i + 1) % 200 == 0 or i + 1 == len(fi):
            print(f'[{now()}] Feature-store hash {i + 1}/{len(fi)}', flush=True)
    return pd.DataFrame(rows)

def rehash_primary_models_cache(project: Path, inventory_path: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_path, low_memory=False)
    rows = []
    for i, r in inv.iterrows():
        p = project / primary_models_CACHE_REL / str(r.relative_path)
        if not p.is_file() or p.stat().st_size != int(r.size_bytes) or sha(p) != str(r.sha256):
            raise RuntimeError(f'primary_models cache integrity failure: {p}')
        rows.append({'relative_path': r.relative_path, 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': r.sha256, 'passed': True})
    return pd.DataFrame(rows)

def flat_full(values: np.ndarray, kind: str) -> np.ndarray:
    a = np.asarray(values)
    if kind == 'cov':
        return a[:, UPPER[0], UPPER[1]]
    return a.reshape(a.shape[0], -1)

def channel_median_iqr(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float32)
    if a.ndim < 3 or a.shape[1] != 64:
        raise RuntimeError(f'Expected channel axis of 64, got {a.shape}')
    a = a.reshape(a.shape[0], 64, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(a, axis=1)
        q25 = np.nanpercentile(a, 25, axis=1)
        q75 = np.nanpercentile(a, 75, axis=1)
    return np.concatenate([med, q75 - q25], axis=1).astype(np.float32)

def build_epoch_store(cache: Path, fr: pd.DataFrame, input_signature: str, recompute: bool) -> Path:
    final = cache / 'epoch_store'
    summary_name = 'epoch_store_summary.json'
    if not recompute and checkpoint_ok(final, summary_name, input_signature):
        return final
    tmp = cache / '.epoch_store.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        manifest = fr.copy().reset_index(drop=True)
        manifest['epoch_store_row_0based'] = np.arange(len(manifest), dtype=int)
        n = len(manifest)
        full = np.lib.format.open_memmap(tmp / 'epoch_full_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_FULL))
        compact = np.lib.format.open_memmap(tmp / 'epoch_compact_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_COMPACT))
        qc = np.lib.format.open_memmap(tmp / 'epoch_qc_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_QC))
        full[:] = np.nan
        compact[:] = np.nan
        qc[:] = np.nan
        full_offset = 0
        compact_slices = {'tf': (0, 60), 'morph': (60, 72), 'plvch': (72, 84), 'dct': (84, 124), 'covsum': (124, 145), 'plvgl': (145, 169)}
        for path_col, kind, width in FULL_BLOCKS:
            for stored_path, group in manifest.groupby(path_col, sort=True):
                arr = np.load(Path(str(stored_path)), mmap_mode='r', allow_pickle=False)
                source_idx = group.feature_row_0based.astype(int).to_numpy()
                dest_idx = group.epoch_store_row_0based.astype(int).to_numpy()
                vals = np.asarray(arr[source_idx])
                flat = flat_full(vals, kind)
                if flat.shape[1] != width:
                    raise RuntimeError(f'Feature width mismatch for {kind}: {flat.shape[1]} != {width}')
                full[dest_idx, full_offset:full_offset + width] = flat.astype(np.float32)
                if kind in {'tf', 'morph', 'plvch', 'dct'}:
                    summaries = channel_median_iqr(vals)
                    start, stop = compact_slices[kind]
                    expected = stop - start
                    if summaries.shape[1] != expected:
                        raise RuntimeError(f'Compact width mismatch for {kind}: {summaries.shape}')
                    compact[dest_idx, start:stop] = summaries
                elif kind == 'covsum':
                    direct = vals.reshape(vals.shape[0], -1).astype(np.float32)
                    start, stop = compact_slices[kind]
                    if direct.shape[1] != stop - start:
                        raise RuntimeError('Covariance-summary compact width mismatch')
                    compact[dest_idx, start:stop] = direct
                elif kind == 'plvgl':
                    direct = vals.reshape(vals.shape[0], -1).astype(np.float32)
                    start, stop = compact_slices[kind]
                    if direct.shape[1] != stop - start:
                        raise RuntimeError('Global-PLV compact width mismatch')
                    compact[dest_idx, start:stop] = direct
            full_offset += width
        if full_offset != PER_EPOCH_FULL or max((stop for _, stop in compact_slices.values())) != PER_EPOCH_COMPACT:
            raise RuntimeError(f'Epoch vector widths incomplete: full={full_offset}, compact={PER_EPOCH_COMPACT}')
        for stored_path, group in manifest.groupby('time_frequency_path', sort=True):
            arr = np.load(Path(str(stored_path)), mmap_mode='r', allow_pickle=False)
            source_idx = group.feature_row_0based.astype(int).to_numpy()
            dest_idx = group.epoch_store_row_0based.astype(int).to_numpy()
            mask = np.isfinite(np.asarray(arr[source_idx])[:, :, 0, 0])
            good = group.final_good_channels.astype(int).to_numpy()
            if not np.array_equal(mask.sum(axis=1), good):
                raise RuntimeError('QC availability mask does not match final_good_channels')
            qc[dest_idx, :64] = mask.astype(np.float32)
            qc[dest_idx, 64] = good.astype(np.float32)
        full.flush()
        compact.flush()
        qc.flush()
        if np.isinf(np.asarray(full)).any() or np.isinf(np.asarray(compact)).any() or (not np.isfinite(np.asarray(qc)).all()):
            raise RuntimeError('Nonfinite epoch-store integrity failure')
        manifest.to_csv(tmp / 'epoch_manifest.csv', index=False)
        atom_json(tmp / summary_name, {'complete': True, 'phase': 'secondary_models', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': input_signature, 'rows': n, 'full_cells_per_epoch': PER_EPOCH_FULL, 'compact_cells_per_epoch': PER_EPOCH_COMPACT, 'qc_cells_per_epoch': PER_EPOCH_QC, 'stored_files': stored_manifest(tmp, {summary_name})})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        if tmp.exists():
            (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def analysis_metadata(trial_outcomes: pd.DataFrame, analysis_rows: pd.DataFrame, analysis_set: str, eligibility_col: str) -> pd.DataFrame:
    rows = analysis_rows[analysis_rows.analysis_set.eq(analysis_set)].copy()
    rows = rows[rows[eligibility_col].map(bval)].copy()
    if rows.empty:
        raise RuntimeError(f'No eligible rows for {analysis_set}/{eligibility_col}')
    keys = ['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'within_subject_block_fold', 'cross_person_fold', 'subject_acquisition_order_1based']
    outcome_cols = keys + ['raw_run', 'raw_source_path', 'acquisition_fraction', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
    meta = rows[keys].merge(trial_outcomes[outcome_cols], on=keys, how='left', validate='one_to_one')
    if meta[['valence', 'liking']].isna().any().any():
        raise RuntimeError(f'Missing outcomes for {analysis_set}')
    meta = meta.sort_values(['subject_id', 'subject_acquisition_order_1based', 'raw_behavior_row_id']).reset_index(drop=True)
    meta['analysis_row_0based'] = np.arange(len(meta), dtype=int)
    meta['analysis_set'] = analysis_set
    return meta

def build_analysis_matrix(cache: Path, dataset_id: str, meta: pd.DataFrame, families: Sequence[str], epoch_store: Path, input_signature: str, recompute: bool) -> Path:
    final = cache / 'analysis_matrices' / dataset_id
    summary_name = 'analysis_matrix_summary.json'
    signature = csha({'base': input_signature, 'dataset_id': dataset_id, 'trial_ids': meta.raw_behavior_row_id.astype(str).tolist(), 'families': list(families)})
    if not recompute and checkpoint_ok(final, summary_name, signature):
        return final
    tmp = cache / 'analysis_matrices' / f'.{dataset_id}.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        epoch_manifest = pd.read_csv(epoch_store / 'epoch_manifest.csv', low_memory=False)
        epoch_full = np.load(epoch_store / 'epoch_full_float32.npy', mmap_mode='r', allow_pickle=False)
        epoch_compact = np.load(epoch_store / 'epoch_compact_float32.npy', mmap_mode='r', allow_pickle=False)
        epoch_qc = np.load(epoch_store / 'epoch_qc_float32.npy', mmap_mode='r', allow_pickle=False)
        lookup = epoch_manifest[['raw_behavior_row_id', 'epoch_family', 'epoch_store_row_0based']].copy()
        n = len(meta)
        full = np.lib.format.open_memmap(tmp / 'signal_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_FULL * len(families)))
        compact = np.lib.format.open_memmap(tmp / 'compact_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_COMPACT * len(families)))
        qc = np.lib.format.open_memmap(tmp / 'qc_float32.npy', mode='w+', dtype=np.float32, shape=(n, PER_EPOCH_QC * len(families)))
        family_rows = []
        for family_index, family in enumerate(families):
            m = meta[['raw_behavior_row_id', 'analysis_row_0based']].merge(lookup[lookup.epoch_family.eq(family)], on='raw_behavior_row_id', how='left', validate='one_to_one')
            if m.epoch_store_row_0based.isna().any():
                raise RuntimeError(f'Missing required epoch {family} in {dataset_id}')
            src = m.epoch_store_row_0based.astype(int).to_numpy()
            dst = m.analysis_row_0based.astype(int).to_numpy()
            full[dst, family_index * PER_EPOCH_FULL:(family_index + 1) * PER_EPOCH_FULL] = epoch_full[src]
            compact[dst, family_index * PER_EPOCH_COMPACT:(family_index + 1) * PER_EPOCH_COMPACT] = epoch_compact[src]
            qc[dst, family_index * PER_EPOCH_QC:(family_index + 1) * PER_EPOCH_QC] = epoch_qc[src]
            family_rows.extend(({'analysis_row_0based': int(d), 'raw_behavior_row_id': meta.iloc[int(d)].raw_behavior_row_id, 'epoch_family': family, 'epoch_store_row_0based': int(s)} for d, s in zip(dst, src)))
        full.flush()
        compact.flush()
        qc.flush()
        meta.to_csv(tmp / 'metadata.csv', index=False)
        pd.DataFrame(family_rows).to_csv(tmp / 'family_row_map.csv', index=False)
        atom_json(tmp / summary_name, {'complete': True, 'phase': 'secondary_models', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': signature, 'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'families': list(families), 'rows': n, 'subjects': int(meta.subject_id.nunique()), 'signal_cells': int(full.shape[1]), 'compact_cells': int(compact.shape[1]), 'qc_cells': int(qc.shape[1]), 'stored_files': stored_manifest(tmp, {summary_name})})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        if tmp.exists():
            (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def cov_transform(meta: pd.DataFrame, train_idx: np.ndarray, eval_idx: np.ndarray, subject_fixed: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tr = meta.iloc[train_idx]
    ev = meta.iloc[eval_idx]
    blocks_train = []
    blocks_eval = []
    subject_levels: list[str] = []
    if subject_fixed:
        subject_levels = sorted(meta.subject_id.astype(str).unique())
        blocks_train.append(np.column_stack([tr.subject_id.eq(s).to_numpy(float) for s in subject_levels]))
        blocks_eval.append(np.column_stack([ev.subject_id.eq(s).to_numpy(float) for s in subject_levels]))
    blocks_train.append(tr.context.eq('video').to_numpy(float)[:, None])
    blocks_eval.append(ev.context.eq('video').to_numpy(float)[:, None])
    ctr = np.column_stack([tr.raw_run.to_numpy(float), tr.acquisition_fraction.to_numpy(float), tr.acquisition_fraction.to_numpy(float) ** 2])
    cev = np.column_stack([ev.raw_run.to_numpy(float), ev.acquisition_fraction.to_numpy(float), ev.acquisition_fraction.to_numpy(float) ** 2])
    mu = ctr.mean(axis=0)
    sd = ctr.std(axis=0)
    keep = sd > 0
    ztr = np.zeros_like(ctr)
    zev = np.zeros_like(cev)
    ztr[:, keep] = (ctr[:, keep] - mu[keep]) / sd[keep]
    zev[:, keep] = (cev[:, keep] - mu[keep]) / sd[keep]
    blocks_train.append(ztr)
    blocks_eval.append(zev)
    return (np.concatenate(blocks_train, axis=1).astype(np.float32), np.concatenate(blocks_eval, axis=1).astype(np.float32), {'subject_fixed_effects': subject_fixed, 'subject_levels': subject_levels, 'output_columns': int(sum((x.shape[1] for x in blocks_train))), 'zero_variance_continuous': int((~keep).sum())})

def standardized_transform(raw: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tr = np.asarray(raw[train_idx], dtype=np.float32)
    ev = np.asarray(raw[eval_idx], dtype=np.float32)
    all_missing = np.isnan(tr).all(axis=0)
    tr = tr[:, ~all_missing]
    ev = ev[:, ~all_missing]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(tr, axis=0)
    if not np.isfinite(med).all():
        raise RuntimeError('Training median is nonfinite after all-missing removal')
    tr = np.where(np.isnan(tr), med[None, :], tr)
    ev = np.where(np.isnan(ev), med[None, :], ev)
    var = tr.var(axis=0)
    keep = np.isfinite(var) & (var > 0)
    tr = tr[:, keep]
    ev = ev[:, keep]
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    if not (sd > 0).all():
        raise RuntimeError('Feature standardization variance failure')
    return (((tr - mu) / sd).astype(np.float32), ((ev - mu) / sd).astype(np.float32), {'input_columns': int(raw.shape[1]), 'all_missing_dropped': int(all_missing.sum()), 'zero_variance_dropped': int((~keep).sum()), 'retained_columns': int(keep.sum())})

def pca_transform(raw: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, components: Sequence[int], seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tr, ev, diag = standardized_transform(raw, train_idx, eval_idx)
    n_components = min(max(components), tr.shape[0] - 1, tr.shape[1])
    if n_components < min(components):
        raise RuntimeError(f'Too few PCA components: {n_components} for candidates {components}')
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=int(seed), n_oversamples=10, iterated_power=4, whiten=False)
    ztr = pca.fit_transform(tr)
    zev = pca.transform(ev)
    diag.update({'pca_components': int(n_components), 'explained_ratio_sum': float(pca.explained_variance_ratio_.sum()), 'seed': int(seed)})
    return (ztr.astype(np.float32), zev.astype(np.float32), diag)

def split_cache(root: Path, split_id: str, meta: pd.DataFrame, signal: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, subject_fixed: bool, components: Sequence[int], input_signature: str, seed: int, recompute: bool) -> Path:
    final = root / split_id
    summary_name = 'split_summary.json'
    signature = csha({'base': input_signature, 'split_id': split_id, 'train': train_idx.astype(int).tolist(), 'eval': eval_idx.astype(int).tolist(), 'subject_fixed': subject_fixed, 'components': list(components), 'seed': int(seed)})
    if not recompute and checkpoint_ok(final, summary_name, signature):
        return final
    tmp = root / f'.{split_id}.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cov_tr, cov_ev, cov_diag = cov_transform(meta, train_idx, eval_idx, subject_fixed)
        pca_tr, pca_ev, pca_diag = pca_transform(signal, train_idx, eval_idx, components, seed)
        arrays = {'train_indices.npy': train_idx.astype(np.int64), 'eval_indices.npy': eval_idx.astype(np.int64), 'cov_train.npy': cov_tr, 'cov_eval.npy': cov_ev, 'pca_train.npy': pca_tr, 'pca_eval.npy': pca_ev}
        for name, array in arrays.items():
            np.save(tmp / name, array, allow_pickle=False)
        atom_json(tmp / 'diagnostics.json', {'covariates': cov_diag, 'signal': pca_diag})
        atom_json(tmp / summary_name, {'complete': True, 'phase': 'secondary_models', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': signature, 'stored_files': stored_manifest(tmp, {summary_name})})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        if tmp.exists():
            (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def load_split(path: Path) -> dict[str, np.ndarray]:
    files = {'train': 'train_indices.npy', 'eval': 'eval_indices.npy', 'cov_train': 'cov_train.npy', 'cov_eval': 'cov_eval.npy', 'pca_train': 'pca_train.npy', 'pca_eval': 'pca_eval.npy'}
    return {k: np.load(path / v, mmap_mode='r', allow_pickle=False) for k, v in files.items()}

def seen_target(meta: pd.DataFrame, outcome: str, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, Any]:
    tr = meta.iloc[train_idx]
    ev = meta.iloc[eval_idx]
    stats = tr.groupby('subject_id')[outcome].agg(['mean', 'std'])
    bad = stats.index[stats['std'].isna() | (stats['std'] <= SD_FLOOR)].tolist()
    if bad:
        raise RuntimeError(f'Undefined participant target SD for {outcome}: {bad}')
    if not set(ev.subject_id).issubset(set(stats.index)):
        raise RuntimeError('Evaluation subject absent from seen-person training data')
    tr_mean = tr.subject_id.map(stats['mean']).to_numpy(float)
    tr_sd = tr.subject_id.map(stats['std']).to_numpy(float)
    ev_mean = ev.subject_id.map(stats['mean']).to_numpy(float)
    ev_sd = ev.subject_id.map(stats['std']).to_numpy(float)
    tr_raw = tr[outcome].to_numpy(float)
    ev_raw = ev[outcome].to_numpy(float)
    return {'train_z': (tr_raw - tr_mean) / tr_sd, 'eval_z': (ev_raw - ev_mean) / ev_sd, 'eval_raw': ev_raw, 'eval_mean': ev_mean, 'eval_sd': ev_sd, 'target_name': outcome, 'scale_type': 'participant_training_mean_sd', 'extra': {}}

def unseen_target(meta: pd.DataFrame, outcome: str, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, Any]:
    tr_raw = meta.iloc[train_idx][outcome].to_numpy(float)
    ev_raw = meta.iloc[eval_idx][outcome].to_numpy(float)
    mean = float(tr_raw.mean())
    sd = float(tr_raw.std(ddof=1))
    if not np.isfinite(sd) or sd <= SD_FLOOR:
        raise RuntimeError(f'Undefined global training SD for {outcome}')
    return {'train_z': (tr_raw - mean) / sd, 'eval_z': (ev_raw - mean) / sd, 'eval_raw': ev_raw, 'eval_mean': np.full(len(ev_raw), mean, dtype=float), 'eval_sd': np.full(len(ev_raw), sd, dtype=float), 'target_name': outcome, 'scale_type': 'global_training_mean_sd_unseen_person', 'extra': {'global_training_mean': mean, 'global_training_sd': sd}}

def conditional_liking_target(meta: pd.DataFrame, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, Any]:
    tr = meta.iloc[train_idx]
    ev = meta.iloc[eval_idx]
    xtr = np.column_stack([np.ones(len(tr)), tr.valence.to_numpy(float), tr.context.eq('video').to_numpy(float)])
    xev = np.column_stack([np.ones(len(ev)), ev.valence.to_numpy(float), ev.context.eq('video').to_numpy(float)])
    beta, *_ = np.linalg.lstsq(xtr, tr.liking.to_numpy(float), rcond=None)
    tr_res = tr.liking.to_numpy(float) - xtr @ beta
    ev_res = ev.liking.to_numpy(float) - xev @ beta
    tmp = pd.DataFrame({'subject_id': tr.subject_id.to_numpy(), 'residual': tr_res})
    stats = tmp.groupby('subject_id').residual.agg(['mean', 'std'])
    bad = stats.index[stats['std'].isna() | (stats['std'] <= SD_FLOOR)].tolist()
    if bad:
        raise RuntimeError(f'Undefined conditional residual SD: {bad}')
    ev_mean = ev.subject_id.map(stats['mean']).to_numpy(float)
    ev_sd = ev.subject_id.map(stats['std']).to_numpy(float)
    tr_mean = tr.subject_id.map(stats['mean']).to_numpy(float)
    tr_sd = tr.subject_id.map(stats['std']).to_numpy(float)
    return {'train_z': (tr_res - tr_mean) / tr_sd, 'eval_z': (ev_res - ev_mean) / ev_sd, 'eval_raw': ev_res, 'eval_mean': ev_mean, 'eval_sd': ev_sd, 'target_name': 'liking_residual_given_valence_and_context', 'scale_type': 'fold_contained_residual_then_participant_training_mean_sd', 'extra': {'residualizer_intercept': float(beta[0]), 'residualizer_valence': float(beta[1]), 'residualizer_video': float(beta[2])}}

def ridge_grid(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    e = np.asarray(x_eval, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    ec = e - x_mean
    yc = y - y_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    uy = u.T @ yc
    out = np.empty((len(e), len(alphas)), dtype=float)
    for j, alpha in enumerate(alphas):
        out[:, j] = y_mean + ec @ (vt.T @ (s / (s * s + float(alpha)) * uy))
    return out

def participant_equal_mse(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> float:
    df = pd.DataFrame({'subject': subjects, 'error': (np.asarray(y) - np.asarray(pred)) ** 2})
    return float(df.groupby('subject').error.mean().mean())

def choose_regression(scores: pd.DataFrame) -> pd.Series:
    best = float(scores.inner_mse.min())
    eligible = scores[scores.inner_mse <= best + TIE].copy()
    eligible['component_sort'] = pd.to_numeric(eligible.pca_components, errors='coerce').fillna(0).astype(int)
    return eligible.sort_values(['component_sort', 'ridge_alpha'], ascending=[True, False]).iloc[0]

def regression_design(split: dict[str, np.ndarray], component: int | None, signal_model: bool) -> tuple[np.ndarray, np.ndarray]:
    cov_tr = np.asarray(split['cov_train'], dtype=np.float64)
    cov_ev = np.asarray(split['cov_eval'], dtype=np.float64)
    if not signal_model:
        return (cov_tr, cov_ev)
    if component is None or component > split['pca_train'].shape[1]:
        raise RuntimeError(f'Invalid PCA component request: {component}')
    return (np.concatenate([cov_tr, np.asarray(split['pca_train'][:, :component], dtype=np.float64)], axis=1), np.concatenate([cov_ev, np.asarray(split['pca_eval'][:, :component], dtype=np.float64)], axis=1))

def fit_regression_nested(*, meta: pd.DataFrame, signal: np.ndarray, dataset_id: str, required_model_id: str, reference_model_id: str, outcomes: Sequence[str], fold_col: str, subject_fixed: bool, components: Sequence[int], target_mode: str, cache_root: Path, input_signature: str, recompute: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fold_labels = sorted(meta[fold_col].astype(int).unique().tolist())
    if fold_labels != FOLDS:
        raise RuntimeError(f'Unexpected fold labels for {dataset_id}: {fold_labels}')
    fold_values = meta[fold_col].astype(int).to_numpy()
    transform_root = cache_root / 'split_transforms' / dataset_id
    transform_root.mkdir(parents=True, exist_ok=True)

    def target_for(outcome: str, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, Any]:
        if target_mode == 'seen':
            return seen_target(meta, outcome, train_idx, eval_idx)
        if target_mode == 'unseen':
            return unseen_target(meta, outcome, train_idx, eval_idx)
        if target_mode == 'conditional':
            return conditional_liking_target(meta, train_idx, eval_idx)
        raise RuntimeError(f'Unknown target mode {target_mode}')
    for outer in fold_labels:
        outer_train = np.flatnonzero(fold_values != outer)
        outer_eval = np.flatnonzero(fold_values == outer)
        outer_path = split_cache(transform_root, f'outer-{outer}', meta, signal, outer_train, outer_eval, subject_fixed, components, input_signature, SEED + 1000 + outer, recompute)
        outer_split = load_split(outer_path)
        outer_lookup = {int(global_i): local_i for local_i, global_i in enumerate(outer_train)}
        for outcome in outcomes:
            candidate_map: dict[tuple[str, int | None, float], np.ndarray] = {}
            for alpha in ALPHAS:
                candidate_map['reference', None, float(alpha)] = np.full(len(outer_train), np.nan)
            for component in components:
                for alpha in ALPHAS:
                    candidate_map['signal', int(component), float(alpha)] = np.full(len(outer_train), np.nan)
            inner_truth = np.full(len(outer_train), np.nan)
            for inner in fold_labels:
                if inner == outer:
                    continue
                inner_train = np.flatnonzero((fold_values != outer) & (fold_values != inner))
                inner_eval = np.flatnonzero(fold_values == inner)
                inner_path = split_cache(transform_root, f'outer-{outer}_inner-{inner}', meta, signal, inner_train, inner_eval, subject_fixed, components, input_signature, SEED + 1000 + 100 * outer + inner, recompute)
                split = load_split(inner_path)
                target = target_for(outcome, inner_train, inner_eval)
                local = np.array([outer_lookup[int(i)] for i in inner_eval], dtype=int)
                inner_truth[local] = target['eval_z']
                xtr_ref, xev_ref = regression_design(split, None, False)
                ref_preds = ridge_grid(xtr_ref, target['train_z'], xev_ref, ALPHAS)
                for j, alpha in enumerate(ALPHAS):
                    candidate_map['reference', None, float(alpha)][local] = ref_preds[:, j]
                for component in components:
                    xtr, xev = regression_design(split, int(component), True)
                    signal_preds = ridge_grid(xtr, target['train_z'], xev, ALPHAS)
                    for j, alpha in enumerate(ALPHAS):
                        candidate_map['signal', int(component), float(alpha)][local] = signal_preds[:, j]
            if np.isnan(inner_truth).any():
                raise RuntimeError(f'Incomplete inner truth for {dataset_id}/{outcome}/outer-{outer}')
            inner_subjects = meta.iloc[outer_train].subject_id.astype(str).to_numpy()
            candidates = []
            for (kind, component, alpha), pred in candidate_map.items():
                if np.isnan(pred).any():
                    raise RuntimeError(f'Incomplete inner predictions for {dataset_id}/{outcome}/{kind}')
                row = {'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'required_model_id': required_model_id, 'candidate_kind': kind, 'outcome': outcome, 'outer_fold': outer, 'pca_components': component, 'ridge_alpha': alpha, 'inner_mse': participant_equal_mse(inner_truth, pred, inner_subjects), 'inner_rows': len(pred), 'inner_subjects': len(set(inner_subjects)), 'target_mode': target_mode}
                candidates.append(row)
                score_rows.append(row)
            candidate_df = pd.DataFrame(candidates)
            for kind, model_id, signal_model in [('reference', reference_model_id, False), ('signal', required_model_id, True)]:
                chosen = choose_regression(candidate_df[candidate_df.candidate_kind.eq(kind)])
                component = None if pd.isna(chosen.pca_components) else int(chosen.pca_components)
                alpha = float(chosen.ridge_alpha)
                outer_target = target_for(outcome, outer_train, outer_eval)
                xtr, xev = regression_design(outer_split, component, signal_model)
                pred_z = ridge_grid(xtr, outer_target['train_z'], xev, [alpha])[:, 0]
                pred_raw = pred_z * outer_target['eval_sd'] + outer_target['eval_mean']
                selections.append({'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'model_id': model_id, 'required_model_id': required_model_id, 'outcome': outcome, 'outer_fold': outer, 'pca_components': component, 'ridge_alpha': alpha, 'selected_inner_mse': float(chosen.inner_mse), 'outer_train_rows': len(outer_train), 'outer_test_rows': len(outer_eval), 'design_columns': int(xtr.shape[1]), 'target_mode': target_mode, **outer_target['extra']})
                eval_meta = meta.iloc[outer_eval]
                for j, global_idx in enumerate(outer_eval):
                    r = eval_meta.iloc[j]
                    predictions.append({'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'raw_behavior_row_id': r.raw_behavior_row_id, 'subject_id': r.subject_id, 'trial_number': int(r.trial_number), 'context': r.context, 'raw_run': int(r.raw_run), 'subject_acquisition_order_1based': int(r.subject_acquisition_order_1based), 'within_subject_block_fold': int(r.within_subject_block_fold), 'cross_person_fold': int(r.cross_person_fold), 'outcome': outcome, 'target_name': outer_target['target_name'], 'target_scale_type': outer_target['scale_type'], 'model_id': model_id, 'required_model_id': required_model_id, 'outer_fold': outer, 'observed_standardized': float(outer_target['eval_z'][j]), 'predicted_standardized': float(pred_z[j]), 'observed_raw_target': float(outer_target['eval_raw'][j]), 'predicted_raw_target': float(pred_raw[j]), 'raw_observed_valence': float(r.valence), 'raw_observed_liking': float(r.liking), 'pca_components': component, 'ridge_alpha': alpha})
            print(f'[{now()}] Regression {dataset_id} / {outcome} / outer {outer} complete', flush=True)
    return (pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(score_rows))

def regression_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['dataset_id', 'analysis_set', 'required_model_id', 'outcome', 'target_name', 'model_id']
    for keys, g in pred.groupby(group_cols, sort=True):
        y = g.observed_standardized.to_numpy(float)
        p = g.predicted_standardized.to_numpy(float)
        yr = g.observed_raw_target.to_numpy(float)
        pr = g.predicted_raw_target.to_numpy(float)
        denom = float(np.sum(y * y))
        target_relative_r2 = float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan
        raw_sst = float(np.sum((yr - yr.mean()) ** 2))
        raw_r2 = float(1 - np.sum((yr - pr) ** 2) / raw_sst) if raw_sst > 0 else np.nan
        participant_rows = []
        for subject, sg in g.groupby('subject_id'):
            sy = sg.observed_raw_target.to_numpy(float)
            sp = sg.predicted_raw_target.to_numpy(float)
            sst = float(np.sum((sy - sy.mean()) ** 2))
            participant_rows.append({'subject_id': subject, 'r2': float(1 - np.sum((sy - sp) ** 2) / sst) if sst > 0 else np.nan, 'mae': float(np.mean(np.abs(sy - sp)))})
        pm = pd.DataFrame(participant_rows)
        pearson = float(np.corrcoef(yr, pr)[0, 1]) if len(yr) > 2 and np.std(yr) > 0 and (np.std(pr) > 0) else np.nan
        spearman = float(pd.Series(yr).corr(pd.Series(pr), method='spearman')) if len(yr) > 2 else np.nan
        rows.append({**dict(zip(group_cols, keys)), 'pooled_target_relative_r2': target_relative_r2, 'pooled_conventional_raw_r2': raw_r2, 'participant_median_raw_r2': float(pm.r2.median(skipna=True)), 'participant_median_raw_mae': float(pm.mae.median()), 'pooled_raw_mae': float(np.mean(np.abs(yr - pr))), 'pooled_pearson_r': pearson, 'pooled_spearman_r': spearman, 'rows': len(g), 'subjects': g.subject_id.nunique()})
    return pd.DataFrame(rows)

def regression_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in metrics.groupby(['dataset_id', 'analysis_set', 'required_model_id', 'outcome', 'target_name'], sort=True):
        required_id = keys[2]
        sig = g[g.model_id.eq(required_id)]
        ref = g[~g.model_id.eq(required_id)]
        if len(sig) != 1 or len(ref) != 1:
            raise RuntimeError(f'Expected one signal and one reference metric for {keys}')
        s = sig.iloc[0]
        r = ref.iloc[0]
        rows.append({'dataset_id': keys[0], 'analysis_set': keys[1], 'required_model_id': required_id, 'outcome': keys[3], 'target_name': keys[4], 'reference_model_id': r.model_id, 'delta_pooled_target_relative_r2': float(s.pooled_target_relative_r2 - r.pooled_target_relative_r2), 'delta_pooled_conventional_raw_r2': float(s.pooled_conventional_raw_r2 - r.pooled_conventional_raw_r2), 'delta_pooled_raw_mae': float(s.pooled_raw_mae - r.pooled_raw_mae)})
    return pd.DataFrame(rows)

def p2_subject_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    p2 = pred[pred.required_model_id.eq('P2_person_specific_compact_all3')].copy()
    rows = []
    for (outcome, subject), g in p2.groupby(['outcome', 'subject_id'], sort=True):
        model_values = {}
        for model_id, mg in g.groupby('model_id'):
            y = mg.observed_raw_target.to_numpy(float)
            p = mg.predicted_raw_target.to_numpy(float)
            sst = float(np.sum((y - y.mean()) ** 2))
            model_values[model_id] = {'r2': float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan, 'mae': float(np.mean(np.abs(y - p)))}
        sig = model_values['P2_person_specific_compact_all3']
        ref_id = [x for x in model_values if x != 'P2_person_specific_compact_all3'][0]
        ref = model_values[ref_id]
        rows.append({'outcome': outcome, 'subject_id': subject, 'signal_model_id': 'P2_person_specific_compact_all3', 'reference_model_id': ref_id, 'signal_r2': sig['r2'], 'reference_r2': ref['r2'], 'delta_r2': sig['r2'] - ref['r2'], 'signal_mae': sig['mae'], 'reference_mae': ref['mae'], 'delta_mae': sig['mae'] - ref['mae'], 'rows': len(g[g.model_id.eq('P2_person_specific_compact_all3')])})
    return pd.DataFrame(rows)

def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)
        adjusted[key] = running
    return adjusted

def p2_sign_tests(subject_metrics: pd.DataFrame, primary_support: list[dict[str, Any]]) -> pd.DataFrame:
    raw_p = {}
    interim = []
    primary_map = {r['outcome']: bval(r['primary_signal_specific_support']) for r in primary_support}
    for outcome, g in subject_metrics.groupby('outcome', sort=True):
        delta = g.delta_r2.to_numpy(float)
        finite = delta[np.isfinite(delta)]
        positive = int(np.sum(finite > TIE))
        negative = int(np.sum(finite < -TIE))
        ties = int(len(finite) - positive - negative)
        n = positive + negative
        p = float(binomtest(positive, n=n, p=0.5, alternative='two-sided').pvalue) if n > 0 else 1.0
        raw_p[outcome] = p
        interim.append({'outcome': outcome, 'subjects': len(g), 'finite_subjects': len(finite), 'positive_delta_subjects': positive, 'negative_delta_subjects': negative, 'ties_excluded': ties, 'median_subject_delta_r2': float(np.median(finite)) if len(finite) else np.nan, 'exact_sign_p_raw': p, 'primary_signal_support_gate': primary_map.get(outcome, False)})
    adjusted = holm_adjust(raw_p)
    rows = []
    for row in interim:
        row['exact_sign_p_holm'] = adjusted[row['outcome']]
        row['median_delta_gate'] = bool(row['median_subject_delta_r2'] > 0)
        row['sign_test_gate'] = bool(row['exact_sign_p_holm'] < 0.05)
        row['person_specific_implementation_support'] = bool(row['primary_signal_support_gate'] and row['median_delta_gate'] and row['sign_test_gate'])
        rows.append(row)
    return pd.DataFrame(rows)

def bootstrap_unseen(pred: pd.DataFrame, replicates: int=BOOTSTRAPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    u1 = pred[pred.required_model_id.eq('U1_unseen_person_universal_all3')].copy()
    outcomes = sorted(u1.outcome.unique())
    subject_sets = [sorted(u1[u1.outcome.eq(outcome)].subject_id.unique()) for outcome in outcomes]
    if not subject_sets or any((subjects != subject_sets[0] for subjects in subject_sets[1:])):
        raise RuntimeError('U1 outcomes do not share the same participant set')
    subject_ids = subject_sets[0]
    rng = np.random.default_rng(SEED)
    draw_indices = rng.integers(0, len(subject_ids), size=(replicates, len(subject_ids)))
    replicate_rows = []
    interval_rows = []
    for outcome in outcomes:
        og = u1[u1.outcome.eq(outcome)]
        models = sorted(og.model_id.unique())
        aggregates = {}
        for model_id in models:
            mg = og[og.model_id.eq(model_id)]
            per_subject = {}
            for subject, sg in mg.groupby('subject_id'):
                y = sg.observed_raw_target.to_numpy(float)
                p = sg.predicted_raw_target.to_numpy(float)
                per_subject[subject] = {'n': len(y), 'sum_y': float(y.sum()), 'sum_y2': float(np.sum(y * y)), 'sse': float(np.sum((y - p) ** 2))}
            aggregates[model_id] = per_subject
        model_values = {m: np.empty(replicates, dtype=float) for m in models}
        for rep in range(replicates):
            draw = [subject_ids[i] for i in draw_indices[rep]]
            for model_id in models:
                n = sum((aggregates[model_id][subject]['n'] for subject in draw))
                sum_y = sum((aggregates[model_id][subject]['sum_y'] for subject in draw))
                sum_y2 = sum((aggregates[model_id][subject]['sum_y2'] for subject in draw))
                sse = sum((aggregates[model_id][subject]['sse'] for subject in draw))
                sst = sum_y2 - sum_y * sum_y / n
                model_values[model_id][rep] = 1 - sse / sst if sst > 0 else np.nan
        required_id = 'U1_unseen_person_universal_all3'
        ref_id = [m for m in models if m != required_id][0]
        delta = model_values[required_id] - model_values[ref_id]
        for rep in range(replicates):
            for model_id in models:
                replicate_rows.append({'replicate': rep + 1, 'outcome': outcome, 'model_id': model_id, 'conventional_raw_r2': model_values[model_id][rep]})
            replicate_rows.append({'replicate': rep + 1, 'outcome': outcome, 'model_id': 'DELTA_U1_MINUS_REFERENCE', 'conventional_raw_r2': delta[rep]})
        for model_id in models + ['DELTA_U1_MINUS_REFERENCE']:
            values = delta if model_id == 'DELTA_U1_MINUS_REFERENCE' else model_values[model_id]
            interval_rows.append({'outcome': outcome, 'model_id': model_id, 'bootstrap_mean': float(np.nanmean(values)), 'ci_lower_95': float(np.nanpercentile(values, 2.5)), 'ci_upper_95': float(np.nanpercentile(values, 97.5)), 'replicates': replicates, 'common_participant_draws_across_outcomes': True, 'universal_decoding_support': bool(model_id == required_id and np.nanpercentile(values, 2.5) > 0)})
    return (pd.DataFrame(replicate_rows), pd.DataFrame(interval_rows))

def participant_equal_logloss(y_true: np.ndarray, probabilities: np.ndarray, subjects: np.ndarray, labels: np.ndarray) -> float:
    rows = []
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        rows.append(log_loss(y_true[mask], probabilities[mask], labels=labels))
    return float(np.mean(rows))

def fit_classification_nested(*, meta: pd.DataFrame, raw_features: np.ndarray, dataset_id: str, model_id: str, representation: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_values = meta.within_subject_block_fold.astype(int).to_numpy()
    predictions = []
    selections = []
    scores = []
    if sorted(np.unique(fold_values).tolist()) != FOLDS:
        raise RuntimeError(f'Unexpected classification fold labels for {dataset_id}')
    if target == 'subject_id':
        labels_text = np.array(sorted(meta.subject_id.astype(str).unique()))
        label_to_int = {s: i for i, s in enumerate(labels_text)}
        y_all = meta.subject_id.astype(str).map(label_to_int).to_numpy(int)
        labels = np.arange(len(labels_text), dtype=int)
    elif target == 'context':
        labels_text = np.array(['imagery', 'video'])
        label_to_int = {'imagery': 0, 'video': 1}
        y_all = meta.context.map(label_to_int).to_numpy(int)
        labels = np.array([0, 1], dtype=int)
    else:
        raise RuntimeError(f'Unknown classification target {target}')
    for outer in FOLDS:
        outer_train = np.flatnonzero(fold_values != outer)
        outer_eval = np.flatnonzero(fold_values == outer)
        outer_lookup = {int(g): i for i, g in enumerate(outer_train)}
        prob_oof = {float(c): np.full((len(outer_train), len(labels)), np.nan) for c in LOGISTIC_C}
        truth_oof = np.full(len(outer_train), -1, dtype=int)
        for inner in FOLDS:
            if inner == outer:
                continue
            inner_train = np.flatnonzero((fold_values != outer) & (fold_values != inner))
            inner_eval = np.flatnonzero(fold_values == inner)
            xtr, xev, _ = standardized_transform(raw_features, inner_train, inner_eval)
            ytr = y_all[inner_train]
            yev = y_all[inner_eval]
            if set(labels) - set(np.unique(ytr)):
                raise RuntimeError(f'Classification training labels missing for {dataset_id}/outer-{outer}/inner-{inner}')
            local = np.array([outer_lookup[int(i)] for i in inner_eval], dtype=int)
            truth_oof[local] = yev
            for c in LOGISTIC_C:
                clf = LogisticRegression(C=float(c), solver='lbfgs', max_iter=5000, random_state=SEED)
                clf.fit(xtr, ytr)
                prob = clf.predict_proba(xev)
                aligned = np.zeros((len(xev), len(labels)), dtype=float)
                for j, cls in enumerate(clf.classes_.astype(int)):
                    aligned[:, int(cls)] = prob[:, j]
                prob_oof[float(c)][local] = aligned
        if (truth_oof < 0).any():
            raise RuntimeError(f'Incomplete classification inner truth for {dataset_id}')
        subject_oof = meta.iloc[outer_train].subject_id.astype(str).to_numpy()
        candidate_rows = []
        for c in LOGISTIC_C:
            prob = prob_oof[float(c)]
            if np.isnan(prob).any():
                raise RuntimeError(f'Incomplete classification probabilities for {dataset_id}/C={c}')
            loss = participant_equal_logloss(truth_oof, prob, subject_oof, labels)
            row = {'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'model_id': model_id, 'representation': representation, 'target': target, 'outer_fold': outer, 'logistic_c': float(c), 'inner_participant_equal_log_loss': loss, 'inner_rows': len(outer_train), 'inner_subjects': len(set(subject_oof))}
            candidate_rows.append(row)
            scores.append(row)
        candidate_df = pd.DataFrame(candidate_rows)
        best = float(candidate_df.inner_participant_equal_log_loss.min())
        chosen = candidate_df[candidate_df.inner_participant_equal_log_loss <= best + TIE].sort_values('logistic_c').iloc[0]
        c = float(chosen.logistic_c)
        xtr, xev, diag = standardized_transform(raw_features, outer_train, outer_eval)
        clf = LogisticRegression(C=c, solver='lbfgs', max_iter=5000, random_state=SEED)
        clf.fit(xtr, y_all[outer_train])
        prob = clf.predict_proba(xev)
        aligned = np.zeros((len(xev), len(labels)), dtype=float)
        for j, cls in enumerate(clf.classes_.astype(int)):
            aligned[:, int(cls)] = prob[:, j]
        pred_int = aligned.argmax(axis=1)
        yev = y_all[outer_eval]
        selections.append({'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'model_id': model_id, 'representation': representation, 'target': target, 'outer_fold': outer, 'logistic_c': c, 'selected_inner_participant_equal_log_loss': float(chosen.inner_participant_equal_log_loss), 'outer_train_rows': len(outer_train), 'outer_test_rows': len(outer_eval), 'retained_feature_columns': diag['retained_columns']})
        eval_meta = meta.iloc[outer_eval]
        for j, global_idx in enumerate(outer_eval):
            true_int = int(yev[j])
            pred_label_int = int(pred_int[j])
            top5 = np.argsort(aligned[j])[-min(5, len(labels)):]
            predictions.append({'dataset_id': dataset_id, 'analysis_set': str(meta.analysis_set.iloc[0]), 'model_id': model_id, 'representation': representation, 'target': target, 'raw_behavior_row_id': eval_meta.iloc[j].raw_behavior_row_id, 'subject_id': eval_meta.iloc[j].subject_id, 'context': eval_meta.iloc[j].context, 'outer_fold': outer, 'true_label': str(labels_text[true_int]), 'predicted_label': str(labels_text[pred_label_int]), 'true_probability': float(aligned[j, true_int]), 'positive_video_probability': float(aligned[j, 1]) if target == 'context' else np.nan, 'top5_correct': bool(true_int in top5), 'logistic_c': c})
        print(f'[{now()}] Classification {dataset_id} / {representation} / outer {outer} complete', flush=True)
    return (pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(scores))

def classification_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['dataset_id', 'analysis_set', 'model_id', 'representation', 'target']
    for keys, g in pred.groupby(group_cols, sort=True):
        target = keys[-1]
        y_true = g.true_label.astype(str).to_numpy()
        y_pred = g.predicted_label.astype(str).to_numpy()
        base = {**dict(zip(group_cols, keys)), 'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro')), 'mean_true_class_probability': float(g.true_probability.mean()), 'rows': len(g), 'subjects': g.subject_id.nunique()}
        if target == 'subject_id':
            labels = sorted(g.true_label.unique())
            y_int = pd.Categorical(y_true, categories=labels).codes
            base['log_loss'] = float(-np.mean(np.log(np.clip(g.true_probability.to_numpy(float), 1e-15, 1))))
            base['top5_accuracy'] = float(g.top5_correct.mean())
            base['chance_accuracy'] = 1.0 / len(labels)
            base['roc_auc'] = np.nan
        else:
            y_int = (y_true == 'video').astype(int)
            prob = g.positive_video_probability.to_numpy(float)
            base['log_loss'] = float(log_loss(y_int, np.column_stack([1 - prob, prob]), labels=[0, 1]))
            base['top5_accuracy'] = np.nan
            base['chance_accuracy'] = max(float(np.mean(y_int)), float(1 - np.mean(y_int)))
            base['roc_auc'] = float(roc_auc_score(y_int, prob))
        rows.append(base)
    return pd.DataFrame(rows)

def cache_inventory(cache: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(cache.rglob('*')):
        if p.is_file():
            rows.append({'stored_path': str(p), 'relative_path': str(p.relative_to(cache)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    return pd.DataFrame(rows)

def write_report(path: Path, summary: dict[str, Any], reg_metrics: pd.DataFrame, contrasts: pd.DataFrame, p2_tests: pd.DataFrame, u1_intervals: pd.DataFrame, class_metrics: pd.DataFrame) -> None:
    lines = ['# Phase secondary_models — prespecified secondary and diagnostic models', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", f"- Execution-policy SHA-256: `{summary['execution_policy_sha256']}`", '- All seven prespecified secondary/diagnostic families executed: **yes**', '- Outer-test family or epoch selection: **no**', '- Validation/confirmation opened: **no**', '', '## Regression model metrics', '', '| Dataset | Outcome | Model | Target-relative R² | Raw R² | Raw MAE |', '|---|---|---|---:|---:|---:|']
    for _, r in reg_metrics.sort_values(['required_model_id', 'dataset_id', 'outcome', 'model_id']).iterrows():
        lines.append(f'| `{r.dataset_id}` | `{r.outcome}` | `{r.model_id}` | {r.pooled_target_relative_r2:.6f} | {r.pooled_conventional_raw_r2:.6f} | {r.pooled_raw_mae:.6f} |')
    lines += ['', 'Matched covariate-only references are descriptive comparators within the same prespecified rows and splits. They do not add or replace a prespecified claim family.', '', '## Incremental regression contrasts', '', '| Dataset | Outcome | Required model | Δ target-relative R² | Δ raw R² |', '|---|---|---|---:|---:|']
    for _, r in contrasts.sort_values(['required_model_id', 'dataset_id', 'outcome']).iterrows():
        lines.append(f'| `{r.dataset_id}` | `{r.outcome}` | `{r.required_model_id}` | {r.delta_pooled_target_relative_r2:.6f} | {r.delta_pooled_conventional_raw_r2:.6f} |')
    lines += ['', '## Person-specific implementation test', '', '| Outcome | Median subject ΔR² | Sign p | Holm p | Primary gate | P2 support |', '|---|---:|---:|---:|---|---|']
    for _, r in p2_tests.sort_values('outcome').iterrows():
        lines.append(f'| `{r.outcome}` | {r.median_subject_delta_r2:.6f} | {r.exact_sign_p_raw:.6f} | {r.exact_sign_p_holm:.6f} | {bool(r.primary_signal_support_gate)} | {bool(r.person_specific_implementation_support)} |')
    lines += ['', 'The prespecified P2 implementation claim also requires the primary signal-specific support gate. Because the primary gate was not met, P2 cannot establish implementation support in this discovery stage even if its descriptive subject-level pattern is favorable.', '', '## Unseen-person bootstrap', '', '| Outcome | Model | Mean R² | 95% CI | Universal support |', '|---|---|---:|---:|---|']
    for _, r in u1_intervals.sort_values(['outcome', 'model_id']).iterrows():
        lines.append(f'| `{r.outcome}` | `{r.model_id}` | {r.bootstrap_mean:.6f} | [{r.ci_lower_95:.6f}, {r.ci_upper_95:.6f}] | {bool(r.universal_decoding_support)} |')
    lines += ['', '## Diagnostic classification', '', '| Dataset | Target | Representation | Accuracy | Balanced accuracy | Macro F1 | ROC AUC / Top-5 |', '|---|---|---|---:|---:|---:|---:|']
    for _, r in class_metrics.sort_values(['model_id', 'dataset_id', 'representation']).iterrows():
        extra = r.roc_auc if r.target == 'context' else r.top5_accuracy
        lines.append(f'| `{r.dataset_id}` | `{r.target}` | `{r.representation}` | {r.accuracy:.6f} | {r.balanced_accuracy:.6f} | {r.macro_f1:.6f} | {extra:.6f} |')
    lines += ['', 'Identity performance is interpreted only as subject separability. Context performance is interpreted only as context information. Neither is emotion decoding.', '', '## Interpretation boundary', '', '- S1 and S2 are reported independently; no best epoch or pair is selected from outer-test performance.', '- E1 is conditional association after fold-contained liking residualization, not deployable liking prediction.', '- Unseen-person failure, if present, indicates nonseparability or misalignment under this representation and does not imply absent physiology.', '- Validation and confirmation remain closed.', '']
    path.write_text(''.join(lines), encoding='utf-8')

def self_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 64, 6, 5)).astype(np.float32)
    compact = channel_median_iqr(x)
    assert compact.shape == (7, 60)
    meta = pd.DataFrame({'subject_id': ['a'] * 5 + ['b'] * 5, 'valence': np.arange(10, dtype=float), 'liking': np.arange(10, dtype=float) + 0.5, 'context': ['imagery', 'video'] * 5})
    t = seen_target(meta, 'valence', np.array([0, 1, 2, 5, 6, 7]), np.array([3, 4, 8, 9]))
    assert np.isfinite(t['train_z']).all() and np.isfinite(t['eval_z']).all()
    xx = rng.normal(size=(20, 5))
    yy = rng.normal(size=20)
    pred = ridge_grid(xx[:15], yy[:15], xx[15:], [0.1, 1.0])
    assert pred.shape == (5, 2)
    adjusted = holm_adjust({'a': 0.01, 'b': 0.04})
    assert abs(adjusted['a'] - 0.02) < 1e-12 and abs(adjusted['b'] - 0.04) < 1e-12
    print('secondary_models self-test: PASS')

def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, contracts, feature_integrity, feature_rows, trial_outcomes, analysis_rows = load_inputs(project)
    cache = project / CACHE_REL
    tables = project / 'outputs/tables'
    qc_dir = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config = project / 'config'
    logs = project / 'outputs/logs'
    if shutil.disk_usage(project).free < 2 * 1024 ** 3:
        raise RuntimeError('At least 2 GiB free space is required for secondary_models caches')
    out = {'feature_rehash': qc_dir / 'secondary_models_feature_store_rehash.csv', 'primary_models_cache_rehash': qc_dir / 'secondary_models_primary_models_cache_rehash.csv', 'reg_predictions': tables / 'secondary_models_secondary_regression_outer_predictions.csv', 'reg_selected': tables / 'secondary_models_secondary_regression_selected_hyperparameters.csv', 'reg_inner': tables / 'secondary_models_secondary_regression_inner_scores.csv', 'reg_metrics': tables / 'secondary_models_secondary_regression_metrics.csv', 'reg_contrasts': tables / 'secondary_models_secondary_regression_contrasts.csv', 'p2_subject_metrics': tables / 'secondary_models_person_specific_subject_metrics.csv', 'p2_sign_tests': tables / 'secondary_models_person_specific_sign_tests.csv', 'u1_bootstrap_replicates': tables / 'secondary_models_unseen_person_bootstrap_replicates.csv', 'u1_bootstrap_intervals': tables / 'secondary_models_unseen_person_bootstrap_intervals.csv', 'class_predictions': tables / 'secondary_models_diagnostic_classification_predictions.csv', 'class_selected': tables / 'secondary_models_diagnostic_classification_selected_hyperparameters.csv', 'class_inner': tables / 'secondary_models_diagnostic_classification_inner_scores.csv', 'class_metrics': tables / 'secondary_models_diagnostic_classification_metrics.csv', 'cache_inventory': tables / 'secondary_models_cache_inventory.csv', 'summary': qc_dir / 'secondary_models_secondary_models_summary.json', 'contract': config / 'secondary_models_secondary_models_contract.json', 'next_policy': config / 'S059_secondary_models_latent_state.json', 'report': reports / 'secondary_models_secondary_models_report.md', 'log': logs / 'secondary_models_secondary_models.log'}
    existing = [p for p in out.values() if p.exists()]
    if existing and (not args.force):
        raise RuntimeError('Outputs exist; use --force:\n  - ' + '\n  - '.join(map(str, existing)))
    with out['log'].open('w', encoding='utf-8') as log:

        def note(text: str) -> None:
            line = f'[{now()}] {text}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Rehashing prespecified outcome_model_integrity.1 feature store.')
        feature_rehash = rehash_feature_store(feature_integrity)
        feature_rehash.to_csv(out['feature_rehash'], index=False)
        note('Rehashing all 254 primary_models cache files.')
        primary_models_rehash = rehash_primary_models_cache(project, paths['primary_models_cache_inventory'])
        primary_models_rehash.to_csv(out['primary_models_cache_rehash'], index=False)
        base_signature = csha({'policy': POLICY_SHA, 'feature_rows': sha(paths['feature_rows']), 'trial_outcomes': sha(paths['trial_outcomes']), 'analysis_rows': sha(paths['analysis_rows']), 'primary_inference_summary': sha(paths['primary_inference_summary']), 'primary_inference_support': sha(paths['primary_inference_support'])})
        note('Building or verifying universal outcome-free epoch store.')
        epoch_store = build_epoch_store(cache, feature_rows, base_signature, args.recompute_cache)
        matrices: dict[str, tuple[pd.DataFrame, Path]] = {}
        for analysis_set, families in SINGLE_SETS.items():
            meta = analysis_metadata(trial_outcomes, analysis_rows, analysis_set, 'within_person_eligible')
            dataset_id = f'within_{analysis_set}'
            path = build_analysis_matrix(cache, dataset_id, meta, families, epoch_store, base_signature, args.recompute_cache)
            matrices[dataset_id] = (meta, path)
        for analysis_set, families in PAIR_SETS.items():
            meta = analysis_metadata(trial_outcomes, analysis_rows, analysis_set, 'within_person_eligible')
            dataset_id = f'within_{analysis_set}'
            path = build_analysis_matrix(cache, dataset_id, meta, families, epoch_store, base_signature, args.recompute_cache)
            matrices[dataset_id] = (meta, path)
        meta_all_seen = analysis_metadata(trial_outcomes, analysis_rows, 'all_three_epoch_families', 'within_person_eligible')
        path_all_seen = build_analysis_matrix(cache, 'within_all_three', meta_all_seen, FAMILIES, epoch_store, base_signature, args.recompute_cache)
        matrices['within_all_three'] = (meta_all_seen, path_all_seen)
        meta_all_unseen = analysis_metadata(trial_outcomes, analysis_rows, 'all_three_epoch_families', 'cross_person_eligible')
        path_all_unseen = build_analysis_matrix(cache, 'unseen_all_three', meta_all_unseen, FAMILIES, epoch_store, base_signature, args.recompute_cache)
        matrices['unseen_all_three'] = (meta_all_unseen, path_all_unseen)
        reg_preds = []
        reg_selected = []
        reg_inner = []
        for analysis_set in SINGLE_SETS:
            dataset_id = f'within_{analysis_set}'
            meta, matrix_path = matrices[dataset_id]
            signal = np.load(matrix_path / 'signal_float32.npy', mmap_mode='r', allow_pickle=False)
            p, s, i = fit_regression_nested(meta=meta, signal=signal, dataset_id=dataset_id, required_model_id='S1_seen_single_epoch_full_signal', reference_model_id='REF_S1_seen_covariate_only', outcomes=OUTCOMES, fold_col='within_subject_block_fold', subject_fixed=True, components=POOLED_COMPONENTS, target_mode='seen', cache_root=cache, input_signature=base_signature, recompute=args.recompute_cache)
            reg_preds.append(p)
            reg_selected.append(s)
            reg_inner.append(i)
        for analysis_set in PAIR_SETS:
            dataset_id = f'within_{analysis_set}'
            meta, matrix_path = matrices[dataset_id]
            signal = np.load(matrix_path / 'signal_float32.npy', mmap_mode='r', allow_pickle=False)
            p, s, i = fit_regression_nested(meta=meta, signal=signal, dataset_id=dataset_id, required_model_id='S2_seen_pair_epoch_full_signal', reference_model_id='REF_S2_seen_covariate_only', outcomes=OUTCOMES, fold_col='within_subject_block_fold', subject_fixed=True, components=POOLED_COMPONENTS, target_mode='seen', cache_root=cache, input_signature=base_signature, recompute=args.recompute_cache)
            reg_preds.append(p)
            reg_selected.append(s)
            reg_inner.append(i)
        meta, matrix_path = matrices['within_all_three']
        compact_all = np.load(matrix_path / 'compact_float32.npy', mmap_mode='r', allow_pickle=False)
        for subject in sorted(meta.subject_id.unique()):
            idx = np.flatnonzero(meta.subject_id.eq(subject).to_numpy())
            subject_meta = meta.iloc[idx].reset_index(drop=True)
            subject_signal = np.asarray(compact_all[idx], dtype=np.float32)
            p, s, i = fit_regression_nested(meta=subject_meta, signal=subject_signal, dataset_id=f'p2_{subject}', required_model_id='P2_person_specific_compact_all3', reference_model_id='REF_P2_person_specific_covariate_only', outcomes=OUTCOMES, fold_col='within_subject_block_fold', subject_fixed=False, components=PERSON_COMPONENTS, target_mode='seen', cache_root=cache, input_signature=base_signature, recompute=args.recompute_cache)
            reg_preds.append(p)
            reg_selected.append(s)
            reg_inner.append(i)
        meta, matrix_path = matrices['unseen_all_three']
        signal = np.load(matrix_path / 'signal_float32.npy', mmap_mode='r', allow_pickle=False)
        p, s, i = fit_regression_nested(meta=meta, signal=signal, dataset_id='unseen_all_three', required_model_id='U1_unseen_person_universal_all3', reference_model_id='REF_U1_unseen_covariate_only', outcomes=OUTCOMES, fold_col='cross_person_fold', subject_fixed=False, components=POOLED_COMPONENTS, target_mode='unseen', cache_root=cache, input_signature=base_signature, recompute=args.recompute_cache)
        reg_preds.append(p)
        reg_selected.append(s)
        reg_inner.append(i)
        meta, matrix_path = matrices['within_all_three']
        signal = np.load(matrix_path / 'signal_float32.npy', mmap_mode='r', allow_pickle=False)
        p, s, i = fit_regression_nested(meta=meta, signal=signal, dataset_id='conditional_liking_all_three', required_model_id='E1_liking_given_valence', reference_model_id='REF_E1_conditional_covariate_only', outcomes=['liking'], fold_col='within_subject_block_fold', subject_fixed=True, components=POOLED_COMPONENTS, target_mode='conditional', cache_root=cache, input_signature=base_signature, recompute=args.recompute_cache)
        reg_preds.append(p)
        reg_selected.append(s)
        reg_inner.append(i)
        reg_pred_df = pd.concat(reg_preds, ignore_index=True)
        reg_sel_df = pd.concat(reg_selected, ignore_index=True)
        reg_inner_df = pd.concat(reg_inner, ignore_index=True)
        reg_metric_df = regression_metrics(reg_pred_df)
        reg_contrast_df = regression_contrasts(reg_metric_df)
        p2_subject_df = p2_subject_metrics(reg_pred_df)
        p2_test_df = p2_sign_tests(p2_subject_df, contracts['primary_support'])
        u1_rep_df, u1_interval_df = bootstrap_unseen(reg_pred_df)
        reg_pred_df.to_csv(out['reg_predictions'], index=False)
        reg_sel_df.to_csv(out['reg_selected'], index=False)
        reg_inner_df.to_csv(out['reg_inner'], index=False)
        reg_metric_df.to_csv(out['reg_metrics'], index=False)
        reg_contrast_df.to_csv(out['reg_contrasts'], index=False)
        p2_subject_df.to_csv(out['p2_subject_metrics'], index=False)
        p2_test_df.to_csv(out['p2_sign_tests'], index=False)
        u1_rep_df.to_csv(out['u1_bootstrap_replicates'], index=False)
        u1_interval_df.to_csv(out['u1_bootstrap_intervals'], index=False)
        class_preds = []
        class_selected = []
        class_inner = []
        meta, matrix_path = matrices['within_all_three']
        compact = np.load(matrix_path / 'compact_float32.npy', mmap_mode='r', allow_pickle=False)
        qc = np.load(matrix_path / 'qc_float32.npy', mmap_mode='r', allow_pickle=False)
        reps = {'compact_signal_507': compact, 'qc_only_195': qc, 'combined_702': np.concatenate([np.asarray(compact), np.asarray(qc)], axis=1).astype(np.float32)}
        for representation, raw in reps.items():
            p, s, i = fit_classification_nested(meta=meta, raw_features=raw, dataset_id='identity_all_three', model_id='D1_subject_identity_probe_all3', representation=representation, target='subject_id')
            class_preds.append(p)
            class_selected.append(s)
            class_inner.append(i)
        for analysis_set in SINGLE_SETS:
            dataset_id = f'within_{analysis_set}'
            meta, matrix_path = matrices[dataset_id]
            compact = np.load(matrix_path / 'compact_float32.npy', mmap_mode='r', allow_pickle=False)
            p, s, i = fit_classification_nested(meta=meta, raw_features=compact, dataset_id=f'context_{analysis_set}', model_id='D2_context_probe', representation='compact_signal_169', target='context')
            class_preds.append(p)
            class_selected.append(s)
            class_inner.append(i)
        class_pred_df = pd.concat(class_preds, ignore_index=True)
        class_sel_df = pd.concat(class_selected, ignore_index=True)
        class_inner_df = pd.concat(class_inner, ignore_index=True)
        class_metric_df = classification_metrics(class_pred_df)
        class_pred_df.to_csv(out['class_predictions'], index=False)
        class_sel_df.to_csv(out['class_selected'], index=False)
        class_inner_df.to_csv(out['class_inner'], index=False)
        class_metric_df.to_csv(out['class_metrics'], index=False)
        note('Hashing secondary_models cache inventory.')
        cache_inv = cache_inventory(cache)
        cache_inv.to_csv(out['cache_inventory'], index=False)
        executed = sorted(set(reg_pred_df.required_model_id.unique()) | set(class_pred_df.model_id.unique()))
        if sorted(MANDATORY_MODELS) != executed:
            raise RuntimeError(f'Mandatory execution incomplete: {executed}')
        expected_reg_families = {'S1_seen_single_epoch_full_signal': 3, 'S2_seen_pair_epoch_full_signal': 3, 'P2_person_specific_compact_all3': int(meta_all_seen.subject_id.nunique()), 'U1_unseen_person_universal_all3': 1, 'E1_liking_given_valence': 1}
        observed_datasets = reg_pred_df[reg_pred_df.model_id.eq(reg_pred_df.required_model_id)].groupby('required_model_id').dataset_id.nunique().to_dict()
        for model_id, expected in expected_reg_families.items():
            if int(observed_datasets.get(model_id, 0)) != expected:
                raise RuntimeError(f'Incomplete dataset execution for {model_id}: {observed_datasets.get(model_id)} != {expected}')
        decision = {'all_mandatory_secondary_models_executed': True, 'all_outer_predictions_complete': True, 'all_transforms_training_only': True, 'outer_test_family_or_epoch_selection_performed': False, 'primary_result_used_to_cancel_secondary_models': False, 'person_specific_support_decided': True, 'unseen_person_support_decided': True, 'identity_probe_complete': True, 'context_probe_complete': True, 'conditional_liking_model_complete': True, 'latent_state_models_fitted': False, 'validation_opened': False, 'confirmation_opened': False}
        summary = {'generated_at': now(), 'phase': 'secondary_models', 'status': 'all_prespecified_secondary_and_diagnostic_discovery_models_complete_candidate', 'execution_policy_sha256': POLICY_SHA, 'counts': {'feature_files_rehashed': len(feature_rehash), 'feature_bytes_rehashed': int(feature_rehash.size_bytes.sum()), 'primary_models_cache_files_rehashed': len(primary_models_rehash), 'primary_models_cache_bytes_rehashed': int(primary_models_rehash.size_bytes.sum()), 'epoch_store_rows': len(feature_rows), 'analysis_matrix_datasets': len(matrices), 'regression_outer_prediction_rows': len(reg_pred_df), 'regression_selected_hyperparameter_rows': len(reg_sel_df), 'regression_inner_score_rows': len(reg_inner_df), 'p2_subject_metric_rows': len(p2_subject_df), 'u1_bootstrap_replicates': BOOTSTRAPS, 'classification_prediction_rows': len(class_pred_df), 'classification_selected_hyperparameter_rows': len(class_sel_df), 'classification_inner_score_rows': len(class_inner_df), 'cache_files': len(cache_inv), 'cache_bytes': int(cache_inv.size_bytes.sum())}, 'mandatory_models': MANDATORY_MODELS, 'executed_models': executed, 'primary_support_decisions': contracts['primary_support'], 'person_specific_sign_tests': p2_test_df.to_dict(orient='records'), 'unseen_person_bootstrap_intervals': u1_interval_df.to_dict(orient='records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        out['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        output_hashes = {}
        for key, path in out.items():
            if key not in {'contract', 'next_policy', 'report', 'log'} and path.is_file():
                output_hashes[key] = sha(path)
        contract = {**summary, 'execution_policy': POLICY, 'cache_root': str(cache), 'large_cache_embedded_in_release': False, 'cache_reuse_policy': 'full SHA-256 verification of every checkpoint before reuse', 'input_hashes': {key: sha(path) for key, path in paths.items()}, 'output_hashes': output_hashes, 'feature_store_modified': False, 'primary_models_cache_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
        out['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        next_policy = {'generated_at': now(), 'phase': 'secondary_models', 'next_phase': 'latent_state_design', 'next_phase_permitted': True, 'next_phase_role': 'specification an outcome-free within-person latent-state and stimulus-to-rating trajectory analysis before fitting any latent-state model', 'required_inputs': {'secondary_contract': str(out['contract']), 'secondary_regression_predictions': str(out['reg_predictions']), 'diagnostic_classification_metrics': str(out['class_metrics']), 'epoch_store': str(epoch_store)}, 'required_design_elements': ['discover latent states within person from neurophysiological features without valence, liking, emotion labels, or other reports', 'represent stimulus_early to stimulus_late to rating_early trajectories on prespecified common trials', 'separate state discovery from post-hoc report/context mapping', 'predeclare participant-level state-count and stability rules', 'predeclare cross-person alignment as a later layer rather than a global preprocessing step', 'retain validation and confirmation closure'], 'forbidden': ['choosing state count from valence or liking association', 'using outer-test secondary_models results to select an epoch family', 'global outcome-informed alignment', 'opening validation or confirmation participants'], 'validation_opened': False, 'confirmation_opened': False}
        out['next_policy'].write_text(json.dumps(next_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(out['report'], summary, reg_metric_df, reg_contrast_df, p2_test_df, u1_interval_df, class_metric_df)
        note('secondary_models complete. All prespecified secondary and diagnostic families were executed.')
    print('' + '=' * 100)
    print('PHASE secondary_models SECONDARY MODELS COMPLETED')
    print('=' * 100)
    print(f'Mandatory models: {len(MANDATORY_MODELS)}/7')
    print(f'Regression prediction rows: {len(reg_pred_df)}')
    print(f'Classification prediction rows: {len(class_pred_df)}')
    print('Validation opened: False')
    print('Confirmation opened: False')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    raise SystemExit(main())