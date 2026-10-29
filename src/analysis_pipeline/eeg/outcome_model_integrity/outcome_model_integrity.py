# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase primary_model_design — feature-store integrity, discovery outcome attachment,
and nested model-analysis specification.

This phase performs three tasks before any EEG-outcome model is fitted:

1. Independently verify every outcome_free_features feature-store file by size and SHA-256,
   every source summary, every NPY header, and every feature array's missing-
   channel semantics and numerical domain.
2. Attach discovery-only reports to exact raw-behavior row IDs. Context must
   agree exactly. No validation or confirmation participant is opened.
3. specification analysis sets, targets, covariates, outer/inner folds, model families,
   hyperparameter grids, negative controls, inference, multiplicity, and claim
   gates. No model is fitted and no result-dependent choice is made.

Primary scientific test prespecified here
-----------------------------------
On the all-three-epoch common-row set, a pooled seen-person model predicts
person-relative valence and liking on contiguous held-out trials. The primary
full-signal model is compared on identical rows with:
- a covariate-only baseline; and
- a QC-only negative-control model.

A signal-specific within-person effect requires improvement over both controls.
Person-specific compact models, unseen-person universal models, single-epoch
ablations, pairwise epoch concatenations, identity probes, and context probes
are fixed secondary/diagnostic analyses.

Interpretation remains constrained:
- stimulus epochs index physiological responses during stimulus/context;
- rating epochs may index report-construction/cognitive state;
- EEG is not ground truth for latent experience;
- within-person predictability does not imply a universal biosignal;
- failed unseen-person generalization does not imply absent physiology."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from numpy.lib import format as npformat
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
outcome_free_features_SUMMARY_REL = Path('outputs/qc/outcome_free_features_outcome_free_feature_summary.json')
outcome_free_features_CONTRACT_REL = Path('config/outcome_free_features_outcome_free_feature_contract.json')
outcome_free_features_POLICY_REL = Path('config/outcome_free_features_prespecified_outcome_model_integrity_model_specification_policy.json')
FEATURE_INVENTORY_REL = Path('outputs/tables/outcome_free_features_feature_store_inventory.csv')
FEATURE_QC_REL = Path('outputs/qc/outcome_free_features_feature_extraction_qc.csv')
FEATURE_DIM_REL = Path('outputs/qc/outcome_free_features_feature_dimensions.csv')
COMMON_MASK_REL = Path('outputs/tables/large_store_integrity_common_trial_mask_manifest.csv')
COMMON_MASK_SUMMARY_REL = Path('outputs/qc/large_store_integrity_common_trial_mask_summary.csv')
TRIAL_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_trial_fold_manifest.csv')
PARTICIPANT_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_participant_fold_manifest.csv')
CORRECTED_MASTER_REL = Path('outputs/tables/derivative_closure_corrected_discovery_master_v1.csv')
EXPECTED = {'source_checkpoints': 57, 'source_family_groups': 171, 'feature_store_files': 1425, 'feature_store_bytes': 126050925, 'feature_epoch_rows': 3853, 'unique_feature_trials': 1301, 'common_trial_rows': 1379, 'discovery_subjects_with_features': 32}
FAMILIES = {'stimulus_early': {'time_frequency': (64, 6, 5), 'morphology': (64, 6), 'covariance': (64, 64), 'covariance_summary': (21,), 'plv_channel': (64, 6), 'plv_global': (6, 4), 'waveform_dct': (64, 20)}, 'stimulus_late': {'time_frequency': (64, 6, 5), 'morphology': (64, 6), 'covariance': (64, 64), 'covariance_summary': (21,), 'plv_channel': (64, 6), 'plv_global': (6, 4), 'waveform_dct': (64, 20)}, 'rating_early': {'time_frequency': (64, 6, 5), 'morphology': (64, 6), 'covariance': (64, 64), 'covariance_summary': (21,), 'plv_channel': (64, 6), 'plv_global': (6, 4), 'waveform_dct': (64, 20)}}
BLOCK_FILE_NAMES = {'time_frequency': 'time_frequency_float32.npy', 'morphology': 'morphology_float32.npy', 'covariance': 'covariance_float32.npy', 'covariance_summary': 'covariance_summary_float32.npy', 'plv_channel': 'plv_channel_float32.npy', 'plv_global': 'plv_global_float32.npy', 'waveform_dct': 'waveform_dct_float32.npy'}
CHANNELWISE_BLOCKS = ['time_frequency', 'morphology', 'plv_channel', 'waveform_dct']
GLOBAL_BLOCKS = ['covariance_summary', 'plv_global']
REPORT_COLUMNS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
PRIMARY_OUTCOMES = ['valence', 'liking']
ANALYSIS_SETS = {'stimulus_early': ['stimulus_early'], 'stimulus_late': ['stimulus_late'], 'rating_early': ['rating_early'], 'stimulus_early_vs_late': ['stimulus_early', 'stimulus_late'], 'stimulus_early_vs_rating': ['stimulus_early', 'rating_early'], 'stimulus_late_vs_rating': ['stimulus_late', 'rating_early'], 'all_three_epoch_families': ['stimulus_early', 'stimulus_late', 'rating_early']}
WITHIN_PERSON_MINIMUM_TOTAL = 16
WITHIN_PERSON_MINIMUM_PER_CONTEXT = 6
OUTER_FOLDS = 5
HASH_CHUNK_BYTES = 16 * 1024 * 1024
ARRAY_ROW_CHUNK = 16
SEED = 540
TARGET_POLICY = {'primary_outcomes': PRIMARY_OUTCOMES, 'observed_scale': 'original reported rating scale; no global transformation', 'seen_person_training_target': "within each outer fold and participant, subtract the participant's outer-training mean and divide by outer-training SD; SD below 1e-8 is fatal", 'prediction_reporting': 'back-transform predictions to the raw rating scale using outer-training participant statistics; also report person-relative standardized metrics', 'unseen_person_training_target': 'fit on raw outcomes with training-participant global standardization only; no test-participant outcome calibration or centering', 'exploratory_conditional_target': {'name': 'liking_given_valence', 'recipe': 'inside each outer-training fold, regress liking on valence and context; apply the prespecified training coefficients to train and test rows and model the residual. This is conditional association, not deployable liking prediction.'}, 'other_reports': 'joy, inspiration, tenderness, sadness, fear, disgust, arousal, and familiarity are attached for later triangulation only and are never predictors of valence/liking'}
PREPROCESSING_POLICY = {'primary_full_signal_vector': {'per_epoch_blocks': ['time_frequency_flat_1920', 'morphology_flat_384', 'covariance_upper_triangle_including_diagonal_2080', 'covariance_summary_21', 'plv_channel_flat_384', 'plv_global_flat_24', 'waveform_dct_flat_1280'], 'scalar_cells_per_epoch': 6093, 'missingness_indicators': False, 'training_only_steps': ['drop all-training-missing columns', 'median imputation fitted on outer-training rows', 'zero-variance filter fitted on outer-training rows', 'standardization fitted on outer-training rows', 'PCA fitted on outer-training rows']}, 'qc_only_vector': {'per_epoch': '64 binary channel-availability indicators plus final_good_channels', 'all_three_cells': 195, 'role': 'negative control; not part of the primary signal vector'}, 'compact_person_specific_vector': {'per_epoch': ['median and IQR across good channels for each of 30 time-frequency cells', 'median and IQR across good channels for each of 6 morphology cells', 'median and IQR across good channels for each of 6 channel-PLV cells', 'median and IQR across good channels for each of 20 DCT coefficients', '21 covariance summaries', '24 global PLV summaries'], 'scalar_cells_per_epoch': 169, 'all_three_cells': 507}, 'forbidden_global_steps': ['imputation', 'standardization', 'PCA', 'supervised feature selection', 'covariance tangent reference', 'CORAL', 'Procrustes', 'manifold alignment', 'outcome-based channel selection']}
HYPERPARAMETER_POLICY = {'ridge_alpha_grid': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], 'pooled_pca_components': [8, 16, 32, 64, 128], 'person_specific_pca_components': [2, 4, 8], 'logistic_c_grid': [0.01, 0.1, 1.0, 10.0], 'component_cap': 'min(candidate, number of retained features, outer-training rows minus 1)', 'inner_selection_metric': 'participant-equal-weighted person-relative mean squared error across inner validation rows', 'tie_break': 'within 1e-12 of the best inner score, choose fewer PCA components, then larger ridge alpha'}
INFERENCE_POLICY = {'primary_metric': 'pooled person-relative R2 = 1 - sum_i,t (y-pred)^2 / sum_i,t (y-mean_i)^2 on outer-fold predictions', 'secondary_metrics': ['participant-median R2', 'participant-median MAE', 'pooled raw-scale MAE', 'pooled Pearson correlation', 'pooled Spearman correlation'], 'bootstrap': {'unit': 'participant', 'replicates': 5000, 'seed': SEED, 'interval': 'percentile 95 percent'}, 'permutation': {'replicates': 1000, 'seed': SEED, 'recipe': 'within each participant and context, circularly shift the trial target by a nonzero offset of at least 3 rows; use the same shifted trial target for every epoch family'}, 'multiplicity': 'Holm correction across valence and liking for the primary full-signal-vs-baseline tests', 'primary_signal_support': ['lower 95 percent participant-bootstrap CI for full-signal minus covariate-baseline R2 is above zero', 'lower 95 percent participant-bootstrap CI for full-signal minus QC-only R2 is above zero', 'Holm-adjusted circular-shift permutation p is below 0.05'], 'person_specific_implementation_support': ['primary signal-support gate is met', 'compact person-specific models have median subject Delta-R2 above zero', 'two-sided exact sign test across subjects is below 0.05 after Holm across the two primary outcomes'], 'universal_decoding_support': 'unseen-person universal model has lower 95 percent participant-bootstrap R2 above zero; failure does not imply absent physiology'}
CLAIM_GUARDRAILS = ['EEG features are physiological measurements and not ground truth for latent experience', 'stimulus-epoch associations are not called a universal hedonic code', 'rating-epoch associations are described as compatible with report-construction or cognitive state', 'within-person prediction precedes person-specific implementation claims', 'identity decoding indicates separability, not emotion decoding', 'unseen-person failure is framed as nonseparability or misalignment, not EEG failure', 'no validation or confirmation participant is opened in outcome_model_integrity or analysis_stage']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify the complete outcome_free_features feature store, attach discovery outcomes, and specification nested model design without fitting models.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Required files are missing:\n  - ' + '\n  - '.join((str(path) for path in missing)))

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def safe_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def python_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    raise RuntimeError(f'Could not interpret boolean value: {value!r}')

def npy_header(path: Path) -> dict[str, Any]:
    with path.open('rb') as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = npformat.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f'Unsupported NPY version {version}: {path}')
    return {'version': '.'.join((str(value) for value in version)), 'shape': tuple((int(value) for value in shape)), 'fortran_order': bool(fortran), 'dtype': str(dtype), 'object_dtype': bool(dtype.hasobject)}

def load_inputs(project: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {'outcome_free_features_summary': project / outcome_free_features_SUMMARY_REL, 'outcome_free_features_contract': project / outcome_free_features_CONTRACT_REL, 'outcome_free_features_policy': project / outcome_free_features_POLICY_REL, 'feature_inventory': project / FEATURE_INVENTORY_REL, 'feature_qc': project / FEATURE_QC_REL, 'feature_dimensions': project / FEATURE_DIM_REL, 'common_masks': project / COMMON_MASK_REL, 'common_mask_summary': project / COMMON_MASK_SUMMARY_REL, 'trial_folds': project / TRIAL_FOLD_REL, 'participant_folds': project / PARTICIPANT_FOLD_REL, 'canonical_master': project / CORRECTED_MASTER_REL}
    require_files(paths.values())
    summary = read_json(paths['outcome_free_features_summary'])
    contract = read_json(paths['outcome_free_features_contract'])
    if summary.get('phase') != 'outcome_free_features':
        raise RuntimeError('Unexpected outcome_free_features prerequisite.')
    required_decisions = ['all_feature_checkpoints_complete', 'all_modeling_eligible_epochs_represented', 'rejected_channel_nan_policy_passed', 'outcome_model_integrity_discovery_model_specification_permitted']
    for key in required_decisions:
        if not bool(summary['decision'][key]):
            raise RuntimeError(f'outcome_free_features prerequisite failed: {key}.')
    if bool(summary['decision']['outcome_values_read']):
        raise RuntimeError('outcome_free_features unexpectedly read outcome values.')
    expected_output_map = {'feature_inventory': 'feature_inventory', 'feature_qc': 'feature_qc', 'feature_dimensions': 'feature_dimensions', 'outcome_free_features_summary': 'summary'}
    for path_key, hash_key in expected_output_map.items():
        observed = sha256_file(paths[path_key])
        expected = contract['output_hashes'][hash_key]
        if observed != expected:
            raise RuntimeError(f'{path_key}: file hash changed after outcome_free_features.')
    inventory = pd.read_csv(paths['feature_inventory'], low_memory=False)
    feature_qc = pd.read_csv(paths['feature_qc'], low_memory=False)
    dimensions = pd.read_csv(paths['feature_dimensions'], low_memory=False)
    common_masks = pd.read_csv(paths['common_masks'], low_memory=False)
    trial_folds = pd.read_csv(paths['trial_folds'], low_memory=False)
    participant_folds = pd.read_csv(paths['participant_folds'], low_memory=False)
    if len(inventory) != EXPECTED['feature_store_files']:
        raise RuntimeError('outcome_free_features feature-inventory row count changed.')
    if int(inventory['size_bytes'].sum()) != EXPECTED['feature_store_bytes']:
        raise RuntimeError('outcome_free_features feature-store byte count changed.')
    if inventory['stored_path'].duplicated().any():
        raise RuntimeError('Feature inventory contains duplicate stored paths.')
    if inventory['source_feature_directory'].nunique() != EXPECTED['source_checkpoints']:
        raise RuntimeError('outcome_free_features source-checkpoint count changed.')
    if len(feature_qc) != EXPECTED['source_family_groups']:
        raise RuntimeError('outcome_free_features source/family-QC row count changed.')
    if int(feature_qc['modeling_eligible_rows'].sum()) != EXPECTED['feature_epoch_rows']:
        raise RuntimeError('outcome_free_features represented epoch count changed.')
    if int(dimensions['scalar_cells_per_epoch'].sum()) != 8109:
        raise RuntimeError('outcome_free_features stored feature dimension changed.')
    if len(common_masks) != EXPECTED['common_trial_rows']:
        raise RuntimeError('raw_eeg_preprocessing_qc.1 common-trial row count changed.')
    return (paths, summary, contract, inventory, feature_qc, dimensions, common_masks, trial_folds, participant_folds)

def verify_feature_store_files(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    source_rows = []
    total = len(inventory)
    for position, row in inventory.iterrows():
        path = Path(str(row['stored_path']))
        if not path.is_file():
            raise RuntimeError(f'Feature-store file is missing: {path}')
        observed_size = int(path.stat().st_size)
        if observed_size != int(row['size_bytes']):
            raise RuntimeError(f'Feature-store size mismatch: {path}')
        observed_hash = sha256_file(path)
        if observed_hash != row['sha256']:
            raise RuntimeError(f'Feature-store SHA-256 mismatch: {path}')
        file_rows.append({'inventory_row_1based': int(position + 1), 'subject_id': row['subject_id'], 'raw_run': int(row['raw_run']), 'source_feature_directory': row['source_feature_directory'], 'relative_path': row['relative_path'], 'stored_path': str(path), 'expected_size_bytes': int(row['size_bytes']), 'observed_size_bytes': observed_size, 'expected_sha256': row['sha256'], 'observed_sha256': observed_hash, 'size_passed': True, 'sha256_passed': True})
        if (position + 1) % 100 == 0 or position + 1 == total:
            print(f'[{now_iso()}] Hashed {position + 1}/{total} feature-store files.', flush=True)
    for source_dir_text, group in inventory.groupby('source_feature_directory', sort=True):
        source_dir = Path(str(source_dir_text))
        summary_path = source_dir / 'source_feature_summary.json'
        if not summary_path.is_file():
            raise RuntimeError(f'Source feature summary is missing: {summary_path}')
        summary = read_json(summary_path)
        if summary.get('complete') is not True:
            raise RuntimeError(f'Incomplete feature checkpoint: {source_dir}')
        if summary.get('implementation_policy_sha256') != group['implementation_policy_sha256'].iloc[0]:
            raise RuntimeError(f'Implementation-policy mismatch: {source_dir}')
        if summary.get('input_signature_sha256') != group['input_signature_sha256'].iloc[0]:
            raise RuntimeError(f'Input-signature mismatch: {source_dir}')
        summary_files = {item['relative_path']: item for item in summary.get('stored_files', [])}
        inventory_files = {item['relative_path']: item for item in group.to_dict(orient='records')}
        if set(summary_files) != set(inventory_files):
            raise RuntimeError(f'Source summary/inventory file set differs: {source_dir}')
        for relative_path, item in summary_files.items():
            inventory_item = inventory_files[relative_path]
            if int(item['size_bytes']) != int(inventory_item['size_bytes']):
                raise RuntimeError(f'Source-summary size mismatch: {source_dir}/{relative_path}')
            if item['sha256'] != inventory_item['sha256']:
                raise RuntimeError(f'Source-summary hash mismatch: {source_dir}/{relative_path}')
        source_rows.append({'subject_id': summary['subject_id'], 'raw_run': int(summary['raw_run']), 'raw_source_path': summary['raw_source_path'], 'source_feature_directory': str(source_dir), 'stored_file_count': int(len(summary_files)), 'available_input_epochs': int(summary['available_input_epochs']), 'modeling_eligible_feature_rows': int(summary['modeling_eligible_feature_rows']), 'implementation_policy_sha256': summary['implementation_policy_sha256'], 'input_signature_sha256': summary['input_signature_sha256'], 'summary_file_set_passed': True, 'summary_hash_metadata_passed': True})
    source_frame = pd.DataFrame(source_rows)
    if len(source_frame) != EXPECTED['source_checkpoints']:
        raise RuntimeError('Feature source-summary count changed.')
    return (pd.DataFrame(file_rows), source_frame)

def inventory_file(group: pd.DataFrame, relative_path: str) -> Path:
    matches = group.loc[group['relative_path'].eq(relative_path), 'stored_path']
    if len(matches) != 1:
        raise RuntimeError(f'Expected one inventory path {relative_path}; observed {len(matches)}.')
    return Path(str(matches.iloc[0]))

def infer_channel_mask(values: np.ndarray, *, block_name: str) -> np.ndarray:
    if block_name == 'time_frequency':
        finite = np.isfinite(values).all(axis=(2, 3))
        all_nan = np.isnan(values).all(axis=(2, 3))
    elif block_name in {'morphology', 'plv_channel', 'waveform_dct'}:
        finite = np.isfinite(values).all(axis=2)
        all_nan = np.isnan(values).all(axis=2)
    else:
        raise RuntimeError(f'Cannot infer channel mask from {block_name}.')
    if not np.logical_xor(finite, all_nan).all():
        raise RuntimeError(f'{block_name}: partial NaN or infinite channel feature cells.')
    return finite

def verify_feature_array_semantics(inventory: pd.DataFrame, feature_qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    unified_indices = []
    for source_dir_text, source_group in inventory.groupby('source_feature_directory', sort=True):
        source_dir = Path(str(source_dir_text))
        source_qc = feature_qc.loc[feature_qc['raw_source_path'].eq(source_group['raw_source_path'].iloc[0])]
        for epoch_family, shape_map in FAMILIES.items():
            index_path = inventory_file(source_group, f'{epoch_family}_feature_index.csv')
            index = pd.read_csv(index_path, low_memory=False)
            expected_qc = source_qc.loc[source_qc['epoch_family'].eq(epoch_family)]
            if len(expected_qc) != 1:
                raise RuntimeError(f'Expected one source/family QC row: {source_dir}, {epoch_family}.')
            expected_rows = int(expected_qc['modeling_eligible_rows'].iloc[0])
            if len(index) != expected_rows:
                raise RuntimeError(f'Feature index row count mismatch: {source_dir}, {epoch_family}.')
            if 'epoch_family' in index.columns:
                if not index['epoch_family'].astype(str).eq(epoch_family).all():
                    raise RuntimeError(f'Feature-index epoch family mismatch: {index_path}')
            else:
                index['epoch_family'] = epoch_family
            if index['feature_row_0based'].astype(int).tolist() != list(range(len(index))):
                raise RuntimeError(f'Noncontiguous feature rows: {index_path}')
            if index['epoch_id'].duplicated().any():
                raise RuntimeError(f'Duplicate epoch IDs in feature index: {index_path}')
            paths = {block_name: inventory_file(source_group, f'{epoch_family}/{file_name}') for block_name, file_name in BLOCK_FILE_NAMES.items()}
            arrays = {}
            for block_name, trailing_shape in shape_map.items():
                header = npy_header(paths[block_name])
                expected_shape = (len(index), *trailing_shape)
                if header['shape'] != expected_shape:
                    raise RuntimeError(f"{paths[block_name]}: expected shape {expected_shape}, observed {header['shape']}.")
                if header['dtype'] != 'float32' or header['object_dtype']:
                    raise RuntimeError(f'{paths[block_name]}: expected non-object float32.')
                arrays[block_name] = np.load(paths[block_name], mmap_mode='r', allow_pickle=False)
            minimum_good = math.inf
            maximum_good = -math.inf
            covariance_symmetry_max = 0.0
            plv_min = math.inf
            plv_max = -math.inf
            entropy_min = math.inf
            entropy_max = -math.inf
            for start in range(0, len(index), ARRAY_ROW_CHUNK):
                stop = min(start + ARRAY_ROW_CHUNK, len(index))
                chunks = {key: np.asarray(value[start:stop]) for key, value in arrays.items()}
                reference_mask = infer_channel_mask(chunks['time_frequency'], block_name='time_frequency')
                for block_name in ['morphology', 'plv_channel', 'waveform_dct']:
                    block_mask = infer_channel_mask(chunks[block_name], block_name=block_name)
                    if not np.array_equal(reference_mask, block_mask):
                        raise RuntimeError(f'Channel-missingness patterns differ in {source_dir}, {epoch_family}, {block_name}.')
                good_counts = reference_mask.sum(axis=1).astype(int)
                if len(good_counts):
                    minimum_good = min(minimum_good, int(good_counts.min()))
                    maximum_good = max(maximum_good, int(good_counts.max()))
                if (good_counts < 48).any():
                    raise RuntimeError(f'A represented feature row has fewer than 48 channels: {source_dir}, {epoch_family}.')
                covariance = chunks['covariance']
                covariance_summary = chunks['covariance_summary']
                plv_channel = chunks['plv_channel']
                plv_global = chunks['plv_global']
                morphology = chunks['morphology']
                if not np.isfinite(covariance_summary).all():
                    raise RuntimeError(f'Nonfinite covariance summary: {source_dir}, {epoch_family}.')
                if not np.isfinite(plv_global).all():
                    raise RuntimeError(f'Nonfinite global PLV: {source_dir}, {epoch_family}.')
                finite_plv = np.concatenate([plv_channel[np.isfinite(plv_channel)], plv_global[np.isfinite(plv_global)]])
                if finite_plv.size:
                    plv_min = min(plv_min, float(finite_plv.min()))
                    plv_max = max(plv_max, float(finite_plv.max()))
                    if finite_plv.min() < -1e-06 or finite_plv.max() > 1.0 + 1e-06:
                        raise RuntimeError(f'PLV outside [0,1]: {source_dir}, {epoch_family}.')
                finite_entropy = morphology[:, :, 5][np.isfinite(morphology[:, :, 5])]
                if finite_entropy.size:
                    entropy_min = min(entropy_min, float(finite_entropy.min()))
                    entropy_max = max(entropy_max, float(finite_entropy.max()))
                    if finite_entropy.min() < -1e-06 or finite_entropy.max() > 1.0 + 1e-06:
                        raise RuntimeError(f'Spectral entropy outside [0,1]: {source_dir}, {epoch_family}.')
                normalized_eigenvalues = covariance_summary[:, :16]
                if (normalized_eigenvalues < -1e-06).any():
                    raise RuntimeError(f'Negative normalized covariance eigenvalue: {source_dir}, {epoch_family}.')
                if (normalized_eigenvalues.sum(axis=1) > 1.0 + 0.0001).any():
                    raise RuntimeError(f'Top normalized covariance eigenvalues sum above one: {source_dir}, {epoch_family}.')
                effective_rank = covariance_summary[:, 16]
                participation = covariance_summary[:, 17]
                if (effective_rank < 1.0 - 0.0001).any() or (effective_rank > 64.0 + 0.0001).any() or (participation < 1.0 - 0.0001).any() or (participation > 64.0 + 0.0001).any():
                    raise RuntimeError(f'Covariance rank summary outside [1,64]: {source_dir}, {epoch_family}.')
                if (covariance_summary[:, 19:] < -1e-06).any():
                    raise RuntimeError(f'Negative GFP summary: {source_dir}, {epoch_family}.')
                for row_offset in range(stop - start):
                    good = reference_mask[row_offset]
                    matrix = covariance[row_offset]
                    good_matrix = matrix[np.ix_(good, good)]
                    if not np.isfinite(good_matrix).all():
                        raise RuntimeError(f'Nonfinite good-channel covariance: {source_dir}, {epoch_family}.')
                    if not np.isnan(matrix[~good, :]).all() or not np.isnan(matrix[:, ~good]).all():
                        raise RuntimeError(f'Rejected covariance row/column is not all NaN: {source_dir}, {epoch_family}.')
                    symmetry = float(np.max(np.abs(good_matrix - good_matrix.T)))
                    covariance_symmetry_max = max(covariance_symmetry_max, symmetry)
                    if symmetry > 0.001:
                        raise RuntimeError(f'Covariance asymmetry exceeds tolerance: {source_dir}, {epoch_family}, {symmetry}.')
                    if (np.diag(good_matrix) < -1e-06).any():
                        raise RuntimeError(f'Negative covariance diagonal: {source_dir}, {epoch_family}.')
            if len(index) == 0:
                minimum_good_value = np.nan
                maximum_good_value = np.nan
                plv_min_value = np.nan
                plv_max_value = np.nan
                entropy_min_value = np.nan
                entropy_max_value = np.nan
            else:
                minimum_good_value = int(minimum_good)
                maximum_good_value = int(maximum_good)
                plv_min_value = float(plv_min)
                plv_max_value = float(plv_max)
                entropy_min_value = float(entropy_min)
                entropy_max_value = float(entropy_max)
            locator_columns = {f'{block_name}_path': str(path) for block_name, path in paths.items()}
            index = index.copy()
            index['source_feature_directory'] = str(source_dir)
            for column, value in locator_columns.items():
                index[column] = value
            unified_indices.append(index)
            group_rows.append({'subject_id': source_group['subject_id'].iloc[0], 'raw_run': int(source_group['raw_run'].iloc[0]), 'raw_source_path': source_group['raw_source_path'].iloc[0], 'source_feature_directory': str(source_dir), 'epoch_family': epoch_family, 'feature_rows': int(len(index)), 'all_headers_float32_and_exact_shape': True, 'channel_missingness_consistent': True, 'minimum_good_channels': minimum_good_value, 'maximum_good_channels': maximum_good_value, 'covariance_symmetry_max_abs': covariance_symmetry_max, 'plv_min': plv_min_value, 'plv_max': plv_max_value, 'spectral_entropy_min': entropy_min_value, 'spectral_entropy_max': entropy_max_value, 'numeric_domain_passed': True})
    unified = pd.concat(unified_indices, ignore_index=True, sort=False)
    groups = pd.DataFrame(group_rows)
    if len(groups) != EXPECTED['source_family_groups']:
        raise RuntimeError('Feature semantic group count changed.')
    if len(unified) != EXPECTED['feature_epoch_rows']:
        raise RuntimeError('Unified feature-index row count changed.')
    if unified['epoch_id'].duplicated().any():
        raise RuntimeError('Unified feature index contains duplicate epoch IDs.')
    if unified['raw_behavior_row_id'].nunique() != EXPECTED['unique_feature_trials']:
        raise RuntimeError('Unique feature-trial count changed.')
    if unified['subject_id'].nunique() != EXPECTED['discovery_subjects_with_features']:
        raise RuntimeError('Feature-subject count changed.')
    return (groups, unified.sort_values(['subject_id', 'subject_acquisition_order_1based', 'epoch_family']).reset_index(drop=True))

def attach_discovery_outcomes(project: Path, unified_features: pd.DataFrame, common_masks: pd.DataFrame, trial_folds: pd.DataFrame, participant_folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_columns = ['raw_behavior_row_id', 'subject_id', 'split', 'trial_number', 'context', 'report_source_attachment_exact', 'report_complete', *REPORT_COLUMNS]
    master = pd.read_csv(project / CORRECTED_MASTER_REL, usecols=use_columns, low_memory=False)
    if master['raw_behavior_row_id'].duplicated().any():
        raise RuntimeError('Corrected master has duplicate raw-behavior row IDs.')
    feature_trial_context = unified_features[['raw_behavior_row_id', 'subject_id', 'trial_number', 'context']].drop_duplicates()
    if feature_trial_context['raw_behavior_row_id'].duplicated().any():
        raise RuntimeError('A feature trial has conflicting subject/trial/context metadata.')
    trial_outcomes = feature_trial_context.merge(master, on=['raw_behavior_row_id', 'subject_id', 'trial_number', 'context'], how='left', validate='one_to_one')
    if len(trial_outcomes) != EXPECTED['unique_feature_trials']:
        raise RuntimeError('Discovery target attachment changed the trial count.')
    if not trial_outcomes['split'].eq('discovery').all():
        raise RuntimeError('A non-discovery row entered outcome attachment.')
    if not trial_outcomes['report_source_attachment_exact'].map(python_bool).all():
        raise RuntimeError('A feature row lacks exact raw-report attachment.')
    if not trial_outcomes['report_complete'].map(python_bool).all():
        raise RuntimeError('A feature trial has an incomplete report vector.')
    if trial_outcomes[REPORT_COLUMNS].isna().any().any():
        raise RuntimeError('A feature trial has a missing attached report.')
    trial_outcomes = trial_outcomes.merge(common_masks, on=['raw_behavior_row_id', 'subject_id', 'trial_number', 'context'], how='left', validate='one_to_one', suffixes=('', '_common'))
    trial_outcomes = trial_outcomes.merge(trial_folds[['raw_behavior_row_id', 'within_subject_block_fold', 'cross_person_fold', 'subject_acquisition_order_1based']], on='raw_behavior_row_id', how='left', validate='one_to_one', suffixes=('', '_fold'))
    trial_outcomes = trial_outcomes.merge(participant_folds[['subject_id', 'cross_person_fold']].rename(columns={'cross_person_fold': 'participant_cross_person_fold_check'}), on='subject_id', how='left', validate='many_to_one')
    if not trial_outcomes['cross_person_fold'].eq(trial_outcomes['participant_cross_person_fold_check']).all():
        raise RuntimeError('Trial and participant cross-person folds differ.')
    trial_outcomes = trial_outcomes.drop(columns=['participant_cross_person_fold_check'])
    subject_trial_counts = trial_outcomes.groupby('subject_id')['raw_behavior_row_id'].transform('size')
    trial_outcomes['acquisition_fraction'] = trial_outcomes['subject_acquisition_order_1based'].astype(float) / subject_trial_counts.astype(float)
    feature_outcomes = unified_features.merge(trial_outcomes, on=['raw_behavior_row_id', 'subject_id', 'trial_number', 'context'], how='left', validate='many_to_one', suffixes=('', '_trial'))
    if len(feature_outcomes) != EXPECTED['feature_epoch_rows']:
        raise RuntimeError('Feature-outcome attachment changed the feature row count.')
    if feature_outcomes[PRIMARY_OUTCOMES].isna().any().any():
        raise RuntimeError('A primary outcome is missing after feature attachment.')
    return (trial_outcomes.sort_values(['subject_id', 'subject_acquisition_order_1based']).reset_index(drop=True), feature_outcomes.sort_values(['subject_id', 'subject_acquisition_order_1based', 'epoch_family']).reset_index(drop=True))

def build_analysis_sets(trial_outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows = []
    subject_rows = []
    summary_rows = []
    for analysis_set, required_families in ANALYSIS_SETS.items():
        if analysis_set not in trial_outcomes.columns:
            raise RuntimeError(f'Common-mask column is missing: {analysis_set}.')
        selected = trial_outcomes.loc[trial_outcomes[analysis_set].map(python_bool)].copy()
        subject_count_frame = selected.groupby('subject_id', sort=True).agg(rows=('raw_behavior_row_id', 'size'), imagery_rows=('context', lambda values: int(pd.Series(values).eq('imagery').sum())), video_rows=('context', lambda values: int(pd.Series(values).eq('video').sum()))).reset_index()
        subject_count_frame['within_person_eligible'] = subject_count_frame['rows'].ge(WITHIN_PERSON_MINIMUM_TOTAL) & subject_count_frame['imagery_rows'].ge(WITHIN_PERSON_MINIMUM_PER_CONTEXT) & subject_count_frame['video_rows'].ge(WITHIN_PERSON_MINIMUM_PER_CONTEXT)
        subject_count_frame['analysis_set'] = analysis_set
        subject_rows.append(subject_count_frame)
        eligibility = subject_count_frame.set_index('subject_id')['within_person_eligible'].to_dict()
        for _, row in selected.iterrows():
            long_rows.append({'analysis_set': analysis_set, 'required_epoch_families': '|'.join(required_families), 'raw_behavior_row_id': row['raw_behavior_row_id'], 'subject_id': row['subject_id'], 'trial_number': int(row['trial_number']), 'context': row['context'], 'within_subject_block_fold': int(row['within_subject_block_fold']), 'cross_person_fold': int(row['cross_person_fold']), 'subject_acquisition_order_1based': int(row['subject_acquisition_order_1based']), 'within_person_eligible': bool(eligibility[row['subject_id']]), 'cross_person_eligible': True, 'same_evaluation_rows_required': True})
        summary_rows.append({'analysis_set': analysis_set, 'required_epoch_families': '|'.join(required_families), 'trial_rows': int(len(selected)), 'subjects_with_rows': int(selected['subject_id'].nunique()), 'within_person_eligible_subjects': int(subject_count_frame['within_person_eligible'].sum()), 'within_person_eligible_rows': int(selected['subject_id'].map(eligibility).sum()), 'outer_within_person_folds': OUTER_FOLDS, 'outer_cross_person_folds': OUTER_FOLDS})
    long_frame = pd.DataFrame(long_rows)
    subject_frame = pd.concat(subject_rows, ignore_index=True, sort=False)
    summary_frame = pd.DataFrame(summary_rows)
    primary = summary_frame.loc[summary_frame['analysis_set'].eq('all_three_epoch_families')]
    if len(primary) != 1:
        raise RuntimeError('Primary all-three analysis-set summary is missing.')
    if int(primary['trial_rows'].iloc[0]) != 1252:
        raise RuntimeError('All-three common-row count changed.')
    if int(primary['within_person_eligible_subjects'].iloc[0]) != 31:
        raise RuntimeError('All-three within-person subject count changed.')
    for analysis_set, group in long_frame.groupby('analysis_set'):
        if not set(group['within_subject_block_fold'].unique()).issubset(set(range(1, OUTER_FOLDS + 1))):
            raise RuntimeError(f'Invalid within-person fold in {analysis_set}.')
        if not set(group['cross_person_fold'].unique()).issubset(set(range(1, OUTER_FOLDS + 1))):
            raise RuntimeError(f'Invalid cross-person fold in {analysis_set}.')
    return (long_frame.sort_values(['analysis_set', 'subject_id', 'subject_acquisition_order_1based']).reset_index(drop=True), subject_frame.sort_values(['analysis_set', 'subject_id']).reset_index(drop=True), summary_frame.sort_values('analysis_set').reset_index(drop=True))

def model_specification_table() -> pd.DataFrame:
    rows = [{'model_id': 'B0_seen_covariate_baseline', 'role': 'primary_baseline', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'within_person_seen_pooled', 'target': 'valence|liking', 'feature_representation': 'none', 'covariates': 'subject_fixed_effect|context|raw_run|acquisition_fraction|acquisition_fraction_squared', 'estimator': 'ridge', 'primary_comparison': True}, {'model_id': 'B1_seen_qc_only', 'role': 'primary_negative_control', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'within_person_seen_pooled', 'target': 'valence|liking', 'feature_representation': 'qc_only_vector_all_three_195', 'covariates': 'same_as_B0', 'estimator': 'ridge', 'primary_comparison': True}, {'model_id': 'P1_seen_all3_full_signal', 'role': 'primary_full_signal', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'within_person_seen_pooled', 'target': 'valence|liking', 'feature_representation': 'full_signal_vector_6093_per_epoch_concatenated_across_three_epochs', 'covariates': 'same_as_B0', 'estimator': 'PCA_plus_ridge', 'primary_comparison': True}, {'model_id': 'S1_seen_single_epoch_full_signal', 'role': 'secondary_epoch_localization', 'analysis_set': 'stimulus_early|stimulus_late|rating_early', 'outer_scheme': 'within_person_seen_pooled', 'target': 'valence|liking', 'feature_representation': 'full_signal_vector_6093', 'covariates': 'same_as_B0', 'estimator': 'PCA_plus_ridge', 'primary_comparison': False}, {'model_id': 'S2_seen_pair_epoch_full_signal', 'role': 'secondary_transition_comparison', 'analysis_set': 'stimulus_early_vs_late|stimulus_early_vs_rating|stimulus_late_vs_rating', 'outer_scheme': 'within_person_seen_pooled', 'target': 'valence|liking', 'feature_representation': 'concatenated_full_signal_vectors_on_prespecified_common_rows', 'covariates': 'same_as_B0', 'estimator': 'PCA_plus_ridge', 'primary_comparison': False}, {'model_id': 'P2_person_specific_compact_all3', 'role': 'key_person_specific_secondary', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'person_specific_contiguous_folds', 'target': 'valence|liking', 'feature_representation': 'compact_169_per_epoch_concatenated_to_507', 'covariates': 'context|raw_run|acquisition_fraction|acquisition_fraction_squared', 'estimator': 'PCA_plus_ridge_per_subject', 'primary_comparison': False}, {'model_id': 'U1_unseen_person_universal_all3', 'role': 'universal_decoding_secondary', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'cross_person_unseen', 'target': 'valence|liking', 'feature_representation': 'full_signal_vector_all_three', 'covariates': 'context|raw_run|acquisition_fraction|acquisition_fraction_squared', 'estimator': 'PCA_plus_ridge', 'primary_comparison': False}, {'model_id': 'D1_subject_identity_probe_all3', 'role': 'diagnostic_separability', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'within_person_contiguous_test_blocks_without_subject_covariate', 'target': 'subject_id', 'feature_representation': 'compact_signal_507|qc_only_195|combined', 'covariates': 'none', 'estimator': 'multinomial_logistic_regression', 'primary_comparison': False}, {'model_id': 'D2_context_probe', 'role': 'diagnostic_context_information', 'analysis_set': 'stimulus_early|stimulus_late|rating_early', 'outer_scheme': 'within_person_seen_pooled', 'target': 'context', 'feature_representation': 'compact_signal_169', 'covariates': 'subject_fixed_effect_excluded', 'estimator': 'logistic_regression', 'primary_comparison': False}, {'model_id': 'E1_liking_given_valence', 'role': 'exploratory_conditional_target', 'analysis_set': 'all_three_epoch_families', 'outer_scheme': 'within_person_seen_pooled', 'target': 'fold_contained_liking_residual_given_valence_and_context', 'feature_representation': 'full_signal_vector_all_three', 'covariates': 'same_as_B0_without_valence', 'estimator': 'PCA_plus_ridge', 'primary_comparison': False}]
    return pd.DataFrame(rows)

def write_report(path: Path, summary: dict[str, Any], analysis_summary: pd.DataFrame, model_table: pd.DataFrame, semantic_groups: pd.DataFrame) -> None:
    lines = ['# Phase primary_model_design — Feature integrity, outcomes, and model specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Discovery outcomes opened: **yes, after feature-store integrity passed**', '- EEG-outcome model fitted: **no**', '- Validation/confirmation opened: **no**', '', '## Feature-store integrity', '', f"- Files hashed: **{summary['counts']['feature_files_hashed']}**", f"- Bytes hashed: **{summary['counts']['feature_bytes_hashed']:,}**", f"- Source summaries verified: **{summary['counts']['source_summaries_verified']}**", f'- Source/family arrays checked: **{len(semantic_groups)}**', f"- Feature rows checked: **{summary['counts']['feature_rows_semantically_checked']}**", '- Exact shapes and float32 dtype: **yes**', '- Missing-channel patterns consistent across blocks: **yes**', '- Covariance symmetry/domain checks: **yes**', '- PLV and entropy domain checks: **yes**', '', '## Exact discovery outcome attachment', '', f"- Unique trial rows with features: **{summary['counts']['feature_trial_rows']}**", f"- Epoch-family feature rows: **{summary['counts']['feature_outcome_rows']}**", f"- Subjects: **{summary['counts']['feature_subjects']}**", '- Missing valence: **0**', '- Missing liking: **0**', '- Context disagreement: **0**', '', '## prespecified analysis sets', '', '| Set | Required epochs | Trial rows | Subjects | Within-person eligible subjects | Eligible rows |', '|---|---|---:|---:|---:|---:|']
    for _, row in analysis_summary.iterrows():
        lines.append(f"| `{row['analysis_set']}` | `{row['required_epoch_families']}` | {int(row['trial_rows'])} | {int(row['subjects_with_rows'])} | {int(row['within_person_eligible_subjects'])} | {int(row['within_person_eligible_rows'])} |")
    lines.extend(['', '## Primary design', '', 'The fixed primary test uses `all_three_epoch_families` and the 31 participants meeting the preregistered-like minimum of 16 total, 6 imagery, and 6 video trials. Five contiguous within-person blocks are outer folds.', '', 'Primary outcomes are valence and liking. `P1_seen_all3_full_signal` must improve on identical rows over both the covariate baseline and the QC-only negative control. All preprocessing, PCA, and ridge tuning is nested inside outer-training data.', '', '## prespecified model families', '', '| Model | Role | Set | Outer scheme | Representation |', '|---|---|---|---|---|'])
    for _, row in model_table.iterrows():
        lines.append(f"| `{row['model_id']}` | {row['role']} | `{row['analysis_set']}` | `{row['outer_scheme']}` | {row['feature_representation']} |")
    decision = summary['decision']
    lines.extend(['', '## Decision', '', f"- Feature-store integrity passed: **{decision['feature_store_integrity_passed']}**", f"- Outcome attachment exact: **{decision['discovery_outcome_attachment_exact']}**", f"- Primary analysis rows prespecified: **{decision['primary_analysis_rows_prespecified']}**", f"- Nested model design prespecified: **{decision['nested_model_design_prespecified']}**", f"- Any model fitted: **{decision['any_model_fitted']}**", f"- Validation opened: **{decision['validation_opened']}**", f"- Confirmation opened: **{decision['confirmation_opened']}**", f"- analysis_stage model fitting permitted: **{decision['primary_models_nested_model_fitting_permitted']}**", '', 'The rating branch is not interpreted as direct access to latent experience. It may reflect report-construction or other cognitive state. EEG remains one physiological implementation layer rather than a universal hedonic ground truth.', ''])
    path.write_text(''.join(lines) + '', encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, outcome_free_features_summary, outcome_free_features_contract, inventory, feature_qc, dimensions, common_masks, trial_folds, participant_folds = load_inputs(project)
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config_dir = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'file_integrity': tables / 'outcome_model_integrity_feature_store_file_integrity.csv', 'source_integrity': qc / 'outcome_model_integrity_source_feature_checkpoint_integrity.csv', 'semantic_integrity': qc / 'outcome_model_integrity_feature_array_semantic_integrity.csv', 'feature_row_manifest': tables / 'outcome_model_integrity_unified_feature_row_manifest.csv', 'trial_outcomes': tables / 'outcome_model_integrity_discovery_trial_outcome_manifest.csv', 'feature_outcomes': tables / 'outcome_model_integrity_feature_outcome_attachment.csv', 'analysis_rows': tables / 'outcome_model_integrity_analysis_set_row_manifest.csv', 'analysis_subjects': qc / 'outcome_model_integrity_analysis_set_subject_eligibility.csv', 'analysis_summary': qc / 'outcome_model_integrity_analysis_set_summary.csv', 'model_table': config_dir / 'outcome_model_integrity_prespecified_model_specification.csv', 'target_policy': config_dir / 'outcome_model_integrity_prespecified_target_policy.json', 'model_policy': config_dir / 'outcome_model_integrity_prespecified_nested_model_policy.json', 'summary': qc / 'outcome_model_integrity_outcome_model_integrity_summary.json', 'contract': config_dir / 'outcome_model_integrity_outcome_model_integrity_contract.json', 'primary_models_policy': config_dir / 'outcome_model_integrity_prespecified_primary_models_execution_policy.json', 'report': reports / 'outcome_model_integrity_outcome_model_integrity_report.md', 'log': logs / 'outcome_model_integrity_outcome_model_integrity.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Outputs already exist. Use --force after review:\n  - ' + '\n  - '.join((str(path) for path in existing)))
    with outputs['log'].open('w', encoding='utf-8') as log:

        def note(message: str) -> None:
            line = f'[{now_iso()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Hashing the complete outcome_free_features feature store and verifying every source summary.')
        file_integrity, source_integrity = verify_feature_store_files(inventory)
        file_integrity.to_csv(outputs['file_integrity'], index=False)
        source_integrity.to_csv(outputs['source_integrity'], index=False)
        note('Scanning all feature arrays for exact shape, dtype, missing-channel consistency, and numerical domains.')
        semantic_integrity, unified_features = verify_feature_array_semantics(inventory, feature_qc)
        semantic_integrity.to_csv(outputs['semantic_integrity'], index=False)
        unified_features.to_csv(outputs['feature_row_manifest'], index=False)
        note('Feature integrity passed; attaching discovery-only reports by exact raw-behavior row ID.')
        trial_outcomes, feature_outcomes = attach_discovery_outcomes(project, unified_features, common_masks, trial_folds, participant_folds)
        trial_outcomes.to_csv(outputs['trial_outcomes'], index=False)
        feature_outcomes.to_csv(outputs['feature_outcomes'], index=False)
        note('Freezing branch-specific, same-row, within-person, and unseen-person analysis sets.')
        analysis_rows, analysis_subjects, analysis_summary = build_analysis_sets(trial_outcomes)
        analysis_rows.to_csv(outputs['analysis_rows'], index=False)
        analysis_subjects.to_csv(outputs['analysis_subjects'], index=False)
        analysis_summary.to_csv(outputs['analysis_summary'], index=False)
        model_table = model_specification_table()
        model_table.to_csv(outputs['model_table'], index=False)
        outputs['target_policy'].write_text(json.dumps(TARGET_POLICY, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        nested_model_policy = {'generated_at': now_iso(), 'phase': 'primary_model_design', 'seed': SEED, 'primary_analysis_set': 'all_three_epoch_families', 'primary_models': ['B0_seen_covariate_baseline', 'B1_seen_qc_only', 'P1_seen_all3_full_signal'], 'primary_outcomes': PRIMARY_OUTCOMES, 'outer_schemes': {'within_person_seen_pooled': 'outer fold k tests within_subject_block_fold k for every eligible participant; all other rows are outer training', 'person_specific_contiguous_folds': "fit one model per participant; outer fold k is that participant's contiguous block k", 'cross_person_unseen': 'outer fold k tests every participant assigned cross_person_fold k'}, 'inner_schemes': {'within_person_seen_pooled': 'leave one of the four remaining within-subject block labels out for inner validation', 'person_specific_contiguous_folds': 'within each participant, leave one of the four remaining blocks out', 'cross_person_unseen': 'leave one of the four remaining participant folds out'}, 'baseline_covariates': {'seen_person': ['subject fixed effects fitted on outer training', 'context', 'raw_run', 'acquisition_fraction', 'acquisition_fraction_squared'], 'unseen_person': ['context', 'raw_run', 'acquisition_fraction', 'acquisition_fraction_squared']}, 'preprocessing': PREPROCESSING_POLICY, 'hyperparameters': HYPERPARAMETER_POLICY, 'inference': INFERENCE_POLICY, 'claim_guardrails': CLAIM_GUARDRAILS, 'model_table': model_table.to_dict(orient='records'), 'forbidden': ['selecting an epoch family from outer-test performance', 'changing QC thresholds or trial masks', 'global imputation, scaling, PCA, tangent mapping, or alignment', 'test-participant outcome calibration', 'using other self-report variables as EEG predictors of valence or liking', 'opening validation or confirmation participants']}
        outputs['model_policy'].write_text(json.dumps(nested_model_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        primary_summary = analysis_summary.loc[analysis_summary['analysis_set'].eq('all_three_epoch_families')].iloc[0]
        decision = {'feature_store_byte_integrity_passed': True, 'feature_store_semantic_integrity_passed': True, 'feature_store_integrity_passed': True, 'discovery_outcome_attachment_exact': True, 'primary_analysis_rows_prespecified': True, 'nested_model_design_prespecified': True, 'global_transform_fitted': False, 'any_model_fitted': False, 'validation_opened': False, 'confirmation_opened': False, 'primary_models_nested_model_fitting_permitted': True}
        summary = {'generated_at': now_iso(), 'phase': 'primary_model_design', 'status': 'feature_store_integrity_discovery_outcome_attachment_and_nested_model_specification_complete_candidate', 'counts': {'feature_files_hashed': int(len(file_integrity)), 'feature_bytes_hashed': int(file_integrity['observed_size_bytes'].sum()), 'source_summaries_verified': int(len(source_integrity)), 'source_family_groups_checked': int(len(semantic_integrity)), 'feature_rows_semantically_checked': int(len(unified_features)), 'feature_trial_rows': int(len(trial_outcomes)), 'feature_outcome_rows': int(len(feature_outcomes)), 'feature_subjects': int(trial_outcomes['subject_id'].nunique()), 'primary_all_three_trial_rows': int(primary_summary['trial_rows']), 'primary_within_person_eligible_subjects': int(primary_summary['within_person_eligible_subjects']), 'primary_within_person_eligible_rows': int(primary_summary['within_person_eligible_rows']), 'prespecified_model_rows': int(len(model_table))}, 'primary_design': {'analysis_set': 'all_three_epoch_families', 'outcomes': PRIMARY_OUTCOMES, 'outer_scheme': 'within_person_seen_pooled', 'baseline_model': 'B0_seen_covariate_baseline', 'qc_negative_control': 'B1_seen_qc_only', 'full_signal_model': 'P1_seen_all3_full_signal'}, 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        contract = {**summary, 'target_policy_sha256': canonical_sha256(TARGET_POLICY), 'preprocessing_policy_sha256': canonical_sha256(PREPROCESSING_POLICY), 'hyperparameter_policy_sha256': canonical_sha256(HYPERPARAMETER_POLICY), 'inference_policy_sha256': canonical_sha256(INFERENCE_POLICY), 'nested_model_policy_sha256': sha256_file(outputs['model_policy']), 'input_hashes': {name: sha256_file(path) for name, path in paths.items()}, 'output_hashes': {name: sha256_file(path) for name, path in outputs.items() if name not in {'contract', 'primary_models_policy', 'report', 'log'} and path.is_file()}, 'source_dataset_modified': False, 'raw_eeg_preprocessing_qc_large_store_modified': False, 'outcome_free_features_feature_store_modified': False, 'canonical_master_modified': False}
        outputs['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        primary_models_policy = {'generated_at': now_iso(), 'phase': 'primary_model_design', 'next_phase': 'primary_inference_design', 'next_phase_permitted': True, 'integrity_contract': str(outputs['contract']), 'feature_row_manifest': str(outputs['feature_row_manifest']), 'trial_outcome_manifest': str(outputs['trial_outcomes']), 'feature_outcome_attachment': str(outputs['feature_outcomes']), 'analysis_set_rows': str(outputs['analysis_rows']), 'analysis_set_subjects': str(outputs['analysis_subjects']), 'model_specification': str(outputs['model_table']), 'target_policy': str(outputs['target_policy']), 'nested_model_policy': str(outputs['model_policy']), 'next_phase_role': 'fit the fully prespecified discovery-only nested models and emit outer-fold predictions; no model-family or epoch selection is permitted from outer-test results', 'validation_opened': False, 'confirmation_opened': False}
        outputs['primary_models_policy'].write_text(json.dumps(primary_models_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(outputs['report'], summary, analysis_summary, model_table, semantic_integrity)
        note('outcome_model_integrity completed. Discovery outcomes were attached only after full feature integrity passed; no model was fitted.')
    print()
    print('=' * 100)
    print('PHASE primary_model_design INTEGRITY / OUTCOME / MODEL specification COMPLETED')
    print('=' * 100)
    print(f'Feature files hashed:           {len(file_integrity)}')
    print(f"Feature bytes hashed:           {int(file_integrity['observed_size_bytes'].sum())}")
    print(f'Feature rows checked:           {len(unified_features)}')
    print(f'Unique feature trials:          {len(trial_outcomes)}')
    print(f"Primary all-three rows:         {int(primary_summary['trial_rows'])}")
    print(f"Primary eligible subjects:      {int(primary_summary['within_person_eligible_subjects'])}")
    print('Any model fitted:                False')
    print(f"Report:                          {outputs['report']}")
    print(f'Runtime:                         {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())