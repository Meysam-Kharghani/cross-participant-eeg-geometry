# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase large_store_integrity — large-store integrity, QC forensics, and discovery_multimodal_inference specification.

Why this phase is required
--------------------------
3.48-GB epoch arrays stored locally. Before scientific signal analysis, the
local store must be checked byte-for-byte and semantically:

- every inventory path exists;
- every stored size and SHA-256 matches;
- every source summary agrees with the phase inventory;
- every NPY header has the prespecified dtype and shape;
- every index row maps exactly to the raw_eeg_preprocessing_qc epoch-QC row;
- every mask-true channel contains only finite samples;
- every mask-false channel contains only NaN samples;
- mask counts equal final_good_channels;
- modeling_eligible is exactly the prespecified recording/epoch gate.

This phase also formalizes the recording failures and specifications common trial
sets for fair cross-epoch comparisons.

No threshold, epoch window, endpoint, fold, channel mask, or eligibility
decision is changed.

discovery_multimodal_inference representation policy
--------------------------
The next phase extracts outcome-free feature blocks only. No scaling,
imputation, PCA, covariance reference, tangent map, feature selection, or
manifold alignment may be fitted globally; those transforms belong inside
the already prespecified outer folds.

prespecified feature blocks:
1. channel time-frequency log power:
   six bands × whole epoch plus four equal temporal quarters × 64 channels;
2. channel temporal morphology:
   robust RMS, line length, Hjorth mobility, Hjorth complexity, lag-1
   autocorrelation, and spectral entropy;
3. spatial covariance:
   64×64 shrinkage covariance with NaN rows/columns for rejected channels,
   plus outcome-free eigen-spectrum summaries;
4. phase synchrony:
   channel-wise mean PLV and global PLV summaries in the six prespecified bands;
5. low-frequency waveform shape:
   first 20 orthonormal DCT coefficients per channel after fixed 50-Hz
   anti-aliased resampling.

Bad-channel-derived feature cells remain NaN. Imputation and standardization
are fold-contained in later modeling.

No behavioral value, outcome, released derivative amplitude, validation, or
confirmation data is read."""
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
raw_eeg_preprocessing_qc_SUMMARY_REL = Path('outputs/qc/raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_summary.json')
raw_eeg_preprocessing_qc_CONTRACT_REL = Path('config/raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_contract.json')
raw_eeg_preprocessing_qc_POLICY_REL = Path('analysis_specifications/S058_raw_eeg_preprocessing_qc_discovery_multimodal.json')
RECORDING_QC_REL = Path('outputs/qc/raw_eeg_preprocessing_qc_recording_channel_qc.csv')
EPOCH_QC_REL = Path('outputs/tables/raw_eeg_preprocessing_qc_epoch_amplitude_qc.csv')
ARRAY_INVENTORY_REL = Path('outputs/tables/raw_eeg_preprocessing_qc_large_array_inventory.csv')
BRANCH_QC_REL = Path('outputs/qc/raw_eeg_preprocessing_qc_branch_retention_qc.csv')
SUBJECT_QC_REL = Path('outputs/qc/raw_eeg_preprocessing_qc_subject_retention_qc.csv')
raw_eeg_epoch_folds_EPOCH_REL = Path('outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv')
raw_eeg_epoch_folds_TRIAL_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_trial_fold_manifest.csv')
raw_eeg_epoch_folds_PARTICIPANT_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_participant_fold_manifest.csv')
EXPECTED_COUNTS = {'sources': 57, 'inventory_rows': 627, 'recording_qc_rows': 3648, 'raw_eeg_epoch_folds_available_epochs': 4083, 'modeling_eligible_epochs': 3853, 'authoritative_trials': 1379}
CHANNEL_COUNT = 64
MINIMUM_GOOD_CHANNELS = 48
SUBJECT_MINIMUM_TOTAL = 16
SUBJECT_MINIMUM_PER_CONTEXT = 6
HASH_CHUNK_BYTES = 16 * 1024 * 1024
ARRAY_ROW_CHUNK = 8
FAMILIES = {'stimulus_early': 2000, 'stimulus_late': 4000, 'rating_early': 4000}
COMPARISON_MASKS = {'stimulus_early': ['stimulus_early'], 'stimulus_late': ['stimulus_late'], 'rating_early': ['rating_early'], 'stimulus_early_vs_late': ['stimulus_early', 'stimulus_late'], 'stimulus_early_vs_rating': ['stimulus_early', 'rating_early'], 'stimulus_late_vs_rating': ['stimulus_late', 'rating_early'], 'all_three_epoch_families': ['stimulus_early', 'stimulus_late', 'rating_early']}
FEATURE_POLICY = {'sampling_rate_hz': 200, 'frequency_bands_hz': {'delta': [1.0, 4.0], 'theta': [4.0, 8.0], 'alpha': [8.0, 13.0], 'low_beta': [13.0, 20.0], 'high_beta': [20.0, 30.0], 'low_gamma': [30.0, 45.0]}, 'time_frequency': {'estimator': 'scipy_signal_welch_hann', 'window_sec': 2.0, 'overlap_fraction': 0.5, 'aggregation': 'log10_integrated_bandpower', 'temporal_bins': ['whole_epoch', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4'], 'channel_layout': 64, 'bad_channel_cells': 'NaN'}, 'temporal_morphology': {'per_channel': ['robust_rms', 'line_length', 'hjorth_mobility', 'hjorth_complexity', 'lag1_autocorrelation', 'spectral_entropy'], 'bad_channel_cells': 'NaN'}, 'spatial_covariance': {'matrix_shape': [64, 64], 'estimator': 'fixed_diagonal_shrinkage_0.10', 'bad_channel_rows_columns': 'NaN', 'outcome_free_summaries': ['top_16_normalized_eigenvalues', 'effective_rank', 'participation_ratio', 'log_determinant_good_submatrix', 'global_field_power_median', 'global_field_power_iqr'], 'tangent_reference': 'not fitted during extraction; fitted inside each prespecified outer fold'}, 'phase_synchrony': {'estimator': 'zero_phase_bandpass_hilbert_plv', 'bands': 'same six prespecified frequency bands', 'stored_features': ['channel_mean_plv', 'global_mean_plv', 'global_sd_plv', 'global_q25_plv', 'global_q75_plv'], 'bad_channel_cells': 'NaN'}, 'waveform_shape': {'fixed_resampling_hz': 50, 'method': 'polyphase', 'transform': 'orthonormal_dct_type_2', 'coefficients_per_channel': 20, 'bad_channel_cells': 'NaN'}, 'forbidden_global_transforms': ['imputation', 'standardization', 'PCA', 'supervised feature selection', 'covariance reference', 'tangent-space map', 'CORAL or Procrustes alignment', 'outcome-based channel or feature selection'], 'later_modeling_requirement': 'all transforms above are fitted on outer-training rows only'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify the complete raw_eeg_preprocessing_qc large store, adjudicate QC failures, and specification discovery_multimodal_inference common-row and feature contracts.')
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
            raise RuntimeError(f'Unsupported NPY version: {version}.')
    return {'npy_version': '.'.join((str(value) for value in version)), 'shape': tuple((int(value) for value in shape)), 'fortran_order': bool(fortran), 'dtype': str(dtype), 'object_dtype': bool(dtype.hasobject)}

def load_inputs(project: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {'raw_eeg_preprocessing_qc_summary': project / raw_eeg_preprocessing_qc_SUMMARY_REL, 'raw_eeg_preprocessing_qc_contract': project / raw_eeg_preprocessing_qc_CONTRACT_REL, 'raw_eeg_preprocessing_qc_policy': project / raw_eeg_preprocessing_qc_POLICY_REL, 'recording_qc': project / RECORDING_QC_REL, 'epoch_qc': project / EPOCH_QC_REL, 'array_inventory': project / ARRAY_INVENTORY_REL, 'branch_qc': project / BRANCH_QC_REL, 'subject_qc': project / SUBJECT_QC_REL, 'raw_eeg_epoch_folds_epoch': project / raw_eeg_epoch_folds_EPOCH_REL, 'raw_eeg_epoch_folds_trial_folds': project / raw_eeg_epoch_folds_TRIAL_FOLD_REL, 'raw_eeg_epoch_folds_participant_folds': project / raw_eeg_epoch_folds_PARTICIPANT_FOLD_REL}
    require_files(paths.values())
    summary = read_json(paths['raw_eeg_preprocessing_qc_summary'])
    contract = read_json(paths['raw_eeg_preprocessing_qc_contract'])
    if summary.get('phase') != 'raw_eeg_preprocessing_qc':
        raise RuntimeError('Unexpected raw_eeg_preprocessing_qc prerequisite.')
    if not bool(summary['decision']['all_source_checkpoints_complete']):
        raise RuntimeError('raw_eeg_preprocessing_qc did not complete every source checkpoint.')
    if not bool(summary['decision']['all_three_branches_passed']):
        raise RuntimeError('raw_eeg_preprocessing_qc did not pass all three branch gates.')
    recording = pd.read_csv(paths['recording_qc'], low_memory=False)
    epoch = pd.read_csv(paths['epoch_qc'], low_memory=False)
    inventory = pd.read_csv(paths['array_inventory'], low_memory=False)
    branch = pd.read_csv(paths['branch_qc'], low_memory=False)
    subject = pd.read_csv(paths['subject_qc'], low_memory=False)
    raw_eeg_epoch_folds_epoch = pd.read_csv(paths['raw_eeg_epoch_folds_epoch'], low_memory=False)
    if len(recording) != EXPECTED_COUNTS['recording_qc_rows']:
        raise RuntimeError('Recording-QC row count changed.')
    if len(epoch) != EXPECTED_COUNTS['raw_eeg_epoch_folds_available_epochs']:
        raise RuntimeError('Epoch-QC row count changed.')
    if int(epoch['modeling_eligible'].map(python_bool).sum()) != EXPECTED_COUNTS['modeling_eligible_epochs']:
        raise RuntimeError('Modeling-eligible epoch count changed.')
    if len(inventory) != EXPECTED_COUNTS['inventory_rows']:
        raise RuntimeError('Large-array inventory row count changed.')
    return (paths, summary, contract, recording, epoch, inventory, branch, subject, raw_eeg_epoch_folds_epoch)

def verify_inventory_files(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory['stored_path'].duplicated().any():
        raise RuntimeError('Large-array inventory contains duplicate stored paths.')
    rows = []
    total = len(inventory)
    for position, row in inventory.iterrows():
        path = Path(str(row['stored_path']))
        if not path.is_file():
            raise RuntimeError(f'Inventory file is missing: {path}')
        observed_size = int(path.stat().st_size)
        if observed_size != int(row['size_bytes']):
            raise RuntimeError(f'Size mismatch: {path}')
        observed_hash = sha256_file(path)
        if observed_hash != row['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {path}')
        rows.append({'inventory_row_1based': int(position + 1), 'subject_id': row['subject_id'], 'raw_run': int(row['raw_run']), 'raw_source_path': row['raw_source_path'], 'source_store_directory': row['source_store_directory'], 'stored_path': str(path), 'relative_path': row['relative_path'], 'expected_size_bytes': int(row['size_bytes']), 'observed_size_bytes': observed_size, 'expected_sha256': row['sha256'], 'observed_sha256': observed_hash, 'size_passed': True, 'sha256_passed': True})
        if (position + 1) % 50 == 0 or position + 1 == total:
            print(f'[{now_iso()}] Hashed {position + 1}/{total} stored files.', flush=True)
    return pd.DataFrame(rows)

def verify_source_summaries(inventory: pd.DataFrame, *, policy_sha256: str, epoch_manifest_sha256: str) -> pd.DataFrame:
    rows = []
    for source_dir_text, group in inventory.groupby('source_store_directory', sort=True):
        source_dir = Path(str(source_dir_text))
        summary_path = source_dir / 'source_summary.json'
        if not summary_path.is_file():
            raise RuntimeError(f'Source summary is missing: {summary_path}')
        summary = read_json(summary_path)
        if summary.get('complete') is not True:
            raise RuntimeError(f'Source checkpoint is not complete: {source_dir}')
        if summary.get('policy_sha256') != policy_sha256:
            raise RuntimeError(f'Policy hash mismatch: {source_dir}')
        if summary.get('epoch_manifest_sha256') != epoch_manifest_sha256:
            raise RuntimeError(f'Epoch-manifest hash mismatch: {source_dir}')
        if summary.get('raw_source_path') != group['raw_source_path'].iloc[0]:
            raise RuntimeError(f'Raw source mismatch: {source_dir}')
        stored_summary = {item['relative_path']: item for item in summary.get('stored_files', [])}
        stored_inventory = {item['relative_path']: item for item in group.to_dict(orient='records')}
        if set(stored_summary) != set(stored_inventory):
            raise RuntimeError(f'Source summary/inventory file set differs: {source_dir}')
        for relative_path in stored_summary:
            left = stored_summary[relative_path]
            right = stored_inventory[relative_path]
            if int(left['size_bytes']) != int(right['size_bytes']):
                raise RuntimeError(f'Source-summary size mismatch: {source_dir}/{relative_path}')
            if left['sha256'] != right['sha256']:
                raise RuntimeError(f'Source-summary hash mismatch: {source_dir}/{relative_path}')
        raw_hashes = set(group['raw_source_sha256'].astype(str))
        if raw_hashes != {summary['raw_source_sha256']}:
            raise RuntimeError(f'Raw-source hash inventory mismatch: {source_dir}')
        rows.append({'subject_id': summary['subject_id'], 'raw_run': int(summary['raw_run']), 'raw_source_path': summary['raw_source_path'], 'source_store_directory': str(source_dir), 'stored_file_count': int(len(stored_summary)), 'available_epoch_rows': int(summary['available_epoch_rows']), 'modeling_eligible_epoch_rows': int(summary['modeling_eligible_epoch_rows']), 'recording_good_channels': int(summary['recording_good_channels']), 'recording_passed': bool(summary['recording_passed']), 'policy_sha256_passed': True, 'epoch_manifest_sha256_passed': True, 'stored_file_set_passed': True, 'stored_file_hash_metadata_passed': True, 'raw_source_hash_metadata_passed': True})
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_COUNTS['sources']:
        raise RuntimeError('Source-summary count changed.')
    return output

def file_for(group: pd.DataFrame, relative_path: str) -> Path:
    matches = group.loc[group['relative_path'].eq(relative_path), 'stored_path']
    if len(matches) != 1:
        raise RuntimeError(f'Expected one inventory file named {relative_path}; observed {len(matches)}.')
    return Path(str(matches.iloc[0]))

def verify_array_semantics(inventory: pd.DataFrame, epoch_qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source_dir, epoch_family), group in inventory.assign(epoch_family=inventory['relative_path'].str.extract('^(stimulus_early|stimulus_late|rating_early)_', expand=False)).dropna(subset=['epoch_family']).groupby(['source_store_directory', 'epoch_family'], sort=True):
        data_path = file_for(group, f'{epoch_family}_epochs_float32.npy')
        mask_path = file_for(group, f'{epoch_family}_channel_mask_bool.npy')
        index_path = file_for(group, f'{epoch_family}_index.csv')
        data_header = npy_header(data_path)
        mask_header = npy_header(mask_path)
        index = pd.read_csv(index_path, low_memory=False)
        expected_samples = FAMILIES[epoch_family]
        expected_data_shape = (len(index), CHANNEL_COUNT, expected_samples)
        expected_mask_shape = (len(index), CHANNEL_COUNT)
        if data_header['shape'] != expected_data_shape:
            raise RuntimeError(f"Data shape mismatch: {data_path}; expected {expected_data_shape}, observed {data_header['shape']}.")
        if mask_header['shape'] != expected_mask_shape:
            raise RuntimeError(f'Mask shape mismatch: {mask_path}.')
        if data_header['dtype'] != 'float32':
            raise RuntimeError(f'Data dtype mismatch: {data_path}.')
        if mask_header['dtype'] != 'bool':
            raise RuntimeError(f'Mask dtype mismatch: {mask_path}.')
        expected_order = list(range(len(index)))
        if index['source_array_row_0based'].astype(int).tolist() != expected_order:
            raise RuntimeError(f'Noncontiguous source-array rows: {index_path}.')
        source_path_values = index['raw_source_path'].astype(str).unique()
        if len(source_path_values) != 1:
            raise RuntimeError(f'Index contains multiple sources: {index_path}.')
        source_path = source_path_values[0]
        qc = epoch_qc.loc[epoch_qc['raw_source_path'].eq(source_path) & epoch_qc['epoch_family'].eq(epoch_family)].sort_values('source_array_row_0based').reset_index(drop=True)
        if len(qc) != len(index):
            raise RuntimeError(f'Index/QC row count mismatch: {source_path}, {epoch_family}.')
        if index['epoch_id'].astype(str).tolist() != qc['epoch_id'].astype(str).tolist():
            raise RuntimeError(f'Index/QC epoch-ID order mismatch: {source_path}, {epoch_family}.')
        data = np.load(data_path, mmap_mode='r', allow_pickle=False)
        mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
        observed_mask_counts = []
        finite_mask_pass = True
        nan_mask_pass = True
        for start in range(0, len(index), ARRAY_ROW_CHUNK):
            stop = min(start + ARRAY_ROW_CHUNK, len(index))
            data_chunk = np.asarray(data[start:stop])
            mask_chunk = np.asarray(mask[start:stop], dtype=bool)
            finite_all = np.isfinite(data_chunk).all(axis=2)
            nan_all = np.isnan(data_chunk).all(axis=2)
            if not np.array_equal(finite_all, mask_chunk):
                finite_mask_pass = False
            if not np.array_equal(nan_all, ~mask_chunk):
                nan_mask_pass = False
            observed_mask_counts.extend(mask_chunk.sum(axis=1).astype(int).tolist())
        if not finite_mask_pass:
            raise RuntimeError(f'Mask-true samples are not all finite: {data_path}.')
        if not nan_mask_pass:
            raise RuntimeError(f'Mask-false samples are not all NaN: {data_path}.')
        expected_counts = qc['final_good_channels'].astype(int).tolist()
        if observed_mask_counts != expected_counts:
            raise RuntimeError(f'Mask counts differ from epoch QC: {source_path}, {epoch_family}.')
        recomputed_modeling = qc['recording_passed'].map(python_bool) & (np.asarray(observed_mask_counts) >= MINIMUM_GOOD_CHANNELS)
        recorded_modeling = qc['modeling_eligible'].map(python_bool)
        if not np.array_equal(recomputed_modeling.to_numpy(dtype=bool), recorded_modeling.to_numpy(dtype=bool)):
            raise RuntimeError(f'Modeling eligibility mismatch: {source_path}, {epoch_family}.')
        rows.append({'subject_id': index['subject_id'].iloc[0], 'raw_run': int(index['raw_run'].iloc[0]), 'raw_source_path': source_path, 'source_store_directory': source_dir, 'epoch_family': epoch_family, 'index_rows': int(len(index)), 'data_shape': 'x'.join((str(value) for value in data_header['shape'])), 'mask_shape': 'x'.join((str(value) for value in mask_header['shape'])), 'data_dtype': data_header['dtype'], 'mask_dtype': mask_header['dtype'], 'index_epoch_order_passed': True, 'mask_true_all_finite': True, 'mask_false_all_nan': True, 'mask_count_matches_epoch_qc': True, 'modeling_eligibility_recomputed': True, 'minimum_good_channels': int(min(observed_mask_counts)), 'median_good_channels': float(np.median(observed_mask_counts)), 'maximum_good_channels': int(max(observed_mask_counts))})
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_COUNTS['sources'] * len(FAMILIES):
        raise RuntimeError('Expected three semantic array families per source.')
    if int(output['index_rows'].sum()) != EXPECTED_COUNTS['raw_eeg_epoch_folds_available_epochs']:
        raise RuntimeError('Semantic array rows do not sum to the raw_eeg_epoch_folds-available count.')
    return output

def recording_failure_forensics(recording: pd.DataFrame, epoch: pd.DataFrame, contract: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = contract['fixed_policy']['recording_channel_qc']
    work = recording.copy()
    work['finite_pass'] = work['finite_fraction'].eq(1.0)
    work['scale_pass'] = work['robust_scale_uv'].between(thresholds['minimum_robust_scale_uv'], thresholds['maximum_robust_scale_uv'], inclusive='both')
    work['p999_pass'] = work['p999_abs_uv'].le(thresholds['maximum_p999_abs_uv'])
    work['recording_channel_passed_recomputed'] = work['finite_pass'] & work['scale_pass'] & work['p999_pass']
    if not np.array_equal(work['recording_channel_passed_recomputed'].to_numpy(dtype=bool), work['recording_channel_passed'].map(python_bool).to_numpy(dtype=bool)):
        raise RuntimeError('Recording-channel pass flags do not match prespecified thresholds.')
    source = work.groupby(['subject_id', 'raw_run', 'raw_source_path'], sort=True).agg(channels=('channel_position_1based', 'size'), passed_channels=('recording_channel_passed_recomputed', 'sum'), finite_fail_channels=('finite_pass', lambda values: int((~values).sum())), scale_fail_channels=('scale_pass', lambda values: int((~values).sum())), p999_fail_channels=('p999_pass', lambda values: int((~values).sum())), median_robust_scale_uv=('robust_scale_uv', 'median'), median_p999_abs_uv=('p999_abs_uv', 'median'), maximum_p999_abs_uv=('p999_abs_uv', 'max')).reset_index()
    source['recording_passed_recomputed'] = source['passed_channels'].ge(MINIMUM_GOOD_CHANNELS)
    epoch_counts = epoch.groupby(['subject_id', 'raw_run', 'raw_source_path'], sort=True).agg(available_epochs=('epoch_id', 'size'), modeling_eligible_epochs=('modeling_eligible', lambda values: int(values.map(python_bool).sum()))).reset_index()
    source = source.merge(epoch_counts, on=['subject_id', 'raw_run', 'raw_source_path'], how='left', validate='one_to_one')
    source['all_epoch_loss_due_to_recording_gate'] = ~source['recording_passed_recomputed'] & source['modeling_eligible_epochs'].eq(0)
    passed_source_epoch = epoch.loc[epoch['recording_passed'].map(python_bool)]
    epoch_level_losses_after_recording_pass = int((~passed_source_epoch['modeling_eligible'].map(python_bool)).sum())
    decision = {'failed_sources': int((~source['recording_passed_recomputed']).sum()), 'failed_source_subjects': sorted(source.loc[~source['recording_passed_recomputed'], 'subject_id'].unique().tolist()), 'epochs_lost_to_failed_recordings': int(source.loc[~source['recording_passed_recomputed'], 'available_epochs'].sum()), 'epoch_level_losses_after_recording_pass': epoch_level_losses_after_recording_pass, 'all_modeling_loss_explained_by_recording_gate': bool(epoch_level_losses_after_recording_pass == 0 and int(source.loc[~source['recording_passed_recomputed'], 'available_epochs'].sum()) == len(epoch) - int(epoch['modeling_eligible'].map(python_bool).sum()))}
    return (source, decision)

def common_trial_masks(raw_eeg_epoch_folds_epoch: pd.DataFrame, epoch_qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial = raw_eeg_epoch_folds_epoch[['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'raw_run', 'raw_source_path', 'subject_acquisition_order_1based', 'within_subject_block_fold']].drop_duplicates().sort_values(['subject_id', 'subject_acquisition_order_1based']).reset_index(drop=True)
    if len(trial) != EXPECTED_COUNTS['authoritative_trials']:
        raise RuntimeError(f"Authoritative trial-row count changed: expected {EXPECTED_COUNTS['authoritative_trials']}, observed {len(trial)}.")
    model = epoch_qc.assign(modeling_bool=epoch_qc['modeling_eligible'].map(python_bool)).pivot(index='raw_behavior_row_id', columns='epoch_family', values='modeling_bool')
    trial = trial.merge(model, left_on='raw_behavior_row_id', right_index=True, how='left', validate='one_to_one')
    for family in FAMILIES:
        trial[family] = trial[family].fillna(False).astype(bool)
    for mask_name, required_families in COMPARISON_MASKS.items():
        trial[mask_name] = trial[required_families].all(axis=1)
    summary_rows = []
    subject_rows = []
    for mask_name in COMPARISON_MASKS:
        selected = trial.loc[trial[mask_name]]
        for subject_id, group in selected.groupby('subject_id', sort=True):
            total = int(len(group))
            imagery = int(group['context'].eq('imagery').sum())
            video = int(group['context'].eq('video').sum())
            subject_rows.append({'comparison_mask': mask_name, 'subject_id': subject_id, 'rows': total, 'imagery_rows': imagery, 'video_rows': video, 'within_person_modeling_eligible': bool(total >= SUBJECT_MINIMUM_TOTAL and imagery >= SUBJECT_MINIMUM_PER_CONTEXT and (video >= SUBJECT_MINIMUM_PER_CONTEXT))})
        subject_frame = pd.DataFrame(subject_rows)
        mask_subjects = subject_frame.loc[subject_frame['comparison_mask'].eq(mask_name)]
        summary_rows.append({'comparison_mask': mask_name, 'required_epoch_families': '|'.join(COMPARISON_MASKS[mask_name]), 'trial_rows': int(len(selected)), 'subjects_with_rows': int(selected['subject_id'].nunique()), 'within_person_eligible_subjects': int(mask_subjects['within_person_modeling_eligible'].map(python_bool).sum()), 'same_evaluation_rows_required': True, 'outcome_values_used': False})
    return (trial, pd.DataFrame(summary_rows), pd.DataFrame(subject_rows))

def write_report(path: Path, summary: dict[str, Any], failure: pd.DataFrame, common_summary: pd.DataFrame, semantic: pd.DataFrame) -> None:
    forensic = summary['recording_failure_forensics']
    lines = ['# Phase large_store_integrity — Large-store integrity and discovery_multimodal_inference specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Outcome or behavioral values read: **no**', '- Validation/confirmation opened: **no**', '', '## Full local-store integrity', '', f"- Inventory files hashed: **{summary['counts']['inventory_files_hashed']}**", f"- Bytes hashed: **{summary['counts']['inventory_bytes_hashed']:,}**", f"- Source summaries verified: **{summary['counts']['source_summaries_verified']}**", f'- Source/family array groups checked: **{len(semantic)}**', '- Every mask-true cell is finite: **yes**', '- Every mask-false cell is entirely NaN: **yes**', '- Every mask count matches epoch QC: **yes**', '', '## Recording failure forensics', '', f"- Failed recordings: **{forensic['failed_sources']}**", '- Failed subjects: ' + ', '.join((f'`{value}`' for value in forensic['failed_source_subjects'])), f"- Epochs lost to failed recordings: **{forensic['epochs_lost_to_failed_recordings']}**", f"- Additional epoch-level losses after a recording passed: **{forensic['epoch_level_losses_after_recording_pass']}**", '', '| Subject | Run | Good channels | Scale failures | p99.9 failures | Available epochs | Eligible epochs |', '|---|---:|---:|---:|---:|---:|---:|']
    failed = failure.loc[~failure['recording_passed_recomputed']]
    for _, row in failed.iterrows():
        lines.append(f"| `{row['subject_id']}` | {int(row['raw_run'])} | {int(row['passed_channels'])} | {int(row['scale_fail_channels'])} | {int(row['p999_fail_channels'])} | {int(row['available_epochs'])} | {int(row['modeling_eligible_epochs'])} |")
    lines.extend(['', 'The two recordings are retained in the store as all-NaN, all-false-mask arrays so that row provenance remains complete. They do not enter modeling.', '', '## prespecified same-row comparison sets', '', '| Comparison mask | Required families | Trial rows | Subjects | Within-person eligible |', '|---|---|---:|---:|---:|'])
    for _, row in common_summary.iterrows():
        lines.append(f"| `{row['comparison_mask']}` | `{row['required_epoch_families']}` | {int(row['trial_rows'])} | {int(row['subjects_with_rows'])} | {int(row['within_person_eligible_subjects'])} |")
    decision = summary['decision']
    lines.extend(['', '## Decision', '', f"- Large store byte integrity passed: **{decision['large_store_byte_integrity_passed']}**", f"- Array/mask semantic integrity passed: **{decision['array_mask_semantic_integrity_passed']}**", f"- discovery_multimodal_inference representation extraction permitted: **{decision['discovery_multimodal_inference_representation_extraction_permitted']}**", f"- QC threshold changed: **{decision['qc_threshold_changed']}**", f"- Modeling rows changed: **{decision['modeling_rows_changed']}**", '', 'The prior raw_eeg_preprocessing_qc resume validator accepted stored files by size. raw_eeg_preprocessing_qc.1 upgrades the scientific acceptance criterion to full SHA-256 plus semantic array verification. Future analysis must depend on this raw_eeg_preprocessing_qc.1 contract rather than size-only checkpoint acceptance.', ''])
    path.write_text(''.join(lines) + '', encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, raw_eeg_preprocessing_qc_summary, raw_eeg_preprocessing_qc_contract, recording, epoch, inventory, branch, subject, raw_eeg_epoch_folds_epoch = load_inputs(project)
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config_dir = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'file_integrity': tables / 'large_store_integrity_large_store_file_integrity.csv', 'source_integrity': qc / 'large_store_integrity_source_checkpoint_integrity.csv', 'array_semantics': qc / 'large_store_integrity_array_mask_semantic_integrity.csv', 'recording_forensics': qc / 'large_store_integrity_recording_failure_forensics.csv', 'common_trial_masks': tables / 'large_store_integrity_common_trial_mask_manifest.csv', 'common_mask_summary': qc / 'large_store_integrity_common_trial_mask_summary.csv', 'common_mask_subjects': qc / 'large_store_integrity_common_trial_subject_eligibility.csv', 'summary': qc / 'large_store_integrity_large_store_integrity_summary.json', 'contract': config_dir / 'large_store_integrity_large_store_integrity_contract.json', 'discovery_multimodal_inference_feature_policy': config_dir / 'S035_large_store_integrity_discovery_multimodal_inference.json', 'report': reports / 'large_store_integrity_large_store_integrity_report.md', 'log': logs / 'large_store_integrity_large_store_integrity.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Outputs already exist. Use --force after review:\n  - ' + '\n  - '.join((str(path) for path in existing)))
    with outputs['log'].open('w', encoding='utf-8') as log:

        def note(message: str) -> None:
            line = f'[{now_iso()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Hashing every locally stored raw_eeg_preprocessing_qc file against the prespecified inventory.')
        file_integrity = verify_inventory_files(inventory)
        file_integrity.to_csv(outputs['file_integrity'], index=False)
        note('Verifying all source summaries against policy, epoch manifest, and inventory metadata.')
        source_integrity = verify_source_summaries(inventory, policy_sha256=raw_eeg_preprocessing_qc_summary['policy_sha256'], epoch_manifest_sha256=raw_eeg_preprocessing_qc_contract['input_hashes']['epoch_manifest'])
        source_integrity.to_csv(outputs['source_integrity'], index=False)
        note('Scanning every NPY array and mask for shape, dtype, row order, finite/NaN semantics, and QC equivalence.')
        array_semantics = verify_array_semantics(inventory, epoch)
        array_semantics.to_csv(outputs['array_semantics'], index=False)
        note('Recomputing recording failure reasons from the prespecified raw_eeg_preprocessing_qc thresholds.')
        recording_forensics, forensic_decision = recording_failure_forensics(recording, epoch, raw_eeg_preprocessing_qc_contract)
        recording_forensics.to_csv(outputs['recording_forensics'], index=False)
        note('Freezing same-row trial masks for branch and cross-epoch comparisons.')
        common_masks, common_summary, common_subjects = common_trial_masks(raw_eeg_epoch_folds_epoch, epoch)
        common_masks.to_csv(outputs['common_trial_masks'], index=False)
        common_summary.to_csv(outputs['common_mask_summary'], index=False)
        common_subjects.to_csv(outputs['common_mask_subjects'], index=False)
        decision = {'large_store_byte_integrity_passed': True, 'source_summary_integrity_passed': True, 'array_mask_semantic_integrity_passed': True, 'recording_qc_recomputed_exactly': True, 'all_modeling_loss_explained_by_recording_gate': bool(forensic_decision['all_modeling_loss_explained_by_recording_gate']), 'same_row_comparison_masks_prespecified': True, 'discovery_multimodal_inference_representation_extraction_permitted': True, 'qc_threshold_changed': False, 'epoch_window_changed': False, 'modeling_rows_changed': False, 'outcome_values_read': False, 'released_derivative_amplitude_read': False, 'validation_opened': False, 'confirmation_opened': False}
        summary = {'generated_at': now_iso(), 'phase': 'large_store_integrity', 'status': 'full_large_store_integrity_qc_forensics_and_discovery_multimodal_inference_specification_complete_candidate', 'counts': {'inventory_files_hashed': int(len(file_integrity)), 'inventory_bytes_hashed': int(file_integrity['observed_size_bytes'].sum()), 'source_summaries_verified': int(len(source_integrity)), 'source_family_array_groups': int(len(array_semantics)), 'epoch_rows_semantically_checked': int(array_semantics['index_rows'].sum()), 'common_trial_rows': int(len(common_masks))}, 'recording_failure_forensics': forensic_decision, 'common_trial_masks': common_summary.to_dict(orient='records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        contract = {**summary, 'scientific_acceptance_upgrade': 'future consumers must verify full stored-file SHA-256 and array/mask semantics; size-only source-checkpoint acceptance is insufficient', 'feature_policy_sha256': hashlib.sha256(json.dumps(FEATURE_POLICY, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest(), 'input_hashes': {name: sha256_file(path) for name, path in paths.items()}, 'output_hashes': {name: sha256_file(path) for name, path in outputs.items() if name not in {'contract', 'discovery_multimodal_inference_feature_policy', 'report', 'log'} and path.is_file()}, 'source_dataset_modified': False, 'large_store_modified': False, 'canonical_master_modified': False}
        outputs['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        feature_policy = {'generated_at': now_iso(), 'phase': 'large_store_integrity', 'next_phase': 'outcome_free_features', 'next_phase_permitted': True, 'large_store_integrity_contract': str(outputs['contract']), 'large_array_inventory': str(paths['array_inventory']), 'epoch_qc_manifest': str(paths['epoch_qc']), 'common_trial_mask_manifest': str(outputs['common_trial_masks']), 'trial_fold_manifest': str(paths['raw_eeg_epoch_folds_trial_folds']), 'participant_fold_manifest': str(paths['raw_eeg_epoch_folds_participant_folds']), 'feature_policy': FEATURE_POLICY, 'representation_role': 'outcome-free full-signal feature extraction only', 'later_modeling_constraints': ['branch-specific models may use only their prespecified branch mask', 'cross-epoch comparisons must use the corresponding common-row mask', 'all imputation/scaling/dimensionality reduction/tangent maps are outer-fold-contained', 'stimulus and rating representations remain separately reported', 'no validation or confirmation participants', 'no biological ground-truth language']}
        outputs['discovery_multimodal_inference_feature_policy'].write_text(json.dumps(feature_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(outputs['report'], summary, recording_forensics, common_summary, array_semantics)
        note('raw_eeg_preprocessing_qc.1 completed. The large store was not modified and no outcomes were read.')
    print()
    print('=' * 100)
    print('PHASE large_store_integrity LARGE-STORE INTEGRITY COMPLETED')
    print('=' * 100)
    print(f'Inventory files hashed:         {len(file_integrity)}')
    print(f"Inventory bytes hashed:         {int(file_integrity['observed_size_bytes'].sum())}")
    print(f'Source summaries verified:      {len(source_integrity)}')
    print(f'Source/family groups checked:   {len(array_semantics)}')
    print('Failed recording subjects:      ' + ', '.join(forensic_decision['failed_source_subjects']))
    print('discovery_multimodal_inference extraction permitted:       True')
    print(f"Report:                          {outputs['report']}")
    print(f'Runtime:                         {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())