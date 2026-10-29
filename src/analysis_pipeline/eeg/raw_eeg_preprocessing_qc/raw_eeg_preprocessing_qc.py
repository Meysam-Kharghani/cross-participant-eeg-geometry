# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase raw_eeg_preprocessing_qc — resumable raw-EEG preprocessing and amplitude QC.

Prerequisite
------------
raw_eeg_epoch_folds froze:
- 1,379 authoritative discovery trials;
- three fixed epoch families;
- 4,083 temporally available epochs;
- source-sample bounds;
- outcome-free within-person and participant folds;
- branch coverage gates.

raw_eeg_preprocessing_qc reads amplitude only for the raw_eeg_epoch_folds-available epochs. It does not change a
window, endpoint, fold, or branch after seeing the signal.

Primary signal pipeline
-----------------------
1. Read each EDF source independently.
2. For each channel, filter the complete continuous recording with a
   fourth-order Butterworth SOS bandpass, 0.5–45 Hz, zero phase.
3. Extract the exact raw_eeg_epoch_folds source-sample interval.
4. Resample with scipy.signal.resample_poly to 200 Hz.
5. Compute recording-level channel QC from a deterministic continuous
   subsample.
6. Compute an iterative 10%-trimmed robust average reference per epoch over
   channels passing the prespecified recording and epoch QC.
7. Preserve 64-channel layout; channels failing final epoch QC are stored as
   NaN. There is no interpolation and no automatic ICA removal.

prespecified recording-channel QC
---------------------------
- all samples finite;
- robust scale in [0.02, 500] microvolts;
- 99.9th percentile absolute amplitude <= 5,000 microvolts.
A recording passes when at least 48 of 64 channels pass.

prespecified epoch-channel QC after robust reference
----------------------------------------------
- all samples finite;
- robust scale in [0.02, 500] microvolts;
- peak-to-peak <= 2,000 microvolts;
- maximum absolute amplitude <= 1,000 microvolts.
An epoch passes when at least 48 channels pass.

The reference/QC iteration is deterministic and capped at three iterations.
No threshold is changed after observing results.

prespecified branch-sufficiency gates after amplitude QC
--------------------------------------------------
For each epoch family:
- overall retention among raw_eeg_epoch_folds-available epochs >= 0.75;
- retention in each context >= 0.65;
- at least 24 participants retain >=16 epochs and >=6 per context.

These are data-sufficiency gates, not evidence of signal validity.

Storage and recovery
--------------------
Large arrays are written under:
outputs/large/raw_eeg_preprocessing_qc_raw_eeg_epochs_v1/

Each EDF is processed into an atomic source directory with:
- referenced float32 arrays by epoch family;
- final epoch-channel masks;
- row-index CSV files;
- recording-channel QC;
- epoch QC;
- source summary and file hashes.

A valid completed source is reused on rerun. --force replaces phase-level
metadata but does not recompute valid source stores. --recompute-sources is
required to rebuild completed source stores.

No behavior/outcome value, released filtered/ICA/reorder amplitude, validation,
or confirmation data is read."""
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
raw_eeg_epoch_folds_SUMMARY_REL = Path('outputs/qc/raw_eeg_epoch_folds_raw_eeg_epoch_folds_summary.json')
raw_eeg_epoch_folds_CONTRACT_REL = Path('config/raw_eeg_epoch_folds_raw_eeg_epoch_folds_contract.json')
raw_eeg_epoch_folds_POLICY_REL = Path('analysis_specifications/S057_raw_eeg_epoch_folds.json')
EPOCH_MANIFEST_REL = Path('outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv')
RAW_HEADER_REL = Path('outputs/qc/raw_eeg_epoch_folds_raw_edf_header_inventory.csv')
TRIAL_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_trial_fold_manifest.csv')
PARTICIPANT_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_participant_fold_manifest.csv')
derivative_closure_ANCHOR_REL = Path('outputs/qc/derivative_closure_exact_anchor_adjudication.csv')
STORE_REL = Path('outputs/large/raw_eeg_preprocessing_qc_raw_eeg_epochs_v1')
CHANNEL_COUNT = 64
OUTPUT_SAMPLING_RATE_HZ = 200
FILTER_ORDER = 4
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 45.0
REFERENCE_TRIM_FRACTION = 0.1
REFERENCE_MAX_ITERATIONS = 3
MINIMUM_GOOD_CHANNELS = 48
QC_SUBSAMPLE_MAX = 1000000
RECORDING_QC = {'minimum_robust_scale_uv': 0.02, 'maximum_robust_scale_uv': 500.0, 'maximum_p999_abs_uv': 5000.0}
EPOCH_CHANNEL_QC = {'minimum_robust_scale_uv': 0.02, 'maximum_robust_scale_uv': 500.0, 'maximum_peak_to_peak_uv': 2000.0, 'maximum_abs_uv': 1000.0}
BRANCH_GATES = {'minimum_overall_retention': 0.75, 'minimum_context_retention': 0.65, 'minimum_eligible_subjects': 24, 'subject_minimum_total': 16, 'subject_minimum_per_context': 6}
FIXED_POLICY = {'channel_count': CHANNEL_COUNT, 'output_sampling_rate_hz': OUTPUT_SAMPLING_RATE_HZ, 'continuous_filter': {'family': 'butterworth', 'order': FILTER_ORDER, 'lowcut_hz': LOWCUT_HZ, 'highcut_hz': HIGHCUT_HZ, 'representation': 'second_order_sections', 'phase': 'zero_phase_sosfiltfilt'}, 'resampling': {'method': 'scipy_signal_resample_poly', 'padtype': 'line'}, 'reference': {'method': 'iterative_trimmed_average', 'trim_fraction_each_tail': REFERENCE_TRIM_FRACTION, 'maximum_iterations': REFERENCE_MAX_ITERATIONS, 'minimum_good_channels': MINIMUM_GOOD_CHANNELS, 'interpolation': False, 'automatic_ica_removal': False}, 'recording_channel_qc': RECORDING_QC, 'epoch_channel_qc': EPOCH_CHANNEL_QC, 'branch_gates': BRANCH_GATES, 'continuous_qc_subsample_max': QC_SUBSAMPLE_MAX}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resumably preprocess raw_eeg_epoch_folds-prespecified raw EEG epochs and apply outcome-free amplitude QC.')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--force', action='store_true', help='Overwrite phase-level metadata. Valid per-source stores are reused.')
    parser.add_argument('--recompute-sources', action='store_true', help='Rebuild completed per-source epoch stores. Use only after review.')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
POLICY_SHA256 = canonical_json_sha256(FIXED_POLICY)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
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

def robust_scale_uv(array: np.ndarray, *, axis: int | None=None) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    return 1.4826 * mad * 1000000.0

def deterministic_subsample(array: np.ndarray, maximum: int=QC_SUBSAMPLE_MAX) -> np.ndarray:
    values = np.asarray(array)
    size = int(values.size)
    if size <= maximum:
        return values
    step = int(math.ceil(size / maximum))
    return values[::step]

def recording_channel_metrics(filtered_channel: np.ndarray) -> dict[str, Any]:
    channel = np.asarray(filtered_channel, dtype=np.float64)
    finite = np.isfinite(channel)
    finite_fraction = float(finite.mean())
    if finite.any():
        sample = deterministic_subsample(channel[finite])
        scale = float(robust_scale_uv(sample))
        p999_abs = float(np.percentile(np.abs(sample) * 1000000.0, 99.9))
    else:
        scale = math.nan
        p999_abs = math.nan
    passed = bool(finite_fraction == 1.0 and math.isfinite(scale) and (RECORDING_QC['minimum_robust_scale_uv'] <= scale <= RECORDING_QC['maximum_robust_scale_uv']) and (p999_abs <= RECORDING_QC['maximum_p999_abs_uv']))
    return {'finite_fraction': finite_fraction, 'robust_scale_uv': scale, 'p999_abs_uv': p999_abs, 'recording_channel_passed': passed}

def epoch_channel_metrics(referenced_epoch: np.ndarray) -> pd.DataFrame:
    epoch = np.asarray(referenced_epoch, dtype=np.float64)
    finite = np.isfinite(epoch).all(axis=1)
    scale = robust_scale_uv(epoch, axis=1)
    peak_to_peak = (np.nanmax(epoch, axis=1) - np.nanmin(epoch, axis=1)) * 1000000.0
    maximum_abs = np.nanmax(np.abs(epoch), axis=1) * 1000000.0
    passed = finite & np.isfinite(scale) & (scale >= EPOCH_CHANNEL_QC['minimum_robust_scale_uv']) & (scale <= EPOCH_CHANNEL_QC['maximum_robust_scale_uv']) & (peak_to_peak <= EPOCH_CHANNEL_QC['maximum_peak_to_peak_uv']) & (maximum_abs <= EPOCH_CHANNEL_QC['maximum_abs_uv'])
    return pd.DataFrame({'finite': finite, 'robust_scale_uv': scale, 'peak_to_peak_uv': peak_to_peak, 'maximum_abs_uv': maximum_abs, 'epoch_channel_passed': passed})

def trimmed_reference(unreferenced_epoch: np.ndarray, active_channels: np.ndarray) -> np.ndarray:
    epoch = np.asarray(unreferenced_epoch, dtype=np.float64)
    active = np.asarray(active_channels, dtype=bool)
    selected = epoch[active]
    if selected.shape[0] < 2:
        raise RuntimeError('At least two channels are required for a reference.')
    ordered = np.sort(selected, axis=0)
    trim = int(math.floor(selected.shape[0] * REFERENCE_TRIM_FRACTION))
    if trim > 0 and selected.shape[0] - 2 * trim >= 2:
        ordered = ordered[trim:selected.shape[0] - trim]
    return np.mean(ordered, axis=0, dtype=np.float64)

def iterative_reference_and_qc(unreferenced_epoch: np.ndarray, recording_good: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    epoch = np.asarray(unreferenced_epoch, dtype=np.float64)
    active = np.asarray(recording_good, dtype=bool).copy()
    iterations = 0
    converged = False
    if int(active.sum()) < MINIMUM_GOOD_CHANNELS:
        referenced = np.full_like(epoch, np.nan, dtype=np.float64)
        return (referenced, np.zeros(CHANNEL_COUNT, dtype=bool), {'reference_iterations': 0, 'reference_converged': False, 'final_good_channels': 0, 'epoch_passed': False, 'maximum_abs_uv': math.nan, 'maximum_peak_to_peak_uv': math.nan, 'median_robust_scale_uv': math.nan})
    referenced = np.empty_like(epoch, dtype=np.float64)
    for iteration in range(1, REFERENCE_MAX_ITERATIONS + 1):
        reference = trimmed_reference(epoch, active)
        referenced = epoch - reference[None, :]
        metrics = epoch_channel_metrics(referenced)
        updated = np.asarray(recording_good, dtype=bool) & metrics['epoch_channel_passed'].to_numpy(dtype=bool)
        iterations = iteration
        if np.array_equal(updated, active):
            converged = True
            active = updated
            break
        active = updated
        if int(active.sum()) < MINIMUM_GOOD_CHANNELS:
            break
    if int(active.sum()) >= MINIMUM_GOOD_CHANNELS:
        reference = trimmed_reference(epoch, active)
        referenced = epoch - reference[None, :]
        final_metrics = epoch_channel_metrics(referenced)
        final_good = np.asarray(recording_good, dtype=bool) & final_metrics['epoch_channel_passed'].to_numpy(dtype=bool)
    else:
        final_metrics = epoch_channel_metrics(referenced)
        final_good = np.zeros(CHANNEL_COUNT, dtype=bool)
    epoch_passed = bool(int(final_good.sum()) >= MINIMUM_GOOD_CHANNELS)
    output = np.asarray(referenced, dtype=np.float32)
    output[~final_good] = np.nan
    good_metrics = final_metrics.loc[final_good]
    if len(good_metrics):
        maximum_abs = float(good_metrics['maximum_abs_uv'].max())
        maximum_p2p = float(good_metrics['peak_to_peak_uv'].max())
        median_scale = float(good_metrics['robust_scale_uv'].median())
    else:
        maximum_abs = math.nan
        maximum_p2p = math.nan
        median_scale = math.nan
    return (output, final_good, {'reference_iterations': int(iterations), 'reference_converged': bool(converged), 'final_good_channels': int(final_good.sum()), 'epoch_passed': epoch_passed, 'maximum_abs_uv': maximum_abs, 'maximum_peak_to_peak_uv': maximum_p2p, 'median_robust_scale_uv': median_scale})

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
    os.replace(temporary, path)

def source_slug(source_path: str, subject_id: str, raw_run: int) -> str:
    digest = hashlib.sha256(source_path.encode('utf-8')).hexdigest()[:12]
    return f'{subject_id}_run-{raw_run:02d}_{digest}'

def load_inputs(project: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paths = {'raw_eeg_epoch_folds_summary': project / raw_eeg_epoch_folds_SUMMARY_REL, 'raw_eeg_epoch_folds_contract': project / raw_eeg_epoch_folds_CONTRACT_REL, 'raw_eeg_epoch_folds_policy': project / raw_eeg_epoch_folds_POLICY_REL, 'epoch_manifest': project / EPOCH_MANIFEST_REL, 'raw_headers': project / RAW_HEADER_REL, 'trial_folds': project / TRIAL_FOLD_REL, 'participant_folds': project / PARTICIPANT_FOLD_REL, 'derivative_closure_anchors': project / derivative_closure_ANCHOR_REL}
    require_files(paths.values())
    summary = read_json(paths['raw_eeg_epoch_folds_summary'])
    contract = read_json(paths['raw_eeg_epoch_folds_contract'])
    if summary.get('phase') != 'raw_eeg_epoch_folds':
        raise RuntimeError('Unexpected raw_eeg_epoch_folds prerequisite.')
    if not bool(summary['decision']['at_least_one_raw_eeg_preprocessing_qc_branch_permitted']):
        raise RuntimeError('raw_eeg_epoch_folds did not permit a raw_eeg_preprocessing_qc branch.')
    if bool(summary['decision']['eeg_amplitude_read']):
        raise RuntimeError('raw_eeg_epoch_folds unexpectedly read EEG amplitude.')
    epoch_manifest = pd.read_csv(paths['epoch_manifest'], low_memory=False)
    headers = pd.read_csv(paths['raw_headers'], low_memory=False)
    available = epoch_manifest.loc[epoch_manifest['epoch_available'].map(python_bool)].copy()
    if len(epoch_manifest) != 4137:
        raise RuntimeError('raw_eeg_epoch_folds epoch manifest row count changed.')
    if len(available) != 4083:
        raise RuntimeError('raw_eeg_epoch_folds available epoch count changed.')
    if available['epoch_id'].duplicated().any():
        raise RuntimeError('Available epoch IDs are not unique.')
    if not available['behavioral_values_used'].eq(False).all():
        raise RuntimeError('raw_eeg_epoch_folds marks behavioral values as used.')
    if not available['eeg_amplitude_read'].eq(False).all():
        raise RuntimeError('raw_eeg_epoch_folds marks EEG amplitude as already read.')
    return (paths, summary, contract, available, headers)

def verify_source_against_raw_eeg_epoch_folds(source_path: Path, header_row: pd.Series, raw: mne.io.BaseRaw) -> None:
    observed = {'source_size_bytes': int(source_path.stat().st_size), 'source_mtime_ns': int(source_path.stat().st_mtime_ns), 'sampling_rate_hz': float(raw.info['sfreq']), 'n_times': int(raw.n_times), 'channel_count': int(len(raw.ch_names)), 'channel_order_sha256': hashlib.sha256(''.join(raw.ch_names).encode('utf-8')).hexdigest()}
    for key, value in observed.items():
        expected = header_row[key]
        if isinstance(value, float):
            equal = math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-09)
        else:
            equal = value == expected
        if not equal:
            raise RuntimeError(f'{source_path}: raw_eeg_epoch_folds header mismatch for {key}: expected {expected!r}, observed {value!r}.')

def resampling_ratio(source_rate_hz: float) -> tuple[int, int]:
    ratio = Fraction(OUTPUT_SAMPLING_RATE_HZ) / Fraction(int(round(source_rate_hz)))
    return (int(ratio.numerator), int(ratio.denominator))

def source_checkpoint_valid(source_dir: Path, *, policy_sha256: str, source_path: str, source_size_bytes: int, source_mtime_ns: int, epoch_manifest_sha256: str) -> bool:
    summary_path = source_dir / 'source_summary.json'
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except Exception:
        return False
    required_equal = {'complete': True, 'policy_sha256': policy_sha256, 'raw_source_path': source_path, 'source_size_bytes': int(source_size_bytes), 'source_mtime_ns': int(source_mtime_ns), 'epoch_manifest_sha256': epoch_manifest_sha256}
    for key, expected in required_equal.items():
        if summary.get(key) != expected:
            return False
    for file_row in summary.get('stored_files', []):
        path = source_dir / file_row['relative_path']
        if not path.is_file():
            return False
        if int(path.stat().st_size) != int(file_row['size_bytes']):
            return False
    return True

def process_source(source_group: pd.DataFrame, header_row: pd.Series, *, store_root: Path, epoch_manifest_sha256: str, recompute: bool) -> Path:
    source_paths = source_group['raw_source_path'].astype(str).unique()
    if len(source_paths) != 1:
        raise RuntimeError('A source group contains multiple EDF paths.')
    source_path = Path(source_paths[0])
    subject_values = source_group['subject_id'].astype(str).unique()
    run_values = source_group['raw_run'].astype(int).unique()
    if len(subject_values) != 1 or len(run_values) != 1:
        raise RuntimeError(f'{source_path}: source identity is not unique.')
    subject_id = subject_values[0]
    raw_run = int(run_values[0])
    slug = source_slug(str(source_path), subject_id, raw_run)
    subject_root = store_root / subject_id
    subject_root.mkdir(parents=True, exist_ok=True)
    final_dir = subject_root / slug
    source_size = int(source_path.stat().st_size)
    source_mtime = int(source_path.stat().st_mtime_ns)
    if not recompute and source_checkpoint_valid(final_dir, policy_sha256=POLICY_SHA256, source_path=str(source_path), source_size_bytes=source_size, source_mtime_ns=source_mtime, epoch_manifest_sha256=epoch_manifest_sha256):
        return final_dir
    temporary_dir = subject_root / ('.' + slug + '.tmp')
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    raw = None
    try:
        raw = mne.io.read_raw_edf(source_path, preload=False, verbose='ERROR')
        verify_source_against_raw_eeg_epoch_folds(source_path, header_row, raw)
        source_sha256 = sha256_file(source_path)
        sampling_rate_hz = float(raw.info['sfreq'])
        n_times = int(raw.n_times)
        picks = list(mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False, stim=False, misc=False, exclude=[]))
        if len(picks) != CHANNEL_COUNT:
            if len(raw.ch_names) == CHANNEL_COUNT:
                picks = list(range(CHANNEL_COUNT))
            else:
                raise RuntimeError(f'{source_path}: expected 64 EEG channels.')
        sos = butter(FILTER_ORDER, [LOWCUT_HZ, HIGHCUT_HZ], btype='bandpass', fs=sampling_rate_hz, output='sos')
        up, down = resampling_ratio(sampling_rate_hz)
        family_rows: dict[str, pd.DataFrame] = {}
        family_arrays: dict[str, np.memmap] = {}
        family_masks: dict[str, np.memmap] = {}
        for epoch_family, family_group in source_group.groupby('epoch_family', sort=True):
            family_group = family_group.sort_values(['subject_acquisition_order_1based', 'epoch_id']).reset_index(drop=True)
            expected_samples = int(family_group['expected_output_samples'].iloc[0])
            if not family_group['expected_output_samples'].eq(expected_samples).all():
                raise RuntimeError(f'{source_path} {epoch_family}: output length changed.')
            index_table = family_group.copy()
            index_table['source_array_row_0based'] = np.arange(len(index_table), dtype=int)
            index_table.to_csv(temporary_dir / f'{epoch_family}_index.csv', index=False)
            data_path = temporary_dir / f'{epoch_family}_epochs_float32.npy'
            mask_path = temporary_dir / f'{epoch_family}_channel_mask_bool.npy'
            family_arrays[epoch_family] = np.lib.format.open_memmap(data_path, mode='w+', dtype=np.float32, shape=(len(index_table), CHANNEL_COUNT, expected_samples))
            family_masks[epoch_family] = np.lib.format.open_memmap(mask_path, mode='w+', dtype=np.bool_, shape=(len(index_table), CHANNEL_COUNT))
            family_masks[epoch_family][:] = False
            family_rows[epoch_family] = index_table
        recording_qc_rows = []
        for channel_position, pick in enumerate(picks):
            channel = raw.get_data(picks=[pick], start=0, stop=n_times)[0]
            filtered = sosfiltfilt(sos, np.asarray(channel, dtype=np.float64), axis=-1)
            metrics = recording_channel_metrics(filtered)
            recording_qc_rows.append({'subject_id': subject_id, 'raw_run': raw_run, 'raw_source_path': str(source_path), 'channel_position_1based': int(channel_position + 1), 'channel_name': raw.ch_names[pick], **metrics})
            for epoch_family, index_table in family_rows.items():
                output_array = family_arrays[epoch_family]
                for row_index, row in index_table.iterrows():
                    start = int(row['start_sample_nearest'])
                    stop = int(row['stop_sample_nearest_exclusive'])
                    if start < 0 or stop > n_times or stop <= start:
                        raise RuntimeError(f"{row['epoch_id']}: invalid raw_eeg_epoch_folds source bounds.")
                    segment = filtered[start:stop]
                    resampled = resample_poly(segment, up, down, padtype='line')
                    expected = output_array.shape[2]
                    if resampled.size != expected:
                        raise RuntimeError(f"{row['epoch_id']}: expected {expected} resampled points; observed {resampled.size}.")
                    output_array[row_index, channel_position, :] = np.asarray(resampled, dtype=np.float32)
            del filtered
            del channel
        recording_qc = pd.DataFrame(recording_qc_rows)
        recording_qc.to_csv(temporary_dir / 'recording_channel_qc.csv', index=False)
        recording_good = recording_qc['recording_channel_passed'].map(python_bool).to_numpy(dtype=bool)
        recording_passed = bool(int(recording_good.sum()) >= MINIMUM_GOOD_CHANNELS)
        epoch_qc_rows = []
        for epoch_family, output_array in family_arrays.items():
            channel_mask = family_masks[epoch_family]
            index_table = family_rows[epoch_family]
            for row_index, row in index_table.iterrows():
                unreferenced = np.asarray(output_array[row_index], dtype=np.float64)
                referenced, final_good, metrics = iterative_reference_and_qc(unreferenced, recording_good)
                output_array[row_index] = referenced
                channel_mask[row_index] = final_good
                epoch_qc_rows.append({'epoch_id': row['epoch_id'], 'raw_behavior_row_id': row['raw_behavior_row_id'], 'subject_id': subject_id, 'context': row['context'], 'raw_run': raw_run, 'raw_source_path': str(source_path), 'epoch_family': epoch_family, 'source_array_row_0based': int(row_index), 'recording_good_channels': int(recording_good.sum()), 'recording_passed': recording_passed, **metrics, 'modeling_eligible': bool(recording_passed and metrics['epoch_passed']), 'behavioral_values_used_for_qc': False})
            output_array.flush()
            channel_mask.flush()
        epoch_qc = pd.DataFrame(epoch_qc_rows)
        epoch_qc.to_csv(temporary_dir / 'epoch_qc.csv', index=False)
        stored_files = []
        for file_path in sorted(temporary_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.name == 'source_summary.json':
                continue
            stored_files.append({'relative_path': file_path.name, 'size_bytes': int(file_path.stat().st_size), 'sha256': sha256_file(file_path)})
        summary = {'generated_at': now_iso(), 'complete': True, 'phase': 'raw_eeg_preprocessing_qc', 'policy_sha256': POLICY_SHA256, 'epoch_manifest_sha256': epoch_manifest_sha256, 'subject_id': subject_id, 'raw_run': raw_run, 'raw_source_path': str(source_path), 'raw_source_sha256': source_sha256, 'source_size_bytes': source_size, 'source_mtime_ns': source_mtime, 'sampling_rate_hz': sampling_rate_hz, 'n_times': n_times, 'recording_good_channels': int(recording_good.sum()), 'recording_passed': recording_passed, 'available_epoch_rows': int(len(epoch_qc)), 'modeling_eligible_epoch_rows': int(epoch_qc['modeling_eligible'].map(python_bool).sum()), 'stored_files': stored_files}
        atomic_json(temporary_dir / 'source_summary.json', summary)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temporary_dir, final_dir)
        return final_dir
    except Exception:
        if temporary_dir.exists():
            failure_path = temporary_dir / 'FAILED.txt'
            failure_path.write_text(f'Failed at {now_iso()}.\nThe directory is not a completed checkpoint and will be removed on the next run.', encoding='utf-8')
        raise
    finally:
        if raw is not None:
            raw.close()

def collect_source_outputs(source_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recording_frames = []
    epoch_frames = []
    inventory_rows = []
    for source_dir in source_dirs:
        summary = read_json(source_dir / 'source_summary.json')
        recording = pd.read_csv(source_dir / 'recording_channel_qc.csv', low_memory=False)
        epoch_qc = pd.read_csv(source_dir / 'epoch_qc.csv', low_memory=False)
        recording_frames.append(recording)
        epoch_frames.append(epoch_qc)
        for file_row in summary['stored_files']:
            path = source_dir / file_row['relative_path']
            inventory_rows.append({'subject_id': summary['subject_id'], 'raw_run': int(summary['raw_run']), 'raw_source_path': summary['raw_source_path'], 'source_store_directory': str(source_dir), 'stored_path': str(path), 'relative_path': file_row['relative_path'], 'size_bytes': int(file_row['size_bytes']), 'sha256': file_row['sha256'], 'policy_sha256': summary['policy_sha256'], 'raw_source_sha256': summary['raw_source_sha256']})
    return (pd.concat(recording_frames, ignore_index=True, sort=False), pd.concat(epoch_frames, ignore_index=True, sort=False), pd.DataFrame(inventory_rows))

def branch_qc(epoch_qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    subject_rows = []
    for epoch_family, family in epoch_qc.groupby('epoch_family', sort=True):
        modeling = family['modeling_eligible'].map(python_bool)
        overall = float(modeling.mean())
        context_retention = family.assign(modeling_bool=modeling).groupby('context')['modeling_bool'].mean().to_dict()
        for subject_id, subject in family.groupby('subject_id', sort=True):
            eligible = subject.loc[subject['modeling_eligible'].map(python_bool)]
            imagery = int(eligible['context'].eq('imagery').sum())
            video = int(eligible['context'].eq('video').sum())
            total = int(len(eligible))
            subject_rows.append({'epoch_family': epoch_family, 'subject_id': subject_id, 'available_before_amplitude_qc': int(len(subject)), 'modeling_eligible_rows': total, 'imagery_rows': imagery, 'video_rows': video, 'within_person_modeling_eligible': bool(total >= BRANCH_GATES['subject_minimum_total'] and imagery >= BRANCH_GATES['subject_minimum_per_context'] and (video >= BRANCH_GATES['subject_minimum_per_context']))})
        subject_frame = pd.DataFrame(subject_rows)
        eligible_subjects = int(subject_frame.loc[subject_frame['epoch_family'].eq(epoch_family), 'within_person_modeling_eligible'].map(python_bool).sum())
        context_pass = bool(set(context_retention) == {'imagery', 'video'} and all((value >= BRANCH_GATES['minimum_context_retention'] for value in context_retention.values())))
        rows.append({'epoch_family': epoch_family, 'available_before_amplitude_qc': int(len(family)), 'modeling_eligible_rows': int(modeling.sum()), 'overall_retention': overall, 'minimum_overall_retention': float(BRANCH_GATES['minimum_overall_retention']), 'imagery_retention': float(context_retention.get('imagery', math.nan)), 'video_retention': float(context_retention.get('video', math.nan)), 'minimum_context_retention': float(BRANCH_GATES['minimum_context_retention']), 'within_person_eligible_subjects': eligible_subjects, 'minimum_eligible_subjects': int(BRANCH_GATES['minimum_eligible_subjects']), 'branch_passed': bool(overall >= BRANCH_GATES['minimum_overall_retention'] and context_pass and (eligible_subjects >= BRANCH_GATES['minimum_eligible_subjects'])), 'threshold_changed_after_amplitude': False})
    return (pd.DataFrame(rows), pd.DataFrame(subject_rows))

def write_report(path: Path, summary: dict[str, Any], branch: pd.DataFrame, recording_qc: pd.DataFrame, epoch_qc: pd.DataFrame, inventory: pd.DataFrame) -> None:
    lines = ['# Phase raw_eeg_preprocessing_qc — Raw-EEG preprocessing and amplitude QC', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Behavioral/outcome values used: **no**', '- Released derivative amplitude used: **no**', '- Validation/confirmation opened: **no**', '', '## Source processing', '', f"- EDF sources completed: **{summary['counts']['raw_edf_sources']}**", f"- Raw source bytes hashed: **{summary['counts']['raw_source_bytes_hashed']:,}**", f'- Recording-channel QC rows: **{len(recording_qc):,}**', f'- Epoch QC rows: **{len(epoch_qc):,}**', f'- Large-store files: **{len(inventory):,}**', f"- Large-store bytes: **{int(inventory['size_bytes'].sum()):,}**", '', '## prespecified branch-sufficiency gates', '', '| Epoch family | Before QC | Eligible | Overall | Imagery | Video | Eligible subjects | Pass |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for _, row in branch.iterrows():
        lines.append(f"| `{row['epoch_family']}` | {int(row['available_before_amplitude_qc'])} | {int(row['modeling_eligible_rows'])} | {row['overall_retention']:.4f} | {row['imagery_retention']:.4f} | {row['video_retention']:.4f} | {int(row['within_person_eligible_subjects'])} | {python_bool(row['branch_passed'])} |")
    recording_passed = recording_qc.groupby('raw_source_path')['recording_channel_passed'].apply(lambda values: int(values.map(python_bool).sum()))
    lines.extend(['', '## Recording QC', '', f'- Sources with at least 48 recording-good channels: **{int((recording_passed >= MINIMUM_GOOD_CHANNELS).sum())}/{len(recording_passed)}**', f'- Minimum recording-good channels: **{int(recording_passed.min())}**', f'- Median recording-good channels: **{float(recording_passed.median()):.1f}**', '', 'Bad channels are not interpolated. They remain NaN in the stored 64-channel arrays.', '', '## Decision', '', f"- At least one post-QC branch passed: **{summary['decision']['at_least_one_branch_passed']}**", f"- All source checkpoints complete: **{summary['decision']['all_source_checkpoints_complete']}**", f"- Threshold changed after amplitude: **{summary['decision']['threshold_changed_after_amplitude']}**", f"- Corrected master modified: **{summary['decision']['canonical_master_modified']}**", ''])
    path.write_text(''.join(lines) + '', encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, raw_eeg_epoch_folds_summary, raw_eeg_epoch_folds_contract, available, headers = load_inputs(project)
    epoch_manifest_sha256 = sha256_file(paths['epoch_manifest'])
    store_root = project / STORE_REL
    store_root.mkdir(parents=True, exist_ok=True)
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config_dir = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'recording_qc': qc / 'raw_eeg_preprocessing_qc_recording_channel_qc.csv', 'epoch_qc': tables / 'raw_eeg_preprocessing_qc_epoch_amplitude_qc.csv', 'array_inventory': tables / 'raw_eeg_preprocessing_qc_large_array_inventory.csv', 'branch_qc': qc / 'raw_eeg_preprocessing_qc_branch_retention_qc.csv', 'subject_qc': qc / 'raw_eeg_preprocessing_qc_subject_retention_qc.csv', 'summary': qc / 'raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_summary.json', 'contract': config_dir / 'raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_contract.json', 'discovery_multimodal_inference_policy': config_dir / 'S058_raw_eeg_preprocessing_qc_discovery_multimodal.json', 'report': reports / 'raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_report.md', 'log': logs / 'raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Phase-level outputs already exist. Use --force after review:\n  - ' + '\n  - '.join((str(path) for path in existing)))
    header_lookup = headers.set_index('raw_source_path')
    source_dirs = []
    print()
    print('=' * 100)
    print('PHASE raw_eeg_preprocessing_qc RAW-EEG PREPROCESSING/QC COMPLETED')
    print('=' * 100)
    print(f'Raw EDF sources:                {len(source_groups)}')
    print(f'raw_eeg_epoch_folds available epochs:           {len(available)}')
    print(f"Modeling-eligible epochs:       {summary['counts']['modeling_eligible_epochs']}")
    print(f"Large-store bytes:              {summary['counts']['large_store_bytes']}")
    print(f"At least one branch passed:     {decision['at_least_one_branch_passed']}")
    print(f'Large store:                    {store_root}')
    print(f"Report:                         {outputs['report']}")
    print(f'Runtime:                        {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    mne.set_log_level('ERROR')
    raise SystemExit(main())