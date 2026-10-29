# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase outcome_free_features — outcome-free full-signal EEG representation extraction.

Prerequisite
------------
raw_eeg_preprocessing_qc.1 established full byte and semantic integrity for the local raw_eeg_preprocessing_qc store:
- 627 files / 3,478,153,368 bytes;
- 57 source checkpoints;
- 171 source × epoch-family groups;
- 4,083 available epochs;
- 3,853 modeling-eligible epochs;
- every true mask channel finite;
- every false mask channel entirely NaN;
- all modeling loss explained by the prespecified recording gate.

outcome_free_features extracts representations only for raw_eeg_preprocessing_qc modeling-eligible epochs.
No report value, valence, liking, category, released derivative amplitude,
validation participant, or confirmation participant is read.

No global fitted transform
--------------------------
outcome_free_features performs deterministic per-epoch feature extraction only. It does not fit:
- imputation;
- standardization;
- PCA;
- supervised feature selection;
- covariance reference or tangent map;
- CORAL, Procrustes, or manifold alignment.

Those operations are permitted only inside the already prespecified outer-training
folds in later modeling.

prespecified feature blocks
---------------------
1. Time-frequency topography
   Six frequency bands × whole epoch plus four equal temporal quarters ×
   64 channels. Welch uses a Hann window, 2 seconds, 50% overlap, nfft=512.
   Features are log10 integrated band power.

2. Temporal morphology
   Six channel-wise features:
   - 5%/95% winsorized RMS in microvolts;
   - mean absolute first difference in microvolts/sample;
   - Hjorth mobility;
   - Hjorth complexity;
   - lag-1 autocorrelation;
   - normalized spectral entropy.

3. Spatial covariance
   A 64×64 covariance matrix in microvolt². Good-channel covariance uses
   fixed 0.10 shrinkage toward its diagonal. Rejected-channel rows and columns
   remain NaN. Outcome-free summaries contain the top 16 normalized
   eigenvalues, effective rank, participation ratio, regularized log
   determinant, and global-field-power median/IQR.

4. Phase synchrony
   For each prespecified band, a zero-phase Butterworth/Hilbert PLV matrix is
   computed on every fourth analytic-phase sample. Stored features are the
   mean PLV of each channel with the other good channels, plus global mean,
   standard deviation, 25th percentile, and 75th percentile.

5. Low-frequency waveform shape
   Each epoch is polyphase-resampled from 200 to 50 Hz and transformed with
   orthonormal DCT-II. The first 20 coefficients per channel are stored in
   microvolts.

Rejected channels remain NaN in every channel-wise block.

Storage and recovery
--------------------
outputs/large/outcome_free_features_outcome_free_features_v1/

Each raw EDF source is an atomic checkpoint. Before extraction or reuse, all
raw_eeg_preprocessing_qc input files for that source are rehashed against the raw_eeg_preprocessing_qc.1-verified
inventory. Valid feature checkpoints are reused only after full SHA-256
verification of every stored output file.
--force rewrites phase-level metadata but does not rebuild valid source
checkpoints. --recompute-sources is required to rebuild them.

contracts, but not the large feature arrays."""
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
from scipy.fft import dct
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
large_store_integrity_SUMMARY_REL = Path('outputs/qc/large_store_integrity_large_store_integrity_summary.json')
large_store_integrity_CONTRACT_REL = Path('config/large_store_integrity_large_store_integrity_contract.json')
large_store_integrity_POLICY_REL = Path('analysis_specifications/S035_large_store_integrity_discovery_multimodal_inference.json')
raw_eeg_preprocessing_qc_INVENTORY_REL = Path('outputs/tables/raw_eeg_preprocessing_qc_large_array_inventory.csv')
raw_eeg_preprocessing_qc_EPOCH_QC_REL = Path('outputs/tables/raw_eeg_preprocessing_qc_epoch_amplitude_qc.csv')
COMMON_MASK_REL = Path('outputs/tables/large_store_integrity_common_trial_mask_manifest.csv')
TRIAL_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_trial_fold_manifest.csv')
PARTICIPANT_FOLD_REL = Path('outputs/tables/raw_eeg_epoch_folds_participant_fold_manifest.csv')
FEATURE_STORE_REL = Path('outputs/large/outcome_free_features_outcome_free_features_v1')
SAMPLING_RATE_HZ = 200.0
CHANNEL_COUNT = 64
BANDS = [('delta', 1.0, 4.0), ('theta', 4.0, 8.0), ('alpha', 8.0, 13.0), ('low_beta', 13.0, 20.0), ('high_beta', 20.0, 30.0), ('low_gamma', 30.0, 45.0)]
TEMPORAL_BINS = ['whole_epoch', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
MORPHOLOGY_NAMES = ['winsorized_rms_uv', 'line_length_uv_per_sample', 'hjorth_mobility', 'hjorth_complexity', 'lag1_autocorrelation', 'normalized_spectral_entropy']
COVARIANCE_SUMMARY_NAMES = [*[f'normalized_eigenvalue_{index:02d}' for index in range(1, 17)], 'effective_rank', 'participation_ratio', 'regularized_log_determinant', 'global_field_power_median_uv', 'global_field_power_iqr_uv']
PLV_GLOBAL_NAMES = ['global_mean_plv', 'global_sd_plv', 'global_q25_plv', 'global_q75_plv']
DCT_COEFFICIENTS = 20
WELCH_WINDOW_SAMPLES = 400
WELCH_OVERLAP_SAMPLES = 200
WELCH_NFFT = 512
COVARIANCE_SHRINKAGE = 0.1
PLV_PHASE_STRIDE = 4
FEATURE_ROW_CHUNK = 1
EXPECTED = {'sources': 57, 'source_family_groups': 171, 'available_epochs': 4083, 'modeling_eligible_epochs': 3853, 'common_trial_rows': 1379}
IMPLEMENTATION_POLICY = {'sampling_rate_hz': SAMPLING_RATE_HZ, 'channel_count': CHANNEL_COUNT, 'frequency_bands_hz': {name: [low, high] for name, low, high in BANDS}, 'time_frequency': {'estimator': 'scipy.signal.welch', 'window': 'hann', 'window_samples': WELCH_WINDOW_SAMPLES, 'overlap_samples': WELCH_OVERLAP_SAMPLES, 'nfft': WELCH_NFFT, 'scaling': 'density', 'band_integration': 'numpy.trapz', 'transform': 'log10(max(power, float64_tiny))', 'temporal_bins': TEMPORAL_BINS, 'shape': [CHANNEL_COUNT, len(BANDS), len(TEMPORAL_BINS)]}, 'temporal_morphology': {'winsorized_rms': 'clip each channel to its 5th and 95th percentiles, then sqrt(mean(x^2)); report microvolts', 'line_length': 'mean(abs(first difference)); report microvolts/sample', 'hjorth_mobility': 'sqrt(var(first difference)/var(signal))', 'hjorth_complexity': 'mobility(first difference)/mobility(signal)', 'lag1_autocorrelation': 'Pearson correlation of x[:-1] and x[1:]', 'spectral_entropy': 'Shannon entropy of whole-epoch Welch bins from 0.5 to 45 Hz, normalized by log(number of bins)', 'shape': [CHANNEL_COUNT, len(MORPHOLOGY_NAMES)]}, 'spatial_covariance': {'units': 'microvolt_squared', 'estimator': 'sample covariance, ddof=1', 'shrinkage': COVARIANCE_SHRINKAGE, 'target': 'diagonal of sample covariance', 'rejected_channels': 'NaN rows and columns', 'shape': [CHANNEL_COUNT, CHANNEL_COUNT], 'summary_names': COVARIANCE_SUMMARY_NAMES}, 'phase_synchrony': {'filter': 'fourth-order Butterworth SOS per band, zero-phase sosfiltfilt', 'analytic_signal': 'scipy.signal.hilbert', 'phase_stride': PLV_PHASE_STRIDE, 'plv': 'abs(U @ U.conj().T / n_samples), diagonal excluded', 'channel_shape': [CHANNEL_COUNT, len(BANDS)], 'global_shape': [len(BANDS), len(PLV_GLOBAL_NAMES)]}, 'waveform_shape': {'input_units': 'microvolts', 'resampling': 'scipy.signal.resample_poly, up=1, down=4, padtype=line', 'target_sampling_rate_hz': 50, 'transform': 'scipy.fft.dct type=2 norm=ortho', 'coefficients': DCT_COEFFICIENTS, 'shape': [CHANNEL_COUNT, DCT_COEFFICIENTS]}, 'missing_channel_policy': 'rejected channels remain NaN in every channel-wise feature block', 'fitted_global_transforms': 'none'}
IMPLEMENTATION_POLICY_SHA256 = hashlib.sha256(json.dumps(IMPLEMENTATION_POLICY, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract five deterministic, outcome-free full-signal feature blocks from the raw_eeg_preprocessing_qc.1-verified EEG store.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--force', action='store_true', help='Rewrite phase-level metadata; valid source checkpoints are reused.')
    parser.add_argument('--recompute-sources', action='store_true', help='Rebuild valid source feature checkpoints after explicit review.')
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

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
    os.replace(temporary, path)

def integrate_band(frequencies: np.ndarray, density: np.ndarray, low_hz: float, high_hz: float, *, include_high: bool) -> np.ndarray:
    if include_high:
        selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    else:
        selected = (frequencies >= low_hz) & (frequencies < high_hz)
    if int(selected.sum()) < 2:
        raise RuntimeError(f'Insufficient Welch bins for {low_hz}-{high_hz} Hz.')
    power = np.trapz(density[..., selected], frequencies[selected], axis=-1)
    return np.log10(np.maximum(power, np.finfo(np.float64).tiny))

def equal_quarter_slices(sample_count: int) -> list[slice]:
    edges = np.rint(np.linspace(0, sample_count, 5)).astype(int)
    slices = [slice(int(edges[index]), int(edges[index + 1])) for index in range(4)]
    if any((value.stop - value.start < WELCH_WINDOW_SAMPLES for value in slices)):
        raise RuntimeError('A temporal quarter is shorter than the prespecified Welch window.')
    return slices

def time_frequency_features(good_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(good_data, dtype=np.float64)
    slices = [slice(0, data.shape[1]), *equal_quarter_slices(data.shape[1])]
    output = np.empty((data.shape[0], len(BANDS), len(TEMPORAL_BINS)), dtype=np.float64)
    whole_frequencies = None
    whole_density = None
    for bin_index, segment_slice in enumerate(slices):
        segment = data[:, segment_slice]
        frequencies, density = welch(segment, fs=SAMPLING_RATE_HZ, window='hann', nperseg=WELCH_WINDOW_SAMPLES, noverlap=WELCH_OVERLAP_SAMPLES, nfft=WELCH_NFFT, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        if bin_index == 0:
            whole_frequencies = frequencies
            whole_density = density
        for band_index, (_, low_hz, high_hz) in enumerate(BANDS):
            output[:, band_index, bin_index] = integrate_band(frequencies, density, low_hz, high_hz, include_high=band_index == len(BANDS) - 1)
    if whole_frequencies is None or whole_density is None:
        raise RuntimeError('Whole-epoch Welch output was not created.')
    return (output, whole_frequencies, whole_density)

def normalized_spectral_entropy(frequencies: np.ndarray, density: np.ndarray) -> np.ndarray:
    selected = (frequencies >= 0.5) & (frequencies <= 45.0)
    spectrum = np.maximum(density[:, selected], 0.0)
    total = spectrum.sum(axis=1, keepdims=True)
    probabilities = np.divide(spectrum, total, out=np.zeros_like(spectrum), where=total > 0)
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0), axis=1)
    denominator = math.log(spectrum.shape[1])
    return entropy / denominator

def temporal_morphology_features(good_data: np.ndarray, frequencies: np.ndarray, density: np.ndarray) -> np.ndarray:
    data = np.asarray(good_data, dtype=np.float64)
    low = np.quantile(data, 0.05, axis=1, keepdims=True)
    high = np.quantile(data, 0.95, axis=1, keepdims=True)
    winsorized = np.clip(data, low, high)
    rms_uv = np.sqrt(np.mean(winsorized ** 2, axis=1)) * 1000000.0
    first_difference = np.diff(data, axis=1)
    second_difference = np.diff(first_difference, axis=1)
    line_length_uv = np.mean(np.abs(first_difference), axis=1) * 1000000.0
    variance_signal = np.var(data, axis=1, ddof=1)
    variance_first = np.var(first_difference, axis=1, ddof=1)
    variance_second = np.var(second_difference, axis=1, ddof=1)
    mobility = np.sqrt(np.divide(variance_first, variance_signal, out=np.full_like(variance_first, np.nan), where=variance_signal > 0))
    mobility_first = np.sqrt(np.divide(variance_second, variance_first, out=np.full_like(variance_second, np.nan), where=variance_first > 0))
    complexity = np.divide(mobility_first, mobility, out=np.full_like(mobility_first, np.nan), where=mobility > 0)
    left = data[:, :-1]
    right = data[:, 1:]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered ** 2, axis=1) * np.sum(right_centered ** 2, axis=1))
    lag1 = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    entropy = normalized_spectral_entropy(frequencies, density)
    return np.column_stack([rms_uv, line_length_uv, mobility, complexity, lag1, entropy])

def covariance_features(good_data: np.ndarray, good_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data_uv = np.asarray(good_data, dtype=np.float64) * 1000000.0
    sample_covariance = np.cov(data_uv, rowvar=True, ddof=1)
    diagonal = np.diag(np.diag(sample_covariance))
    covariance = (1.0 - COVARIANCE_SHRINKAGE) * sample_covariance + COVARIANCE_SHRINKAGE * diagonal
    full = np.full((CHANNEL_COUNT, CHANNEL_COUNT), np.nan, dtype=np.float64)
    full[np.ix_(good_indices, good_indices)] = covariance
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigen_total = float(eigenvalues.sum())
    if eigen_total <= 0:
        raise RuntimeError('Covariance eigenvalue total is nonpositive.')
    normalized = eigenvalues / eigen_total
    positive = normalized[normalized > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation_ratio = float(1.0 / np.sum(normalized ** 2))
    regularizer = max(float(np.trace(covariance) / covariance.shape[0]) * 1e-12, 1e-12)
    sign, logdet = np.linalg.slogdet(covariance + regularizer * np.eye(covariance.shape[0]))
    if sign <= 0:
        raise RuntimeError('Regularized covariance log determinant is nonpositive.')
    global_field_power = np.std(data_uv, axis=0, ddof=1)
    top = np.full(16, np.nan, dtype=np.float64)
    top[:min(16, normalized.size)] = normalized[:16]
    summary = np.concatenate([top, np.asarray([effective_rank, participation_ratio, float(logdet), float(np.median(global_field_power)), float(np.quantile(global_field_power, 0.75) - np.quantile(global_field_power, 0.25))])])
    return (full, summary)

def phase_synchrony_features(good_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(good_data, dtype=np.float64)
    channel_output = np.empty((data.shape[0], len(BANDS)), dtype=np.float64)
    global_output = np.empty((len(BANDS), len(PLV_GLOBAL_NAMES)), dtype=np.float64)
    for band_index, (_, low_hz, high_hz) in enumerate(BANDS):
        sos = butter(4, [low_hz, high_hz], btype='bandpass', fs=SAMPLING_RATE_HZ, output='sos')
        bandpassed = sosfiltfilt(sos, data, axis=1)
        analytic = hilbert(bandpassed, axis=1)[:, ::PLV_PHASE_STRIDE]
        unit_phase = np.divide(analytic, np.abs(analytic), out=np.ones_like(analytic, dtype=np.complex128), where=np.abs(analytic) > 0)
        plv = np.abs(unit_phase @ unit_phase.conj().T / unit_phase.shape[1])
        np.fill_diagonal(plv, np.nan)
        channel_output[:, band_index] = np.nanmean(plv, axis=1)
        upper = plv[np.triu_indices(plv.shape[0], k=1)]
        global_output[band_index,] = np.asarray([float(np.mean(upper)), float(np.std(upper, ddof=1)), float(np.quantile(upper, 0.25)), float(np.quantile(upper, 0.75))])
    return (channel_output, global_output)

def waveform_dct_features(good_data: np.ndarray) -> np.ndarray:
    data_uv = np.asarray(good_data, dtype=np.float64) * 1000000.0
    downsampled = resample_poly(data_uv, 1, 4, axis=1, padtype='line')
    if downsampled.shape[1] < DCT_COEFFICIENTS:
        raise RuntimeError('Downsampled epoch is shorter than the prespecified DCT block.')
    coefficients = dct(downsampled, type=2, axis=1, norm='ortho')
    return coefficients[:, :DCT_COEFFICIENTS]

def extract_epoch_features(epoch: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(epoch, dtype=np.float64)
    good = np.asarray(mask, dtype=bool)
    if data.shape[0] != CHANNEL_COUNT:
        raise RuntimeError('Epoch does not have 64 channels.')
    if good.shape != (CHANNEL_COUNT,):
        raise RuntimeError('Channel mask does not have 64 entries.')
    if int(good.sum()) < 48:
        raise RuntimeError('A modeling-eligible epoch has fewer than 48 good channels.')
    if not np.isfinite(data[good]).all():
        raise RuntimeError('A mask-true channel contains a nonfinite sample.')
    if not np.isnan(data[~good]).all():
        raise RuntimeError('A mask-false channel is not entirely NaN.')
    good_indices = np.flatnonzero(good)
    good_data = data[good]
    tf_good, frequencies, density = time_frequency_features(good_data)
    morphology_good = temporal_morphology_features(good_data, frequencies, density)
    covariance, covariance_summary = covariance_features(good_data, good_indices)
    plv_good, plv_global = phase_synchrony_features(good_data)
    dct_good = waveform_dct_features(good_data)
    time_frequency = np.full((CHANNEL_COUNT, len(BANDS), len(TEMPORAL_BINS)), np.nan, dtype=np.float32)
    morphology = np.full((CHANNEL_COUNT, len(MORPHOLOGY_NAMES)), np.nan, dtype=np.float32)
    plv_channel = np.full((CHANNEL_COUNT, len(BANDS)), np.nan, dtype=np.float32)
    waveform = np.full((CHANNEL_COUNT, DCT_COEFFICIENTS), np.nan, dtype=np.float32)
    time_frequency[good] = np.asarray(tf_good, dtype=np.float32)
    morphology[good] = np.asarray(morphology_good, dtype=np.float32)
    plv_channel[good] = np.asarray(plv_good, dtype=np.float32)
    waveform[good] = np.asarray(dct_good, dtype=np.float32)
    return {'time_frequency': time_frequency, 'morphology': morphology, 'covariance': np.asarray(covariance, dtype=np.float32), 'covariance_summary': np.asarray(covariance_summary, dtype=np.float32), 'plv_channel': plv_channel, 'plv_global': np.asarray(plv_global, dtype=np.float32), 'waveform_dct': waveform}

def load_inputs(project: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {'large_store_integrity_summary': project / large_store_integrity_SUMMARY_REL, 'large_store_integrity_contract': project / large_store_integrity_CONTRACT_REL, 'large_store_integrity_policy': project / large_store_integrity_POLICY_REL, 'array_inventory': project / raw_eeg_preprocessing_qc_INVENTORY_REL, 'epoch_qc': project / raw_eeg_preprocessing_qc_EPOCH_QC_REL, 'common_masks': project / COMMON_MASK_REL, 'trial_folds': project / TRIAL_FOLD_REL, 'participant_folds': project / PARTICIPANT_FOLD_REL}
    require_files(paths.values())
    summary = read_json(paths['large_store_integrity_summary'])
    contract = read_json(paths['large_store_integrity_contract'])
    policy = read_json(paths['large_store_integrity_policy'])
    if summary.get('phase') != 'large_store_integrity':
        raise RuntimeError('Unexpected raw_eeg_preprocessing_qc.1 prerequisite.')
    required_decisions = ['large_store_byte_integrity_passed', 'array_mask_semantic_integrity_passed', 'discovery_multimodal_inference_representation_extraction_permitted']
    for key in required_decisions:
        if not bool(summary['decision'][key]):
            raise RuntimeError(f'raw_eeg_preprocessing_qc.1 prerequisite failed: {key}.')
    if bool(summary['decision']['outcome_values_read']):
        raise RuntimeError('raw_eeg_preprocessing_qc.1 unexpectedly opened outcomes.')
    feature_policy = policy['feature_policy']
    if canonical_sha256(feature_policy) != contract['feature_policy_sha256']:
        raise RuntimeError('raw_eeg_preprocessing_qc.1 feature-policy hash does not match its contract.')
    for logical_name, contract_key in [('array_inventory', 'array_inventory'), ('epoch_qc', 'epoch_qc')]:
        observed = sha256_file(paths[logical_name])
        expected = contract['input_hashes'][contract_key]
        if observed != expected:
            raise RuntimeError(f'{logical_name}: hash changed after raw_eeg_preprocessing_qc.1.')
    observed_common = sha256_file(paths['common_masks'])
    expected_common = contract['output_hashes']['common_trial_masks']
    if observed_common != expected_common:
        raise RuntimeError('raw_eeg_preprocessing_qc.1 common-row manifest hash changed.')
    inventory = pd.read_csv(paths['array_inventory'], low_memory=False)
    epoch_qc = pd.read_csv(paths['epoch_qc'], low_memory=False)
    common_masks = pd.read_csv(paths['common_masks'], low_memory=False)
    trial_folds = pd.read_csv(paths['trial_folds'], low_memory=False)
    participant_folds = pd.read_csv(paths['participant_folds'], low_memory=False)
    if len(inventory['source_store_directory'].unique()) != EXPECTED['sources']:
        raise RuntimeError('Expected 57 source stores.')
    if len(epoch_qc) != EXPECTED['available_epochs']:
        raise RuntimeError('Expected 4,083 raw_eeg_preprocessing_qc epoch-QC rows.')
    if int(epoch_qc['modeling_eligible'].map(python_bool).sum()) != EXPECTED['modeling_eligible_epochs']:
        raise RuntimeError('Expected 3,853 modeling-eligible epochs.')
    if len(common_masks) != EXPECTED['common_trial_rows']:
        raise RuntimeError('Expected 1,379 common-trial rows.')
    return (paths, summary, contract, policy, inventory, epoch_qc, common_masks, trial_folds, participant_folds)

def inventory_path(source_inventory: pd.DataFrame, relative_path: str) -> Path:
    matches = source_inventory.loc[source_inventory['relative_path'].eq(relative_path), 'stored_path']
    if len(matches) != 1:
        raise RuntimeError(f'Expected one input file {relative_path}; observed {len(matches)}.')
    path = Path(str(matches.iloc[0]))
    if not path.is_file():
        raise RuntimeError(f'Input store file is missing: {path}')
    return path

def source_feature_checkpoint_valid(source_dir: Path, *, input_signature_sha256: str) -> bool:
    summary_path = source_dir / 'source_feature_summary.json'
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except Exception:
        return False
    if summary.get('complete') is not True:
        return False
    if summary.get('implementation_policy_sha256') != IMPLEMENTATION_POLICY_SHA256:
        return False
    if summary.get('input_signature_sha256') != input_signature_sha256:
        return False
    for item in summary.get('stored_files', []):
        path = source_dir / item['relative_path']
        if not path.is_file():
            return False
        if int(path.stat().st_size) != int(item['size_bytes']):
            return False
        if sha256_file(path) != item['sha256']:
            return False
    return True

def feature_shapes(row_count: int) -> dict[str, tuple[int, ...]]:
    return {'time_frequency': (row_count, CHANNEL_COUNT, len(BANDS), len(TEMPORAL_BINS)), 'morphology': (row_count, CHANNEL_COUNT, len(MORPHOLOGY_NAMES)), 'covariance': (row_count, CHANNEL_COUNT, CHANNEL_COUNT), 'covariance_summary': (row_count, len(COVARIANCE_SUMMARY_NAMES)), 'plv_channel': (row_count, CHANNEL_COUNT, len(BANDS)), 'plv_global': (row_count, len(BANDS), len(PLV_GLOBAL_NAMES)), 'waveform_dct': (row_count, CHANNEL_COUNT, DCT_COEFFICIENTS)}

def open_feature_memmaps(directory: Path, row_count: int) -> dict[str, np.memmap]:
    maps = {}
    for block_name, shape in feature_shapes(row_count).items():
        maps[block_name] = np.lib.format.open_memmap(directory / f'{block_name}_float32.npy', mode='w+', dtype=np.float32, shape=shape)
        maps[block_name][:] = np.nan
    return maps

def process_source(source_store_directory: str, source_inventory: pd.DataFrame, epoch_qc: pd.DataFrame, common_masks: pd.DataFrame, trial_folds: pd.DataFrame, participant_folds: pd.DataFrame, *, feature_store_root: Path, recompute: bool) -> Path:
    source_input = Path(source_store_directory)
    subject_values = source_inventory['subject_id'].astype(str).unique()
    run_values = source_inventory['raw_run'].astype(int).unique()
    raw_path_values = source_inventory['raw_source_path'].astype(str).unique()
    raw_hash_values = source_inventory['raw_source_sha256'].astype(str).unique()
    if len(subject_values) != 1 or len(run_values) != 1 or len(raw_path_values) != 1 or (len(raw_hash_values) != 1):
        raise RuntimeError(f'Input inventory identity is not unique: {source_store_directory}.')
    subject_id = subject_values[0]
    raw_run = int(run_values[0])
    raw_source_path = raw_path_values[0]
    raw_source_sha256 = raw_hash_values[0]
    for inventory_row in source_inventory.to_dict(orient='records'):
        input_path = Path(str(inventory_row['stored_path']))
        if not input_path.is_file():
            raise RuntimeError(f'raw_eeg_preprocessing_qc input file is missing: {input_path}')
        if int(input_path.stat().st_size) != int(inventory_row['size_bytes']):
            raise RuntimeError(f'raw_eeg_preprocessing_qc input file size changed: {input_path}')
        if sha256_file(input_path) != inventory_row['sha256']:
            raise RuntimeError(f'raw_eeg_preprocessing_qc input file SHA-256 changed after raw_eeg_preprocessing_qc.1: {input_path}')
    slug = source_input.name
    subject_root = feature_store_root / subject_id
    subject_root.mkdir(parents=True, exist_ok=True)
    final_dir = subject_root / slug
    input_signature = {'source_store_directory': source_store_directory, 'raw_source_path': raw_source_path, 'raw_source_sha256': raw_source_sha256, 'source_input_files': sorted([{'relative_path': row['relative_path'], 'size_bytes': int(row['size_bytes']), 'sha256': row['sha256']} for row in source_inventory.to_dict(orient='records')], key=lambda value: value['relative_path'])}
    input_signature_sha256 = canonical_sha256(input_signature)
    if not recompute and source_feature_checkpoint_valid(final_dir, input_signature_sha256=input_signature_sha256):
        return final_dir
    temporary_dir = subject_root / ('.' + slug + '.tmp')
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    qc_rows = []
    try:
        for epoch_family in ['stimulus_early', 'stimulus_late', 'rating_early']:
            data_path = inventory_path(source_inventory, f'{epoch_family}_epochs_float32.npy')
            mask_path = inventory_path(source_inventory, f'{epoch_family}_channel_mask_bool.npy')
            index_path = inventory_path(source_inventory, f'{epoch_family}_index.csv')
            source_index = pd.read_csv(index_path, low_memory=False)
            source_qc = epoch_qc.loc[epoch_qc['raw_source_path'].eq(raw_source_path) & epoch_qc['epoch_family'].eq(epoch_family)].sort_values('source_array_row_0based').reset_index(drop=True)
            if source_index['epoch_id'].astype(str).tolist() != source_qc['epoch_id'].astype(str).tolist():
                raise RuntimeError(f'{subject_id} run {raw_run} {epoch_family}: input index/QC order differs.')
            eligible_qc = source_qc.loc[source_qc['modeling_eligible'].map(python_bool)].copy()
            eligible_rows = eligible_qc['source_array_row_0based'].astype(int).tolist()
            feature_index = eligible_qc[['epoch_id', 'raw_behavior_row_id', 'subject_id', 'context', 'raw_run', 'raw_source_path', 'epoch_family', 'source_array_row_0based', 'final_good_channels']].copy()
            feature_index['feature_row_0based'] = np.arange(len(feature_index), dtype=int)
            comparison_columns = ['stimulus_early', 'stimulus_late', 'rating_early', 'stimulus_early_vs_late', 'stimulus_early_vs_rating', 'stimulus_late_vs_rating', 'all_three_epoch_families']
            feature_index = feature_index.merge(common_masks[['raw_behavior_row_id', *comparison_columns]], on='raw_behavior_row_id', how='left', validate='many_to_one')
            feature_index = feature_index.merge(trial_folds[['raw_behavior_row_id', 'subject_acquisition_order_1based', 'within_subject_block_fold', 'cross_person_fold']], on='raw_behavior_row_id', how='left', validate='many_to_one')
            feature_index = feature_index.merge(participant_folds[['subject_id', 'cross_person_fold']].rename(columns={'cross_person_fold': 'participant_fold_check'}), on='subject_id', how='left', validate='many_to_one')
            if not feature_index['cross_person_fold'].eq(feature_index['participant_fold_check']).all():
                raise RuntimeError('Trial and participant cross-person folds differ.')
            if feature_index[comparison_columns].isna().any().any():
                raise RuntimeError('A feature row lacks a prespecified common-trial mask.')
            if not feature_index[epoch_family].map(python_bool).all():
                raise RuntimeError(f'A modeling-eligible {epoch_family} row is false in its prespecified branch mask.')
            feature_index = feature_index.drop(columns=['participant_fold_check'])
            feature_index.to_csv(temporary_dir / f'{epoch_family}_feature_index.csv', index=False)
            family_dir = temporary_dir / epoch_family
            family_dir.mkdir(parents=True, exist_ok=False)
            feature_maps = open_feature_memmaps(family_dir, len(feature_index))
            data = np.load(data_path, mmap_mode='r', allow_pickle=False)
            mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
            started_family = time.time()
            for feature_row, source_row in enumerate(eligible_rows):
                features = extract_epoch_features(data[source_row], mask[source_row])
                for block_name, values in features.items():
                    feature_maps[block_name][feature_row] = values
                if (feature_row + 1) % 10 == 0 or feature_row + 1 == len(eligible_rows):
                    print(f'[{now_iso()}] {subject_id} run {raw_run} {epoch_family}: {feature_row + 1}/{len(eligible_rows)} epochs.', flush=True)
            for feature_map in feature_maps.values():
                feature_map.flush()
            nonfinite_good_cells = 0
            feature_mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
            for feature_row, source_row in enumerate(eligible_rows):
                good = np.asarray(feature_mask[source_row], dtype=bool)
                for block_name in ['time_frequency', 'morphology', 'plv_channel', 'waveform_dct']:
                    values = np.asarray(feature_maps[block_name][feature_row])
                    if not np.isfinite(values[good]).all():
                        nonfinite_good_cells += 1
                    if not np.isnan(values[~good]).all():
                        raise RuntimeError(f'{epoch_family} {block_name}: rejected channel feature cells are not all NaN.')
                covariance = np.asarray(feature_maps['covariance'][feature_row])
                if not np.isfinite(covariance[np.ix_(good, good)]).all():
                    nonfinite_good_cells += 1
                if not np.isnan(covariance[~good, :]).all() or not np.isnan(covariance[:, ~good]).all():
                    raise RuntimeError('Rejected covariance rows/columns are not all NaN.')
                if not np.isfinite(feature_maps['covariance_summary'][feature_row]).all():
                    nonfinite_good_cells += 1
                if not np.isfinite(feature_maps['plv_global'][feature_row]).all():
                    nonfinite_good_cells += 1
            if nonfinite_good_cells != 0:
                raise RuntimeError(f'{epoch_family}: nonfinite values occurred in valid feature cells.')
            qc_rows.append({'subject_id': subject_id, 'raw_run': raw_run, 'raw_source_path': raw_source_path, 'epoch_family': epoch_family, 'available_input_rows': int(len(source_qc)), 'modeling_eligible_rows': int(len(feature_index)), 'excluded_input_rows': int(len(source_qc) - len(feature_index)), 'feature_blocks': int(len(feature_maps)), 'nonfinite_good_feature_checks': nonfinite_good_cells, 'rejected_channel_nan_policy_passed': True, 'runtime_seconds': round(time.time() - started_family, 3)})
        qc_frame = pd.DataFrame(qc_rows)
        qc_frame.to_csv(temporary_dir / 'source_feature_qc.csv', index=False)
        stored_files = []
        for file_path in sorted(temporary_dir.rglob('*')):
            if not file_path.is_file():
                continue
            if file_path.name == 'source_feature_summary.json':
                continue
            stored_files.append({'relative_path': str(file_path.relative_to(temporary_dir)), 'size_bytes': int(file_path.stat().st_size), 'sha256': sha256_file(file_path)})
        source_feature_summary = {'generated_at': now_iso(), 'complete': True, 'phase': 'outcome_free_features', 'implementation_policy_sha256': IMPLEMENTATION_POLICY_SHA256, 'input_signature_sha256': input_signature_sha256, 'subject_id': subject_id, 'raw_run': raw_run, 'raw_source_path': raw_source_path, 'source_input_store_directory': source_store_directory, 'available_input_epochs': int(qc_frame['available_input_rows'].sum()), 'modeling_eligible_feature_rows': int(qc_frame['modeling_eligible_rows'].sum()), 'stored_files': stored_files}
        atomic_json(temporary_dir / 'source_feature_summary.json', source_feature_summary)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(temporary_dir, final_dir)
        return final_dir
    except Exception:
        if temporary_dir.exists():
            (temporary_dir / 'FAILED.txt').write_text(f'Failed at {now_iso()}.\nThis is not a valid completed feature checkpoint.', encoding='utf-8')
        raise

def collect_outputs(source_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows = []
    qc_frames = []
    for source_dir in source_dirs:
        summary = read_json(source_dir / 'source_feature_summary.json')
        qc_frames.append(pd.read_csv(source_dir / 'source_feature_qc.csv', low_memory=False))
        for item in summary['stored_files']:
            path = source_dir / item['relative_path']
            inventory_rows.append({'subject_id': summary['subject_id'], 'raw_run': int(summary['raw_run']), 'raw_source_path': summary['raw_source_path'], 'source_feature_directory': str(source_dir), 'stored_path': str(path), 'relative_path': item['relative_path'], 'size_bytes': int(item['size_bytes']), 'sha256': item['sha256'], 'implementation_policy_sha256': summary['implementation_policy_sha256'], 'input_signature_sha256': summary['input_signature_sha256']})
    return (pd.DataFrame(inventory_rows), pd.concat(qc_frames, ignore_index=True, sort=False))

def feature_dimension_table() -> pd.DataFrame:
    rows = [{'feature_block': 'time_frequency', 'per_epoch_shape': f'{CHANNEL_COUNT}x{len(BANDS)}x{len(TEMPORAL_BINS)}', 'scalar_cells_per_epoch': CHANNEL_COUNT * len(BANDS) * len(TEMPORAL_BINS), 'units': 'log10 volt_squared', 'channel_wise': True}, {'feature_block': 'morphology', 'per_epoch_shape': f'{CHANNEL_COUNT}x{len(MORPHOLOGY_NAMES)}', 'scalar_cells_per_epoch': CHANNEL_COUNT * len(MORPHOLOGY_NAMES), 'units': 'mixed; defined in implementation policy', 'channel_wise': True}, {'feature_block': 'covariance', 'per_epoch_shape': f'{CHANNEL_COUNT}x{CHANNEL_COUNT}', 'scalar_cells_per_epoch': CHANNEL_COUNT * CHANNEL_COUNT, 'units': 'microvolt_squared', 'channel_wise': False}, {'feature_block': 'covariance_summary', 'per_epoch_shape': str(len(COVARIANCE_SUMMARY_NAMES)), 'scalar_cells_per_epoch': len(COVARIANCE_SUMMARY_NAMES), 'units': 'mixed; defined in implementation policy', 'channel_wise': False}, {'feature_block': 'plv_channel', 'per_epoch_shape': f'{CHANNEL_COUNT}x{len(BANDS)}', 'scalar_cells_per_epoch': CHANNEL_COUNT * len(BANDS), 'units': 'dimensionless', 'channel_wise': True}, {'feature_block': 'plv_global', 'per_epoch_shape': f'{len(BANDS)}x{len(PLV_GLOBAL_NAMES)}', 'scalar_cells_per_epoch': len(BANDS) * len(PLV_GLOBAL_NAMES), 'units': 'dimensionless', 'channel_wise': False}, {'feature_block': 'waveform_dct', 'per_epoch_shape': f'{CHANNEL_COUNT}x{DCT_COEFFICIENTS}', 'scalar_cells_per_epoch': CHANNEL_COUNT * DCT_COEFFICIENTS, 'units': 'microvolt', 'channel_wise': True}]
    frame = pd.DataFrame(rows)
    frame['total_scalar_cells_all_blocks'] = int(frame['scalar_cells_per_epoch'].sum())
    return frame

def write_report(path: Path, summary: dict[str, Any], feature_qc: pd.DataFrame, dimensions: pd.DataFrame) -> None:
    lines = ['# Phase outcome_free_features — Outcome-free full-signal representations', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Outcomes/report values read: **no**', '- Validation/confirmation opened: **no**', '- Globally fitted transform: **none**', '', '## Extraction', '', f"- Source checkpoints: **{summary['counts']['source_checkpoints']}**", f"- Source/family groups: **{summary['counts']['source_family_groups']}**", f"- Modeling-eligible epochs represented: **{summary['counts']['feature_epoch_rows']}**", f"- Feature-store files: **{summary['counts']['feature_store_files']}**", f"- Feature-store bytes: **{summary['counts']['feature_store_bytes']:,}**", '', '## prespecified blocks', '', '| Block | Shape per epoch | Scalar cells | Units |', '|---|---|---:|---|']
    for _, row in dimensions.iterrows():
        lines.append(f"| `{row['feature_block']}` | `{row['per_epoch_shape']}` | {int(row['scalar_cells_per_epoch'])} | {row['units']} |")
    family_counts = feature_qc.groupby('epoch_family')['modeling_eligible_rows'].sum()
    lines.extend(['', '## Epoch-family rows', ''])
    for family, count in family_counts.items():
        lines.append(f'- `{family}`: **{int(count)}** rows.')
    decision = summary['decision']
    lines.extend(['', '## Decision', '', f"- All feature checkpoints complete: **{decision['all_feature_checkpoints_complete']}**", f"- All eligible epochs represented: **{decision['all_modeling_eligible_epochs_represented']}**", f"- Rejected-channel NaN policy passed: **{decision['rejected_channel_nan_policy_passed']}**", f"- Global fitted transform used: **{decision['global_fitted_transform_used']}**", f"- Outcome attachment permitted for next discovery-only model-specification phase: **{decision['outcome_model_integrity_discovery_model_specification_permitted']}**", '', 'The feature store is a deterministic representation layer, not a biological ground truth. Imputation, standardization, PCA, covariance tangent mapping, alignment, and feature selection remain outer-fold-contained tasks for later modeling.', ''])
    path.write_text(''.join(lines) + '', encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    project = args.project.expanduser().resolve()
    paths, large_store_integrity_summary, large_store_integrity_contract, large_store_integrity_policy, inventory, epoch_qc, common_masks, trial_folds, participant_folds = load_inputs(project)
    feature_store_root = project / FEATURE_STORE_REL
    feature_store_root.mkdir(parents=True, exist_ok=True)
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config_dir = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'feature_inventory': tables / 'outcome_free_features_feature_store_inventory.csv', 'feature_qc': qc / 'outcome_free_features_feature_extraction_qc.csv', 'feature_dimensions': qc / 'outcome_free_features_feature_dimensions.csv', 'summary': qc / 'outcome_free_features_outcome_free_feature_summary.json', 'contract': config_dir / 'outcome_free_features_outcome_free_feature_contract.json', 'outcome_model_integrity_policy': config_dir / 'outcome_free_features_prespecified_outcome_model_integrity_model_specification_policy.json', 'report': reports / 'outcome_free_features_outcome_free_feature_report.md', 'log': logs / 'outcome_free_features_outcome_free_feature.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Phase outputs already exist. Use --force after review:\n  - ' + '\n  - '.join((str(path) for path in existing)))
    source_dirs = []
    source_groups = list(inventory.groupby('source_store_directory', sort=True))
    with outputs['log'].open('w', encoding='utf-8') as log:

        def note(message: str) -> None:
            line = f'[{now_iso()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note(f'Starting {len(source_groups)} resumable source feature checkpoints under implementation policy {IMPLEMENTATION_POLICY_SHA256}.')
        for source_index, (source_store_directory, source_inventory) in enumerate(source_groups, start=1):
            note(f'[{source_index}/{len(source_groups)}] Processing or reusing {source_store_directory}.')
            source_dir = process_source(source_store_directory, source_inventory, epoch_qc, common_masks, trial_folds, participant_folds, feature_store_root=feature_store_root, recompute=args.recompute_sources)
            source_dirs.append(source_dir)
        note('Collecting feature-store inventory and source/family QC.')
        feature_inventory, feature_qc = collect_outputs(source_dirs)
        dimensions = feature_dimension_table()
        feature_inventory.to_csv(outputs['feature_inventory'], index=False)
        feature_qc.to_csv(outputs['feature_qc'], index=False)
        dimensions.to_csv(outputs['feature_dimensions'], index=False)
        represented = int(feature_qc['modeling_eligible_rows'].sum())
        if len(source_dirs) != EXPECTED['sources']:
            raise RuntimeError('Feature source checkpoint count is incomplete.')
        if len(feature_qc) != EXPECTED['source_family_groups']:
            raise RuntimeError('Feature source/family QC count is incomplete.')
        if represented != EXPECTED['modeling_eligible_epochs']:
            raise RuntimeError(f'Expected 3,853 represented epochs; observed {represented}.')
        if not feature_qc['rejected_channel_nan_policy_passed'].map(python_bool).all():
            raise RuntimeError('Rejected-channel NaN policy failed.')
        if not feature_qc['nonfinite_good_feature_checks'].eq(0).all():
            raise RuntimeError('A valid feature cell is nonfinite.')
        decision = {'all_feature_checkpoints_complete': True, 'all_modeling_eligible_epochs_represented': True, 'rejected_channel_nan_policy_passed': True, 'global_fitted_transform_used': False, 'outcome_values_read': False, 'category_values_read': False, 'released_derivative_amplitude_read': False, 'validation_opened': False, 'confirmation_opened': False, 'outcome_model_integrity_discovery_model_specification_permitted': True}
        summary = {'generated_at': now_iso(), 'phase': 'outcome_free_features', 'status': 'outcome_free_full_signal_representation_extraction_complete_candidate', 'implementation_policy_sha256': IMPLEMENTATION_POLICY_SHA256, 'counts': {'source_checkpoints': int(len(source_dirs)), 'source_family_groups': int(len(feature_qc)), 'feature_epoch_rows': represented, 'feature_store_files': int(len(feature_inventory)), 'feature_store_bytes': int(feature_inventory['size_bytes'].sum()), 'scalar_feature_cells_per_epoch': int(dimensions['scalar_cells_per_epoch'].sum())}, 'epoch_family_rows': {str(key): int(value) for key, value in feature_qc.groupby('epoch_family')['modeling_eligible_rows'].sum().to_dict().items()}, 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        contract = {**summary, 'implementation_policy': IMPLEMENTATION_POLICY, 'feature_store_root': str(feature_store_root), 'large_features_embedded_in_release': False, 'resume_policy': 'valid source feature checkpoints are reused only after full SHA-256 verification of every stored file', 'input_hashes': {name: sha256_file(path) for name, path in paths.items()}, 'output_hashes': {name: sha256_file(path) for name, path in outputs.items() if name not in {'contract', 'outcome_model_integrity_policy', 'report', 'log'} and path.is_file()}, 'source_dataset_modified': False, 'raw_eeg_preprocessing_qc_large_store_modified': False, 'canonical_master_modified': False}
        outputs['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        outcome_model_integrity_policy = {'generated_at': now_iso(), 'phase': 'outcome_free_features', 'next_phase': 'primary_model_design', 'next_phase_permitted': True, 'feature_contract': str(outputs['contract']), 'feature_inventory': str(outputs['feature_inventory']), 'feature_qc': str(outputs['feature_qc']), 'common_trial_masks': str(paths['common_masks']), 'trial_folds': str(paths['trial_folds']), 'participant_folds': str(paths['participant_folds']), 'next_phase_role': 'discovery-only outcome attachment and nested model-analysis specification; not model fitting or result selection', 'required_modeling_principles': ['use only outcome_free_features feature rows', 'branch models use their branch-specific masks', 'cross-epoch comparisons use the prespecified same-row masks', 'all imputation, scaling, dimensionality reduction, covariance reference, tangent mapping, and alignment are fitted inside outer-training folds only', 'compare models on identical evaluation rows', 'stimulus and rating branches remain separately reported', 'within-person decoding precedes person-specific implementation claims', 'no universal biosignal or biological ground-truth claim', 'validation and confirmation remain closed']}
        outputs['outcome_model_integrity_policy'].write_text(json.dumps(outcome_model_integrity_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(outputs['report'], summary, feature_qc, dimensions)
        note('outcome_free_features completed. No outcome or validation/confirmation data was read.')
    print()
    print('=' * 100)
    print('PHASE outcome_free_features OUTCOME-FREE REPRESENTATION EXTRACTION COMPLETED')
    print('=' * 100)
    print(f'Source checkpoints:             {len(source_dirs)}')
    print(f'Source/family groups:           {len(feature_qc)}')
    print(f'Feature epoch rows:             {represented}')
    print(f"Scalar cells per epoch:         {int(dimensions['scalar_cells_per_epoch'].sum())}")
    print(f"Feature-store bytes:            {int(feature_inventory['size_bytes'].sum())}")
    print(f'Feature store:                  {feature_store_root}')
    print(f"Report:                         {outputs['report']}")
    print(f'Runtime:                        {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    raise SystemExit(main())