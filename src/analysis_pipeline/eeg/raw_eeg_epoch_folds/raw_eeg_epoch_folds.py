# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase raw_eeg_epoch_folds — raw-EEG epoch, coverage, and fold specification.

Prerequisite
------------
derivative_closure closed released-derivative trial attachment and froze a corrected,
layered discovery master. This phase uses only rows marked
continuous_raw_eeg_analysis_eligible.

No EEG sample amplitude is read.

Fixed epoch families
--------------------
stimulus_early
    [stimulus + 2 s, stimulus + 12 s], duration 10 s.

stimulus_late
    [source-supported endpoint - 25 s, endpoint - 5 s], duration 20 s.
    Video endpoint is rating onset.
    Imagery endpoint is the unique fade between stimulus and rating; when no
    fade exists, rating onset is the prespecified fallback. Multiple fades make this
    epoch unavailable. No fade is selected after seeing EEG.

rating_early
    [rating + 5 s, rating + 25 s], duration 20 s.
    It is available only when the entire interval ends at least 5 s before the
    next authoritative stimulus in the same EDF, or 5 s before recording end
    for the last authoritative trial in that EDF.

These windows are fixed before amplitude access. Coverage does not trigger a
shorter or shifted replacement window.

Fixed coverage gates
--------------------
- stimulus_early: overall >= .995 and each context >= .990;
- stimulus_late: overall >= .950 and each context >= .900;
- rating_early: overall >= .900 and each context >= .850.

A failed branch is bfixed; another window is not selected automatically.

Outcome-free folds
------------------
- Within-person outer folds: five contiguous acquisition-order blocks.
- Cross-person outer folds: deterministic greedy balancing by eligible row
  count and context count only.
- All epoch families from one trial share the same folds.
- No report value, valence, liking, category, or EEG amplitude contributes to
  fold construction.

The resulting contract is the sole prerequisite for raw_eeg_preprocessing_qc raw-amplitude
preprocessing. Validation and confirmation remain closed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import mne
import numpy as np
import pandas as pd
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
derivative_closure_SUMMARY_REL = Path('outputs/qc/derivative_closure_derivative_closure_canonical_master_summary.json')
derivative_closure_CONTRACT_REL = Path('config/derivative_closure_derivative_closure_canonical_master_contract.json')
MASTER_REL = Path('outputs/tables/derivative_closure_corrected_discovery_master_v1.csv')
ATTACHMENT_REL = Path('outputs/tables/derivative_closure_raw_event_source_attachment.csv')
ANCHOR_REL = Path('outputs/qc/derivative_closure_exact_anchor_adjudication.csv')
GLOBAL_EVENTS_REL = Path('outputs/tables/eeg_events_global_time.csv')
EPOCH_SPECS = {'stimulus_early': {'duration_sec': 10.0, 'overall_gate': 0.995, 'context_gate': 0.99, 'output_sampling_rate_hz': 200}, 'stimulus_late': {'duration_sec': 20.0, 'overall_gate': 0.95, 'context_gate': 0.9, 'output_sampling_rate_hz': 200}, 'rating_early': {'duration_sec': 20.0, 'overall_gate': 0.9, 'context_gate': 0.85, 'output_sampling_rate_hz': 200}}
WITHIN_PERSON_FOLDS = 5
CROSS_PERSON_FOLDS = 5
MIN_SUBJECT_TOTAL = 16
MIN_SUBJECT_PER_CONTEXT = 6
TIME_TOLERANCE_SEC = 1e-06

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='specification raw-EEG epoch windows, temporal coverage, and outcome-free folds without reading sample amplitude.')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
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

def normalize_subject(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith('sub-'):
        suffix = text[4:]
        if suffix.isdigit():
            return f'sub-{int(suffix):02d}'
    if text.isdigit():
        return f'sub-{int(text):02d}'
    return text

def normalize_event_class(value: object) -> str:
    return str(value).strip().lower()

def quantize_nearest(seconds: float, sampling_rate_hz: float) -> int:
    return int(np.rint(float(seconds) * float(sampling_rate_hz)))

def read_raw_headers(source_paths: list[str]) -> pd.DataFrame:
    rows = []
    for source_text in sorted(set(source_paths)):
        source_path = Path(source_text)
        if not source_path.is_file():
            raise RuntimeError(f'Raw EDF does not exist: {source_path}')
        raw = mne.io.read_raw_edf(source_path, preload=False, verbose='ERROR')
        sfreq = float(raw.info['sfreq'])
        n_times = int(raw.n_times)
        channel_names = list(raw.ch_names)
        rows.append({'raw_source_path': str(source_path), 'source_size_bytes': int(source_path.stat().st_size), 'source_mtime_ns': int(source_path.stat().st_mtime_ns), 'sampling_rate_hz': sfreq, 'n_times': n_times, 'recording_duration_sec': float(n_times / sfreq), 'channel_count': int(len(channel_names)), 'channel_order_sha256': hashlib.sha256(''.join(channel_names).encode('utf-8')).hexdigest(), 'highpass_hz': float(raw.info.get('highpass', math.nan)), 'lowpass_hz': float(raw.info.get('lowpass', math.nan)), 'line_frequency_hz': raw.info.get('line_freq', None), 'annotation_count': int(len(raw.annotations)), 'header_readable': True, 'sample_amplitude_read': False})
        raw.close()
    return pd.DataFrame(rows)

def authoritative_trials(master: pd.DataFrame) -> pd.DataFrame:
    required = {'raw_behavior_row_id', 'subject_id', 'split', 'trial_number', 'context', 'continuous_raw_eeg_analysis_eligible', 'continuous_event_authoritative', 'source_stimulus_run', 'source_stimulus_path', 'source_stimulus_annotation_index', 'source_stimulus_local_onset_sec', 'source_rating_run', 'source_rating_path', 'source_rating_annotation_index', 'source_rating_local_onset_sec', 'stimulus_onset_global_sec', 'rating_onset_global_sec'}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise RuntimeError('Corrected master lacks columns: ' + ', '.join(missing))
    trial = master.loc[master['continuous_raw_eeg_analysis_eligible'].map(python_bool), sorted(required)].copy()
    trial['subject_id'] = trial['subject_id'].map(normalize_subject)
    if len(trial) != 1379:
        raise RuntimeError(f'Expected 1,379 raw-EEG eligible rows; observed {len(trial)}.')
    if trial['subject_id'].nunique() != 34:
        raise RuntimeError('Expected 34 raw-EEG eligible discovery subjects.')
    if not trial['split'].eq('discovery').all():
        raise RuntimeError('A non-discovery row entered raw_eeg_epoch_folds.')
    if not trial['continuous_event_authoritative'].map(python_bool).all():
        raise RuntimeError('A non-authoritative event entered raw_eeg_epoch_folds.')
    if not trial['source_stimulus_run'].eq(trial['source_rating_run']).all():
        raise RuntimeError('Stimulus/rating run mismatch in raw_eeg_epoch_folds input.')
    if not trial['source_stimulus_path'].eq(trial['source_rating_path']).all():
        raise RuntimeError('Stimulus/rating EDF mismatch in raw_eeg_epoch_folds input.')
    duplicate_checks = [['source_stimulus_path', 'source_stimulus_annotation_index'], ['source_rating_path', 'source_rating_annotation_index'], ['source_stimulus_path', 'source_stimulus_annotation_index', 'source_rating_annotation_index']]
    for columns in duplicate_checks:
        if trial.duplicated(columns).any():
            raise RuntimeError('Authoritative source events are reused: ' + ', '.join(columns))
    trial = trial.sort_values(['subject_id', 'stimulus_onset_global_sec', 'raw_behavior_row_id']).reset_index(drop=True)
    trial['subject_acquisition_order_1based'] = trial.groupby('subject_id').cumcount() + 1
    trial['within_subject_block_fold'] = trial.groupby('subject_id', group_keys=False)['subject_acquisition_order_1based'].transform(lambda series: np.floor((series.to_numpy(dtype=float) - 1.0) * WITHIN_PERSON_FOLDS / float(len(series))).astype(int) + 1)
    return trial

def deterministic_participant_folds(trial: pd.DataFrame) -> pd.DataFrame:
    counts = trial.groupby('subject_id', sort=True).agg(eligible_rows=('raw_behavior_row_id', 'size'), imagery_rows=('context', lambda values: int(pd.Series(values).eq('imagery').sum())), video_rows=('context', lambda values: int(pd.Series(values).eq('video').sum()))).reset_index()
    counts = counts.sort_values(['eligible_rows', 'imagery_rows', 'video_rows', 'subject_id'], ascending=[False, False, False, True]).reset_index(drop=True)
    fold_state = {fold: {'eligible_rows': 0, 'imagery_rows': 0, 'video_rows': 0, 'subjects': 0} for fold in range(1, CROSS_PERSON_FOLDS + 1)}
    assignments = []
    for _, row in counts.iterrows():
        fold = min(fold_state, key=lambda value: (fold_state[value]['eligible_rows'], abs(fold_state[value]['imagery_rows'] - fold_state[value]['video_rows']), fold_state[value]['subjects'], value))
        assignments.append({**row.to_dict(), 'cross_person_fold': int(fold)})
        for key in ['eligible_rows', 'imagery_rows', 'video_rows']:
            fold_state[fold][key] += int(row[key])
        fold_state[fold]['subjects'] += 1
    output = pd.DataFrame(assignments).sort_values('subject_id').reset_index(drop=True)
    if output['subject_id'].duplicated().any():
        raise RuntimeError('Participant fold assignment is not unique.')
    if set(output['cross_person_fold']) != set(range(1, CROSS_PERSON_FOLDS + 1)):
        raise RuntimeError('Not all participant folds are populated.')
    return output

def event_index(global_events: pd.DataFrame, source_paths: set[str]) -> pd.DataFrame:
    required = {'subject_id', 'run', 'source_path', 'annotation_index', 'onset_sec', 'event_class'}
    missing = sorted(required.difference(global_events.columns))
    if missing:
        raise RuntimeError('Global event table lacks columns: ' + ', '.join(missing))
    events = global_events[list(required)].copy()
    events['subject_id'] = events['subject_id'].map(normalize_subject)
    events['event_class'] = events['event_class'].map(normalize_event_class)
    events = events.loc[events['source_path'].astype(str).isin(source_paths)].copy()
    return events

def derive_trial_boundaries(trial: pd.DataFrame, events: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    header_lookup = headers.set_index('raw_source_path').to_dict(orient='index')
    rows = []
    for source_path, source_group in trial.groupby('source_stimulus_path', sort=True):
        source_group = source_group.sort_values('source_stimulus_annotation_index').reset_index(drop=True)
        source_events = events.loc[events['source_path'].eq(source_path)].sort_values('annotation_index')
        if source_path not in header_lookup:
            raise RuntimeError(f'No EDF header for {source_path}.')
        recording_end = float(header_lookup[source_path]['recording_duration_sec'])
        for position, row in source_group.iterrows():
            stimulus_index = int(row['source_stimulus_annotation_index'])
            rating_index = int(row['source_rating_annotation_index'])
            stimulus_time = float(row['source_stimulus_local_onset_sec'])
            rating_time = float(row['source_rating_local_onset_sec'])
            next_stimulus_time = float(source_group.iloc[position + 1]['source_stimulus_local_onset_sec']) if position + 1 < len(source_group) else math.nan
            rating_boundary = next_stimulus_time - 5.0 if math.isfinite(next_stimulus_time) else recording_end - 5.0
            fades = source_events.loc[source_events['event_class'].eq('fade') & source_events['annotation_index'].astype(int).gt(stimulus_index) & source_events['annotation_index'].astype(int).lt(rating_index)].sort_values('annotation_index')
            context = row['context']
            if context == 'video':
                late_endpoint = rating_time
                endpoint_rule = 'video_rating_onset'
                endpoint_available = True
            elif len(fades) == 1:
                late_endpoint = float(fades.iloc[0]['onset_sec'])
                endpoint_rule = 'imagery_unique_fade'
                endpoint_available = True
            elif len(fades) == 0:
                late_endpoint = rating_time
                endpoint_rule = 'imagery_rating_fallback_no_fade'
                endpoint_available = True
            else:
                late_endpoint = math.nan
                endpoint_rule = 'imagery_multiple_fades_unresolved'
                endpoint_available = False
            rows.append({**row.to_dict(), 'recording_end_sec': recording_end, 'next_authoritative_stimulus_local_sec': next_stimulus_time, 'rating_epoch_latest_stop_sec': rating_boundary, 'fade_count_between_stimulus_and_rating': int(len(fades)), 'late_stimulus_endpoint_rule': endpoint_rule, 'late_stimulus_endpoint_sec': late_endpoint, 'late_stimulus_endpoint_available': endpoint_available})
    output = pd.DataFrame(rows).sort_values(['subject_id', 'subject_acquisition_order_1based']).reset_index(drop=True)
    if len(output) != len(trial):
        raise RuntimeError('Trial-boundary derivation changed row count.')
    return output

def window_status(start_sec: float, stop_sec: float, *, recording_end_sec: float, earliest_allowed_sec: float=0.0, latest_allowed_sec: float | None=None) -> tuple[bool, str]:
    if not (math.isfinite(start_sec) and math.isfinite(stop_sec)):
        return (False, 'endpoint_unresolved')
    if stop_sec <= start_sec:
        return (False, 'nonpositive_duration')
    if start_sec < earliest_allowed_sec - TIME_TOLERANCE_SEC:
        return (False, 'starts_before_allowed_boundary')
    if stop_sec > recording_end_sec + TIME_TOLERANCE_SEC:
        return (False, 'ends_after_recording')
    if latest_allowed_sec is not None and stop_sec > latest_allowed_sec + TIME_TOLERANCE_SEC:
        return (False, 'crosses_next_authoritative_boundary')
    return (True, 'available')

def build_epoch_manifest(trial: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    header_lookup = headers.set_index('raw_source_path').to_dict(orient='index')
    rows = []
    for _, row in trial.iterrows():
        source_path = row['source_stimulus_path']
        header = header_lookup[source_path]
        sfreq = float(header['sampling_rate_hz'])
        recording_end = float(row['recording_end_sec'])
        stimulus = float(row['source_stimulus_local_onset_sec'])
        rating = float(row['source_rating_local_onset_sec'])
        late_endpoint = float(row['late_stimulus_endpoint_sec']) if pd.notna(row['late_stimulus_endpoint_sec']) else math.nan
        windows = {'stimulus_early': {'start_sec': stimulus + 2.0, 'stop_sec': stimulus + 12.0, 'latest_allowed_sec': rating - 5.0, 'endpoint_rule': 'fixed_stimulus_onset_plus_2_to_12'}, 'stimulus_late': {'start_sec': late_endpoint - 25.0, 'stop_sec': late_endpoint - 5.0, 'latest_allowed_sec': rating - 5.0, 'endpoint_rule': row['late_stimulus_endpoint_rule']}, 'rating_early': {'start_sec': rating + 5.0, 'stop_sec': rating + 25.0, 'latest_allowed_sec': float(row['rating_epoch_latest_stop_sec']), 'endpoint_rule': 'fixed_rating_onset_plus_5_to_25'}}
        for epoch_family, window in windows.items():
            available, reason = window_status(float(window['start_sec']), float(window['stop_sec']), recording_end_sec=recording_end, earliest_allowed_sec=0.0, latest_allowed_sec=float(window['latest_allowed_sec']))
            start_sample = quantize_nearest(window['start_sec'], sfreq) if math.isfinite(float(window['start_sec'])) else pd.NA
            stop_sample = quantize_nearest(window['stop_sec'], sfreq) if math.isfinite(float(window['stop_sec'])) else pd.NA
            source_samples = int(stop_sample - start_sample) if start_sample is not pd.NA and stop_sample is not pd.NA else pd.NA
            expected_output_samples = int(EPOCH_SPECS[epoch_family]['duration_sec'] * EPOCH_SPECS[epoch_family]['output_sampling_rate_hz'])
            rows.append({'epoch_id': f"{row['raw_behavior_row_id']}|{epoch_family}", 'raw_behavior_row_id': row['raw_behavior_row_id'], 'subject_id': row['subject_id'], 'trial_number': int(row['trial_number']), 'context': row['context'], 'raw_run': int(row['source_stimulus_run']), 'raw_source_path': source_path, 'subject_acquisition_order_1based': int(row['subject_acquisition_order_1based']), 'within_subject_block_fold': int(row['within_subject_block_fold']), 'epoch_family': epoch_family, 'epoch_start_sec': float(window['start_sec']), 'epoch_stop_sec': float(window['stop_sec']), 'epoch_duration_sec': float(EPOCH_SPECS[epoch_family]['duration_sec']), 'epoch_endpoint_rule': window['endpoint_rule'], 'epoch_available': available, 'unavailability_reason': reason, 'original_sampling_rate_hz': sfreq, 'start_sample_nearest': start_sample, 'stop_sample_nearest_exclusive': stop_sample, 'source_sample_count': source_samples, 'output_sampling_rate_hz': int(EPOCH_SPECS[epoch_family]['output_sampling_rate_hz']), 'expected_output_samples': expected_output_samples, 'behavioral_values_used': False, 'eeg_amplitude_read': False})
    output = pd.DataFrame(rows)
    if len(output) != len(trial) * len(EPOCH_SPECS):
        raise RuntimeError('Epoch manifest row count changed.')
    if output['epoch_id'].duplicated().any():
        raise RuntimeError('Epoch IDs are not unique.')
    return output

def coverage_tables(epoch_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for epoch_family, group in epoch_manifest.groupby('epoch_family', sort=True):
        available = group['epoch_available'].map(python_bool)
        overall_coverage = float(available.mean())
        context_coverage = group.assign(available_bool=available).groupby('context')['available_bool'].mean().to_dict()
        spec = EPOCH_SPECS[epoch_family]
        overall_pass = bool(overall_coverage >= spec['overall_gate'])
        context_pass = bool(set(context_coverage) == {'imagery', 'video'} and all((value >= spec['context_gate'] for value in context_coverage.values())))
        summary_rows.append({'epoch_family': epoch_family, 'total_rows': int(len(group)), 'available_rows': int(available.sum()), 'overall_coverage': overall_coverage, 'overall_gate': float(spec['overall_gate']), 'overall_gate_passed': overall_pass, 'imagery_coverage': float(context_coverage.get('imagery', math.nan)), 'video_coverage': float(context_coverage.get('video', math.nan)), 'context_gate': float(spec['context_gate']), 'context_gate_passed': context_pass, 'branch_permitted_for_raw_eeg_preprocessing_qc': bool(overall_pass and context_pass), 'alternate_window_selected': False})
    subject_rows = []
    for (epoch_family, subject_id), group in epoch_manifest.groupby(['epoch_family', 'subject_id'], sort=True):
        available = group.loc[group['epoch_available'].map(python_bool)]
        imagery = int(available['context'].eq('imagery').sum())
        video = int(available['context'].eq('video').sum())
        total = int(len(available))
        subject_rows.append({'epoch_family': epoch_family, 'subject_id': subject_id, 'available_rows': total, 'imagery_rows': imagery, 'video_rows': video, 'minimum_total_required': MIN_SUBJECT_TOTAL, 'minimum_per_context_required': MIN_SUBJECT_PER_CONTEXT, 'within_person_modeling_eligible': bool(total >= MIN_SUBJECT_TOTAL and imagery >= MIN_SUBJECT_PER_CONTEXT and (video >= MIN_SUBJECT_PER_CONTEXT))})
    reason_table = epoch_manifest.loc[~epoch_manifest['epoch_available'].map(python_bool)].groupby(['epoch_family', 'unavailability_reason']).size().rename('rows').reset_index()
    return (pd.DataFrame(summary_rows), pd.DataFrame(subject_rows), reason_table)

def fold_qc(trial: pd.DataFrame, participant_folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_fold = trial[['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'source_stimulus_run', 'source_stimulus_path', 'stimulus_onset_global_sec', 'subject_acquisition_order_1based', 'within_subject_block_fold']].merge(participant_folds[['subject_id', 'cross_person_fold']], on='subject_id', how='left', validate='many_to_one')
    trial_fold = trial_fold.rename(columns={'source_stimulus_run': 'raw_run', 'source_stimulus_path': 'raw_source_path'})
    if trial_fold['cross_person_fold'].isna().any():
        raise RuntimeError('At least one trial lacks a participant fold.')
    subject_fold_counts = trial_fold.groupby(['subject_id', 'within_subject_block_fold']).agg(rows=('raw_behavior_row_id', 'size'), first_order=('subject_acquisition_order_1based', 'min'), last_order=('subject_acquisition_order_1based', 'max'), imagery_rows=('context', lambda values: int(pd.Series(values).eq('imagery').sum())), video_rows=('context', lambda values: int(pd.Series(values).eq('video').sum()))).reset_index()
    for subject_id, group in subject_fold_counts.groupby('subject_id'):
        if set(group['within_subject_block_fold']) != set(range(1, WITHIN_PERSON_FOLDS + 1)):
            raise RuntimeError(f'{subject_id}: not all within-person folds are populated.')
        ordered = group.sort_values('within_subject_block_fold')
        if not (ordered['first_order'].to_numpy()[1:] > ordered['last_order'].to_numpy()[:-1]).all():
            raise RuntimeError(f'{subject_id}: within-person folds are not contiguous blocks.')
    participant_qc = participant_folds.groupby('cross_person_fold').agg(subjects=('subject_id', 'size'), eligible_rows=('eligible_rows', 'sum'), imagery_rows=('imagery_rows', 'sum'), video_rows=('video_rows', 'sum')).reset_index()
    return (trial_fold, pd.concat([subject_fold_counts.assign(qc_level='within_subject'), participant_qc.assign(qc_level='cross_person')], ignore_index=True, sort=False))

def write_report(path: Path, summary: dict[str, Any], coverage: pd.DataFrame, subject_coverage: pd.DataFrame, header_inventory: pd.DataFrame, participant_folds: pd.DataFrame) -> None:
    lines = ['# Phase raw_eeg_epoch_folds — Raw-EEG epoch and fold specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- EEG sample amplitude read: **no**', '- Behavior/outcomes used for windows or folds: **no**', '- Validation/confirmation opened: **no**', '', '## prespecified epoch coverage', '', '| Epoch family | Available | Total | Coverage | Overall gate | Imagery | Video | Context gate | raw_eeg_preprocessing_qc branch |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for _, row in coverage.iterrows():
        lines.append(f"| `{row['epoch_family']}` | {int(row['available_rows'])} | {int(row['total_rows'])} | {row['overall_coverage']:.4f} | {row['overall_gate']:.3f} | {row['imagery_coverage']:.4f} | {row['video_coverage']:.4f} | {row['context_gate']:.3f} | {python_bool(row['branch_permitted_for_raw_eeg_preprocessing_qc'])} |")
    eligible_counts = subject_coverage.groupby('epoch_family')['within_person_modeling_eligible'].apply(lambda values: int(values.map(python_bool).sum())).to_dict()
    lines.extend(['', '## Header inventory', '', f'- Raw EDF files: **{len(header_inventory)}**', '- Sampling rates: ' + ', '.join((f'`{value:g} Hz`' for value in sorted(header_inventory['sampling_rate_hz'].unique()))), '- Channel counts: ' + ', '.join((f'`{int(value)}`' for value in sorted(header_inventory['channel_count'].unique()))), f"- Unique channel-order hashes: **{header_inventory['channel_order_sha256'].nunique()}**", '', '## Outcome-free folds', '', f'- Within-person contiguous folds: **{WITHIN_PERSON_FOLDS}**', f'- Cross-person folds: **{CROSS_PERSON_FOLDS}**', f'- Participants assigned: **{len(participant_folds)}**', ''])
    for epoch_family in sorted(eligible_counts):
        lines.append(f'- `{epoch_family}` within-person eligible subjects: **{eligible_counts[epoch_family]}**.')
    decision = summary['decision']
    lines.extend(['', '## Decision', '', f"- Header and interval specification passed: **{decision['header_and_interval_specification_passed']}**", f"- At least one raw_eeg_preprocessing_qc branch permitted: **{decision['at_least_one_raw_eeg_preprocessing_qc_branch_permitted']}**", f"- Stimulus branch permitted: **{decision['stimulus_branch_permitted']}**", f"- Rating branch permitted: **{decision['rating_branch_permitted']}**", f"- Alternate window selected after coverage: **{decision['alternate_window_selected_after_coverage']}**", f"- EEG amplitude read: **{decision['eeg_amplitude_read']}**", ''])
    path.write_text(''.join(lines) + '', encoding='utf-8')

def main() -> int:
    args = parse_args()
    started = time.time()
    dataset = args.dataset.expanduser().resolve()
    project = args.project.expanduser().resolve()
    paths = {'derivative_closure_summary': project / derivative_closure_SUMMARY_REL, 'derivative_closure_contract': project / derivative_closure_CONTRACT_REL, 'master': project / MASTER_REL, 'attachment': project / ATTACHMENT_REL, 'anchors': project / ANCHOR_REL, 'global_events': project / GLOBAL_EVENTS_REL}
    require_files(paths.values())
    derivative_closure_summary = read_json(paths['derivative_closure_summary'])
    derivative_closure_contract = read_json(paths['derivative_closure_contract'])
    if derivative_closure_summary.get('phase') != 'derivative_closure':
        raise RuntimeError('Unexpected derivative_closure prerequisite.')
    if not bool(derivative_closure_summary['decision']['corrected_layered_discovery_master_prespecified']):
        raise RuntimeError('derivative_closure did not specification the corrected discovery master.')
    if not bool(derivative_closure_summary['decision']['new_raw_edf_preprocessing_permitted']):
        raise RuntimeError('derivative_closure did not permit raw EDF preprocessing.')
    if not bool(derivative_closure_summary['decision']['released_filtered_trial_attachment_closed']):
        raise RuntimeError('Released derivative attachment is not formally closed.')
    master = pd.read_csv(paths['master'], low_memory=False)
    attachment = pd.read_csv(paths['attachment'], low_memory=False)
    anchors = pd.read_csv(paths['anchors'], low_memory=False)
    global_events = pd.read_csv(paths['global_events'], low_memory=False)
    if len(attachment) != len(master):
        raise RuntimeError('derivative_closure master/attachment row counts differ.')
    if len(anchors) != 13:
        raise RuntimeError('Expected 13 pipeline-QC anchors.')
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config_dir = project / 'config'
    logs = project / 'outputs/logs'
    outputs = {'trial_manifest': tables / 'raw_eeg_epoch_folds_authoritative_trial_timing_manifest.csv', 'epoch_manifest': tables / 'raw_eeg_epoch_folds_prespecified_epoch_manifest.csv', 'raw_headers': qc / 'raw_eeg_epoch_folds_raw_edf_header_inventory.csv', 'coverage': qc / 'raw_eeg_epoch_folds_epoch_coverage_summary.csv', 'subject_coverage': qc / 'raw_eeg_epoch_folds_subject_epoch_coverage.csv', 'unavailability': qc / 'raw_eeg_epoch_folds_epoch_unavailability_reasons.csv', 'trial_folds': tables / 'raw_eeg_epoch_folds_trial_fold_manifest.csv', 'participant_folds': tables / 'raw_eeg_epoch_folds_participant_fold_manifest.csv', 'fold_qc': qc / 'raw_eeg_epoch_folds_fold_balance_qc.csv', 'summary': qc / 'raw_eeg_epoch_folds_raw_eeg_epoch_folds_summary.json', 'contract': config_dir / 'raw_eeg_epoch_folds_raw_eeg_epoch_folds_contract.json', 'raw_eeg_preprocessing_qc_policy': config_dir / 'S057_raw_eeg_epoch_folds.json', 'report': reports / 'raw_eeg_epoch_folds_raw_eeg_epoch_folds_report.md', 'log': logs / 'raw_eeg_epoch_folds_raw_eeg_epoch_folds.log'}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and (not args.force):
        raise RuntimeError('Outputs already exist. Use --force only after review:\n  - ' + '\n  - '.join((str(path) for path in existing)))
    with outputs['log'].open('w', encoding='utf-8') as log:

        def note(message: str) -> None:
            line = f'[{now_iso()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Selecting only derivative_closure-authoritative discovery raw-EEG trials.')
        trial = authoritative_trials(master)
        note('Reading raw EDF headers only; sample amplitude remains unopened.')
        headers = read_raw_headers(trial['source_stimulus_path'].astype(str).tolist())
        headers.to_csv(outputs['raw_headers'], index=False)
        if not headers['header_readable'].map(python_bool).all():
            raise RuntimeError('At least one raw EDF header is unreadable.')
        if not headers['channel_count'].eq(64).all():
            raise RuntimeError('At least one raw_eeg_epoch_folds raw EDF does not contain 64 channels.')
        if not np.allclose(headers['sampling_rate_hz'], np.rint(headers['sampling_rate_hz']), rtol=0.0, atol=1e-09):
            raise RuntimeError('At least one sampling rate is not integer-valued.')
        events = event_index(global_events, set(trial['source_stimulus_path'].astype(str)))
        note('Deriving unique-fade, no-fade fallback, and next-authoritative-stimulus boundaries.')
        trial = derive_trial_boundaries(trial, events, headers)
        trial.to_csv(outputs['trial_manifest'], index=False)
        note('Freezing three fixed epoch families and nearest source-sample bounds.')
        epoch_manifest = build_epoch_manifest(trial, headers)
        epoch_manifest.to_csv(outputs['epoch_manifest'], index=False)
        coverage, subject_coverage, unavailability = coverage_tables(epoch_manifest)
        coverage.to_csv(outputs['coverage'], index=False)
        subject_coverage.to_csv(outputs['subject_coverage'], index=False)
        unavailability.to_csv(outputs['unavailability'], index=False)
        note('Freezing outcome-free contiguous within-person and participant-level folds.')
        participant_folds = deterministic_participant_folds(trial)
        trial_folds, folds_qc = fold_qc(trial, participant_folds)
        trial_folds.to_csv(outputs['trial_folds'], index=False)
        participant_folds.to_csv(outputs['participant_folds'], index=False)
        folds_qc.to_csv(outputs['fold_qc'], index=False)
        branch_lookup = coverage.set_index('epoch_family')['branch_permitted_for_raw_eeg_preprocessing_qc'].map(python_bool).to_dict()
        stimulus_branch = bool(branch_lookup.get('stimulus_early', False) and branch_lookup.get('stimulus_late', False))
        rating_branch = bool(branch_lookup.get('rating_early', False))
        decision = {'header_and_interval_specification_passed': True, 'stimulus_early_branch_permitted': bool(branch_lookup.get('stimulus_early', False)), 'stimulus_late_branch_permitted': bool(branch_lookup.get('stimulus_late', False)), 'rating_early_branch_permitted': rating_branch, 'stimulus_branch_permitted': stimulus_branch, 'rating_branch_permitted': rating_branch, 'at_least_one_raw_eeg_preprocessing_qc_branch_permitted': bool(stimulus_branch or rating_branch), 'alternate_window_selected_after_coverage': False, 'outcomes_used_for_folds': False, 'eeg_amplitude_read': False, 'released_derivative_amplitude_read': False, 'validation_opened': False, 'confirmation_opened': False}
        summary = {'generated_at': now_iso(), 'phase': 'raw_eeg_epoch_folds', 'status': 'raw_eeg_epoch_coverage_and_outcome_free_fold_specification_complete_candidate', 'counts': {'authoritative_trials': int(len(trial)), 'subjects': int(trial['subject_id'].nunique()), 'raw_edf_files': int(len(headers)), 'epoch_manifest_rows': int(len(epoch_manifest)), 'available_epoch_rows': int(epoch_manifest['epoch_available'].map(python_bool).sum()), 'participant_fold_rows': int(len(participant_folds)), 'trial_fold_rows': int(len(trial_folds))}, 'epoch_coverage': coverage.to_dict(orient='records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        contract = {**summary, 'epoch_specs': EPOCH_SPECS, 'within_person_folds': WITHIN_PERSON_FOLDS, 'cross_person_folds': CROSS_PERSON_FOLDS, 'within_person_fold_policy': 'five contiguous blocks of corrected global acquisition order', 'cross_person_fold_policy': 'deterministic greedy balance using eligible-row and context counts only', 'subject_modeling_minimums': {'total': MIN_SUBJECT_TOTAL, 'per_context': MIN_SUBJECT_PER_CONTEXT}, 'source_sample_quantization': 'nearest integer sample at original EDF sampling rate', 'alternate_window_policy': 'none; failed branches are bfixed', 'behavioral_values_read_for_design': False, 'eeg_amplitude_read': False, 'validation_opened': False, 'confirmation_opened': False, 'source_dataset_modified': False, 'source_derivatives_modified': False, 'canonical_master_modified': False, 'input_hashes': {name: sha256_file(path) for name, path in paths.items()}, 'output_hashes': {name: sha256_file(path) for name, path in outputs.items() if name not in {'contract', 'raw_eeg_preprocessing_qc_policy', 'report', 'log'} and path.is_file()}}
        outputs['contract'].write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=safe_json_value) + '', encoding='utf-8')
        raw_eeg_preprocessing_qc_policy = {'generated_at': now_iso(), 'phase': 'raw_eeg_epoch_folds', 'next_phase': 'raw_eeg_preprocessing_qc', 'next_phase_permitted': bool(decision['at_least_one_raw_eeg_preprocessing_qc_branch_permitted']), 'permitted_epoch_branches': {row['epoch_family']: python_bool(row['branch_permitted_for_raw_eeg_preprocessing_qc']) for _, row in coverage.iterrows()}, 'fixed_signal_policy': {'continuous_filter': 'fourth-order Butterworth SOS, 0.5-45 Hz, zero-phase', 'resampling': 'polyphase resampling to 200 Hz after source-window extraction', 'primary_reference': 'robust average reference over channels passing prespecified recording/epoch QC; no ICA component removal in primary', 'no_automatic_ica_removal': True, 'epoch_windows': EPOCH_SPECS, 'fold_manifest': str(outputs['trial_folds']), 'participant_fold_manifest': str(outputs['participant_folds'])}, 'forbidden': ['changing an epoch window after amplitude inspection', 'using released filtered trial attachment', 'using validation or confirmation participants', 'using outcomes to define QC or folds', 'automatic ICA component rejection without an independently prespecified classifier']}
        outputs['raw_eeg_preprocessing_qc_policy'].write_text(json.dumps(raw_eeg_preprocessing_qc_policy, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(outputs['report'], summary, coverage, subject_coverage, headers, participant_folds)
        note('raw_eeg_epoch_folds completed. No EEG amplitude, outcome, validation, or confirmation data was read.')
    print()
    print('=' * 100)
    print('PHASE raw_eeg_epoch_folds RAW-EEG EPOCH/FOLD specification COMPLETED')
    print('=' * 100)
    print(f'Authoritative trials:           {len(trial)}')
    print(f'Raw EDF files:                  {len(headers)}')
    print(f'Epoch manifest rows:            {len(epoch_manifest)}')
    print(f"Available epoch rows:           {epoch_manifest['epoch_available'].map(python_bool).sum()}")
    print(f'Stimulus branch permitted:      {stimulus_branch}')
    print(f'Rating branch permitted:        {rating_branch}')
    print('EEG sample amplitude read:      False')
    print(f"Report:                         {outputs['report']}")
    print(f'Runtime:                        {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    mne.set_log_level('ERROR')
    raise SystemExit(main())