# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, glob, hashlib, json, shutil
from pathlib import Path
from datetime import datetime
PHASE = 'raw_dynamic_physio_alignment_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
REQ = {'integrated_synthesis_lock': '<INTEGRATED_SYNTHESIS_CHECKPOINT>', 'three_cohort_structural_synthesis_lock': 'outputs/qc/three_cohort_structural_synthesis_final_output_lock_hashes.json', 'discovery_multimodal_synthesis_lock': 'outputs/qc/discovery_multimodal_synthesis_final_output_lock_hashes.json', 'p02_contract': 'outputs/qc/physiology_source_contract.v1.json', 'raw_inventory': 'outputs/tables/raw_physio_inventory.csv', 'raw_eeg_epoch_folds_timing': 'outputs/tables/raw_eeg_epoch_folds_authoritative_trial_timing_manifest.csv', 'raw_eeg_epoch_folds_epochs': 'outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv', 'split': 'config/participant_split_manifest.v1.csv'}
EPOCH_RULES = [{'epoch_family': 'stimulus_early', 'definition': 'stimulus + 2 s to stimulus + 12 s', 'duration_sec': 10.0, 'source': 'fixed raw_eeg_epoch_folds epoch manifest'}, {'epoch_family': 'stimulus_late', 'definition': 'source-supported endpoint - 25 s to endpoint - 5 s', 'duration_sec': 20.0, 'source': 'fixed raw_eeg_epoch_folds epoch manifest'}, {'epoch_family': 'rating_early', 'definition': 'rating + 5 s to rating + 25 s', 'duration_sec': 20.0, 'source': 'fixed raw_eeg_epoch_folds epoch manifest'}]

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def boolish(x):
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y'}

def rows(p):
    with open(p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def writecsv(p, rs, fields=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rs:
        p.write_text('', encoding='utf-8')
        return
    if fields is None:
        fields = list(rs[0].keys())
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rs)

def require(ps):
    miss = [p for p in ps if not p.is_file()]
    if miss:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(map(str, miss)))

def verify_boundaries(project: Path):
    c5 = readj(project / REQ['integrated_synthesis_lock'])
    if c5.get('status') != 'full_publication_draft_assembled':
        raise RuntimeError('integrated evidence assembly not in expected status')
    b = c5.get('boundaries', {})
    if b.get('new_inference_performed') is not False or b.get('raw_dynamic_physio_tested') is not False:
        raise RuntimeError('integrated_synthesis boundary mismatch')
    if b.get('heldout_physio_opened') is not False:
        raise RuntimeError('Held-out physiology was unexpectedly opened')
    if not c5.get('next', {}).get('raw_dynamic_physio_requires_new_design_specification'):
        raise RuntimeError('integrated_synthesis does not authorize a new raw-dynamic analysis specification')
    discovery_multimodal_synthesis = readj(project / REQ['discovery_multimodal_synthesis_lock'])
    if discovery_multimodal_synthesis.get('multimodal_analysis_concluded') is not True:
        raise RuntimeError('discovery_multimodal_synthesis multimodal branch is not closed as expected')
    if discovery_multimodal_synthesis.get('Validation_physio_values_opened') is not False:
        raise RuntimeError('Validation physiology was unexpectedly opened in discovery_multimodal_synthesis')
    if discovery_multimodal_synthesis.get('Confirmation_physio_values_opened') is not False:
        raise RuntimeError('Confirmation physiology was unexpectedly opened in discovery_multimodal_synthesis')
    p02 = readj(project / REQ['p02_contract'])
    if p02.get('status') != 'prespecified':
        raise RuntimeError('P0.2 physiology source contract is not prespecified')
    rc = p02.get('raw_continuous_recordings', {})
    if int(rc.get('participants', 0)) != 47 or int(rc.get('gsr_files', 0)) != 47 or int(rc.get('ppg_files', 0)) != 47:
        raise RuntimeError('Unexpected P0.2 raw physiology inventory contract')
    if p02.get('analysis_boundary', {}).get('raw_dynamic_timeseries_model_allowed_now') is not False:
        raise RuntimeError('Historical P0.2 boundary unexpectedly allowed raw dynamic modelling')
    return (c5, p02)

def build_source_roster(project: Path, dataset: Path):
    inv = rows(project / REQ['raw_inventory'])
    d = [r for r in inv if r.get('split', '').strip().lower() == 'discovery']
    by = {}
    for r in d:
        s = r.get('subject_id', '').strip()
        m = r.get('modality', '').strip().upper()
        if not s or m not in {'GSR', 'PPG'}:
            continue
        by.setdefault(s, {})[m] = r
    out = []
    for s in sorted(by):
        g = by[s].get('GSR', {})
        p = by[s].get('PPG', {})
        out.append({'subject_id': s, 'split': 'discovery', 'gsr_tsv_path': g.get('tsv_path', ''), 'gsr_json_path': g.get('json_path', ''), 'gsr_exists': str(boolish(g.get('tsv_exists', 'False')) and boolish(g.get('json_exists', 'False'))), 'gsr_sampling_frequency_sidecar': g.get('sampling_frequency', ''), 'gsr_time_column_declared': 'csv_time_GSR', 'ppg_tsv_path': p.get('tsv_path', ''), 'ppg_json_path': p.get('json_path', ''), 'ppg_exists': str(boolish(p.get('tsv_exists', 'False')) and boolish(p.get('json_exists', 'False'))), 'ppg_sampling_frequency_sidecar': p.get('sampling_frequency', ''), 'ppg_time_column_declared': 'csv_time_PPG', 'both_raw_modalities_exist': str(boolish(g.get('tsv_exists', 'False')) and boolish(g.get('json_exists', 'False')) and boolish(p.get('tsv_exists', 'False')) and boolish(p.get('json_exists', 'False')))})
    return out

def timing_counts(project: Path):
    timing = rows(project / REQ['raw_eeg_epoch_folds_timing'])
    ep = rows(project / REQ['raw_eeg_epoch_folds_epochs'])
    timing = [r for r in timing if r.get('split', '').lower() == 'discovery' and boolish(r.get('continuous_raw_eeg_analysis_eligible', 'False'))]
    eps = [r for r in ep if boolish(r.get('epoch_available', 'False'))]
    return (timing, eps)

def policy_dict(source_hashes, counts):
    return {'phase': PHASE, 'generated_at': now(), 'role': 'raw time-resolved peripheral physiology alignment and analysis analysis specification', 'scientific_scope': {'primary_question': 'Can raw event-aligned GSR/PPG dynamics reveal reproducible autonomic temporal structure or brain-body coupling that was not captured by the prespecified 40-feature trial-level physiology representation?', 'this_phase_inference': False, 'this_phase_raw_signal_values_read': False, 'previous_trial_summary_result_immutable': 'M1-M3 from discovery_multimodal_inference remain unsupported and are not reinterpreted by opening a distinct raw-dynamic representation.'}, 'source_hashes': source_hashes, 'split_lockbox': {'Discovery': 'timing audit and, only after later specifications, development/inference permitted', 'Validation': 'raw physiology values and timing columns remain unopened until a prespecified Discovery dynamic support gate', 'Confirmation': 'raw physiology values and timing columns remain unopened until one-shot Validation transfer passes'}, 'raw_sources': {'GSR': 'sub-XX/eeg/sub-XX_task-emotion_recording-GSR_physio.tsv.gz + JSON sidecar', 'PPG': 'sub-XX/eeg/sub-XX_task-emotion_recording-PPG_physio.tsv.gz + JSON sidecar', 'historical_inventory': 'raw_physio_inventory.csv', 'expected_sidecar_sampling_hz': {'GSR': 40.0, 'PPG': 20.0}, 'amplitude_columns_forbidden_in_raw_dynamic_physio_timebase_audit': ['GSR', 'PPG'], 'timing_columns_allowed_in_raw_dynamic_physio_timebase_audit': ['csv_time_GSR', 'csv_time_PPG']}, 'epoch_contract': {'source': 'raw_eeg_epoch_folds prespecified EEG epoch manifest; windows are copied, never reselected from physiology', 'families': EPOCH_RULES, 'physiology_lag_window_reselection_forbidden': True, 'shorter_or_shifted_rescue_windows_forbidden': True}, 'raw_dynamic_physio_timebase_audit_timebase_audit': {'signal_amplitude_read': False, 'candidate_clock_interpretations_in_fixed_order': ['outcome_model_integrity: csv_time minus JSON StartTime, if finite/monotonic and scale matches sidecar sampling frequency', 'C2: csv_time minus first finite csv_time, if finite/monotonic and scale matches sidecar sampling frequency', 'C3: row-index divided by sidecar SamplingFrequency, timing-only fallback'], 'model_selection_uses_only': ['monotonicity', 'finite timing coverage', 'median local dt vs 1/SamplingFrequency', 'epoch-window sample-density coverage', 'GSR-PPG clock agreement'], 'model_selection_must_not_use': ['GSR amplitude', 'PPG amplitude', 'EEG amplitude', 'Family 1/2', 'Valence', 'Liking', 'any report', 'emotion category'], 'fixed_diagnostics': {'median_dt_relative_error_max': 0.02, 'within_epoch_expected_sample_fraction_min': 0.95, 'cross_modality_epoch_boundary_disagreement_sec_max': 0.25, 'participant_min_complete_three_epoch_trials': 16}, 'primary_feasibility_gate': {'minimum_participants_with_both_modalities_and_16_complete_trajectories': 18, 'minimum_fraction_of_candidate_raw_overlap_trials_with_all_three_epochs': 0.7, 'if_failed': 'stop the joint raw-dynamic branch; do not auto-switch to a favourable single modality or alternate windows'}}, 'future_sequence_if_raw_dynamic_physio_timebase_audit_passes': ['raw_dynamic_physio_preprocessing_specification: specification amplitude preprocessing, artifact/QC, resampling and dynamic representation before amplitude access', 'raw_dynamic_physio_synthesis: execute Discovery raw GSR/PPG preprocessing and build event-aligned autonomic trajectories', 'raw_dynamic_physio_sync_correction: prespecified outcome-free autonomic structure and EEG-autonomic coupling inference in Discovery', 'exact_physio_eeg_sync_recovery+: only if the pre-specified Discovery support+robustness gate passes, one-shot Validation then Confirmation transfer'], 'future_analysis_boundaries': {'EEG_Family_1_2_immutable': True, 'semantic_relabeling_forbidden': True, 'outcome_driven_feature_selection_forbidden': True, 'heldout_peeking_forbidden': True, 'dynamic_results_cannot_retroactively_change_discovery_multimodal_inference_trial_summary_result': True}, 'counts_at_specification': counts}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    dataset = a.dataset.expanduser().resolve()
    require([project / x for x in REQ.values()])
    c5, p02 = verify_boundaries(project)
    src_roster = build_source_roster(project, dataset)
    timing, eps = timing_counts(project)
    raw_both = [r for r in src_roster if boolish(r['both_raw_modalities_exist'])]
    if len(raw_both) != 29:
        raise RuntimeError(f'Expected 29 Discovery participants with both raw modalities from fixed audit; observed {len(raw_both)}')
    source_hashes = {k: sha256(project / v) for k, v in REQ.items()}
    counts = {'discovery_raw_both_participants_inventory': len(raw_both), 'discovery_authoritative_EEG_trials': len(timing), 'available_fixed_EEG_epochs': len(eps), 'epoch_families': 3, 'raw_modalities': 2, 'raw_signal_values_read': False, 'validation_raw_physio_opened': False, 'confirmation_raw_physio_opened': False}
    policy = policy_dict(source_hashes, counts)
    out_cfg = project / 'config'
    out_t = project / 'outputs/tables'
    out_q = project / 'outputs/qc'
    out_r = project / 'outputs/reports'
    out_l = project / 'outputs/logs'
    outputs = [out_cfg / 'S052_raw_dynamic_physio_alignment.json', out_cfg / 'S053_raw_dynamic_physio_alignment.json', out_t / 'raw_dynamic_physio_alignment_design_discovery_raw_physio_source_roster.csv', out_t / 'raw_dynamic_physio_alignment_design_epoch_window_contract.csv', out_q / 'raw_dynamic_physio_alignment_design_source_rehash.csv', out_q / 'raw_dynamic_physio_alignment_design_summary.json', out_r / 'raw_dynamic_physio_alignment_design_raw_dynamic_physio_design_report.md', out_l / 'raw_dynamic_physio_alignment_design_raw_dynamic_physio_design_specification.log', out_cfg / 'raw_dynamic_physio_alignment_design_raw_dynamic_physio_design_contract.json', out_q / 'raw_dynamic_physio_alignment_design_final_output_lock_hashes.json']
    if any((p.exists() for p in outputs)) and (not a.force):
        raise RuntimeError('raw_dynamic_physio_alignment_design outputs already exist; rerun with --force only if intentionally replacing this analysis specification.')
    outputs[0].write_text(json.dumps(policy, indent=2, ensure_ascii=False) + '', encoding='utf-8')
    nextp = {'phase': PHASE, 'generated_at': now(), 'next_phase': 'raw_dynamic_physio_timebase_audit', 'next_phase_permitted': True, 'role': 'Discovery-only timing-column/source alignment audit for raw GSR/PPG; no amplitude access and no inference', 'allowed_reads': ['fixed Discovery raw_eeg_epoch_folds timing/epoch manifests', 'Discovery raw GSR/PPG JSON sidecars', 'Discovery raw GSR/PPG csv_time_* columns only'], 'forbidden_reads': ['GSR amplitude', 'PPG amplitude', 'Validation raw physiology timing or amplitude', 'Confirmation raw physiology timing or amplitude', 'behavioral outcomes/reports for alignment selection'], 'required_outputs': ['per-file time-column diagnostics', 'per-participant selected clock interpretation or no-solution', 'epoch coverage audit', 'GSR-PPG clock agreement', 'prespecified feasible Discovery dynamic roster'], 'inference_forbidden': True}
    outputs[1].write_text(json.dumps(nextp, indent=2) + '', encoding='utf-8')
    writecsv(outputs[2], src_roster)
    writecsv(outputs[3], EPOCH_RULES)
    rehash = [{'source_key': k, 'relative_path': REQ[k], 'sha256': v} for k, v in source_hashes.items()]
    writecsv(outputs[4], rehash)
    summ = {'phase': PHASE, 'generated_at': now(), 'status': 'raw_dynamic_physio_alignment_design_prespecified', 'counts': counts, 'boundaries': {'raw_signal_values_read': False, 'new_inference_performed': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False}, 'raw_dynamic_physio_timebase_audit_permitted': True}
    outputs[5].write_text(json.dumps(summ, indent=2) + '', encoding='utf-8')
    report = f'# raw_dynamic_physio_alignment_design — raw dynamic physiology alignment analysis specification\n\nStatus: **prespecified**.\n\nThis phase opens no raw GSR/PPG amplitude. It specifications a timing-first attempt to align the raw peripheral recordings to the already fixed EEG windows.\n\n## fixed facts\n- Discovery participants with both raw GSR and PPG in the prior timing inventory: **{len(raw_both)}**.\n- fixed Discovery EEG authoritative trials: **{len(timing)}**.\n- Available fixed EEG epoch rows: **{len(eps)}**.\n- EEG windows are copied from raw_eeg_epoch_folds and may not be shifted after viewing physiology.\n- The prior 40-feature trial-level physiology result remains unchanged and unsupported for M1-M3.\n\n## Next gate\nraw_dynamic_physio_timebase_audit may read only `csv_time_GSR` / `csv_time_PPG` and sidecar timing metadata in Discovery. It may not read GSR/PPG amplitude. A joint dynamic branch proceeds only if at least 18 participants have both modalities and >=16 complete three-epoch trajectories, with >=70% complete trial coverage in the candidate raw overlap.\n\nValidation and Confirmation raw physiology remain closed.'
    outputs[6].write_text(report, encoding='utf-8')
    outputs[7].write_text(f'[{now()}] raw_dynamic_physio_alignment_design analysis specification complete. Raw amplitude read: False. raw_dynamic_physio_timebase_audit permitted: True.', encoding='utf-8')
    prehash = {str(p.relative_to(project)): sha256(p) for p in outputs[:8]}
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'raw_dynamic_physio_alignment_design_prespecified', 'source_hashes': source_hashes, 'output_hashes_prelock': prehash, 'boundaries': summ['boundaries'], 'raw_dynamic_physio_timebase_audit_permitted': True}
    outputs[8].write_text(json.dumps(contract, indent=2) + '', encoding='utf-8')
    hashes = {str(p.relative_to(project)): sha256(p) for p in outputs[:9]}
    lock = {'phase': PHASE, 'generated_at': now(), 'status': 'raw_dynamic_physio_alignment_design_prespecified', 'hashes': hashes, 'raw_dynamic_physio_timebase_audit_permitted': True, 'raw_signal_values_read': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'new_inference_performed': False}
    outputs[9].write_text(json.dumps(lock, indent=2) + '', encoding='utf-8')
    zp = project / '' / f'raw_dynamic_physio_alignment_design_raw_dynamic_physio_alignment_design_specification_{stamp()}'
    s = sha256(zp)
    Path(str(zp) + '.sha256').write_text(f'{s}  {zp.name}', encoding='utf-8')
    print('raw_dynamic_physio_alignment_design PASS')
    print('Discovery raw-both participants:', len(raw_both))
    print('fixed EEG trials:', len(timing))
    print('Raw signal amplitude read: False')
    print('raw_dynamic_physio_timebase_audit permitted: True')
    print('', zp)
    print('SHA-256:', s)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())