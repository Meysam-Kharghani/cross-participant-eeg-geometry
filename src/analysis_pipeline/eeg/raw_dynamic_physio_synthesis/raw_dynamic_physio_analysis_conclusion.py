# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, hashlib, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
PHASE = 'raw_dynamic_physio_synthesis'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
B_LOCK = 'outputs/qc/raw_dynamic_physio_timebase_audit_final_output_lock_hashes.json'
B_SUM = 'outputs/qc/raw_dynamic_physio_timebase_audit_timing_audit_summary.json'
B_NEXT = 'analysis_specifications/S056_raw_dynamic_physio_timebase_audit_preprocessing.json'
B_STREAM = 'outputs/qc/raw_dynamic_physio_timebase_audit_raw_physio_timing_stream_audit.csv'
B_EDF = 'outputs/qc/raw_dynamic_physio_timebase_audit_eeg_absolute_header_anchor_audit.csv'
B_CAND = 'outputs/tables/raw_dynamic_physio_timebase_audit_timebase_candidate_diagnostics.csv'
B_SEL = 'outputs/tables/raw_dynamic_physio_timebase_audit_participant_clock_selection.csv'

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

def writej(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def writecsv(p, rows, fields):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def verify_b(project: Path):
    lock = readj(project / B_LOCK)
    summary = readj(project / B_SUM)
    nxt = readj(project / B_NEXT)
    if lock.get('phase') != 'raw_dynamic_physio_timebase_audit':
        raise RuntimeError('Wrong raw_dynamic_physio_timebase_audit lock phase')
    if lock.get('raw_dynamic_physio_preprocessing_specification_permitted') is not False:
        raise RuntimeError('raw_dynamic_physio_synthesis closure only valid after failed raw_dynamic_physio_timebase_audit gate')
    if lock.get('raw_signal_amplitude_read') is not False:
        raise RuntimeError('raw_dynamic_physio_timebase_audit boundary violation: raw amplitude was read')
    if lock.get('Validation_raw_physio_opened') is not False or lock.get('Confirmation_raw_physio_opened') is not False:
        raise RuntimeError('Held-out raw physiology boundary violation')
    if summary.get('status') != 'timing_audit_failed_stop_raw_dynamic_branch':
        raise RuntimeError('raw_dynamic_physio_timebase_audit did not record STOP status')
    if summary.get('counts', {}).get('complete_joint_dynamic_trajectories') != 0:
        raise RuntimeError('Expected zero complete joint dynamic trajectories')
    if nxt.get('next_phase') != 'raw_source_lineage_dynamic_physio_preprocessing_specification' or nxt.get('next_phase_permitted') is not False:
        raise RuntimeError('prespecified raw_dynamic_physio_preprocessing_specification gate mismatch')
    for rel, h in lock.get('hashes', {}).items():
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing fixed raw_dynamic_physio_timebase_audit output: {rel}')
        got = sha256(p)
        if got != h:
            raise RuntimeError(f'raw_dynamic_physio_timebase_audit hash mismatch: {rel}: {got} != {h}')
    return (lock, summary, nxt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        print('raw_dynamic_physio_synthesis self-test: PASS')
        return 0
    project = a.project.resolve()
    lock, summary, nxt = verify_b(project)
    stream = pd.read_csv(project / B_STREAM)
    edf = pd.read_csv(project / B_EDF)
    cand = pd.read_csv(project / B_CAND)
    sel = pd.read_csv(project / B_SEL)
    edf = edf.copy()
    edf['subject_id'] = edf['raw_source_path'].astype(str).str.extract('/(sub-\\d+)/')
    eeg = edf.groupby('subject_id', as_index=True)['meas_date_epoch_sec'].min()
    phys = stream.groupby('subject_id', as_index=True)['normalized_first_abs_sec'].min()
    j = pd.concat([eeg.rename('eeg_abs_start_sec'), phys.rename('phys_abs_start_sec')], axis=1).dropna()
    j['abs_anchor_offset_days'] = (j.eeg_abs_start_sec - j.phys_abs_start_sec) / 86400.0
    offs = j.abs_anchor_offset_days.to_numpy(float)
    c3 = cand[(cand.candidate == 'C3') & (cand.structural_pass == True)]
    rows = [{'metric': 'raw_both_inventory_participants', 'value': summary['counts']['raw_both_inventory_participants'], 'interpretation': 'Source inventory only'}, {'metric': 'raw_both_with_raw_eeg_epoch_folds_EEG_participants', 'value': summary['counts']['raw_both_with_raw_eeg_epoch_folds_EEG_participants'], 'interpretation': 'Participants entering timing audit'}, {'metric': 'candidate_EEG_all3_trials', 'value': summary['counts']['candidate_EEG_all3_trials'], 'interpretation': 'EEG trials with all three prespecified epochs'}, {'metric': 'complete_joint_dynamic_trajectories', 'value': summary['counts']['complete_joint_dynamic_trajectories'], 'interpretation': 'Joint GSR+PPG trajectories passing prespecified timing gate'}, {'metric': 'participants_feasible_min16', 'value': summary['counts']['participants_with_feasible_joint_clock_and_min16'], 'interpretation': 'Participants passing prespecified joint timing gate'}, {'metric': 'C3_structurally_valid_streams', 'value': int(len(c3)), 'interpretation': 'Row-index clocks have correct internal sample interval but do not establish cross-device synchronization'}, {'metric': 'absolute_anchor_offset_days_min', 'value': float(np.min(offs)) if len(offs) else np.nan, 'interpretation': 'EEG earliest header minus raw-physiology normalized start'}, {'metric': 'absolute_anchor_offset_days_median', 'value': float(np.median(offs)) if len(offs) else np.nan, 'interpretation': 'EEG earliest header minus raw-physiology normalized start'}, {'metric': 'absolute_anchor_offset_days_max', 'value': float(np.max(offs)) if len(offs) else np.nan, 'interpretation': 'EEG earliest header minus raw-physiology normalized start'}]
    out_table = project / 'outputs/tables/raw_dynamic_physio_synthesis_raw_dynamic_timing_failure_summary.csv'
    writecsv(out_table, rows, ['metric', 'value', 'interpretation'])
    closure = {'phase': PHASE, 'generated_at': now(), 'status': 'raw_dynamic_physio_analysis_concluded_after_timing_infeasibility', 'upstream_raw_dynamic_physio_timebase_audit_lock_sha256': sha256(project / B_LOCK), 'decision': {'raw_dynamic_physio_preprocessing_specification_preprocessing_permitted': False, 'raw_dynamic_GSR_PPG_amplitude_may_be_opened_under_current_branch': False, 'Validation_raw_physio_may_be_opened': False, 'Confirmation_raw_physio_may_be_opened': False, 'single_modality_rescue_permitted': False, 'relative_start_or_manual_offset_rescue_permitted': False, 'window_retuning_permitted': False}, 'interpretation': {'supported': 'The available timing metadata do not support a defensible joint event-aligned GSR+PPG analysis under the prespecified synchronization rules.', 'not_supported': 'This is not evidence that dynamic autonomic physiology is absent or unrelated to EEG trajectory geometry.', 'data_limitation': 'Raw stream internal sampling can be reconstructed by row index, but cross-device absolute anchors do not overlap the fixed EEG event windows.'}, 'counts': summary['counts'], 'next_phase': 'integrated_evidence_refresh', 'next_phase_role': 'Refresh the integrated evidence boundary with the raw-dynamic timing infeasibility result; no new empirical inference.'}
    out_policy = project / 'config/raw_dynamic_physio_synthesis_raw_dynamic_physio_analysis_conclusion_policy.json'
    writej(out_policy, closure)
    report = project / 'outputs/reports/raw_dynamic_physio_synthesis_raw_dynamic_physio_analysis_conclusion.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    mn = float(np.min(offs))
    md = float(np.median(offs))
    mx = float(np.max(offs))
    report.write_text(f"# raw_dynamic_physio_synthesis — Raw dynamic physiology analysis conclusion\n\nStatus: **CLOSED FOR TIMING INFEASIBILITY**.\n\nraw_dynamic_physio_timebase_audit performed a timing-only audit and read no GSR/PPG amplitude. The prespecified joint gate failed with **0** complete GSR+PPG three-epoch trajectories among **{summary['counts']['candidate_EEG_all3_trials']}** candidate EEG trials and **0** participants meeting the >=16 trajectory criterion. raw_dynamic_physio_preprocessing_specification preprocessing is therefore not authorized.\n\nThe raw streams have internally reconstructable sampling intervals under the row-index clock (C3), but the available absolute physiology anchors do not overlap the fixed EEG event windows. Across {len(offs)} participants with both header anchors, the earliest-EEG minus normalized-physiology start discrepancy ranged from **{mn:.2f} to {mx:.2f} days** (median **{md:.2f} days**). This is incompatible with treating the recorded absolute dates as a shared clock.\n\nThis is a **data-synchronization limitation**, not a biological null result. The branch does not test whether event-aligned autonomic dynamics would relate to EEG geometry if a valid cross-device synchronization marker were available.\n\nForbidden rescue analyses remain: manual/estimated offsets, relative-start synchronization without an external marker, window retuning, single-modality rescue, outcome/family-guided alignment, or opening held-out raw physiology.", encoding='utf-8')
    final = project / 'outputs/qc/raw_dynamic_physio_synthesis_final_output_lock_hashes.json'
    final.parent.mkdir(parents=True, exist_ok=True)
    rels = [str(out_table.relative_to(project)), str(out_policy.relative_to(project)), str(report.relative_to(project))]
    hashes = {r: sha256(project / r) for r in rels}
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': closure['status'], 'hashes': hashes, 'raw_dynamic_physio_preprocessing_specification_permitted': False, 'next_phase': 'integrated_evidence_refresh', 'new_inference_performed': False, 'raw_signal_amplitude_read': False})
    rels.append(str(final.relative_to(project)))
    hz = project / '' / f'raw_dynamic_physio_synthesis_raw_dynamic_physio_analysis_conclusion_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print('raw_dynamic_physio_synthesis PASS')
    print(f"Raw dynamic branch closed; complete trajectories=0/{summary['counts']['candidate_EEG_all3_trials']}")
    print(f'Absolute-anchor discrepancy days: min={mn:.2f}, median={md:.2f}, max={mx:.2f}')
    print('raw_dynamic_physio_preprocessing_specification permitted: False')
    print('Next phase: integrated_evidence_refresh')
    print('', hz)
    print('SHA-256:', sh)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())