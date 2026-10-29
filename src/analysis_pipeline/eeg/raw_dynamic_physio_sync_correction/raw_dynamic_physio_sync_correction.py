# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'raw_dynamic_physio_sync_correction'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
SNAPSHOT_SHA = 'f3b086adfe0fbe7066ce578a2189985818ef1106a10b9f5eac9dc0998bc0b320'
B_LOCK = 'outputs/qc/raw_dynamic_physio_timebase_audit_final_output_lock_hashes.json'
D_LOCK = 'outputs/qc/raw_dynamic_physio_synthesis_final_output_lock_hashes.json'
B_STREAM = 'outputs/qc/raw_dynamic_physio_timebase_audit_raw_physio_timing_stream_audit.csv'
B_PART = 'outputs/tables/raw_dynamic_physio_timebase_audit_participant_clock_selection.csv'
raw_eeg_epoch_folds_CONTRACT = 'config/raw_eeg_epoch_folds_raw_eeg_epoch_folds_contract.json'
raw_eeg_epoch_folds_EPOCH = 'outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv'
raw_eeg_epoch_folds_HEADERS = 'outputs/qc/raw_eeg_epoch_folds_raw_edf_header_inventory.csv'
EPOCH_FAMS = ['stimulus_early', 'stimulus_late', 'rating_early']
TZ = timezone(timedelta(hours=8))
ENDTIME_UNCERTAINTY_SEC = 60.0

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def writej(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def writecsv(p, rows, fields=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict('records')
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with p.open('w', newline='', encoding='utf-8') as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

def boolish(x):
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

def verify_lock(project: Path, rel: str, phase: str):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'Missing upstream lock: {rel}')
    lock = readj(p)
    if lock.get('phase') != phase:
        raise RuntimeError(f"Unexpected phase in {rel}: {lock.get('phase')}")
    for r, h in lock.get('hashes', {}).items():
        q = project / r
        if not q.is_file():
            raise RuntimeError(f'Missing fixed upstream output: {r}')
        g = sha256(q)
        if g != h:
            raise RuntimeError(f'Upstream hash mismatch {r}: {g} != {h}')
    return lock

def verify_raw_eeg_epoch_folds(project: Path):
    p = project / raw_eeg_epoch_folds_CONTRACT
    if not p.is_file():
        raise RuntimeError('Missing raw_eeg_epoch_folds contract')
    c = readj(p)
    if c.get('phase') != 'raw_eeg_epoch_folds':
        raise RuntimeError('Wrong raw_eeg_epoch_folds contract phase')
    if c.get('eeg_amplitude_read') is not False:
        raise RuntimeError('raw_eeg_epoch_folds boundary mismatch')
    oh = c.get('output_hashes', {})
    for rel, key in [(raw_eeg_epoch_folds_EPOCH, 'epoch_manifest'), (raw_eeg_epoch_folds_HEADERS, 'raw_headers')]:
        q = project / rel
        if not q.is_file():
            raise RuntimeError(f'Missing raw_eeg_epoch_folds input: {rel}')
        exp = oh.get(key)
        if not exp:
            raise RuntimeError(f'raw_eeg_epoch_folds contract lacks output hash {key}')
        got = sha256(q)
        if got != exp:
            raise RuntimeError(f'raw_eeg_epoch_folds {key} hash mismatch: {got} != {exp}')
    return c

def normalize_date(s: str) -> str:
    return str(s).strip().replace('-', '/') if re.match('^\\d{4}/\\d{1,2}-\\d{1,2}$', str(s).strip()) else str(s).strip()

def parse_readme_endtimes(text: str):
    out = {}
    insec = False
    for line in text.splitlines():
        if 'Endtime of original EEG files' in line:
            insec = True
            continue
        if not insec:
            continue
        if line.startswith('## ') and out:
            break
        m = re.match('^\\s*(\\d{1,2})\\s+(\\d{4}/\\d{1,2}(?:/|-)\\d{1,2})\\s+(\\d{1,2}:\\d{2})(?:\\s+(\\d{1,2}:\\d{2}))?\\s*$', line.replace('\t', ' '))
        if m:
            sid = f'sub-{int(m.group(1)):02d}'
            date = normalize_date(m.group(2))
            out[sid] = (date, m.group(3), m.group(4) or '')
    return out

def local_readme_audit(dataset: Path, snap: pd.DataFrame, relevant: set[str]):
    paths = [dataset / 'README.md', dataset / 'README', dataset / 'Readme.md']
    p = next((x for x in paths if x.is_file()), None)
    rows = []
    if p is None:
        for sid in sorted(relevant):
            rows.append({'subject_id': sid, 'local_readme_found': False, 'local_match_to_prespecified_snapshot': None, 'detail': 'local README not found; prespecified factual snapshot used'})
        return (rows, None)
    parsed = parse_readme_endtimes(p.read_text(encoding='utf-8', errors='replace'))
    ss = snap.set_index('subject_id')
    mism = []
    for sid in sorted(relevant):
        if sid not in parsed:
            ok = False
            detail = 'subject row not parsed from local README'
        else:
            a = parsed[sid]
            b = (normalize_date(ss.loc[sid, 'date']), str(ss.loc[sid, 'endtime1']), '' if pd.isna(ss.loc[sid, 'endtime2']) else str(ss.loc[sid, 'endtime2']))
            ok = a == b
            detail = f'local={a}; snapshot={b}'
        rows.append({'subject_id': sid, 'local_readme_found': True, 'local_match_to_prespecified_snapshot': ok, 'detail': detail})
        if not ok:
            mism.append(sid)
    if mism:
        raise RuntimeError(f'Local README end-time table disagrees with prespecified snapshot for: {mism}')
    return (rows, p)

def end_epoch(date_s, time_s):
    d = normalize_date(date_s)
    return datetime.strptime(f'{d} {time_s}', '%Y/%m/%d %H:%M').replace(tzinfo=TZ).timestamp()

def nearest_boundary(start, fs, n, x):
    if n <= 0:
        return math.nan
    k = int(round((x - start) * fs))
    k = max(0, min(n - 1, k))
    return start + k / fs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        assert abs(end_epoch('2023/8/20', '18:34') - 1692527640.0) < 1e-06
        assert normalize_date('2023/10-4') == '2023/10/4'
        print('raw_dynamic_physio_sync_correction self-test: PASS')
        return 0
    project = a.project
    dataset = a.dataset
    if sha256(a.snapshot) != SNAPSHOT_SHA:
        raise RuntimeError('prespecified README end-time snapshot SHA mismatch')
    b = verify_lock(project, B_LOCK, 'raw_dynamic_physio_timebase_audit')
    d = verify_lock(project, D_LOCK, 'raw_dynamic_physio_synthesis')
    verify_raw_eeg_epoch_folds(project)
    if b.get('raw_signal_amplitude_read') is not False or d.get('raw_signal_amplitude_read') is not False:
        raise RuntimeError('Historical raw-amplitude boundary mismatch')
    stream = pd.read_csv(project / B_STREAM, low_memory=False)
    oldpart = pd.read_csv(project / B_PART, low_memory=False)
    epochs = pd.read_csv(project / raw_eeg_epoch_folds_EPOCH, low_memory=False)
    headers = pd.read_csv(project / raw_eeg_epoch_folds_HEADERS, low_memory=False)
    snap = pd.read_csv(a.snapshot, dtype=str).fillna('')
    relevant = set(stream['subject_id'].astype(str).unique())
    missing = relevant - set(snap['subject_id'])
    if missing:
        raise RuntimeError(f'prespecified end-time snapshot lacks subjects: {sorted(missing)}')
    sel = oldpart[oldpart['subject_id'].astype(str).isin(relevant)].copy()
    badsel = sel[(sel.gsr_selected_candidate.astype(str) != 'C3') | (sel.ppg_selected_candidate.astype(str) != 'C3')]
    if len(badsel):
        raise RuntimeError('Correction is only valid for streams previously prespecified to C3; unexpected non-C3 subjects present')
    readme_rows, local_readme = local_readme_audit(dataset, snap, relevant)
    spans = {}
    stream_rows = []
    for _, r in stream.iterrows():
        sid = str(r.subject_id)
        mod = str(r.modality)
        fs = float(r.sampling_frequency_hz)
        n = int(r.data_rows)
        s = float(r.normalized_first_abs_sec)
        e = s + (n - 1) / fs if n > 0 else math.nan
        ok_start = boolish(r.json_start_agreement_after_timezone_normalization)
        if not ok_start:
            raise RuntimeError(f'{sid} {mod}: C3 normalized first sample did not agree with JSON StartTime in fixed raw_dynamic_physio_timebase_audit')
        spans[sid, mod] = (s, e, fs, n)
        stream_rows.append({'subject_id': sid, 'modality': mod, 'c3_start_epoch_sec': s, 'c3_stop_epoch_sec': e, 'sampling_frequency_hz': fs, 'data_rows': n, 'source': 'fixed_raw_dynamic_physio_timebase_audit_timing_stream_only', 'raw_amplitude_read': False})
    for sid in relevant:
        if (sid, 'GSR') not in spans or (sid, 'PPG') not in spans:
            raise RuntimeError(f'{sid}: missing GSR or PPG C3 span')
        if abs(spans[sid, 'GSR'][0] - spans[sid, 'PPG'][0]) > 0.25:
            raise RuntimeError(f'{sid}: GSR/PPG C3 start disagreement >0.25s')
    dur = dict(zip(headers.raw_source_path.astype(str), pd.to_numeric(headers.recording_duration_sec, errors='coerce')))
    ss = snap.set_index('subject_id')
    ep = epochs[epochs.subject_id.astype(str).isin(relevant)].copy()
    ep['epoch_available'] = ep['epoch_available'].map(boolish)
    run_rows = []
    anchor_map = {}
    for (sid, run, path), g in ep.groupby(['subject_id', 'raw_run', 'raw_source_path'], dropna=False):
        sid = str(sid)
        run = int(run)
        path = str(path)
        if path not in dur or not math.isfinite(float(dur[path])):
            raise RuntimeError(f'Missing fixed EDF duration: {path}')
        endcol = 'endtime1' if run == 1 else 'endtime2'
        et = str(ss.loc[sid, endcol]).strip()
        if not et:
            raise RuntimeError(f'{sid} run {run}: prespecified README snapshot has no {endcol}')
        ee = end_epoch(ss.loc[sid, 'date'], et)
        rs = ee - float(dur[path])
        anchor_map[sid, run, path] = (rs, ee)
        run_rows.append({'subject_id': sid, 'raw_run': run, 'raw_source_path': path, 'readme_date': ss.loc[sid, 'date'], 'readme_endtime': et, 'readme_end_resolution': 'minute', 'endtime_uncertainty_sec_conservative': ENDTIME_UNCERTAINTY_SEC, 'recording_duration_sec': float(dur[path]), 'estimated_run_start_epoch_sec': rs, 'estimated_run_end_epoch_sec': ee, 'anchor_source': 'dataset README original EEG endtime minus fixed EDF duration', 'edf_meas_date_used': False, 'raw_amplitude_read': False})
    epoch_rows = []
    trial_map = {}
    for _, r in ep.sort_values(['subject_id', 'trial_number', 'epoch_family']).iterrows():
        sid = str(r.subject_id)
        tr = int(r.trial_number)
        fam = str(r.epoch_family)
        key = (sid, int(r.raw_run), str(r.raw_source_path))
        rs, ee = anchor_map[key]
        if r.epoch_available and math.isfinite(float(r.epoch_start_sec)) and math.isfinite(float(r.epoch_stop_sec)):
            es = rs + float(r.epoch_start_sec)
            ez = rs + float(r.epoch_stop_sec)
            mods = {}
            for mod in ['GSR', 'PPG']:
                ps, pz, fs, n = spans[sid, mod]
                nominal = es >= ps and ez <= pz
                robust = es - ENDTIME_UNCERTAINTY_SEC >= ps and ez + ENDTIME_UNCERTAINTY_SEC <= pz
                nb0 = nearest_boundary(ps, fs, n, es)
                nb1 = nearest_boundary(ps, fs, n, ez)
                mods[mod] = (nominal, robust, nb0, nb1)
            bd = max(abs(mods['GSR'][2] - mods['PPG'][2]), abs(mods['GSR'][3] - mods['PPG'][3]))
            joint_nom = mods['GSR'][0] and mods['PPG'][0] and (bd <= 0.25)
            joint_rob = mods['GSR'][1] and mods['PPG'][1] and (bd <= 0.25)
        else:
            es = ez = math.nan
            bd = math.nan
            joint_nom = joint_rob = False
            mods = {'GSR': (False, False, math.nan, math.nan), 'PPG': (False, False, math.nan, math.nan)}
        epoch_rows.append({'subject_id': sid, 'trial_number': tr, 'epoch_family': fam, 'raw_run': int(r.raw_run), 'eeg_epoch_available': bool(r.epoch_available), 'coarse_epoch_abs_start_sec': es, 'coarse_epoch_abs_stop_sec': ez, 'readme_anchor_uncertainty_sec': ENDTIME_UNCERTAINTY_SEC, 'gsr_nominal_span_covered': mods['GSR'][0], 'ppg_nominal_span_covered': mods['PPG'][0], 'gsr_robust_span_covered_plusminus60s': mods['GSR'][1], 'ppg_robust_span_covered_plusminus60s': mods['PPG'][1], 'cross_modality_nearest_boundary_disagreement_sec': bd, 'joint_nominal_span_covered': joint_nom, 'joint_robust_span_covered_plusminus60s': joint_rob, 'exact_event_sync_established': False, 'raw_amplitude_read': False})
        trial_map.setdefault((sid, tr), {})[fam] = (bool(r.epoch_available), joint_nom, joint_rob)
    trial_rows = []
    participant_rows = []
    bysub = {sid: [] for sid in sorted(relevant)}
    for (sid, tr), st in sorted(trial_map.items()):
        all3 = all((st.get(f, (False, False, False))[0] for f in EPOCH_FAMS))
        nom = all3 and all((st.get(f, (False, False, False))[1] for f in EPOCH_FAMS))
        rob = all3 and all((st.get(f, (False, False, False))[2] for f in EPOCH_FAMS))
        rr = {'subject_id': sid, 'trial_number': tr, 'eeg_all_three_epochs_available': all3, 'joint_nominal_three_epoch_span_coverage': nom, 'joint_robust_three_epoch_span_coverage_plusminus60s': rob, 'exact_event_sync_established': False}
        trial_rows.append(rr)
        bysub[sid].append(rr)
    for sid in sorted(relevant):
        rr = bysub[sid]
        cand = sum((x['eeg_all_three_epochs_available'] for x in rr))
        nom = sum((x['joint_nominal_three_epoch_span_coverage'] for x in rr))
        rob = sum((x['joint_robust_three_epoch_span_coverage_plusminus60s'] for x in rr))
        participant_rows.append({'subject_id': sid, 'candidate_eeg_all3_trials': cand, 'nominal_joint_span_covered_trials': nom, 'robust_joint_span_covered_trials_plusminus60s': rob, 'coarse_min16_pass': rob >= 16, 'exact_sync_status': 'unresolved', 'raw_amplitude_analysis_permitted': False})
    total_cand = sum((x['candidate_eeg_all3_trials'] for x in participant_rows))
    total_nom = sum((x['nominal_joint_span_covered_trials'] for x in participant_rows))
    total_rob = sum((x['robust_joint_span_covered_trials_plusminus60s'] for x in participant_rows))
    passn = sum((x['coarse_min16_pass'] for x in participant_rows))
    frac = total_rob / total_cand if total_cand else 0.0
    coarse_pass = passn >= 18 and frac >= 0.7
    exact_sync = False
    amp_permitted = False
    next_permitted = coarse_pass
    paths = {'correction': project / 'config/raw_dynamic_physio_sync_correction_sync_correction_addendum.json', 'readme': project / 'outputs/qc/raw_dynamic_physio_sync_correction_local_readme_endtime_verification.csv', 'stream': project / 'outputs/qc/raw_dynamic_physio_sync_correction_inherited_c3_stream_spans.csv', 'runs': project / 'outputs/tables/raw_dynamic_physio_sync_correction_readme_anchored_run_spans.csv', 'epochs': project / 'outputs/tables/raw_dynamic_physio_sync_correction_coarse_epoch_span_coverage.csv', 'trials': project / 'outputs/tables/raw_dynamic_physio_sync_correction_coarse_trial_span_coverage.csv', 'participants': project / 'outputs/tables/raw_dynamic_physio_sync_correction_coarse_participant_feasibility.csv', 'summary': project / 'outputs/qc/raw_dynamic_physio_sync_correction_sync_correction_summary.json', 'report': project / 'outputs/reports/raw_dynamic_physio_sync_correction_raw_dynamic_physio_sync_correction_report.md', 'next': project / 'analysis_specifications/S054_raw_dynamic_physio_sync_correction_exact.json', 'contract': project / 'config/raw_dynamic_physio_sync_correction_raw_dynamic_physio_sync_correction_contract.json'}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    correction = {'phase': PHASE, 'generated_at': now(), 'status': 'correction_addendum_fixed', 'supersedes_only': ['raw_dynamic_physio_timebase_audit assumption that EDF meas_date is the original acquisition-time anchor', 'raw_dynamic_physio_synthesis/topographic_analysis conclusion that raw-dynamic physiology has zero temporal overlap / is infeasible based on that EDF-meas_date anchor'], 'preserves': ['all EEG empirical results', 'all subjective results', 'analysis_stage trial-level physiology M1-M3 results', 'held-out physiology closure', 'all raw physiology amplitudes still unopened'], 'reason': 'Dataset README provides original EEG end times on acquisition dates that differ from the converted EDF meas_date dates. The fixed raw_dynamic_physio_timebase_audit C3 physiology clocks therefore must be re-audited against dataset-provided acquisition timing rather than EDF meas_date.', 'prespecified_anchor_for_coarse_reaudit': 'README original EEG run endtime - fixed EDF duration; timezone UTC+08:00; conservative +/-60 s because README endtimes are minute-resolution', 'important_boundary': 'Coarse temporal overlap is a feasibility check only. It does NOT establish <=1 s event synchronization and does NOT permit raw-amplitude analysis.', 'outcomes_family_reports_used': False, 'raw_signal_amplitude_read': False, 'heldout_raw_physio_opened': False, 'prespecified_readme_endtime_snapshot_sha256': SNAPSHOT_SHA, 'source_notes': {'dataset': 'OpenNeuro ds005540 / EmoEEG-MC', 'readme_section': 'Endtime of original EEG files', 'scientific_data_usage_note': 'Published dataset documentation states that physiology JSON start times enable EEG alignment at one-second precision; exact recovery remains to be established locally before amplitude access.'}}
    writej(paths['correction'], correction)
    writecsv(paths['readme'], readme_rows)
    writecsv(paths['stream'], stream_rows)
    writecsv(paths['runs'], run_rows)
    writecsv(paths['epochs'], epoch_rows)
    writecsv(paths['trials'], trial_rows)
    writecsv(paths['participants'], participant_rows)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'coarse_overlap_reopened_exact_sync_still_unresolved' if coarse_pass else 'coarse_overlap_gate_failed', 'historical_correction': True, 'counts': {'audited_participants': len(participant_rows), 'candidate_EEG_all3_trials': total_cand, 'nominal_joint_span_covered_trials': total_nom, 'robust_joint_span_covered_trials_plusminus60s': total_rob, 'participants_with_min16_robust_span_covered': passn}, 'fractions': {'robust_joint_span_coverage_fraction': frac}, 'gates': {'minimum_participants_required': 18, 'minimum_coverage_fraction_required': 0.7, 'coarse_coverage_gate_pass': coarse_pass}, 'synchronization': {'EDF_meas_date_rejected_as_original_acquisition_anchor': True, 'README_endtime_resolution': 'minute', 'coarse_anchor_uncertainty_sec': ENDTIME_UNCERTAINTY_SEC, 'exact_event_sync_established': exact_sync, 'target_exact_sync_precision_sec': 1.0}, 'boundaries': {'raw_GSR_amplitude_read': False, 'raw_PPG_amplitude_read': False, 'EEG_sample_amplitude_read': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False}, 'raw_amplitude_analysis_permitted': amp_permitted, 'exact_physio_eeg_sync_recovery_permitted': next_permitted}
    writej(paths['summary'], summary)
    nextpol = {'phase': PHASE, 'generated_at': now(), 'next_phase': 'exact_physio_eeg_sync_recovery', 'next_phase_permitted': next_permitted, 'role': 'Outcome-blind exact synchronization recovery/audit. Establish per-run EEG↔physiology offset to <=1 s before any raw amplitude access.', 'required_order': ['audit dataset-provided exact timing/acquisition metadata and code', 'audit timing-column discontinuities and run-boundary consistency without signal amplitude', 'audit author-released physiology segmentation metadata as an independent calibration source if available', 'only after an exact-sync solution is prespecified may a later phase open Discovery raw amplitudes'], 'forbidden': ['using Valence/Liking/reports/emotion labels/family assignments to choose offsets', 'using EEG/GSR/PPG amplitude to choose an offset in exact_physio_eeg_sync_recovery', 'moving epoch windows', 'opening Validation or Confirmation raw physiology', 'treating coarse README-minute overlap as exact event synchronization'], 'fallback_if_exact_sync_not_recovered': 'Keep raw event-aligned amplitude branch closed, but separately audit author-provided within-trial 3-s GSR derivatives as a stimulus-only dynamic physiology representation.'}
    writej(paths['next'], nextpol)
    report = f"# raw_dynamic_physio_sync_correction — raw dynamic physiology synchronization correction\n\nStatus: **{('COARSE FEASIBILITY RE-OPENED' if coarse_pass else 'COARSE GATE FAILED')}**.\n\nraw_dynamic_physio_timebase_audit used EDF `meas_date` as an absolute acquisition anchor. That added assumption is superseded here. The dataset README's original EEG end times are used only for a coarse, outcome-blind overlap re-audit.\n\n- Audited raw-both Discovery participants: **{len(participant_rows)}**\n- Candidate EEG three-epoch trials: **{total_cand}**\n- Robust joint GSR+PPG span-covered trajectories (allowing ±60 s README endtime uncertainty): **{total_rob}** ({frac:.4f})\n- Participants with >=16 robust span-covered trajectories: **{passn}**\n- prespecified coarse gate (>=18 participants and >=70%): **{('PASS' if coarse_pass else 'FAIL')}**\n\n**This does not yet establish exact event synchronization.** README end times are minute-resolution, while the dataset paper describes physiology alignment at one-second precision. Therefore raw GSR/PPG amplitudes remain unopened and event-fixed amplitude analysis remains forbidden.\n\nThe next authorized step is exact_physio_eeg_sync_recovery, a timing/metadata-only exact-sync recovery audit. Historical raw_dynamic_physio_synthesis/topographic_analysis raw-feasibility statements are superseded only on this point; all other empirical results remain unchanged."
    paths['report'].write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'inputs': {'raw_dynamic_physio_timebase_audit_lock_sha256': sha256(project / B_LOCK), 'raw_dynamic_physio_synthesis_lock_sha256': sha256(project / D_LOCK), 'raw_eeg_epoch_folds_contract_sha256': sha256(project / raw_eeg_epoch_folds_CONTRACT), 'raw_eeg_epoch_folds_epoch_sha256': sha256(project / raw_eeg_epoch_folds_EPOCH), 'raw_eeg_epoch_folds_header_inventory_sha256': sha256(project / raw_eeg_epoch_folds_HEADERS), 'raw_dynamic_physio_timebase_audit_stream_sha256': sha256(project / B_STREAM), 'prespecified_readme_endtime_snapshot_sha256': SNAPSHOT_SHA, 'local_dataset_readme_sha256': sha256(local_readme) if local_readme else None}, 'outputs': {}, 'coarse_coverage_gate_pass': coarse_pass, 'exact_event_sync_established': False, 'raw_signal_amplitude_read': False, 'heldout_raw_physio_opened': False}
    lockhash = {}
    for k, p in paths.items():
        if k == 'contract':
            continue
        lockhash[str(p.relative_to(project))] = sha256(p)
    contract['outputs'] = lockhash
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/raw_dynamic_physio_sync_correction_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': lockhash, 'exact_physio_eeg_sync_recovery_permitted': next_permitted, 'raw_amplitude_analysis_permitted': False, 'exact_event_sync_established': False, 'raw_signal_amplitude_read': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'raw_dynamic_physio_sync_correction_raw_dynamic_physio_sync_correction_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] raw_dynamic_physio_sync_correction PASS. coarse participants={passn}; robust complete={total_rob}/{total_cand} ({frac:.4f}); exact_sync=False; amplitude_opened=False', flush=True)
    print('', hz, flush=True)
    print('SHA-256:', sh, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())