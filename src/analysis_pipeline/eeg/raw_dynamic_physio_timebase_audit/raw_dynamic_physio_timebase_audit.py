# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, warnings
from array import array
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'raw_dynamic_physio_timebase_audit'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
A_LOCK = 'outputs/qc/raw_dynamic_physio_alignment_design_final_output_lock_hashes.json'
A_POLICY = 'analysis_specifications/S052_raw_dynamic_physio_alignment.json'
A_NEXT = 'analysis_specifications/S053_raw_dynamic_physio_alignment.json'
A_ROSTER = 'outputs/tables/raw_dynamic_physio_alignment_design_discovery_raw_physio_source_roster.csv'
raw_eeg_epoch_folds_TIMING = 'outputs/tables/raw_eeg_epoch_folds_authoritative_trial_timing_manifest.csv'
raw_eeg_epoch_folds_EPOCHS = 'outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv'
CANDIDATES = ['outcome_model_integrity', 'C2', 'C3']
EPS = 1e-09

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

def writecsv(p, rows, fields=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with p.open('w', newline='', encoding='utf-8') as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

def boolish(x):
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

def finite(x):
    try:
        return math.isfinite(float(x))
    except:
        return False

def verify_raw_dynamic_physio_alignment_design(project: Path):
    lock = readj(project / A_LOCK)
    pol = readj(project / A_POLICY)
    nxt = readj(project / A_NEXT)
    if lock.get('phase') != 'raw_dynamic_physio_alignment_design' or not lock.get('raw_dynamic_physio_timebase_audit_permitted'):
        raise RuntimeError('raw_dynamic_physio_alignment_design does not authorize raw_dynamic_physio_timebase_audit')
    if pol.get('counts_at_specification', {}).get('raw_signal_values_read') is not False:
        raise RuntimeError('raw_dynamic_physio_alignment_design boundary mismatch')
    if nxt.get('next_phase') != 'raw_dynamic_physio_timebase_audit' or not nxt.get('next_phase_permitted'):
        raise RuntimeError('prespecified raw_dynamic_physio_timebase_audit timing policy mismatch')
    for rel, h in lock.get('hashes', {}).items():
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing fixed raw_dynamic_physio_alignment_design output: {rel}')
        got = sha256(p)
        if got != h:
            raise RuntimeError(f'raw_dynamic_physio_alignment_design hash mismatch: {rel}: {got} != {h}')
    return (lock, pol, nxt)

def parse_meas_date(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        d = v
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    if isinstance(v, (tuple, list)) and len(v) >= 1:
        try:
            return float(v[0]) + (float(v[1]) / 1000000.0 if len(v) > 1 else 0.0)
        except:
            return None
    return None

def read_edf_header_start(path: Path):
    try:
        import mne
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose='ERROR')
        try:
            t = parse_meas_date(raw.info.get('meas_date'))
            sf = float(raw.info['sfreq']) if raw.info.get('sfreq') else None
            n = int(raw.n_times)
        finally:
            try:
                raw.close()
            except:
                pass
        return (t, sf, n, '')
    except Exception as e:
        return (None, None, None, f'{type(e).__name__}: {e}')

def raw_token_to_epoch(token, cache):
    s = token.strip().strip('"').strip("'")
    if not s:
        return (None, 'empty')
    if s in cache:
        return cache[s]
    try:
        x = float(s)
        if math.isfinite(x):
            out = (x, 'numeric')
            cache[s] = out
            return out
    except Exception:
        pass
    try:
        ts = pd.to_datetime(s, utc=True, errors='raise')
        if pd.isna(ts):
            return (None, 'unparseable')
        out = (float(ts.timestamp()), 'datetime_epoch_seconds')
        cache[s] = out
        return out
    except Exception:
        return (None, 'unparseable')

def sidecar_columns(meta):
    c = meta.get('Columns', meta.get('columns', []))
    if isinstance(c, str):
        c = [x.strip() for x in c.split('|')]
    return list(c) if isinstance(c, (list, tuple)) else []

def stream_time_column(tsv: Path, json_path: Path, expected_name: str, expected_fs: float):
    meta = readj(json_path)
    cols = sidecar_columns(meta)
    idx = cols.index(expected_name) if expected_name in cols else None
    sf = meta.get('SamplingFrequency', meta.get('sampling_frequency', expected_fs))
    try:
        sf = float(sf)
    except:
        sf = float(expected_fs)
    st = meta.get('StartTime', meta.get('start_time'))
    try:
        st = float(st)
    except:
        st = None
    vals = array('d')
    cache = {}
    malformed = nonfinite = neg = dup = 0
    positive = []
    kinds = set()
    data_rows = 0
    header_skipped = False
    stream_error = ''
    try:
        with gzip.open(tsv, 'rt', encoding='utf-8', errors='replace', newline='') as f:
            for li, line in enumerate(f):
                toks = line.rstrip('').split('\t')
                if idx is None:
                    if li == 0 and expected_name in toks:
                        idx = toks.index(expected_name)
                        header_skipped = True
                        continue
                    raise RuntimeError(f'{tsv}: sidecar Columns lacks {expected_name} and no exact TSV header fallback exists')
                if li == 0 and idx < len(toks) and (toks[idx].strip() == expected_name):
                    header_skipped = True
                    continue
                data_rows += 1
                if idx >= len(toks):
                    malformed += 1
                    continue
                val, kind = raw_token_to_epoch(toks[idx], cache)
                if val is None:
                    malformed += 1
                    continue
                if not math.isfinite(val):
                    nonfinite += 1
                    continue
                vals.append(val)
                kinds.add(kind)
                if len(vals) > 1:
                    d = vals[-1] - vals[-2]
                    if d < 0:
                        neg += 1
                    elif d == 0:
                        dup += 1
                    elif len(positive) < 100000:
                        positive.append(d)
    except Exception as e:
        stream_error = f'{type(e).__name__}: {e}'
    arr = np.asarray(vals, dtype=float)
    posmed = float(np.median(positive)) if positive else math.nan
    first = float(arr[0]) if len(arr) else math.nan
    last = float(arr[-1]) if len(arr) else math.nan
    tzshift = 0.0
    tz_rule = 'none'
    if len(arr) and st is not None and math.isfinite(st):
        delta = first - st
        h = round(delta / 3600.0)
        if abs(h) <= 14 and abs(delta - h * 3600.0) <= 2.0:
            tzshift = float(h * 3600.0)
            tz_rule = f'integer_hour_{h:+d}'
    normfirst = first - tzshift if len(arr) else math.nan
    start_agreement = abs(normfirst - st) <= 2.0 if len(arr) and st is not None and math.isfinite(st) else None
    return {'meta': meta, 'columns': cols, 'time_index': idx, 'sampling_frequency': sf, 'json_start_time': st, 'parsed_times': arr, 'data_rows': data_rows, 'parsed_values': len(arr), 'malformed': malformed, 'nonfinite': nonfinite, 'negative_deltas': neg, 'duplicate_values': dup, 'median_positive_dt': posmed, 'first_raw': first, 'last_raw': last, 'timestamp_kinds': '|'.join(sorted(kinds)), 'timezone_shift_sec': tzshift, 'timezone_rule': tz_rule, 'normalized_first_abs': normfirst, 'json_start_agreement_after_timezone_normalization': start_agreement, 'header_skipped': header_skipped, 'stream_error': stream_error}

def candidate_diag(info, code, windows):
    fs = float(info['sampling_frequency'])
    n = int(info['data_rows'])
    arr = info['parsed_times']
    shift = float(info['timezone_shift_sec'])
    starttime = info['json_start_time']
    first = info['first_raw']
    normfirst = info['normalized_first_abs']
    if code == 'outcome_model_integrity':
        absarr = arr - shift if len(arr) else np.array([])
        dt = info['median_positive_dt']
        origin_ok = bool(len(arr) and starttime is not None and math.isfinite(starttime))
    elif code == 'C2':
        absarr = arr - shift if len(arr) else np.array([])
        dt = info['median_positive_dt']
        origin_ok = bool(len(arr))
    else:
        absarr = None
        dt = 1.0 / fs if fs > 0 else math.nan
        origin_ok = bool(n > 0 and math.isfinite(normfirst))
    relerr = abs(dt - 1.0 / fs) / (1.0 / fs) if fs > 0 and math.isfinite(dt) else math.inf
    structural = bool(origin_ok and info['negative_deltas'] == 0 and (relerr <= 0.02) and (not info['stream_error']))
    covered = 0
    sample_fracs = []
    if structural:
        for w in windows:
            s = float(w['epoch_abs_start'])
            e = float(w['epoch_abs_stop'])
            expected = max(1, int(round((e - s) * fs)))
            if code == 'C3':
                lo = math.ceil((s - normfirst) * fs - EPS)
                hi = math.ceil((e - normfirst) * fs - EPS)
                lo = max(0, min(n, lo))
                hi = max(0, min(n, hi))
                count = max(0, hi - lo)
            else:
                lo = int(np.searchsorted(absarr, s, side='left'))
                hi = int(np.searchsorted(absarr, e, side='left'))
                count = max(0, hi - lo)
            frac = count / expected
            sample_fracs.append(frac)
            if frac >= 0.95:
                covered += 1
    return {'candidate': code, 'origin_available': origin_ok, 'median_dt_sec': dt, 'median_dt_relative_error': relerr, 'structural_pass': structural, 'fixed_available_epoch_rows': len(windows), 'fully_covered_epoch_rows': covered, 'median_expected_sample_fraction': float(np.median(sample_fracs)) if sample_fracs else 0.0}

def candidate_sample_metrics(info, code, s, e):
    fs = float(info['sampling_frequency'])
    n = int(info['data_rows'])
    arr = info['parsed_times']
    shift = float(info['timezone_shift_sec'])
    normfirst = float(info['normalized_first_abs'])
    expected = max(1, int(round((e - s) * fs)))
    if code == 'C3':
        lo = math.ceil((s - normfirst) * fs - EPS)
        hi = math.ceil((e - normfirst) * fs - EPS)
        lo = max(0, min(n, lo))
        hi = max(0, min(n, hi))
        count = max(0, hi - lo)
        if n > 0:

            def nearest(x):
                k = int(round((x - normfirst) * fs))
                k = max(0, min(n - 1, k))
                return normfirst + k / fs
            ns = nearest(s)
            ne = nearest(e)
        else:
            ns = ne = math.nan
    else:
        absarr = arr - shift
        lo = int(np.searchsorted(absarr, s, side='left'))
        hi = int(np.searchsorted(absarr, e, side='left'))
        count = max(0, hi - lo)

        def near(x):
            if len(absarr) == 0:
                return math.nan
            k = int(np.searchsorted(absarr, x, side='left'))
            cand = []
            if k < len(absarr):
                cand.append(absarr[k])
            if k > 0:
                cand.append(absarr[k - 1])
            return float(min(cand, key=lambda y: abs(y - x))) if cand else math.nan
        ns = near(s)
        ne = near(e)
    return (count, count / expected, ns, ne)

def build_epoch_abs(project, subject_set):
    ep = pd.read_csv(project / raw_eeg_epoch_folds_EPOCHS, low_memory=False)
    ep = ep[ep['subject_id'].isin(subject_set)].copy()
    ep['epoch_available'] = ep['epoch_available'].map(boolish)
    paths = sorted(set(ep.loc[ep.epoch_available, 'raw_source_path'].dropna().astype(str)))
    header = []
    startmap = {}
    for p in paths:
        t, sf, n, err = read_edf_header_start(Path(p))
        startmap[p] = t
        header.append({'raw_source_path': p, 'meas_date_epoch_sec': t, 'header_sfreq_hz': sf, 'header_n_times': n, 'meas_date_available': t is not None, 'header_error': err, 'sample_amplitude_read': False})
    ep['edf_meas_date_epoch_sec'] = ep['raw_source_path'].map(startmap)
    ep['epoch_abs_start'] = pd.to_numeric(ep['epoch_start_sec'], errors='coerce') + pd.to_numeric(ep['edf_meas_date_epoch_sec'], errors='coerce')
    ep['epoch_abs_stop'] = pd.to_numeric(ep['epoch_stop_sec'], errors='coerce') + pd.to_numeric(ep['edf_meas_date_epoch_sec'], errors='coerce')
    return (ep, header)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        fake = {'sampling_frequency': 20.0, 'data_rows': 2000, 'parsed_times': np.arange(2000) // 20 + 1700000000.0, 'timezone_shift_sec': 0.0, 'normalized_first_abs': 1700000000.0, 'json_start_time': 1700000000.0, 'first_raw': 1700000000.0, 'median_positive_dt': 1.0, 'negative_deltas': 0, 'stream_error': ''}
        ws = [{'epoch_abs_start': 1700000010.0, 'epoch_abs_stop': 1700000020.0}]
        assert not candidate_diag(fake, 'outcome_model_integrity', ws)['structural_pass']
        assert candidate_diag(fake, 'C3', ws)['structural_pass']
        c, f, _, _ = candidate_sample_metrics(fake, 'C3', 1700000010.0, 1700000020.0)
        assert c == 200 and abs(f - 1) < 1e-12
        print('raw_dynamic_physio_timebase_audit self-test: PASS')
        return 0
    project = a.project
    verify_raw_dynamic_physio_alignment_design(project)
    addendum = {'phase': PHASE, 'generated_at': now(), 'status': 'pre_timing_absolute_anchor_safety_addendum', 'reason': 'Coverage alone cannot establish synchronization. Discovery EDF measurement-date headers are therefore read without preload to anchor the already prespecified raw_eeg_epoch_folds local epoch windows to absolute time before physiology timing-column inspection.', 'additional_allowed_read': 'Discovery EEG EDF header meas_date and sfreq only; preload=False; no sample amplitude', 'raw_physio_amplitude_read': False, 'outcome_or_family_used_for_alignment': False, 'selection_resolution': {'candidate_rank': 'Among candidates with origin available, zero negative timing jumps, <=2% median-dt relative error and no stream error: maximize fully covered fixed EEG epoch rows, then median expected sample fraction, then fixed order outcome_model_integrity,C2,C3.', 'timezone_normalization': 'If first raw timestamp minus JSON StartTime is within 2 seconds of an integer-hour offset in [-14,+14], subtract that fixed integer-hour offset from raw timestamps. Timing metadata only.', 'absolute_sample_clock_for_C3': 'normalized first raw timestamp + row_index / SamplingFrequency', 'participant_complete_trial': 'all three prespecified EEG epochs available; >=95% expected samples in BOTH GSR and PPG for each epoch; nearest GSR/PPG boundary disagreement <=0.25 s'}}
    outcfg = project / 'config/raw_dynamic_physio_timebase_audit_absolute_anchor_safety_addendum.json'
    writej(outcfg, addendum)
    add_sha = sha256(outcfg)
    print(f'[{now()}] raw_dynamic_physio_alignment_design verified; pre-timing absolute-anchor addendum fixed: {add_sha}', flush=True)
    roster = pd.read_csv(project / A_ROSTER, low_memory=False)
    both = roster[roster['both_raw_modalities_exist'].map(boolish)].copy()
    rawsubs = set(both.subject_id.astype(str))
    ep, header = build_epoch_abs(project, rawsubs)
    eegsubs = set(ep.subject_id.astype(str))
    cand = both[both.subject_id.astype(str).isin(eegsubs)].copy()
    subjects = sorted(cand.subject_id.astype(str).unique())
    print(f'[{now()}] Header-only anchor audit: raw-both inventory={len(rawsubs)}, with raw_eeg_epoch_folds EEG={len(subjects)}.', flush=True)
    win_by_sub = {}
    for sub, g in ep[ep.epoch_available & ep.epoch_abs_start.notna() & ep.epoch_abs_stop.notna()].groupby('subject_id'):
        win_by_sub[sub] = g[['trial_number', 'epoch_family', 'epoch_abs_start', 'epoch_abs_stop']].to_dict('records')
    file_diags = []
    info_by = {}
    selected_by = {}
    timing_stream_rows = []
    for _, r in cand.sort_values('subject_id').iterrows():
        sub = str(r.subject_id)
        windows = win_by_sub.get(sub, [])
        for mod, pcol, jcol, fscol, tcol in [('GSR', 'gsr_tsv_path', 'gsr_json_path', 'gsr_sampling_frequency_sidecar', 'csv_time_GSR'), ('PPG', 'ppg_tsv_path', 'ppg_json_path', 'ppg_sampling_frequency_sidecar', 'csv_time_PPG')]:
            tsv = Path(str(r[pcol]))
            js = Path(str(r[jcol]))
            expfs = float(r[fscol])
            if not tsv.is_file() or not js.is_file():
                timing_stream_rows.append({'subject_id': sub, 'modality': mod, 'status': 'missing_source', 'tsv_path': str(tsv), 'json_path': str(js), 'raw_signal_amplitude_read': False})
                selected_by[sub, mod] = 'NO_SOLUTION'
                continue
            info = stream_time_column(tsv, js, tcol, expfs)
            info_by[sub, mod] = info
            timing_stream_rows.append({'subject_id': sub, 'modality': mod, 'status': 'parsed' if not info['stream_error'] else 'stream_error', 'tsv_path': str(tsv), 'json_path': str(js), 'sampling_frequency_hz': info['sampling_frequency'], 'data_rows': info['data_rows'], 'parsed_timing_values': info['parsed_values'], 'malformed_timing_rows': info['malformed'], 'negative_deltas': info['negative_deltas'], 'duplicate_timestamps': info['duplicate_values'], 'median_positive_dt_sec': info['median_positive_dt'], 'first_raw_timestamp_sec': info['first_raw'], 'last_raw_timestamp_sec': info['last_raw'], 'json_start_time_sec': info['json_start_time'], 'timezone_shift_sec': info['timezone_shift_sec'], 'timezone_rule': info['timezone_rule'], 'normalized_first_abs_sec': info['normalized_first_abs'], 'json_start_agreement_after_timezone_normalization': info['json_start_agreement_after_timezone_normalization'], 'timestamp_kinds': info['timestamp_kinds'], 'stream_error': info['stream_error'], 'raw_signal_amplitude_read': False})
            cds = []
            for code in CANDIDATES:
                d = candidate_diag(info, code, windows)
                d.update({'subject_id': sub, 'modality': mod, 'tsv_path': str(tsv)})
                cds.append(d)
                file_diags.append(d)
            eligible = [d for d in cds if d['structural_pass']]
            if eligible:
                eligible.sort(key=lambda d: (-d['fully_covered_epoch_rows'], -d['median_expected_sample_fraction'], CANDIDATES.index(d['candidate'])))
                selected_by[sub, mod] = eligible[0]['candidate']
            else:
                selected_by[sub, mod] = 'NO_SOLUTION'
    epoch_rows = []
    traj_rows = []
    participant_rows = []
    for sub in subjects:
        g = ep[ep.subject_id.astype(str).eq(sub)].copy()
        cg = selected_by.get((sub, 'GSR'), 'NO_SOLUTION')
        cp = selected_by.get((sub, 'PPG'), 'NO_SOLUTION')
        ig = info_by.get((sub, 'GSR'))
        ip = info_by.get((sub, 'PPG'))
        trial_status = {}
        for _, e in g.sort_values(['trial_number', 'epoch_family']).iterrows():
            base = {'subject_id': sub, 'trial_number': int(e.trial_number), 'epoch_family': e.epoch_family, 'eeg_epoch_available': bool(e.epoch_available), 'eeg_abs_start_sec': e.epoch_abs_start, 'eeg_abs_stop_sec': e.epoch_abs_stop, 'gsr_candidate': cg, 'ppg_candidate': cp}
            if not bool(e.epoch_available) or not finite(e.epoch_abs_start) or (not finite(e.epoch_abs_stop)) or (cg == 'NO_SOLUTION') or (cp == 'NO_SOLUTION') or (ig is None) or (ip is None):
                base.update({'gsr_sample_fraction': 0.0, 'ppg_sample_fraction': 0.0, 'gsr_boundary_start_sec': math.nan, 'ppg_boundary_start_sec': math.nan, 'gsr_boundary_stop_sec': math.nan, 'ppg_boundary_stop_sec': math.nan, 'max_cross_modality_boundary_disagreement_sec': math.nan, 'joint_epoch_covered': False})
            else:
                s = float(e.epoch_abs_start)
                z = float(e.epoch_abs_stop)
                _, fg, gbs, gbe = candidate_sample_metrics(ig, cg, s, z)
                _, fp, pbs, pbe = candidate_sample_metrics(ip, cp, s, z)
                dis = max(abs(gbs - pbs), abs(gbe - pbe)) if all(map(math.isfinite, [gbs, gbe, pbs, pbe])) else math.inf
                ok = fg >= 0.95 and fp >= 0.95 and (dis <= 0.25)
                base.update({'gsr_sample_fraction': fg, 'ppg_sample_fraction': fp, 'gsr_boundary_start_sec': gbs, 'ppg_boundary_start_sec': pbs, 'gsr_boundary_stop_sec': gbe, 'ppg_boundary_stop_sec': pbe, 'max_cross_modality_boundary_disagreement_sec': dis, 'joint_epoch_covered': ok})
            epoch_rows.append(base)
            trial_status.setdefault(int(e.trial_number), {})[str(e.epoch_family)] = bool(base['joint_epoch_covered'])
        complete = candidate_all3 = 0
        for tr, st in sorted(trial_status.items()):
            eeg_all3 = all((bool(g[(g.trial_number == tr) & (g.epoch_family == fam)].epoch_available.iloc[0]) if len(g[(g.trial_number == tr) & (g.epoch_family == fam)]) else False for fam in ['stimulus_early', 'stimulus_late', 'rating_early']))
            joint = all((st.get(f, False) for f in ['stimulus_early', 'stimulus_late', 'rating_early']))
            candidate_all3 += int(eeg_all3)
            complete += int(joint)
            traj_rows.append({'subject_id': sub, 'trial_number': tr, 'eeg_all_three_epochs_available': eeg_all3, 'joint_gsr_ppg_all_three_epochs_complete': joint, 'gsr_candidate': cg, 'ppg_candidate': cp})
        feasible = cg != 'NO_SOLUTION' and cp != 'NO_SOLUTION' and (complete >= 16)
        participant_rows.append({'subject_id': sub, 'gsr_selected_candidate': cg, 'ppg_selected_candidate': cp, 'candidate_eeg_all3_trials': candidate_all3, 'complete_joint_dynamic_trajectories': complete, 'participant_min16_pass': complete >= 16, 'clock_solution_both_modalities': cg != 'NO_SOLUTION' and cp != 'NO_SOLUTION', 'participant_feasible': feasible})
    total_candidate = sum((r['candidate_eeg_all3_trials'] for r in participant_rows))
    total_complete = sum((r['complete_joint_dynamic_trajectories'] for r in participant_rows))
    feasible_n = sum((r['participant_feasible'] for r in participant_rows))
    frac = total_complete / total_candidate if total_candidate else 0.0
    gate_part = feasible_n >= 18
    gate_frac = frac >= 0.7
    passed = gate_part and gate_frac
    paths = {'addendum': outcfg, 'header': project / 'outputs/qc/raw_dynamic_physio_timebase_audit_eeg_absolute_header_anchor_audit.csv', 'stream': project / 'outputs/qc/raw_dynamic_physio_timebase_audit_raw_physio_timing_stream_audit.csv', 'candidates': project / 'outputs/tables/raw_dynamic_physio_timebase_audit_timebase_candidate_diagnostics.csv', 'participants': project / 'outputs/tables/raw_dynamic_physio_timebase_audit_participant_clock_selection.csv', 'epochs': project / 'outputs/tables/raw_dynamic_physio_timebase_audit_epoch_coverage_audit.csv', 'trials': project / 'outputs/tables/raw_dynamic_physio_timebase_audit_trial_dynamic_coverage.csv', 'roster': project / 'outputs/tables/raw_dynamic_physio_timebase_audit_prespecified_feasible_dynamic_roster.csv', 'summary': project / 'outputs/qc/raw_dynamic_physio_timebase_audit_timing_audit_summary.json', 'report': project / 'outputs/reports/raw_dynamic_physio_timebase_audit_raw_dynamic_physio_timebase_audit_report.md', 'next': project / 'analysis_specifications/S056_raw_dynamic_physio_timebase_audit_preprocessing.json', 'contract': project / 'config/raw_dynamic_physio_timebase_audit_raw_dynamic_physio_timebase_audit_contract.json', 'log': project / 'outputs/logs/raw_dynamic_physio_timebase_audit_raw_dynamic_physio_timebase_audit.log'}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    writecsv(paths['header'], header)
    writecsv(paths['stream'], timing_stream_rows)
    writecsv(paths['candidates'], file_diags)
    writecsv(paths['participants'], participant_rows)
    writecsv(paths['epochs'], epoch_rows)
    writecsv(paths['trials'], traj_rows)
    writecsv(paths['roster'], [r for r in participant_rows if r['participant_feasible']])
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'timing_audit_passed' if passed else 'timing_audit_failed_stop_raw_dynamic_branch', 'counts': {'raw_both_inventory_participants': len(rawsubs), 'raw_both_with_raw_eeg_epoch_folds_EEG_participants': len(subjects), 'participants_with_feasible_joint_clock_and_min16': feasible_n, 'candidate_EEG_all3_trials': total_candidate, 'complete_joint_dynamic_trajectories': total_complete, 'complete_fraction': frac}, 'gates': {'minimum_participants_required': 18, 'participant_gate_pass': gate_part, 'minimum_complete_fraction_required': 0.7, 'coverage_gate_pass': gate_frac, 'primary_feasibility_gate_pass': passed}, 'boundaries': {'GSR_amplitude_read': False, 'PPG_amplitude_read': False, 'EEG_sample_amplitude_read': False, 'EDF_header_only_read': True, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False}, 'raw_dynamic_physio_preprocessing_specification_permitted': passed}
    writej(paths['summary'], summary)
    nextpol = {'phase': PHASE, 'generated_at': now(), 'next_phase': 'raw_source_lineage_dynamic_physio_preprocessing_specification', 'next_phase_permitted': passed, 'role': 'specification raw GSR/PPG amplitude preprocessing, artifact/QC, resampling and dynamic representation before amplitude access.' if passed else 'STOP: joint raw-dynamic physiology alignment feasibility gate failed.', 'forbidden': ['Validation/Confirmation raw physiology', 'outcome-driven window changes', 'single-modality rescue after failed joint gate', 'semantic relabeling of EEG families']}
    writej(paths['next'], nextpol)
    report = f"# raw_dynamic_physio_timebase_audit — Discovery raw physiology timebase / epoch coverage audit\n\nStatus: **{('PASS' if passed else 'STOP')}**.\n\nTiming metadata/columns only; no GSR, PPG or EEG sample amplitude was loaded. A pre-timing safety addendum required absolute EEG header anchors.\n\n- Raw-both inventory participants: **{len(rawsubs)}**\n- Raw-both with prespecified raw_eeg_epoch_folds EEG: **{len(subjects)}**\n- Feasible participants with >=16 complete trajectories: **{feasible_n}**\n- Candidate EEG all-three-epoch trials: **{total_candidate}**\n- Complete joint GSR+PPG trajectories: **{total_complete}** ({frac:.4f})\n- >=18 participant gate: **{('PASS' if gate_part else 'FAIL')}**\n- >=70% coverage gate: **{('PASS' if gate_frac else 'FAIL')}**\n- Primary gate: **{('PASS' if passed else 'FAIL')}**\n\nraw_dynamic_physio_preprocessing_specification is {('permitted' if passed else 'not permitted')}. Validation and Confirmation raw physiology remain unopened."
    paths['report'].write_text(report, encoding='utf-8')
    paths['log'].write_text(f'[{now()}] raw_dynamic_physio_timebase_audit completed. gate={passed}; amplitude_read=False', encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'raw_dynamic_physio_alignment_design_lock_sha256': sha256(project / A_LOCK), 'absolute_anchor_addendum_sha256': add_sha, 'inputs': {'raw_eeg_epoch_folds_timing_sha256': sha256(project / raw_eeg_epoch_folds_TIMING), 'raw_eeg_epoch_folds_epochs_sha256': sha256(project / raw_eeg_epoch_folds_EPOCHS), 'raw_dynamic_physio_alignment_design_roster_sha256': sha256(project / A_ROSTER)}, 'outputs': {}, 'primary_feasibility_gate_pass': passed, 'raw_signal_amplitude_read': False}
    lockhash = {}
    for p in paths.values():
        if p.is_file():
            lockhash[str(p.relative_to(project))] = sha256(p)
    contract['outputs'] = lockhash
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/raw_dynamic_physio_timebase_audit_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': lockhash, 'raw_dynamic_physio_preprocessing_specification_permitted': passed, 'raw_signal_amplitude_read': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'raw_dynamic_physio_timebase_audit_raw_dynamic_physio_timebase_audit_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print(f"[{now()}] raw_dynamic_physio_timebase_audit {('PASS' if passed else 'STOP')}. feasible participants={feasible_n}; complete={total_complete}/{total_candidate} ({frac:.4f})", flush=True)
    print('', hz, flush=True)
    print('SHA-256:', sh, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())