# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, re, warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PHASE = 'exact_physio_eeg_sync_recovery'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
TZ8 = timezone(timedelta(hours=8))
SNAPSHOT_SHA = 'f3b086adfe0fbe7066ce578a2189985818ef1106a10b9f5eac9dc0998bc0b320'
E_LOCK = 'outputs/qc/raw_dynamic_physio_sync_correction_final_output_lock_hashes.json'
E_SUM = 'outputs/qc/raw_dynamic_physio_sync_correction_sync_correction_summary.json'
E_POLICY = 'analysis_specifications/S054_raw_dynamic_physio_sync_correction_exact.json'
E_RUNS = 'outputs/tables/raw_dynamic_physio_sync_correction_readme_anchored_run_spans.csv'
A_ROSTER = 'outputs/tables/raw_dynamic_physio_alignment_design_discovery_raw_physio_source_roster.csv'
A_EPOCH = 'outputs/tables/raw_eeg_epoch_folds_prespecified_epoch_manifest.csv'
A_TIMING = 'outputs/tables/raw_eeg_epoch_folds_authoritative_trial_timing_manifest.csv'
A_HEADER = 'outputs/qc/raw_eeg_epoch_folds_raw_edf_header_inventory.csv'
A_CONTRACT = 'config/raw_eeg_epoch_folds_raw_eeg_epoch_folds_contract.json'
B_STREAM = 'outputs/qc/raw_dynamic_physio_timebase_audit_raw_physio_timing_stream_audit.csv'
EPOCHS = ['stimulus_early', 'stimulus_late', 'rating_early']
MIN_PARTICIPANTS = 18
MIN_COMPLETE_FRAC = 0.7
MIN_TRAJ_PER_SUB = 16
TARGET_SYNC_UNCERTAINTY_SEC = 1.0
PHYSIO_MIN_SAMPLE_FRAC = 0.95
FORBIDDEN_NAME_RE = re.compile('(valence|liking|arousal|joy|fear|sad|disgust|tender|inspir|emotion|family|report)', re.I)
TIMING_NAME_RE = re.compile('(time|timestamp|date|acq|record|start|stop|end|onset|offset|sample|index|segment|trial)', re.I)
ABS_START_KEY_RE = re.compile('(acq.*time|acquisition.*time|record.*start|start.*time|measurement.*time|meas.*date|orig.*time)', re.I)
ABS_END_KEY_RE = re.compile('(record.*end|end.*time|stop.*time)', re.I)

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

def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def verify_lock(project: Path, rel: str, phase: str):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'Missing upstream lock: {rel}')
    lock = readj(p)
    if lock.get('phase') != phase:
        raise RuntimeError(f"Wrong upstream phase in {rel}: {lock.get('phase')}")
    for r, h in lock.get('hashes', {}).items():
        q = project / r
        if not q.is_file():
            raise RuntimeError(f'Missing fixed upstream artifact: {r}')
        g = sha256(q)
        if g != h:
            raise RuntimeError(f'Upstream hash mismatch {r}: {g} != {h}')
    return lock

def verify_inputs(project: Path):
    e = verify_lock(project, E_LOCK, 'raw_dynamic_physio_sync_correction')
    if not e.get('exact_physio_eeg_sync_recovery_permitted'):
        raise RuntimeError('raw_dynamic_physio_sync_correction did not authorize exact_physio_eeg_sync_recovery')
    if e.get('raw_signal_amplitude_read') is not False:
        raise RuntimeError('raw_dynamic_physio_sync_correction amplitude boundary mismatch')
    es = readj(project / E_SUM)
    ep = readj(project / E_POLICY)
    if not es.get('exact_physio_eeg_sync_recovery_permitted') or ep.get('next_phase') != 'exact_physio_eeg_sync_recovery' or (not ep.get('next_phase_permitted')):
        raise RuntimeError('raw_dynamic_physio_sync_correction synchronization policy mismatch')
    c = readj(project / A_CONTRACT)
    if c.get('phase') != 'raw_eeg_epoch_folds' or c.get('eeg_amplitude_read') is not False:
        raise RuntimeError('raw_eeg_epoch_folds contract boundary mismatch')
    oh = c.get('output_hashes', {})
    for rel, key in [(A_EPOCH, 'epoch_manifest'), (A_HEADER, 'raw_headers')]:
        q = project / rel
        exp = oh.get(key)
        if not q.is_file() or not exp or sha256(q) != exp:
            raise RuntimeError(f'raw_eeg_epoch_folds fixed input mismatch: {rel}')
    for rel in [A_ROSTER, A_TIMING, B_STREAM, E_RUNS]:
        if not (project / rel).is_file():
            raise RuntimeError(f'Missing required input: {rel}')
    return (e, es, ep, c)

def parse_abs_timestamp(v: Any):
    """Return (epoch_sec, precision_sec, parse_kind) or (None,None,reason)."""
    if v is None:
        return (None, None, 'none')
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if math.isfinite(x) and 100000000.0 < x < 4000000000.0:
            return (x, 1.0, 'unix_numeric')
        return (None, None, 'numeric_not_absolute_epoch')
    s = str(v).strip().strip('"').strip("'")
    if not s or s.lower() in {'n/a', 'na', 'nan', 'none'}:
        return (None, None, 'empty')
    try:
        x = float(s)
        if math.isfinite(x) and 100000000.0 < x < 4000000000.0:
            return (x, 1.0, 'unix_string')
    except Exception:
        pass
    if not (re.search('\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}', s) and re.search('\\d{1,2}:\\d{2}', s)):
        return (None, None, 'not_absolute_datetime')
    try:
        ts = pd.to_datetime(s, errors='raise')
        if getattr(ts, 'tzinfo', None) is None:
            ts = ts.tz_localize('Asia/Shanghai')
        precision = 1.0 if re.search(':\\d{2}:\\d{2}', s) else 60.0
        return (float(ts.timestamp()), precision, 'datetime')
    except Exception:
        return (None, None, 'datetime_parse_failed')

def read_edf_header_text(path: Path):
    row = {'raw_source_path': str(path), 'header_readable': False, 'edf_start_epoch_from_standard_header': math.nan, 'edf_start_precision_sec': math.nan, 'edf_patient_id': '', 'edf_recording_id': '', 'edf_reserved': '', 'error': ''}
    try:
        with path.open('rb') as f:
            b = f.read(256)
        if len(b) < 256:
            raise RuntimeError('EDF header shorter than 256 bytes')
        dec = lambda a, b_: b[a:b_].decode('latin1', 'replace').strip()
        row['edf_patient_id'] = dec(8, 88)
        row['edf_recording_id'] = dec(88, 168)
        date = dec(168, 176)
        tm = dec(176, 184)
        row['edf_reserved'] = dec(192, 236)
        row['header_readable'] = True
        m = re.match('^(\\d{2})\\.(\\d{2})\\.(\\d{2})$', date)
        t = re.match('^(\\d{2})\\.(\\d{2})\\.(\\d{2})$', tm)
        if m and t:
            yy = int(m.group(3))
            year = 2000 + yy if yy < 85 else 1900 + yy
            dt = datetime(year, int(m.group(2)), int(m.group(1)), int(t.group(1)), int(t.group(2)), int(t.group(3)), tzinfo=TZ8)
            row['edf_start_epoch_from_standard_header'] = dt.timestamp()
            row['edf_start_precision_sec'] = 1.0
    except Exception as ex:
        row['error'] = f'{type(ex).__name__}: {ex}'
    return row

def recursive_timing_values(obj: Any, prefix=''):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.extend(recursive_timing_values(v, p))
            elif TIMING_NAME_RE.search(str(k)) and (not FORBIDDEN_NAME_RE.search(str(k))):
                out.append((p, v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(recursive_timing_values(v, f'{prefix}[{i}]'))
    return out

def read_small_timing_text_candidates(path: Path, sid: str, run_hint: int | None, source_kind: str):
    rows = []
    try:
        if path.suffix.lower() == '.json':
            obj = readj(path)
            for key, v in recursive_timing_values(obj):
                x, prec, kind = parse_abs_timestamp(v)
                rows.append({'subject_id': sid, 'run_hint': run_hint, 'source_kind': source_kind, 'source_path': str(path), 'field': key, 'raw_value': str(v)[:200], 'parsed_epoch_sec': x if x is not None else math.nan, 'precision_sec': prec if prec is not None else math.nan, 'parse_kind': kind, 'candidate_role': 'start' if ABS_START_KEY_RE.search(key) else 'end' if ABS_END_KEY_RE.search(key) else 'unspecified', 'forbidden_semantic_used': False, 'raw_signal_amplitude_read': False})
        elif path.suffix.lower() == '.tsv':
            hdr = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
            cols = [c for c in hdr if TIMING_NAME_RE.search(c) and (not FORBIDDEN_NAME_RE.search(c))]
            if cols:
                d = pd.read_csv(path, sep='\t', usecols=cols, nrows=500)
                for c in cols:
                    vals = d[c].dropna().astype(str).unique()[:20]
                    for v in vals:
                        x, prec, kind = parse_abs_timestamp(v)
                        rows.append({'subject_id': sid, 'run_hint': run_hint, 'source_kind': source_kind, 'source_path': str(path), 'field': c, 'raw_value': str(v)[:200], 'parsed_epoch_sec': x if x is not None else math.nan, 'precision_sec': prec if prec is not None else math.nan, 'parse_kind': kind, 'candidate_role': 'start' if ABS_START_KEY_RE.search(c) else 'end' if ABS_END_KEY_RE.search(c) else 'unspecified', 'forbidden_semantic_used': False, 'raw_signal_amplitude_read': False})
    except Exception as ex:
        rows.append({'subject_id': sid, 'run_hint': run_hint, 'source_kind': source_kind, 'source_path': str(path), 'field': '<read_error>', 'raw_value': f'{type(ex).__name__}: {ex}', 'parsed_epoch_sec': math.nan, 'precision_sec': math.nan, 'parse_kind': 'read_error', 'candidate_role': 'unspecified', 'forbidden_semantic_used': False, 'raw_signal_amplitude_read': False})
    return rows

def parse_stream_time_token(raw: str, json_start: float):
    """Parse one timing token without consulting any signal amplitude.

    Returns (epoch_seconds, parse_kind).  The ds005540 raw csv_time_* columns are
    strings in at least some released files, so float-only parsing is invalid.
    Naive datetimes/time-of-day values are reconciled to the sidecar StartTime
    using the acquisition timezone (+08:00) and never using outcomes or amplitudes.
    """
    s = str(raw).strip().strip('"').strip("'")
    if not s or s.lower() in {'n/a', 'na', 'nan', 'none', 'null'}:
        return (None, 'empty')
    try:
        x = float(s)
        if math.isfinite(x):
            if 100000000000.0 < abs(x) < 4000000000000.0:
                return (x / 1000.0, 'unix_milliseconds')
            if 100000000.0 < x < 4000000000.0:
                return (x, 'unix_seconds')
            if abs(x) < 10000000.0:
                return (float(json_start) + x, 'relative_seconds_from_json_start')
    except Exception:
        pass
    m = re.fullmatch('\\s*(\\d{1,2}):(\\d{2})(?::(\\d{2})(?:[.,](\\d+))?)?\\s*', s)
    if m:
        hh, mm = (int(m.group(1)), int(m.group(2)))
        ss = int(m.group(3) or 0)
        frac = float('0.' + (m.group(4) or '0'))
        if hh < 24 and mm < 60 and (ss < 60):
            base = datetime.fromtimestamp(float(json_start), TZ8)
            cand = datetime(base.year, base.month, base.day, hh, mm, ss, tzinfo=TZ8).timestamp() + frac
            opts = [cand - 86400.0, cand, cand + 86400.0]
            return (min(opts, key=lambda z: abs(z - float(json_start))), 'local_time_of_day')
    py = None
    try:
        py = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        for fmt in ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try:
                py = datetime.strptime(s, fmt)
                break
            except Exception:
                pass
    if py is None:
        try:
            ts = pd.to_datetime(s, errors='raise')
            if isinstance(ts, pd.DatetimeIndex):
                ts = ts[0]
            py = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        except Exception:
            return (None, 'unparsed')
    if getattr(py, 'tzinfo', None) is not None:
        return (float(py.timestamp()), 'datetime_tzaware')
    c_local = py.replace(tzinfo=TZ8).timestamp()
    c_utc = py.replace(tzinfo=timezone.utc).timestamp()
    return (min((c_local, c_utc), key=lambda z: abs(z - float(json_start))), 'datetime_naive_reconciled')

def stream_second_labels(tsv: Path, json_path: Path, expected_col: str):
    meta = readj(json_path)
    cols = meta.get('Columns', [])
    if isinstance(cols, str):
        cols = [x.strip() for x in cols.split('|')]
    if expected_col not in cols:
        raise RuntimeError(f'{json_path}: Columns lacks {expected_col}')
    idx = cols.index(expected_col)
    fs = float(meta.get('SamplingFrequency'))
    start = float(meta.get('StartTime'))
    last_label = None
    group_count = 0
    groups = []
    nrows = 0
    malformed = 0
    negative = 0
    parse_kinds = Counter()
    rejected_examples = []
    cache = {}

    def flush(label, startrow, count):
        if label is not None:
            groups.append((float(label), int(startrow), int(count)))
    with gzip.open(tsv, 'rt', encoding='utf-8', errors='replace', newline='') as f:
        row_start = 0
        for li, line in enumerate(f):
            toks = line.rstrip('').split('\t')
            if li == 0 and idx < len(toks) and (toks[idx].strip().strip('"') == expected_col):
                continue
            nrows += 1
            if idx >= len(toks):
                malformed += 1
                continue
            tok = toks[idx].strip().strip('"').strip("'")
            if tok in cache:
                lab, kind = cache[tok]
            else:
                lab, kind = parse_stream_time_token(tok, start)
                cache[tok] = (lab, kind)
            parse_kinds[kind] += 1
            if lab is None or not math.isfinite(float(lab)):
                malformed += 1
                if len(rejected_examples) < 5 and tok not in rejected_examples:
                    rejected_examples.append(tok[:120])
                continue
            lab = float(lab)
            if last_label is None:
                row_start = nrows - 1
                last_label = lab
                group_count = 1
                continue
            if lab < last_label - 1e-09:
                negative += 1
            if abs(lab - last_label) <= 1e-09:
                group_count += 1
            else:
                flush(last_label, row_start, group_count)
                row_start = nrows - 1
                group_count = 1
                last_label = lab
    flush(last_label, row_start, group_count)
    if not groups:
        raise RuntimeError(f'{tsv}: no timing labels parsed; rejected timing examples={rejected_examples!r}; parse_kinds={dict(parse_kinds)}')
    raw_labels = np.array([g[0] for g in groups], float)
    raw_counts = np.array([g[2] for g in groups], int)
    uniq_d = np.diff(raw_labels)
    pos_d = uniq_d[uniq_d > 1e-09]
    med_delta = float(np.median(pos_d)) if len(pos_d) else math.nan
    med_repeat = float(np.median(raw_counts)) if len(raw_counts) else math.nan
    second_label_mode = bool(finite(med_repeat) and med_repeat >= max(2.0, 0.25 * fs) or (finite(med_delta) and 0.5 <= med_delta <= 1.5))
    bins = defaultdict(int)
    if second_label_mode:
        for lab, cnt in zip(raw_labels, raw_counts):
            sec = int(round(lab)) if abs(lab - round(lab)) <= 0.25 else int(math.floor(lab))
            bins[sec] += int(cnt)
        clock_mode = 'repeated_second_labels'
    else:
        for lab, cnt in zip(raw_labels, raw_counts):
            bins[int(math.floor(lab))] += int(cnt)
        clock_mode = 'per_sample_or_subsecond_timestamps'
    labels = np.array(sorted(bins), float)
    counts = np.array([bins[int(x)] for x in labels], int)
    if not len(labels):
        raise RuntimeError(f'{tsv}: timing labels parsed but no clock-second bins constructed')
    deltas = np.diff(labels)
    gaps = []
    for i, d in enumerate(deltas):
        if d > 1.5:
            gaps.append({'gap_before_epoch_sec': labels[i], 'gap_after_epoch_sec': labels[i + 1], 'gap_delta_sec': float(d), 'missing_clock_sec': float(max(0, d - 1))})
    return {'fs': fs, 'json_start': start, 'nrows': nrows, 'malformed': malformed, 'negative': negative, 'timezone_shift_sec': 0.0, 'clock_mode': clock_mode, 'parse_kinds': json.dumps(dict(parse_kinds), sort_keys=True), 'rejected_timing_examples': json.dumps(rejected_examples), 'raw_unique_timing_groups': len(groups), 'median_raw_group_repeat_count': med_repeat, 'median_positive_raw_timestamp_delta_sec': med_delta, 'first_second': float(labels[0]), 'last_second': float(labels[-1]), 'second_groups': len(labels), 'median_samples_per_second': float(np.median(counts)), 'p05_samples_per_second': float(np.quantile(counts, 0.05)), 'p95_samples_per_second': float(np.quantile(counts, 0.95)), 'fraction_second_groups_ge_95pct_fs': float(np.mean(counts >= 0.95 * fs)), 'gaps': gaps, 'labels': labels, 'counts': counts}

def minute_end_epoch(date_s: str, time_s: str):
    d = str(date_s).replace('-', '/') if re.match('^\\d{4}/\\d{1,2}-\\d{1,2}$', str(date_s)) else str(date_s)
    return datetime.strptime(f'{d} {time_s}', '%Y/%m/%d %H:%M').replace(tzinfo=TZ8).timestamp()

def circular_sec_diff(a, b):
    return (float(a) - float(b) + 43200) % 86400 - 43200

def source_to_run_map(runs: pd.DataFrame):
    return {(str(r.raw_source_path), str(r.subject_id), int(r.raw_run)): r for _, r in runs.iterrows()}

def build_anchor_candidates(dataset: Path, runs: pd.DataFrame, meta_rows: list[dict], edf_rows: list[dict], gap_support: list[dict]):
    out = []
    for m in meta_rows:
        if not finite(m.get('parsed_epoch_sec')) or not finite(m.get('precision_sec')):
            continue
        if float(m['precision_sec']) > TARGET_SYNC_UNCERTAINTY_SEC:
            continue
        sid = m['subject_id']
        hint = m.get('run_hint')
        role = m.get('candidate_role')
        target = runs[runs.subject_id.astype(str).eq(str(sid))]
        if hint is not None and finite(hint):
            target = target[target.raw_run.astype(int).eq(int(hint))]
        for _, r in target.iterrows():
            x = float(m['parsed_epoch_sec'])
            start = x if role == 'start' else x - float(r.recording_duration_sec) if role == 'end' else math.nan
            if not finite(start):
                continue
            end = start + float(r.recording_duration_sec)
            readme_end = minute_end_epoch(str(r.readme_date), str(r.readme_endtime))
            consistent = abs(end - readme_end) <= 60.0
            out.append({'subject_id': sid, 'raw_run': int(r.raw_run), 'raw_source_path': str(r.raw_source_path), 'anchor_source': f"metadata:{m['source_kind']}:{m['field']}", 'candidate_start_epoch_sec': start, 'candidate_uncertainty_sec': float(m['precision_sec']), 'readme_end_residual_sec': end - readme_end, 'readme_coarse_consistent': consistent, 'authoritative_candidate': bool(consistent), 'support_detail': str(m['source_path']), 'raw_signal_amplitude_read': False})
    for e in edf_rows:
        if not finite(e.get('edf_start_epoch_from_standard_header')):
            continue
        rr = runs[runs.raw_source_path.astype(str).eq(str(e['raw_source_path']))]
        if len(rr) != 1:
            continue
        r = rr.iloc[0]
        start = float(e['edf_start_epoch_from_standard_header'])
        end = start + float(r.recording_duration_sec)
        re_ = minute_end_epoch(str(r.readme_date), str(r.readme_endtime))
        consistent = abs(end - re_) <= 60.0
        out.append({'subject_id': str(r.subject_id), 'raw_run': int(r.raw_run), 'raw_source_path': str(r.raw_source_path), 'anchor_source': 'edf_standard_header_startdate_starttime', 'candidate_start_epoch_sec': start, 'candidate_uncertainty_sec': 1.0, 'readme_end_residual_sec': end - re_, 'readme_coarse_consistent': consistent, 'authoritative_candidate': bool(consistent), 'support_detail': f"patient={e.get('edf_patient_id', '')}; recording={e.get('edf_recording_id', '')}", 'raw_signal_amplitude_read': False})
    for g in gap_support:
        out.append({'subject_id': g['subject_id'], 'raw_run': g['raw_run'], 'raw_source_path': g['raw_source_path'], 'anchor_source': 'dual_modality_physio_gap_support_only', 'candidate_start_epoch_sec': math.nan, 'candidate_uncertainty_sec': math.nan, 'readme_end_residual_sec': g.get('readme_boundary_residual_sec', math.nan), 'readme_coarse_consistent': g.get('coarse_match', False), 'authoritative_candidate': False, 'support_detail': f"gap_before={g.get('gap_before_epoch_sec')}; gap_after={g.get('gap_after_epoch_sec')}; requires independent EEG absolute anchor", 'raw_signal_amplitude_read': False})
    return out

def select_anchors(runs: pd.DataFrame, cands: pd.DataFrame):
    rows = []
    for _, r in runs.sort_values(['subject_id', 'raw_run']).iterrows():
        cc = cands[(cands.subject_id.astype(str) == str(r.subject_id)) & (pd.to_numeric(cands.raw_run, errors='coerce') == int(r.raw_run)) & (cands.authoritative_candidate == True)].copy() if len(cands) else pd.DataFrame()
        status = 'no_exact_anchor'
        selected = None
        detail = ''
        if len(cc):
            vals = cc.candidate_start_epoch_sec.astype(float).to_numpy()
            spread = float(np.max(vals) - np.min(vals))
            if spread <= TARGET_SYNC_UNCERTAINTY_SEC:
                cc['priority'] = cc.anchor_source.map(lambda x: 0 if str(x).startswith('metadata:') else 1)
                selected = cc.sort_values(['priority', 'candidate_uncertainty_sec']).iloc[0]
                status = 'exact_anchor_selected'
                detail = f'{len(cc)} exact candidates; spread={spread:.3f}s'
            else:
                status = 'conflicting_exact_candidates'
                detail = f'{len(cc)} candidates; spread={spread:.3f}s'
        rows.append({'subject_id': str(r.subject_id), 'raw_run': int(r.raw_run), 'raw_source_path': str(r.raw_source_path), 'selection_status': status, 'selected_anchor_source': selected.anchor_source if selected is not None else '', 'selected_start_epoch_sec': float(selected.candidate_start_epoch_sec) if selected is not None else math.nan, 'selected_uncertainty_sec': float(selected.candidate_uncertainty_sec) if selected is not None else math.nan, 'selection_detail': detail, 'raw_signal_amplitude_read': False})
    return rows

def second_group_window_fraction(info: dict, start: float, stop: float):
    labels = info['labels']
    counts = info['counts']
    fs = float(info['fs'])
    if stop <= start:
        return 0.0
    sec0 = math.floor(start)
    sec1 = math.ceil(stop)
    needed = list(range(sec0, sec1))
    if not needed:
        return 0.0
    mp = {int(round(l)): int(c) for l, c in zip(labels, counts) if abs(l - round(l)) <= 0.01}
    expected = 0.0
    present = 0.0
    for sec in needed:
        overlap = max(0.0, min(stop, sec + 1) - max(start, sec))
        expected += overlap * fs
        c = mp.get(sec, 0)
        usable = min(float(c), fs)
        present += overlap * usable
    return present / expected if expected > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        assert parse_abs_timestamp('2024-07-12 15:24:33')[1] == 1.0
        assert parse_abs_timestamp('2024-07-12 15:24')[1] == 60.0
        assert parse_abs_timestamp('15:24:33')[0] is None
        js = 1692513751.0
        assert abs(parse_stream_time_token('2023-08-20 14:42:31', js)[0] - js) < 1e-06
        assert abs(parse_stream_time_token('2023/08/20 14:42:31', js)[0] - js) < 1e-06
        assert abs(parse_stream_time_token('14:42:31', js)[0] - js) < 1e-06
        assert abs(parse_stream_time_token('1692513751', js)[0] - js) < 1e-06
        assert abs(circular_sec_diff(10, 86390) - 20) < 1e-09
        print('exact_physio_eeg_sync_recovery self-test: PASS')
        return 0
    project = a.project
    dataset = a.dataset
    if sha256(a.snapshot) != SNAPSHOT_SHA:
        raise RuntimeError('prespecified README endtime snapshot SHA mismatch')
    e, es, ep, raw_eeg_epoch_folds = verify_inputs(project)
    runs = pd.read_csv(project / E_RUNS, low_memory=False)
    roster = pd.read_csv(project / A_ROSTER, low_memory=False)
    epochs = pd.read_csv(project / A_EPOCH, low_memory=False)
    bstream = pd.read_csv(project / B_STREAM, low_memory=False)
    relevant = set(runs.subject_id.astype(str))
    stream_info = {}
    stream_rows = []
    gap_rows = []
    rr = roster[roster.subject_id.astype(str).isin(relevant) & roster.both_raw_modalities_exist.map(boolish)].copy()
    for _, r in rr.sort_values('subject_id').iterrows():
        sid = str(r.subject_id)
        for mod, tc in [('GSR', 'csv_time_GSR'), ('PPG', 'csv_time_PPG')]:
            tsv = Path(r.gsr_tsv_path if mod == 'GSR' else r.ppg_tsv_path)
            js = Path(r.gsr_json_path if mod == 'GSR' else r.ppg_json_path)
            if not tsv.is_file() or not js.is_file():
                continue
            info = stream_second_labels(tsv, js, tc)
            stream_info[sid, mod] = info
            stream_rows.append({'subject_id': sid, 'modality': mod, 'tsv_path': str(tsv), 'json_path': str(js), 'sampling_frequency_hz': info['fs'], 'json_start_time_sec': info['json_start'], 'first_second_label_sec': info['first_second'], 'last_second_label_sec': info['last_second'], 'timezone_shift_sec': info['timezone_shift_sec'], 'data_rows': info['nrows'], 'second_groups': info['second_groups'], 'clock_mode': info.get('clock_mode', ''), 'parse_kinds': info.get('parse_kinds', ''), 'raw_unique_timing_groups': info.get('raw_unique_timing_groups', math.nan), 'median_raw_group_repeat_count': info.get('median_raw_group_repeat_count', math.nan), 'median_positive_raw_timestamp_delta_sec': info.get('median_positive_raw_timestamp_delta_sec', math.nan), 'median_samples_per_second': info['median_samples_per_second'], 'p05_samples_per_second': info['p05_samples_per_second'], 'p95_samples_per_second': info['p95_samples_per_second'], 'fraction_second_groups_ge_95pct_fs': info['fraction_second_groups_ge_95pct_fs'], 'negative_time_transitions': info['negative'], 'malformed_time_rows': info['malformed'], 'large_gap_count': len(info['gaps']), 'clock_reconstruction': 'text/numeric timing token -> epoch; then repeated-second or per-sample clock bins', 'raw_signal_amplitude_read': False})
            for g in info['gaps']:
                gap_rows.append({'subject_id': sid, 'modality': mod, **g, 'raw_signal_amplitude_read': False})
    gap_support = []
    for sid in sorted(relevant):
        gi = stream_info.get((sid, 'GSR'))
        pi = stream_info.get((sid, 'PPG'))
        if not gi or not pi:
            continue
        for run in [1, 2]:
            subrun = runs[(runs.subject_id.astype(str) == sid) & (runs.raw_run.astype(int) == run)]
            if len(subrun) != 1:
                continue
            sr = subrun.iloc[0]
            if len(runs[runs.subject_id.astype(str) == sid]) < 2:
                continue
            target = float(sr.estimated_run_end_epoch_sec if run == 1 else sr.estimated_run_start_epoch_sec)
            candidates = []
            for gg in gi['gaps']:
                for pg in pi['gaps']:
                    if abs(gg['gap_before_epoch_sec'] - pg['gap_before_epoch_sec']) <= 1.0 and abs(gg['gap_after_epoch_sec'] - pg['gap_after_epoch_sec']) <= 1.0:
                        boundary = gg['gap_before_epoch_sec'] if run == 1 else gg['gap_after_epoch_sec']
                        if abs(boundary - target) <= 60.0:
                            candidates.append((gg, pg, boundary))
            if len(candidates) == 1:
                gg, pg, boundary = candidates[0]
                gap_support.append({'subject_id': sid, 'raw_run': run, 'raw_source_path': str(sr.raw_source_path), 'gap_before_epoch_sec': gg['gap_before_epoch_sec'], 'gap_after_epoch_sec': gg['gap_after_epoch_sec'], 'readme_boundary_residual_sec': boundary - target, 'coarse_match': True})
    meta_rows = []
    edf_rows = []
    source_inventory = []
    for _, r in runs.sort_values(['subject_id', 'raw_run']).iterrows():
        sid = str(r.subject_id)
        run = int(r.raw_run)
        edf = Path(str(r.raw_source_path))
        eegdir = edf.parent
        er = read_edf_header_text(edf)
        er.update({'subject_id': sid, 'raw_run': run})
        edf_rows.append(er)
        source_inventory.append({'subject_id': sid, 'raw_run': run, 'source_kind': 'edf_standard_header', 'source_path': str(edf), 'exists': edf.is_file(), 'raw_signal_amplitude_read': False})
        stem = edf.name
        for suff in ['_run-01_eeg.edf', '_run-02_eeg.edf', '_eeg.edf']:
            if stem.endswith(suff):
                js = eegdir / (stem[:-4] + '.json')
                if js.is_file():
                    meta_rows += read_small_timing_text_candidates(js, sid, run, 'eeg_json')
                    source_inventory.append({'subject_id': sid, 'raw_run': run, 'source_kind': 'eeg_json', 'source_path': str(js), 'exists': True, 'raw_signal_amplitude_read': False})
                break
        subjdir = dataset / sid
        if subjdir.is_dir():
            for p in sorted(subjdir.rglob('*.tsv')):
                name = p.name.lower()
                if ('events' in name or 'scans' in name or 'timing' in name) and 'beh' not in str(p.parent).lower():
                    meta_rows += read_small_timing_text_candidates(p, sid, run, 'timing_tsv')
                    source_inventory.append({'subject_id': sid, 'raw_run': run, 'source_kind': 'timing_tsv', 'source_path': str(p), 'exists': True, 'raw_signal_amplitude_read': False})
            for p in sorted((subjdir / 'eeg').glob('*.json')) if (subjdir / 'eeg').is_dir() else []:
                if 'physio' in p.name.lower() or p.name.endswith('_eeg.json'):
                    continue
                if p.stat().st_size <= 2000000:
                    meta_rows += read_small_timing_text_candidates(p, sid, run, 'other_eeg_json')
                    source_inventory.append({'subject_id': sid, 'raw_run': run, 'source_kind': 'other_eeg_json', 'source_path': str(p), 'exists': True, 'raw_signal_amplitude_read': False})
    deriv_rows = []
    for sid in sorted(relevant):
        for ctx, ses in [('imagery', 'ses-ima'), ('video', 'ses-vid')]:
            p = dataset / 'derivatives' / sid / ses / 'physio' / f'{sid}_{ses}_task-emotion_GSRfeatures.csv'
            if not p.is_file():
                hits = list(dataset.rglob(f'{sid}_{ses}_task-emotion_GSRfeatures.csv'))
                p = hits[0] if hits else p
            if not p.is_file():
                deriv_rows.append({'subject_id': sid, 'context': ctx, 'source_path': '', 'found': False, 'header_columns': '', 'timing_metadata_columns': '', 'absolute_timestamp_values_found': 0, 'raw_signal_amplitude_read': False})
                continue
            hdr = pd.read_csv(p, nrows=0).columns.tolist()
            timing_cols = [x for x in hdr if TIMING_NAME_RE.search(str(x)) and (not FORBIDDEN_NAME_RE.search(str(x))) and (str(x) not in {'SCR', 'SCL', 'iSCR'})]
            absn = 0
            samples = []
            if timing_cols:
                d = pd.read_csv(p, usecols=timing_cols, nrows=300)
                for cc in timing_cols:
                    for v in d[cc].dropna().astype(str).unique()[:20]:
                        x, prec, kind = parse_abs_timestamp(v)
                        if x is not None and prec is not None and (prec <= 1.0):
                            absn += 1
                            samples.append(f'{cc}={v}')
            deriv_rows.append({'subject_id': sid, 'context': ctx, 'source_path': str(p), 'found': True, 'header_columns': '|'.join(map(str, hdr)), 'timing_metadata_columns': '|'.join(map(str, timing_cols)), 'absolute_timestamp_values_found': absn, 'absolute_timestamp_examples': '|'.join(samples[:5]), 'raw_signal_amplitude_read': False})
    cand_rows = build_anchor_candidates(dataset, runs, meta_rows, edf_rows, gap_support)
    cand_df = pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame(columns=['subject_id', 'raw_run', 'raw_source_path', 'anchor_source', 'candidate_start_epoch_sec', 'candidate_uncertainty_sec', 'readme_end_residual_sec', 'readme_coarse_consistent', 'authoritative_candidate', 'support_detail', 'raw_signal_amplitude_read'])
    sel_rows = select_anchors(runs, cand_df)
    sel = pd.DataFrame(sel_rows)
    anchor_map = {(r.subject_id, int(r.raw_run)): float(r.selected_start_epoch_sec) for _, r in sel.iterrows() if r.selection_status == 'exact_anchor_selected'}
    ep = epochs[epochs.subject_id.astype(str).isin(relevant)].copy()
    ep['epoch_available'] = ep.epoch_available.map(boolish)
    epoch_rows = []
    trial_rows = []
    part_rows = []
    bytrial = defaultdict(dict)
    for _, r in ep.sort_values(['subject_id', 'trial_number', 'epoch_family']).iterrows():
        sid = str(r.subject_id)
        run = int(r.raw_run)
        fam = str(r.epoch_family)
        tr = int(r.trial_number)
        anchor = anchor_map.get((sid, run))
        g = stream_info.get((sid, 'GSR'))
        p = stream_info.get((sid, 'PPG'))
        available = bool(r.epoch_available) and finite(r.epoch_start_sec) and finite(r.epoch_stop_sec)
        if anchor is not None and available and g and p:
            s = anchor + float(r.epoch_start_sec)
            e2 = anchor + float(r.epoch_stop_sec)
            fg = second_group_window_fraction(g, s, e2)
            fp = second_group_window_fraction(p, s, e2)
            joint = fg >= PHYSIO_MIN_SAMPLE_FRAC and fp >= PHYSIO_MIN_SAMPLE_FRAC
        else:
            s = e2 = fg = fp = math.nan
            joint = False
        row = {'subject_id': sid, 'trial_number': tr, 'raw_run': run, 'epoch_family': fam, 'eeg_epoch_available': available, 'exact_eeg_run_anchor_available': anchor is not None, 'epoch_abs_start_sec': s, 'epoch_abs_stop_sec': e2, 'gsr_timing_sample_fraction': fg, 'ppg_timing_sample_fraction': fp, 'joint_exact_timing_epoch_covered': joint, 'raw_signal_amplitude_read': False}
        epoch_rows.append(row)
        bytrial[sid, tr][fam] = (available, joint)
    for (sid, tr), st in sorted(bytrial.items()):
        eegall = all((st.get(f, (False, False))[0] for f in EPOCHS))
        joint = all((st.get(f, (False, False))[1] for f in EPOCHS))
        trial_rows.append({'subject_id': sid, 'trial_number': tr, 'eeg_all_three_epochs_available': eegall, 'joint_exact_timing_all_three_epochs_covered': joint, 'raw_signal_amplitude_read': False})
    tdf = pd.DataFrame(trial_rows)
    for sid in sorted(relevant):
        ss = tdf[tdf.subject_id.astype(str) == sid] if len(tdf) else pd.DataFrame()
        cand = int(ss.eeg_all_three_epochs_available.sum()) if len(ss) else 0
        comp = int(ss.joint_exact_timing_all_three_epochs_covered.sum()) if len(ss) else 0
        sr = sel[sel.subject_id.astype(str) == sid]
        allruns = bool(len(sr) and (sr.selection_status == 'exact_anchor_selected').all())
        part_rows.append({'subject_id': sid, 'candidate_eeg_all3_trials': cand, 'complete_joint_exact_timing_trajectories': comp, 'all_required_eeg_runs_exactly_anchored': allruns, 'participant_min16_pass': bool(allruns and comp >= MIN_TRAJ_PER_SUB), 'raw_signal_amplitude_read': False})
    total_cand = sum((x['candidate_eeg_all3_trials'] for x in part_rows))
    total_comp = sum((x['complete_joint_exact_timing_trajectories'] for x in part_rows))
    n_pass = sum((x['participant_min16_pass'] for x in part_rows))
    frac = total_comp / total_cand if total_cand else 0.0
    exact_gate = bool(n_pass >= MIN_PARTICIPANTS and frac >= MIN_COMPLETE_FRAC)
    paths = {'stream': project / 'outputs/qc/exact_physio_eeg_sync_recovery_physio_second_label_clock_audit.csv', 'gaps': project / 'outputs/qc/exact_physio_eeg_sync_recovery_physio_timestamp_gap_audit.csv', 'sources': project / 'outputs/qc/exact_physio_eeg_sync_recovery_eeg_timing_metadata_source_inventory.csv', 'meta': project / 'outputs/qc/exact_physio_eeg_sync_recovery_eeg_absolute_timestamp_candidates_raw.csv', 'edf': project / 'outputs/qc/exact_physio_eeg_sync_recovery_edf_standard_header_timing_audit.csv', 'deriv': project / 'outputs/qc/exact_physio_eeg_sync_recovery_author_derivative_timing_metadata_audit.csv', 'cands': project / 'outputs/tables/exact_physio_eeg_sync_recovery_run_anchor_candidates.csv', 'sel': project / 'outputs/tables/exact_physio_eeg_sync_recovery_exact_run_anchor_selection.csv', 'epochs': project / 'outputs/tables/exact_physio_eeg_sync_recovery_exact_epoch_timing_coverage.csv', 'trials': project / 'outputs/tables/exact_physio_eeg_sync_recovery_exact_trial_timing_coverage.csv', 'parts': project / 'outputs/tables/exact_physio_eeg_sync_recovery_participant_exact_sync_feasibility.csv', 'summary': project / 'outputs/qc/exact_physio_eeg_sync_recovery_exact_sync_recovery_summary.json', 'report': project / 'outputs/reports/exact_physio_eeg_sync_recovery_exact_physio_eeg_sync_recovery_report.md', 'next': project / 'analysis_specifications/S031_exact_physio_eeg_sync_recovery.json', 'contract': project / 'config/exact_physio_eeg_sync_recovery_exact_physio_eeg_sync_recovery_contract.json'}
    for pth in paths.values():
        pth.parent.mkdir(parents=True, exist_ok=True)
    writecsv(paths['stream'], stream_rows)
    writecsv(paths['gaps'], gap_rows)
    writecsv(paths['sources'], source_inventory)
    writecsv(paths['meta'], meta_rows)
    writecsv(paths['edf'], edf_rows)
    writecsv(paths['deriv'], deriv_rows)
    writecsv(paths['cands'], cand_rows)
    writecsv(paths['sel'], sel_rows)
    writecsv(paths['epochs'], epoch_rows)
    writecsv(paths['trials'], trial_rows)
    writecsv(paths['parts'], part_rows)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'exact_sync_gate_passed' if exact_gate else 'exact_sync_not_recovered', 'counts': {'audited_participants': len(part_rows), 'eeg_runs': len(runs), 'exactly_anchored_eeg_runs': int((sel.selection_status == 'exact_anchor_selected').sum()), 'participants_with_min16_exact_joint_trajectories': n_pass, 'candidate_EEG_all3_trials': total_cand, 'complete_joint_exact_timing_trajectories': total_comp, 'physio_streams_audited': len(stream_rows), 'dual_modality_gap_support_rows': len(gap_support)}, 'fractions': {'complete_joint_exact_timing_fraction': frac}, 'gates': {'minimum_participants_required': MIN_PARTICIPANTS, 'minimum_complete_fraction_required': MIN_COMPLETE_FRAC, 'minimum_trials_per_participant': MIN_TRAJ_PER_SUB, 'exact_sync_gate_pass': exact_gate}, 'synchronization': {'target_uncertainty_sec': TARGET_SYNC_UNCERTAINTY_SEC, 'exact_event_sync_established': exact_gate, 'physio_clock_reconstruction': 'second label + within-second sample index / SamplingFrequency'}, 'boundaries': {'raw_GSR_amplitude_read': False, 'raw_PPG_amplitude_read': False, 'EEG_sample_amplitude_read': False, 'behavioral_outcomes_read_for_sync': False, 'family_assignments_read_for_sync': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False}, 'raw_amplitude_analysis_permitted': exact_gate, 'fallback_author_GSR_stimulus_dynamic_audit_permitted': not exact_gate}
    writej(paths['summary'], summary)
    nextpol = {'phase': PHASE, 'generated_at': now(), 'if_exact_sync_gate_passes': {'next_phase': 'raw_dynamic_physio_preprocessing_specification', 'role': 'specification Discovery raw GSR/PPG amplitude preprocessing/QC and trajectory representation before first amplitude access.', 'raw_amplitude_access_in_exact_physio_eeg_sync_recovery': False}, 'if_exact_sync_gate_fails': {'next_phase': 'source_gsr_dynamic_design', 'role': 'specification author-provided stimulus-only dynamic GSR derivative route; do not claim rating-period raw physiology was tested.', 'raw_amplitude_access_in_exact_physio_eeg_sync_recovery': False}, 'selected_next_phase': 'raw_dynamic_physio_preprocessing_specification' if exact_gate else 'source_gsr_dynamic_design', 'forbidden': ['using Valence/Liking/reports/emotion labels/family assignments to choose synchronization', 'using EEG/GSR/PPG amplitude to choose synchronization', 'moving EEG epoch windows', 'opening Validation or Confirmation raw physiology', 'treating coarse README-minute overlap or physiology gaps alone as exact EEG anchors']}
    writej(paths['next'], nextpol)
    report = f"# exact_physio_eeg_sync_recovery — outcome-blind exact EEG↔physiology synchronization recovery\n\nStatus: **{('PASS — exact sync recovered' if exact_gate else 'EXACT SYNC NOT RECOVERED')}**.\n\nThis phase read timing metadata/columns only. It did **not** read GSR amplitude, PPG amplitude, EEG sample amplitude, behavioral outcomes, or EEG family assignments.\n\n## Timing reconstruction\nRaw physiology `csv_time_*` values are treated as one-second labels repeated across samples. The timing-only reconstruction is `second_label + within-second sample index / SamplingFrequency`, while preserving detected clock gaps.\n\n## Exact anchor result\n- EEG runs audited: **{len(runs)}**\n- EEG runs with accepted <=1 s absolute start anchor: **{int((sel.selection_status == 'exact_anchor_selected').sum())}**\n- Participants with >=16 complete exact joint trajectories: **{n_pass}**\n- Candidate EEG three-epoch trials: **{total_cand}**\n- Complete exact joint GSR+PPG timing trajectories: **{total_comp}** ({frac:.4f})\n- prespecified exact-sync gate (>=18 participants and >=70%): **{('PASS' if exact_gate else 'FAIL')}**\n\nPhysiology timestamp gaps were audited only as supporting evidence and were never accepted as an EEG absolute anchor by themselves.\n\nIf the exact-sync gate fails, raw event-aligned amplitude analysis remains closed. The only pre-authorized fallback is an audit/specification of author-provided 10×3-s GSR stimulus derivatives, which can test stimulus-period dynamics but cannot test rating-period raw physiology."
    paths['report'].write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'inputs': {'raw_dynamic_physio_sync_correction_lock_sha256': sha256(project / E_LOCK), 'raw_dynamic_physio_sync_correction_summary_sha256': sha256(project / E_SUM), 'raw_dynamic_physio_sync_correction_policy_sha256': sha256(project / E_POLICY), 'raw_eeg_epoch_folds_epoch_sha256': sha256(project / A_EPOCH), 'raw_eeg_epoch_folds_timing_sha256': sha256(project / A_TIMING), 'raw_dynamic_physio_alignment_design_roster_sha256': sha256(project / A_ROSTER), 'raw_dynamic_physio_timebase_audit_stream_sha256': sha256(project / B_STREAM), 'prespecified_readme_endtime_snapshot_sha256': SNAPSHOT_SHA}, 'outputs': {}, 'exact_sync_gate_pass': exact_gate, 'raw_signal_amplitude_read': False, 'behavioral_outcomes_read_for_sync': False, 'family_assignments_read_for_sync': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False}
    lockhash = {}
    for k, pth in paths.items():
        if k == 'contract':
            continue
        lockhash[str(pth.relative_to(project))] = sha256(pth)
    contract['outputs'] = lockhash
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/exact_physio_eeg_sync_recovery_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': lockhash, 'exact_sync_gate_pass': exact_gate, 'next_phase': nextpol['selected_next_phase'], 'raw_signal_amplitude_read': False, 'behavioral_outcomes_read_for_sync': False, 'family_assignments_read_for_sync': False, 'Validation_raw_physio_opened': False, 'Confirmation_raw_physio_opened': False, 'inference_performed': False})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'exact_physio_eeg_sync_recovery_exact_physio_eeg_sync_recovery_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print(f"[{now()}] exact_physio_eeg_sync_recovery {('PASS' if exact_gate else 'STOP-RAW')}: exact_runs={(sel.selection_status == 'exact_anchor_selected').sum()}/{len(sel)}; participants={n_pass}; complete={total_comp}/{total_cand} ({frac:.4f})", flush=True)
    print('Next phase:', nextpol['selected_next_phase'], flush=True)
    print('', hz, flush=True)
    print('SHA-256:', sh, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())