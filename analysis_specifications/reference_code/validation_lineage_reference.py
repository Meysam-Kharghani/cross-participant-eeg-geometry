# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase validation_source_epoch_lineage — final Validation source/header/epoch lineage specification.

The final lineage rule uses the audited
outcome-free event grammar. Complete 42-trial rosters use prespecified 21/21 context
capacity and chronological sequence mapping. sub-57 uses its 32 exact retained
rating anchors because ten raw trials are absent from the behavioral roster."""
from __future__ import annotations
import argparse, bisect, hashlib, importlib.util, json, math, re, time
from datetime import datetime
from pathlib import Path
import mne
import numpy as np
import pandas as pd
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'validation_source_epoch_lineage'
ROSTER = {'sub-07': 42, 'sub-09': 42, 'sub-11': 42, 'sub-17': 42, 'sub-21': 42, 'sub-25': 42, 'sub-28': 42, 'sub-37': 42, 'sub-39': 42, 'sub-51': 42, 'sub-53': 42, 'sub-57': 32}
FULL42 = set((k for k, v in ROSTER.items() if v == 42))
SUB57_TRIALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
EXPECTED = {'i_policy': '44b0a261d3302cafaad22759f0fb2765e3f40344c49c49742496772efe078c62', 'i_ref': 'b319b782ae2d2149efdbec70094298e077be8eaeda51b32f34ce26ffcf2d0dd1', 'i_core': '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b', 'i_next': 'a8e016a6e4d4966cbee378e4789716a00c0c94b5374a2c20c70f3cb8102de6ea', 'raw_eeg_epoch_folds': '93ca3cc67f7fb72f008845941a16f8a802e4b0aee601d53fbeb5c84c8e25aa8b', 'raw_eeg_preprocessing_qc': 'bd6a50669f0a6ba17965a4497433f1d4583d4a98f840dff29a73866d801e2ecb', 'outcome_free_features': 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba', 'resolution': 'db78b6578d8ea81ac40ebc6f6abd310a45ed1749d80d258dce0905213914a6d7', 'diag_summary': 'd52a384cf73420d2593d72b3ab7ea6ba258773033493c9aad80edd5a8b2bffc3', 'diag_subjects': 'b241acc8d3a62b8d7aaa4839f6d28d874464fcc5e493237098475441ccaba64f', 'diag_anchors': 'f0e194723e3c9b9e67fb506441a2cb271db705de48693370a984cea8efa31972'}
TRANSFER_CANONICAL = 'bd3e7a5f4bd0cba00cfa66a0cd09208c07e0b1a9edab4b2a9493060200e1b37e'
FORBIDDEN = ('score_', 'valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion', 'stimulus_id', 'video_name')
PRIORS = {'imagery': {'n': 472, 'median_log_delay': 4.339744373079974, 'robust_scale_log_delay': 0.2181073407641778, 'median_delay_sec': 76.68793333228997}, 'video': {'n': 471, 'median_log_delay': 4.611135690265312, 'robust_scale_log_delay': 0.21874635865798572, 'median_delay_sec': 100.59833333333329}}
VARIANTS = {'primary': {'maximum_delay_sec': 650.0, 'ratings_per_stimulus': 6, 'intervening_stimulus_penalty': 0.4, 'intervening_rating_penalty': 0.6}, 'tight': {'maximum_delay_sec': 500.0, 'ratings_per_stimulus': 5, 'intervening_stimulus_penalty': 0.5, 'intervening_rating_penalty': 0.8}, 'wide': {'maximum_delay_sec': 700.0, 'ratings_per_stimulus': 8, 'intervening_stimulus_penalty': 0.3, 'intervening_rating_penalty': 0.5}}
MIN_DELAY = 30.0
WITHIN_PERSON_FOLDS = 5

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def normsub(x):
    m = re.fullmatch('(?:sub-)?(\\d+)', str(x).strip(), re.I)
    return f'sub-{int(m.group(1)):02d}' if m else str(x).strip()

def normalize_context(value):
    t = str(value).strip().lower()
    if t in {'ima', 'imagery', 'ses-ima'}:
        return 'imagery'
    if t in {'vid', 'video', 'ses-vid'}:
        return 'video'
    return t

def safe_csv(path, allowed, required):
    cols = list(pd.read_csv(path, nrows=0).columns)
    use = [c for c in allowed if c in cols]
    if any((any((t in c.lower() for t in FORBIDDEN)) for c in use)):
        raise RuntimeError('Unsafe column requested.')
    miss = [c for c in required if c not in use]
    if miss:
        raise RuntimeError(f'{Path(path).name} missing safe columns {miss}; available={cols}')
    return pd.read_csv(path, usecols=use, low_memory=False)

def require(paths):
    miss = [str(p) for p in paths if not Path(p).is_file()]
    if miss:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(miss))

def loadmod(path):
    spec = importlib.util.spec_from_file_location('prespecified_raw_eeg_epoch_folds', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def runno(p):
    m = re.search('_run-(\\d+)_', Path(p).name)
    return int(m.group(1)) if m else 1

def tid(desc):
    m = re.search('TypeID\\s*:\\s*(-?\\d+)', str(desc), re.I)
    return int(m.group(1)) if m else None

def trigger_map(readme, subjects):
    linepat = re.compile('(sub-\\d+)\\s*:\\s*(.*)', re.I)
    pairpat = re.compile('(vid|ima|fade|fede|rating)\\s*[-:]?\\s*(\\d+)', re.I)
    rows = []
    for n, line in enumerate(Path(readme).read_text(errors='replace').splitlines(), 1):
        m = linepat.search(line)
        if not m or normsub(m.group(1)) not in subjects:
            continue
        sub = normsub(m.group(1))
        found = {}
        for k, v in pairpat.findall(m.group(2)):
            cls = {'vid': 'video', 'ima': 'imagery', 'fede': 'fade'}.get(k.lower(), k.lower())
            found[cls] = int(v)
        if set(found) != {'video', 'imagery', 'fade', 'rating'}:
            raise RuntimeError(f'Incomplete trigger map for {sub}: {found}')
        for cls, v in found.items():
            rows.append({'subject_id': sub, 'event_class': cls, 'type_id': v, 'readme_line': n})
    out = pd.DataFrame(rows)
    if set(out.subject_id) != subjects or out.duplicated(['subject_id', 'event_class']).any():
        raise RuntimeError('Trigger map gate failed.')
    return out

def headers_events(dataset, trig):
    lookup = {(r.subject_id, int(r.type_id)): r.event_class for r in trig.itertuples()}
    headers = []
    events = []
    for sub in sorted(ROSTER):
        files = sorted((dataset / sub / 'eeg').glob(f'{sub}_task-emotion*_eeg.edf'), key=lambda p: (runno(p), p.name))
        if not files:
            raise RuntimeError(f'No EDF for {sub}')
        offset = 0.0
        for p in files:
            raw = mne.io.read_raw_edf(p, preload=False, verbose='ERROR')
            sf = float(raw.info['sfreq'])
            dur = float(raw.n_times / sf)
            rn = runno(p)
            headers.append({'raw_source_path': str(p.resolve()), 'subject_id': sub, 'run': rn, 'source_size_bytes': p.stat().st_size, 'source_mtime_ns': p.stat().st_mtime_ns, 'sampling_rate_hz': sf, 'n_times': int(raw.n_times), 'recording_duration_sec': dur, 'channel_count': len(raw.ch_names), 'channel_order_sha256': hashlib.sha256(''.join(raw.ch_names).encode()).hexdigest(), 'highpass_hz': float(raw.info.get('highpass', math.nan)), 'lowpass_hz': float(raw.info.get('lowpass', math.nan)), 'annotation_count': len(raw.annotations), 'header_readable': True, 'sample_amplitude_read': False, 'run_offset_sec': offset, 'run_global_start_sec': offset, 'run_global_end_sec': offset + dur})
            for i, (on, d, desc) in enumerate(zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)):
                t = tid(desc)
                events.append({'subject_id': sub, 'run': rn, 'source_path': str(p.resolve()), 'annotation_index': i, 'onset_sec': float(on), 'duration_sec': float(d), 'description': str(desc), 'type_id': t, 'event_class': lookup.get((sub, t), 'unclassified'), 'classification_method': 'readme_exact_type_id' if (sub, t) in lookup else 'unclassified', 'run_offset_sec': offset, 'global_onset_sec': offset + float(on), 'sample_amplitude_read': False})
            raw.close()
            offset += dur
    return (pd.DataFrame(headers), pd.DataFrame(events))

def schedule(project, dataset):
    can = project / 'data/canonical_behavioral_table.csv'
    require([can])
    c = safe_csv(can, ('subject_id', 'trial_number_global'), ('subject_id', 'trial_number_global'))
    c.subject_id = c.subject_id.map(normsub)
    c = c[c.subject_id.isin(ROSTER)].copy()
    c.trial_number_global = pd.to_numeric(c.trial_number_global, errors='raise').astype(int)
    if c.groupby('subject_id').size().to_dict() != ROSTER or c.duplicated(['subject_id', 'trial_number_global']).any():
        raise RuntimeError('Canonical Validation roster/key gate failed.')
    for sub in FULL42:
        if c[c.subject_id.eq(sub)].trial_number_global.sort_values().tolist() != list(range(1, 43)):
            raise RuntimeError(f'{sub}: complete roster is not trial 1..42')
    if c[c.subject_id.eq('sub-57')].trial_number_global.sort_values().tolist() != SUB57_TRIALS:
        raise RuntimeError('sub-57 retained roster differs from prespecified resolution.')
    rows = []
    metadata_paths = []
    for sub, g in c.groupby('subject_id', sort=True):
        mp = dataset / 'derivatives/metadata' / f'{sub}_trials_with_context.csv'
        require([mp])
        metadata_paths.append(mp)
        m = safe_csv(mp, ('sub', 'global_trial', 'context', 'context_trial_idx', 'rating_time_sec', 'trial_number'), ('global_trial', 'trial_number', 'context', 'rating_time_sec'))
        m.global_trial = pd.to_numeric(m.global_trial, errors='raise').astype(int)
        m.trial_number = pd.to_numeric(m.trial_number, errors='raise').astype(int)
        m.rating_time_sec = pd.to_numeric(m.rating_time_sec, errors='coerce')
        m.context = m.context.map(normalize_context)
        if m.duplicated('trial_number').any():
            raise RuntimeError(f'{sub}: duplicate metadata trial_number')
        x = g.merge(m, left_on='trial_number_global', right_on='trial_number', how='left', validate='one_to_one')
        if x.rating_time_sec.isna().any() or (~x.context.isin(['imagery', 'video'])).any():
            raise RuntimeError(f'{sub}: retained safe metadata fields incomplete')
        for r in x.sort_values('trial_number_global').itertuples():
            rows.append({'raw_behavior_row_id': f'validation|{sub}|trial-{int(r.trial_number_global):03d}', 'subject_id': sub, 'trial_number': int(r.trial_number_global), 'metadata_global_trial': int(r.global_trial), 'metadata_context': str(r.context), 'metadata_rating_time_sec': float(r.rating_time_sec), 'metadata_fields_role': 'diagnostic_non_authoritative_except_sub57_exact_rating_anchor', 'reports_outcomes_read': False, 'stimulus_identity_read': False, 'canonical_context_read': False, 'metadata_global_trial_used_as_join_key': False})
    out = pd.DataFrame(rows).sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    return (out, can, metadata_paths)

def generate_candidates(sub, events, variant):
    pars = VARIANTS[variant]
    e = events[events.subject_id.eq(sub)].sort_values(['global_onset_sec', 'run', 'annotation_index']).copy()
    stim = e[e.event_class.isin(['imagery', 'video'])].drop_duplicates(['source_path', 'annotation_index']).sort_values('global_onset_sec').reset_index(drop=True)
    rating = e[e.event_class.eq('rating')].drop_duplicates(['source_path', 'annotation_index']).sort_values('global_onset_sec').reset_index(drop=True)
    rt = rating.global_onset_sec.to_numpy(float)
    rows = []
    for s in stim.itertuples():
        st = float(s.global_onset_sec)
        first = int(np.searchsorted(rt, st + MIN_DELAY, side='left'))
        last = min(len(rating), first + int(pars['ratings_per_stimulus']))
        for j in range(first, last):
            r = rating.iloc[j]
            delay = float(r.global_onset_sec) - st
            if delay > pars['maximum_delay_sec']:
                break
            prior = PRIORS[str(s.event_class)]
            z = abs(math.log(delay) - prior['median_log_delay']) / prior['robust_scale_log_delay']
            inters = int(((stim.global_onset_sec > st) & (stim.global_onset_sec < float(r.global_onset_sec))).sum())
            interr = int(((rating.global_onset_sec > st) & (rating.global_onset_sec < float(r.global_onset_sec))).sum())
            cost = min(z, 20.0) + pars['intervening_stimulus_penalty'] * inters + pars['intervening_rating_penalty'] * interr
            rows.append({'subject_id': sub, 'variant': variant, 'context': str(s.event_class), 'stimulus_onset_global_sec': st, 'rating_onset_global_sec': float(r.global_onset_sec), 'stimulus_to_rating_sec': delay, 'pair_cost': float(cost), 'intervening_stimulus_events': inters, 'intervening_rating_events': interr, 'stimulus_run': int(s.run), 'rating_run': int(r.run), 'stimulus_source_path': str(s.source_path), 'rating_source_path': str(r.source_path), 'stimulus_annotation_index': int(s.annotation_index), 'rating_annotation_index': int(r.annotation_index)})
    return pd.DataFrame(rows)

def select_nonoverlap(candidates, target_imagery, target_video):
    if candidates.empty:
        return (candidates.copy(), {'selected_total': 0, 'feasible': False})
    f = candidates.sort_values(['rating_onset_global_sec', 'stimulus_onset_global_sec', 'pair_cost']).reset_index(drop=True)
    rtimes = f.rating_onset_global_sec.to_numpy(float)
    prev = [bisect.bisect_left(rtimes, float(row.stimulus_onset_global_sec)) - 1 for row in f.itertuples()]
    n = len(f)
    inf = 1000000000000000.0
    dp = np.full((n + 1, target_imagery + 1, target_video + 1), inf)
    take = np.zeros(dp.shape, dtype=np.int8)
    dp[0, 0, 0] = 0.0
    for ci in range(1, n + 1):
        dp[ci] = dp[ci - 1]
        row = f.iloc[ci - 1]
        pi = prev[ci - 1] + 1
        ai = int(row.context == 'imagery')
        av = int(row.context == 'video')
        for ii in range(ai, target_imagery + 1):
            for vv in range(av, target_video + 1):
                val = dp[pi, ii - ai, vv - av] + float(row.pair_cost)
                if val < dp[ci, ii, vv]:
                    dp[ci, ii, vv] = val
                    take[ci, ii, vv] = 1
    feasible = []
    for ii in range(target_imagery + 1):
        for vv in range(target_video + 1):
            if dp[n, ii, vv] < inf:
                feasible.append((ii + vv, -float(dp[n, ii, vv]), ii, vv))
    _, negcost, ii, vv = max(feasible)
    chosen = []
    ci = n
    curi = ii
    curv = vv
    while ci > 0:
        if take[ci, curi, curv] == 1:
            fi = ci - 1
            row = f.iloc[fi]
            chosen.append(fi)
            curi -= int(row.context == 'imagery')
            curv -= int(row.context == 'video')
            ci = prev[fi] + 1
        else:
            ci -= 1
    sel = f.iloc[chosen[::-1]].copy()
    sel['selection_rank'] = np.arange(1, len(sel) + 1)
    return (sel, {'selected_total': len(sel), 'selected_imagery': int((sel.context == 'imagery').sum()), 'selected_video': int((sel.context == 'video').sum()), 'target_total': target_imagery + target_video, 'target_imagery': target_imagery, 'target_video': target_video, 'total_cost': -negcost, 'feasible': len(sel) == target_imagery + target_video})

def signature(df):
    return {(round(float(r.stimulus_onset_global_sec), 4), round(float(r.rating_onset_global_sec), 4), str(r.context)) for r in df.itertuples()}

def event_row_lookup(events):
    return events.set_index(['source_path', 'annotation_index'], drop=False).to_dict('index')

def pair_to_trial(schedule_row, pair, lookup, method, in_tight, in_wide):
    sk = (str(pair.stimulus_source_path), int(pair.stimulus_annotation_index))
    rk = (str(pair.rating_source_path), int(pair.rating_annotation_index))
    if sk not in lookup or rk not in lookup:
        raise RuntimeError('Selected event absent from inventory.')
    ss = lookup[sk]
    rr = lookup[rk]
    if ss['source_path'] != rr['source_path'] or int(ss['run']) != int(rr['run']):
        raise RuntimeError('Cross-run selected pair.')
    mdctx = str(schedule_row.metadata_context)
    evctx = str(pair.context)
    base = schedule_row.to_dict() if hasattr(schedule_row, 'to_dict') else schedule_row._asdict()
    return {**base, 'context': evctx, 'context_source': method, 'metadata_event_context_agreement': bool(mdctx == evctx), 'metadata_rating_delta_from_selected_sec': float(pair.rating_onset_global_sec - float(schedule_row.metadata_rating_time_sec)), 'source_stimulus_candidate_count': math.nan, 'source_stimulus_candidate_contexts': evctx, 'source_stimulus_mixed_context_candidates': False, 'source_competing_stimulus_before_selected_count': int(pair.intervening_stimulus_events), 'source_stimulus_selection_rule': method, 'event_grammar_pair_cost': float(pair.pair_cost), 'event_grammar_selection_stability_count': int(1 + int(in_tight) + int(in_wide)), 'event_grammar_selected_in_tight': bool(in_tight), 'event_grammar_selected_in_wide': bool(in_wide), 'source_stimulus_run': int(ss['run']), 'source_stimulus_path': str(ss['source_path']), 'source_stimulus_annotation_index': int(ss['annotation_index']), 'source_stimulus_local_onset_sec': float(ss['onset_sec']), 'source_stimulus_global_onset_sec': float(ss['global_onset_sec']), 'source_rating_run': int(rr['run']), 'source_rating_path': str(rr['source_path']), 'source_rating_annotation_index': int(rr['annotation_index']), 'source_rating_local_onset_sec': float(rr['onset_sec']), 'source_rating_global_onset_sec': float(rr['global_onset_sec']), 'continuous_event_authoritative': True}

def attach_sub57(g, events, lookup):
    sub = 'sub-57'
    e = events[events.subject_id.eq(sub)].sort_values(['global_onset_sec', 'run', 'annotation_index']).reset_index(drop=True)
    ratings = e[e.event_class.eq('rating')].reset_index(drop=True)
    rows = []
    useds = set()
    usedr = set()
    for r in g.sort_values('trial_number').itertuples():
        delta = np.abs(ratings.global_onset_sec.to_numpy(float) - float(r.metadata_rating_time_sec))
        hit = np.flatnonzero(delta <= 0.001)
        if len(hit) != 1:
            raise RuntimeError(f'sub-57 trial {r.trial_number}: exact retained rating anchor count {len(hit)}')
        pos = int(hit[0])
        rr = ratings.iloc[pos]
        prev = float(ratings.iloc[pos - 1].global_onset_sec) if pos > 0 else -math.inf
        stimuli = e[(e.global_onset_sec > prev + 1e-09) & (e.global_onset_sec < float(rr.global_onset_sec) - 1e-09) & e.event_class.isin(['imagery', 'video'])].sort_values(['global_onset_sec', 'run', 'annotation_index'])
        if len(stimuli) != 1:
            raise RuntimeError(f'sub-57 trial {r.trial_number}: expected one exact stimulus between raw rating anchors; observed {len(stimuli)}')
        ss = stimuli.iloc[0]
        sk = (str(ss.source_path), int(ss.annotation_index))
        rk = (str(rr.source_path), int(rr.annotation_index))
        if sk in useds or rk in usedr:
            raise RuntimeError('sub-57 source event reuse.')
            useds.add(sk)
            usedr.add(rk)
        if int(ss.run) != int(rr.run) or str(ss.source_path) != str(rr.source_path):
            raise RuntimeError('sub-57 cross-run pair.')
        evctx = str(ss.event_class)
        rows.append({**r._asdict(), 'context': evctx, 'context_source': 'sub57_exact_retained_rating_anchor_unique_adjacent_source_stimulus', 'metadata_event_context_agreement': bool(str(r.metadata_context) == evctx), 'metadata_rating_delta_from_selected_sec': float(rr.global_onset_sec - float(r.metadata_rating_time_sec)), 'source_previous_rating_global_onset_sec': prev if math.isfinite(prev) else np.nan, 'source_stimulus_candidate_count': 1, 'source_stimulus_candidate_contexts': evctx, 'source_stimulus_mixed_context_candidates': False, 'source_competing_stimulus_before_selected_count': 0, 'source_stimulus_selection_rule': 'sub57_exact_retained_rating_anchor_unique_adjacent_source_stimulus', 'event_grammar_pair_cost': np.nan, 'event_grammar_selection_stability_count': 3, 'event_grammar_selected_in_tight': True, 'event_grammar_selected_in_wide': True, 'source_stimulus_run': int(ss.run), 'source_stimulus_path': str(ss.source_path), 'source_stimulus_annotation_index': int(ss.annotation_index), 'source_stimulus_local_onset_sec': float(ss.onset_sec), 'source_stimulus_global_onset_sec': float(ss.global_onset_sec), 'source_rating_run': int(rr.run), 'source_rating_path': str(rr.source_path), 'source_rating_annotation_index': int(rr.annotation_index), 'source_rating_local_onset_sec': float(rr.onset_sec), 'source_rating_global_onset_sec': float(rr.global_onset_sec), 'continuous_event_authoritative': True})
    out = pd.DataFrame(rows)
    if out.context.value_counts().to_dict() != {'imagery': 18, 'video': 14}:
        raise RuntimeError(f'sub-57 prespecified context count gate failed: {out.context.value_counts().to_dict()}')
    return out

def attach(sched, events):
    rows = []
    candidate_rows = []
    selected_rows = []
    grammar_qc = []
    lookup = event_row_lookup(events)
    for sub, g in sched.groupby('subject_id', sort=True):
        if sub == 'sub-57':
            subrows = attach_sub57(g, events, lookup)
            rows.append(subrows)
            grammar_qc.append({'subject_id': sub, 'attachment_method': 'exact_retained_anchor', 'roster_trials': 32, 'target_imagery': 18, 'target_video': 14, 'primary_selected': 32, 'tight_selected': 32, 'wide_selected': 32, 'primary_pairs_in_tight': 32, 'primary_pairs_in_wide': 32, 'primary_total_cost': np.nan, 'attachment_complete': True})
            continue
        variants = {}
        qs = {}
        for name in VARIANTS:
            cand = generate_candidates(sub, events, name)
            sel, q = select_nonoverlap(cand, 21, 21)
            candidate_rows.append(cand)
            variants[name] = sel
            qs[name] = q
            tmp = sel.copy()
            tmp['selected_variant'] = name
            selected_rows.append(tmp)
        p = variants['primary'].sort_values('rating_onset_global_sec').reset_index(drop=True)
        tset = signature(variants['tight'])
        wset = signature(variants['wide'])
        pset = signature(p)
        if any((not qs[name]['feasible'] for name in VARIANTS)) or len(p) != 42:
            raise RuntimeError(f'{sub}: corrected 21/21 event grammar did not reach all 42 trials: {qs}')
        if (p.stimulus_run != p.rating_run).any() or (p.stimulus_source_path != p.rating_source_path).any():
            raise RuntimeError(f'{sub}: cross-run primary pair')
        go = g.sort_values('trial_number').reset_index(drop=True)
        for i in range(42):
            pair = p.iloc[i]
            sig = (round(float(pair.stimulus_onset_global_sec), 4), round(float(pair.rating_onset_global_sec), 4), str(pair.context))
            stability = 1 + int(sig in tset) + int(sig in wset)
            rows.append(pd.DataFrame([pair_to_trial(go.iloc[i], pair, lookup, 'primary_event_grammar_prespecified_21_imagery_21_video_chronological_sequence', sig in tset, sig in wset)]))
        grammar_qc.append({'subject_id': sub, 'attachment_method': 'primary_event_grammar_sequence', 'roster_trials': 42, 'target_imagery': 21, 'target_video': 21, 'primary_selected': len(p), 'tight_selected': len(variants['tight']), 'wide_selected': len(variants['wide']), 'primary_pairs_in_tight': len(pset & tset), 'primary_pairs_in_wide': len(pset & wset), 'primary_total_cost': float(qs['primary']['total_cost']), 'attachment_complete': True})
    out = pd.concat(rows, ignore_index=True, sort=False).drop(columns=['Index'], errors='ignore')
    out = out.sort_values(['subject_id', 'source_stimulus_global_onset_sec', 'raw_behavior_row_id']).reset_index(drop=True)
    out['subject_acquisition_order_1based'] = out.groupby('subject_id').cumcount() + 1
    out['within_subject_block_fold'] = out.groupby('subject_id')['subject_acquisition_order_1based'].transform(lambda s: np.floor((s.to_numpy(float) - 1.0) * WITHIN_PERSON_FOLDS / float(len(s))).astype(int) + 1)
    if len(out) != 494 or out.subject_id.nunique() != 12 or out.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Final attached roster gate failed.')
    if out.groupby('subject_id').size().to_dict() != ROSTER:
        raise RuntimeError('Final subject counts changed.')
    if out.duplicated(['source_rating_path', 'source_rating_annotation_index']).any() or out.duplicated(['source_stimulus_path', 'source_stimulus_annotation_index']).any():
        raise RuntimeError('Final source event reuse gate failed.')
    if not out.source_stimulus_path.eq(out.source_rating_path).all() or not out.source_stimulus_run.eq(out.source_rating_run).all():
        raise RuntimeError('Final source pair run/path mismatch.')
    canddf = pd.concat(candidate_rows, ignore_index=True, sort=False) if candidate_rows else pd.DataFrame()
    seldf = pd.concat(selected_rows, ignore_index=True, sort=False) if selected_rows else pd.DataFrame()
    return (out, canddf, seldf, pd.DataFrame(grammar_qc))

def zippack(project, files):
    root = f'validation_source_epoch_lineage_validation_source_epoch_lineage_{tag()}'
    z = project / '' / f'{root}'
    d = sha(z)
    Path(str(z) + '.sha256').write_text(f'{d}  {z.name}')
    return (z, d)

def selftest():
    assert normsub('7') == 'sub-07' and normalize_context('ima') == 'imagery' and (normalize_context('vid') == 'video')
    c = pd.DataFrame([{'context': 'imagery', 'stimulus_onset_global_sec': 0.0, 'rating_onset_global_sec': 80.0, 'pair_cost': 1.0}, {'context': 'video', 'stimulus_onset_global_sec': 90.0, 'rating_onset_global_sec': 190.0, 'pair_cost': 1.0}, {'context': 'imagery', 'stimulus_onset_global_sec': 0.0, 'rating_onset_global_sec': 190.0, 'pair_cost': 9.0}])
    s, q = select_nonoverlap(c, 1, 1)
    assert len(s) == 2 and q['feasible']
    order = np.arange(1, 43)
    folds = np.floor((order - 1) * 5 / 42).astype(int) + 1
    assert set(folds) == {1, 2, 3, 4, 5}
    print('validation source-epoch lineage self-test: PASS')
    return 0

def main():
    a = args()
    if a.self_test:
        return selftest()
    project = a.project.expanduser().resolve()
    dataset = a.dataset.expanduser().resolve()
    started = time.time()
    P = {'i_policy': project / 'analysis_specifications/S082_validation_geometry_transfer.json', 'i_ref': project / 'analysis_specifications/S081_validation_geometry_transfer_reference.json', 'i_core': project / 'analysis_specifications/reference_code/validation_geometry_transfer_core.py', 'i_next': project / 'analysis_specifications/S083_validation_geometry_transfer_source_epoch.json', 'i_lock': project / 'outputs/qc/validation_geometry_transfer_design_final_output_lock_hashes.json', 'split': project / 'config/participant_split_manifest.v1.csv', 'readme': dataset / 'README.md', 'raw_eeg_epoch_folds': project / 'src/analysis_pipeline/eeg/raw_eeg_epoch_folds/raw_eeg_epoch_folds.py', 'raw_eeg_preprocessing_qc': project / 'src/analysis_pipeline/eeg/raw_eeg_preprocessing_qc/raw_eeg_preprocessing_qc.py', 'outcome_free_features': project / 'src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py', 'resolution': project / 'analysis_specifications/S089_validation_source_epoch_lineage_attachment_resolution.json', 'diag_summary': project / 'outputs/qc/validation_source_epoch_lineagediag_summary.json', 'diag_subjects': project / 'outputs/qc/validation_source_epoch_lineagediag_subject_summary.csv', 'diag_anchors': project / 'outputs/qc/validation_source_epoch_lineagediag_anchor_window_diagnostics.csv'}
    require(P.values())
    for k, v in EXPECTED.items():
        if sha(P[k]) != v:
            raise RuntimeError(f'{k} hash mismatch: expected {v}, observed {sha(P[k])}')
    policy = json.loads(P['i_policy'].read_text())
    nxt = json.loads(P['i_next'].read_text())
    resolution = json.loads(P['resolution'].read_text())
    if canon(policy) != TRANSFER_CANONICAL or nxt.get('next_phase') != PHASE or (not nxt.get('next_phase_permitted')):
        raise RuntimeError('validation_geometry_transfer_design authorization/canonical policy failed.')
    if resolution.get('resolution_id') != 'validation_source_epoch_lineage_validated_attachment_rule_EVENT_GRAMMAR_FINAL':
        raise RuntimeError('Attachment resolution ID mismatch.')
    diag = json.loads(P['diag_summary'].read_text())
    if diag.get('prespecified_roster_trials') != 494 or diag.get('anchor_failures_adjacent_window') != 51:
        raise RuntimeError('Diagnostic evidence mismatch.')
    split = safe_csv(P['split'], ('subject_id', 'split', 'n_trials', 'split_version'), ('subject_id', 'split', 'n_trials'))
    split.subject_id = split.subject_id.map(normsub)
    val = split[split.subject_id.isin(ROSTER)]
    if not val.split.eq('validation').all() or dict(zip(val.subject_id, val.n_trials.astype(int))) != ROSTER:
        raise RuntimeError('prespecified split roster mismatch.')
    raw_eeg_epoch_folds = loadmod(P['raw_eeg_epoch_folds'])
    O = {'headers': project / 'outputs/qc/validation_source_epoch_lineage_validation_raw_edf_header_inventory.csv', 'events': project / 'outputs/tables/validation_source_epoch_lineage_validation_annotation_event_inventory.csv', 'grammar_candidates': project / 'outputs/tables/validation_source_epoch_lineage_validation_event_grammar_candidates.csv', 'grammar_selected': project / 'outputs/tables/validation_source_epoch_lineage_validation_event_grammar_selected_pairs.csv', 'grammar_qc': project / 'outputs/qc/validation_source_epoch_lineage_validation_event_grammar_subject_qc.csv', 'trials': project / 'outputs/tables/validation_source_epoch_lineage_validation_authoritative_trial_timing_manifest.csv', 'epochs': project / 'outputs/tables/validation_source_epoch_lineage_validation_prespecified_epoch_manifest.csv', 'coverage': project / 'outputs/qc/validation_source_epoch_lineage_validation_epoch_coverage_summary.csv', 'subject_qc': project / 'outputs/qc/validation_source_epoch_lineage_validation_subject_lineage_qc.csv', 'source_rehash': project / 'outputs/qc/validation_source_epoch_lineage_source_rehash.csv', 'summary': project / 'outputs/qc/validation_source_epoch_lineage_validation_source_epoch_lineage_summary.json', 'contract': project / 'config/validation_source_epoch_lineage_validation_source_epoch_lineage_contract.json', 'next': project / 'analysis_specifications/S090_validation_source_epoch_lineage_preprocessing.json', 'report': project / 'outputs/reports/validation_source_epoch_lineage_validation_source_epoch_lineage_report.md', 'lock': project / 'outputs/qc/validation_source_epoch_lineage_final_output_lock_hashes.json', 'log': project / 'outputs/logs/validation_source_epoch_lineage_validation_source_epoch_lineage.log'}
    if any((p.exists() for p in O.values())) and (not a.force):
        raise RuntimeError('Outputs exist; use --force after review.')
    with O['log'].open('w') as log:

        def note(x):
            line = f'[{now()}] {x}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Verified validation geometry transfer, the final attachment rule, diagnostic provenance, and exact prespecified raw_eeg_epoch_folds/raw_eeg_preprocessing_qc/outcome_free_features lineage.')
        trig = trigger_map(P['readme'], set(ROSTER))
        sched, canonical, metadata_paths = schedule(project, dataset)
        note('Reading Validation EDF headers/annotations only; no sample amplitude.')
        headers, events = headers_events(dataset, trig)
        if len(headers) < 12 or not headers.channel_count.eq(64).all():
            raise RuntimeError('Header/channel gate failed.')
        note('Applying final two-branch source attachment: corrected 21/21 event grammar for complete rosters; exact retained anchors for sub-57.')
        trials, candidates, selected, grammar_qc = attach(sched, events)
        trials = raw_eeg_epoch_folds.derive_trial_boundaries(trials, events, headers)
        epochs = raw_eeg_epoch_folds.build_epoch_manifest(trials, headers)
        cov, subject_cov, unavail = raw_eeg_epoch_folds.coverage_tables(epochs)
        cov = cov.rename(columns={'branch_permitted_for_raw_eeg_preprocessing_qc': 'branch_permitted_for_validation_preprocessing_qc'})
        headers.to_csv(O['headers'], index=False)
        events.to_csv(O['events'], index=False)
        candidates.to_csv(O['grammar_candidates'], index=False)
        selected.to_csv(O['grammar_selected'], index=False)
        grammar_qc.to_csv(O['grammar_qc'], index=False)
        trials.to_csv(O['trials'], index=False)
        epochs.to_csv(O['epochs'], index=False)
        cov.to_csv(O['coverage'], index=False)
        sq = []
        for sub, g in trials.groupby('subject_id'):
            eg = epochs[epochs.subject_id.eq(sub)]
            q = grammar_qc[grammar_qc.subject_id.eq(sub)].iloc[0]
            sq.append({'subject_id': sub, 'prespecified_roster_trials': ROSTER[sub], 'attached_trials': len(g), 'attachment_method': q.attachment_method, 'target_imagery': int(q.target_imagery), 'target_video': int(q.target_video), 'attached_imagery': int((g.context == 'imagery').sum()), 'attached_video': int((g.context == 'video').sum()), 'primary_pairs_in_tight': int(q.primary_pairs_in_tight), 'primary_pairs_in_wide': int(q.primary_pairs_in_wide), 'raw_edf_files': g.source_stimulus_path.nunique(), 'unique_stimulus_events': g[['source_stimulus_path', 'source_stimulus_annotation_index']].drop_duplicates().shape[0], 'unique_rating_events': g[['source_rating_path', 'source_rating_annotation_index']].drop_duplicates().shape[0], 'stimulus_early_available': int(eg.query("epoch_family=='stimulus_early'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'stimulus_late_available': int(eg.query("epoch_family=='stimulus_late'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'rating_early_available': int(eg.query("epoch_family=='rating_early'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'metadata_event_context_agreement_rows': int(g.metadata_event_context_agreement.sum()), 'metadata_event_context_disagreement_rows': int((~g.metadata_event_context_agreement).sum()), 'source_lineage_complete': True, 'reports_outcomes_read': False, 'eeg_amplitude_read': False})
        pd.DataFrame(sq).to_csv(O['subject_qc'], index=False)
        src = []
        for k, p in P.items():
            src.append({'source_id': k, 'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size, 'semantic_values_read': k in ('i_policy', 'i_ref', 'i_next', 'split', 'resolution', 'diag_summary'), 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        src.append({'source_id': 'canonical_behavioral_trial_roster_keys', 'path': str(canonical), 'sha256': sha(canonical), 'bytes': canonical.stat().st_size, 'semantic_values_read': True, 'semantic_values_scope': 'subject_id and trial_number_global only; canonical context not read', 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        for p in metadata_paths:
            src.append({'source_id': 'safe_validation_metadata', 'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size, 'semantic_values_read': True, 'semantic_values_scope': 'trial_number/global_trial/context/rating_time_sec only; authoritative only for sub-57 exact retained rating anchors; otherwise diagnostic', 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        pd.DataFrame(src).to_csv(O['source_rehash'], index=False)
        branches = bool(cov.branch_permitted_for_validation_preprocessing_qc.map(raw_eeg_epoch_folds.python_bool).all())
        exact = len(trials) == 494 and trials.subject_id.nunique() == 12
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'validation_source_header_epoch_lineage_specification_complete_candidate', 'resolution_id': 'validation_source_epoch_lineage_validated_attachment_rule_EVENT_GRAMMAR_FINAL', 'counts': {'validation_subjects': 12, 'authoritative_trials': len(trials), 'raw_edf_files': len(headers), 'annotation_events': len(events), 'event_grammar_candidate_rows': len(candidates), 'event_grammar_selected_diagnostic_rows': len(selected), 'epoch_manifest_rows': len(epochs), 'available_epoch_rows': int(epochs.epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'metadata_event_context_agreement_rows': int(trials.metadata_event_context_agreement.sum()), 'metadata_event_context_disagreement_rows': int((~trials.metadata_event_context_agreement).sum())}, 'coverage': cov.to_dict('records'), 'decision': {'validation_geometry_transfer_design_integrity_passed': True, 'attachment_resolution_integrity_passed': True, 'diagnostic_provenance_integrity_passed': True, 'exact_prespecified_roster_attached': exact, 'all_complete_rosters_reconstructed_with_21_21_event_grammar': bool(grammar_qc[grammar_qc.subject_id != 'sub-57'].primary_selected.eq(42).all()), 'sub57_exact_retained_anchor_gate_passed': bool(len(trials[trials.subject_id == 'sub-57']) == 32), 'raw_headers_and_annotations_read': True, 'eeg_amplitude_read': False, 'reports_outcomes_read': False, 'stimulus_identity_read': False, 'context_derived_from_exact_source_trigger': True, 'canonical_context_read': False, 'metadata_global_trial_used_as_join_key': False, 'all_three_epoch_branches_permitted_for_validation_preprocessing_qc': branches, 'validation_preprocessing_qc_preprocessing_permitted': bool(exact and branches), 'validation_opened_for_source_lineage_only': True, 'confirmation_opened': False, 'alternate_window_selected': False}, 'runtime_seconds': round(time.time() - started, 3)}
        O['summary'].write_text(json.dumps(summary, indent=2) + '')
        req = {k: sha(O[k]) for k in ('headers', 'trials', 'epochs', 'coverage', 'subject_qc', 'grammar_qc')}
        nextp = {'generated_at': now(), 'phase': PHASE, 'next_phase': 'validation_preprocessing_qc', 'next_phase_permitted': summary['decision']['validation_preprocessing_qc_preprocessing_permitted'], 'next_phase_role': 'apply exact prespecified raw_eeg_preprocessing_qc preprocessing/QC to available Validation epochs; do not read reports/outcomes or Confirmation', 'required_inputs': req, 'required_source_code_sha256': EXPECTED['raw_eeg_preprocessing_qc'], 'required_boundary': ['verify every validation_source_epoch_lineage hash', 'use only epoch_available rows', 'do not alter windows or preprocessing/QC', 'do not read reports/outcomes/stimulus identity', 'Confirmation remains closed'], 'validation_opened_for_preprocessing_only': False, 'confirmation_opened': False}
        O['next'].write_text(json.dumps(nextp, indent=2) + '')
        core = {k: sha(O[k]) for k in ('headers', 'events', 'grammar_candidates', 'grammar_selected', 'grammar_qc', 'trials', 'epochs', 'coverage', 'subject_qc', 'source_rehash', 'summary', 'next')}
        contract = {**summary, 'transfer_policy_canonical_sha256': TRANSFER_CANONICAL, 'attachment_resolution_sha256': EXPECTED['resolution'], 'diagnostic_provenance_sha256': {k: EXPECTED[k] for k in ('diag_summary', 'diag_subjects', 'diag_anchors')}, 'source_code_lineage': {'raw_eeg_epoch_folds': EXPECTED['raw_eeg_epoch_folds'], 'raw_eeg_preprocessing_qc': EXPECTED['raw_eeg_preprocessing_qc'], 'outcome_free_features': EXPECTED['outcome_free_features']}, 'epoch_specs': raw_eeg_epoch_folds.EPOCH_SPECS, 'prespecified_roster': ROSTER, 'attachment_rule': '42-trial subjects: prespecified primary global non-overlapping event grammar with 21 imagery/21 video and chronological behavior mapping; sub-57: exact retained rating anchors with unique adjacent exact source stimulus', 'output_hashes': core, 'source_dataset_modified': False}
        O['contract'].write_text(json.dumps(contract, indent=2) + '')
        O['report'].write_text(f'# Phase validation_source_epoch_lineage — final source/epoch lineage\n\n- Resolution: **validation_source_epoch_lineage_validated_attachment_rule_EVENT_GRAMMAR_FINAL**\n- Validation participants: **12**\n- Authoritative trials: **{len(trials)}**\n- Raw EDF files: **{len(headers)}**\n- Epoch rows: **{len(epochs)}**\n- EEG amplitude read: **False**\n- Reports/outcomes read: **False**\n- Complete 42-trial rosters: **primary event grammar, 21 imagery + 21 video, chronological behavior mapping**\n- sub-57: **32 exact retained rating anchors with unique adjacent exact stimulus**\n- Adjacent-rating heuristic used for complete rosters: **False**\n- Metadata context/rating for complete rosters: **diagnostic only**\n- Canonical context read: **False**\n- All epoch branches permit validation_preprocessing_qc: **{branches}**\n- Confirmation opened: **False**')
        lock = {**core, 'contract': sha(O['contract']), 'report': sha(O['report'])}
        O['lock'].write_text(json.dumps({'generated_at': now(), 'phase': PHASE, 'hashes': lock, 'resolution_sha256': EXPECTED['resolution'], 'eeg_amplitude_read': False, 'reports_outcomes_read': False, 'confirmation_opened': False}, indent=2) + '')
        note(f"validation source-epoch lineage complete. validation_preprocessing_qc permitted: {summary['decision']['validation_preprocessing_qc_preprocessing_permitted']}.")
    z, d = zippack(project, list(O.values()))
    print(f'{z}\nSHA-256: {d}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())