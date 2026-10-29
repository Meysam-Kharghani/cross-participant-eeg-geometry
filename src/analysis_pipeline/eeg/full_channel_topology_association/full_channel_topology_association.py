# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, hashlib, json, math, os, re, warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
PHASE = 'full_channel_topology_association'
SEED = 20260812
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
D_BASE = Path('outputs/interpretability/outcome_free_family_source_decomposition_outcome_free_family_source_decomposition')
D_LOCK = D_BASE / 'qc/outcome_free_family_source_decomposition_final_output_lock_hashes.json'
cross_person_alignment_fit_LOCK = Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json')
cross_person_alignment_fit_ASSIGN = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv')
outcome_free_features_SUM = Path('outputs/qc/outcome_free_features_outcome_free_feature_summary.json')
outcome_free_features_CONTRACT = Path('config/outcome_free_features_outcome_free_feature_contract.json')
outcome_free_features_INV = Path('outputs/tables/outcome_free_features_feature_store_inventory.csv')
outcome_free_features_QC = Path('outputs/qc/outcome_free_features_feature_extraction_qc.csv')
outcome_free_features_DIM = Path('outputs/qc/outcome_free_features_feature_dimensions.csv')
outcome_free_features_ROOT = Path('outputs/large/outcome_free_features_outcome_free_features_v1')
raw_eeg_preprocessing_qc_CONTRACT = Path('config/raw_eeg_preprocessing_qc_raw_eeg_preprocessing_qc_contract.json')
raw_eeg_preprocessing_qc_REC_QC = Path('outputs/qc/raw_eeg_preprocessing_qc_recording_channel_qc.csv')
EPOCHS = ['stimulus_early', 'stimulus_late', 'rating_early']
QUANTITIES = ['early_to_late', 'late_to_rating', 'segment_cross_product']
BANDS = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma']
TF_BINS = ['whole_epoch', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
MORPH = ['winsorized_rms_uv', 'line_length_uv_per_sample', 'hjorth_mobility', 'hjorth_complexity', 'lag1_autocorrelation', 'normalized_spectral_entropy']
DCT = [f'dct_{i:02d}' for i in range(1, 21)]
BLOCKS = {'time_frequency': BANDS, 'morphology': MORPH, 'plv_channel': BANDS, 'waveform_dct': DCT}
ARRAY_FILENAMES = {'time_frequency': 'time_frequency_float32.npy', 'morphology': 'morphology_float32.npy', 'plv_channel': 'plv_channel_float32.npy', 'waveform_dct': 'waveform_dct_float32.npy', 'covariance': 'covariance_float32.npy'}
MIN_PARTICIPANTS_PER_CELL = 24
MIN_VALID_CHANNELS_PER_MAP = 48
ROBUST_IQR_EPS = 1e-12

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def truth(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def canonical_row_ids(series: pd.Series, label: str):
    if series.isna().any():
        raise RuntimeError(f'{label}: raw_behavior_row_id contains missing values')
    out = series.astype(str).str.strip()
    if out.eq('').any() or out.eq('nan').any():
        raise RuntimeError(f'{label}: raw_behavior_row_id contains empty/invalid string values')
    return out

def require(project: Path, rels: Iterable[Path]):
    miss = [str(r) for r in rels if not (project / r).is_file()]
    if miss:
        raise RuntimeError('Missing prerequisite files:\n  - ' + '\n  - '.join(miss))

def verify_lineage(project: Path):
    require(project, [D_LOCK, cross_person_alignment_fit_LOCK, cross_person_alignment_fit_ASSIGN, outcome_free_features_SUM, outcome_free_features_CONTRACT, outcome_free_features_INV, outcome_free_features_QC, outcome_free_features_DIM, raw_eeg_preprocessing_qc_CONTRACT, raw_eeg_preprocessing_qc_REC_QC])
    dl = readj(project / D_LOCK)
    if dl.get('phase') != 'outcome_free_family_source_decomposition' or dl.get('next_phase') != PHASE:
        raise RuntimeError('outcome_free_family_source_decomposition does not authorize full_channel_topology_association')
    if dl.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('outcome_free_family_source_decomposition semantic boundary drift')
    if dl.get('behavioral_report_values_read') is not False:
        raise RuntimeError('outcome_free_family_source_decomposition outcome-free boundary drift')
    if dl.get('topography_used_in_family_definition') is not False:
        raise RuntimeError('outcome_free_family_source_decomposition topography boundary drift')
    for rel, exp in dl.get('hashes', {}).items():
        p = project / rel
        if not p.is_file() or sha256(p) != exp:
            raise RuntimeError(f'outcome_free_family_source_decomposition fixed artifact mismatch: {rel}')
    d3 = readj(project / cross_person_alignment_fit_LOCK)
    if d3.get('phase') != 'cross_person_alignment_fit' or truth(d3.get('reports_outcomes_context_qc_read')):
        raise RuntimeError('cross_person_alignment_fit outcome-free family lock boundary drift')
    if sha256(project / cross_person_alignment_fit_ASSIGN) != d3['hashes']['trajectory_selected_assignments']:
        raise RuntimeError('cross_person_alignment_fit family assignment hash mismatch')
    c0s = readj(project / outcome_free_features_SUM)
    c0c = readj(project / outcome_free_features_CONTRACT)
    if c0s.get('phase') != 'outcome_free_features' or c0c.get('phase') != 'outcome_free_features':
        raise RuntimeError('outcome_free_features phase mismatch')
    for key in ['all_feature_checkpoints_complete', 'all_modeling_eligible_epochs_represented', 'rejected_channel_nan_policy_passed']:
        if c0s['decision'].get(key) is not True:
            raise RuntimeError(f'outcome_free_features prerequisite failed: {key}')
    for key in ['outcome_values_read', 'category_values_read', 'validation_opened', 'confirmation_opened']:
        if c0s['decision'].get(key) is not False:
            raise RuntimeError(f'outcome_free_features outcome/heldout boundary failed: {key}')
    for key, rel in [('feature_inventory', outcome_free_features_INV), ('feature_qc', outcome_free_features_QC), ('feature_dimensions', outcome_free_features_DIM), ('summary', outcome_free_features_SUM)]:
        exp = c0c['output_hashes'].get(key)
        if exp is None or sha256(project / rel) != exp:
            raise RuntimeError(f'outcome_free_features output hash mismatch: {key}')
    b = readj(project / raw_eeg_preprocessing_qc_CONTRACT)
    exp = b.get('output_hashes', {}).get('recording_qc')
    if exp is None or sha256(project / raw_eeg_preprocessing_qc_REC_QC) != exp:
        raise RuntimeError('raw_eeg_preprocessing_qc recording-channel QC hash mismatch')
    return (dl, d3, c0s, c0c, b)

def normalize_channel_name(name: str):
    x = str(name).upper().strip()
    x = re.sub('[\\s_.-]+', '', x)
    x = re.sub('(REF|LE|EEG)$', '', x)
    return x

def coarse_region(ch: str):
    x = normalize_channel_name(ch)
    if x.startswith(('FP', 'AF')):
        return 'frontopolar'
    if x.startswith(('FT', 'TP')) or re.match('^T\\d', x):
        return 'temporal'
    if x.startswith('FC'):
        return 'frontocentral'
    if x.startswith('CP'):
        return 'centroparietal'
    if x.startswith('PO'):
        return 'parieto_occipital'
    if x.startswith('F'):
        return 'frontal'
    if x.startswith('C'):
        return 'central'
    if x.startswith('P'):
        return 'parietal'
    if x.startswith(('O', 'I')):
        return 'occipital'
    return 'other'

def resolve_source_dir(project: Path, subject_id: str, stored: str):
    p = Path(str(stored))
    if p.is_dir():
        return p
    fallback = project / outcome_free_features_ROOT / str(subject_id) / p.name
    if fallback.is_dir():
        return fallback
    raise RuntimeError(f'Feature source directory not found: {stored} or {fallback}')

def build_source_registry(project: Path, inv: pd.DataFrame, recqc: pd.DataFrame, discovery_subjects: set[str]):
    used = inv[inv.subject_id.astype(str).isin(discovery_subjects)].copy()
    if used.empty:
        raise RuntimeError('No outcome_free_features feature-store rows for Discovery subjects')
    source_rows = []
    source_channel_maps = {}
    rehash_rows = []
    for (sid, stored), q in used.groupby(['subject_id', 'source_feature_directory'], sort=True):
        sdir = resolve_source_dir(project, str(sid), str(stored))
        raw_paths = q.raw_source_path.astype(str).unique().tolist()
        if len(raw_paths) != 1:
            raise RuntimeError(f'{sid} {sdir}: nonunique raw source identity')
        raw = raw_paths[0]
        cr = recqc[(recqc.subject_id.astype(str) == str(sid)) & (recqc.raw_source_path.astype(str) == raw)].copy()
        if len(cr) != 64:
            raise RuntimeError(f'{sid} {raw}: expected 64 recording-channel rows, got {len(cr)}')
        cr = cr.sort_values('channel_position_1based')
        if cr.channel_position_1based.astype(int).tolist() != list(range(1, 65)):
            raise RuntimeError(f'{sid} {raw}: channel position roster drift')
        names = [normalize_channel_name(x) for x in cr.channel_name.astype(str)]
        if len(set(names)) != 64:
            raise RuntimeError(f'{sid} {raw}: normalized channel names are not unique')
        source_channel_maps[str(sdir)] = names
        source_rows.append({'subject_id': str(sid), 'raw_source_path': raw, 'source_dir': str(sdir), 'raw_run': int(q.raw_run.iloc[0]), 'channel_name_sha256': hashlib.sha256('|'.join(names).encode()).hexdigest()})
    sets = [set(x) for x in source_channel_maps.values()]
    common = set.intersection(*sets)
    union = set.union(*sets)
    if len(common) != 64 or len(union) != 64:
        raise RuntimeError(f'Topographic channel roster is not identical across used sources: intersection={len(common)}, union={len(union)}')
    canonical = sorted(common)
    return (pd.DataFrame(source_rows), source_channel_maps, canonical)

def inventory_hash_map(inv: pd.DataFrame, source_dir: Path):
    q = inv[inv.source_feature_directory.astype(str).map(lambda x: Path(x).name == source_dir.name)]
    q2 = q[inv.loc[q.index, 'source_feature_directory'].astype(str).eq(str(source_dir))]
    if not q2.empty:
        q = q2
    out = {}
    for _, r in q.iterrows():
        out[str(r.relative_path)] = (str(r.sha256), int(r.size_bytes))
    return out

def verify_file_against_inventory(path: Path, rel: str, hm: dict, rehash_rows: list):
    if rel not in hm:
        raise RuntimeError(f'Feature-store inventory does not contain {rel} under {path.parent}')
    exp, size = hm[rel]
    if not path.is_file():
        raise RuntimeError(f'Missing feature-store file: {path}')
    got_size = path.stat().st_size
    got = sha256(path)
    ok = got == exp and got_size == size
    rehash_rows.append({'path': str(path), 'relative_path': rel, 'expected_sha256': exp, 'observed_sha256': got, 'expected_size_bytes': size, 'observed_size_bytes': got_size, 'passed': ok})
    if not ok:
        raise RuntimeError(f'Feature-store integrity mismatch: {path}')

def build_locator(project: Path, inv: pd.DataFrame, fam: pd.DataFrame, registry: pd.DataFrame):
    target = fam[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based']].copy()
    target['subject_id'] = target.subject_id.astype(str)
    target['raw_behavior_row_id'] = canonical_row_ids(target.raw_behavior_row_id, 'family target')
    wanted = set(target.raw_behavior_row_id)
    chunks = []
    rehash = []
    for _, sr in registry.iterrows():
        sdir = Path(sr.source_dir)
        qinv = inv[(inv.subject_id.astype(str) == str(sr.subject_id)) & inv.source_feature_directory.astype(str).map(lambda x: Path(x).name == sdir.name)]
        hm = {str(r.relative_path): (str(r.sha256), int(r.size_bytes)) for _, r in qinv.iterrows()}
        for ep in EPOCHS:
            rel = f'{ep}_feature_index.csv'
            p = sdir / rel
            verify_file_against_inventory(p, rel, hm, rehash)
            ix = pd.read_csv(p, low_memory=False)
            ix['raw_behavior_row_id'] = canonical_row_ids(ix.raw_behavior_row_id, f'{sdir.name} {ep} feature index')
            ix = ix[ix.raw_behavior_row_id.isin(wanted)].copy()
            if ix.empty:
                continue
            ix['source_dir'] = str(sdir)
            ix['epoch_family'] = ep
            chunks.append(ix[['subject_id', 'raw_behavior_row_id', 'epoch_family', 'feature_row_0based', 'source_dir', 'raw_source_path']])
    loc = pd.concat(chunks, ignore_index=True)
    loc['subject_id'] = loc.subject_id.astype(str)
    loc['raw_behavior_row_id'] = canonical_row_ids(loc.raw_behavior_row_id, 'full-channel locator')
    if loc.duplicated(['subject_id', 'raw_behavior_row_id', 'epoch_family']).any():
        raise RuntimeError('Duplicate trial/epoch feature locator')
    counts = loc.groupby(['subject_id', 'raw_behavior_row_id']).epoch_family.nunique()
    if len(counts) != len(target) or not counts.eq(3).all():
        raise RuntimeError(f'Full-channel locator incomplete: complete={int(counts.eq(3).sum())}/{len(target)}')
    loc = loc.merge(target, on=['subject_id', 'raw_behavior_row_id'], how='left', validate='many_to_one')
    return (loc, pd.DataFrame(rehash))

def robust_spatial_normalize(x: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(x, axis=1, keepdims=True)
        q25 = np.nanquantile(x, 0.25, axis=1, keepdims=True)
        q75 = np.nanquantile(x, 0.75, axis=1, keepdims=True)
    iqr = q75 - q25
    nvalid = np.sum(np.isfinite(x), axis=1, keepdims=True)
    good = np.isfinite(iqr) & (iqr > ROBUST_IQR_EPS) & (nvalid >= MIN_VALID_CHANNELS_PER_MAP)
    z = np.divide(x - med, iqr, out=np.full_like(x, np.nan, dtype=float), where=good)
    return z

def participant_contrasts(d: pd.DataFrame, values: np.ndarray):
    plist = sorted(d.subject_id.astype(str).unique())
    out = np.full((len(plist), values.shape[1]), np.nan, float)
    sidarr = d.subject_id.astype(str).to_numpy()
    fam = d.trajectory_family_1based.astype(int).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for pi, sid in enumerate(plist):
            idx = np.where(sidarr == sid)[0]
            a = values[idx][fam[idx] == 1]
            b = values[idx][fam[idx] == 2]
            if len(a) == 0 or len(b) == 0:
                continue
            out[pi] = np.nanmean(b, axis=0) - np.nanmean(a, axis=0)
    return (plist, out)

def signflip_null_stats(C: np.ndarray, signs: np.ndarray):
    counts = np.sum(np.isfinite(C), axis=0)
    valid = counts >= MIN_PARTICIPANTS_PER_CELL
    if not valid.any():
        raise RuntimeError('No cells satisfy participant-coverage threshold')
    X = np.where(np.isfinite(C[:, valid]), C[:, valid], 0.0)
    den = counts[valid].astype(float)
    obs_mean = X.sum(axis=0) / den
    obs = float(np.mean(obs_mean ** 2))
    null = np.empty(signs.shape[0], float)
    chunk = 250
    for start in range(0, len(signs), chunk):
        S = signs[start:start + chunk].astype(float)
        means = S @ X / den[None, :]
        null[start:start + len(S)] = np.mean(means ** 2, axis=1)
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return (obs, p, null, valid, obs_mean)

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out

def maxT_localization(C: np.ndarray, groups: list[np.ndarray], signs: np.ndarray):
    counts = np.sum(np.isfinite(C), axis=0)
    basevalid = counts >= MIN_PARTICIPANTS_PER_CELL
    X = np.where(np.isfinite(C), C, 0.0)
    den = np.maximum(counts, 1).astype(float)
    gobs = X.sum(axis=0) / den
    obs = []
    usable = []
    for g in groups:
        g = np.asarray(g, int)
        v = g[basevalid[g]]
        usable.append(v)
        obs.append(float(np.mean(gobs[v] ** 2)) if len(v) else np.nan)
    obs = np.asarray(obs, float)
    nullmax = np.empty(len(signs), float)
    ge = np.zeros(len(groups), int)
    chunk = 200
    for start in range(0, len(signs), chunk):
        S = signs[start:start + chunk].astype(float)
        means = S @ X / den[None, :]
        stats = np.full((len(S), len(groups)), np.nan, float)
        for gi, v in enumerate(usable):
            if len(v):
                stats[:, gi] = np.mean(means[:, v] ** 2, axis=1)
        mx = np.nanmax(stats, axis=1)
        nullmax[start:start + len(S)] = mx
        for gi in range(len(groups)):
            if np.isfinite(obs[gi]):
                ge[gi] += int(np.sum(mx >= obs[gi]))
    pmax = np.where(np.isfinite(obs), (1 + ge) / (len(signs) + 1), np.nan)
    return (obs, pmax)

def load_channel_block(project: Path, inv: pd.DataFrame, loc: pd.DataFrame, channel_maps: dict, canonical: list[str], block: str, rehash_rows: list):
    k = {'time_frequency': 30, 'morphology': 6, 'plv_channel': 6, 'waveform_dct': 20}[block]
    n = loc[['subject_id', 'raw_behavior_row_id']].drop_duplicates().shape[0]
    trialkeys = loc[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based']].drop_duplicates().sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    trialkeys['trial_row'] = np.arange(len(trialkeys))
    keymap = {(str(r.subject_id), str(r.raw_behavior_row_id)): int(r.trial_row) for r in trialkeys.itertuples()}
    epdata = {ep: np.full((len(trialkeys), 64, k), np.nan, np.float32) for ep in EPOCHS}
    for (sdir, ep), q in loc.groupby(['source_dir', 'epoch_family'], sort=True):
        sdir = Path(sdir)
        q = q.copy()
        q['raw_behavior_row_id'] = canonical_row_ids(q.raw_behavior_row_id, f'{sdir.name} {ep} locator')
        q['trial_row'] = [keymap[str(s), str(r)] for s, r in zip(q.subject_id, q.raw_behavior_row_id)]
        q = q.sort_values('feature_row_0based')
        qinv = inv[inv.subject_id.astype(str).isin(q.subject_id.astype(str).unique()) & inv.source_feature_directory.astype(str).map(lambda x: Path(x).name == sdir.name)]
        hm = {str(r.relative_path): (str(r.sha256), int(r.size_bytes)) for _, r in qinv.iterrows()}
        rel = f'{ep}/{ARRAY_FILENAMES[block]}'
        p = sdir / rel
        verify_file_against_inventory(p, rel, hm, rehash_rows)
        mm = np.load(p, mmap_mode='r', allow_pickle=False)
        names = channel_maps[str(sdir)]
        pos = {name: i for i, name in enumerate(names)}
        reorder = [pos[c] for c in canonical]
        rows = q.feature_row_0based.astype(int).to_numpy()
        vals = np.asarray(mm[rows][:, reorder, ...], dtype=np.float32)
        vals = vals.reshape(len(q), 64, k)
        epdata[ep][q.trial_row.astype(int).to_numpy()] = vals
        del mm
    return (trialkeys, epdata)

def subfeature_labels(block):
    if block == 'time_frequency':
        return [f'{b}|{tb}' for b in BANDS for tb in TF_BINS]
    if block == 'morphology':
        return MORPH
    if block == 'plv_channel':
        return BANDS
    if block == 'waveform_dct':
        return DCT
    raise KeyError(block)

def channel_parent_analysis(block, trialkeys, epdata, canonical, signs, parent_rows, channel_rows, sub_rows, effect_rows):
    z = {ep: robust_spatial_normalize(epdata[ep].astype(float)) for ep in EPOCHS}
    vals = {'early_to_late': z['stimulus_late'] - z['stimulus_early'], 'late_to_rating': z['rating_early'] - z['stimulus_late']}
    vals['segment_cross_product'] = vals['early_to_late'] * vals['late_to_rating']
    labels = subfeature_labels(block)
    nsub = len(labels)
    ch_groups = [np.arange(ci * nsub, (ci + 1) * nsub, dtype=int) for ci in range(64)]
    sf_groups = [np.arange(si, 64 * nsub, nsub, dtype=int) for si in range(nsub)]
    store = {}
    for qty, V in vals.items():
        flat = V.reshape(len(trialkeys), -1)
        plist, C = participant_contrasts(trialkeys, flat)
        obs, p, null, valid, gmean = signflip_null_stats(C, signs)
        key = f'{block}|{qty}'
        store[key] = (C, null, valid, gmean)
        parent_rows.append({'parent_test': key, 'domain': 'channel_topography', 'feature_block': block, 'quantity': qty, 'n_participants': len(plist), 'n_cells_total': flat.shape[1], 'n_cells_valid': int(valid.sum()), 'observed_multivariate_effect_energy': obs, 'raw_signflip_p': p, 'inferential_role': 'exploratory_conditional_same_dataset_fixed_family_association'})
        chobs, chp = maxT_localization(C, ch_groups, signs)
        for ci, ch in enumerate(canonical):
            channel_rows.append({'parent_test': key, 'channel_name': ch, 'region': coarse_region(ch), 'channel_multivariate_effect_energy': chobs[ci], 'maxT_p_within_parent': chp[ci]})
        sfobs, sfp = maxT_localization(C, sf_groups, signs)
        for si, label in enumerate(labels):
            sub_rows.append({'parent_test': key, 'subfeature': label, 'subfeature_multivariate_effect_energy': sfobs[si], 'maxT_p_within_parent': sfp[si]})
        gfull = np.full(flat.shape[1], np.nan)
        gfull[valid] = gmean
        G = gfull.reshape(64, nsub)
        finite_count = np.sum(np.isfinite(C), axis=0).reshape(64, nsub)
        signcons = np.full((64, nsub), np.nan)
        for ci in range(64):
            for si in range(nsub):
                j = ci * nsub + si
                v = C[:, j]
                v = v[np.isfinite(v)]
                if len(v):
                    s = np.sign(G[ci, si])
                    signcons[ci, si] = np.mean(np.sign(v) == s) if s != 0 else np.nan
                effect_rows.append({'parent_test': key, 'channel_name': canonical[ci], 'region': coarse_region(canonical[ci]), 'subfeature': labels[si], 'family2_minus_family1_participant_mean': G[ci, si], 'participants_finite': int(finite_count[ci, si]), 'participant_sign_consistency': signcons[ci, si]})
    return store

def load_covariance(project: Path, inv: pd.DataFrame, loc: pd.DataFrame, channel_maps: dict, canonical: list[str], rehash_rows: list):
    trialkeys = loc[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based']].drop_duplicates().sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    trialkeys['trial_row'] = np.arange(len(trialkeys))
    keymap = {(str(r.subject_id), str(r.raw_behavior_row_id)): int(r.trial_row) for r in trialkeys.itertuples()}
    epdata = {ep: np.full((len(trialkeys), 64, 64), np.nan, np.float32) for ep in EPOCHS}
    for (sdir, ep), q in loc.groupby(['source_dir', 'epoch_family'], sort=True):
        sdir = Path(sdir)
        q = q.copy()
        q['raw_behavior_row_id'] = canonical_row_ids(q.raw_behavior_row_id, f'{sdir.name} {ep} locator')
        q['trial_row'] = [keymap[str(s), str(r)] for s, r in zip(q.subject_id, q.raw_behavior_row_id)]
        q = q.sort_values('feature_row_0based')
        qinv = inv[inv.subject_id.astype(str).isin(q.subject_id.astype(str).unique()) & inv.source_feature_directory.astype(str).map(lambda x: Path(x).name == sdir.name)]
        hm = {str(r.relative_path): (str(r.sha256), int(r.size_bytes)) for _, r in qinv.iterrows()}
        rel = f"{ep}/{ARRAY_FILENAMES['covariance']}"
        p = sdir / rel
        verify_file_against_inventory(p, rel, hm, rehash_rows)
        mm = np.load(p, mmap_mode='r', allow_pickle=False)
        names = channel_maps[str(sdir)]
        pos = {name: i for i, name in enumerate(names)}
        rr = [pos[c] for c in canonical]
        rows = q.feature_row_0based.astype(int).to_numpy()
        vals = np.asarray(mm[rows][:, rr, :][:, :, rr], dtype=np.float32)
        epdata[ep][q.trial_row.astype(int).to_numpy()] = vals
        del mm
    return (trialkeys, epdata)

def cov_to_fisher_edges(cov: np.ndarray):
    diag = np.diagonal(cov, axis1=1, axis2=2)
    den = np.sqrt(diag[:, :, None] * diag[:, None, :])
    corr = np.divide(cov, den, out=np.full_like(cov, np.nan, dtype=float), where=np.isfinite(den) & (den > 0))
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    iu = np.triu_indices(64, k=1)
    return (z[:, iu[0], iu[1]], iu)

def covariance_analysis(trialkeys, epcov, canonical, signs, parent_rows, region_rows, edge_rows):
    edges = {}
    iu = None
    for ep in EPOCHS:
        edges[ep], iu = cov_to_fisher_edges(epcov[ep].astype(float))
    vals = {'early_to_late': edges['stimulus_late'] - edges['stimulus_early'], 'late_to_rating': edges['rating_early'] - edges['stimulus_late']}
    vals['segment_cross_product'] = vals['early_to_late'] * vals['late_to_rating']
    regions = [coarse_region(c) for c in canonical]
    pair_names = []
    pair_to_edges = {}
    for ei, (a, b) in enumerate(zip(iu[0], iu[1])):
        rp = '--'.join(sorted([regions[a], regions[b]]))
        pair_to_edges.setdefault(rp, []).append(ei)
    pair_names = sorted(pair_to_edges)
    groups = [np.asarray(pair_to_edges[p], int) for p in pair_names]
    for qty, V in vals.items():
        plist, C = participant_contrasts(trialkeys, V)
        obs, p, null, valid, gmean = signflip_null_stats(C, signs)
        key = f'covariance_topology|{qty}'
        parent_rows.append({'parent_test': key, 'domain': 'covariance_topology', 'feature_block': 'covariance_correlation', 'quantity': qty, 'n_participants': len(plist), 'n_cells_total': V.shape[1], 'n_cells_valid': int(valid.sum()), 'observed_multivariate_effect_energy': obs, 'raw_signflip_p': p, 'inferential_role': 'exploratory_conditional_same_dataset_fixed_family_association'})
        robs, rp = maxT_localization(C, groups, signs)
        for name, o, pp, gg in zip(pair_names, robs, rp, groups):
            region_rows.append({'parent_test': key, 'region_pair': name, 'n_edges': len(gg), 'region_pair_effect_energy': o, 'maxT_p_within_parent': pp})
        gfull = np.full(V.shape[1], np.nan)
        gfull[valid] = gmean
        order = np.argsort(np.nan_to_num(np.abs(gfull), nan=-1))[::-1][:100]
        for rank, ei in enumerate(order, 1):
            a, b = (iu[0][ei], iu[1][ei])
            vv = C[:, ei]
            vv = vv[np.isfinite(vv)]
            gm = gfull[ei]
            edge_rows.append({'parent_test': key, 'rank_abs_effect': rank, 'channel_a': canonical[a], 'channel_b': canonical[b], 'region_a': regions[a], 'region_b': regions[b], 'family2_minus_family1_participant_mean_fisher_z_delta_or_product': gm, 'participants_finite': len(vv), 'participant_sign_consistency': float(np.mean(np.sign(vv) == np.sign(gm))) if len(vv) and gm != 0 else np.nan, 'role': 'descriptive_top_edge_not_independently_selected_hypothesis'})

def self_test():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(10, 64, 3))
    z1 = robust_spatial_normalize(x)
    z2 = robust_spatial_normalize(4 * x + 11)
    if not np.nanmax(np.abs(z1 - z2)) < 1e-10:
        raise RuntimeError('Robust spatial normalization is not affine-invariant')
    C = rng.normal(size=(31, 40))
    signs = rng.choice([-1, 1], size=(200, 31))
    obs, p, null, valid, g = signflip_null_stats(C, signs)
    if not (np.isfinite(obs) and 0 < p <= 1 and (len(null) == 200)):
        raise RuntimeError('Sign-flip self-test failed')
    if coarse_region('Fp1') != 'frontopolar' or coarse_region('FCz') != 'frontocentral':
        raise RuntimeError('Region parser self-test failed')
    rid = canonical_row_ids(pd.Series(['sub-02|trial1', 'sub-02|trial2']), 'self-test')
    if rid.tolist() != ['sub-02|trial1', 'sub-02|trial2']:
        raise RuntimeError('String raw_behavior_row_id regression self-test failed')
    keymap = {('sub-02', rid.iloc[0]): 7}
    if keymap['sub-02', 'sub-02|trial1'] != 7:
        raise RuntimeError('String trial-key regression self-test failed')
    print('full_channel_topology_association self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--signflip-permutations', type=int, default=10000)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    verify_lineage(project)
    out = project / 'outputs/interpretability/full_channel_topology_association_full_channel_topology_association'
    if out.exists():
        existing_final = out / 'qc/full_channel_topology_association_final_output_lock_hashes.json'
        if existing_final.is_file() and (not a.force):
            raise RuntimeError(f'Completed output already exists: {out}; use --force only after deliberate review')
        import shutil
        if not existing_final.is_file():
            print(f'[{now()}] Removing incomplete full_channel_topology_association output from a prior failed run: {out}', flush=True)
        shutil.rmtree(out)
    (out / 'tables').mkdir(parents=True)
    (out / 'qc').mkdir()
    (out / 'reports').mkdir()
    fam = pd.read_csv(project / cross_person_alignment_fit_ASSIGN, low_memory=False)
    fam = fam[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based']].copy()
    fam['subject_id'] = fam.subject_id.astype(str)
    fam['raw_behavior_row_id'] = canonical_row_ids(fam.raw_behavior_row_id, 'cross_person_alignment_fit family assignments')
    if len(fam) != 1237 or fam.subject_id.nunique() != 31 or set(fam.trajectory_family_1based.astype(int)) != {1, 2}:
        raise RuntimeError('cross_person_alignment_fit Discovery family roster drift')
    discovery = set(fam.subject_id)
    inv = pd.read_csv(project / outcome_free_features_INV, low_memory=False)
    recqc = pd.read_csv(project / raw_eeg_preprocessing_qc_REC_QC, low_memory=False)
    registry, channel_maps, canonical = build_source_registry(project, inv, recqc, discovery)
    loc, rehash_idx = build_locator(project, inv, fam, registry)
    roster = pd.DataFrame({'channel_name': canonical, 'channel_1based': np.arange(1, 65), 'coarse_region': [coarse_region(c) for c in canonical]})
    roster.to_csv(out / 'tables/full_channel_topology_association_channel_roster.csv', index=False)
    registry.to_csv(out / 'qc/full_channel_topology_association_source_channel_registry.csv', index=False)
    locq = loc.groupby(['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based'], as_index=False).agg(epoch_families=('epoch_family', 'nunique'), source_directories=('source_dir', 'nunique'))
    locq.to_csv(out / 'qc/full_channel_topology_association_trial_epoch_locator_qc.csv', index=False)
    rng = np.random.default_rng(SEED)
    signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(a.signflip_permutations, 31), replace=True)
    parent_rows = []
    channel_rows = []
    sub_rows = []
    effect_rows = []
    region_rows = []
    edge_rows = []
    rehash_rows = rehash_idx.to_dict(orient='records')
    canonical_trialkeys = None
    for block in ['time_frequency', 'morphology', 'plv_channel', 'waveform_dct']:
        trialkeys, epdata = load_channel_block(project, inv, loc, channel_maps, canonical, block, rehash_rows)
        if canonical_trialkeys is None:
            canonical_trialkeys = trialkeys.copy()
        elif not trialkeys[['subject_id', 'raw_behavior_row_id']].equals(canonical_trialkeys[['subject_id', 'raw_behavior_row_id']]):
            raise RuntimeError(f'Trial order drift while loading {block}')
        channel_parent_analysis(block, trialkeys, epdata, canonical, signs, parent_rows, channel_rows, sub_rows, effect_rows)
        del epdata
    trialkeys, epcov = load_covariance(project, inv, loc, channel_maps, canonical, rehash_rows)
    if not trialkeys[['subject_id', 'raw_behavior_row_id']].equals(canonical_trialkeys[['subject_id', 'raw_behavior_row_id']]):
        raise RuntimeError('Trial order drift while loading covariance')
    covariance_analysis(trialkeys, epcov, canonical, signs, parent_rows, region_rows, edge_rows)
    del epcov
    parents = pd.DataFrame(parent_rows)
    parents['holm_p_across_15_parents'] = holm(parents.raw_signflip_p.to_numpy(float))
    parents['exploratory_parent_supported'] = parents.holm_p_across_15_parents.le(0.05)
    parents['independent_confirmation'] = False
    parents['semantic_relabeling_permitted'] = False
    parents.to_csv(out / 'tables/full_channel_topology_association_global_topology_association_tests.csv', index=False)
    support = {r.parent_test: bool(r.exploratory_parent_supported) for r in parents.itertuples()}
    ch = pd.DataFrame(channel_rows)
    if len(ch):
        ch['parent_Holm_supported'] = ch.parent_test.map(support)
        ch['localization_gate_pass'] = ch.parent_Holm_supported & ch.maxT_p_within_parent.le(0.05)
        ch['interpretation_role'] = np.where(ch.parent_Holm_supported, 'gated_exploratory_localization', 'descriptive_only_parent_not_supported')
        ch.to_csv(out / 'tables/full_channel_topology_association_channel_localization.csv', index=False)
    sf = pd.DataFrame(sub_rows)
    if len(sf):
        sf['parent_Holm_supported'] = sf.parent_test.map(support)
        sf['localization_gate_pass'] = sf.parent_Holm_supported & sf.maxT_p_within_parent.le(0.05)
        sf['interpretation_role'] = np.where(sf.parent_Holm_supported, 'gated_exploratory_localization', 'descriptive_only_parent_not_supported')
        sf.to_csv(out / 'tables/full_channel_topology_association_subfeature_localization.csv', index=False)
    eff = pd.DataFrame(effect_rows)
    if len(eff):
        eff['parent_Holm_supported'] = eff.parent_test.map(support)
        eff.to_csv(out / 'tables/full_channel_topology_association_signed_channel_subfeature_effect_atlas.csv', index=False)
    rr = pd.DataFrame(region_rows)
    if len(rr):
        rr['parent_Holm_supported'] = rr.parent_test.map(support)
        rr['localization_gate_pass'] = rr.parent_Holm_supported & rr.maxT_p_within_parent.le(0.05)
        rr['interpretation_role'] = np.where(rr.parent_Holm_supported, 'gated_exploratory_network_localization', 'descriptive_only_parent_not_supported')
        rr.to_csv(out / 'tables/full_channel_topology_association_covariance_region_pair_localization.csv', index=False)
    ee = pd.DataFrame(edge_rows)
    if len(ee):
        ee['parent_Holm_supported'] = ee.parent_test.map(support)
        ee.to_csv(out / 'tables/full_channel_topology_association_covariance_top_edges.csv', index=False)
    rh = pd.DataFrame(rehash_rows).drop_duplicates(['path', 'relative_path'])
    if not rh.passed.all():
        raise RuntimeError('A feature-store rehash audit row failed')
    rh.to_csv(out / 'qc/full_channel_topology_association_opened_feature_store_rehash_audit.csv', index=False)
    supported = parents.loc[parents.exploratory_parent_supported, 'parent_test'].tolist()
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'full_channel_topology_exploratory_association_complete', 'counts': {'Discovery_trials': int(len(fam)), 'Discovery_participants': int(fam.subject_id.nunique()), 'canonical_channels': len(canonical), 'source_directories_used': int(registry.source_dir.nunique()), 'global_parent_tests': int(len(parents)), 'global_parent_tests_Holm_supported': int(len(supported)), 'feature_store_files_rehashed_before_open': int(len(rh))}, 'supported_exploratory_parents': supported, 'scientific_boundaries': {'family_labels_fixed_before_full_channel_topology_association': True, 'behavioral_report_values_read': False, 'emotion_category_values_read': False, 'Validation_opened': False, 'Confirmation_opened': False, 'new_family_discovery': False, 'semantic_relabeling_permitted': False, 'independent_confirmation': False, 'why_not_independent': 'The full-channel maps derive from the same EEG epochs used to construct the compact family geometry. Spatial median/IQR normalization removes the exact channel-summary location/scale used in the family-defining compact blocks, but the analysis remains same-dataset exploratory attribution/association.', 'topography_role': 'exploratory association with already-fixed families; topography was not used to define K=2'}, 'next_phase': 'heldout_topography_transfer_design', 'next_phase_role': 'Synthesis of family neurophysiology and interpretation boundaries; if a spatial/topological parent survives Holm, design a fixed held-out topographic transfer using existing Validation/Confirmation EEG features without reopening behavioral semantics.'}
    writej(out / 'qc/full_channel_topology_association_summary.json', summary)
    show = parents.sort_values(['holm_p_across_15_parents', 'raw_signflip_p']).head(15)
    lines = ['# full_channel_topology_association — Full-channel / covariance topology association with fixed EEG trajectory families', '', f"Status: **{summary['status']}**", '', 'This phase is outcome-free and uses only the fixed Discovery F1/F2 assignments plus the previously prespecified full-signal EEG feature store.', '', '## Anti-circularity step', 'For channel-wise TF, morphology, channel-PLV and DCT maps, every trial/epoch/subfeature map is robustly standardized across channels by its own channel median and IQR before family comparison. This removes the exact global channel median/IQR location/scale summaries used in the 169-cell family-defining representation. It does not make the phase independent evidence; the same EEG epochs are still being interrogated.', '', 'The full 64x64 covariance matrices are converted to Fisher-z correlation topology. The compact family representation used covariance summaries, not individual correlation edges; edge/region localization is therefore new spatial detail but remains exploratory same-dataset analysis.', '', '## Global parents (Holm across 15)', '', '| Parent | Raw p | Holm p | Exploratory support |', '|---|---:|---:|---|']
    for r in show.itertuples():
        lines.append(f'| {r.parent_test} | {r.raw_signflip_p:.6g} | {r.holm_p_across_15_parents:.6g} | {bool(r.exploratory_parent_supported)} |')
    lines += ['', f'Holm-supported exploratory parents: **{len(supported)}/15**.', '', '## Interpretation boundary', 'Even a supported topographic or covariance result cannot rename F1/F2 as a psychological state. It can only describe where/frequency/feature topology the already-fixed temporal reconfiguration is expressed.', '', 'Validation and Confirmation remain unopened in full_channel_topology_association.']
    (out / 'reports/full_channel_topology_association_report.md').write_text(''.join(lines) + '', encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'role': 'outcome-free exploratory full-channel and covariance-topology association with fixed Discovery F1/F2', 'primary_family_definition_reused': True, 'spatial_normalization': 'per-trial per-epoch per-subfeature channel median/IQR', 'channelwise_quantities': QUANTITIES, 'covariance_transform': 'covariance -> correlation -> clip +/-0.999999 -> Fisher z', 'global_test': 'participant-balanced F2-F1 contrast vector; 10,000 participant sign flips by default; mean squared group-effect vector', 'multiplicity': 'Holm across 15 global parent tests; maxT within supported parent for channels/subfeatures/region-pairs', 'minimum_participants_per_cell': MIN_PARTICIPANTS_PER_CELL, 'minimum_valid_channels_per_map': MIN_VALID_CHANNELS_PER_MAP, 'behavioral_report_values_read': False, 'Validation_opened': False, 'Confirmation_opened': False, 'semantic_relabeling_permitted': False, 'independent_confirmation': False, 'next_phase': 'heldout_topography_transfer_design'}
    writej(out / 'qc/full_channel_topology_association_contract.json', contract)
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    hashes = {str(p.relative_to(project)): sha256(p) for p in files}
    final = out / 'qc/full_channel_topology_association_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': hashes, 'global_parent_tests': 15, 'global_parent_tests_Holm_supported': len(supported), 'behavioral_report_values_read': False, 'Validation_opened': False, 'Confirmation_opened': False, 'semantic_relabeling_permitted': False, 'independent_confirmation': False, 'next_phase': 'heldout_topography_transfer_design'})
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    hz = project / '' / f'full_channel_topology_association_full_channel_topology_association_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    h = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{h}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] full_channel_topology_association PASS')
    print(f'Discovery trials={len(fam)} participants={fam.subject_id.nunique()} channels={len(canonical)}')
    print(f'Global Holm-supported exploratory parents: {len(supported)}/15')
    if supported:
        print('Supported:', '; '.join(supported))
    print('Behavioral reports opened: False')
    print('Validation/Confirmation opened: False')
    print('Semantic relabeling permitted: False')
    print('Next phase: heldout_topography_transfer_design')
    print('', hz)
    print('SHA-256:', h)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())