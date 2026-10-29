# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'validation_full_channel_transfer'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
STORE_REL = Path('outputs/large/validation_full_channel_transfer_validation_full_channel_transfer_v1')
heldout_topography_transfer_design_BASE = Path('outputs/interpretability/heldout_topography_transfer_design_heldout_topography_transfer_design_specification')
heldout_topography_transfer_design_LOCK = heldout_topography_transfer_design_BASE / 'qc/heldout_topography_transfer_design_final_output_lock_hashes.json'
heldout_topography_transfer_design_LOCK_SHA = 'a5590556ee9e4a32ff709a2abb3e19cbee45a698fb85f440f0ae5cf768abbbf0'
heldout_topography_transfer_design_POLICY = heldout_topography_transfer_design_BASE / 'qc/heldout_topography_transfer_design_prespecified_transfer_policy.json'
validation_preprocessing_qc_LOCK = Path('outputs/qc/validation_preprocessing_qc_final_output_lock_hashes.json')
validation_preprocessing_qc_LOCK_SHA = '984aba20ce015884184a037dba87b7ed4b4b8228610b479b5a881090a9c6f460'
validation_preprocessing_qc_EPOCH = Path('outputs/tables/validation_preprocessing_qc_validation_epoch_amplitude_qc.csv')
validation_preprocessing_qc_INV = Path('outputs/tables/validation_preprocessing_qc_validation_large_array_inventory.csv')
validation_preprocessing_qc_CH = Path('outputs/qc/validation_preprocessing_qc_validation_recording_channel_qc.csv')
validation_fixed_assignment_LOCK = Path('outputs/qc/validation_fixed_assignment_final_output_lock_hashes.json')
validation_fixed_assignment_LOCK_SHA = 'b6ea679c97243167292bf2e167883b17b067bb4b220d38ed0e1c0e35fd783450'
validation_fixed_assignment_ASSIGN = Path('outputs/tables/validation_fixed_assignment_validation_trajectory_assignments.csv')
outcome_free_features_SCRIPT = Path('src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py')
outcome_free_features_SCRIPT_SHA = 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba'
EPOCHS = ['stimulus_early', 'stimulus_late', 'rating_early']
BANDS = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma']
TF_BINS = ['whole_epoch', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
TF_LABELS = [f'{b}|{t}' for b in BANDS for t in TF_BINS]
DCT_LABELS = [f'dct_{i:02d}' for i in range(1, 21)]
EXPECTED_TRIALS = 437
EXPECTED_PARTICIPANTS = 12
EXPECTED_F1 = 263
EXPECTED_F2 = 174
MIN_VALID_CHANNELS = 48
IQR_EPS = 1e-12
PARENTS = ['time_frequency|early_to_late', 'waveform_dct|segment_cross_product', 'covariance_topology|segment_cross_product']

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
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
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def normalize_channel_name(name: str):
    import re
    x = str(name).upper().strip()
    x = re.sub('[\\s_.-]+', '', x)
    x = re.sub('(REF|LE|EEG)$', '', x)
    return x

def import_verified(path: Path, expected_sha: str, name: str):
    if not path.is_file():
        raise RuntimeError(f'Missing prespecified source: {path}')
    got = sha256(path)
    if got != expected_sha:
        raise RuntimeError(f'{name} hash drift: {got} != {expected_sha}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def verify_final_lock(project: Path, rel: Path, expected_sha: str, phase: str):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'Missing {phase} final lock: {p}')
    got = sha256(p)
    if got != expected_sha:
        raise RuntimeError(f'{phase} final-lock SHA drift: {got}')
    lock = readj(p)
    if lock.get('phase') != phase:
        raise RuntimeError(f"Wrong phase in {p}: {lock.get('phase')}")
    return lock

def verify_child(project: Path, lock: dict, key: str, rel: Path):
    exp = lock.get('hashes', {}).get(key)
    if not exp:
        raise RuntimeError(f'Missing hash key {key} in upstream lock')
    got = sha256(project / rel)
    if got != exp:
        raise RuntimeError(f'Hash mismatch for {rel}: {got} != {exp}')

def verify_upstream(project: Path, bundle_dir: Path):
    th = readj(bundle_dir / 'prespecified_TEMPLATE_HASHES.json')
    for n, exp in th.items():
        p = bundle_dir / n
        if not p.is_file() or sha256(p) != exp:
            raise RuntimeError(f'Bundled prespecified template mismatch: {n}')
    f = verify_final_lock(project, heldout_topography_transfer_design_LOCK, heldout_topography_transfer_design_LOCK_SHA, 'heldout_topography_transfer_design')
    if f.get('heldout_signal_values_opened') is not False:
        raise RuntimeError('heldout_topography_transfer_design held-out boundary drift')
    if f.get('Confirmation_transfer_values_may_open_now') is not False:
        raise RuntimeError('heldout_topography_transfer_design unexpectedly permits Confirmation before Validation')
    if f.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('heldout_topography_transfer_design semantic boundary drift')
    for rel, exp in f.get('hashes', {}).items():
        p = project / rel
        if not p.is_file() or sha256(p) != exp:
            raise RuntimeError(f'heldout_topography_transfer_design fixed artifact mismatch: {rel}')
    policy = readj(project / heldout_topography_transfer_design_POLICY)
    if policy.get('phase') != 'heldout_topography_transfer_design':
        raise RuntimeError('heldout_topography_transfer_design policy phase drift')
    if set(policy.get('Discovery_supported_parent_roster', [])) != set(PARENTS):
        raise RuntimeError('prespecified parent roster drift')
    if policy['heldout_sequence']['first'] != 'Validation':
        raise RuntimeError('Validation is not prespecified first held-out split')
    for n in ['prespecified_DISCOVERY_TEMPLATE_TF_EARLY_TO_LATE.csv', 'prespecified_DISCOVERY_TEMPLATE_DCT_SEGMENT_CROSS_PRODUCT.csv', 'prespecified_DISCOVERY_TEMPLATE_COV_TOP100_SEGMENT_CROSS_PRODUCT.csv', 'prespecified_DISCOVERY_LOCALIZATIONS.csv']:
        rel = str(heldout_topography_transfer_design_BASE / 'templates' / n)
        exp = f['hashes'].get(rel)
        if exp is None:
            raise RuntimeError(f'heldout_topography_transfer_design final lock missing template: {n}')
        if sha256(bundle_dir / n) != exp:
            raise RuntimeError(f'Bundled {n} is not the heldout_topography_transfer_design-fixed template')
    k = verify_final_lock(project, validation_preprocessing_qc_LOCK, validation_preprocessing_qc_LOCK_SHA, 'validation_preprocessing_qc')
    if k.get('reports_outcomes_read') is not False or k.get('confirmation_opened') is not False:
        raise RuntimeError('validation_preprocessing_qc boundary drift')
    for key, rel in [('epoch_qc', validation_preprocessing_qc_EPOCH), ('inventory', validation_preprocessing_qc_INV), ('recording_qc', validation_preprocessing_qc_CH)]:
        verify_child(project, k, key, rel)
    m = verify_final_lock(project, validation_fixed_assignment_LOCK, validation_fixed_assignment_LOCK_SHA, 'validation_fixed_assignment')
    if m.get('structural_transfer_gate_passed') is not True:
        raise RuntimeError('validation_fixed_assignment structural transfer gate is no longer PASS')
    if m.get('reports_outcomes_read') is not False or m.get('family_refit_or_rename') is not False:
        raise RuntimeError('validation_fixed_assignment boundary drift')
    verify_child(project, m, 'assignments', validation_fixed_assignment_ASSIGN)
    if sha256(project / outcome_free_features_SCRIPT) != outcome_free_features_SCRIPT_SHA:
        raise RuntimeError('Exact outcome_free_features extractor SHA drift')
    return (f, policy, k, m, th)

def load_validation_metadata(project: Path):
    epoch = pd.read_csv(project / validation_preprocessing_qc_EPOCH, usecols=['epoch_id', 'raw_behavior_row_id', 'subject_id', 'raw_run', 'raw_source_path', 'epoch_family', 'source_array_row_0based', 'modeling_eligible'], low_memory=False)
    epoch['subject_id'] = epoch.subject_id.astype(str)
    epoch['raw_behavior_row_id'] = epoch.raw_behavior_row_id.astype(str)
    epoch = epoch[epoch.modeling_eligible.map(truth)].copy()
    ass = pd.read_csv(project / validation_fixed_assignment_ASSIGN, usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based'], low_memory=False)
    ass['subject_id'] = ass.subject_id.astype(str)
    ass['raw_behavior_row_id'] = ass.raw_behavior_row_id.astype(str)
    ass['trial_number'] = ass.trial_number.astype(int)
    ass['trajectory_family_1based'] = ass.trajectory_family_1based.astype(int)
    if len(ass) != EXPECTED_TRIALS or ass.subject_id.nunique() != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f'Validation assignment roster drift: {len(ass)}, {ass.subject_id.nunique()}')
    vc = ass.trajectory_family_1based.value_counts().to_dict()
    if vc.get(1) != EXPECTED_F1 or vc.get(2) != EXPECTED_F2:
        raise RuntimeError(f'Validation family counts drift: {vc}')
    needed = epoch.merge(ass[['subject_id', 'raw_behavior_row_id', 'trial_number', 'trajectory_family_1based']], on=['subject_id', 'raw_behavior_row_id'], how='inner', validate='many_to_one')
    c = needed.groupby(['subject_id', 'raw_behavior_row_id']).epoch_family.nunique()
    if len(c) != EXPECTED_TRIALS or not c.eq(3).all():
        raise RuntimeError(f'Validation three-epoch coverage drift: {len(c)}/{EXPECTED_TRIALS}')
    if len(needed) != EXPECTED_TRIALS * 3:
        raise RuntimeError(f'Expected {EXPECTED_TRIALS * 3} needed epochs, got {len(needed)}')
    return (epoch, ass, needed)

def load_inventory_and_channels(project: Path, needed: pd.DataFrame, template_channels: list[str]):
    inv = pd.read_csv(project / validation_preprocessing_qc_INV, low_memory=False)
    rec = pd.read_csv(project / validation_preprocessing_qc_CH, low_memory=False)
    rec['subject_id'] = rec.subject_id.astype(str)
    used_sources = sorted(needed.raw_source_path.astype(str).unique())
    channel_maps = {}
    registry = []
    for source in used_sources:
        q = rec[rec.raw_source_path.astype(str).eq(source)].sort_values('channel_position_1based')
        if len(q) != 64 or q.channel_position_1based.astype(int).tolist() != list(range(1, 65)):
            raise RuntimeError(f'Validation source does not have a 64-position channel roster: {source}')
        names = [normalize_channel_name(x) for x in q.channel_name.astype(str)]
        if len(set(names)) != 64:
            raise RuntimeError(f'Nonunique normalized channel roster: {source}')
        if set(names) != set(template_channels):
            missing = sorted(set(template_channels) - set(names))
            extra = sorted(set(names) - set(template_channels))
            raise RuntimeError(f'Channel roster differs from Discovery template for {source}; missing={missing}; extra={extra}')
        channel_maps[source] = names
        registry.append({'raw_source_path': source, 'subject_id': str(q.subject_id.iloc[0]), 'raw_run': int(q.raw_run.iloc[0]), 'channels': 64, 'canonical_roster_match': True, 'channel_roster_sha256': hashlib.sha256('|'.join(names).encode()).hexdigest()})
    inv_used = inv[inv.raw_source_path.astype(str).isin(used_sources)].copy()
    if inv_used.empty:
        raise RuntimeError('No validation_preprocessing_qc inventory rows for Validation sources')
    return (inv, inv_used, channel_maps, pd.DataFrame(registry))

def inventory_file(inv_source: pd.DataFrame, relative_path: str):
    q = inv_source[inv_source.relative_path.astype(str).eq(relative_path)]
    if len(q) != 1:
        raise RuntimeError(f'Expected one inventory row for {relative_path}, found {len(q)}')
    r = q.iloc[0]
    return (Path(str(r.stored_path)), str(r.sha256), int(r.size_bytes))

def verify_before_open(path: Path, exp_sha: str, exp_size: int, audit: list, source: str, relative_path: str):
    if not path.is_file():
        raise RuntimeError(f'Missing Validation preprocessed array: {path}')
    got_size = path.stat().st_size
    got_sha = sha256(path)
    passed = got_size == exp_size and got_sha == exp_sha
    audit.append({'raw_source_path': source, 'relative_path': relative_path, 'stored_path': str(path), 'expected_size_bytes': exp_size, 'observed_size_bytes': got_size, 'expected_sha256': exp_sha, 'observed_sha256': got_sha, 'passed': passed})
    if not passed:
        raise RuntimeError(f'Validation array integrity mismatch: {path}')

def source_slug(subject: str, run: int, source: str):
    return f'{subject}_run-{run:02d}_{hashlib.sha256(source.encode()).hexdigest()[:12]}'

def checkpoint_signature(rows: pd.DataFrame, array_audit: list, extractor_sha: str, script_sha: str):
    payload = {'rows': rows[['epoch_id', 'raw_behavior_row_id', 'epoch_family', 'source_array_row_0based']].sort_values('epoch_id').to_dict(orient='records'), 'arrays': sorted([{'relative_path': r['relative_path'], 'sha256': r['observed_sha256']} for r in array_audit], key=lambda x: x['relative_path']), 'extractor_sha': extractor_sha, 'script_sha': script_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def checkpoint_valid(directory: Path, signature: str):
    sp = directory / 'source_summary.json'
    if not sp.is_file():
        return False
    try:
        s = readj(sp)
    except Exception:
        return False
    if s.get('complete') is not True or s.get('input_signature_sha256') != signature:
        return False
    for r in s.get('stored_files', []):
        p = directory / r['relative_path']
        if not p.is_file() or p.stat().st_size != int(r['size_bytes']) or sha256(p) != r['sha256']:
            return False
    return True

def stored_manifest(directory: Path):
    rows = []
    for p in sorted((x for x in directory.rglob('*') if x.is_file() and x.name != 'source_summary.json')):
        rows.append({'relative_path': str(p.relative_to(directory)), 'size_bytes': p.stat().st_size, 'sha256': sha256(p)})
    return rows

def process_source(source_rows: pd.DataFrame, inv_source: pd.DataFrame, channel_names: list[str], canonical_channels: list[str], store_root: Path, outcome_free_features, script_sha: str, global_audit: list, recompute: bool):
    source = str(source_rows.raw_source_path.iloc[0])
    subject = str(source_rows.subject_id.iloc[0])
    run = int(source_rows.raw_run.iloc[0])
    final = store_root / 'sources' / source_slug(subject, run, source)
    final.parent.mkdir(parents=True, exist_ok=True)
    pos = {n: i for i, n in enumerate(channel_names)}
    reorder = [pos[c] for c in canonical_channels]
    local_audit = []
    array_paths = {}
    for fam in EPOCHS:
        for suffix in ['epochs_float32.npy', 'channel_mask_bool.npy']:
            rel = f'{fam}_{suffix}'
            p, hs, sz = inventory_file(inv_source, rel)
            verify_before_open(p, hs, sz, local_audit, source, rel)
            array_paths[fam, suffix] = p
    global_audit.extend(local_audit)
    sig = checkpoint_signature(source_rows, local_audit, outcome_free_features_SCRIPT_SHA, script_sha)
    if not recompute and checkpoint_valid(final, sig):
        return final
    tmp = final.with_name('.' + final.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        manifest = []
        tf_rows = []
        dct_rows = []
        cov_rows = []
        opened = {}
        for fam in EPOCHS:
            opened[fam, 'epochs_float32.npy'] = np.load(array_paths[fam, 'epochs_float32.npy'], mmap_mode='r', allow_pickle=False)
            opened[fam, 'channel_mask_bool.npy'] = np.load(array_paths[fam, 'channel_mask_bool.npy'], mmap_mode='r', allow_pickle=False)
        for r in source_rows.sort_values(['raw_behavior_row_id', 'epoch_family']).itertuples(index=False):
            fam = str(r.epoch_family)
            idx = int(r.source_array_row_0based)
            epochs = opened[fam, 'epochs_float32.npy']
            masks = opened[fam, 'channel_mask_bool.npy']
            if idx < 0 or idx >= epochs.shape[0] or idx >= masks.shape[0]:
                raise RuntimeError(f'{source} {fam}: source row {idx} outside prespecified array')
            blocks = outcome_free_features.extract_epoch_features(np.asarray(epochs[idx]), np.asarray(masks[idx]))
            tf = np.asarray(blocks['time_frequency'], dtype=np.float32).reshape(64, 30)[reorder]
            dct = np.asarray(blocks['waveform_dct'], dtype=np.float32).reshape(64, 20)[reorder]
            cov = np.asarray(blocks['covariance'], dtype=np.float32)[np.ix_(reorder, reorder)]
            if tf.shape != (64, 30) or dct.shape != (64, 20) or cov.shape != (64, 64):
                raise RuntimeError('prespecified extractor block shape drift')
            row = len(manifest)
            tf_rows.append(tf)
            dct_rows.append(dct)
            cov_rows.append(cov)
            manifest.append({'checkpoint_row_0based': row, 'epoch_id': str(r.epoch_id), 'raw_behavior_row_id': str(r.raw_behavior_row_id), 'subject_id': subject, 'trial_number': int(r.trial_number), 'trajectory_family_1based': int(r.trajectory_family_1based), 'epoch_family': fam, 'source_array_row_0based': idx, 'raw_source_path': source})
        tf_arr = np.asarray(tf_rows, dtype=np.float32)
        dct_arr = np.asarray(dct_rows, dtype=np.float32)
        cov_arr = np.asarray(cov_rows, dtype=np.float32)
        if tf_arr.shape != (len(manifest), 64, 30):
            raise RuntimeError(f'TF checkpoint shape drift: {tf_arr.shape}')
        if dct_arr.shape != (len(manifest), 64, 20):
            raise RuntimeError(f'DCT checkpoint shape drift: {dct_arr.shape}')
        if cov_arr.shape != (len(manifest), 64, 64):
            raise RuntimeError(f'Covariance checkpoint shape drift: {cov_arr.shape}')
        np.save(tmp / 'time_frequency_float32.npy', tf_arr, allow_pickle=False)
        np.save(tmp / 'waveform_dct_float32.npy', dct_arr, allow_pickle=False)
        np.save(tmp / 'covariance_float32.npy', cov_arr, allow_pickle=False)
        pd.DataFrame(manifest).to_csv(tmp / 'manifest.csv', index=False)
        summary = {'complete': True, 'phase': PHASE, 'subject_id': subject, 'raw_run': run, 'raw_source_path': source, 'rows': len(manifest), 'input_signature_sha256': sig, 'extractor_sha256': outcome_free_features_SCRIPT_SHA, 'stored_files': stored_manifest(tmp)}
        writej(tmp / 'source_summary.json', summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def robust_spatial_normalize(x: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(x, axis=1, keepdims=True)
        q25 = np.nanquantile(x, 0.25, axis=1, keepdims=True)
        q75 = np.nanquantile(x, 0.75, axis=1, keepdims=True)
    iqr = q75 - q25
    nvalid = np.sum(np.isfinite(x), axis=1, keepdims=True)
    good = np.isfinite(iqr) & (iqr > IQR_EPS) & (nvalid >= MIN_VALID_CHANNELS)
    return np.divide(x - med, iqr, out=np.full_like(x, np.nan, dtype=float), where=good)

def cov_to_edges(cov: np.ndarray):
    diag = np.diagonal(cov, axis1=1, axis2=2)
    den = np.sqrt(diag[:, :, None] * diag[:, None, :])
    corr = np.divide(cov, den, out=np.full_like(cov, np.nan, dtype=float), where=np.isfinite(den) & (den > 0))
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    iu = np.triu_indices(64, k=1)
    return (z[:, iu[0], iu[1]], iu)

def assemble_trials(assign: pd.DataFrame, source_dirs: list[Path]):
    manifests = []
    blocks = {}
    for d in source_dirs:
        m = pd.read_csv(d / 'manifest.csv', low_memory=False)
        m['checkpoint_dir'] = str(d)
        manifests.append(m)
    loc = pd.concat(manifests, ignore_index=True)
    loc['raw_behavior_row_id'] = loc.raw_behavior_row_id.astype(str)
    if loc.duplicated(['raw_behavior_row_id', 'epoch_family']).any():
        raise RuntimeError('Duplicate Validation checkpoint epoch locator')
    c = loc.groupby('raw_behavior_row_id').epoch_family.nunique()
    if len(c) != EXPECTED_TRIALS or not c.eq(3).all():
        raise RuntimeError(f'Checkpoint trial coverage drift: {len(c)}/{EXPECTED_TRIALS}')
    trials = assign.sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id']).reset_index(drop=True).copy()
    n = len(trials)
    tf = {e: np.full((n, 64, 30), np.nan, np.float32) for e in EPOCHS}
    dct = {e: np.full((n, 64, 20), np.nan, np.float32) for e in EPOCHS}
    cov = {e: np.full((n, 64, 64), np.nan, np.float32) for e in EPOCHS}
    lookup = {(str(r.raw_behavior_row_id), str(r.epoch_family)): (Path(r.checkpoint_dir), int(r.checkpoint_row_0based)) for r in loc.itertuples(index=False)}
    cache = {}
    for ti, r in enumerate(trials.itertuples(index=False)):
        rid = str(r.raw_behavior_row_id)
        for e in EPOCHS:
            d, row = lookup[rid, e]
            key = str(d)
            if key not in cache:
                cache[key] = {'tf': np.load(d / 'time_frequency_float32.npy', mmap_mode='r', allow_pickle=False), 'dct': np.load(d / 'waveform_dct_float32.npy', mmap_mode='r', allow_pickle=False), 'cov': np.load(d / 'covariance_float32.npy', mmap_mode='r', allow_pickle=False)}
            tf[e][ti] = cache[key]['tf'][row]
            dct[e][ti] = cache[key]['dct'][row]
            cov[e][ti] = cache[key]['cov'][row]
    return (trials, tf, dct, cov)

def participant_contrasts(trials: pd.DataFrame, values: np.ndarray):
    subjects = sorted(trials.subject_id.astype(str).unique())
    C = np.full((len(subjects), values.shape[1]), np.nan, float)
    for i, sid in enumerate(subjects):
        idx = np.where(trials.subject_id.astype(str).to_numpy() == sid)[0]
        y = trials.iloc[idx].trajectory_family_1based.to_numpy(int)
        a = values[idx][y == 1]
        b = values[idx][y == 2]
        if len(a) == 0 or len(b) == 0:
            raise RuntimeError(f'{sid} does not contain both prespecified families')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            C[i] = np.nanmean(b, axis=0) - np.nanmean(a, axis=0)
    return (subjects, C)

def all_signs(n: int):
    m = 1 << n
    bits = np.arange(m, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)[None, :] & 1
    return np.where(bits == 1, -1.0, 1.0)

def exact_energy_test(C: np.ndarray, signs: np.ndarray):
    count = np.sum(np.isfinite(C), axis=0)
    valid = count >= max(8, C.shape[0] - 2)
    if not valid.any():
        raise RuntimeError('No sufficiently covered cells for energy test')
    X = np.where(np.isfinite(C[:, valid]), C[:, valid], 0.0)
    den = count[valid].astype(float)
    obs_mean = X.sum(axis=0) / den
    obs = float(np.mean(obs_mean ** 2))
    means = signs @ X / den[None, :]
    null = np.mean(means ** 2, axis=1)
    p = float(np.sum(null >= obs - 1e-15) / len(null))
    return (obs, p, int(valid.sum()), obs_mean, valid)

def cosine(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa = a[m]
    bb = b[m]
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 0 else np.nan

def exact_concordance_test(C: np.ndarray, template: np.ndarray, signs: np.ndarray):
    cos = np.asarray([cosine(C[i], template) for i in range(C.shape[0])], float)
    if not np.isfinite(cos).all():
        raise RuntimeError(f'Nonfinite participant-template cosines: {cos}')
    obs = float(np.mean(cos))
    null = signs @ cos / len(cos)
    p = float(np.sum(null >= obs - 1e-15) / len(null))
    return (cos, obs, float(np.median(cos)), p)

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        v = min(1.0, (m - rank) * p[idx])
        running = max(running, v)
        out[idx] = running
    return out

def template_vector_tf(path: Path, canonical: list[str]):
    q = pd.read_csv(path)
    d = {(str(r.channel_name), str(r.subfeature)): float(r.family2_minus_family1_participant_mean) for r in q.itertuples(index=False)}
    out = []
    for ch in canonical:
        for sf in TF_LABELS:
            if (ch, sf) not in d:
                raise RuntimeError(f'TF Discovery template missing {(ch, sf)}')
            out.append(d[ch, sf])
    return np.asarray(out, float)

def template_vector_dct(path: Path, canonical: list[str]):
    q = pd.read_csv(path)
    d = {(str(r.channel_name), str(r.subfeature)): float(r.family2_minus_family1_participant_mean) for r in q.itertuples(index=False)}
    out = []
    for ch in canonical:
        for sf in DCT_LABELS:
            if (ch, sf) not in d:
                raise RuntimeError(f'DCT Discovery template missing {(ch, sf)}')
            out.append(d[ch, sf])
    return np.asarray(out, float)

def covariance_template(path: Path, canonical: list[str], iu):
    q = pd.read_csv(path).sort_values('rank_abs_effect')
    edge_index = {(canonical[a], canonical[b]): i for i, (a, b) in enumerate(zip(iu[0], iu[1]))}
    idx = []
    vals = []
    for r in q.itertuples(index=False):
        key = (str(r.channel_a), str(r.channel_b))
        if key not in edge_index:
            key = (str(r.channel_b), str(r.channel_a))
        if key not in edge_index:
            raise RuntimeError(f'prespecified covariance template edge absent in Validation roster: {r.channel_a}-{r.channel_b}')
        idx.append(edge_index[key])
        vals.append(float(r.family2_minus_family1_participant_mean_fisher_z_delta_or_product))
    if len(idx) != 100 or len(set(idx)) != 100:
        raise RuntimeError('prespecified covariance top-100 template index drift')
    return (np.asarray(idx, int), np.asarray(vals, float))

def secondary_specificity(trials, tf_quantity, dct_quantity, cov_quantity, canonical, iu, bundle_dir):
    rows = []
    if 'PO8' in canonical:
        ci = canonical.index('PO8')
        _, C = participant_contrasts(trials, tf_quantity[:, ci * 30:(ci + 1) * 30])
        t = template_vector_tf(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_TF_EARLY_TO_LATE.csv', canonical)[ci * 30:(ci + 1) * 30]
        pcs = [cosine(C[i], t) for i in range(len(C))]
        rows.append({'parent': 'T1', 'specificity': 'PO8|30D_TF', 'validation_mean_template_cosine': float(np.mean(pcs)), 'validation_median_template_cosine': float(np.median(pcs)), 'inferential_role': 'secondary_descriptive_does_not_drive_gate'})
    if 'FP2' in canonical:
        ci = canonical.index('FP2')
        _, C = participant_contrasts(trials, dct_quantity[:, ci * 20:(ci + 1) * 20])
        t = template_vector_dct(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_DCT_SEGMENT_CROSS_PRODUCT.csv', canonical)[ci * 20:(ci + 1) * 20]
        pcs = [cosine(C[i], t) for i in range(len(C))]
        rows.append({'parent': 'T2', 'specificity': 'FP2|20D_DCT', 'validation_mean_template_cosine': float(np.mean(pcs)), 'validation_median_template_cosine': float(np.median(pcs)), 'inferential_role': 'secondary_descriptive_does_not_drive_gate'})
    sf = DCT_LABELS.index('dct_16')
    idx = np.asarray([ci * 20 + sf for ci in range(64)], int)
    _, C = participant_contrasts(trials, dct_quantity[:, idx])
    td = template_vector_dct(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_DCT_SEGMENT_CROSS_PRODUCT.csv', canonical)[idx]
    pcs = [cosine(C[i], td) for i in range(len(C))]
    rows.append({'parent': 'T2', 'specificity': 'DCT16|64D_spatial', 'validation_mean_template_cosine': float(np.mean(pcs)), 'validation_median_template_cosine': float(np.median(pcs)), 'inferential_role': 'secondary_descriptive_does_not_drive_gate'})
    loc = pd.read_csv(bundle_dir / 'prespecified_DISCOVERY_LOCALIZATIONS.csv')
    pairs = loc[(loc.parent_test == 'covariance_topology|segment_cross_product') & (loc.localization_type == 'region_pair')].localization.astype(str).tolist()

    def region(ch):
        x = ch.upper()
        if x.startswith(('FP', 'AF')):
            return 'frontopolar'
        if x.startswith(('FT', 'TP')) or (x.startswith('T') and len(x) > 1 and x[1].isdigit()):
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
    regs = [region(c) for c in canonical]
    for pair in pairs:
        a, b = pair.split('--')
        eidx = [ei for ei, (u, v) in enumerate(zip(iu[0], iu[1])) if '--'.join(sorted([regs[u], regs[v]])) == pair]
        if not eidx:
            continue
        _, C = participant_contrasts(trials, cov_quantity[:, eidx])
        gm = np.nanmean(C, axis=0)
        rows.append({'parent': 'T3', 'specificity': f'{pair}|{len(eidx)}edges', 'validation_mean_template_cosine': np.nan, 'validation_median_template_cosine': np.nan, 'validation_effect_energy': float(np.nanmean(gm ** 2)), 'inferential_role': 'secondary_descriptive_does_not_drive_gate'})
    return pd.DataFrame(rows)

def self_test():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 64, 5))
    z1 = robust_spatial_normalize(x)
    z2 = robust_spatial_normalize(9 * x + 17)
    if np.nanmax(np.abs(z1 - z2)) > 1e-10:
        raise RuntimeError('Spatial-normalization self-test failed')
    n = 12
    p = 80
    template = rng.normal(size=p)
    C = np.asarray([template + 0.15 * rng.normal(size=p) for _ in range(n)])
    S = all_signs(n)
    _, pe, _, _, _ = exact_energy_test(C, S)
    cos, mc, med, pc = exact_concordance_test(C, template, S)
    if pe > 0.01 or pc > 0.01 or mc <= 0 or (med <= 0):
        raise RuntimeError(f'Exact transfer self-test failed: energy={pe}, cosine={pc}, mean={mc}')
    iu = np.triu_indices(64, k=1)
    if len(iu[0]) != 2016 or S.shape != (4096, 12):
        raise RuntimeError('Cardinality self-test failed')
    print('validation_full_channel_transfer self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--bundle-dir', type=Path)
    ap.add_argument('--recompute-sources', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.bundle_dir is None:
        raise RuntimeError('--bundle-dir is required for a real run')
    project = a.project.resolve()
    bundle_dir = a.bundle_dir.resolve()
    script_sha = sha256(Path(__file__).resolve())
    verify_upstream(project, bundle_dir)
    outcome_free_features = import_verified(project / outcome_free_features_SCRIPT, outcome_free_features_SCRIPT_SHA, 'prespecified_outcome_free_features')
    epoch, assign, needed = load_validation_metadata(project)
    tf_template_df = pd.read_csv(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_TF_EARLY_TO_LATE.csv')
    canonical = sorted(tf_template_df.channel_name.astype(str).unique().tolist())
    if len(canonical) != 64:
        raise RuntimeError(f'Discovery template channel count drift: {len(canonical)}')
    inv, inv_used, channel_maps, registry = load_inventory_and_channels(project, needed, canonical)
    out = project / 'outputs/interpretability/validation_full_channel_transfer_validation_full_channel_transfer'
    final = out / 'qc/validation_full_channel_transfer_final_output_lock_hashes.json'
    if out.exists():
        if final.is_file():
            if a.force:
                shutil.rmtree(out)
            else:
                raise RuntimeError(f'Completed validation_full_channel_transfer output exists: {out}')
        else:
            shutil.rmtree(out)
    (out / 'tables').mkdir(parents=True)
    (out / 'qc').mkdir()
    (out / 'reports').mkdir()
    store = project / STORE_REL
    store.mkdir(parents=True, exist_ok=True)
    rehash = []
    source_dirs = []
    groups = list(needed.groupby(['raw_source_path', 'subject_id', 'raw_run'], sort=True))
    for i, ((source, subject, run), g) in enumerate(groups, 1):
        print(f'[{now()}] Validation source {i}/{len(groups)}: {subject} run {run} ({len(g)} needed epochs)', flush=True)
        inv_source = inv[inv.raw_source_path.astype(str).eq(str(source))].copy()
        d = process_source(g.copy(), inv_source, channel_maps[str(source)], canonical, store, outcome_free_features, script_sha, rehash, a.recompute_sources)
        source_dirs.append(d)
    rh = pd.DataFrame(rehash).drop_duplicates(['stored_path'])
    if rh.empty or not rh.passed.all():
        raise RuntimeError('Validation source rehash audit failed')
    rh.to_csv(out / 'qc/validation_full_channel_transfer_opened_validation_array_rehash.csv', index=False)
    registry.to_csv(out / 'qc/validation_full_channel_transfer_validation_channel_registry.csv', index=False)
    ci = []
    for d in source_dirs:
        s = readj(d / 'source_summary.json')
        for r in s['stored_files']:
            ci.append({'checkpoint_dir': str(d), 'raw_source_path': s['raw_source_path'], 'relative_path': r['relative_path'], 'size_bytes': r['size_bytes'], 'sha256': r['sha256']})
    pd.DataFrame(ci).to_csv(out / 'qc/validation_full_channel_transfer_checkpoint_inventory.csv', index=False)
    trials, tf, dct, cov = assemble_trials(assign, source_dirs)
    ztf = {e: robust_spatial_normalize(tf[e].astype(float)) for e in EPOCHS}
    tf_q = (ztf['stimulus_late'] - ztf['stimulus_early']).reshape(len(trials), -1)
    zdct = {e: robust_spatial_normalize(dct[e].astype(float)) for e in EPOCHS}
    dct_a = zdct['stimulus_late'] - zdct['stimulus_early']
    dct_b = zdct['rating_early'] - zdct['stimulus_late']
    dct_q = (dct_a * dct_b).reshape(len(trials), -1)
    cov_edges = {}
    iu = None
    for e in EPOCHS:
        cov_edges[e], iu = cov_to_edges(cov[e].astype(float))
    cov_a = cov_edges['stimulus_late'] - cov_edges['stimulus_early']
    cov_b = cov_edges['rating_early'] - cov_edges['stimulus_late']
    cov_q = cov_a * cov_b
    t_tf = template_vector_tf(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_TF_EARLY_TO_LATE.csv', canonical)
    t_dct = template_vector_dct(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_DCT_SEGMENT_CROSS_PRODUCT.csv', canonical)
    cov_idx, t_cov = covariance_template(bundle_dir / 'prespecified_DISCOVERY_TEMPLATE_COV_TOP100_SEGMENT_CROSS_PRODUCT.csv', canonical, iu)
    representations = {'time_frequency|early_to_late': (tf_q, t_tf, np.arange(tf_q.shape[1])), 'waveform_dct|segment_cross_product': (dct_q, t_dct, np.arange(dct_q.shape[1])), 'covariance_topology|segment_cross_product': (cov_q, t_cov, cov_idx)}
    signs = all_signs(EXPECTED_PARTICIPANTS)
    primary = []
    participant_cosines = []
    parent_aux = {}
    for parent, (X, template, template_indices) in representations.items():
        subjects, C = participant_contrasts(trials, X)
        energy, p_energy, ncells, group_mean, valid = exact_energy_test(C, signs)
        Cdir = C[:, template_indices]
        cos, mean_cos, median_cos, p_cos = exact_concordance_test(Cdir, template, signs)
        parent_aux[parent] = {'C': C, 'group_mean': group_mean, 'energy': energy, 'raw_energy_p': p_energy, 'raw_cosine_p': p_cos, 'mean_cosine': mean_cos, 'median_cosine': median_cos, 'energy_cells': ncells}
        primary.extend([{'parent_test': parent, 'test_type': 'A_family_association_energy', 'observed_statistic': energy, 'raw_exact_signflip_p': p_energy, 'n_participants': len(subjects), 'n_features_used': ncells, 'direction_requirement_applicable': False}, {'parent_test': parent, 'test_type': 'B_discovery_template_concordance', 'observed_statistic': mean_cos, 'raw_exact_signflip_p': p_cos, 'n_participants': len(subjects), 'n_features_used': len(template), 'direction_requirement_applicable': True, 'median_participant_cosine': median_cos}])
        for sid, c in zip(subjects, cos):
            participant_cosines.append({'parent_test': parent, 'subject_id': sid, 'participant_template_cosine': float(c)})
    primary = pd.DataFrame(primary)
    primary['holm_p_across_6_validation_tests'] = holm(primary.raw_exact_signflip_p.to_numpy(float))
    primary['holm_supported'] = primary.holm_p_across_6_validation_tests.le(0.05)
    primary.to_csv(out / 'tables/validation_full_channel_transfer_validation_primary_tests.csv', index=False)
    pd.DataFrame(participant_cosines).to_csv(out / 'tables/validation_full_channel_transfer_validation_participant_template_cosines.csv', index=False)
    reps = []
    for parent in PARENTS:
        arow = primary[(primary.parent_test == parent) & (primary.test_type == 'A_family_association_energy')].iloc[0]
        brow = primary[(primary.parent_test == parent) & (primary.test_type == 'B_discovery_template_concordance')].iloc[0]
        mean_cos = float(parent_aux[parent]['mean_cosine'])
        med_cos = float(parent_aux[parent]['median_cosine'])
        replicated = bool(arow.holm_supported and brow.holm_supported and (mean_cos > 0) and (med_cos > 0))
        reps.append({'parent_test': parent, 'validation_energy_observed': float(arow.observed_statistic), 'validation_energy_raw_p': float(arow.raw_exact_signflip_p), 'validation_energy_Holm_p': float(arow.holm_p_across_6_validation_tests), 'validation_mean_template_cosine': mean_cos, 'validation_median_template_cosine': med_cos, 'validation_concordance_raw_p': float(brow.raw_exact_signflip_p), 'validation_concordance_Holm_p': float(brow.holm_p_across_6_validation_tests), 'validation_parent_replicated': replicated})
    reps = pd.DataFrame(reps)
    reps.to_csv(out / 'tables/validation_full_channel_transfer_validation_parent_replication.csv', index=False)
    nrep = int(reps.validation_parent_replicated.sum())
    gate = bool(nrep >= 2)
    sec = secondary_specificity(trials, tf_q, dct_q, cov_q, canonical, iu, bundle_dir)
    sec.to_csv(out / 'tables/validation_full_channel_transfer_validation_secondary_specificity.csv', index=False)
    next_phase = 'topography_transfer_followup' if gate else 'topographic_transfer_synthesis'
    next_role = 'One-shot Confirmation full-channel transfer of the same three prespecified parents.' if gate else 'Close/synthesize the topographic transfer branch without opening Confirmation signal-derived features.'
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_full_channel_transfer_complete', 'counts': {'Validation_participants': EXPECTED_PARTICIPANTS, 'Validation_trajectories': EXPECTED_TRIALS, 'family1': EXPECTED_F1, 'family2': EXPECTED_F2, 'sources_processed': len(source_dirs), 'exact_signflip_patterns': int(len(signs)), 'primary_tests': 6, 'parents_replicated': nrep}, 'parent_replication': {r.parent_test: bool(r.validation_parent_replicated) for r in reps.itertuples()}, 'Validation_gate_passed': gate, 'Confirmation_transfer_values_may_open_now': gate, 'scientific_boundaries': {'behavioral_report_values_read': False, 'emotion_category_values_read': False, 'semantic_relabeling_permitted': False, 'family_refit_or_rename': False, 'Confirmation_signal_or_feature_values_read': False, 'external_dataset_replication': False, 'interpretation': 'same-dataset fixed structural EEG transfer only'}, 'next_phase': next_phase, 'next_phase_role': next_role}
    writej(out / 'qc/validation_full_channel_transfer_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'upstream_design_specification': 'heldout_topography_transfer_design', 'exact_outcome_free_features_extractor_sha256': outcome_free_features_SCRIPT_SHA, 'template_hashes': readj(bundle_dir / 'prespecified_TEMPLATE_HASHES.json'), 'Validation_primary_tests': 6, 'multiplicity': 'Holm across all 6 tests', 'parent_replication_rule': 'energy Holm<=.05 AND concordance Holm<=.05 AND mean cosine>0 AND median cosine>0', 'Validation_gate': 'at least 2 of 3 parents replicate', 'behavioral_report_values_read': False, 'Confirmation_signal_or_feature_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': next_phase}
    writej(out / 'qc/validation_full_channel_transfer_contract.json', contract)
    lines = ['# validation_full_channel_transfer — One-shot Validation full-channel structural transfer', '', f"Validation gate: **{('PASS' if gate else 'FAIL')}**", '', '| Parent | Energy Holm p | Mean cosine | Median cosine | Concordance Holm p | Replicated |', '|---|---:|---:|---:|---:|---|']
    for r in reps.itertuples():
        lines.append(f'| {r.parent_test} | {r.validation_energy_Holm_p:.6g} | {r.validation_mean_template_cosine:.6f} | {r.validation_median_template_cosine:.6f} | {r.validation_concordance_Holm_p:.6g} | {bool(r.validation_parent_replicated)} |')
    lines += ['', f'Replicated parents: **{nrep}/3**.', '', 'Confirmation signal-derived features were **not opened** in this phase.', '', 'No behavioral semantic label may be attached to F1/F2 from this result.', '', f'Next phase: **{next_phase}** — {next_role}']
    (out / 'reports/validation_full_channel_transfer_report.md').write_text(''.join(lines) + '', encoding='utf-8')
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    hashes = {str(p.relative_to(project)): sha256(p) for p in files}
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': 'validation_full_channel_transfer_complete', 'hashes': hashes, 'Validation_gate_passed': gate, 'parents_replicated': nrep, 'Confirmation_transfer_values_may_open_now': gate, 'Confirmation_signal_or_feature_values_read': False, 'behavioral_report_values_read': False, 'semantic_relabeling_permitted': False, 'next_phase': next_phase})
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    hz = project / '' / f'validation_full_channel_transfer_validation_full_channel_transfer_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    hs = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{hs}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] validation_full_channel_transfer PASS')
    print(f'Validation replicated parents: {nrep}/3')
    for r in reps.itertuples():
        print(f'{r.parent_test}: energy Holm={r.validation_energy_Holm_p:.6g}; mean cosine={r.validation_mean_template_cosine:.6f}; median cosine={r.validation_median_template_cosine:.6f}; concordance Holm={r.validation_concordance_Holm_p:.6g}; replicated={bool(r.validation_parent_replicated)}')
    print(f'Validation gate passed: {gate}')
    print('Confirmation signal/features opened: False')
    print('Semantic relabeling permitted: False')
    print(f'Next phase: {next_phase}')
    print('', hz)
    print('SHA-256:', hs)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())