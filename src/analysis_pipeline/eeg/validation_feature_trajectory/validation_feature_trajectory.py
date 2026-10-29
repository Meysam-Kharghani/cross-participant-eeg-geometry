# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase validation_feature_trajectory — Validation outcome-free compact features and trajectories.

This cohort adapter performs only the validation_geometry_transfer_design-prespecified Validation transfer steps:

1. verify every validation_preprocessing_qc phase output and every large-store file byte-for-byte;
2. call the exact prespecified outcome_free_features per-epoch feature extractor;
3. construct the exact 169-cell compact representation used by secondary_models/latent_state_fit;
4. apply the exact latent_state_fit within-participant, epoch-family robust
   standardization and whitened PCA to complete three-epoch trajectories;
5. write raw six-coordinate trajectory geometry for validation_fixed_assignment.

No report, outcome, emotion category, stimulus identity, context target, or
Confirmation file is read. Context is not loaded into the feature or geometry
frames. No Discovery centroid is applied and no family assignment is made in
this phase.

Large per-source compact checkpoints and the global compact array are stored
outputs/large/validation_feature_trajectory_validation_compact_features_v1/"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'validation_feature_trajectory'
NEXT_PHASE = 'validation_fixed_assignment'
PREFIX = 'validation_feature_trajectory'
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
PER_EPOCH_COMPACT = 169
EXPECTED_K_COUNTS = {'subjects': 12, 'trials': 494, 'eligible_epochs': 1424, 'complete_trajectories': 437, 'large_files': 209}
EXPECTED_outcome_free_features_SHA = 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba'
EXPECTED_secondary_models_SHA = '6cbe259167269fa1dc6927a749c3cfbf8492ac6b4ff4c048fc183f6082cf774d'
EXPECTED_latent_state_fit_SHA = 'c9f4c963d77a57d75e634fdbdabec5f2ae67ba1c0b76fe0a1621f3f266136c1d'
EXPECTED_latent_state_design_POLICY_CANONICAL_SHA = '1383dd7fd2faf6035391899dca9adad580b20e7d86ecf322acf7eddf7e3866d7'
EXPECTED_validation_geometry_transfer_design_TRANSFER_CANONICAL_SHA = 'bd3e7a5f4bd0cba00cfa66a0cd09208c07e0b1a9edab4b2a9493060200e1b37e'
EXPECTED_validation_geometry_transfer_design_CORE_SHA = '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b'
STORE_REL = Path('outputs/large/validation_feature_trajectory_validation_compact_features_v1')
outcome_free_features_REL = Path('src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py')
secondary_models_REL = Path('src/analysis_pipeline/eeg/secondary_models/secondary_models.py')
latent_state_fit_REL = Path('src/analysis_pipeline/eeg/latent_state_fit/latent_state_fit.py')
latent_state_design_POLICY_REL = Path('analysis_specifications/S037_latent_state.json')
latent_state_design_CONTRACT_REL = Path('config/latent_state_design_latent_state_design_contract.json')
latent_state_design_SCHEMA_REL = Path('outputs/tables/latent_state_design_compact_feature_schema.csv')
validation_geometry_transfer_design_POLICY_REL = Path('analysis_specifications/S082_validation_geometry_transfer.json')
validation_geometry_transfer_design_REFERENCE_REL = Path('analysis_specifications/S081_validation_geometry_transfer_reference.json')
validation_geometry_transfer_design_CORE_REL = Path('analysis_specifications/reference_code/validation_geometry_transfer_core.py')
K_REL = {'recording_qc': Path('outputs/qc/validation_preprocessing_qc_validation_recording_channel_qc.csv'), 'epoch_qc': Path('outputs/tables/validation_preprocessing_qc_validation_epoch_amplitude_qc.csv'), 'inventory': Path('outputs/tables/validation_preprocessing_qc_validation_large_array_inventory.csv'), 'branch_qc': Path('outputs/qc/validation_preprocessing_qc_validation_branch_retention_qc.csv'), 'subject_branch_qc': Path('outputs/qc/validation_preprocessing_qc_validation_subject_branch_retention_qc.csv'), 'trajectory_manifest': Path('outputs/tables/validation_preprocessing_qc_validation_post_qc_trajectory_availability.csv'), 'subject_trajectory_qc': Path('outputs/qc/validation_preprocessing_qc_validation_subject_trajectory_availability.csv'), 'transfer_gate': Path('outputs/qc/validation_preprocessing_qc_validation_pre_feature_transfer_gate.csv'), 'source_rehash': Path('outputs/qc/validation_preprocessing_qc_validation_source_rehash.csv'), 'summary': Path('outputs/qc/validation_preprocessing_qc_validation_preprocessing_qc_summary.json'), 'next': Path('analysis_specifications/S084_validation_preprocessing_qc_feature_trajectory.json'), 'contract': Path('config/validation_preprocessing_qc_validation_preprocessing_qc_contract.json'), 'report': Path('outputs/reports/validation_preprocessing_qc_validation_preprocessing_qc_report.md'), 'lock': Path('outputs/qc/validation_preprocessing_qc_final_output_lock_hashes.json')}
K_LOCK_PATHS = {k: v for k, v in K_REL.items() if k != 'lock'}
TRANSFER_MIN_PARTICIPANTS = 9
TRANSFER_MIN_FRACTION = 0.75
TRANSFER_MIN_PER_SUBJECT = 16
FINITE_FRACTION = 0.9
MIN_RETAINED = 30

def parse_args():
    p = argparse.ArgumentParser(description='Extract exact outcome_free_features compact features and latent_state_fit trajectories in Validation.')
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--recompute-sources', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def csha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def jread(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)

def atom_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + '', encoding='utf-8')
    os.replace(tmp, path)

def bval(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def require(paths: Iterable[Path]):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n  - ' + '\n  - '.join(missing))

def import_verified(path: Path, expected_sha: str, name: str):
    require([path])
    observed = sha(path)
    if observed != expected_sha:
        raise RuntimeError(f'{name} source hash mismatch: expected {expected_sha}, observed {observed}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {name}: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def output_paths(project: Path):
    return {'log': project / 'outputs/logs/validation_feature_trajectory_validation_feature_trajectory.log', 'large_rehash': project / 'outputs/qc/validation_feature_trajectory_validation_validation_preprocessing_qc_large_store_rehash.csv', 'source_inventory': project / 'outputs/qc/validation_feature_trajectory_validation_feature_checkpoint_inventory.csv', 'feature_qc': project / 'outputs/qc/validation_feature_trajectory_validation_compact_feature_qc.csv', 'participant_feature_audit': project / 'outputs/tables/validation_feature_trajectory_validation_participant_feature_audit.csv', 'feature_schema': project / 'outputs/tables/validation_feature_trajectory_validation_compact_feature_schema.csv', 'epoch_manifest': project / 'outputs/tables/validation_feature_trajectory_validation_compact_epoch_manifest.csv', 'pca_scores': project / 'outputs/tables/validation_feature_trajectory_validation_epoch_pca_scores.csv', 'pca_loadings': project / 'outputs/tables/validation_feature_trajectory_validation_pca_loadings.csv', 'preprocessing': project / 'outputs/tables/validation_feature_trajectory_validation_participant_preprocessing.csv', 'trajectories': project / 'outputs/tables/validation_feature_trajectory_validation_raw_trajectory_geometry.csv', 'participant_qc': project / 'outputs/qc/validation_feature_trajectory_validation_participant_trajectory_qc.csv', 'gate': project / 'outputs/qc/validation_feature_trajectory_validation_pre_assignment_transfer_gate.csv', 'global_store_inventory': project / 'outputs/tables/validation_feature_trajectory_validation_compact_store_inventory.csv', 'summary': project / 'outputs/qc/validation_feature_trajectory_validation_feature_trajectory_summary.json', 'report': project / 'outputs/reports/validation_feature_trajectory_validation_feature_trajectory_report.md', 'contract': project / 'config/validation_feature_trajectory_validation_feature_trajectory_contract.json', 'next': project / 'analysis_specifications/S079_validation_feature_trajectory_fixed_assignment.json', 'lock': project / 'outputs/qc/validation_feature_trajectory_final_output_lock_hashes.json'}

def verify_k(project: Path):
    paths = {k: project / v for k, v in K_REL.items()}
    require(paths.values())
    lock = jread(paths['lock'])
    fixed = lock.get('hashes', {})
    for key, rel in K_LOCK_PATHS.items():
        if key not in fixed:
            raise RuntimeError(f'validation_preprocessing_qc final lock missing {key}')
        if sha(paths[key]) != str(fixed[key]):
            raise RuntimeError(f'validation_preprocessing_qc fixed output mismatch: {key}')
    summary = jread(paths['summary'])
    nxt = jread(paths['next'])
    contract = jread(paths['contract'])
    if summary.get('phase') != 'validation_preprocessing_qc' or not summary.get('decision', {}).get('validation_feature_trajectory_feature_extraction_permitted'):
        raise RuntimeError('validation_preprocessing_qc did not permit validation_feature_trajectory.')
    if nxt.get('next_phase') != PHASE or not nxt.get('next_phase_permitted'):
        raise RuntimeError('validation_preprocessing_qc next-phase policy does not authorize validation_feature_trajectory.')
    if str(nxt.get('required_outcome_free_features_source_code_sha256')) != EXPECTED_outcome_free_features_SHA:
        raise RuntimeError('validation_preprocessing_qc expected different outcome_free_features code.')
    if str(contract.get('validation_geometry_transfer_design_transfer_policy_canonical_sha256')) != EXPECTED_validation_geometry_transfer_design_TRANSFER_CANONICAL_SHA:
        raise RuntimeError('validation_geometry_transfer_design transfer lineage mismatch.')
    counts = summary.get('counts', {})
    for k, expected in [('validation_subjects', 12), ('authoritative_trials', 494), ('modeling_eligible_epochs', 1424), ('complete_three_epoch_trajectories', 437), ('large_store_files', 209)]:
        if int(counts.get(k, -1)) != expected:
            raise RuntimeError(f'validation_preprocessing_qc count changed for {k}: {counts.get(k)}')
    if summary.get('decision', {}).get('reports_outcomes_read') or summary.get('decision', {}).get('confirmation_opened'):
        raise RuntimeError('validation_preprocessing_qc boundary was violated.')
    epoch = pd.read_csv(paths['epoch_qc'], usecols=['epoch_id', 'raw_behavior_row_id', 'subject_id', 'raw_run', 'raw_source_path', 'epoch_family', 'source_array_row_0based', 'final_good_channels', 'modeling_eligible'], low_memory=False)
    epoch = epoch[epoch.modeling_eligible.map(bval)].copy().reset_index(drop=True)
    if len(epoch) != 1424 or epoch.epoch_id.duplicated().any():
        raise RuntimeError('validation_preprocessing_qc eligible epoch manifest is invalid.')
    traj = pd.read_csv(paths['trajectory_manifest'], usecols=['raw_behavior_row_id', 'subject_id', 'trial_number', 'rating_early', 'stimulus_early', 'stimulus_late', 'complete_three_epoch_trajectory'], low_memory=False)
    if len(traj) != 494 or traj.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('validation_preprocessing_qc trajectory manifest is invalid.')
    subj = pd.read_csv(paths['subject_trajectory_qc'], low_memory=False)
    inv = pd.read_csv(paths['inventory'], low_memory=False)
    if len(inv) != 209:
        raise RuntimeError('validation_preprocessing_qc large inventory row count changed.')
    return (paths, summary, nxt, contract, epoch, traj, subj, inv)

def verify_discovery_lineage(project: Path):
    p_policy = project / validation_geometry_transfer_design_POLICY_REL
    p_ref = project / validation_geometry_transfer_design_REFERENCE_REL
    p_core = project / validation_geometry_transfer_design_CORE_REL
    p_latent = project / latent_state_design_POLICY_REL
    p_contract = project / latent_state_design_CONTRACT_REL
    p_schema = project / latent_state_design_SCHEMA_REL
    require([p_policy, p_ref, p_core, p_latent, p_contract, p_schema])
    policy = jread(p_policy)
    ref = jread(p_ref)
    latent = jread(p_latent)
    contract = jread(p_contract)
    if csha(policy) != EXPECTED_validation_geometry_transfer_design_TRANSFER_CANONICAL_SHA:
        raise RuntimeError('validation_geometry_transfer_design transfer policy canonical hash mismatch.')
    if sha(p_core) != EXPECTED_validation_geometry_transfer_design_CORE_SHA:
        raise RuntimeError('validation_geometry_transfer_design prespecified transfer core hash mismatch.')
    if csha(latent) != EXPECTED_latent_state_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('latent_state_design latent policy canonical hash mismatch.')
    expected_schema = str(contract.get('output_hashes', {}).get('feature_schema', ''))
    if not expected_schema or sha(p_schema) != expected_schema:
        raise RuntimeError('latent_state_design compact feature schema hash mismatch.')
    schema = pd.read_csv(p_schema, low_memory=False)
    if len(schema) != 169 or not np.array_equal(schema.compact_cell_0based.to_numpy(int), np.arange(169)):
        raise RuntimeError('Compact schema invalid.')
    if policy.get('trajectory_construction', {}).get('epoch_order') != FAMILIES:
        raise RuntimeError('validation_geometry_transfer_design epoch order mismatch.')
    return (policy, ref, latent, schema)

def rehash_large_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    required = {'stored_path', 'size_bytes', 'sha256', 'raw_source_path', 'relative_path'}
    if not required.issubset(inv.columns):
        raise RuntimeError(f'validation_preprocessing_qc inventory missing {sorted(required - set(inv.columns))}')
    rows = []
    for i, r in inv.iterrows():
        p = Path(str(r.stored_path))
        if not p.is_file():
            raise RuntimeError(f'Missing validation_preprocessing_qc large-store file: {p}')
        obs = sha(p)
        ok = p.stat().st_size == int(r.size_bytes) and obs == str(r.sha256)
        if not ok:
            raise RuntimeError(f'validation_preprocessing_qc large-store integrity failure: {p}')
        rows.append({'inventory_row_1based': i + 1, 'raw_source_path': str(r.raw_source_path), 'relative_path': str(r.relative_path), 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': obs, 'passed': True})
        if (i + 1) % 25 == 0 or i + 1 == len(inv):
            print(f'[{now()}] Large-store hash {i + 1}/{len(inv)}', flush=True)
    return pd.DataFrame(rows)

def source_slug(source: str, subject: str, run: int) -> str:
    return f'{subject}_run-{int(run):02d}_{hashlib.sha256(source.encode()).hexdigest()[:12]}'

def stored_manifest(directory: Path, exclude: set[str] | None=None):
    exclude = exclude or set()
    rows = []
    for p in sorted((x for x in directory.rglob('*') if x.is_file() and x.name not in exclude)):
        rows.append({'relative_path': str(p.relative_to(directory)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    return rows

def checkpoint_valid(directory: Path, input_signature: str, script_sha: str) -> bool:
    sp = directory / 'source_summary.json'
    if not sp.is_file():
        return False
    try:
        s = jread(sp)
    except Exception:
        return False
    if not s.get('complete') or s.get('input_signature_sha256') != input_signature or s.get('executor_sha256') != script_sha:
        return False
    for r in s.get('stored_files', []):
        p = directory / r['relative_path']
        if not p.is_file() or p.stat().st_size != int(r['size_bytes']) or sha(p) != r['sha256']:
            return False
    return True

def compact_from_blocks(blocks: dict[str, np.ndarray], secondary_models) -> np.ndarray:
    parts = [secondary_models.channel_median_iqr(np.asarray(blocks['time_frequency'])[None, ...])[0], secondary_models.channel_median_iqr(np.asarray(blocks['morphology'])[None, ...])[0], secondary_models.channel_median_iqr(np.asarray(blocks['plv_channel'])[None, ...])[0], secondary_models.channel_median_iqr(np.asarray(blocks['waveform_dct'])[None, ...])[0], np.asarray(blocks['covariance_summary'], dtype=np.float32).reshape(-1), np.asarray(blocks['plv_global'], dtype=np.float32).reshape(-1)]
    out = np.concatenate(parts).astype(np.float32, copy=False)
    if out.shape != (169,) or np.isinf(out).any():
        raise RuntimeError(f'Compact vector integrity failure: {out.shape}')
    return out

def source_input_signature(source_rows: pd.DataFrame, inv_source: pd.DataFrame, outcome_free_features_sha: str, secondary_models_sha: str, script_sha: str) -> str:
    payload = {'source': str(source_rows.raw_source_path.iloc[0]), 'rows': source_rows[['epoch_id', 'epoch_family', 'source_array_row_0based', 'final_good_channels']].sort_values('epoch_id').to_dict(orient='records'), 'files': inv_source[['relative_path', 'size_bytes', 'sha256']].sort_values('relative_path').to_dict(orient='records'), 'outcome_free_features': outcome_free_features_sha, 'secondary_models': secondary_models_sha, 'executor': script_sha}
    return csha(payload)

def find_inventory_file(inv_source: pd.DataFrame, name: str) -> Path:
    m = inv_source[inv_source.relative_path.astype(str).eq(name)]
    if len(m) != 1:
        raise RuntimeError(f'Expected one {name} in source inventory, found {len(m)}')
    return Path(str(m.stored_path.iloc[0]))

def process_source(source_rows: pd.DataFrame, inv_source: pd.DataFrame, root: Path, outcome_free_features, secondary_models, script_sha: str, recompute: bool) -> Path:
    source = str(source_rows.raw_source_path.iloc[0])
    subject = str(source_rows.subject_id.iloc[0])
    run = int(source_rows.raw_run.iloc[0])
    final = root / 'sources' / source_slug(source, subject, run)
    sig = source_input_signature(source_rows, inv_source, EXPECTED_outcome_free_features_SHA, EXPECTED_secondary_models_SHA, script_sha)
    if not recompute and checkpoint_valid(final, sig, script_sha):
        return final
    tmp = final.with_name('.' + final.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        manifest_rows = []
        vectors = []
        for family in FAMILIES:
            g = source_rows[source_rows.epoch_family.eq(family)].sort_values('source_array_row_0based')
            if g.empty:
                continue
            epochs = np.load(find_inventory_file(inv_source, f'{family}_epochs_float32.npy'), mmap_mode='r', allow_pickle=False)
            masks = np.load(find_inventory_file(inv_source, f'{family}_channel_mask_bool.npy'), mmap_mode='r', allow_pickle=False)
            if epochs.shape[0] != masks.shape[0]:
                raise RuntimeError(f'{source} {family}: epoch/mask row mismatch')
            for r in g.itertuples(index=False):
                idx = int(r.source_array_row_0based)
                if idx < 0 or idx >= epochs.shape[0]:
                    raise RuntimeError(f'{source} {family}: row outside array')
                blocks = outcome_free_features.extract_epoch_features(np.asarray(epochs[idx]), np.asarray(masks[idx]))
                comp = compact_from_blocks(blocks, secondary_models)
                vectors.append(comp)
                manifest_rows.append({'source_checkpoint_row_0based': len(vectors) - 1, 'epoch_id': str(r.epoch_id), 'raw_behavior_row_id': str(r.raw_behavior_row_id), 'subject_id': subject, 'raw_run': run, 'raw_source_path': source, 'epoch_family': family, 'source_array_row_0based': idx, 'final_good_channels': int(r.final_good_channels)})
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape != (len(source_rows), 169):
            raise RuntimeError(f'{source}: compact checkpoint shape {arr.shape}')
        np.save(tmp / 'compact_float32.npy', arr, allow_pickle=False)
        pd.DataFrame(manifest_rows).to_csv(tmp / 'manifest.csv', index=False)
        summary = {'complete': True, 'phase': PHASE, 'subject_id': subject, 'raw_run': run, 'raw_source_path': source, 'rows': len(arr), 'cells_per_epoch': 169, 'input_signature_sha256': sig, 'executor_sha256': script_sha, 'outcome_free_features_source_sha256': EXPECTED_outcome_free_features_SHA, 'secondary_models_source_sha256': EXPECTED_secondary_models_SHA}
        summary['stored_files'] = stored_manifest(tmp, {'source_summary.json'})
        atom_json(tmp / 'source_summary.json', summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def combine_store(root: Path, source_dirs: list[Path], epoch_qc: pd.DataFrame, script_sha: str, recompute: bool):
    final = root / 'global'
    summary_path = final / 'global_summary.json'
    signature = csha({'epoch_ids': epoch_qc.epoch_id.astype(str).tolist(), 'source_summaries': [sha(d / 'source_summary.json') for d in source_dirs], 'executor': script_sha})
    if not recompute and checkpoint_valid(final, signature, script_sha):
        return (final, pd.read_csv(final / 'epoch_manifest.csv', low_memory=False))
    tmp = final.with_name('.global.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        records = []
        arrays = []
        for d in source_dirs:
            m = pd.read_csv(d / 'manifest.csv', low_memory=False)
            a = np.load(d / 'compact_float32.npy', mmap_mode='r', allow_pickle=False)
            if len(m) != a.shape[0]:
                raise RuntimeError(f'Checkpoint manifest mismatch: {d}')
            for r in m.itertuples(index=False):
                records.append(r._asdict())
                arrays.append(np.asarray(a[int(r.source_checkpoint_row_0based)], dtype=np.float32))
        manifest = pd.DataFrame(records)
        if len(manifest) != len(epoch_qc) or manifest.epoch_id.duplicated().any() or set(manifest.epoch_id) != set(epoch_qc.epoch_id):
            raise RuntimeError('Global compact manifest does not match validation_preprocessing_qc eligible epochs.')
        order = {eid: i for i, eid in enumerate(epoch_qc.epoch_id.astype(str))}
        manifest['epoch_compact_row_0based'] = manifest.epoch_id.astype(str).map(order).astype(int)
        source_array = np.asarray(arrays, dtype=np.float32)
        dest = np.empty((len(manifest), 169), dtype=np.float32)
        for src_i, r in enumerate(manifest.itertuples(index=False)):
            dest[int(r.epoch_compact_row_0based)] = source_array[src_i]
        manifest = manifest.sort_values('epoch_compact_row_0based').reset_index(drop=True)
        if np.isinf(dest).any():
            raise RuntimeError('Infinite compact values in global store.')
        np.save(tmp / 'epoch_compact_float32.npy', dest, allow_pickle=False)
        manifest.to_csv(tmp / 'epoch_manifest.csv', index=False)
        summary = {'complete': True, 'phase': PHASE, 'rows': len(dest), 'compact_cells_per_epoch': 169, 'input_signature_sha256': signature, 'executor_sha256': script_sha, 'stored_files': stored_manifest(tmp, {'global_summary.json'})}
        atom_json(tmp / 'global_summary.json', summary)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return (final, manifest)
    except Exception:
        (tmp / 'FAILED.txt').write_text(f'Failed {now()}', encoding='utf-8')
        raise

def participant_feature_audit(subject: str, meta: pd.DataFrame, raw: np.ndarray):
    rows = []
    retain = np.ones(169, dtype=bool)
    for j in range(169):
        fam_fracs = []
        for fam in FAMILIES:
            mask = meta.epoch_family.eq(fam).to_numpy()
            fam_fracs.append(float(np.isfinite(raw[mask, j]).mean()))
        ok = all((x >= FINITE_FRACTION for x in fam_fracs))
        retain[j] = ok
        rows.append({'subject_id': subject, 'compact_cell_0based': j, 'stimulus_early_finite_fraction': fam_fracs[0], 'stimulus_late_finite_fraction': fam_fracs[1], 'rating_early_finite_fraction': fam_fracs[2], 'finite_fraction_threshold': FINITE_FRACTION, 'passes_finite_fraction_rule': ok})
    return (pd.DataFrame(rows), np.flatnonzero(retain))

def make_policy(latent_state_fit):
    return latent_state_fit.prespecifiedPolicy(policy_sha256=EXPECTED_latent_state_design_POLICY_CANONICAL_SHA, candidate_k=[2, 3, 4, 5], pca_components=8, iqr_floor=1e-08, winsor_limit=6.0, covariance_type='diag', reg_covar=0.0001, n_init=20, max_iter=1000, stability_repeats=30, trial_fraction=0.8, median_ari_min=0.6, p10_ari_min=0.3, collapse_rate_max=0.1, min_state_observations=8, min_state_unique_trials=4, bic_delta=10.0, null_repeats=1000)

def build_trajectories(traj_manifest: pd.DataFrame, epoch_manifest: pd.DataFrame, compact: np.ndarray, latent_state_fit, schema: pd.DataFrame):
    complete = traj_manifest[traj_manifest.complete_three_epoch_trajectory.map(bval)].copy().sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id']).reset_index(drop=True)
    if len(complete) != 437:
        raise RuntimeError(f'Expected 437 complete trajectories, found {len(complete)}')
    idx = epoch_manifest.set_index(['raw_behavior_row_id', 'epoch_family']).epoch_compact_row_0based.to_dict()
    all_audit = []
    all_scores = []
    all_loadings = []
    all_prep = []
    all_traj = []
    sub_qc = []
    policy = make_policy(latent_state_fit)
    for subject, st in complete.groupby('subject_id', sort=True):
        st = st.sort_values(['trial_number', 'raw_behavior_row_id']).reset_index(drop=True)
        meta_rows = []
        raw_rows = []
        epoch_rows = []
        for r in st.itertuples(index=False):
            for fam in FAMILIES:
                key = (str(r.raw_behavior_row_id), fam)
                if key not in idx:
                    raise RuntimeError(f'{subject} missing compact epoch {key}')
                erow = int(idx[key])
                epoch_rows.append(erow)
                raw_rows.append(np.asarray(compact[erow], dtype=np.float32))
                meta_rows.append({'subject_id': subject, 'raw_behavior_row_id': str(r.raw_behavior_row_id), 'trial_number': int(r.trial_number), 'epoch_family': fam, 'epoch_compact_row_0based': erow})
        meta = pd.DataFrame(meta_rows)
        raw = np.asarray(raw_rows, dtype=np.float32)
        audit, retained = participant_feature_audit(subject, meta, raw)
        all_audit.append(audit)
        if len(retained) < MIN_RETAINED:
            raise RuntimeError(f'{subject}: only {len(retained)} features pass finite rule')
        data = latent_state_fit.SubjectData(subject_id=subject, trial_ids=meta.raw_behavior_row_id.to_numpy(str), trial_numbers=meta.trial_number.to_numpy(np.int32), epoch_rows=np.asarray(epoch_rows, dtype=np.int32), epoch_families=meta.epoch_family.to_numpy(str), compact=raw[:, retained], retained_features=retained.astype(np.int16), input_sha256=csha({'subject': subject, 'trial_ids': meta.raw_behavior_row_id.tolist(), 'epoch_rows': epoch_rows, 'compact_sha': hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()}))
        prep = latent_state_fit.preprocess_subject(data, policy)
        scores = np.asarray(prep['pca_scores'], float)
        score_frame = meta.copy()
        for j in range(scores.shape[1]):
            score_frame[f'pc{j + 1}'] = scores[:, j]
        all_scores.append(score_frame)
        for pc in range(scores.shape[1]):
            for local_idx, cell in enumerate(prep['final_features'].astype(int)):
                all_loadings.append({'subject_id': subject, 'pc_1based': pc + 1, 'compact_cell_0based': cell, 'feature_name': str(schema.loc[schema.compact_cell_0based.eq(cell), 'feature_name'].iloc[0]), 'pca_loading': float(prep['pca_components'][pc, local_idx]), 'explained_variance': float(prep['pca_explained_variance'][pc]), 'explained_variance_ratio': float(prep['pca_explained_variance_ratio'][pc])})
        for fam_i, fam in enumerate(FAMILIES):
            for local_idx, cell in enumerate(prep['final_features'].astype(int)):
                all_prep.append({'subject_id': subject, 'epoch_family': fam, 'compact_cell_0based': cell, 'feature_name': str(schema.loc[schema.compact_cell_0based.eq(cell), 'feature_name'].iloc[0]), 'family_median': float(prep['family_medians'][fam_i, local_idx]), 'family_scale': float(prep['family_scales'][fam_i, local_idx]), 'pooled_iqr': float(prep['pooled_iqr'][local_idx])})
        rows = latent_state_fit.continuous_trajectory_rows(subject, meta, scores)
        all_traj.extend(rows)
        finite = sum((np.isfinite([x[k] for k in ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']]).all() for x in rows))
        sub_qc.append({'subject_id': subject, 'complete_input_trajectories': len(rows), 'retained_compact_features_initial': len(retained), 'retained_compact_features_after_pooled_iqr': len(prep['final_features']), 'pca_components': scores.shape[1], 'finite_geometry_trajectories': int(finite), 'minimum_required': TRANSFER_MIN_PER_SUBJECT, 'eligible_for_validation_fixed_assignment': bool(finite >= TRANSFER_MIN_PER_SUBJECT)})
    return (complete, pd.concat(all_audit, ignore_index=True), pd.concat(all_scores, ignore_index=True), pd.DataFrame(all_loadings), pd.DataFrame(all_prep), pd.DataFrame(all_traj), pd.DataFrame(sub_qc))

def global_store_inventory(store: Path):
    rows = []
    for p in sorted((x for x in store.rglob('*') if x.is_file())):
        rows.append({'relative_path': str(p.relative_to(store)), 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    return pd.DataFrame(rows)

def self_test(project: Path):
    outcome_free_features = import_verified(project / outcome_free_features_REL, EXPECTED_outcome_free_features_SHA, 'outcome_free_features_selftest')
    secondary_models = import_verified(project / secondary_models_REL, EXPECTED_secondary_models_SHA, 'secondary_models_selftest')
    latent_state_fit = import_verified(project / latent_state_fit_REL, EXPECTED_latent_state_fit_SHA, 'latent_state_fit_selftest')
    rng = np.random.default_rng(540)
    epoch = rng.normal(scale=8e-06, size=(64, 2000))
    mask = np.ones(64, bool)
    mask[[2, 9, 41, 63]] = False
    epoch[~mask] = np.nan
    comp = compact_from_blocks(outcome_free_features.extract_epoch_features(epoch, mask), secondary_models)
    assert comp.shape == (169,) and np.isfinite(comp).all()
    meta = []
    raw = []
    for t in range(1, 21):
        for fi, fam in enumerate(FAMILIES):
            meta.append({'subject_id': 'sub-test', 'raw_behavior_row_id': f'trial-{t:03d}', 'trial_number': t, 'epoch_family': fam})
            raw.append(comp + rng.normal(scale=0.05, size=169) + fi * 0.02)
    meta = pd.DataFrame(meta)
    raw = np.asarray(raw, np.float32)
    retained = np.arange(169, dtype=np.int16)
    data = latent_state_fit.SubjectData('sub-test', meta.raw_behavior_row_id.to_numpy(str), meta.trial_number.to_numpy(np.int32), np.arange(len(meta), dtype=np.int32), meta.epoch_family.to_numpy(str), raw, retained, 'synthetic')
    prep = latent_state_fit.preprocess_subject(data, make_policy(latent_state_fit))
    rows = latent_state_fit.continuous_trajectory_rows('sub-test', meta, prep['pca_scores'])
    assert len(rows) == 20 and all((np.isfinite([r[x] for x in ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']]).all() for r in rows))
    print('validation_feature_trajectory self-test: PASS')

def main():
    args = parse_args()
    project = args.project.resolve()
    if args.self_test:
        self_test(project)
        return 0
    out = output_paths(project)
    out['log'].parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"validation_feature_trajectory outputs already exist: {out['contract']}; use --force after review.")
    started = time.time()
    logf = out['log'].open('a', encoding='utf-8')

    def note(msg):
        print(f'[{now()}] {msg}', flush=True)
        print(f'[{now()}] {msg}', file=logf, flush=True)
    script_sha = sha(Path(__file__).resolve())
    try:
        kpaths, ksummary, knext, kcontract, epoch, traj_manifest, subject_k, inventory = verify_k(project)
        transfer_policy, reference, latent, schema = verify_discovery_lineage(project)
        outcome_free_features = import_verified(project / outcome_free_features_REL, EXPECTED_outcome_free_features_SHA, 'prespecified_outcome_free_features')
        secondary_models = import_verified(project / secondary_models_REL, EXPECTED_secondary_models_SHA, 'prespecified_secondary_models')
        latent_state_fit = import_verified(project / latent_state_fit_REL, EXPECTED_latent_state_fit_SHA, 'prespecified_latent_state_fit')
        note('Verified validation_preprocessing_qc, validation_geometry_transfer_design, exact outcome_free_features/secondary_models/latent_state_fit code, and all fixed boundaries.')
        large_rehash = rehash_large_inventory(inventory)
        out['large_rehash'].parent.mkdir(parents=True, exist_ok=True)
        large_rehash.to_csv(out['large_rehash'], index=False)
        store = project / STORE_REL
        store.mkdir(parents=True, exist_ok=True)
        source_dirs = []
        sources = list(epoch.groupby(['raw_source_path', 'subject_id', 'raw_run'], sort=True))
        for i, ((source, subject, run), g) in enumerate(sources, 1):
            note(f'[{i}/{len(sources)}] Extracting/reusing exact compact features for {source} ({len(g)} epochs).')
            inv_source = inventory[inventory.raw_source_path.astype(str).eq(str(source))].copy()
            source_dirs.append(process_source(g.copy(), inv_source, store, outcome_free_features, secondary_models, script_sha, args.recompute_sources))
        global_dir, epoch_manifest = combine_store(store, source_dirs, epoch, script_sha, args.recompute_sources)
        compact = np.load(global_dir / 'epoch_compact_float32.npy', mmap_mode='r', allow_pickle=False)
        if compact.shape != (1424, 169):
            raise RuntimeError(f'Unexpected global compact shape {compact.shape}')
        complete, audit, scores, loadings, preprocessing, trajectories, participant_qc = build_trajectories(traj_manifest, epoch_manifest, compact, latent_state_fit, schema)
        geometry_cols = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
        trajectories['geometry_finite'] = np.isfinite(trajectories[geometry_cols]).all(axis=1)
        trajectories['geometry_exclusion_reason'] = np.where(trajectories.geometry_finite, '', 'nonfinite_raw_geometry')
        finite_count = int(trajectories.geometry_finite.sum())
        eligible_subjects = int(participant_qc.eligible_for_validation_fixed_assignment.sum())
        fraction = finite_count / 494.0
        gate = pd.DataFrame([{'gate_id': 'L1', 'metric': 'eligible_participants', 'observed': eligible_subjects, 'threshold': TRANSFER_MIN_PARTICIPANTS, 'comparison': '>=', 'pass': eligible_subjects >= TRANSFER_MIN_PARTICIPANTS}, {'gate_id': 'L2', 'metric': 'finite_geometry_trajectory_fraction_of_494', 'observed': fraction, 'threshold': TRANSFER_MIN_FRACTION, 'comparison': '>=', 'pass': fraction >= TRANSFER_MIN_FRACTION}, {'gate_id': 'L3', 'metric': 'exact_source_feature_preprocessing_hash_gates', 'observed': 1, 'threshold': 1, 'comparison': '==', 'pass': True}, {'gate_id': 'L4', 'metric': 'reports_outcomes_context_targets_read', 'observed': 0, 'threshold': 0, 'comparison': '==', 'pass': True}])
        permitted = bool(gate['pass'].all())
        out['feature_schema'].parent.mkdir(parents=True, exist_ok=True)
        schema.to_csv(out['feature_schema'], index=False)
        epoch_manifest.to_csv(out['epoch_manifest'], index=False)
        audit.to_csv(out['participant_feature_audit'], index=False)
        scores.to_csv(out['pca_scores'], index=False)
        loadings.to_csv(out['pca_loadings'], index=False)
        preprocessing.to_csv(out['preprocessing'], index=False)
        trajectories.to_csv(out['trajectories'], index=False)
        participant_qc.to_csv(out['participant_qc'], index=False)
        gate.to_csv(out['gate'], index=False)
        feature_qc = pd.DataFrame([{'subject_id': s, 'epoch_family': f, 'rows': len(g), 'finite_compact_cells': int(np.isfinite(np.asarray(compact[g.epoch_compact_row_0based.astype(int)])).sum()), 'total_compact_cells': int(len(g) * 169), 'infinite_cells': int(np.isinf(np.asarray(compact[g.epoch_compact_row_0based.astype(int)])).sum())} for (s, f), g in epoch_manifest.groupby(['subject_id', 'epoch_family'], sort=True)])
        feature_qc.to_csv(out['feature_qc'], index=False)
        checkpoint_rows = []
        for d in source_dirs:
            s = jread(d / 'source_summary.json')
            checkpoint_rows.append({'subject_id': s['subject_id'], 'raw_run': s['raw_run'], 'raw_source_path': s['raw_source_path'], 'checkpoint_directory': str(d), 'rows': s['rows'], 'input_signature_sha256': s['input_signature_sha256'], 'source_summary_sha256': sha(d / 'source_summary.json'), 'checkpoint_valid': True})
        pd.DataFrame(checkpoint_rows).to_csv(out['source_inventory'], index=False)
        store_inv = global_store_inventory(store)
        store_inv.to_csv(out['global_store_inventory'], index=False)
        required_hash_files = {'large_rehash': out['large_rehash'], 'source_inventory': out['source_inventory'], 'feature_qc': out['feature_qc'], 'participant_feature_audit': out['participant_feature_audit'], 'feature_schema': out['feature_schema'], 'epoch_manifest': out['epoch_manifest'], 'pca_scores': out['pca_scores'], 'pca_loadings': out['pca_loadings'], 'preprocessing': out['preprocessing'], 'trajectories': out['trajectories'], 'participant_qc': out['participant_qc'], 'gate': out['gate'], 'global_store_inventory': out['global_store_inventory']}
        primary_hashes = {k: sha(p) for k, p in required_hash_files.items()}
        next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': permitted, 'next_phase_role': 'apply the validation_geometry_transfer_design-prespecified monotone transforms, within-participant geometry scaling, two fixed Discovery centroids, support thresholds, and structural transfer gate without refitting or report/outcome/context-target access', 'required_inputs': primary_hashes, 'required_validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_TRANSFER_CANONICAL_SHA, 'required_validation_geometry_transfer_design_prespecified_core_sha256': EXPECTED_validation_geometry_transfer_design_CORE_SHA, 'expected_counts': {'validation_subjects': 12, 'authoritative_trials': 494, 'eligible_epochs': 1424, 'complete_three_epoch_trajectories': 437, 'finite_geometry_trajectories': finite_count}, 'forbidden': ['centroid refit', 'threshold tuning', 'family renaming', 'H1-H6 mapping', 'reports/outcomes/context targets', 'Confirmation access'], 'validation_opened_for_fixed_assignment_only': False, 'confirmation_opened': False}
        atom_json(out['next'], next_policy)
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'validation_exact_outcome_free_features_compact_features_and_latent_state_fit_trajectories_complete_candidate', 'source_code_lineage': {'outcome_free_features': EXPECTED_outcome_free_features_SHA, 'secondary_models': EXPECTED_secondary_models_SHA, 'latent_state_fit': EXPECTED_latent_state_fit_SHA, 'validation_geometry_transfer_design_core': EXPECTED_validation_geometry_transfer_design_CORE_SHA}, 'counts': {'validation_subjects': 12, 'authoritative_trials': 494, 'modeling_eligible_epochs': len(epoch), 'source_checkpoints': len(source_dirs), 'compact_cells_per_epoch': 169, 'complete_three_epoch_trajectories': len(complete), 'finite_geometry_trajectories': finite_count, 'eligible_participants_for_validation_fixed_assignment': eligible_subjects, 'global_store_files': len(store_inv), 'global_store_bytes': int(store_inv.size_bytes.sum())}, 'decision': {'all_validation_preprocessing_qc_hashes_verified': True, 'all_validation_preprocessing_qc_large_store_files_verified': True, 'exact_outcome_free_features_extractor_used': True, 'exact_secondary_models_compact_builder_used': True, 'exact_latent_state_fit_preprocessing_and_pca_used': True, 'reports_outcomes_read': False, 'context_target_read': False, 'stimulus_identity_read': False, 'family_assignment_performed': False, 'validation_opened_for_feature_extraction_only': True, 'confirmation_opened': False, 'validation_fixed_assignment_assignment_permitted': permitted}, 'runtime_seconds': time.time() - started}
        atom_json(out['summary'], summary)
        contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'executor_sha256': script_sha, 'source_code_lineage': summary['source_code_lineage'], 'validation_geometry_transfer_design_transfer_policy_canonical_sha256': EXPECTED_validation_geometry_transfer_design_TRANSFER_CANONICAL_SHA, 'latent_state_design_latent_policy_canonical_sha256': EXPECTED_latent_state_design_POLICY_CANONICAL_SHA, 'validation_preprocessing_qc_input_hashes': jread(kpaths['lock'])['hashes'], 'output_hashes': {**primary_hashes, 'summary': sha(out['summary']), 'next': sha(out['next'])}, 'large_store_root': str(store), 'large_arrays_embedded_in_release': False, 'boundary': summary['decision']}
        atom_json(out['contract'], contract)
        report = ''.join(['# Phase validation_feature_trajectory — Validation feature extraction and raw trajectory geometry', '', f'- Generated: `{now()}`', f'- Exact outcome_free_features source SHA-256: `{EXPECTED_outcome_free_features_SHA}`', f'- Exact secondary_models compact-builder SHA-256: `{EXPECTED_secondary_models_SHA}`', f'- Exact latent_state_fit trajectory-engine SHA-256: `{EXPECTED_latent_state_fit_SHA}`', f'- Modeling-eligible epochs: **{len(epoch)}**', f'- Complete three-epoch trajectories: **{len(complete)}**', f'- Finite raw geometries: **{finite_count}**', f'- Eligible participants for fixed assignment: **{eligible_subjects}/12**', f'- validation_fixed_assignment permitted: **{permitted}**', '', '## Boundary', '', '- Reports/outcomes read: **no**', '- Context used as a target: **no**', '- Discovery centroids applied: **no**', '- Family assignment performed: **no**', '- Confirmation opened: **no**', ''])
        out['report'].parent.mkdir(parents=True, exist_ok=True)
        out['report'].write_text(report, encoding='utf-8')
        lock_files = {**required_hash_files, 'summary': out['summary'], 'next': out['next'], 'contract': out['contract'], 'report': out['report']}
        atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': {k: sha(p) for k, p in lock_files.items()}, 'reports_outcomes_read': False, 'context_target_read': False, 'family_assignment_performed': False, 'confirmation_opened': False})
        files = list(lock_files.values()) + [out['lock'], project / outcome_free_features_REL, project / secondary_models_REL, project / latent_state_fit_REL]
        note(f'validation_feature_trajectory complete. Finite trajectories={finite_count}/{len(complete)}; eligible participants={eligible_subjects}/12; validation_fixed_assignment permitted={permitted}.')
        return 0
    finally:
        logf.close()
if __name__ == '__main__':
    raise SystemExit(main())