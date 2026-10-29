# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase confirmation_feature_trajectory — Confirmation compact features and trajectories.

This is a cohort/namespace adapter around the exact prespecified validation_feature_trajectory executor.
It performs only the validation_confirmation_synthesis-prespecified Confirmation transfer steps:

1. verify every confirmation_preprocessing_qc phase output and every large-store file byte-for-byte;
2. import the exact prespecified validation_feature_trajectory executor and its exact outcome_free_features/secondary_models/latent_state_fit
   source lineage;
3. extract the exact 169-cell outcome-free compact representation;
4. apply the exact latent_state_fit within-participant, epoch-family robust
   standardization and whitened PCA to complete three-epoch trajectories;
5. write raw six-coordinate trajectory geometry for confirmation_fixed_assignment.

No report, outcome, emotion category, stimulus identity, context target, or
Validation target is read. Context is not loaded into feature/geometry frames.
No Discovery centroid is applied and no family assignment is made here.

Large per-source checkpoints and the global compact array are stored outside
outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1/"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'confirmation_feature_trajectory'
NEXT_PHASE = 'confirmation_fixed_assignment'
PREFIX = 'confirmation_feature_trajectory'
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
PER_EPOCH_COMPACT = 169
EXPECTED_COUNTS = {'subjects': 12, 'trials': 494, 'eligible_epochs': 1413, 'complete_trajectories': 459, 'large_files': 253}
EXPECTED_outcome_free_features_SHA = 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba'
EXPECTED_secondary_models_SHA = '6cbe259167269fa1dc6927a749c3cfbf8492ac6b4ff4c048fc183f6082cf774d'
EXPECTED_latent_state_fit_SHA = 'c9f4c963d77a57d75e634fdbdabec5f2ae67ba1c0b76fe0a1621f3f266136c1d'
EXPECTED_validation_feature_trajectory_EXECUTOR_SHA = '444748f05f3c85eeacc183540df0180023ab5c4dba6c7c41b29c36a84d3c9f3c'
EXPECTED_latent_state_design_POLICY_CANONICAL_SHA = '1383dd7fd2faf6035391899dca9adad580b20e7d86ecf322acf7eddf7e3866d7'
EXPECTED_validation_confirmation_synthesis_TRANSFER_CANONICAL_SHA = '73df540947a247e541f20766de290b3c7cf6878ec2e1cc3bb17684f642c286c9'
EXPECTED_validation_confirmation_synthesis_CORE_SHA = '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b'
EXPECTED_validation_confirmation_synthesis_REFERENCE_SHA = '383c3528514944e8f76eb03a6b8ee39c22fa1ad5e6f53d6278e4691285380afa'
STORE_REL = Path('outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1')
outcome_free_features_REL = Path('src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py')
secondary_models_REL = Path('src/analysis_pipeline/eeg/secondary_models/secondary_models.py')
latent_state_fit_REL = Path('src/analysis_pipeline/eeg/latent_state_fit/latent_state_fit.py')
validation_feature_trajectory_REL = Path('src/analysis_pipeline/eeg/validation_feature_trajectory/validation_feature_trajectory.py')
latent_state_design_POLICY_REL = Path('analysis_specifications/S037_latent_state.json')
latent_state_design_CONTRACT_REL = Path('config/latent_state_design_latent_state_design_contract.json')
latent_state_design_SCHEMA_REL = Path('outputs/tables/latent_state_design_compact_feature_schema.csv')
validation_confirmation_synthesis_POLICY_REL = Path('analysis_specifications/S075_validation_confirmation_synthesis_geometry_transfer.json')
validation_confirmation_synthesis_REFERENCE_REL = Path('analysis_specifications/S074_validation_confirmation_synthesis_geometry_reference.json')
validation_confirmation_synthesis_CORE_REL = Path('analysis_specifications/reference_code/validation_confirmation_transfer_core.py')
validation_confirmation_synthesis_CONTRACT_REL = Path('config/validation_confirmation_synthesis_validation_confirmation_synthesis_contract.json')
P_REL = {'recording_qc': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_recording_channel_qc.csv'), 'epoch_qc': Path('outputs/tables/confirmation_preprocessing_qc_confirmation_epoch_amplitude_qc.csv'), 'inventory': Path('outputs/tables/confirmation_preprocessing_qc_confirmation_large_array_inventory.csv'), 'branch_qc': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_branch_retention_qc.csv'), 'subject_branch_qc': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_subject_branch_retention_qc.csv'), 'trajectory_manifest': Path('outputs/tables/confirmation_preprocessing_qc_confirmation_post_qc_trajectory_availability.csv'), 'subject_trajectory_qc': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_subject_trajectory_availability.csv'), 'transfer_gate': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_pre_feature_transfer_gate.csv'), 'source_rehash': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_source_rehash.csv'), 'summary': Path('outputs/qc/confirmation_preprocessing_qc_confirmation_preprocessing_qc_summary.json'), 'next': Path('analysis_specifications/S004_confirmation_preprocessing_qc_feature_trajectory.json'), 'contract': Path('config/confirmation_preprocessing_qc_confirmation_preprocessing_qc_contract.json'), 'report': Path('outputs/reports/confirmation_preprocessing_qc_confirmation_preprocessing_qc_report.md'), 'lock': Path('outputs/qc/confirmation_preprocessing_qc_final_output_lock_hashes.json')}
P_LOCK_PATHS = {k: v for k, v in P_REL.items() if k != 'lock'}
TRANSFER_MIN_PARTICIPANTS = 9
TRANSFER_MIN_FRACTION = 0.75
TRANSFER_MIN_PER_SUBJECT = 16
FINITE_FRACTION = 0.9
MIN_RETAINED = 30

def parse_args():
    p = argparse.ArgumentParser(description='Extract exact prespecified compact features and trajectories in Confirmation.')
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
    return {'log': project / 'outputs/logs/confirmation_feature_trajectory_confirmation_feature_trajectory.log', 'large_rehash': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_confirmation_preprocessing_qc_large_store_rehash.csv', 'source_inventory': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_feature_checkpoint_inventory.csv', 'feature_qc': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_compact_feature_qc.csv', 'participant_feature_audit': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_participant_feature_audit.csv', 'feature_schema': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_compact_feature_schema.csv', 'epoch_manifest': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_compact_epoch_manifest.csv', 'pca_scores': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_epoch_pca_scores.csv', 'pca_loadings': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_pca_loadings.csv', 'preprocessing': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_participant_preprocessing.csv', 'trajectories': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_raw_trajectory_geometry.csv', 'participant_qc': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_participant_trajectory_qc.csv', 'gate': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_pre_assignment_transfer_gate.csv', 'global_store_inventory': project / 'outputs/tables/confirmation_feature_trajectory_confirmation_compact_store_inventory.csv', 'summary': project / 'outputs/qc/confirmation_feature_trajectory_confirmation_feature_trajectory_summary.json', 'report': project / 'outputs/reports/confirmation_feature_trajectory_confirmation_feature_trajectory_report.md', 'contract': project / 'config/confirmation_feature_trajectory_confirmation_feature_trajectory_contract.json', 'next': project / 'analysis_specifications/S002_confirmation_feature_trajectory_fixed_assignment.json', 'lock': project / 'outputs/qc/confirmation_feature_trajectory_final_output_lock_hashes.json'}

def verify_p(project: Path):
    paths = {k: project / v for k, v in P_REL.items()}
    require(paths.values())
    lock = jread(paths['lock'])
    fixed = lock.get('hashes', {})
    for key in P_LOCK_PATHS:
        if key not in fixed:
            raise RuntimeError(f'confirmation_preprocessing_qc final lock missing {key}')
        if sha(paths[key]) != str(fixed[key]):
            raise RuntimeError(f'confirmation_preprocessing_qc fixed output mismatch: {key}')
    summary = jread(paths['summary'])
    nxt = jread(paths['next'])
    contract = jread(paths['contract'])
    if summary.get('phase') != 'confirmation_preprocessing_qc' or not summary.get('decision', {}).get('confirmation_feature_trajectory_feature_extraction_permitted'):
        raise RuntimeError('confirmation_preprocessing_qc did not permit confirmation_feature_trajectory.')
    if nxt.get('next_phase') != PHASE or not nxt.get('next_phase_permitted'):
        raise RuntimeError('confirmation_preprocessing_qc next-phase policy does not authorize confirmation_feature_trajectory.')
    if str(nxt.get('required_outcome_free_features_source_code_sha256')) != EXPECTED_outcome_free_features_SHA:
        raise RuntimeError('confirmation_preprocessing_qc expected different outcome_free_features code.')
    if str(summary.get('validation_confirmation_synthesis_transfer_policy_canonical_sha256')) != EXPECTED_validation_confirmation_synthesis_TRANSFER_CANONICAL_SHA:
        raise RuntimeError('confirmation_preprocessing_qc Confirmation transfer-policy lineage mismatch.')
    counts = summary.get('counts', {})
    expected = [('confirmation_subjects', 12), ('authoritative_trials', 494), ('modeling_eligible_epochs', 1413), ('complete_three_epoch_trajectories', 459), ('large_store_files', 253)]
    for k, v in expected:
        if int(counts.get(k, -1)) != v:
            raise RuntimeError(f'confirmation_preprocessing_qc count changed for {k}: {counts.get(k)}')
    decision = summary.get('decision', {})
    if decision.get('reports_outcomes_read') or decision.get('confirmation_reports_outcomes_read') or decision.get('validation_targets_read'):
        raise RuntimeError('confirmation_preprocessing_qc boundary was violated.')
    epoch = pd.read_csv(paths['epoch_qc'], usecols=['epoch_id', 'raw_behavior_row_id', 'subject_id', 'raw_run', 'raw_source_path', 'epoch_family', 'source_array_row_0based', 'final_good_channels', 'modeling_eligible'], low_memory=False)
    epoch = epoch[epoch.modeling_eligible.map(bval)].copy().reset_index(drop=True)
    if len(epoch) != 1413 or epoch.epoch_id.duplicated().any():
        raise RuntimeError('confirmation_preprocessing_qc eligible epoch manifest is invalid.')
    traj = pd.read_csv(paths['trajectory_manifest'], usecols=['raw_behavior_row_id', 'subject_id', 'trial_number', 'rating_early', 'stimulus_early', 'stimulus_late', 'complete_three_epoch_trajectory'], low_memory=False)
    if len(traj) != 494 or traj.raw_behavior_row_id.duplicated().any():
        raise RuntimeError('confirmation_preprocessing_qc trajectory manifest is invalid.')
    subj = pd.read_csv(paths['subject_trajectory_qc'], low_memory=False)
    inv = pd.read_csv(paths['inventory'], low_memory=False)
    if len(inv) != 253:
        raise RuntimeError('confirmation_preprocessing_qc large inventory row count changed.')
    if int(traj.complete_three_epoch_trajectory.map(bval).sum()) != 459:
        raise RuntimeError('confirmation_preprocessing_qc complete trajectory count changed.')
    return (paths, summary, nxt, contract, epoch, traj, subj, inv)

def verify_prespecified_lineage(project: Path):
    p_policy = project / validation_confirmation_synthesis_POLICY_REL
    p_ref = project / validation_confirmation_synthesis_REFERENCE_REL
    p_core = project / validation_confirmation_synthesis_CORE_REL
    p_contract_n = project / validation_confirmation_synthesis_CONTRACT_REL
    p_latent = project / latent_state_design_POLICY_REL
    p_contract = project / latent_state_design_CONTRACT_REL
    p_schema = project / latent_state_design_SCHEMA_REL
    p_l = project / validation_feature_trajectory_REL
    require([p_policy, p_ref, p_core, p_contract_n, p_latent, p_contract, p_schema, p_l])
    policy = jread(p_policy)
    ref = jread(p_ref)
    ncontract = jread(p_contract_n)
    latent = jread(p_latent)
    contract = jread(p_contract)
    if csha(policy) != EXPECTED_validation_confirmation_synthesis_TRANSFER_CANONICAL_SHA:
        raise RuntimeError('validation_confirmation_synthesis transfer policy canonical hash mismatch.')
    if sha(p_core) != EXPECTED_validation_confirmation_synthesis_CORE_SHA:
        raise RuntimeError('validation_confirmation_synthesis prespecified transfer core hash mismatch.')
    if sha(p_ref) != EXPECTED_validation_confirmation_synthesis_REFERENCE_SHA:
        raise RuntimeError('validation_confirmation_synthesis prespecified geometry reference hash mismatch.')
    if sha(p_l) != EXPECTED_validation_feature_trajectory_EXECUTOR_SHA:
        raise RuntimeError('Exact prespecified validation_feature_trajectory executor hash mismatch.')
    if csha(latent) != EXPECTED_latent_state_design_POLICY_CANONICAL_SHA:
        raise RuntimeError('latent_state_design latent policy canonical hash mismatch.')
    expected_schema = str(contract.get('output_hashes', {}).get('feature_schema', ''))
    if not expected_schema or sha(p_schema) != expected_schema:
        raise RuntimeError('latent_state_design compact feature schema hash mismatch.')
    schema = pd.read_csv(p_schema, low_memory=False)
    if len(schema) != 169 or not np.array_equal(schema.compact_cell_0based.to_numpy(int), np.arange(169)):
        raise RuntimeError('Compact schema invalid.')
    if policy.get('trajectory_construction', {}).get('epoch_order') != FAMILIES:
        raise RuntimeError('validation_confirmation_synthesis epoch order mismatch.')
    if ncontract.get('source_lineage', {}).get('validation_geometry_transfer_design_prespecified_transfer_core_sha256') != EXPECTED_validation_confirmation_synthesis_CORE_SHA:
        raise RuntimeError('validation_confirmation_synthesis contract core lineage mismatch.')
    lmod = import_verified(p_l, EXPECTED_validation_feature_trajectory_EXECUTOR_SHA, 'prespecified_validation_feature_trajectory_executor')
    lmod.PHASE = PHASE
    lmod.STORE_REL = STORE_REL
    return (policy, ref, latent, schema, lmod)

def build_trajectories(traj_manifest: pd.DataFrame, epoch_manifest: pd.DataFrame, compact: np.ndarray, latent_state_fit, schema: pd.DataFrame, lmod):
    complete = traj_manifest[traj_manifest.complete_three_epoch_trajectory.map(bval)].copy().sort_values(['subject_id', 'trial_number', 'raw_behavior_row_id']).reset_index(drop=True)
    if len(complete) != 459:
        raise RuntimeError(f'Expected 459 complete trajectories, found {len(complete)}')
    idx = epoch_manifest.set_index(['raw_behavior_row_id', 'epoch_family']).epoch_compact_row_0based.to_dict()
    all_audit = []
    all_scores = []
    all_loadings = []
    all_prep = []
    all_traj = []
    sub_qc = []
    policy = lmod.make_policy(latent_state_fit)
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
        audit, retained = lmod.participant_feature_audit(subject, meta, raw)
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
        geom = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
        finite = sum((np.isfinite([x[k] for k in geom]).all() for x in rows))
        sub_qc.append({'subject_id': subject, 'complete_input_trajectories': len(rows), 'retained_compact_features_initial': len(retained), 'retained_compact_features_after_pooled_iqr': len(prep['final_features']), 'pca_components': scores.shape[1], 'finite_geometry_trajectories': int(finite), 'minimum_required': TRANSFER_MIN_PER_SUBJECT, 'eligible_for_confirmation_fixed_assignment': bool(finite >= TRANSFER_MIN_PER_SUBJECT)})
    return (complete, pd.concat(all_audit, ignore_index=True), pd.concat(all_scores, ignore_index=True), pd.DataFrame(all_loadings), pd.DataFrame(all_prep), pd.DataFrame(all_traj), pd.DataFrame(sub_qc))

def global_store_inventory(store: Path):
    rows = []
    for p in sorted((x for x in store.rglob('*') if x.is_file())):
        rows.append({'relative_path': str(p.relative_to(store)), 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)})
    return pd.DataFrame(rows)

def self_test(project: Path):
    _, _, _, schema, lmod = verify_prespecified_lineage(project)
    outcome_free_features = import_verified(project / outcome_free_features_REL, EXPECTED_outcome_free_features_SHA, 'outcome_free_features_selftest_q')
    secondary_models = import_verified(project / secondary_models_REL, EXPECTED_secondary_models_SHA, 'secondary_models_selftest_q')
    latent_state_fit = import_verified(project / latent_state_fit_REL, EXPECTED_latent_state_fit_SHA, 'latent_state_fit_selftest_q')
    rng = np.random.default_rng(540)
    epoch = rng.normal(scale=8e-06, size=(64, 2000))
    mask = np.ones(64, bool)
    mask[[2, 9, 41, 63]] = False
    epoch[~mask] = np.nan
    comp = lmod.compact_from_blocks(outcome_free_features.extract_epoch_features(epoch, mask), secondary_models)
    assert comp.shape == (169,) and np.isfinite(comp).all() and (len(schema) == 169)
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
    prep = latent_state_fit.preprocess_subject(data, lmod.make_policy(latent_state_fit))
    rows = latent_state_fit.continuous_trajectory_rows('sub-test', meta, prep['pca_scores'])
    geom = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
    assert len(rows) == 20 and all((np.isfinite([r[x] for x in geom]).all() for r in rows))
    print('confirmation_feature_trajectory self-test: PASS')

def main():
    args = parse_args()
    project = args.project.resolve()
    if args.self_test:
        self_test(project)
        return 0
    out = output_paths(project)
    out['log'].parent.mkdir(parents=True, exist_ok=True)
    if out['contract'].exists() and (not args.force):
        raise RuntimeError(f"confirmation_feature_trajectory outputs already exist: {out['contract']}; use --force after review.")
    started = time.time()
    logf = out['log'].open('a', encoding='utf-8')

    def note(msg):
        print(f'[{now()}] {msg}', flush=True)
        print(f'[{now()}] {msg}', file=logf, flush=True)
    script_sha = sha(Path(__file__).resolve())
    try:
        ppaths, psummary, pnext, pcontract, epoch, traj_manifest, subject_p, inventory = verify_p(project)
        transfer_policy, reference, latent, schema, lmod = verify_prespecified_lineage(project)
        outcome_free_features = import_verified(project / outcome_free_features_REL, EXPECTED_outcome_free_features_SHA, 'prespecified_outcome_free_features_q')
        secondary_models = import_verified(project / secondary_models_REL, EXPECTED_secondary_models_SHA, 'prespecified_secondary_models_q')
        latent_state_fit = import_verified(project / latent_state_fit_REL, EXPECTED_latent_state_fit_SHA, 'prespecified_latent_state_fit_q')
        note('Verified confirmation_preprocessing_qc, validation_confirmation_synthesis, exact validation_feature_trajectory/outcome_free_features/secondary_models/latent_state_fit lineage, and all fixed boundaries.')
        large_rehash = lmod.rehash_large_inventory(inventory)
        out['large_rehash'].parent.mkdir(parents=True, exist_ok=True)
        large_rehash.to_csv(out['large_rehash'], index=False)
        store = project / STORE_REL
        store.mkdir(parents=True, exist_ok=True)
        source_dirs = []
        sources = list(epoch.groupby(['raw_source_path', 'subject_id', 'raw_run'], sort=True))
        for i, ((source, subject, run), g) in enumerate(sources, 1):
            note(f'[{i}/{len(sources)}] Extracting/reusing exact compact features for {source} ({len(g)} epochs).')
            inv_source = inventory[inventory.raw_source_path.astype(str).eq(str(source))].copy()
            source_dirs.append(lmod.process_source(g.copy(), inv_source, store, outcome_free_features, secondary_models, script_sha, args.recompute_sources))
        global_dir, epoch_manifest = lmod.combine_store(store, source_dirs, epoch, script_sha, args.recompute_sources)
        compact = np.load(global_dir / 'epoch_compact_float32.npy', mmap_mode='r', allow_pickle=False)
        if compact.shape != (1413, 169):
            raise RuntimeError(f'Unexpected global compact shape {compact.shape}')
        complete, audit, scores, loadings, preprocessing, trajectories, participant_qc = build_trajectories(traj_manifest, epoch_manifest, compact, latent_state_fit, schema, lmod)
        geometry_cols = ['early_to_late_norm', 'late_to_rating_norm', 'early_to_rating_direct_norm', 'total_path_length', 'directness_ratio', 'segment_cosine']
        trajectories['geometry_finite'] = np.isfinite(trajectories[geometry_cols]).all(axis=1)
        trajectories['geometry_exclusion_reason'] = np.where(trajectories.geometry_finite, '', 'nonfinite_raw_geometry')
        finite_count = int(trajectories.geometry_finite.sum())
        eligible_subjects = int(participant_qc.eligible_for_confirmation_fixed_assignment.sum())
        fraction = finite_count / 494.0
        gate = pd.DataFrame([{'gate_id': 'Q1', 'metric': 'eligible_participants', 'observed': eligible_subjects, 'threshold': TRANSFER_MIN_PARTICIPANTS, 'comparison': '>=', 'pass': eligible_subjects >= TRANSFER_MIN_PARTICIPANTS}, {'gate_id': 'Q2', 'metric': 'finite_geometry_trajectory_fraction_of_494', 'observed': fraction, 'threshold': TRANSFER_MIN_FRACTION, 'comparison': '>=', 'pass': fraction >= TRANSFER_MIN_FRACTION}, {'gate_id': 'Q3', 'metric': 'exact_source_feature_preprocessing_hash_gates', 'observed': 1, 'threshold': 1, 'comparison': '==', 'pass': True}, {'gate_id': 'Q4', 'metric': 'reports_outcomes_context_targets_or_validation_targets_read', 'observed': 0, 'threshold': 0, 'comparison': '==', 'pass': True}])
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
        next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': permitted, 'next_phase_role': 'apply the validation_confirmation_synthesis-prespecified monotone transforms, within-participant geometry scaling, two fixed Discovery centroids, support thresholds, and unchanged structural transfer gate without refitting or report/outcome/context-target/Validation-target access', 'required_inputs': primary_hashes, 'required_validation_confirmation_synthesis_transfer_policy_canonical_sha256': EXPECTED_validation_confirmation_synthesis_TRANSFER_CANONICAL_SHA, 'required_validation_confirmation_synthesis_prespecified_core_sha256': EXPECTED_validation_confirmation_synthesis_CORE_SHA, 'expected_counts': {'confirmation_subjects': 12, 'authoritative_trials': 494, 'eligible_epochs': 1413, 'complete_three_epoch_trajectories': 459, 'finite_geometry_trajectories': finite_count}, 'forbidden': ['centroid refit', 'threshold tuning', 'family renaming', 'H1-H6 mapping', 'reports/outcomes/context targets', 'Validation targets'], 'confirmation_opened_for_feature_extraction_only': True, 'confirmation_opened_for_fixed_assignment_only': False, 'confirmation_reports_outcomes_read': False, 'validation_targets_read': False}
        atom_json(out['next'], next_policy)
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'confirmation_exact_outcome_free_features_compact_features_and_latent_state_fit_trajectories_complete_candidate', 'source_code_lineage': {'outcome_free_features': EXPECTED_outcome_free_features_SHA, 'secondary_models': EXPECTED_secondary_models_SHA, 'latent_state_fit': EXPECTED_latent_state_fit_SHA, 'validation_feature_trajectory_executor': EXPECTED_validation_feature_trajectory_EXECUTOR_SHA, 'validation_confirmation_synthesis_core': EXPECTED_validation_confirmation_synthesis_CORE_SHA}, 'counts': {'confirmation_subjects': 12, 'authoritative_trials': 494, 'modeling_eligible_epochs': len(epoch), 'source_checkpoints': len(source_dirs), 'compact_cells_per_epoch': 169, 'complete_three_epoch_trajectories': len(complete), 'finite_geometry_trajectories': finite_count, 'eligible_participants_for_confirmation_fixed_assignment': eligible_subjects, 'global_store_files': len(store_inv), 'global_store_bytes': int(store_inv.size_bytes.sum())}, 'decision': {'all_confirmation_preprocessing_qc_hashes_verified': True, 'all_confirmation_preprocessing_qc_large_store_files_verified': True, 'exact_validation_feature_trajectory_executor_used_as_algorithm_reference': True, 'exact_outcome_free_features_extractor_used': True, 'exact_secondary_models_compact_builder_used': True, 'exact_latent_state_fit_preprocessing_and_pca_used': True, 'reports_outcomes_read': False, 'context_target_read': False, 'stimulus_identity_read': False, 'validation_targets_read': False, 'family_assignment_performed': False, 'confirmation_opened_for_feature_extraction_only': True, 'confirmation_reports_outcomes_read': False, 'confirmation_fixed_assignment_assignment_permitted': permitted}, 'runtime_seconds': time.time() - started}
        atom_json(out['summary'], summary)
        contract = {'generated_at': now(), 'phase': PHASE, 'status': summary['status'], 'executor_sha256': script_sha, 'source_code_lineage': summary['source_code_lineage'], 'validation_confirmation_synthesis_transfer_policy_canonical_sha256': EXPECTED_validation_confirmation_synthesis_TRANSFER_CANONICAL_SHA, 'latent_state_design_latent_policy_canonical_sha256': EXPECTED_latent_state_design_POLICY_CANONICAL_SHA, 'confirmation_preprocessing_qc_input_hashes': jread(ppaths['lock'])['hashes'], 'output_hashes': {**primary_hashes, 'summary': sha(out['summary']), 'next': sha(out['next'])}, 'large_store_root': str(store), 'large_arrays_embedded_in_release': False, 'boundary': summary['decision']}
        atom_json(out['contract'], contract)
        report = ''.join(['# Phase confirmation_feature_trajectory — Confirmation feature extraction and raw trajectory geometry', '', f'- Generated: `{now()}`', f'- Exact validation_feature_trajectory executor SHA-256: `{EXPECTED_validation_feature_trajectory_EXECUTOR_SHA}`', f'- Exact outcome_free_features source SHA-256: `{EXPECTED_outcome_free_features_SHA}`', f'- Exact secondary_models compact-builder SHA-256: `{EXPECTED_secondary_models_SHA}`', f'- Exact latent_state_fit trajectory-engine SHA-256: `{EXPECTED_latent_state_fit_SHA}`', f'- Modeling-eligible epochs: **{len(epoch)}**', f'- Complete three-epoch trajectories: **{len(complete)}**', f'- Finite raw geometries: **{finite_count}**', f'- Eligible participants for fixed assignment: **{eligible_subjects}/12**', f'- confirmation_fixed_assignment permitted: **{permitted}**', '', '## Boundary', '', '- Reports/outcomes read: **no**', '- Context used as a target: **no**', '- Validation targets read: **no**', '- Discovery centroids applied: **no**', '- Family assignment performed: **no**', '- Confirmation opened only for prespecified feature extraction: **yes**', ''])
        out['report'].parent.mkdir(parents=True, exist_ok=True)
        out['report'].write_text(report, encoding='utf-8')
        lock_files = {**required_hash_files, 'summary': out['summary'], 'next': out['next'], 'contract': out['contract'], 'report': out['report']}
        atom_json(out['lock'], {'generated_at': now(), 'phase': PHASE, 'hashes': {k: sha(p) for k, p in lock_files.items()}, 'reports_outcomes_read': False, 'context_target_read': False, 'validation_targets_read': False, 'family_assignment_performed': False, 'confirmation_opened_for_feature_extraction_only': True})
        files = list(lock_files.values()) + [out['lock'], project / outcome_free_features_REL, project / secondary_models_REL, project / latent_state_fit_REL, project / validation_feature_trajectory_REL, project / validation_confirmation_synthesis_POLICY_REL, project / validation_confirmation_synthesis_REFERENCE_REL, project / validation_confirmation_synthesis_CORE_REL]
        note(f'confirmation_feature_trajectory complete. Finite trajectories={finite_count}/{len(complete)}; eligible participants={eligible_subjects}/12; confirmation_fixed_assignment permitted={permitted}.')
        return 0
    finally:
        logf.close()
if __name__ == '__main__':
    raise SystemExit(main())