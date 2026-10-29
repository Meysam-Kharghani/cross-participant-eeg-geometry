# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase discovery_report_transition_multiresolution_fit — Discovery-only report-transition multi-resolution fit.

This phase executes the exact discovery_report_transition_multiresolution_fit policy prespecified by semantic_report_dissociation_design. It opens only:
  * the prespecified Discovery report-transition pair manifest;
  * the already-prespecified Discovery 169-cell compact EEG store;
  * stimulus_late and rating_early rows referenced by that manifest; and
  * the immutable discovery_stimulus_experience_multiresolution_fit experience-family assignments only AFTER the transition
    fits/eligibility decisions are fixed, for outcome-free structural comparison.

It does NOT read behavioral reports, nominal emotion categories, context or
stimulus identity as targets, physiology, Validation, or Confirmation.

To preserve numerical implementation identity with discovery_stimulus_experience_multiresolution_fit, discovery_report_transition_multiresolution_fit imports the exact
fixed discovery_stimulus_experience_multiresolution_fit source (verified by SHA-256) and reuses its robust standardization,
14-cell signature construction, participant weighting, weighted KMeans,
stability resampling, collapse metrics, and cross-resolution helpers.

The discovery_stimulus_experience_multiresolution_fit helper names its two generic stages "stimulus_early" and
"stimulus_late" in the preprocessing audit. discovery_report_transition_multiresolution_fit remaps those audit labels
immediately and deterministically to the actual stages:
  stimulus_early -> stimulus_late
  stimulus_late  -> rating_early
No numerical value is affected by this audit-label remapping."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
PHASE = 'discovery_report_transition_multiresolution_fit'
NEXT_PHASE = 'validation_report_transition_transfer_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
OUT_REL = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit')
CACHE_REL = Path('outputs/large/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_cache_v1')
semantic_report_dissociation_design_OUT = Path('outputs/interpretability/semantic_report_dissociation_design_semantic_report_dissociation_design_specification')
semantic_report_dissociation_design_LOCK = semantic_report_dissociation_design_OUT / 'qc/semantic_report_dissociation_design_final_output_lock_hashes.json'
semantic_report_dissociation_design_LOCK_SHA256 = '318879c9f111667d7f3027cc10d24b35aac0150da03d972c96c56cb1a3fe1f87'
semantic_report_dissociation_design_POLICY = Path('analysis_specifications/S060_semantic_report_dissociation_discovery_transition.json')
semantic_report_dissociation_design_POLICY_SHA256 = 'e3b8c87970da9387318e843cdf01994464fb13d3ca476dce5033c1091678515a'
semantic_report_dissociation_design_PAIR_MANIFEST = semantic_report_dissociation_design_OUT / 'tables/semantic_report_dissociation_design_prespecified_discovery_report_transition_pair_manifest.csv'
semantic_report_dissociation_design_PAIR_MANIFEST_SHA256 = '9deb3788034a3e1d458e2c66b125fb86d0766ada0a6c30ffe9066a5490341983'
DISCOVERY_COMPACT = Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy')
DISCOVERY_MANIFEST = Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_manifest.csv')
DISCOVERY_COMPACT_SHA256 = '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'
DISCOVERY_MANIFEST_SHA256 = '7c25a3f77a00fcccd0a03bade656d6066c0641a7e34a76f092fcc43171b296e1'
discovery_stimulus_experience_multiresolution_fit_SOURCE = Path('src/analysis_pipeline/eeg/discovery_stimulus_experience_multiresolution_fit/discovery_stimulus_experience_multiresolution_fit.py')
discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256 = 'd829035f335109b0faca87642d60dacc20e44e96e7abe501eff4f9a97b0844e9'
discovery_stimulus_experience_multiresolution_fit_ASSIGN = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv')
discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256 = '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'
EXPERIENCE_K = [2, 3, 4]
SIGNATURE_COORDINATES = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']
TRANSFORMED_COORDINATES = [f'transformed__{x}' for x in SIGNATURE_COORDINATES]
STANDARDIZED_COORDINATES = [f'standardized__{x}' for x in SIGNATURE_COORDINATES]

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readj(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def verify_exact(path: Path, expected: str, label: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f'{label} SHA-256 mismatch:\n  path: {path}\n  expected: {expected}\n  observed: {observed}')

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv.tmp', dir=path.parent, delete=False, encoding='utf-8') as f:
        tmp = Path(f.name)
        df.to_csv(f, index=False)
    os.replace(tmp, path)

def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json.tmp', dir=path.parent, delete=False, encoding='utf-8') as f:
        tmp = Path(f.name)
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '')
    os.replace(tmp, path)

def import_discovery_stimulus_experience_multiresolution_fit_core(project: Path):
    source = project / discovery_stimulus_experience_multiresolution_fit_SOURCE
    verify_exact(source, discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256, 'fixed discovery_stimulus_experience_multiresolution_fit numerical source')
    spec = importlib.util.spec_from_file_location('discovery_stimulus_experience_multiresolution_fit_fixed_numeric_core', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not construct import spec for discovery_stimulus_experience_multiresolution_fit numerical source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module

def verify_semantic_report_dissociation_design(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_exact(project / semantic_report_dissociation_design_LOCK, semantic_report_dissociation_design_LOCK_SHA256, 'semantic_report_dissociation_design final lock')
    lock = readj(project / semantic_report_dissociation_design_LOCK)
    if lock.get('phase') != 'semantic_report_dissociation_design':
        raise RuntimeError('semantic_report_dissociation_design final-lock phase drift')
    if lock.get('next_phase') != PHASE:
        raise RuntimeError(f'semantic_report_dissociation_design does not authorize {PHASE}')
    for key in ['new_signal_values_opened', 'rating_early_EEG_values_opened', 'behavioral_report_values_opened', 'semantic_relabeling_permitted']:
        if bool(lock.get(key)):
            raise RuntimeError(f'semantic_report_dissociation_design boundary drift: {key}=True')
    hashes = lock.get('hashes', {})
    if hashes.get(str(semantic_report_dissociation_design_POLICY)) != semantic_report_dissociation_design_POLICY_SHA256:
        raise RuntimeError('semantic_report_dissociation_design final lock policy reference drift')
    if hashes.get(str(semantic_report_dissociation_design_PAIR_MANIFEST)) != semantic_report_dissociation_design_PAIR_MANIFEST_SHA256:
        raise RuntimeError('semantic_report_dissociation_design final lock pair-manifest reference drift')
    for rel, expected in hashes.items():
        verify_exact(project / rel, expected, f'semantic_report_dissociation_design fixed child {rel}')
    verify_exact(project / semantic_report_dissociation_design_POLICY, semantic_report_dissociation_design_POLICY_SHA256, 'discovery_report_transition_multiresolution_fit prespecified policy')
    policy = readj(project / semantic_report_dissociation_design_POLICY)
    if policy.get('phase') != PHASE or policy.get('discovery_only') is not True:
        raise RuntimeError('discovery_report_transition_multiresolution_fit prespecified policy identity drift')
    permitted = policy['data_values_permitted_in_discovery_report_transition_multiresolution_fit']
    allowed_true = {k for k, v in permitted.items() if bval(v)}
    if allowed_true != {'stimulus_late_compact_EEG', 'rating_early_compact_EEG'}:
        raise RuntimeError(f'Unexpected discovery_report_transition_multiresolution_fit permission set: {sorted(allowed_true)}')
    if policy['multi_resolution_discovery']['candidate_K'] != [2, 3, 4, 5, 6, 7, 8]:
        raise RuntimeError('discovery_report_transition_multiresolution_fit candidate-K drift')
    if int(policy['source_store']['source_cells_per_epoch']) != 169:
        raise RuntimeError('discovery_report_transition_multiresolution_fit compact-width drift')
    if policy['representation']['stages'] != ['stimulus_late', 'rating_early']:
        raise RuntimeError('discovery_report_transition_multiresolution_fit stage drift')
    blocks = policy['representation']['blocks']
    if [(int(b['start']), int(b['stop'])) for b in blocks] != [(0, 60), (60, 72), (72, 84), (84, 124), (124, 145), (145, 169)]:
        raise RuntimeError('discovery_report_transition_multiresolution_fit compact block-boundary drift')
    return (lock, policy)

def load_prespecified_pairs(project: Path) -> pd.DataFrame:
    need = ['raw_behavior_row_id', 'subject_id', 'stimulus_late_row_0based', 'rating_early_row_0based', 'report_transition_paired_trials', 'report_transition_eligible']
    verify_exact(project / semantic_report_dissociation_design_PAIR_MANIFEST, semantic_report_dissociation_design_PAIR_MANIFEST_SHA256, 'semantic_report_dissociation_design prespecified Discovery report-transition pair manifest')
    header = pd.read_csv(project / semantic_report_dissociation_design_PAIR_MANIFEST, nrows=0)
    missing = set(need) - set(header.columns)
    if missing:
        raise RuntimeError(f'semantic_report_dissociation_design pair-manifest schema drift: {sorted(missing)}')
    df = pd.read_csv(project / semantic_report_dissociation_design_PAIR_MANIFEST, usecols=need, low_memory=False)
    df['raw_behavior_row_id'] = df['raw_behavior_row_id'].astype(str)
    df['subject_id'] = df['subject_id'].astype(str)
    df = df[df.report_transition_eligible.map(bval)].copy()
    if df.empty:
        raise RuntimeError('No semantic_report_dissociation_design report-transition-eligible pairs')
    if df.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Duplicate raw_behavior_row_id in report-transition cohort')
    if len(df) != 1237 or df.subject_id.nunique() != 31:
        raise RuntimeError(f'prespecified semantic_report_dissociation_design cohort drift: n={len(df)}, subjects={df.subject_id.nunique()}')
    if int(df.groupby('subject_id').size().min()) < 30:
        raise RuntimeError('prespecified semantic_report_dissociation_design minimum paired-trial rule drift')
    return df.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def remap_stage_audit(audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()
    mapping = {'stimulus_early': 'stimulus_late', 'stimulus_late': 'rating_early'}
    original = out['stage'].astype(str).copy()
    if set(original.unique()) != set(mapping):
        raise RuntimeError(f'Unexpected discovery_stimulus_experience_multiresolution_fit helper stage labels: {sorted(original.unique())}')
    out['stage'] = original.map(mapping)
    return out

def cache_meta(script_sha: str, compact_sha: str, manifest_sha: str, pair_sha: str, policy_sha: str, core_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'script_sha256': script_sha, 'Discovery_compact_sha256': compact_sha, 'Discovery_manifest_sha256': manifest_sha, 'semantic_report_dissociation_design_pair_manifest_sha256': pair_sha, 'discovery_report_transition_multiresolution_fit_policy_sha256': policy_sha, 'discovery_stimulus_experience_multiresolution_fit_numeric_core_sha256': core_sha}

def structural_comparison(base: pd.DataFrame, transition_assignments: pd.DataFrame, eligible_transition_k: list[int], experience_assignments_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not eligible_transition_k:
        return (pd.DataFrame(columns=['transition_k', 'experience_k', 'common_trials', 'ARI', 'NMI']), pd.DataFrame(columns=['transition_k', 'transition_family_1based', 'experience_k', 'experience_family_1based', 'trial_count', 'participant_weighted_mass']), pd.DataFrame(columns=['transition_k', 'transition_family_1based', 'experience_k', 'experience_family_1based', 'participant_weighted_mass', 'p_transition_given_experience', 'p_experience_given_transition']))
    exp = pd.read_csv(experience_assignments_path, usecols=['raw_behavior_row_id', 'subject_id', 'k', 'family_1based'], low_memory=False)
    exp['raw_behavior_row_id'] = exp['raw_behavior_row_id'].astype(str)
    exp['subject_id'] = exp['subject_id'].astype(str)
    exp = exp[exp['k'].astype(int).isin(EXPERIENCE_K)].copy()
    if exp.duplicated(['raw_behavior_row_id', 'subject_id', 'k']).any():
        raise RuntimeError('Duplicate immutable experience assignment key')
    weight = base[['raw_behavior_row_id', 'subject_id']].copy()
    weight['participant_weight'] = 1.0 / weight.groupby('subject_id')['subject_id'].transform('size').astype(float)
    metric_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    for tk in eligible_transition_k:
        tr = transition_assignments[transition_assignments['k'].astype(int).eq(int(tk))][['raw_behavior_row_id', 'subject_id', 'family_1based']].copy()
        tr = tr.rename(columns={'family_1based': 'transition_family_1based'})
        tr = tr.merge(weight, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
        for ek in EXPERIENCE_K:
            ex = exp[exp['k'].astype(int).eq(int(ek))][['raw_behavior_row_id', 'subject_id', 'family_1based']].copy()
            ex = ex.rename(columns={'family_1based': 'experience_family_1based'})
            j = tr.merge(ex, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
            if j.empty:
                raise RuntimeError(f'No common trials for transition K={tk}, experience K={ek}')
            metric_rows.append({'transition_k': int(tk), 'experience_k': int(ek), 'common_trials': int(len(j)), 'ARI': float(adjusted_rand_score(j['transition_family_1based'], j['experience_family_1based'])), 'NMI': float(normalized_mutual_info_score(j['transition_family_1based'], j['experience_family_1based']))})
            mass_by_t = j.groupby('transition_family_1based')['participant_weight'].sum().to_dict()
            mass_by_e = j.groupby('experience_family_1based')['participant_weight'].sum().to_dict()
            for tf in range(1, int(tk) + 1):
                for ef in range(1, int(ek) + 1):
                    cell = j[j.transition_family_1based.eq(tf) & j.experience_family_1based.eq(ef)]
                    mass = float(cell['participant_weight'].sum())
                    cell_rows.append({'transition_k': int(tk), 'transition_family_1based': tf, 'experience_k': int(ek), 'experience_family_1based': ef, 'trial_count': int(len(cell)), 'participant_weighted_mass': mass})
                    cond_rows.append({'transition_k': int(tk), 'transition_family_1based': tf, 'experience_k': int(ek), 'experience_family_1based': ef, 'participant_weighted_mass': mass, 'p_transition_given_experience': mass / float(mass_by_e.get(ef, 0.0)) if float(mass_by_e.get(ef, 0.0)) > 0 else np.nan, 'p_experience_given_transition': mass / float(mass_by_t.get(tf, 0.0)) if float(mass_by_t.get(tf, 0.0)) > 0 else np.nan})
    return (pd.DataFrame(metric_rows), pd.DataFrame(cell_rows), pd.DataFrame(cond_rows))

def self_test() -> None:
    assert PHASE == 'discovery_report_transition_multiresolution_fit'
    assert NEXT_PHASE == 'validation_report_transition_transfer_design'
    assert EXPERIENCE_K == [2, 3, 4]
    assert len(SIGNATURE_COORDINATES) == 14
    assert discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256 == 'd829035f335109b0faca87642d60dacc20e44e96e7abe501eff4f9a97b0844e9'
    x = pd.DataFrame({'stage': ['stimulus_early', 'stimulus_late']})
    y = remap_stage_audit(x)
    assert y['stage'].tolist() == ['stimulus_late', 'rating_early']
    print('discovery_report_transition_multiresolution_fit self-test PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--recompute-cache', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    print(f'[{now()}] verifying exact semantic_report_dissociation_design analysis specification and fixed discovery_stimulus_experience_multiresolution_fit numerical core', flush=True)
    semantic_report_dissociation_design_lock, policy = verify_semantic_report_dissociation_design(project)
    core = import_discovery_stimulus_experience_multiresolution_fit_core(project)
    verify_exact(project / DISCOVERY_COMPACT, DISCOVERY_COMPACT_SHA256, 'Discovery compact EEG array')
    verify_exact(project / DISCOVERY_MANIFEST, DISCOVERY_MANIFEST_SHA256, 'Discovery epoch manifest')
    verify_exact(project / discovery_stimulus_experience_multiresolution_fit_ASSIGN, discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256, 'immutable discovery_stimulus_experience_multiresolution_fit experience assignments')
    base = load_prespecified_pairs(project)
    pair_manifest_sha = sha256(project / semantic_report_dissociation_design_PAIR_MANIFEST)
    compact = np.load(project / DISCOVERY_COMPACT, mmap_mode='r', allow_pickle=False)
    if compact.ndim != 2 or compact.shape[1] != 169:
        raise RuntimeError(f'Unexpected Discovery compact shape: {compact.shape}')
    lrow = base.stimulus_late_row_0based.to_numpy(int)
    rrow = base.rating_early_row_0based.to_numpy(int)
    if np.min(lrow) < 0 or np.min(rrow) < 0 or np.max(lrow) >= len(compact) or (np.max(rrow) >= len(compact)):
        raise RuntimeError('prespecified semantic_report_dissociation_design pair manifest references compact rows outside the store')
    print(f'[{now()}] opening only prespecified Discovery stimulus_late/rating_early compact EEG values', flush=True)
    stimulus_late = np.asarray(compact[lrow], dtype=np.float64)
    rating_early = np.asarray(compact[rrow], dtype=np.float64)
    del compact
    blocks = policy['representation']['blocks']
    print(f'[{now()}] constructing the prespecified 14-cell report-transition signature', flush=True)
    full_sig = core.build_signatures(base, stimulus_late, rating_early, blocks)
    full_sig.stage_audit = remap_stage_audit(full_sig.stage_audit)
    mres = policy['multi_resolution_discovery']
    ks = [int(x) for x in mres['candidate_K']]
    n_init = int(mres['n_init'])
    max_iter = int(mres['max_iter'])
    tol = float(mres['tol'])
    stab = mres['stability']
    repeats = int(stab['repeats'])
    pfrac = float(stab['participant_fraction_without_replacement'])
    tfrac = float(stab['trial_fraction_within_retained_participant_without_replacement'])
    thr = stab['eligibility_thresholds']
    median_min = float(thr['median_ARI_at_least'])
    p10_min = float(thr['ARI_10th_percentile_at_least'])
    collapse_max = float(thr['collapse_rate_at_most'])
    min_participants = int(thr['minimum_participants_per_family'])
    min_trials = int(thr['minimum_trials_per_family'])
    min_occ = float(thr['minimum_participant_weighted_occupancy'])
    script_sha = sha256(Path(__file__).resolve())
    cache_root = project / CACHE_REL
    core.prepare_cache(cache_root, cache_meta(script_sha, DISCOVERY_COMPACT_SHA256, DISCOVERY_MANIFEST_SHA256, pair_manifest_sha, semantic_report_dissociation_design_POLICY_SHA256, discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256), args.recompute_cache)
    out = project / OUT_REL
    final_lock = out / 'qc/discovery_report_transition_multiresolution_fit_final_output_lock_hashes.json'
    if out.exists():
        if final_lock.is_file() and (not args.force):
            raise RuntimeError(f'Completed discovery_report_transition_multiresolution_fit output exists: {out}\nUse --force only if intentionally rematerializing identical analysis.')
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    participants = np.array(sorted(base.subject_id.astype(str).unique()), dtype=object)
    subjects = base.subject_id.astype(str).to_numpy()
    n_keep_p = max(max(ks), int(round(pfrac * len(participants))))
    full_fits = {}
    stability_parts: list[pd.DataFrame] = []
    for k in ks:
        print(f'[{now()}] K={k}: full participant-weighted fit + {repeats} prespecified two-level stability resamples', flush=True)
        kroot = cache_root / f'k{k}'
        kroot.mkdir(parents=True, exist_ok=True)
        fit_path = kroot / 'full_fit.npz'
        if fit_path.is_file():
            fit = core.load_fit(fit_path)
        else:
            fit = core.fit_weighted_kmeans(full_sig.standardized, full_sig.weights, k, core.stable_seed('discovery_report_transition_multiresolution_fit-full', k), n_init=n_init, max_iter=max_iter, tol=tol)
            core.save_fit(fit_path, fit)
        full_fits[k] = fit
        stab_path = kroot / 'stability.csv'
        existing = pd.read_csv(stab_path) if stab_path.is_file() else pd.DataFrame()
        done = set(existing['repeat'].astype(int)) if len(existing) else set()
        rows = existing.to_dict('records') if len(existing) else []
        for r in range(repeats):
            if r in done:
                continue
            rng = np.random.default_rng(core.stable_seed('discovery_report_transition_multiresolution_fit-resample', k, r))
            kept_p = rng.choice(participants, size=n_keep_p, replace=False)
            idx_parts: list[np.ndarray] = []
            for subject in kept_p:
                idx_s = np.flatnonzero(subjects == str(subject))
                n_keep_t = max(2, int(round(tfrac * len(idx_s))))
                idx_parts.append(rng.choice(idx_s, size=n_keep_t, replace=False))
            idx = np.sort(np.concatenate(idx_parts))
            sub_base = base.iloc[idx].reset_index(drop=True)
            try:
                sub_sig = core.build_signatures(sub_base, stimulus_late[idx], rating_early[idx], blocks)
                rf = core.fit_weighted_kmeans(sub_sig.standardized, sub_sig.weights, k, core.stable_seed('discovery_report_transition_multiresolution_fit-resample-fit', k, r), n_init=n_init, max_iter=max_iter, tol=tol)
                ari = float(adjusted_rand_score(fit.labels[idx], rf.labels))
                collapsed, minp, minn, mino = core.collapse_metrics(rf.labels, sub_base.subject_id.astype(str).to_numpy(), sub_sig.weights, k, min_participants, min_trials, min_occ)
                row = {'k': k, 'repeat': r, 'retained_participants': int(len(kept_p)), 'retained_trials': int(len(idx)), 'ari_vs_full': ari, 'collapsed': bool(collapsed), 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_participant_weighted_occupancy': mino, 'objective': rf.objective, 'converged': rf.converged, 'failure': ''}
            except Exception as exc:
                row = {'k': k, 'repeat': r, 'retained_participants': int(len(kept_p)), 'retained_trials': int(len(idx)), 'ari_vs_full': np.nan, 'collapsed': True, 'minimum_participants_per_family': 0, 'minimum_trials_per_family': 0, 'minimum_participant_weighted_occupancy': 0.0, 'objective': np.nan, 'converged': False, 'failure': f'{type(exc).__name__}: {exc}'}
            rows.append(row)
            if (r + 1) % 5 == 0 or r + 1 == repeats:
                atomic_csv(stab_path, pd.DataFrame(rows).sort_values('repeat'))
                print(f'  K={k}: stability checkpoint {r + 1}/{repeats}', flush=True)
        stability_parts.append(pd.DataFrame(rows).sort_values('repeat'))
    stability = pd.concat(stability_parts, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    assignment_rows: list[pd.DataFrame] = []
    centroid_rows: list[dict[str, Any]] = []
    for k in ks:
        fit = full_fits[k]
        labels_by_k[k] = fit.labels
        reps = stability[stability.k.eq(k)].copy()
        valid = reps[np.isfinite(reps.ari_vs_full)]
        collapsed, minp, minn, mino = core.collapse_metrics(fit.labels, subjects, full_sig.weights, k, min_participants, min_trials, min_occ)
        median_ari = float(valid.ari_vs_full.median()) if len(valid) else float('nan')
        p10 = float(valid.ari_vs_full.quantile(0.1)) if len(valid) else float('nan')
        collapse_rate = float(reps.collapsed.map(bval).mean()) if len(reps) else 1.0
        silhouette = core.weighted_silhouette(full_sig.standardized, fit.labels, full_sig.weights)
        eligible = bool(fit.converged and (not collapsed) and (len(valid) == repeats) and (median_ari >= median_min) and (p10 >= p10_min) and (collapse_rate <= collapse_max))
        metric_rows.append({'k': k, 'full_fit_converged': fit.converged, 'full_fit_iterations': fit.iterations, 'objective': fit.objective, 'participant_weighted_silhouette': silhouette, 'full_fit_collapsed': collapsed, 'minimum_participants_per_family': minp, 'minimum_trials_per_family': minn, 'minimum_participant_weighted_occupancy': mino, 'stability_valid_repeats': int(len(valid)), 'stability_failed_repeats': int(repeats - len(valid)), 'median_ari': median_ari, 'ari_10th_percentile': p10, 'collapse_rate': collapse_rate, 'eligible': eligible})
        adf = base[['raw_behavior_row_id', 'subject_id']].copy()
        adf['k'] = k
        adf['family_1based'] = fit.labels + 1
        adf['assigned_squared_distance'] = fit.assigned_squared_distance
        adf['assignment_margin'] = fit.assignment_margin
        adf['participant_weight'] = full_sig.weights
        assignment_rows.append(adf)
        for fam in range(k):
            idxf = fit.labels == fam
            row: dict[str, Any] = {'k': k, 'family_1based': fam + 1, 'trial_count': int(idxf.sum()), 'participant_count': int(len(set(subjects[idxf]))), 'participant_weighted_occupancy': float(full_sig.weights[idxf].sum() / full_sig.weights.sum())}
            for j, name in enumerate(STANDARDIZED_COORDINATES):
                row[name] = float(fit.centers[fam, j])
            centroid_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values('k').reset_index(drop=True)
    eligible_k = metrics.loc[metrics.eligible.map(bval), 'k'].astype(int).tolist()
    assignments = pd.concat(assignment_rows, ignore_index=True)
    centroids = pd.DataFrame(centroid_rows)
    cross_metrics, overlap, purity = core.cross_resolution_tables(base, full_sig.weights, labels_by_k, eligible_k)
    print(f'[{now()}] transition K eligibility fixed; computing outcome-free experience/transition overlap', flush=True)
    et_metrics, et_cells, et_conditionals = structural_comparison(base, assignments, eligible_k, project / discovery_stimulus_experience_multiresolution_fit_ASSIGN)
    sig_table = base[['raw_behavior_row_id', 'subject_id']].copy()
    for j, name in enumerate(SIGNATURE_COORDINATES):
        sig_table[name] = full_sig.raw[:, j]
    for j, name in enumerate(TRANSFORMED_COORDINATES):
        sig_table[name] = full_sig.transformed[:, j]
    for j, name in enumerate(STANDARDIZED_COORDINATES):
        sig_table[name] = full_sig.standardized[:, j]
    sig_table['participant_weight'] = full_sig.weights
    coord_table = pd.DataFrame({'coordinate_0based': np.arange(14, dtype=int), 'raw_name': SIGNATURE_COORDINATES, 'transformed_name': TRANSFORMED_COORDINATES, 'standardized_name': STANDARDIZED_COORDINATES, 'raw_transform': ['log1p' if i % 2 == 0 else 'arctanh_clip_1e-6' for i in range(14)]})
    zero_table = pd.DataFrame([{'signature_scope': k, 'trial_count': v} for k, v in sorted(full_sig.zero_norm_counts.items())])
    source_rehash = pd.DataFrame([{'source': 'semantic_report_dissociation_design_final_lock', 'path': str(semantic_report_dissociation_design_LOCK), 'sha256': semantic_report_dissociation_design_LOCK_SHA256, 'verified': True, 'values_opened': False}, {'source': 'semantic_report_dissociation_design_prespecified_report_transition_pair_manifest', 'path': str(semantic_report_dissociation_design_PAIR_MANIFEST), 'sha256': semantic_report_dissociation_design_PAIR_MANIFEST_SHA256, 'verified': True, 'values_opened': 'structural_identifiers_only'}, {'source': 'Discovery_compact_EEG_array', 'path': str(DISCOVERY_COMPACT), 'sha256': DISCOVERY_COMPACT_SHA256, 'verified': True, 'values_opened': 'stimulus_late_and_rating_early_rows_only'}, {'source': 'Discovery_epoch_manifest', 'path': str(DISCOVERY_MANIFEST), 'sha256': DISCOVERY_MANIFEST_SHA256, 'verified': True, 'values_opened': 'hash_verification_only'}, {'source': 'fixed_discovery_stimulus_experience_multiresolution_fit_numerical_core', 'path': str(discovery_stimulus_experience_multiresolution_fit_SOURCE), 'sha256': discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256, 'verified': True, 'values_opened': 'source_code_only'}, {'source': 'immutable_discovery_stimulus_experience_multiresolution_fit_experience_assignments', 'path': str(discovery_stimulus_experience_multiresolution_fit_ASSIGN), 'sha256': discovery_stimulus_experience_multiresolution_fit_ASSIGN_SHA256, 'verified': True, 'values_opened': 'after_transition_eligibility_for_outcome_free_structural_comparison_only'}])
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv', sig_table)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_family_candidate_metrics.csv', metrics)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_family_stability_resamples.csv', stability.sort_values(['k', 'repeat']))
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv', assignments.sort_values(['k', 'subject_id', 'raw_behavior_row_id']))
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_family_centroids_all_k.csv', centroids.sort_values(['k', 'family_1based']))
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_cross_resolution_metrics_eligible_k.csv', cross_metrics)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_family_overlap_eligible_k.csv', overlap)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_fine_to_coarse_parent_purity.csv', purity)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_signature_coordinate_schema.csv', coord_table)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_cross_metrics.csv', et_metrics)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_contingency.csv', et_cells)
    atomic_csv(out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_directional_conditionals.csv', et_conditionals)
    atomic_csv(out / 'qc/discovery_report_transition_multiresolution_fit_participant_stage_preprocessing_audit.csv', full_sig.stage_audit.sort_values(['subject_id', 'stage']))
    atomic_csv(out / 'qc/discovery_report_transition_multiresolution_fit_cross_person_standardization_audit.csv', full_sig.cross_person_audit.sort_values('subject_id'))
    atomic_csv(out / 'qc/discovery_report_transition_multiresolution_fit_zero_norm_cosine_audit.csv', zero_table)
    atomic_csv(out / 'qc/discovery_report_transition_multiresolution_fit_source_rehash.csv', source_rehash)
    status = 'one_or_more_stable_report_transition_resolutions' if eligible_k else 'no_stable_discrete_report_transition_resolution_continuous_signature_only'
    summary = {'phase': PHASE, 'generated_at': now(), 'status': status, 'cohort': {'Discovery_report_transition_eligible_participants': int(base.subject_id.nunique()), 'Discovery_report_transition_pairs': int(len(base)), 'minimum_pairs_per_participant': int(base.groupby('subject_id').size().min()), 'maximum_pairs_per_participant': int(base.groupby('subject_id').size().max())}, 'representation': {'source_compact_cells_per_epoch': 169, 'report_transition_signature_cells': 14, 'signature_coordinate_order': SIGNATURE_COORDINATES, 'stages': ['stimulus_late', 'rating_early']}, 'candidate_K': ks, 'eligible_K': eligible_k, 'candidate_metrics': metrics.to_dict('records'), 'experience_transition_structural_comparison_performed_after_transition_eligibility_lock': True, 'experience_reference_K': EXPERIENCE_K, 'scientific_boundaries': {'Discovery_stimulus_late_values_read': True, 'Discovery_rating_early_values_read': True, 'behavioral_report_values_read': False, 'emotion_category_values_read': False, 'context_as_target_values_read': False, 'stimulus_identity_as_target_values_read': False, 'physiology_values_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'experience_assignments_used_to_choose_transition_K': False, 'semantic_relabeling_permitted': False, 'distortion_dishonesty_suppression_exaggeration_claim_permitted': False}, 'next_phase': NEXT_PHASE, 'next_phase_role': 'if one or more transition K are eligible, specification one-shot Validation structural transfer for all eligible K; otherwise formally close the discrete transition-family sub-branch while retaining the continuous signature'}
    writej(out / 'qc/discovery_report_transition_multiresolution_fit_summary.json', summary)
    report_lines = ['# discovery_report_transition_multiresolution_fit — Discovery report-transition multi-resolution structural fit', '', f'Status: **{status}**', '', '## prespecified boundary', 'Only Discovery `stimulus_late` and `rating_early` compact EEG values were opened. Behavioral reports, nominal emotion category, context/stimulus targets, physiology, Validation, and Confirmation were not opened.', '', '## Cohort', f'- eligible participants: **{base.subject_id.nunique()}**', f'- paired `stimulus_late -> rating_early` trials: **{len(base)}**', '', '## Resolution scan', '```text', metrics.to_string(index=False), '```', '', f"Eligible report-transition K resolutions: **{(eligible_k if eligible_k else 'none')}**", '', 'All passing K values are retained. No behavior, emotion category, or experience-family label was allowed to choose K, centroids, or stability thresholds.', '', '## Outcome-free relationship to the experience structure']
    if len(et_metrics):
        report_lines += ['```text', et_metrics.to_string(index=False), '```']
    else:
        report_lines += ['No discrete report-transition K passed, so no discrete experience/transition comparison was materialized.']
    report_lines += ['', 'These overlap statistics are descriptive structural comparisons performed only after transition eligibility was fixed. They do not give semantic meaning to either partition.', '', '## Claim boundary', 'A stable `stimulus_late -> rating_early` family would describe report/evaluation-stage EEG reconfiguration only. It may not be called an emotion, cognitive state, distortion, dishonesty, suppression, exaggeration, or correction of the reported feeling.', '', '## Next', f'**{NEXT_PHASE}** — conditional design-specification for held-out transition transfer or closure of the discrete transition sub-branch.', '']
    (out / 'reports/discovery_report_transition_multiresolution_fit_report.md').write_text(''.join(report_lines), encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'status': status, 'source_code_sha256': script_sha, 'semantic_report_dissociation_design_final_lock_sha256': semantic_report_dissociation_design_LOCK_SHA256, 'discovery_report_transition_multiresolution_fit_policy_sha256': semantic_report_dissociation_design_POLICY_SHA256, 'Discovery_compact_sha256': DISCOVERY_COMPACT_SHA256, 'Discovery_manifest_sha256': DISCOVERY_MANIFEST_SHA256, 'semantic_report_dissociation_design_report_transition_pair_manifest_sha256': pair_manifest_sha, 'discovery_stimulus_experience_multiresolution_fit_numeric_core_sha256': discovery_stimulus_experience_multiresolution_fit_SOURCE_SHA256, 'candidate_K': ks, 'eligible_K': eligible_k, 'family_label_rule': 'lexicographic centroid order in the fixed 14-cell signature coordinate order', 'signature_coordinate_order': SIGNATURE_COORDINATES, 'behavioral_values_read': False, 'emotion_category_values_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'experience_assignments_used_to_choose_transition_K': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/discovery_report_transition_multiresolution_fit_contract.json', contract)
    lock_targets = [out / 'qc/discovery_report_transition_multiresolution_fit_contract.json', out / 'qc/discovery_report_transition_multiresolution_fit_summary.json', out / 'qc/discovery_report_transition_multiresolution_fit_participant_stage_preprocessing_audit.csv', out / 'qc/discovery_report_transition_multiresolution_fit_cross_person_standardization_audit.csv', out / 'qc/discovery_report_transition_multiresolution_fit_zero_norm_cosine_audit.csv', out / 'qc/discovery_report_transition_multiresolution_fit_source_rehash.csv', out / 'tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv', out / 'tables/discovery_report_transition_multiresolution_fit_family_candidate_metrics.csv', out / 'tables/discovery_report_transition_multiresolution_fit_family_stability_resamples.csv', out / 'tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv', out / 'tables/discovery_report_transition_multiresolution_fit_family_centroids_all_k.csv', out / 'tables/discovery_report_transition_multiresolution_fit_cross_resolution_metrics_eligible_k.csv', out / 'tables/discovery_report_transition_multiresolution_fit_family_overlap_eligible_k.csv', out / 'tables/discovery_report_transition_multiresolution_fit_fine_to_coarse_parent_purity.csv', out / 'tables/discovery_report_transition_multiresolution_fit_signature_coordinate_schema.csv', out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_cross_metrics.csv', out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_contingency.csv', out / 'tables/discovery_report_transition_multiresolution_fit_experience_transition_directional_conditionals.csv', out / 'reports/discovery_report_transition_multiresolution_fit_report.md']
    hashes = {str(p.relative_to(project)): sha256(p) for p in lock_targets if p.is_file()}
    final = {'phase': PHASE, 'generated_at': now(), 'status': status, 'hashes': hashes, 'candidate_K': ks, 'eligible_K': eligible_k, 'Discovery_stimulus_late_values_read': True, 'Discovery_rating_early_values_read': True, 'behavioral_values_read': False, 'emotion_category_values_read': False, 'context_as_target_values_read': False, 'stimulus_identity_as_target_values_read': False, 'physiology_values_read': False, 'Validation_signal_values_read': False, 'Confirmation_signal_values_read': False, 'experience_assignments_used_to_choose_transition_K': False, 'semantic_relabeling_permitted': False, 'next_phase': NEXT_PHASE}
    writej(final_lock, final)
    phase_dir = Path(__file__).resolve().parent
    print('', flush=True)
    print('discovery_report_transition_multiresolution_fit PASS', flush=True)
    print(f'Discovery report-transition participants: {base.subject_id.nunique()}', flush=True)
    print(f'Discovery report-transition pairs: {len(base)}', flush=True)
    print(f'Candidate K: {ks}', flush=True)
    print(f"Eligible K: {(eligible_k if eligible_k else 'NONE')}", flush=True)
    print('stimulus_late EEG opened: True', flush=True)
    print('rating_early EEG opened: True', flush=True)
    print('Behavior/emotion-category/context-target/stimulus-target opened: False', flush=True)
    print('Experience assignments used to choose transition K: False', flush=True)
    print('Validation opened: False', flush=True)
    print('Confirmation opened: False', flush=True)
    print('Semantic relabeling permitted: False', flush=True)
    print(f'Next phase: {NEXT_PHASE}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())