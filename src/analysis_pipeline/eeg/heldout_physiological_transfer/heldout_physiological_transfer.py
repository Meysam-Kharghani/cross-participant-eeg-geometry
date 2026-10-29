# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PHASE = 'heldout_physiological_transfer'
NEXT_PHASE = 'signal_space_feature_decomposition'
PHASE_SLUG = 'heldout_physiological_transfer_heldout_physiological_fingerprint_transfer'
OUT_REL = Path('outputs/interpretability') / PHASE_SLUG
CACHE_REL = Path('outputs/large') / f'{PHASE_SLUG}_bootstrap_v1'
POLICY_REL = Path('analysis_specifications/S034_heldout_physiological_transfer_signal_space_feature.json')
physiological_characterization_LOCK_REL = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design/qc/physiological_characterization_final_output_lock_hashes.json')
physiological_characterization_LOCK_SHA = '5105434cc56803cae4dca6e5008d243c956be2e0012602262311aefa5915d19b'
physiological_characterization_COORD_REL = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_coordinate_schema.csv')
physiological_characterization_COORD_SHA = '8d3d68f4f9884e1021e62bd22459f087c758c7095968d52ee045765df16415f7'
physiological_characterization_CONTRAST_REL = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_family_contrast_registry.csv')
physiological_characterization_CONTRAST_SHA = '3f82b2075f3b86c7561cd7738d0a6a94de49a461d6b26f42d490bb33d16ef286'
physiological_characterization_BLOCK_REL = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_feature_block_schema.csv')
physiological_characterization_BLOCK_SHA = 'd73f13f3bbacfc8f51bb01e4473b879770d697e68a91abe353d001b1b07d83b4'
discovery_physiological_reference_atlas_LOCK_REL = Path('outputs/interpretability/discovery_physiological_reference_atlas_discovery_physiological_reference_atlas/qc/discovery_physiological_reference_atlas_final_output_lock_hashes.json')
discovery_physiological_reference_atlas_LOCK_SHA = '7ce0d16d5bac520ced1c5e9c5cb26abddac270e309fa57fa67ab052a5be77a46'
discovery_physiological_reference_atlas_POLICY_REL = Path('analysis_specifications/S023_discovery_physiological_reference_atlas_heldout.json')
discovery_physiological_reference_atlas_POLICY_SHA = '948ab4c5934527e0ca9607ac06a6e8f574ad4496e62bd20e9f63d88ca7d9ef6b'
DISC_REF_REL = Path('outputs/interpretability/discovery_physiological_reference_atlas_discovery_physiological_reference_atlas/tables/discovery_physiological_reference_atlas_discovery_reference_contrast_vectors_14d.csv')
DISC_REF_SHA = '3154196fdf57086d5870011376b8cc65f33967aad8a7d722d7a67d405eea30fe'
DISC_COORD_REL = Path('outputs/interpretability/discovery_physiological_reference_atlas_discovery_physiological_reference_atlas/tables/discovery_physiological_reference_atlas_discovery_coordinate_contrasts.csv')
DISC_COORD_SHA = 'afc347f0e991adf077a33d4b3e78fe66178596f32053fb0dac2921a68cd4fb27'
DISC_BLOCK_REL = Path('outputs/interpretability/discovery_physiological_reference_atlas_discovery_physiological_reference_atlas/tables/discovery_physiological_reference_atlas_discovery_non_global_block_contributions.csv')
DISC_BLOCK_SHA = '2da42f6a811282001e752555595d3e1a5fda07562931b1ccc02f90e6be59dced'
HELDOUT = {('Validation', 'experience'): {'ks': [2, 3, 4], 'lock_rel': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'), 'lock_sha': '5c8431957ac2fe3ffdc8886724b7145359062f83580463b527937164daa9b0ca', 'signature_rel': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv'), 'signature_sha': '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298', 'assignment_rel': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv'), 'assignment_sha': 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'}, ('Confirmation', 'experience'): {'ks': [2, 3, 4], 'lock_rel': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/qc/confirmation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'), 'lock_sha': '3b6f03c0d44903345871470cc40e35b0d9d0aaa9d836954e1c0009a7485b6b8b', 'signature_rel': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'), 'signature_sha': '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d', 'assignment_rel': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'), 'assignment_sha': '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'}, ('Validation', 'report_transition'): {'ks': [2, 3], 'lock_rel': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/qc/validation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'), 'lock_sha': '8025e9b1ca3357ebcce055554bda3df6b12e1d1bae08737757cb586ba1faf715', 'signature_rel': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv'), 'signature_sha': 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067', 'assignment_rel': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv'), 'assignment_sha': '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'}, ('Confirmation', 'report_transition'): {'ks': [2, 3], 'lock_rel': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/qc/confirmation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'), 'lock_sha': '14bb0cb021ae071204c90e5be71959bacbc6fc2259e409ed111fc299578d9375', 'signature_rel': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv'), 'signature_sha': '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6', 'assignment_rel': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'), 'assignment_sha': 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'}}
N_BOOT = 2000
MIN_FINITE = 1900
SEED_BASE = 540100000
SPLIT_OFFSET = {'Validation': 1000, 'Confirmation': 2000}
DOMAIN_OFFSET = {'experience': 0, 'report_transition': 10000}
SIGN_EPS = 1e-12
COS_EPS = 1e-12

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')

def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

def sign_int(x: float, eps: float=SIGN_EPS) -> int:
    if not np.isfinite(x) or abs(float(x)) <= eps:
        return 0
    return 1 if x > 0 else -1

def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float=COS_EPS) -> float:
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= eps or (nb <= eps):
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))

def verify_exact(project: Path, rel: Path, expected_sha: str, role: str, rehash: list[dict[str, Any]], values_parsed: bool=False) -> Path:
    path = project / rel
    if not path.exists():
        raise FileNotFoundError(f'{role}: missing {path}')
    observed = sha256_path(path)
    ok = observed == expected_sha
    rehash.append({'role': role, 'path': str(rel), 'expected_sha256': expected_sha, 'observed_sha256': observed, 'verified': bool(ok), 'values_parsed': bool(values_parsed)})
    if not ok:
        raise RuntimeError(f'{role}: SHA drift: expected {expected_sha}, observed {observed}')
    return path

def verify_lock_all_children(project: Path, role: str, rel: Path, expected_sha: str, rehash: list[dict[str, Any]]) -> dict[str, Any]:
    p = verify_exact(project, rel, expected_sha, role, rehash, values_parsed=False)
    obj = readj(p)
    hashes = obj.get('hashes', {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{role}: lock contains no child hashes')
    for child_rel, child_sha in hashes.items():
        cp = project / child_rel
        if not cp.exists():
            raise FileNotFoundError(f'{role}: missing fixed child {child_rel}')
        obs = sha256_path(cp)
        if obs != child_sha:
            raise RuntimeError(f'{role}: child hash drift: {child_rel}')
    return obj

def participant_family_means(sig: pd.DataFrame, assignments: pd.DataFrame, k: int, coord_cols: list[str], domain: str, split: str, eligible_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ['raw_behavior_row_id', 'subject_id']
    if sig.duplicated(key).any():
        raise RuntimeError(f'{split}/{domain}: duplicate signature keys')
    aa = assignments.loc[assignments['k'].astype(int).eq(int(k))].copy()
    if eligible_only:
        if 'participant_eligible' not in aa.columns:
            raise RuntimeError(f'{split}/{domain}: missing participant_eligible')
        aa = aa.loc[bool_series(aa['participant_eligible'])].copy()
    if aa.empty:
        raise RuntimeError(f'{split}/{domain}: no assignments K={k}, eligible_only={eligible_only}')
    if aa.duplicated(key).any():
        raise RuntimeError(f'{split}/{domain}: duplicate assignment keys K={k}')
    required = key + coord_cols
    missing = [c for c in required if c not in sig.columns]
    if missing:
        raise RuntimeError(f'{split}/{domain}: missing signature columns: {missing}')
    m = aa[key + ['family_1based']].merge(sig[required], on=key, how='left', validate='one_to_one')
    if m[coord_cols].isna().any().any() or not np.isfinite(m[coord_cols].to_numpy(float)).all():
        raise RuntimeError(f'{split}/{domain}: missing/nonfinite signature values after join K={k}')
    gp = m.groupby(['subject_id', 'family_1based'], sort=True, observed=True)
    means = gp[coord_cols].mean().reset_index()
    counts = gp.size().rename('trial_count').reset_index()
    means = means.merge(counts, on=['subject_id', 'family_1based'], validate='one_to_one')
    means.insert(0, 'k', int(k))
    means.insert(0, 'domain', domain)
    means.insert(0, 'split', split)
    means.insert(4, 'eligible_only', bool(eligible_only))
    fam_rows: list[dict[str, Any]] = []
    for fam in range(1, int(k) + 1):
        pf = means.loc[means['family_1based'].astype(int).eq(fam)]
        if pf.empty:
            raise RuntimeError(f'{split}/{domain}: family {fam} absent K={k}')
        row: dict[str, Any] = {'split': split, 'domain': domain, 'k': int(k), 'family_1based': fam, 'eligible_only': bool(eligible_only), 'trial_count': int(m['family_1based'].astype(int).eq(fam).sum()), 'participant_count': int(pf['subject_id'].nunique())}
        for c in coord_cols:
            row[c] = float(pf[c].mean())
        fam_rows.append(row)
    return (means, pd.DataFrame(fam_rows))

def contrast_vector(fam: pd.DataFrame, domain: str, k: int, low: int, high: int, coord_cols: list[str]) -> np.ndarray:
    sub = fam.loc[(fam['domain'] == domain) & fam['k'].astype(int).eq(int(k))]
    lo = sub.loc[sub['family_1based'].astype(int).eq(int(low))]
    hi = sub.loc[sub['family_1based'].astype(int).eq(int(high))]
    if len(lo) != 1 or len(hi) != 1:
        raise RuntimeError(f'missing family centroid {domain} K{k} F{high}-F{low}')
    return hi.iloc[0][coord_cols].to_numpy(float) - lo.iloc[0][coord_cols].to_numpy(float)

def bootstrap_contrast(pfc: pd.DataFrame, low: int, high: int, coord_cols: list[str], ref: np.ndarray, seed: int, n_boot: int=N_BOOT) -> tuple[pd.DataFrame, int]:
    subjects = sorted(pfc['subject_id'].astype(str).unique().tolist())
    if not subjects:
        raise RuntimeError('bootstrap: no participants')
    lookup: dict[tuple[str, int], np.ndarray] = {}
    for _, r in pfc.iterrows():
        lookup[str(r['subject_id']), int(r['family_1based'])] = r[coord_cols].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    n = len(subjects)
    for b in range(int(n_boot)):
        sampled = rng.choice(subjects, size=n, replace=True)
        lo_vals = [lookup[str(s), int(low)] for s in sampled if (str(s), int(low)) in lookup]
        hi_vals = [lookup[str(s), int(high)] for s in sampled if (str(s), int(high)) in lookup]
        if not lo_vals or not hi_vals:
            continue
        v = np.mean(np.vstack(hi_vals), axis=0) - np.mean(np.vstack(lo_vals), axis=0)
        if not np.isfinite(v).all():
            continue
        row = {'bootstrap_index_0based': b, 'contrast_cosine': safe_cosine(ref, v)}
        for j, val in enumerate(v):
            row[f'coord_{j:02d}'] = float(val)
        rows.append(row)
    return (pd.DataFrame(rows), len(rows))

def q025975(a: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float('nan'), float('nan'))
    q = np.quantile(x, [0.025, 0.975], method='linear')
    return (float(q[0]), float(q[1]))

def block_contributions(v: np.ndarray, blocks: pd.DataFrame) -> list[dict[str, Any]]:
    denom = float(np.sum(np.asarray(v[:12], float) ** 2))
    rows: list[dict[str, Any]] = []
    for _, br in blocks.iterrows():
        i = int(br['rms_coordinate_0based'])
        j = int(br['cosine_coordinate_0based'])
        ss = float(v[i] ** 2 + v[j] ** 2)
        contrib = float(ss / denom) if denom > 1e-12 else 0.0
        rows.append({'block': str(br['block']), 'rms_coordinate_0based': i, 'cosine_coordinate_0based': j, 'block_squared_contrast': ss, 'non_global_block_contribution': contrib, 'above_uniform_reference': bool(contrib > 1.0 / 6.0)})
    return rows

def make_signal_space_feature_decomposition_policy(authorization_rel: str, authorization_sha: str, allowed: list[dict[str, Any]], source_block_schema_rel: str, source_block_schema_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'next_phase': NEXT_PHASE, 'authorization': 'signal_space_feature_decomposition may open prespecified 169-feature structural values only for contrast-block pairs explicitly authorized by heldout_physiological_transfer replicated-prominent-block gate', 'authorization_registry': {'path': authorization_rel, 'sha256': authorization_sha}, 'feature_block_schema': {'path': source_block_schema_rel, 'sha256': source_block_schema_sha}, 'allowed_contrast_block_pairs': allowed, 'selection_rule_was_prespecified_before_heldout_values': True, 'assignment_conditioned_gate': True, 'independent_replication_claim_permitted': False, 'new_block_or_feature_selection_permitted': False, 'all_features_within_authorized_block_required': True, 'heldout_behavioral_or_semantic_values_permitted': False, 'family_relabeling_permitted': False, 'K_reselection_permitted': False, 'topographic_or_source_localization_inference_permitted': False, 'crosssplit_p_value_combination_permitted': False, 'interpretation': 'feature decomposition may characterize reproducible signal-space contributors only; it cannot assign emotion semantics or causal generators'}

def self_test() -> None:
    assert sign_int(0.0) == 0 and sign_int(2.0) == 1 and (sign_int(-2.0) == -1)
    assert abs(safe_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-12
    assert safe_cosine(np.zeros(2), np.ones(2)) == 0.0
    cols = [f'x{i}' for i in range(14)]
    rows = []
    for s in ['s1', 's2', 's3']:
        for f in [1, 2]:
            r = {'split': 'V', 'domain': 'experience', 'k': 2, 'subject_id': s, 'family_1based': f}
            for i, c in enumerate(cols):
                r[c] = float((1 if f == 2 else 0) + i * 0.01)
            rows.append(r)
    pfc = pd.DataFrame(rows)
    ref = np.ones(14)
    draws, nf = bootstrap_contrast(pfc, 1, 2, cols, ref, seed=123, n_boot=20)
    assert nf == 20 and len(draws) == 20
    blocks = pd.DataFrame([{'block': f'b{i}', 'rms_coordinate_0based': 2 * i, 'cosine_coordinate_0based': 2 * i + 1} for i in range(6)])
    bc = block_contributions(np.arange(1, 15, dtype=float), blocks)
    assert abs(sum((x['non_global_block_contribution'] for x in bc)) - 1.0) < 1e-12
    probe_allowed = [{'contrast_id': 'x', 'block': 'b'}]
    probe = make_signal_space_feature_decomposition_policy('auth.csv', 'a' * 64, probe_allowed, 'blocks.csv', 'b' * 64)
    assert probe['authorization_registry'] == {'path': 'auth.csv', 'sha256': 'a' * 64}
    assert probe['feature_block_schema'] == {'path': 'blocks.csv', 'sha256': 'b' * 64}
    assert probe['allowed_contrast_block_pairs'] == probe_allowed
    print('heldout_physiological_transfer self-test PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.project:
        raise SystemExit('--project required')
    t0 = time.time()
    project = Path(args.project).resolve()
    out = project / OUT_REL
    cache = project / CACHE_REL
    if out.exists() and any(out.rglob('*')) and (not args.force):
        raise RuntimeError(f'nonempty heldout_physiological_transfer output already exists: {out}; quarantine it or use --force intentionally')
    if args.force:
        if out.exists():
            shutil.rmtree(out)
        if cache.exists():
            shutil.rmtree(cache)
    for d in [out / 'tables', out / 'qc', out / 'reports', cache]:
        d.mkdir(parents=True, exist_ok=True)
    rehash: list[dict[str, Any]] = []
    parent_refs: list[Path] = []
    physiological_characterization_lock = verify_exact(project, physiological_characterization_LOCK_REL, physiological_characterization_LOCK_SHA, 'physiological_characterization_final_lock', rehash)
    parent_refs.append(physiological_characterization_lock)
    discovery_physiological_reference_atlas_lock_obj = verify_lock_all_children(project, 'discovery_physiological_reference_atlas_final_lock', discovery_physiological_reference_atlas_LOCK_REL, discovery_physiological_reference_atlas_LOCK_SHA, rehash)
    parent_refs.append(project / discovery_physiological_reference_atlas_LOCK_REL)
    discovery_physiological_reference_atlas_policy_path = verify_exact(project, discovery_physiological_reference_atlas_POLICY_REL, discovery_physiological_reference_atlas_POLICY_SHA, 'discovery_physiological_reference_atlas_heldout_physiological_transfer_policy', rehash, values_parsed=True)
    parent_refs.append(discovery_physiological_reference_atlas_policy_path)
    policy = readj(discovery_physiological_reference_atlas_policy_path)
    if policy.get('next_phase') != PHASE:
        raise RuntimeError('discovery_physiological_reference_atlas policy authorization drift')
    if policy.get('behavioral_or_semantic_values_permitted') is not False:
        raise RuntimeError('discovery_physiological_reference_atlas policy behavioral boundary drift')
    if policy.get('K_reselection_permitted') is not False or policy.get('family_relabeling_permitted') is not False:
        raise RuntimeError('discovery_physiological_reference_atlas policy family/K boundary drift')
    b = policy.get('bootstrap', {})
    if int(b.get('n_resamples', -1)) != N_BOOT or int(b.get('finite_replicate_requirement', -1)) != MIN_FINITE:
        raise RuntimeError('discovery_physiological_reference_atlas bootstrap specification drift')
    coord_path = verify_exact(project, physiological_characterization_COORD_REL, physiological_characterization_COORD_SHA, 'physiological_characterization_coordinate_schema', rehash, values_parsed=True)
    contrast_reg_path = verify_exact(project, physiological_characterization_CONTRAST_REL, physiological_characterization_CONTRAST_SHA, 'physiological_characterization_contrast_registry', rehash, values_parsed=True)
    block_path = verify_exact(project, physiological_characterization_BLOCK_REL, physiological_characterization_BLOCK_SHA, 'physiological_characterization_block_schema', rehash, values_parsed=True)
    ref_path = verify_exact(project, DISC_REF_REL, DISC_REF_SHA, 'discovery_physiological_reference_atlas_discovery_reference_vectors', rehash, values_parsed=True)
    disc_coord_path = verify_exact(project, DISC_COORD_REL, DISC_COORD_SHA, 'discovery_physiological_reference_atlas_discovery_coordinate_contrasts', rehash, values_parsed=True)
    disc_block_path = verify_exact(project, DISC_BLOCK_REL, DISC_BLOCK_SHA, 'discovery_physiological_reference_atlas_discovery_block_contributions', rehash, values_parsed=True)
    parent_refs.extend([coord_path, contrast_reg_path, block_path, ref_path, disc_coord_path, disc_block_path])
    coord_schema = pd.read_csv(coord_path).sort_values('coordinate_0based')
    contrast_reg = pd.read_csv(contrast_reg_path)
    blocks = pd.read_csv(block_path)
    ref_df = pd.read_csv(ref_path)
    disc_coord = pd.read_csv(disc_coord_path)
    disc_block = pd.read_csv(disc_block_path)
    coord_cols = coord_schema['standardized_name'].astype(str).tolist()
    if len(coord_cols) != 14 or list(coord_schema['coordinate_0based'].astype(int)) != list(range(14)):
        raise RuntimeError('coordinate schema drift')
    if len(contrast_reg) != 14 or len(ref_df) != 14:
        raise RuntimeError('contrast registry/reference row count drift')
    if set(contrast_reg['contrast_id']) != set(ref_df['contrast_id']):
        raise RuntimeError('reference/registry contrast ID drift')
    if len(blocks) != 6:
        raise RuntimeError('block schema drift')
    heldout_data: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    cohort_rows: list[dict[str, Any]] = []
    for (split, domain), spec in HELDOUT.items():
        lock_obj = verify_lock_all_children(project, f'{split}_{domain}_final_lock', spec['lock_rel'], spec['lock_sha'], rehash)
        parent_refs.append(project / spec['lock_rel'])
        sig_path = verify_exact(project, spec['signature_rel'], spec['signature_sha'], f'{split}_{domain}_signatures', rehash, values_parsed=True)
        ass_path = verify_exact(project, spec['assignment_rel'], spec['assignment_sha'], f'{split}_{domain}_assignments', rehash, values_parsed=True)
        sig = pd.read_csv(sig_path)
        ass = pd.read_csv(ass_path)
        heldout_data[split, domain] = (sig, ass)
        for k in spec['ks']:
            ak = ass.loc[ass['k'].astype(int).eq(int(k))]
            eligible_mask = bool_series(ak['participant_eligible'])
            cohort_rows.append({'split': split, 'domain': domain, 'k': int(k), 'signature_rows': int(len(sig)), 'assignment_rows_at_k': int(len(ak)), 'all_assigned_participants': int(ak['subject_id'].nunique()), 'primary_eligible_assignment_rows': int(eligible_mask.sum()), 'primary_eligible_participants': int(ak.loc[eligible_mask, 'subject_id'].nunique()), 'behavioral_or_semantic_values_opened': False, 'structural_signature_values_opened': True})
    primary_pfc_parts: list[pd.DataFrame] = []
    primary_fam_parts: list[pd.DataFrame] = []
    sensitivity_fam: dict[tuple[str, str, int], pd.DataFrame] = {}
    for (split, domain), spec in HELDOUT.items():
        sig, ass = heldout_data[split, domain]
        for k in spec['ks']:
            pfc, fam = participant_family_means(sig, ass, int(k), coord_cols, domain, split, eligible_only=True)
            primary_pfc_parts.append(pfc)
            primary_fam_parts.append(fam)
            _, fam_all = participant_family_means(sig, ass, int(k), coord_cols, domain, split, eligible_only=False)
            sensitivity_fam[split, domain, int(k)] = fam_all
    primary_pfc = pd.concat(primary_pfc_parts, ignore_index=True)
    primary_fam = pd.concat(primary_fam_parts, ignore_index=True)
    point_rows: list[dict[str, Any]] = []
    coord_ci_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    boot_audit_rows: list[dict[str, Any]] = []
    boot_draw_parts: list[pd.DataFrame] = []
    ref_index = ref_df.set_index('contrast_id')
    coord_meta = coord_schema.set_index('coordinate_0based').to_dict('index')
    for _, rr in contrast_reg.iterrows():
        domain = str(rr['domain'])
        k = int(rr['k'])
        low = int(rr['family_low_1based'])
        high = int(rr['family_high_1based'])
        cid = str(rr['contrast_id'])
        ref = ref_index.loc[cid, coord_cols].to_numpy(float)
        if not np.isfinite(ref).all():
            raise RuntimeError(f'{cid}: nonfinite Discovery reference')
        dsign = np.array([sign_int(x) for x in ref], dtype=int)
        for split in ['Validation', 'Confirmation']:
            fam = primary_fam.loc[(primary_fam['split'] == split) & (primary_fam['domain'] == domain) & primary_fam['k'].astype(int).eq(k)]
            v = contrast_vector(fam, domain, k, low, high, coord_cols)
            hsign = np.array([sign_int(x) for x in v], dtype=int)
            concord = int(np.sum(dsign == hsign))
            cosine = safe_cosine(ref, v)
            seed = SEED_BASE + DOMAIN_OFFSET[domain] + SPLIT_OFFSET[split] + 100 * k + 10 * low + high
            pfc = primary_pfc.loc[(primary_pfc['split'] == split) & (primary_pfc['domain'] == domain) & primary_pfc['k'].astype(int).eq(k)]
            draws, nfinite = bootstrap_contrast(pfc, low, high, coord_cols, ref, seed, N_BOOT)
            if nfinite < MIN_FINITE:
                raise RuntimeError(f'{split}/{cid}: only {nfinite} finite bootstrap replicates (<{MIN_FINITE})')
            draws.insert(0, 'seed', int(seed))
            draws.insert(0, 'contrast_id', cid)
            draws.insert(0, 'split', split)
            boot_draw_parts.append(draws)
            cos_lo, cos_hi = q025975(draws['contrast_cosine'].to_numpy(float))
            point_rows.append({'split': split, 'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'orientation': 'family_high_minus_family_low', 'contrast_cosine_to_discovery': cosine, 'contrast_cosine_bootstrap_ci_low': cos_lo, 'contrast_cosine_bootstrap_ci_high': cos_hi, 'point_sign_concordance_n_of_14': concord, 'point_sign_concordance_fraction': concord / 14.0, 'fingerprint_split_gate': bool(cos_lo > 0.0 and concord >= 10), 'bootstrap_seed': int(seed), 'bootstrap_requested': N_BOOT, 'bootstrap_finite': int(nfinite)})
            boot_audit_rows.append({'split': split, 'domain': domain, 'k': k, 'contrast_id': cid, 'seed': int(seed), 'requested': N_BOOT, 'finite': int(nfinite), 'finite_requirement': MIN_FINITE, 'finite_requirement_met': bool(nfinite >= MIN_FINITE), 'ci_method': 'numpy.quantile linear 2.5/97.5'})
            for j, c in enumerate(coord_cols):
                lo_ci, hi_ci = q025975(draws[f'coord_{j:02d}'].to_numpy(float))
                coord_ci_rows.append({'split': split, 'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'coordinate_0based': j, 'standardized_name': c, 'block': coord_meta[j]['block'], 'metric': coord_meta[j]['metric'], 'discovery_contrast': float(ref[j]), 'heldout_contrast': float(v[j]), 'discovery_sign': int(dsign[j]), 'heldout_sign': int(hsign[j]), 'sign_matches_discovery': bool(dsign[j] == hsign[j]), 'bootstrap_ci_low': lo_ci, 'bootstrap_ci_high': hi_ci})
            for br in block_contributions(v, blocks):
                block_rows.append({'split': split, 'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, **br})
            fam_all = sensitivity_fam[split, domain, k]
            v_all = contrast_vector(fam_all, domain, k, low, high, coord_cols)
            s_all = np.array([sign_int(x) for x in v_all], dtype=int)
            srow = {'split': split, 'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'all_assigned_contrast_cosine_to_discovery': safe_cosine(ref, v_all), 'all_assigned_sign_concordance_n_of_14': int(np.sum(dsign == s_all)), 'primary_gate_can_be_changed_by_sensitivity': False}
            for sbr in block_contributions(v_all, blocks):
                srow[f"all_assigned_block_contribution__{sbr['block']}"] = float(sbr['non_global_block_contribution'])
            sensitivity_rows.append(srow)
    point_df = pd.DataFrame(point_rows)
    coord_df = pd.DataFrame(coord_ci_rows)
    block_df = pd.DataFrame(block_rows)
    sens_df = pd.DataFrame(sensitivity_rows)
    boot_audit = pd.DataFrame(boot_audit_rows)
    boot_draws = pd.concat(boot_draw_parts, ignore_index=True)
    gate_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    prominent_rows: list[dict[str, Any]] = []
    allowed_rows: list[dict[str, Any]] = []
    disc_coord_idx = disc_coord.set_index(['contrast_id', 'coordinate_0based'])
    disc_block_idx = disc_block.set_index(['contrast_id', 'block'])
    for _, rr in contrast_reg.iterrows():
        domain = str(rr['domain'])
        k = int(rr['k'])
        low = int(rr['family_low_1based'])
        high = int(rr['family_high_1based'])
        cid = str(rr['contrast_id'])
        pp = point_df.loc[point_df['contrast_id'] == cid].set_index('split')
        if set(pp.index) != {'Validation', 'Confirmation'}:
            raise RuntimeError(f'{cid}: heldout split rows missing')
        vg = bool(pp.loc['Validation', 'fingerprint_split_gate'])
        cg = bool(pp.loc['Confirmation', 'fingerprint_split_gate'])
        gate_rows.append({'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'validation_cosine': float(pp.loc['Validation', 'contrast_cosine_to_discovery']), 'validation_cosine_ci_low': float(pp.loc['Validation', 'contrast_cosine_bootstrap_ci_low']), 'validation_sign_concordance_n_of_14': int(pp.loc['Validation', 'point_sign_concordance_n_of_14']), 'validation_split_gate': vg, 'confirmation_cosine': float(pp.loc['Confirmation', 'contrast_cosine_to_discovery']), 'confirmation_cosine_ci_low': float(pp.loc['Confirmation', 'contrast_cosine_bootstrap_ci_low']), 'confirmation_sign_concordance_n_of_14': int(pp.loc['Confirmation', 'point_sign_concordance_n_of_14']), 'confirmation_split_gate': cg, 'fingerprint_direction_replicated_both_heldout': bool(vg and cg)})
        stable_by_coord: dict[int, bool] = {}
        for j, c in enumerate(coord_cols):
            drow = disc_coord_idx.loc[cid, j]
            dval = float(drow['contrast_high_minus_low'])
            dsign = sign_int(dval)
            vv = coord_df.loc[(coord_df['contrast_id'] == cid) & (coord_df['split'] == 'Validation') & coord_df['coordinate_0based'].astype(int).eq(j)].iloc[0]
            cc = coord_df.loc[(coord_df['contrast_id'] == cid) & (coord_df['split'] == 'Confirmation') & coord_df['coordinate_0based'].astype(int).eq(j)].iloc[0]
            vsign = sign_int(float(vv['heldout_contrast']))
            csign = sign_int(float(cc['heldout_contrast']))
            same_sign = bool(dsign != 0 and dsign == vsign == csign)
            if dsign > 0:
                v_excl = float(vv['bootstrap_ci_low']) > 0.0
                c_excl = float(cc['bootstrap_ci_low']) > 0.0
            elif dsign < 0:
                v_excl = float(vv['bootstrap_ci_high']) < 0.0
                c_excl = float(cc['bootstrap_ci_high']) < 0.0
            else:
                v_excl = c_excl = False
            stable = bool(same_sign and v_excl and c_excl)
            stable_by_coord[j] = stable
            stable_rows.append({'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'coordinate_0based': j, 'standardized_name': c, 'block': coord_meta[j]['block'], 'metric': coord_meta[j]['metric'], 'discovery_contrast': dval, 'validation_contrast': float(vv['heldout_contrast']), 'confirmation_contrast': float(cc['heldout_contrast']), 'same_nonzero_sign_all_three_splits': same_sign, 'validation_ci_excludes_zero_in_discovery_direction': bool(v_excl), 'confirmation_ci_excludes_zero_in_discovery_direction': bool(c_excl), 'stable_coordinate': stable})
        for _, br in blocks.iterrows():
            block = str(br['block'])
            i = int(br['rms_coordinate_0based'])
            j = int(br['cosine_coordinate_0based'])
            dcon = float(disc_block_idx.loc[(cid, block), 'non_global_block_contribution'])
            vb = block_df.loc[(block_df['contrast_id'] == cid) & (block_df['split'] == 'Validation') & (block_df['block'] == block)].iloc[0]
            cb = block_df.loc[(block_df['contrast_id'] == cid) & (block_df['split'] == 'Confirmation') & (block_df['block'] == block)].iloc[0]
            vcon = float(vb['non_global_block_contribution'])
            ccon = float(cb['non_global_block_contribution'])
            above_all = bool(dcon > 1 / 6 and vcon > 1 / 6 and (ccon > 1 / 6))
            stable_member = bool(stable_by_coord.get(i, False) or stable_by_coord.get(j, False))
            passed = bool(above_all and stable_member)
            prow = {'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'block': block, 'feature_start_0based_inclusive': int(br['feature_start_0based_inclusive']), 'feature_stop_0based_exclusive': int(br['feature_stop_0based_exclusive']), 'n_features': int(br['n_features']), 'rms_coordinate_0based': i, 'cosine_coordinate_0based': j, 'discovery_contribution': dcon, 'validation_contribution': vcon, 'confirmation_contribution': ccon, 'contribution_above_1_over_6_all_three': above_all, 'stable_member_coordinate_present': stable_member, 'replicated_prominent_block': passed, 'signal_space_feature_decomposition_authorized': passed}
            prominent_rows.append(prow)
            if passed:
                allowed_rows.append({'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'block': block, 'feature_start_0based_inclusive': int(br['feature_start_0based_inclusive']), 'feature_stop_0based_exclusive': int(br['feature_stop_0based_exclusive']), 'n_features': int(br['n_features'])})
    gate_df = pd.DataFrame(gate_rows)
    stable_df = pd.DataFrame(stable_rows)
    prominent_df = pd.DataFrame(prominent_rows)
    authorization_df = prominent_df.copy()
    if len(point_df) != 28 or len(gate_df) != 14 or len(coord_df) != 392 or (len(stable_df) != 196) or (len(block_df) != 168) or (len(prominent_df) != 84):
        raise RuntimeError('heldout_physiological_transfer output row count invariant failed')
    block_sums = block_df.groupby(['split', 'contrast_id'])['non_global_block_contribution'].sum()
    if not np.allclose(block_sums.to_numpy(float), 1.0, atol=1e-10, rtol=0):
        raise RuntimeError('heldout block contributions do not sum to one')
    table_files: list[Path] = []

    def save_csv(df: pd.DataFrame, filename: str) -> Path:
        p = out / 'tables' / filename
        df.to_csv(p, index=False)
        table_files.append(p)
        return p
    save_csv(primary_pfc, 'heldout_physiological_transfer_heldout_participant_family_centroids_primary.csv')
    save_csv(primary_fam, 'heldout_physiological_transfer_heldout_participant_balanced_family_centroids_primary.csv')
    save_csv(point_df, 'heldout_physiological_transfer_heldout_contrast_point_metrics.csv')
    save_csv(coord_df, 'heldout_physiological_transfer_heldout_coordinate_contrasts_with_bootstrap_ci.csv')
    save_csv(block_df, 'heldout_physiological_transfer_heldout_non_global_block_contributions.csv')
    save_csv(gate_df, 'heldout_physiological_transfer_fingerprint_direction_replication_gate.csv')
    save_csv(stable_df, 'heldout_physiological_transfer_stable_coordinate_registry.csv')
    authorization_path = save_csv(authorization_df, 'heldout_physiological_transfer_replicated_prominent_block_registry.csv')
    save_csv(sens_df, 'heldout_physiological_transfer_all_assigned_sensitivity.csv')
    boot_path = cache / 'heldout_physiological_transfer_bootstrap_contrast_draws.csv'
    boot_draws.to_csv(boot_path, index=False)
    boot_sha = sha256_path(boot_path)
    cache_meta = {'phase': PHASE, 'n_resamples_per_split_contrast': N_BOOT, 'rows': int(len(boot_draws)), 'file': str(boot_path.relative_to(project)), 'sha256': boot_sha, 'contains_behavioral_or_semantic_values': False}
    cache_meta_path = cache / 'cache_meta.json'
    writej(cache_meta_path, cache_meta)
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_path = out / 'qc' / 'heldout_physiological_transfer_heldout_source_cohort_audit.csv'
    cohort_df.to_csv(cohort_path, index=False)
    boot_audit_path = out / 'qc' / 'heldout_physiological_transfer_bootstrap_audit.csv'
    boot_audit.to_csv(boot_audit_path, index=False)
    rehash_path = out / 'qc' / 'heldout_physiological_transfer_source_rehash.csv'
    pd.DataFrame(rehash).to_csv(rehash_path, index=False)
    auth_rel = str(authorization_path.relative_to(project))
    auth_sha = sha256_path(authorization_path)
    signal_space_feature_decomposition_policy = make_signal_space_feature_decomposition_policy(auth_rel, auth_sha, allowed_rows, str(physiological_characterization_BLOCK_REL), physiological_characterization_BLOCK_SHA)
    signal_space_feature_decomposition_policy_path = project / POLICY_REL
    writej(signal_space_feature_decomposition_policy_path, signal_space_feature_decomposition_policy)
    summary = {'phase': PHASE, 'status': 'heldout_physiological_fingerprint_transfer_complete', 'next_phase': NEXT_PHASE, 'heldout_splits': ['Validation', 'Confirmation'], 'experience_K': [2, 3, 4], 'report_transition_K': [2, 3], 'n_contrasts': 14, 'n_fingerprint_direction_replicated': int(gate_df['fingerprint_direction_replicated_both_heldout'].sum()), 'n_stable_coordinates': int(stable_df['stable_coordinate'].sum()), 'n_replicated_prominent_contrast_blocks': int(prominent_df['replicated_prominent_block'].sum()), 'n_signal_space_feature_decomposition_authorized_contrast_blocks': int(len(allowed_rows)), 'bootstrap_resamples_per_split_contrast': N_BOOT, 'bootstrap_finite_minimum_observed': int(boot_audit['finite'].min()), 'behavioral_or_semantic_values_opened': False, 'heldout_structural_signature_values_opened': True, 'K_reselected': False, 'families_refit': False, 'family_ids_relabelled': False, 'new_inferential_p_values_run': False, 'topographic_extension_analysis_reopened': False, 'structural_replication_semantic_analysis_reopened': False, 'all_assigned_sensitivity_can_change_primary_gate': False, 'fingerprint_gate_is_assignment_conditioned': True, 'fingerprint_gate_is_independent_replication': False, 'runtime_seconds': round(time.time() - t0, 3)}
    summary_path = out / 'qc' / 'heldout_physiological_transfer_summary.json'
    writej(summary_path, summary)
    contract = {**summary, 'physiological_characterization_final_lock_sha256': physiological_characterization_LOCK_SHA, 'discovery_physiological_reference_atlas_final_lock_sha256': discovery_physiological_reference_atlas_LOCK_SHA, 'discovery_physiological_reference_atlas_heldout_physiological_transfer_policy_sha256': discovery_physiological_reference_atlas_POLICY_SHA, 'discovery_reference_vectors_sha256': DISC_REF_SHA, 'contrast_orientation': 'family_high_minus_family_low', 'fingerprint_direction_gate': 'both heldout: cosine bootstrap lower 95% CI > 0 and point sign concordance >=10/14', 'stable_coordinate_rule': 'same nonzero point sign all three splits and both heldout 95% CIs exclude zero in that direction', 'replicated_prominent_block_rule': 'block contribution >1/6 in D/V/C and at least one member coordinate stable', 'bootstrap': {'unit': 'participant', 'n': N_BOOT, 'minimum_finite': MIN_FINITE, 'seed_base': SEED_BASE}, 'interpretation_boundary': 'heldout family assignments were transferred from Discovery in the same 14D space; therefore heldout_physiological_transfer fingerprint alignment is assignment-conditioned descriptive stability, not an independent replication test'}
    contract_path = out / 'qc' / 'heldout_physiological_transfer_contract.json'
    writej(contract_path, contract)
    report_lines = ['# heldout_physiological_transfer — Held-out physiological fingerprint transfer', '', 'heldout_physiological_transfer opened only prespecified structural signature/assignment values from Validation and Confirmation.', 'No behavioral/semantic values, new family fitting, K selection, family relabeling, p-value inference, or topographic rescue were permitted.', '', f"- Fingerprint-direction replicated contrasts: {int(gate_df['fingerprint_direction_replicated_both_heldout'].sum())}/14", f"- Stable coordinate instances: {int(stable_df['stable_coordinate'].sum())}/196", f"- Replicated-prominent contrast-block pairs: {int(prominent_df['replicated_prominent_block'].sum())}/84", f'- signal_space_feature_decomposition-authorized contrast-block pairs: {len(allowed_rows)}', f"- Minimum finite bootstrap replicates: {int(boot_audit['finite'].min())}/{N_BOOT}", '', 'Because heldout family IDs were assigned by prespecified Discovery transfer in the same 14D space, this fingerprint gate is assignment-conditioned descriptive stability, not an independent replication test.', 'A passing fingerprint licenses only a reproducible transferred feature-space geometry statement.', 'It does not license emotion labels, causal neural generators, source localization, or reopening the topographic extension/structural_replication.']
    report_path = out / 'reports' / 'heldout_physiological_transfer_report.md'
    report_path.write_text(''.join(report_lines) + '', encoding='utf-8')
    files_to_lock = table_files + [cohort_path, boot_audit_path, rehash_path, summary_path, contract_path, report_path, signal_space_feature_decomposition_policy_path, cache_meta_path, boot_path]
    hashes = {str(p.relative_to(project)): sha256_path(p) for p in files_to_lock}
    final_lock = {'phase': PHASE, 'status': 'heldout_physiological_fingerprint_transfer_fixed', 'next_phase': NEXT_PHASE, 'generated_at': datetime.now().astimezone().isoformat(), 'physiological_characterization_final_lock_sha256': physiological_characterization_LOCK_SHA, 'discovery_physiological_reference_atlas_final_lock_sha256': discovery_physiological_reference_atlas_LOCK_SHA, 'behavioral_or_semantic_values_opened': False, 'heldout_structural_signature_values_opened': True, 'K_reselected': False, 'families_refit': False, 'family_ids_relabelled': False, 'new_inferential_p_values_run': False, 'fingerprint_gate_is_assignment_conditioned': True, 'fingerprint_gate_is_independent_replication': False, 'topographic_extension_analysis_reopened': False, 'structural_replication_semantic_analysis_reopened': False, 'hashes': hashes}
    lock_path = out / 'qc' / 'heldout_physiological_transfer_final_output_lock_hashes.json'
    writej(lock_path, final_lock)
    print('heldout_physiological_transfer PASS')
    print(f"Fingerprint-direction gate passes (assignment-conditioned): {int(gate_df['fingerprint_direction_replicated_both_heldout'].sum())}/14")
    print(f"Stable coordinate instances: {int(stable_df['stable_coordinate'].sum())}/196")
    print(f"Replicated-prominent contrast-block pairs: {int(prominent_df['replicated_prominent_block'].sum())}/84")
    print(f'signal_space_feature_decomposition-authorized contrast-block pairs: {len(allowed_rows)}')
    print(f"Minimum finite bootstrap replicates: {int(boot_audit['finite'].min())}/{N_BOOT}")
    print('Behavioral/semantic values opened: False')
    print('Heldout structural signature values opened: True')
    print('topographic/structural semantic-mapping branches reopened: False / False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())