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
PHASE = 'discovery_physiological_reference_atlas'
NEXT_PHASE = 'heldout_physiological_transfer'
PHASE_SLUG = 'discovery_physiological_reference_atlas_discovery_physiological_reference_atlas'
OUT_REL = Path('outputs/interpretability') / PHASE_SLUG
POLICY_REL = Path('analysis_specifications/S023_discovery_physiological_reference_atlas_heldout.json')
physiological_characterization_LOCK_REL = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design/qc/physiological_characterization_final_output_lock_hashes.json')
physiological_characterization_LOCK_SHA = '5105434cc56803cae4dca6e5008d243c956be2e0012602262311aefa5915d19b'
physiological_characterization_POLICY_REL = Path('analysis_specifications/S048_physiological_characterization.json')
physiological_characterization_POLICY_SHA = '66e1ddd4721c88c909f8b171f1fc38f8cdc50ce0999aa82cc125c24c042575d0'
physiological_characterization_COORD_REL = OUT_REL.parent / 'physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_coordinate_schema.csv'
physiological_characterization_COORD_SHA = '8d3d68f4f9884e1021e62bd22459f087c758c7095968d52ee045765df16415f7'
physiological_characterization_CONTRAST_REL = OUT_REL.parent / 'physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_family_contrast_registry.csv'
physiological_characterization_CONTRAST_SHA = '3f82b2075f3b86c7561cd7738d0a6a94de49a461d6b26f42d490bb33d16ef286'
physiological_characterization_BLOCK_REL = OUT_REL.parent / 'physiological_characterization_physiological_characterization_design/tables/physiological_characterization_prespecified_feature_block_schema.csv'
physiological_characterization_BLOCK_SHA = 'd73f13f3bbacfc8f51bb01e4473b879770d697e68a91abe353d001b1b07d83b4'
discovery_stimulus_experience_multiresolution_fit_LOCK_REL = Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json')
discovery_stimulus_experience_multiresolution_fit_LOCK_SHA = '35b9fa3efacfbe1e3884e2d622e1701538a5375fe2b72ffe1155fc429903c2b7'
discovery_report_transition_multiresolution_fit_LOCK_REL = Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/qc/discovery_report_transition_multiresolution_fit_final_output_lock_hashes.json')
discovery_report_transition_multiresolution_fit_LOCK_SHA = 'aa09937f090a17eadac760c38ebbe36310c5e88af2e92429f8e4680ca59a1473'
DISCOVERY = {'experience': {'ks': [2, 3, 4], 'lock': 'discovery_stimulus_experience_multiresolution_fit', 'signature_rel': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'), 'signature_sha': 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326', 'assignment_rel': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'), 'assignment_sha': '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5', 'centroid_rel': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_centroids_all_k.csv'), 'centroid_sha': '9f2913d4df1774d91e19f8106d320daace75b2297a26ac94472d71cab84d0a05', 'coord_rel': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_signature_coordinate_schema.csv'), 'coord_sha': '383907c12aa4c7bdffd41e11c321525a3c3588199a7e7a7819970a92fa0284fd'}, 'report_transition': {'ks': [2, 3], 'lock': 'discovery_report_transition_multiresolution_fit', 'signature_rel': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv'), 'signature_sha': '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83', 'assignment_rel': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'), 'assignment_sha': '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a', 'centroid_rel': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_centroids_all_k.csv'), 'centroid_sha': 'a9484298b3a55bdec096b7e09efb1fb2a770901fd41f9f5c00579c26dc336cff', 'coord_rel': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_signature_coordinate_schema.csv'), 'coord_sha': '383907c12aa4c7bdffd41e11c321525a3c3588199a7e7a7819970a92fa0284fd'}}

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

def verify_exact(project: Path, rel: Path, expected_sha: str, role: str, rehash: list[dict[str, Any]], values_parsed: bool=False) -> Path:
    p = project / rel
    if not p.exists():
        raise RuntimeError(f'missing {role}: {rel}')
    obs = sha256_path(p)
    ok = obs == expected_sha
    rehash.append({'role': role, 'path': str(rel), 'expected_sha256': expected_sha, 'observed_sha256': obs, 'verified': ok, 'values_parsed': values_parsed})
    if not ok:
        raise RuntimeError(f'{role} hash drift: {rel}')
    return p

def verify_lock(project: Path, role: str, rel: Path, expected_sha: str, rehash: list[dict[str, Any]]) -> dict[str, Any]:
    p = verify_exact(project, rel, expected_sha, f'{role}_final_lock', rehash, False)
    lock = readj(p)
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{role} final lock has no child hashes')
    for child_rel, child_sha in hashes.items():
        cp = project / child_rel
        if not cp.exists():
            raise RuntimeError(f'{role} fixed child missing: {child_rel}')
        if sha256_path(cp) != child_sha:
            raise RuntimeError(f'{role} child drift: {child_rel}')
    return lock

def sign_int(x: float, eps: float=1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0

def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float=1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def participant_balanced_atlas(sig: pd.DataFrame, assignments: pd.DataFrame, ks: list[int], coord_cols: list[str], domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ['raw_behavior_row_id', 'subject_id']
    if sig.duplicated(key).any():
        raise RuntimeError(f'{domain}: duplicate signature keys')
    required_sig = set(key + coord_cols)
    if not required_sig.issubset(sig.columns):
        raise RuntimeError(f'{domain}: missing signature columns: {sorted(required_sig - set(sig.columns))}')
    required_a = set(key + ['k', 'family_1based'])
    if not required_a.issubset(assignments.columns):
        raise RuntimeError(f'{domain}: missing assignment columns')
    pfc_rows: list[pd.DataFrame] = []
    fam_rows: list[dict[str, Any]] = []
    for k in ks:
        aa = assignments.loc[assignments['k'].astype(int).eq(k), key + ['family_1based']].copy()
        if aa.empty:
            raise RuntimeError(f'{domain}: no assignments for K={k}')
        if aa.duplicated(key).any():
            raise RuntimeError(f'{domain}: duplicate assignment keys at K={k}')
        m = aa.merge(sig[key + coord_cols], on=key, how='left', validate='one_to_one')
        if m[coord_cols].isna().any().any():
            raise RuntimeError(f'{domain}: missing signature values after K={k} join')
        if not np.isfinite(m[coord_cols].to_numpy(float)).all():
            raise RuntimeError(f'{domain}: non-finite signature values')
        gp = m.groupby(['subject_id', 'family_1based'], sort=True, observed=True)
        means = gp[coord_cols].mean().reset_index()
        counts = gp.size().rename('trial_count').reset_index()
        means = means.merge(counts, on=['subject_id', 'family_1based'], validate='one_to_one')
        means.insert(0, 'k', k)
        means.insert(0, 'domain', domain)
        pfc_rows.append(means)
        for fam in range(1, k + 1):
            pf = means.loc[means['family_1based'].astype(int).eq(fam)]
            if pf.empty:
                raise RuntimeError(f'{domain}: family {fam} absent at K={k}')
            row: dict[str, Any] = {'domain': domain, 'k': k, 'family_1based': fam, 'trial_count': int(m['family_1based'].astype(int).eq(fam).sum()), 'participant_count': int(pf['subject_id'].nunique())}
            for c in coord_cols:
                row[c] = float(pf[c].mean())
            fam_rows.append(row)
    return (pd.concat(pfc_rows, ignore_index=True), pd.DataFrame(fam_rows))

def make_contrasts(fam: pd.DataFrame, registry: pd.DataFrame, coord_cols: list[str], coord_schema: pd.DataFrame, blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vec_rows: list[dict[str, Any]] = []
    coord_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    coord_meta = coord_schema.set_index('standardized_name').to_dict('index')
    block_pairs = []
    for _, br in blocks.iterrows():
        block_pairs.append((str(br['block']), int(br['rms_coordinate_0based']), int(br['cosine_coordinate_0based'])))
    for _, rr in registry.iterrows():
        domain = str(rr['domain'])
        k = int(rr['k'])
        low = int(rr['family_low_1based'])
        high = int(rr['family_high_1based'])
        cid = str(rr['contrast_id'])
        sub = fam.loc[(fam['domain'] == domain) & (fam['k'].astype(int) == k)]
        lo = sub.loc[sub['family_1based'].astype(int).eq(low)]
        hi = sub.loc[sub['family_1based'].astype(int).eq(high)]
        if len(lo) != 1 or len(hi) != 1:
            raise RuntimeError(f'missing family centroid for {cid}')
        lv = lo.iloc[0][coord_cols].to_numpy(float)
        hv = hi.iloc[0][coord_cols].to_numpy(float)
        v = hv - lv
        norm14 = float(np.linalg.norm(v))
        norm12 = float(np.linalg.norm(v[:12]))
        global_norm = float(np.linalg.norm(v[12:14]))
        row = {'domain': domain, 'k': k, 'family_low_1based': low, 'family_high_1based': high, 'contrast_id': cid, 'orientation': 'family_high_minus_family_low', 'contrast_norm_14d': norm14, 'contrast_norm_non_global_12d': norm12, 'global_pair_norm': global_norm, 'global_pair_fraction_of_14d_sqnorm': float(global_norm ** 2 / norm14 ** 2) if norm14 > 1e-12 else 0.0}
        for c, val in zip(coord_cols, v):
            row[c] = float(val)
        vec_rows.append(row)
        denom14 = float(np.sum(v ** 2))
        for j, (c, val) in enumerate(zip(coord_cols, v)):
            meta = coord_meta[c]
            coord_rows.append({'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'coordinate_0based': j, 'standardized_name': c, 'block': meta['block'], 'metric': meta['metric'], 'family_low_centroid': float(lv[j]), 'family_high_centroid': float(hv[j]), 'contrast_high_minus_low': float(val), 'contrast_sign': sign_int(float(val)), 'abs_contrast': abs(float(val)), 'fraction_of_14d_squared_contrast': float(val ** 2 / denom14) if denom14 > 1e-12 else 0.0, 'semantic_interpretation_permitted': False})
        denom12 = float(np.sum(v[:12] ** 2))
        for block, i, j in block_pairs:
            ss = float(v[i] ** 2 + v[j] ** 2)
            block_rows.append({'domain': domain, 'k': k, 'contrast_id': cid, 'family_low_1based': low, 'family_high_1based': high, 'block': block, 'rms_coordinate_0based': i, 'cosine_coordinate_0based': j, 'block_squared_contrast': ss, 'non_global_block_contribution': float(ss / denom12) if denom12 > 1e-12 else 0.0, 'uniform_reference_1_over_6': 1.0 / 6.0, 'discovery_above_uniform_reference': bool(ss / denom12 > 1.0 / 6.0) if denom12 > 1e-12 else False, 'crosssplit_prominence_decidable_in_discovery_physiological_reference_atlas': False})
    return (pd.DataFrame(vec_rows), pd.DataFrame(coord_rows), pd.DataFrame(block_rows))

def heldout_physiological_transfer_policy(reference_vector_rel: str, reference_vector_sha: str, coord_table_rel: str, coord_table_sha: str, block_rel: str, block_sha: str) -> dict[str, Any]:
    return {'phase': PHASE, 'next_phase': NEXT_PHASE, 'authorization': 'heldout_physiological_transfer may open only prespecified Validation/Confirmation structural signature and assignment values after verifying physiological_characterization and discovery_physiological_reference_atlas locks', 'structural_replication_semantic_analysis_reopened': False, 'topographic_extension_analysis_reopened': False, 'new_family_fit_permitted': False, 'K_reselection_permitted': False, 'family_relabeling_permitted': False, 'behavioral_or_semantic_values_permitted': False, 'discovery_reference_vectors': {'path': reference_vector_rel, 'sha256': reference_vector_sha, 'orientation': 'family_high_minus_family_low', 'all_14_coordinates_required': True}, 'discovery_coordinate_contrasts': {'path': coord_table_rel, 'sha256': coord_table_sha}, 'discovery_non_global_block_contributions': {'path': block_rel, 'sha256': block_sha}, 'heldout_splits': ['Validation', 'Confirmation'], 'domains': {'experience': {'K': [2, 3, 4]}, 'report_transition': {'K': [2, 3]}}, 'primary_eligibility': 'participant_eligible == True in each prespecified transfer assignment table', 'all_assigned_sensitivity': 'repeat point-estimate characterization using all assigned held-out trials/participants; sensitivity cannot satisfy or overturn primary gate', 'family_centroid': 'within-participant family mean of all 14 standardized coordinates, then equal-weight mean across participants represented in that family', 'contrast_cosine': {'definition': 'cosine between fixed Discovery 14D contrast vector and heldout 14D contrast vector', 'zero_norm_convention': 0.0, 'epsilon': 1e-12}, 'sign_concordance': {'definition': 'exact sign agreement across all 14 coordinates relative to Discovery', 'sign_epsilon': 1e-12, 'denominator': 14}, 'non_global_block_contribution': 'sum of squared heldout contrast coordinates within each prespecified RMS/cosine pair divided by sum over coordinates 0..11', 'bootstrap': {'n_resamples': 2000, 'unit': 'participant', 'sampling': 'sample eligible participant IDs with replacement within each split; a duplicated participant contributes again; for each family average available sampled participant-family means', 'finite_replicate_requirement': 1900, 'ci': '2.5th and 97.5th percentiles using NumPy quantile linear method over finite replicates', 'seed_formula': '540100000 + domain_offset + split_offset + 100*k + 10*family_low + family_high', 'domain_offset': {'experience': 0, 'report_transition': 10000}, 'split_offset': {'Validation': 1000, 'Confirmation': 2000}}, 'fingerprint_direction_replication_gate': 'for BOTH Validation and Confirmation: bootstrap 95% lower CI of contrast cosine > 0 AND point-estimate sign concordance >= 10/14', 'stable_coordinate_rule': 'same point-estimate contrast sign in Discovery, Validation, Confirmation AND both heldout participant-bootstrap 95% CIs exclude zero in that direction', 'replicated_prominent_block_rule': 'non-global block contribution > 1/6 in Discovery, Validation, Confirmation AND at least one member coordinate satisfies stable-coordinate rule', 'global_pair': 'coordinates 12-13 reported separately; never counted as a seventh source block', 'no_inferential_p_values': True, 'no_crosssplit_selection': True, 'signal_space_feature_decomposition_authorization': 'only blocks satisfying replicated_prominent_block_rule in heldout_physiological_transfer may enter 169-feature decomposition', 'interpretation': 'replicated fingerprint means reproducible feature-space geometry only; no emotion semantics, generator localization, causal mechanism, or topographic rescue'}

def self_test() -> None:
    cols = [f'c{i}' for i in range(14)]
    sig = pd.DataFrame([{'raw_behavior_row_id': 'a1', 'subject_id': 's1', **{c: float(i) for i, c in enumerate(cols)}}, {'raw_behavior_row_id': 'a2', 'subject_id': 's1', **{c: float(i + 2) for i, c in enumerate(cols)}}, {'raw_behavior_row_id': 'b1', 'subject_id': 's2', **{c: float(i + 4) for i, c in enumerate(cols)}}, {'raw_behavior_row_id': 'b2', 'subject_id': 's2', **{c: float(i + 6) for i, c in enumerate(cols)}}])
    ass = pd.DataFrame([{'raw_behavior_row_id': 'a1', 'subject_id': 's1', 'k': 2, 'family_1based': 1}, {'raw_behavior_row_id': 'a2', 'subject_id': 's1', 'k': 2, 'family_1based': 2}, {'raw_behavior_row_id': 'b1', 'subject_id': 's2', 'k': 2, 'family_1based': 1}, {'raw_behavior_row_id': 'b2', 'subject_id': 's2', 'k': 2, 'family_1based': 2}])
    _, fam = participant_balanced_atlas(sig, ass, [2], cols, 'experience')
    f1 = fam.loc[fam.family_1based.eq(1), cols].iloc[0].to_numpy(float)
    f2 = fam.loc[fam.family_1based.eq(2), cols].iloc[0].to_numpy(float)
    if not np.allclose(f2 - f1, np.repeat(2.0, 14)):
        raise RuntimeError('self-test contrast orientation failed')
    if safe_cosine(np.ones(14), np.ones(14)) < 0.999999:
        raise RuntimeError('self-test cosine failed')
    print('discovery_physiological_reference_atlas self-test PASS')

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
        raise SystemExit('--project is required')
    t0 = time.time()
    project = Path(args.project).resolve()
    out = project / OUT_REL
    if out.exists() and any(out.rglob('*')) and (not args.force):
        raise RuntimeError(f'refusing nonempty output directory without --force: {out}')
    if out.exists() and args.force:
        shutil.rmtree(out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    (out / 'reports').mkdir(parents=True, exist_ok=True)
    rehash: list[dict[str, Any]] = []
    physiological_characterization = verify_lock(project, 'physiological_characterization', physiological_characterization_LOCK_REL, physiological_characterization_LOCK_SHA, rehash)
    if physiological_characterization.get('phase') != 'physiological_characterization' or physiological_characterization.get('next_phase') != PHASE:
        raise RuntimeError('physiological_characterization authorization drift')
    if physiological_characterization.get('behavioral_values_read') is not False or physiological_characterization.get('structural_values_parsed') is not False:
        raise RuntimeError('physiological_characterization purity drift')
    p10a = verify_exact(project, physiological_characterization_POLICY_REL, physiological_characterization_POLICY_SHA, 'physiological_characterization_policy', rehash, True)
    policy10a = readj(p10a)
    if policy10a.get('experience_K') != [2, 3, 4] or policy10a.get('report_transition_K') != [2, 3]:
        raise RuntimeError('physiological_characterization K specification drift')
    if policy10a.get('semantic_rescue_permitted') is not False:
        raise RuntimeError('physiological_characterization semantic boundary drift')
    verify_lock(project, 'discovery_stimulus_experience_multiresolution_fit', discovery_stimulus_experience_multiresolution_fit_LOCK_REL, discovery_stimulus_experience_multiresolution_fit_LOCK_SHA, rehash)
    verify_lock(project, 'discovery_report_transition_multiresolution_fit', discovery_report_transition_multiresolution_fit_LOCK_REL, discovery_report_transition_multiresolution_fit_LOCK_SHA, rehash)
    physiological_characterization_coord_path = verify_exact(project, physiological_characterization_COORD_REL, physiological_characterization_COORD_SHA, 'physiological_characterization_coordinate_schema', rehash, True)
    physiological_characterization_contrast_path = verify_exact(project, physiological_characterization_CONTRAST_REL, physiological_characterization_CONTRAST_SHA, 'physiological_characterization_contrast_registry', rehash, True)
    physiological_characterization_block_path = verify_exact(project, physiological_characterization_BLOCK_REL, physiological_characterization_BLOCK_SHA, 'physiological_characterization_block_schema', rehash, True)
    coord_schema = pd.read_csv(physiological_characterization_coord_path)
    registry = pd.read_csv(physiological_characterization_contrast_path)
    blocks = pd.read_csv(physiological_characterization_block_path)
    coord_cols = coord_schema.sort_values('coordinate_0based')['standardized_name'].astype(str).tolist()
    if len(coord_cols) != 14 or len(set(coord_cols)) != 14:
        raise RuntimeError('physiological_characterization coordinate schema is not exactly 14 unique coordinates')
    if len(registry) != 14:
        raise RuntimeError('physiological_characterization contrast registry is not exactly 14 contrasts')
    pfc_all = []
    fam_all = []
    discovery_source_rows = []
    for domain, spec in DISCOVERY.items():
        sigp = verify_exact(project, spec['signature_rel'], spec['signature_sha'], f'{domain}_discovery_signatures', rehash, True)
        assp = verify_exact(project, spec['assignment_rel'], spec['assignment_sha'], f'{domain}_discovery_assignments', rehash, True)
        verify_exact(project, spec['centroid_rel'], spec['centroid_sha'], f'{domain}_historical_centroids_reference_only', rehash, False)
        coordp = verify_exact(project, spec['coord_rel'], spec['coord_sha'], f'{domain}_coordinate_schema', rehash, True)
        orig_coord = pd.read_csv(coordp).sort_values('coordinate_0based')
        if orig_coord['standardized_name'].astype(str).tolist() != coord_cols:
            raise RuntimeError(f'{domain}: standardized coordinate identity drift')
        sig = pd.read_csv(sigp)
        ass = pd.read_csv(assp)
        pfc, fam = participant_balanced_atlas(sig, ass, spec['ks'], coord_cols, domain)
        pfc_all.append(pfc)
        fam_all.append(fam)
        discovery_source_rows.append({'domain': domain, 'signature_rows': len(sig), 'signature_participants': sig['subject_id'].nunique(), 'assignment_rows_all_k': len(ass), 'assignment_participants': ass['subject_id'].nunique(), 'prespecified_K': ','.join(map(str, spec['ks'])), 'heldout_values_opened': False})
    pfc_df = pd.concat(pfc_all, ignore_index=True)
    fam_df = pd.concat(fam_all, ignore_index=True)
    vec_df, coord_df, block_df = make_contrasts(fam_df, registry, coord_cols, coord_schema, blocks)
    atlas_summary = []
    for domain in ['experience', 'report_transition']:
        vv = vec_df.loc[vec_df.domain.eq(domain)]
        bb = block_df.loc[block_df.domain.eq(domain)]
        atlas_summary.append({'domain': domain, 'n_contrasts': len(vv), 'n_K': vv['k'].nunique(), 'mean_14d_contrast_norm': float(vv['contrast_norm_14d'].mean()), 'median_14d_contrast_norm': float(vv['contrast_norm_14d'].median()), 'n_discovery_block_cells_above_uniform_reference': int(bb['discovery_above_uniform_reference'].sum()), 'crosssplit_replication_decidable': False})
    atlas_summary_df = pd.DataFrame(atlas_summary)
    p_pfc = out / 'tables/discovery_physiological_reference_atlas_discovery_participant_family_centroids_14d.csv'
    p_fam = out / 'tables/discovery_physiological_reference_atlas_discovery_participant_balanced_family_centroids_14d.csv'
    p_vec = out / 'tables/discovery_physiological_reference_atlas_discovery_reference_contrast_vectors_14d.csv'
    p_coord = out / 'tables/discovery_physiological_reference_atlas_discovery_coordinate_contrasts.csv'
    p_block = out / 'tables/discovery_physiological_reference_atlas_discovery_non_global_block_contributions.csv'
    p_atlas = out / 'tables/discovery_physiological_reference_atlas_discovery_atlas_summary.csv'
    p_srcaudit = out / 'qc/discovery_physiological_reference_atlas_discovery_source_cohort_audit.csv'
    p_rehash = out / 'qc/discovery_physiological_reference_atlas_source_rehash.csv'
    p_contract = out / 'qc/discovery_physiological_reference_atlas_contract.json'
    p_summary = out / 'qc/discovery_physiological_reference_atlas_summary.json'
    p_report = out / 'reports/discovery_physiological_reference_atlas_report.md'
    pfc_df.to_csv(p_pfc, index=False, float_format='%.12g')
    fam_df.to_csv(p_fam, index=False, float_format='%.12g')
    vec_df.to_csv(p_vec, index=False, float_format='%.12g')
    coord_df.to_csv(p_coord, index=False, float_format='%.12g')
    block_df.to_csv(p_block, index=False, float_format='%.12g')
    atlas_summary_df.to_csv(p_atlas, index=False, float_format='%.12g')
    pd.DataFrame(discovery_source_rows).to_csv(p_srcaudit, index=False)
    pd.DataFrame(rehash).to_csv(p_rehash, index=False)
    rel_vec = str(p_vec.relative_to(project))
    rel_coord = str(p_coord.relative_to(project))
    rel_block = str(p_block.relative_to(project))
    policy10c = heldout_physiological_transfer_policy(rel_vec, sha256_path(p_vec), rel_coord, sha256_path(p_coord), rel_block, sha256_path(p_block))
    p_policy = project / POLICY_REL
    writej(p_policy, policy10c)
    contract = {'phase': PHASE, 'next_phase': NEXT_PHASE, 'status': 'Discovery_14D_reference_atlas_materialized', 'physiological_characterization_final_lock_sha256': physiological_characterization_LOCK_SHA, 'physiological_characterization_policy_sha256': physiological_characterization_POLICY_SHA, 'experience_K': [2, 3, 4], 'report_transition_K': [2, 3], 'n_coordinates': 14, 'n_contrasts': 14, 'family_centroid_estimator': 'participant-balanced within-participant-family mean then equal participant mean', 'contrast_orientation': 'family_high_minus_family_low', 'heldout_values_opened': False, 'behavioral_or_semantic_values_opened': False, 'new_inference_run': False, 'crosssplit_gate_evaluated': False, 'topographic_extension_analysis_reopened': False, 'structural_replication_semantic_analysis_reopened': False, 'heldout_physiological_transfer_policy_path': str(POLICY_REL)}
    writej(p_contract, contract)
    summary = {**contract, 'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'experience_discovery_signature_rows': int(discovery_source_rows[0]['signature_rows']), 'transition_discovery_signature_rows': int(discovery_source_rows[1]['signature_rows']), 'n_participant_family_centroid_rows': int(len(pfc_df)), 'n_family_centroid_rows': int(len(fam_df)), 'n_coordinate_contrast_rows': int(len(coord_df)), 'n_block_contribution_rows': int(len(block_df)), 'signal_space_feature_decomposition_blocks_selected': [], 'signal_space_feature_decomposition_selection_permitted_in_discovery_physiological_reference_atlas': False}
    writej(p_summary, summary)
    report_lines = ['# discovery_physiological_reference_atlas Discovery physiological reference atlas', '', 'discovery_physiological_reference_atlas opened Discovery structural signature/assignment values only. Validation and Confirmation values were not read.', '', '## prespecified scope', '', '- Experience: K=2,3,4', '- Report-transition: K=2,3', '- All 14 standardized signature coordinates', '- All 14 canonical family contrasts', '', '## Estimator', '', 'Family centroids are participant-balanced: within-participant family means followed by equal participant averaging.', 'Each contrast is higher numeric family ID minus lower numeric family ID.', '', '## Outputs', '', 'The atlas reports every coordinate contrast and every non-global block contribution. No coordinate/block is selected in discovery_physiological_reference_atlas.', 'The global pair is reported separately and is not a seventh independent source block.', '', '## Scientific boundary', '', 'These are Discovery reference fingerprints. Cross-split physiological replication is not decidable until heldout_physiological_transfer.', 'No semantic interpretation, topographic rescue, causal generator claim, or emotion-family naming is permitted.', '', f'Next phase: {NEXT_PHASE}', '']
    p_report.write_text(''.join(report_lines), encoding='utf-8')
    lock_targets = [p_policy, p_pfc, p_fam, p_vec, p_coord, p_block, p_atlas, p_srcaudit, p_rehash, p_contract, p_summary, p_report]
    hashes = {str(p.relative_to(project)): sha256_path(p) for p in lock_targets}
    p_lock = out / 'qc/discovery_physiological_reference_atlas_final_output_lock_hashes.json'
    final_lock = {'phase': PHASE, 'next_phase': NEXT_PHASE, 'status': 'Discovery_14D_reference_atlas_fixed', 'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'physiological_characterization_final_lock_sha256': physiological_characterization_LOCK_SHA, 'heldout_values_opened': False, 'behavioral_or_semantic_values_opened': False, 'new_inference_run': False, 'crosssplit_gate_evaluated': False, 'K_reselected': False, 'families_refit': False, 'semantic_relabeling_permitted': False, 'hashes': hashes, 'runtime_seconds': round(time.time() - t0, 3)}
    writej(p_lock, final_lock)
    parent_refs = [project / physiological_characterization_LOCK_REL, project / physiological_characterization_POLICY_REL, project / discovery_stimulus_experience_multiresolution_fit_LOCK_REL, project / discovery_report_transition_multiresolution_fit_LOCK_REL]
    files = lock_targets + [p_lock]
    print('discovery_physiological_reference_atlas PASS')
    print('Discovery experience signatures:', discovery_source_rows[0]['signature_rows'], 'trials /', discovery_source_rows[0]['signature_participants'], 'participants')
    print('Discovery report-transition signatures:', discovery_source_rows[1]['signature_rows'], 'trials /', discovery_source_rows[1]['signature_participants'], 'participants')
    print('Participant-balanced family centroids:', len(fam_df))
    print('prespecified contrast vectors:', len(vec_df), 'x 14 coordinates')
    print('Heldout values opened: False')
    print('Cross-split gate evaluated: False')
    print('Next phase:', NEXT_PHASE)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())