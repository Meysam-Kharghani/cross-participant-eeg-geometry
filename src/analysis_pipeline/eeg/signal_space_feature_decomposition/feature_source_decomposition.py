# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PHASE = 'signal_space_feature_decomposition'
NEXT_PHASE = 'physiological_synthesis'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
IQR_FLOOR = 1e-08
WINSOR_LIMIT = 6.0
ZERO_NORM_EPS = 1e-12
IDENTITY_TOL = 1e-09
OUT = Path('outputs/interpretability/signal_space_feature_decomposition_feature_source_decomposition')
NEXT_CONFIG = Path('analysis_specifications/S064_signal_space_feature_decomposition_physiological_synthesis.json')
heldout_physiological_transfer_OUT = Path('outputs/interpretability/heldout_physiological_transfer_heldout_physiological_fingerprint_transfer')
heldout_physiological_transfer_LOCK = heldout_physiological_transfer_OUT / 'qc/heldout_physiological_transfer_final_output_lock_hashes.json'
heldout_physiological_transfer_LOCK_SHA256 = '190b9226fc272d901b30ae52cde8666c035262bf5adc653ef562614fecc3b1df'
heldout_physiological_transfer_POLICY = Path('analysis_specifications/S034_heldout_physiological_transfer_signal_space_feature.json')
heldout_physiological_transfer_POLICY_SHA256 = '873ee51cde201986041780a89726b2a45245e304369b1e9088c6c7a49e2cfe25'
heldout_physiological_transfer_AUTH = heldout_physiological_transfer_OUT / 'tables/heldout_physiological_transfer_replicated_prominent_block_registry.csv'
heldout_physiological_transfer_AUTH_SHA256 = '873d08f61663d15f79829f1921f43e1c21dc04ed70bc98f50e4130d8a4d800ba'
physiological_characterization_OUT = Path('outputs/interpretability/physiological_characterization_physiological_characterization_design')
physiological_characterization_LOCK = physiological_characterization_OUT / 'qc/physiological_characterization_final_output_lock_hashes.json'
physiological_characterization_LOCK_SHA256 = '5105434cc56803cae4dca6e5008d243c956be2e0012602262311aefa5915d19b'
physiological_characterization_BLOCK_SCHEMA = physiological_characterization_OUT / 'tables/physiological_characterization_prespecified_feature_block_schema.csv'
physiological_characterization_BLOCK_SCHEMA_SHA256 = 'd73f13f3bbacfc8f51bb01e4473b879770d697e68a91abe353d001b1b07d83b4'
OPTIONAL_FEATURE_SCHEMA_SHA256 = 'ebbd3b32c394e0ebcfc6137027a2736bee113a975df11168504e6a1242999471'
COMPACT = {'Discovery': (Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy'), '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'), 'Validation': (Path('outputs/large/validation_feature_trajectory_validation_compact_features_v1/global/epoch_compact_float32.npy'), '6f8ab1755f8d950a33351d2032ccd190092656923b09b701ab04e78430fec402'), 'Confirmation': (Path('outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1/global/epoch_compact_float32.npy'), '734fa3d2581d95adc172ea854d4f1fccc2432eaeb167960fdf1e76da0c1509d3')}
SPECS = {('experience', 'Discovery'): {'pair': Path('outputs/interpretability/multiresolution_experience_report_design_multiresolution_experience_report_design_specification/tables/multiresolution_experience_report_design_discovery_experience_pair_manifest.csv'), 'pair_sha': '80ef6d2e3ed49613c2e5646d37d958ba58946606fc007a5231341f5cf21767c8', 'assign': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'), 'assign_sha': '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5', 'signature': Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'), 'signature_sha': 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326', 'stage_a_col': 'stimulus_early_row_0based', 'stage_b_col': 'stimulus_late_row_0based', 'stage_a_name': 'stimulus_early', 'stage_b_name': 'stimulus_late', 'discovery_eligibility': 'experience_eligible'}, ('report_transition', 'Discovery'): {'pair': Path('outputs/interpretability/semantic_report_dissociation_design_semantic_report_dissociation_design_specification/tables/semantic_report_dissociation_design_prespecified_discovery_report_transition_pair_manifest.csv'), 'pair_sha': '9deb3788034a3e1d458e2c66b125fb86d0766ada0a6c30ffe9066a5490341983', 'assign': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'), 'assign_sha': '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a', 'signature': Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv'), 'signature_sha': '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83', 'stage_a_col': 'stimulus_late_row_0based', 'stage_b_col': 'rating_early_row_0based', 'stage_a_name': 'stimulus_late', 'stage_b_name': 'rating_early', 'discovery_eligibility': 'report_transition_eligible'}, ('experience', 'Validation'): {'pair': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_pair_manifest.csv'), 'pair_sha': '9580d1b07e65d059919a6c70d11b3c03327103887c236f1386e0826a5a4ceb43', 'assign': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv'), 'assign_sha': 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1', 'signature': Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv'), 'signature_sha': '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298', 'stage_a_col': 'stimulus_early_row_0based', 'stage_b_col': 'stimulus_late_row_0based', 'stage_a_name': 'stimulus_early', 'stage_b_name': 'stimulus_late', 'heldout_eligibility': 'participant_eligible'}, ('experience', 'Confirmation'): {'pair': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_pair_manifest.csv'), 'pair_sha': 'cad7e3e2a1f21191f73ce728b5251d8cbdcf45fbdaaada911121fff1a86ea3ae', 'assign': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'), 'assign_sha': '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed', 'signature': Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'), 'signature_sha': '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d', 'stage_a_col': 'stimulus_early_row_0based', 'stage_b_col': 'stimulus_late_row_0based', 'stage_a_name': 'stimulus_early', 'stage_b_name': 'stimulus_late', 'heldout_eligibility': 'participant_eligible'}, ('report_transition', 'Validation'): {'pair': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_pair_manifest.csv'), 'pair_sha': 'a3f377c398729d8d8eab1a32c4b1fd508efee6659403a6750e9c221172ef1e2e', 'assign': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv'), 'assign_sha': '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53', 'signature': Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv'), 'signature_sha': 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067', 'stage_a_col': 'stimulus_late_row_0based', 'stage_b_col': 'rating_early_row_0based', 'stage_a_name': 'stimulus_late', 'stage_b_name': 'rating_early', 'heldout_eligibility': 'participant_eligible'}, ('report_transition', 'Confirmation'): {'pair': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_pair_manifest.csv'), 'pair_sha': '8e02e5339860beb8c47132cc46a71e4f6eef38b7ab7c3c5b9905d3ab53dc9130', 'assign': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'), 'assign_sha': 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b', 'signature': Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv'), 'signature_sha': '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6', 'stage_a_col': 'stimulus_late_row_0based', 'stage_b_col': 'rating_early_row_0based', 'stage_a_name': 'stimulus_late', 'stage_b_name': 'rating_early', 'heldout_eligibility': 'participant_eligible'}}
BLOCKS = [('time_frequency', 0, 60), ('morphology', 60, 72), ('plv_channel', 72, 84), ('waveform_dct', 84, 124), ('covariance', 124, 145), ('plv_global', 145, 169)]
BLOCK_LOOKUP = {name: (start, stop) for name, start, stop in BLOCKS}
RAW_COORDS = ['time_frequency_channel_median_iqr__rms_displacement', 'time_frequency_channel_median_iqr__cosine_similarity', 'morphology_channel_median_iqr__rms_displacement', 'morphology_channel_median_iqr__cosine_similarity', 'plv_channel_median_iqr__rms_displacement', 'plv_channel_median_iqr__cosine_similarity', 'waveform_dct_channel_median_iqr__rms_displacement', 'waveform_dct_channel_median_iqr__cosine_similarity', 'covariance_summary__rms_displacement', 'covariance_summary__cosine_similarity', 'plv_global__rms_displacement', 'plv_global__cosine_similarity', 'global__rms_displacement', 'global__cosine_similarity']

def now() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime('%Y%m%d_%H%M%S')

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'1', 'true', 't', 'yes', 'y'}

def verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}; observed {observed}; path={path}')
    return observed

def stage_standardize(block: np.ndarray, pooled_iqr: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=float).copy()
    med = np.nanmedian(block, axis=0)
    if not np.isfinite(med).all():
        bad = np.flatnonzero(~np.isfinite(med))
        raise RuntimeError('Participant/stage median undefined for compact cell(s): ' + ','.join(map(str, bad[:20])))
    missing = ~np.isfinite(block)
    if np.any(missing):
        block[missing] = np.broadcast_to(med, block.shape)[missing]
    q1 = np.quantile(block, 0.25, axis=0)
    q3 = np.quantile(block, 0.75, axis=0)
    iq = q3 - q1
    fallback = ~np.isfinite(iq) | (iq <= IQR_FLOOR)
    scale = np.where(fallback, pooled_iqr, iq)
    constant = ~np.isfinite(scale) | (scale <= IQR_FLOOR)
    scale = np.where(constant, 1.0, scale)
    z = np.clip((block - med) / scale, -WINSOR_LIMIT, WINSOR_LIMIT)
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite participant/stage standardized EEG values')
    return z

def preprocess_stage_pair(base: pd.DataFrame, compact: np.ndarray, a_col: str, b_col: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(base)
    ai = base[a_col].astype(int).to_numpy()
    bi = base[b_col].astype(int).to_numpy()
    if n == 0:
        raise RuntimeError('Empty pair manifest after prespecified eligibility')
    if min(ai.min(), bi.min()) < 0 or max(ai.max(), bi.max()) >= len(compact):
        raise RuntimeError('Pair manifest compact row is outside the prespecified store')
    raw_a = np.asarray(compact[ai], dtype=np.float64)
    raw_b = np.asarray(compact[bi], dtype=np.float64)
    if raw_a.shape != (n, 169) or raw_b.shape != (n, 169):
        raise RuntimeError('Unexpected compact stage shape')
    subjects = base.subject_id.astype(str).to_numpy()
    az = np.empty_like(raw_a, dtype=float)
    bz = np.empty_like(raw_b, dtype=float)
    audits: list[dict[str, Any]] = []
    for subject in pd.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        a = raw_a[idx]
        b = raw_b[idx]
        pooled = np.vstack([a, b])
        pooled_iqr = np.nanquantile(pooled, 0.75, axis=0) - np.nanquantile(pooled, 0.25, axis=0)
        pooled_iqr = np.where(np.isfinite(pooled_iqr), pooled_iqr, 0.0)
        az[idx] = stage_standardize(a, pooled_iqr)
        bz[idx] = stage_standardize(b, pooled_iqr)
        audits.append({'subject_id': str(subject), 'paired_trials': int(len(idx)), 'stage_a_finite': bool(np.isfinite(az[idx]).all()), 'stage_b_finite': bool(np.isfinite(bz[idx]).all())})
    return (az, bz, audits)

def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    na = np.sqrt(np.sum(a * a, axis=1))
    nb = np.sqrt(np.sum(b * b, axis=1))
    za = na <= ZERO_NORM_EPS
    zb = nb <= ZERO_NORM_EPS
    both = za & zb
    one = za ^ zb
    out = np.empty(len(a), dtype=float)
    regular = ~(za | zb)
    out[regular] = num[regular] / (na[regular] * nb[regular])
    out[both] = 1.0
    out[one] = 0.0
    return np.clip(out, -1.0, 1.0)

def raw_signatures_from_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = []
    for _, start, stop in BLOCKS:
        aa = a[:, start:stop]
        bb = b[:, start:stop]
        cols.append(np.sqrt(np.mean((bb - aa) ** 2, axis=1)))
        cols.append(row_cosine(aa, bb))
    cols.append(np.sqrt(np.mean((b - a) ** 2, axis=1)))
    cols.append(row_cosine(a, b))
    out = np.column_stack(cols)
    if out.shape != (len(a), 14) or not np.isfinite(out).all():
        raise RuntimeError('Invalid reconstructed raw signature')
    return out

def block_feature_metrics(a: np.ndarray, b: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aa = np.asarray(a[:, start:stop], dtype=float)
    bb = np.asarray(b[:, start:stop], dtype=float)
    delta = bb - aa
    sq = delta * delta
    denom = sq.sum(axis=1)
    nfeat = stop - start
    rms_share = np.empty_like(sq)
    regular = denom > ZERO_NORM_EPS
    rms_share[regular] = sq[regular] / denom[regular, None]
    rms_share[~regular] = 1.0 / float(nfeat)
    na = np.sqrt(np.sum(aa * aa, axis=1))
    nb = np.sqrt(np.sum(bb * bb, axis=1))
    za = na <= ZERO_NORM_EPS
    zb = nb <= ZERO_NORM_EPS
    both = za & zb
    one = za ^ zb
    reg = ~(za | zb)
    cos_contrib = np.zeros_like(aa)
    if np.any(reg):
        cos_contrib[reg] = aa[reg] * bb[reg] / (na[reg, None] * nb[reg, None])
    if np.any(both):
        cos_contrib[both] = 1.0 / float(nfeat)
    cos_contrib[one] = 0.0
    if not np.allclose(rms_share.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError('RMS contribution decomposition does not sum to 1')
    if not np.allclose(cos_contrib.sum(axis=1), row_cosine(aa, bb), atol=1e-12, rtol=1e-12):
        raise RuntimeError('Cosine feature contributions do not reconstruct block cosine')
    return (rms_share, delta, cos_contrib)

def participant_balanced_family_vector(values: np.ndarray, subjects: np.ndarray, families: np.ndarray, family: int) -> tuple[np.ndarray, int, int]:
    family = int(family)
    mask = families.astype(int) == family
    if not np.any(mask):
        raise RuntimeError(f'No rows for family {family}')
    s = subjects[mask].astype(str)
    x = values[mask]
    pmeans = []
    for subject in pd.unique(s):
        pmeans.append(np.mean(x[s == subject], axis=0))
    arr = np.vstack(pmeans)
    return (np.mean(arr, axis=0), int(len(pmeans)), int(mask.sum()))

def discover_feature_labels(project: Path) -> tuple[list[str], str, str | None]:
    labels = [f'feature_{i:03d}' for i in range(169)]
    roots = [project / 'outputs/tables', project / 'outputs/qc', project / 'outputs/large']
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            try:
                candidates.extend(root.rglob('*feature*schema*.csv'))
            except OSError:
                pass
    for p in sorted(set(candidates)):
        try:
            if sha256_path(p) != OPTIONAL_FEATURE_SCHEMA_SHA256:
                continue
            df = pd.read_csv(p)
            if len(df) != 169:
                return (labels, 'exact_schema_hash_found_but_rowcount_not_169', str(p.relative_to(project)))
            preferred = ['feature_name', 'feature', 'name', 'label', 'column_name', 'feature_label', 'description']
            col = next((c for c in preferred if c in df.columns), None)
            if col is None:
                text_cols = [c for c in df.columns if df[c].dtype == object and c.lower() not in {'block', 'group'}]
                col = text_cols[0] if text_cols else None
            if col is not None:
                vals = df[col].astype(str).tolist()
                if len(vals) == 169 and all((v.strip() for v in vals)):
                    return (vals, 'historical_feature_schema_exact_hash', str(p.relative_to(project)))
            return (labels, 'historical_schema_exact_hash_generic_labels', str(p.relative_to(project)))
        except Exception:
            continue
    return (labels, 'generic_index_labels_no_exact_schema_file_found', None)

def verify_heldout_physiological_transfer_authorization(project: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    verify(project / heldout_physiological_transfer_LOCK, heldout_physiological_transfer_LOCK_SHA256, 'heldout_physiological_transfer corrected final lock')
    rows.append({'role': 'heldout_physiological_transfer_corrected_final_lock', 'path': str(heldout_physiological_transfer_LOCK), 'expected_sha256': heldout_physiological_transfer_LOCK_SHA256, 'observed_sha256': sha256_path(project / heldout_physiological_transfer_LOCK), 'verified': True, 'values_parsed': False})
    lock = readj(project / heldout_physiological_transfer_LOCK)
    if lock.get('phase') != 'heldout_physiological_transfer' or lock.get('next_phase') != PHASE:
        raise RuntimeError('heldout_physiological_transfer phase/authorization drift')
    if lock.get('behavioral_or_semantic_values_opened') is not False:
        raise RuntimeError('heldout_physiological_transfer semantic boundary drift')
    if lock.get('fingerprint_gate_is_assignment_conditioned') is not True:
        raise RuntimeError('heldout_physiological_transfer assignment-conditioned boundary drift')
    for rel, expected in lock.get('hashes', {}).items():
        p = project / rel
        verify(p, expected, 'heldout_physiological_transfer child')
    verify(project / heldout_physiological_transfer_POLICY, heldout_physiological_transfer_POLICY_SHA256, 'heldout_physiological_transfer -> signal_space_feature_decomposition policy')
    rows.append({'role': 'heldout_physiological_transfer_signal_space_feature_decomposition_policy', 'path': str(heldout_physiological_transfer_POLICY), 'expected_sha256': heldout_physiological_transfer_POLICY_SHA256, 'observed_sha256': sha256_path(project / heldout_physiological_transfer_POLICY), 'verified': True, 'values_parsed': True})
    policy = readj(project / heldout_physiological_transfer_POLICY)
    if policy.get('phase') != 'heldout_physiological_transfer' or policy.get('next_phase') != PHASE:
        raise RuntimeError('signal_space_feature_decomposition policy phase drift')
    if policy.get('new_block_or_feature_selection_permitted') is not False:
        raise RuntimeError('signal_space_feature_decomposition feature-selection boundary drift')
    if policy.get('all_features_within_authorized_block_required') is not True:
        raise RuntimeError('signal_space_feature_decomposition all-feature requirement drift')
    verify(project / heldout_physiological_transfer_AUTH, heldout_physiological_transfer_AUTH_SHA256, 'heldout_physiological_transfer authorization registry')
    rows.append({'role': 'heldout_physiological_transfer_authorization_registry', 'path': str(heldout_physiological_transfer_AUTH), 'expected_sha256': heldout_physiological_transfer_AUTH_SHA256, 'observed_sha256': sha256_path(project / heldout_physiological_transfer_AUTH), 'verified': True, 'values_parsed': True})
    auth = pd.read_csv(project / heldout_physiological_transfer_AUTH)
    required = {'domain', 'k', 'contrast_id', 'family_low_1based', 'family_high_1based', 'block', 'feature_start_0based_inclusive', 'feature_stop_0based_exclusive', 'n_features', 'signal_space_feature_decomposition_authorized'}
    if required - set(auth.columns):
        raise RuntimeError('heldout_physiological_transfer authorization registry schema drift')
    allowed = auth[auth.signal_space_feature_decomposition_authorized.map(bval)].copy()
    if len(allowed) != 31:
        raise RuntimeError(f'Expected exactly 31 signal_space_feature_decomposition-authorized pairs, found {len(allowed)}')
    policy_allowed = policy.get('allowed_contrast_block_pairs')
    if not isinstance(policy_allowed, list) or len(policy_allowed) != 31:
        raise RuntimeError('signal_space_feature_decomposition policy allowed-pair serialization drift')
    reg_tuples = {(str(r.domain), int(r.k), str(r.contrast_id), str(r.block)) for r in allowed.itertuples()}
    pol_tuples = {(str(r['domain']), int(r['k']), str(r['contrast_id']), str(r['block'])) for r in policy_allowed}
    if reg_tuples != pol_tuples:
        raise RuntimeError('signal_space_feature_decomposition policy/authorization registry pair mismatch')
    verify(project / physiological_characterization_LOCK, physiological_characterization_LOCK_SHA256, 'physiological_characterization final lock')
    verify(project / physiological_characterization_BLOCK_SCHEMA, physiological_characterization_BLOCK_SCHEMA_SHA256, 'physiological_characterization feature block schema')
    rows.append({'role': 'physiological_characterization_feature_block_schema', 'path': str(physiological_characterization_BLOCK_SCHEMA), 'expected_sha256': physiological_characterization_BLOCK_SCHEMA_SHA256, 'observed_sha256': sha256_path(project / physiological_characterization_BLOCK_SCHEMA), 'verified': True, 'values_parsed': True})
    block_schema = pd.read_csv(project / physiological_characterization_BLOCK_SCHEMA)
    for r in allowed.itertuples():
        if str(r.block) not in BLOCK_LOOKUP:
            raise RuntimeError(f'Unknown authorized block: {r.block}')
        start, stop = BLOCK_LOOKUP[str(r.block)]
        if int(r.feature_start_0based_inclusive) != start or int(r.feature_stop_0based_exclusive) != stop:
            raise RuntimeError(f'Authorized block boundary drift for {r.block}')
        if int(r.n_features) != stop - start:
            raise RuntimeError(f'Authorized block n_features drift for {r.block}')
    return (auth, allowed, rows)

def load_and_verify_sources(project: Path, source_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    compact_arrays: dict[str, np.ndarray] = {}
    for split, (rel, expected) in COMPACT.items():
        p = project / rel
        verify(p, expected, f'{split} prespecified compact 169-feature store')
        source_rows.append({'role': f'{split}_compact_169', 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': sha256_path(p), 'verified': True, 'values_parsed': True})
        arr = np.load(p, mmap_mode='r', allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 169:
            raise RuntimeError(f'Unexpected {split} compact shape: {arr.shape}')
        compact_arrays[split] = arr
    seen: set[tuple[str, str]] = set()
    for (domain, split), spec in SPECS.items():
        for role_key, path_key, sha_key in [('pair_manifest', 'pair', 'pair_sha'), ('family_assignments', 'assign', 'assign_sha'), ('raw_signature_identity', 'signature', 'signature_sha')]:
            rel = spec[path_key]
            expected = spec[sha_key]
            key = (str(rel), expected)
            if key in seen:
                continue
            seen.add(key)
            p = project / rel
            verify(p, expected, f'{domain}/{split} {role_key}')
            source_rows.append({'role': f'{domain}_{split}_{role_key}', 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': sha256_path(p), 'verified': True, 'values_parsed': True})
    return compact_arrays

def prepare_domain_split(project: Path, domain: str, split: str, compact: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    spec = SPECS[domain, split]
    pair = pd.read_csv(project / spec['pair'], low_memory=False)
    if pair.raw_behavior_row_id.astype(str).duplicated().any():
        raise RuntimeError(f'Duplicate pair IDs for {domain}/{split}')
    pair['raw_behavior_row_id'] = pair.raw_behavior_row_id.astype(str)
    pair['subject_id'] = pair.subject_id.astype(str)
    if split == 'Discovery':
        elig = spec['discovery_eligibility']
        if elig not in pair.columns:
            raise RuntimeError(f'Missing Discovery eligibility column {elig}')
        base = pair[pair[elig].map(bval)].copy()
    else:
        base = pair.copy()
    a, b, stage_audit = preprocess_stage_pair(base, compact, spec['stage_a_col'], spec['stage_b_col'])
    reconstructed = raw_signatures_from_z(a, b)
    sig = pd.read_csv(project / spec['signature'], low_memory=False)
    sig['raw_behavior_row_id'] = sig.raw_behavior_row_id.astype(str)
    raw_cols = [f'raw__{c}' for c in RAW_COORDS]
    missing = set(['raw_behavior_row_id'] + raw_cols) - set(sig.columns)
    if missing:
        raise RuntimeError(f'Signature identity columns missing for {domain}/{split}: {sorted(missing)}')
    sig2 = base[['raw_behavior_row_id']].merge(sig[['raw_behavior_row_id'] + raw_cols], on='raw_behavior_row_id', how='left', validate='one_to_one')
    if sig2[raw_cols].isna().any().any():
        raise RuntimeError(f'Missing signature identity rows for {domain}/{split}')
    expected = sig2[raw_cols].to_numpy(float)
    abs_err = np.abs(expected - reconstructed)
    max_err = float(np.nanmax(abs_err))
    if max_err > IDENTITY_TOL:
        raise RuntimeError(f'169-feature preprocessing identity failed for {domain}/{split}: max abs raw-signature error={max_err}')
    identity = {'domain': domain, 'split': split, 'n_rows_identity_checked': int(len(base)), 'n_participants_identity_checked': int(base.subject_id.nunique()), 'max_abs_raw_signature_error': max_err, 'identity_tolerance': IDENTITY_TOL, 'passed': True}
    if split == 'Discovery':
        primary_mask = np.ones(len(base), dtype=bool)
    else:
        ecol = spec['heldout_eligibility']
        if ecol not in base.columns:
            raise RuntimeError(f'Missing held-out eligibility column {ecol}')
        primary_mask = base[ecol].map(bval).to_numpy(bool)
    base_primary = base.loc[primary_mask].copy().reset_index(drop=True)
    a_primary = a[primary_mask]
    b_primary = b[primary_mask]
    assign = pd.read_csv(project / spec['assign'], low_memory=False)
    assign['raw_behavior_row_id'] = assign.raw_behavior_row_id.astype(str)
    assign['subject_id'] = assign.subject_id.astype(str)
    if split != 'Discovery' and 'participant_eligible' in assign.columns:
        assign = assign[assign.participant_eligible.map(bval)].copy()
    return (base_primary, a_primary, b_primary, assign, identity, stage_audit)

def build_family_feature_centroids(base: pd.DataFrame, a: np.ndarray, b: np.ndarray, assignments: pd.DataFrame, authorized_domain: pd.DataFrame) -> tuple[dict[tuple[int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    ids = base.raw_behavior_row_id.astype(str).to_numpy()
    subjects = base.subject_id.astype(str).to_numpy()
    id_to_idx = {rid: i for i, rid in enumerate(ids)}
    if len(id_to_idx) != len(ids):
        raise RuntimeError('Primary pair IDs are not unique')
    needed_blocks = sorted(set(authorized_domain.block.astype(str)))
    block_metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in needed_blocks:
        start, stop = BLOCK_LOOKUP[block]
        block_metrics[block] = block_feature_metrics(a, b, start, stop)
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for k in sorted(set(authorized_domain.k.astype(int))):
        aa = assignments[assignments.k.astype(int).eq(k)].copy()
        aa = aa[aa.raw_behavior_row_id.isin(id_to_idx)].copy()
        if aa.raw_behavior_row_id.duplicated().any():
            raise RuntimeError(f'Duplicate assignment rows at K={k}')
        idx = np.array([id_to_idx[x] for x in aa.raw_behavior_row_id.astype(str)], dtype=int)
        fam = aa.family_1based.astype(int).to_numpy()
        sub = aa.subject_id.astype(str).to_numpy()
        if not np.array_equal(sub, subjects[idx]):
            raise RuntimeError('Assignment subject/pair subject mismatch')
        blocks_for_k = sorted(set(authorized_domain.loc[authorized_domain.k.astype(int).eq(k), 'block'].astype(str)))
        families_needed = sorted(set(authorized_domain.loc[authorized_domain.k.astype(int).eq(k), 'family_low_1based'].astype(int)) | set(authorized_domain.loc[authorized_domain.k.astype(int).eq(k), 'family_high_1based'].astype(int)))
        for block in blocks_for_k:
            rms, delta, cos = block_metrics[block]
            rms_k, delta_k, cos_k = (rms[idx], delta[idx], cos[idx])
            for family in families_needed:
                rv, npart, ntrial = participant_balanced_family_vector(rms_k, sub, fam, family)
                dv, npart2, ntrial2 = participant_balanced_family_vector(delta_k, sub, fam, family)
                cv, npart3, ntrial3 = participant_balanced_family_vector(cos_k, sub, fam, family)
                if (npart, ntrial) != (npart2, ntrial2) or (npart, ntrial) != (npart3, ntrial3):
                    raise RuntimeError('Family metric cohort mismatch')
                out[k, family, block] = {'rms_share': rv, 'signed_delta': dv, 'cosine_contribution': cv, 'n_participants': npart, 'n_trials': ntrial}
                audit_rows.append({'k': k, 'family_1based': family, 'block': block, 'n_participants': npart, 'n_trials': ntrial, 'rms_share_sum': float(np.sum(rv)), 'cosine_contribution_sum': float(np.sum(cv))})
                if abs(float(np.sum(rv)) - 1.0) > 1e-10:
                    raise RuntimeError('Participant-balanced RMS share does not sum to 1')
    return (out, audit_rows)

def sign_int(x: float, tol: float=1e-15) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0

def build_crosssplit_registry(long_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ['domain', 'k', 'contrast_id', 'family_low_1based', 'family_high_1based', 'block', 'feature_global_0based', 'feature_block_local_0based', 'feature_label', 'n_features_in_block', 'uniform_share_reference']
    rows = []
    for key, g in long_df.groupby(key_cols, sort=False, dropna=False):
        by = {str(r.split): r for r in g.itertuples()}
        if set(by) != {'Discovery', 'Validation', 'Confirmation'}:
            raise RuntimeError(f'Missing split in cross-split feature registry: {key}')
        d, v, c = (by['Discovery'], by['Validation'], by['Confirmation'])
        signs = [sign_int(float(d.contrast_rms_share)), sign_int(float(v.contrast_rms_share)), sign_int(float(c.contrast_rms_share))]
        same_nonzero = signs[0] != 0 and signs[0] == signs[1] == signs[2]
        dominant = None
        above = False
        if same_nonzero:
            dominant = int(d.family_high_1based) if signs[0] > 0 else int(d.family_low_1based)
            shares = []
            for r in [d, v, c]:
                shares.append(float(r.high_family_rms_share) if signs[0] > 0 else float(r.low_family_rms_share))
            ref = float(d.uniform_share_reference)
            above = all((x > ref for x in shares))
        signed_delta_signs = [sign_int(float(d.contrast_signed_delta)), sign_int(float(v.contrast_signed_delta)), sign_int(float(c.contrast_signed_delta))]
        cosine_signs = [sign_int(float(d.contrast_cosine_contribution)), sign_int(float(v.contrast_cosine_contribution)), sign_int(float(c.contrast_cosine_contribution))]
        row = dict(zip(key_cols, key))
        row.update({'discovery_low_family_rms_share': float(d.low_family_rms_share), 'discovery_high_family_rms_share': float(d.high_family_rms_share), 'validation_low_family_rms_share': float(v.low_family_rms_share), 'validation_high_family_rms_share': float(v.high_family_rms_share), 'confirmation_low_family_rms_share': float(c.low_family_rms_share), 'confirmation_high_family_rms_share': float(c.high_family_rms_share), 'discovery_contrast_rms_share': float(d.contrast_rms_share), 'validation_contrast_rms_share': float(v.contrast_rms_share), 'confirmation_contrast_rms_share': float(c.contrast_rms_share), 'rms_share_contrast_same_nonzero_direction_all_three': bool(same_nonzero), 'dominant_family_1based': dominant, 'dominant_family_share_above_uniform_all_three': bool(above), 'stable_subfeature': bool(same_nonzero and above), 'discovery_contrast_signed_delta': float(d.contrast_signed_delta), 'validation_contrast_signed_delta': float(v.contrast_signed_delta), 'confirmation_contrast_signed_delta': float(c.contrast_signed_delta), 'signed_delta_contrast_same_nonzero_direction_all_three': bool(signed_delta_signs[0] != 0 and signed_delta_signs[0] == signed_delta_signs[1] == signed_delta_signs[2]), 'discovery_contrast_cosine_contribution': float(d.contrast_cosine_contribution), 'validation_contrast_cosine_contribution': float(v.contrast_cosine_contribution), 'confirmation_contrast_cosine_contribution': float(c.contrast_cosine_contribution), 'cosine_contribution_contrast_same_nonzero_direction_all_three': bool(cosine_signs[0] != 0 and cosine_signs[0] == cosine_signs[1] == cosine_signs[2])})
        rows.append(row)
    return pd.DataFrame(rows)

def make_summary_tables(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows = []
    for key, g in registry.groupby(['domain', 'k', 'contrast_id', 'block', 'n_features_in_block'], sort=False):
        domain, k, cid, block, nfeat = key
        stable = g.stable_subfeature.map(bval)
        block_rows.append({'domain': domain, 'k': int(k), 'contrast_id': cid, 'block': block, 'n_features': int(nfeat), 'n_stable_subfeatures': int(stable.sum()), 'fraction_stable_subfeatures': float(stable.mean()), 'any_stable_subfeature': bool(stable.any()), 'all_features_emitted': int(len(g)) == int(nfeat)})
    block_summary = pd.DataFrame(block_rows)
    phys_rows = []
    for key, g in registry.groupby(['domain', 'block'], sort=False):
        domain, block = key
        phys_rows.append({'domain': domain, 'block': block, 'n_authorized_contrasts': int(g.contrast_id.nunique()), 'n_authorized_contrast_feature_rows': int(len(g)), 'n_stable_subfeature_rows': int(g.stable_subfeature.map(bval).sum()), 'n_unique_feature_indices_stable_at_least_once': int(g.loc[g.stable_subfeature.map(bval), 'feature_global_0based'].nunique())})
    return (block_summary, pd.DataFrame(phys_rows))

def run_self_test() -> None:
    rng = np.random.default_rng(54010)
    base = pd.DataFrame({'raw_behavior_row_id': [f'r{i}' for i in range(12)], 'subject_id': np.repeat(['s1', 's2', 's3'], 4), 'a': np.arange(12), 'b': np.arange(12, 24)})
    compact = rng.normal(size=(24, 169)).astype(np.float64)
    a, b, _ = preprocess_stage_pair(base, compact, 'a', 'b')
    raw = raw_signatures_from_z(a, b)
    assert raw.shape == (12, 14)
    for block, start, stop in BLOCKS:
        rms, delta, cos = block_feature_metrics(a, b, start, stop)
        assert rms.shape == (12, stop - start)
        assert np.allclose(rms.sum(axis=1), 1.0)
        assert np.allclose(cos.sum(axis=1), row_cosine(a[:, start:stop], b[:, start:stop]))
    z = np.zeros((2, 12))
    x = np.zeros((2, 12))
    x[1, 0] = 1.0
    rms, delta, cos = block_feature_metrics(z, x, 0, 12)
    assert np.allclose(rms[0], np.repeat(1 / 12, 12))
    assert np.allclose(cos[0], np.repeat(1 / 12, 12))
    assert np.allclose(cos[1], 0.0)
    print('signal_space_feature_decomposition self-test PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    t0 = time.time()
    project = args.project.resolve()
    out = project / OUT
    if out.exists() and any(out.rglob('*')) and (not args.force):
        raise RuntimeError(f'Nonempty signal_space_feature_decomposition output already exists: {out}. Quarantine it or use --force only for an intentional rerun.')
    if args.force and out.exists():
        shutil.rmtree(out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    (out / 'reports').mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] verifying corrected heldout_physiological_transfer authorization and prespecified physiological_characterization block schema', flush=True)
    auth, allowed, source_rows = verify_heldout_physiological_transfer_authorization(project)
    print(f'[{now()}] signal_space_feature_decomposition authorization: {len(allowed)} contrast-block pairs', flush=True)
    print(f'[{now()}] verifying prespecified compact stores, pair manifests, assignments, and signature identity sources', flush=True)
    compact_arrays = load_and_verify_sources(project, source_rows)
    labels, label_status, label_source = discover_feature_labels(project)
    label_rows = []
    for block, start, stop in BLOCKS:
        for i in range(start, stop):
            label_rows.append({'feature_global_0based': i, 'block': block, 'feature_block_local_0based': i - start, 'feature_label': labels[i], 'label_status': label_status, 'label_source_path': label_source, 'label_used_for_selection_or_computation': False})
    all_long: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    stage_audit_rows: list[dict[str, Any]] = []
    centroid_audit_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for domain in ['experience', 'report_transition']:
        auth_domain = allowed[allowed.domain.astype(str).eq(domain)].copy()
        if auth_domain.empty:
            continue
        for split in ['Discovery', 'Validation', 'Confirmation']:
            print(f'[{now()}] {domain}/{split}: reconstructing prespecified participant-stage 169-feature representation', flush=True)
            base, az, bz, assignments, identity, stage_audit = prepare_domain_split(project, domain, split, compact_arrays[split])
            identity_rows.append(identity)
            for r in stage_audit:
                r2 = dict(r)
                r2.update({'domain': domain, 'split': split})
                stage_audit_rows.append(r2)
            cohort_rows.append({'domain': domain, 'split': split, 'primary_trials': int(len(base)), 'primary_participants': int(base.subject_id.nunique()), 'behavioral_or_semantic_values_opened': False, 'compact_169_values_opened': True})
            centroids, caudit = build_family_feature_centroids(base, az, bz, assignments, auth_domain)
            for r in caudit:
                r2 = dict(r)
                r2.update({'domain': domain, 'split': split})
                centroid_audit_rows.append(r2)
            for ar in auth_domain.itertuples():
                k = int(ar.k)
                low = int(ar.family_low_1based)
                high = int(ar.family_high_1based)
                block = str(ar.block)
                start, stop = BLOCK_LOOKUP[block]
                nfeat = stop - start
                lo = centroids[k, low, block]
                hi = centroids[k, high, block]
                if len(lo['rms_share']) != nfeat or len(hi['rms_share']) != nfeat:
                    raise RuntimeError('Authorized block feature-length drift')
                for local in range(nfeat):
                    gi = start + local
                    all_long.append({'domain': domain, 'split': split, 'k': k, 'contrast_id': str(ar.contrast_id), 'family_low_1based': low, 'family_high_1based': high, 'block': block, 'feature_global_0based': gi, 'feature_block_local_0based': local, 'feature_label': labels[gi], 'n_features_in_block': nfeat, 'uniform_share_reference': 1.0 / float(nfeat), 'low_family_n_participants': int(lo['n_participants']), 'high_family_n_participants': int(hi['n_participants']), 'low_family_n_trials': int(lo['n_trials']), 'high_family_n_trials': int(hi['n_trials']), 'low_family_rms_share': float(lo['rms_share'][local]), 'high_family_rms_share': float(hi['rms_share'][local]), 'contrast_rms_share': float(hi['rms_share'][local] - lo['rms_share'][local]), 'low_family_signed_delta': float(lo['signed_delta'][local]), 'high_family_signed_delta': float(hi['signed_delta'][local]), 'contrast_signed_delta': float(hi['signed_delta'][local] - lo['signed_delta'][local]), 'low_family_cosine_contribution': float(lo['cosine_contribution'][local]), 'high_family_cosine_contribution': float(hi['cosine_contribution'][local]), 'contrast_cosine_contribution': float(hi['cosine_contribution'][local] - lo['cosine_contribution'][local])})
    long_df = pd.DataFrame(all_long)
    expected_rows = int(allowed.n_features.astype(int).sum()) * 3
    if len(long_df) != expected_rows:
        raise RuntimeError(f'All-feature emission failure: expected {expected_rows}, got {len(long_df)}')
    registry = build_crosssplit_registry(long_df)
    if len(registry) != int(allowed.n_features.astype(int).sum()):
        raise RuntimeError('Cross-split registry does not contain every authorized feature')
    block_summary, physiological_summary = make_summary_tables(registry)
    if not block_summary.all_features_emitted.map(bval).all():
        raise RuntimeError('At least one authorized block omitted feature rows')
    stable_count = int(registry.stable_subfeature.map(bval).sum())
    authorized_blocks_with_stable = int(block_summary.any_stable_subfeature.map(bval).sum())
    feature_metric_schema = pd.DataFrame([{'metric': 'rms_share', 'definition': 'per-trial squared participant/stage-standardized stage difference divided by the sum of squared differences within the prespecified block; if the block difference norm is zero, uniform 1/block_size is used', 'family_estimator': 'within-participant family mean then equal-weight mean across participants', 'role': 'prespecified primary subfeature contribution metric'}, {'metric': 'signed_delta', 'definition': 'stage_b standardized feature minus stage_a standardized feature', 'family_estimator': 'within-participant family mean then equal-weight mean across participants', 'role': 'directional stage-change descriptor retained separately; not used for stable-subfeature gate'}, {'metric': 'cosine_contribution', 'definition': 'per-feature contribution to normalized block dot product; both-zero block allocates 1/block_size uniformly, one-zero block allocates zero', 'family_estimator': 'within-participant family mean then equal-weight mean across participants', 'role': 'directional/profile-preservation decomposition retained separately; not used for stable-subfeature gate'}])
    tables = out / 'tables'
    qc = out / 'qc'
    reports = out / 'reports'
    p_labels = tables / 'signal_space_feature_decomposition_feature_label_map.csv'
    p_metrics = tables / 'signal_space_feature_decomposition_feature_metric_schema.csv'
    p_long = tables / 'signal_space_feature_decomposition_authorized_feature_decomposition_long.csv'
    p_registry = tables / 'signal_space_feature_decomposition_crosssplit_stable_subfeature_registry.csv'
    p_block = tables / 'signal_space_feature_decomposition_authorized_block_summary.csv'
    p_phys = tables / 'signal_space_feature_decomposition_physiological_block_summary.csv'
    p_identity = qc / 'signal_space_feature_decomposition_preprocessing_identity_audit.csv'
    p_stage = qc / 'signal_space_feature_decomposition_participant_stage_preprocessing_audit.csv'
    p_centroid = qc / 'signal_space_feature_decomposition_family_feature_centroid_audit.csv'
    p_cohort = qc / 'signal_space_feature_decomposition_cohort_audit.csv'
    p_rehash = qc / 'signal_space_feature_decomposition_source_rehash.csv'
    atomic_csv(p_labels, pd.DataFrame(label_rows))
    atomic_csv(p_metrics, feature_metric_schema)
    atomic_csv(p_long, long_df.sort_values(['domain', 'k', 'contrast_id', 'block', 'split', 'feature_global_0based']))
    atomic_csv(p_registry, registry.sort_values(['domain', 'k', 'contrast_id', 'block', 'feature_global_0based']))
    atomic_csv(p_block, block_summary.sort_values(['domain', 'k', 'contrast_id', 'block']))
    atomic_csv(p_phys, physiological_summary.sort_values(['domain', 'block']))
    atomic_csv(p_identity, pd.DataFrame(identity_rows).sort_values(['domain', 'split']))
    atomic_csv(p_stage, pd.DataFrame(stage_audit_rows).sort_values(['domain', 'split', 'subject_id']))
    atomic_csv(p_centroid, pd.DataFrame(centroid_audit_rows).sort_values(['domain', 'split', 'k', 'family_1based', 'block']))
    atomic_csv(p_cohort, pd.DataFrame(cohort_rows).sort_values(['domain', 'split']))
    atomic_csv(p_rehash, pd.DataFrame(source_rows))
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'authorization': 'physiological_synthesis synthesis/claims ledger only; no new data opening or inferential testing', 'heldout_physiological_transfer_corrected_final_lock_sha256': heldout_physiological_transfer_LOCK_SHA256, 'heldout_physiological_transfer_signal_space_feature_decomposition_policy_sha256': heldout_physiological_transfer_POLICY_SHA256, 'heldout_physiological_transfer_authorization_registry_sha256': heldout_physiological_transfer_AUTH_SHA256, 'all_authorized_features_emitted': True, 'stable_subfeature_rule': 'within a heldout_physiological_transfer-authorized contrast-block pair: RMS-share family contrast has the same nonzero direction in Discovery/Validation/Confirmation and the dominant family RMS share is > 1/block_size in all three splits', 'signed_delta_and_cosine_decomposition': 'reported for every authorized feature as descriptive decompositions; not used to redefine the stable-subfeature gate', 'new_inferential_tests_permitted': False, 'new_feature_or_block_selection_permitted': False, 'behavioral_or_semantic_values_permitted': False, 'topographic_or_source_localization_inference_permitted': False, 'family_semantic_naming_permitted': False, 'topographic_extension_or_structural_replication_reopening_permitted': False, 'physiological_synthesis_must_report_all_authorized_blocks': True, 'physiological_synthesis_must_preserve_assignment_conditioned_limitation': True, 'interpretation_boundary': 'stable subfeatures describe reproducible contributions within the prespecified compact EEG representation conditional on transferred family geometry; they do not establish causal generators, source localization, or true emotion kinds'}
    atomic_json(project / NEXT_CONFIG, next_policy)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'feature_source_decomposition_complete', 'next_phase': NEXT_PHASE, 'heldout_physiological_transfer_authorized_contrast_blocks': int(len(allowed)), 'authorized_feature_instances_before_split_expansion': int(allowed.n_features.astype(int).sum()), 'long_rows_three_splits': int(len(long_df)), 'crosssplit_feature_rows': int(len(registry)), 'stable_subfeature_rows': stable_count, 'authorized_blocks_with_at_least_one_stable_subfeature': authorized_blocks_with_stable, 'all_authorized_features_emitted': True, 'preprocessing_identity_all_pass': bool(pd.DataFrame(identity_rows).passed.map(bval).all()), 'preprocessing_identity_max_abs_error': float(pd.DataFrame(identity_rows).max_abs_raw_signature_error.max()), 'feature_label_status': label_status, 'feature_label_source_path': label_source, 'behavioral_or_semantic_values_opened': False, 'compact_169_structural_values_opened': True, 'new_inferential_p_values_run': False, 'new_feature_or_block_selection_run': False, 'topographic_or_source_localization_inference_run': False, 'topographic_extension_analysis_reopened': False, 'structural_replication_semantic_analysis_reopened': False, 'assignment_conditioned_limitation_remains_binding': True, 'runtime_seconds': round(time.time() - t0, 3)}
    p_summary = qc / 'signal_space_feature_decomposition_summary.json'
    atomic_json(p_summary, summary)
    contract = {'phase': PHASE, 'scientific_role': 'outcome-free 169-feature decomposition of only heldout_physiological_transfer-authorized replicated-prominent contrast-block pairs', 'authoritative_parent': {'heldout_physiological_transfer_final_lock': heldout_physiological_transfer_LOCK_SHA256, 'heldout_physiological_transfer_policy': heldout_physiological_transfer_POLICY_SHA256, 'heldout_physiological_transfer_authorization_registry': heldout_physiological_transfer_AUTH_SHA256}, 'preprocessing_identity': 'exact structural_replication participant/stage robust standardization; verified by reconstructing all 14 raw signatures from the 169-feature arrays', 'primary_metric': 'participant-balanced per-feature RMS-share family contrast', 'stable_subfeature_rule': next_policy['stable_subfeature_rule'], 'all_features_within_authorized_block_required': True, 'n_authorized_contrast_blocks': int(len(allowed)), 'n_crosssplit_feature_rows': int(len(registry)), 'no_behavior_or_semantics': True, 'no_new_inference': True, 'no_topographic_rescue': True}
    p_contract = qc / 'signal_space_feature_decomposition_contract.json'
    atomic_json(p_contract, contract)
    report_lines = ['# signal_space_feature_decomposition — authorized 169-feature source decomposition', '', f'- heldout_physiological_transfer-authorized contrast×block pairs: **{len(allowed)}**', f'- Cross-split authorized feature rows: **{len(registry)}**', f'- Stable subfeature rows under the prespecified descriptive rule: **{stable_count}**', f'- Authorized block instances containing ≥1 stable subfeature: **{authorized_blocks_with_stable}/{len(allowed)}**', f"- Maximum raw-signature reconstruction error: **{summary['preprocessing_identity_max_abs_error']:.3e}**", f'- Feature-label status: **{label_status}**', '', '## Interpretation boundary', '', 'signal_space_feature_decomposition decomposes only block×contrast pairs authorized before opening the 169-feature source representation.', 'Every feature within every authorized block is emitted. The stable-subfeature rule is descriptive and pre-prespecified.', 'Because held-out family identity remains conditional on the prespecified transferred 14D geometry, these results characterize', 'reproducible compact-EEG feature contributions; they do not establish causal generators, source localization, or emotion semantics.', '', '## Next phase', '', 'physiological_synthesis is synthesis/claims-ledger only. It may not run new tests, select new features, or reopen the closed topographic/structural semantic-mapping branches.']
    p_report = reports / 'signal_space_feature_decomposition_report.md'
    p_report.write_text(''.join(report_lines) + '', encoding='utf-8')
    lock_files = [p_labels, p_metrics, p_long, p_registry, p_block, p_phys, p_identity, p_stage, p_centroid, p_cohort, p_rehash, p_summary, p_contract, p_report, project / NEXT_CONFIG]
    hashes = {str(p.relative_to(project)): sha256_path(p) for p in lock_files}
    final_lock = {'phase': PHASE, 'generated_at': now(), 'status': 'feature_source_decomposition_fixed', 'next_phase': NEXT_PHASE, 'heldout_physiological_transfer_final_lock_sha256': heldout_physiological_transfer_LOCK_SHA256, 'heldout_physiological_transfer_signal_space_feature_decomposition_policy_sha256': heldout_physiological_transfer_POLICY_SHA256, 'heldout_physiological_transfer_authorization_registry_sha256': heldout_physiological_transfer_AUTH_SHA256, 'n_heldout_physiological_transfer_authorized_contrast_blocks': int(len(allowed)), 'n_crosssplit_feature_rows': int(len(registry)), 'n_stable_subfeature_rows': stable_count, 'behavioral_or_semantic_values_opened': False, 'new_inferential_p_values_run': False, 'new_feature_or_block_selection_run': False, 'topographic_or_source_localization_inference_run': False, 'all_authorized_features_emitted': True, 'assignment_conditioned_limitation_remains_binding': True, 'hashes': hashes}
    p_lock = qc / 'signal_space_feature_decomposition_final_output_lock_hashes.json'
    atomic_json(p_lock, final_lock)
    source_path = Path(__file__).resolve()
    include = [*lock_files, p_lock, source_path, project / heldout_physiological_transfer_LOCK, project / heldout_physiological_transfer_POLICY, project / heldout_physiological_transfer_AUTH, project / physiological_characterization_BLOCK_SCHEMA]
    print('signal_space_feature_decomposition PASS')
    print(f'heldout_physiological_transfer-authorized contrast-block pairs: {len(allowed)}')
    print(f'Authorized cross-split feature rows: {len(registry)}')
    print(f'Stable subfeature rows: {stable_count}')
    print(f'Authorized block instances with >=1 stable subfeature: {authorized_blocks_with_stable}/{len(allowed)}')
    print(f"Preprocessing identity max abs raw-signature error: {summary['preprocessing_identity_max_abs_error']:.3e}")
    print('All authorized features emitted: True')
    print('Behavioral/semantic values opened: False')
    print('New inferential p-values run: False')
    print('Topographic/source-localization inference run: False')
    print('Assignment-conditioned limitation remains binding: True')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())