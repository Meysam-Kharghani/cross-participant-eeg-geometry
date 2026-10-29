# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
PHASE = 'physiological_characterization'
NEXT_PHASE = 'discovery_physiological_reference_atlas'
PHASE_SLUG = 'physiological_characterization_physiological_characterization_design'
OUT_REL = Path('outputs/interpretability') / PHASE_SLUG
POLICY_REL = Path('analysis_specifications/S048_physiological_characterization.json')
EXPECTED_LOCKS = {'topographic_transfer_synthesis': (Path('outputs/interpretability/topographic_transfer_synthesis_topographic_transfer_analysis_conclusion/qc/topographic_transfer_synthesis_final_output_lock_hashes.json'), 'c77d8ede1e309d28588dd3088628d7bfb6b94e6e47ef1e3d995fc7a4c5c19e64'), 'discovery_stimulus_experience_multiresolution_fit': (Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/qc/discovery_stimulus_experience_multiresolution_fit_final_output_lock_hashes.json'), '35b9fa3efacfbe1e3884e2d622e1701538a5375fe2b72ffe1155fc429903c2b7'), 'validation_stimulus_experience_multiresolution_transfer': (Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/qc/validation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'), '5c8431957ac2fe3ffdc8886724b7145359062f83580463b527937164daa9b0ca'), 'confirmation_stimulus_experience_multiresolution_transfer': (Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/qc/confirmation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'), '3b6f03c0d44903345871470cc40e35b0d9d0aaa9d836954e1c0009a7485b6b8b'), 'discovery_report_transition_multiresolution_fit': (Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/qc/discovery_report_transition_multiresolution_fit_final_output_lock_hashes.json'), 'aa09937f090a17eadac760c38ebbe36310c5e88af2e92429f8e4680ca59a1473'), 'validation_report_transition_multiresolution_transfer': (Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/qc/validation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'), '8025e9b1ca3357ebcce055554bda3df6b12e1d1bae08737757cb586ba1faf715'), 'confirmation_report_transition_multiresolution_transfer': (Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/qc/confirmation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'), '14bb0cb021ae071204c90e5be71959bacbc6fc2259e409ed111fc299578d9375'), 'cross_split_semantic_report_synthesis': (Path('outputs/interpretability/cross_split_semantic_report_synthesis_final_crosssplit_semantic_report_synthesis/qc/cross_split_semantic_report_synthesis_final_output_lock_hashes.json'), '967c5cc35877e0e788a672d621c7825300eba7ab88ca2508a49030b081d2bd25')}
KEY_ARTIFACTS = [('experience_discovery_signatures', 'discovery_stimulus_experience_multiresolution_fit', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_experience_signatures.csv'), 'f9128f5225eaa8fd4fc701426a1da5f8385cec0f92d4783b743ae2bf52841326'), ('experience_discovery_assignments', 'discovery_stimulus_experience_multiresolution_fit', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_assignments_all_k.csv'), '91a95a7c891bb53beec1b3c893c8018ed85a77875938efd4817d9c35ae5b21a5'), ('experience_discovery_centroids', 'discovery_stimulus_experience_multiresolution_fit', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_family_centroids_all_k.csv'), '9f2913d4df1774d91e19f8106d320daace75b2297a26ac94472d71cab84d0a05'), ('signature_coordinate_schema', 'discovery_stimulus_experience_multiresolution_fit', Path('outputs/interpretability/discovery_stimulus_experience_multiresolution_fit_discovery_stimulus_experience_multiresolution_fit/tables/discovery_stimulus_experience_multiresolution_fit_signature_coordinate_schema.csv'), '383907c12aa4c7bdffd41e11c321525a3c3588199a7e7a7819970a92fa0284fd'), ('experience_validation_signatures', 'validation_stimulus_experience_multiresolution_transfer', Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv'), '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'), ('experience_validation_assignments', 'validation_stimulus_experience_multiresolution_transfer', Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv'), 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'), ('experience_confirmation_signatures', 'confirmation_stimulus_experience_multiresolution_transfer', Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'), '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'), ('experience_confirmation_assignments', 'confirmation_stimulus_experience_multiresolution_transfer', Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer/tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'), '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'), ('transition_discovery_signatures', 'discovery_report_transition_multiresolution_fit', Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_report_transition_signatures.csv'), '6da778106a3d1ae8649d1aa6757f3ad77e6f0b2e95d57d7421e65e2436252c83'), ('transition_discovery_assignments', 'discovery_report_transition_multiresolution_fit', Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_assignments_all_k.csv'), '8703e1f998a0ffb1c4d0b4ff89fa60b22ade8af2e05a248f35d0ae2f4d6aa75a'), ('transition_discovery_centroids', 'discovery_report_transition_multiresolution_fit', Path('outputs/interpretability/discovery_report_transition_multiresolution_fit_discovery_report_transition_multiresolution_fit/tables/discovery_report_transition_multiresolution_fit_family_centroids_all_k.csv'), 'a9484298b3a55bdec096b7e09efb1fb2a770901fd41f9f5c00579c26dc336cff'), ('transition_validation_signatures', 'validation_report_transition_multiresolution_transfer', Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv'), 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067'), ('transition_validation_assignments', 'validation_report_transition_multiresolution_transfer', Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv'), '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'), ('transition_confirmation_signatures', 'confirmation_report_transition_multiresolution_transfer', Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv'), '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6'), ('transition_confirmation_assignments', 'confirmation_report_transition_multiresolution_transfer', Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer/tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'), 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b')]
HEAVY_SOURCES = [('Discovery', Path('outputs/large/secondary_models_secondary_models_cache_v1/epoch_store/epoch_compact_float32.npy'), '0e6ba0ffe3f72c4f639d443f422b040b3f9db8461e56cfd3e18fb6de072d2d3e'), ('Validation', Path('outputs/large/validation_feature_trajectory_validation_compact_features_v1/global/epoch_compact_float32.npy'), '6f8ab1755f8d950a33351d2032ccd190092656923b09b701ab04e78430fec402'), ('Confirmation', Path('outputs/large/confirmation_feature_trajectory_confirmation_compact_features_v1/global/epoch_compact_float32.npy'), '734fa3d2581d95adc172ea854d4f1fccc2432eaeb167960fdf1e76da0c1509d3')]
COORDINATES = [(0, 'time_frequency_channel_median_iqr', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in robust channel-level time-frequency profile'), (1, 'time_frequency_channel_median_iqr', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of robust channel-level time-frequency pattern'), (2, 'morphology_channel_median_iqr', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in robust waveform-morphology profile'), (3, 'morphology_channel_median_iqr', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of robust waveform-morphology pattern'), (4, 'plv_channel_median_iqr', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in channel-resolved phase-locking summary'), (5, 'plv_channel_median_iqr', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of channel-resolved phase-locking summary'), (6, 'waveform_dct_channel_median_iqr', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in temporal waveform/DCT structure'), (7, 'waveform_dct_channel_median_iqr', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of temporal waveform/DCT structure'), (8, 'covariance_summary', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in covariance-structure summary'), (9, 'covariance_summary', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of covariance-structure summary'), (10, 'plv_global', 'rms_displacement', 'log1p', 'magnitude of stage-to-stage change in global phase-locking summary'), (11, 'plv_global', 'cosine_similarity', 'arctanh_clip_1e-6', 'directional/profile preservation of global phase-locking summary'), (12, 'global', 'rms_displacement', 'log1p', 'holistic 169-feature stage-to-stage displacement; aggregate, not an independent physiological source'), (13, 'global', 'cosine_similarity', 'arctanh_clip_1e-6', 'holistic 169-feature directional preservation; aggregate, not an independent physiological source')]
BLOCKS = [('time_frequency', 0, 60, 60, 'spectro-temporal/time-frequency feature block', 0, 1), ('morphology', 60, 72, 12, 'waveform morphology feature block', 2, 3), ('plv_channel', 72, 84, 12, 'channel-resolved phase-locking feature block', 4, 5), ('waveform_dct', 84, 124, 40, 'temporal waveform/DCT feature block', 6, 7), ('covariance', 124, 145, 21, 'covariance-structure feature block', 8, 9), ('plv_global', 145, 169, 24, 'global phase-locking feature block', 10, 11)]

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def verify_exact(project: Path, rel: Path, expected_sha: str, role: str, rehash: list[dict[str, Any]]) -> Path:
    p = project / rel
    if not p.exists():
        raise RuntimeError(f'missing {role}: {rel}')
    obs = sha256_path(p)
    ok = obs == expected_sha
    rehash.append({'role': role, 'path': str(rel), 'expected_sha256': expected_sha, 'observed_sha256': obs, 'verified': ok, 'values_parsed': False})
    if not ok:
        raise RuntimeError(f'{role} hash drift: {rel}')
    return p

def verify_lock(project: Path, role: str, rel: Path, expected_sha: str, rehash: list[dict[str, Any]]) -> dict[str, Any]:
    p = verify_exact(project, rel, expected_sha, f'{role}_final_lock', rehash)
    lock = readj(p)
    hashes = lock.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{role} lock has no child hashes')
    for child_rel, child_sha in hashes.items():
        cp = project / child_rel
        if not cp.exists():
            raise RuntimeError(f'{role} fixed child missing: {child_rel}')
        obs = sha256_path(cp)
        if obs != child_sha:
            raise RuntimeError(f'{role} child drift: {child_rel}')
    return lock

def canonical_contrasts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, ks in [('experience', [2, 3, 4]), ('report_transition', [2, 3])]:
        for k in ks:
            for low in range(1, k + 1):
                for high in range(low + 1, k + 1):
                    rows.append({'domain': domain, 'k': k, 'family_low_1based': low, 'family_high_1based': high, 'contrast_id': f'{domain}_K{k}_F{high}-F{low}', 'orientation': 'family_high_minus_family_low', 'family_ids_relabelable': False, 'all_14_coordinates_required': True})
    return rows

def policy_object() -> dict[str, Any]:
    return {'phase': PHASE, 'next_phase': NEXT_PHASE, 'program_name': 'replicated_family_physiological_characterization', 'scientific_role': 'outcome-free neurophysiological/feature-space characterization of already-replicated structural_replication families', 'structural_replication_analysis_reopened': False, 'semantic_rescue_permitted': False, 'behavioral_outcome_values_permitted': False, 'emotion_category_values_permitted': False, 'subjective_report_values_permitted': False, 'peripheral_physio_analysis_reopened': False, 'new_family_fit_permitted': False, 'K_reselection_permitted': False, 'family_relabeling_permitted': False, 'topographic_hypothesis_screen_reopened': False, 'external_replication_claim_permitted': False, 'experience_K': [2, 3, 4], 'report_transition_K': [2, 3], 'family_contrast_orientation': 'higher_numeric_family_id minus lower_numeric_family_id; IDs remain Discovery-anchored', 'primary_representation': {'dimension': 14, 'coordinates': 'all 14 standardized signature coordinates; no coordinate subset selection', 'coordinate_pairs': 'RMS displacement + cosine similarity for six non-global blocks, plus a two-coordinate global aggregate', 'global_pair_is_independent_source': False}, 'primary_estimation': {'family_centroid': 'participant-balanced: within-participant family mean, then equal-weight mean across eligible participants', 'Discovery_eligibility': 'all participants/trials in the prespecified discovery_stimulus_experience_multiresolution_fit/discovery_report_transition_multiresolution_fit assignment tables', 'heldout_primary_eligibility': 'participant_eligible == True in prespecified transfer assignment tables', 'heldout_all_assigned_sensitivity': True, 'pairwise_contrasts': 'all canonical pairwise family contrasts at every replicated K'}, 'heldout_fingerprint_metrics': {'contrast_cosine': 'cosine between fixed Discovery contrast vector and held-out participant-balanced contrast vector', 'coordinate_sign_concordance': 'fraction of all 14 coordinates with same sign as Discovery reference', 'non_global_block_contribution': 'sum of squared contrast coordinates for each RMS/cosine block pair divided by sum across coordinates 0..11', 'global_pair': 'reported separately; never counted as a seventh independent source block', 'bootstrap': '2000 participant-level bootstrap resamples per held-out split; deterministic seed namespace 540/signal_space_characterization', 'fingerprint_direction_replication_gate': 'for both Validation and Confirmation: bootstrap 95% lower CI of contrast cosine > 0 AND sign concordance >= 10/14', 'stable_coordinate_rule': 'same contrast sign in Discovery/Validation/Confirmation AND both held-out participant-bootstrap 95% CIs exclude zero in that same direction', 'replicated_prominent_block_rule': 'block contribution > 1/6 in Discovery, Validation, and Confirmation AND at least one member coordinate satisfies the stable-coordinate rule', 'binary_gate_scope': 'descriptive characterization only; failure cannot invalidate structural_replication structural replication and success cannot establish a biological mechanism'}, 'source_decomposition_169': {'stage': 'signal_space_feature_decomposition only, after heldout_physiological_transfer cross-split fingerprint results', 'allowed_blocks': 'only blocks satisfying the pre-prespecified replicated-prominent-block rule; all features within each allowed block are emitted', 'feature_blocks': [{'name': b[0], 'start_0based_inclusive': b[1], 'stop_0based_exclusive': b[2], 'n_features': b[3]} for b in BLOCKS], 'preprocessing_identity': 'must reproduce discovery_stimulus_experience_multiresolution_fit stage_standardize/raw_signatures conventions; no new normalization tuned to signal_space_characterization results', 'rms_decomposition': 'per-feature squared standardized stage difference, normalized within its prespecified block; signed stage difference retained separately', 'cosine_decomposition': 'per-feature contribution to the normalized dot product using the same zero-norm convention as discovery_stimulus_experience_multiresolution_fit', 'stable_subfeature_rule': 'within an allowed block, contribution above the uniform 1/block_size reference in all three splits plus same-direction family contrast across all three; descriptive only', 'topographic_inference_permitted': False}, 'program_stages': {'physiological_characterization': 'analysis specification only; byte/hash verification, no structural values parsed', 'discovery_physiological_reference_atlas': 'Discovery 14D reference atlas; all K and all pairwise contrasts, no selection', 'heldout_physiological_transfer': 'Validation + Confirmation transfer of fixed 14D physiological fingerprints with prespecified metrics/gates', 'signal_space_feature_decomposition': '169-feature source decomposition only for pre-prespecified replicated-prominent blocks; no scalp/topographic rescue', 'physiological_synthesis': 'cross-split physiological synthesis and claims ledger; no new tests or semantics'}, 'interpretation_boundaries': ['RMS displacement is a magnitude-of-change descriptor, not activation strength or emotional intensity.', 'Cosine similarity is profile/directional preservation, not psychological similarity.', 'PLV-derived coordinates describe phase-locking/synchrony summaries and do not localize a generator.', 'Covariance-summary coordinates describe covariance structure and do not by themselves establish anatomical topology.', 'DCT coordinates describe temporal waveform structure and are not source-localized oscillatory generators.', 'Global coordinates are overlapping aggregate descriptors, not independent physiological systems.', 'Stable fingerprints may characterize signal geometry of replicated families; they do not make families true emotion kinds.', 'Low experience/report-transition overlap does not establish independent biological mechanisms.', 'topographic extension analysis conclusion remains binding; signal_space_characterization cannot rescue it with a renamed topographic analysis.']}

def self_test() -> None:
    assert len(COORDINATES) == 14
    assert sum((b[3] for b in BLOCKS)) == 169
    assert [b[1] for b in BLOCKS] == [0, 60, 72, 84, 124, 145]
    assert [b[2] for b in BLOCKS] == [60, 72, 84, 124, 145, 169]
    cons = canonical_contrasts()
    assert len(cons) == 14
    assert sum((r['domain'] == 'experience' for r in cons)) == 10
    assert sum((r['domain'] == 'report_transition' for r in cons)) == 4
    assert policy_object()['structural_replication_analysis_reopened'] is False
    print('physiological_characterization self-test PASS')

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
    project = Path(args.project).resolve()
    out = project / OUT_REL
    if out.exists() and any(out.rglob('*')) and (not args.force):
        raise RuntimeError(f'nonempty physiological_characterization output exists: {out}; quarantine it or use --force intentionally')
    if args.force and out.exists():
        shutil.rmtree(out)
    qc, tables, reports = (out / 'qc', out / 'tables', out / 'reports')
    for d in [qc, tables, reports]:
        d.mkdir(parents=True, exist_ok=True)
    generated = now_iso()
    started = time.time()
    rehash: list[dict[str, Any]] = []
    locks: dict[str, dict[str, Any]] = {}
    parent_refs: list[Path] = []
    for role, (rel, expected) in EXPECTED_LOCKS.items():
        locks[role] = verify_lock(project, role, rel, expected, rehash)
        parent_refs.append(project / rel)
    if locks['cross_split_semantic_report_synthesis'].get('status') != 'structural_replication_branch_complete':
        raise RuntimeError('cross_split_semantic_report_synthesis analysis conclusion status drift')
    if locks['cross_split_semantic_report_synthesis'].get('experience_structurally_replicated_K') != [2, 3, 4]:
        raise RuntimeError('cross_split_semantic_report_synthesis experience K drift')
    if locks['cross_split_semantic_report_synthesis'].get('report_transition_structurally_replicated_K') != [2, 3]:
        raise RuntimeError('cross_split_semantic_report_synthesis transition K drift')
    if locks['cross_split_semantic_report_synthesis'].get('split_level_corrected_parent_rejections') != 0:
        raise RuntimeError('cross_split_semantic_report_synthesis semantic state drift')
    if locks['topographic_transfer_synthesis'].get('status') != 'topographic_transfer_analysis_concluded' or locks['topographic_transfer_synthesis'].get('validation_full_channel_transfer_Validation_gate_passed') is not False:
        raise RuntimeError('topographic_transfer_synthesis topographic closure drift')
    if locks['discovery_stimulus_experience_multiresolution_fit'].get('eligible_K') != [2, 3, 4] or locks['validation_stimulus_experience_multiresolution_transfer'].get('passing_K') != [2, 3, 4] or locks['confirmation_stimulus_experience_multiresolution_transfer'].get('passing_K') != [2, 3, 4]:
        raise RuntimeError('experience structural K lineage drift')
    if locks['discovery_report_transition_multiresolution_fit'].get('eligible_K') != [2, 3] or locks['validation_report_transition_multiresolution_transfer'].get('passing_K') != [2, 3] or locks['confirmation_report_transition_multiresolution_transfer'].get('passing_K') != [2, 3]:
        raise RuntimeError('transition structural K lineage drift')
    source_rows: list[dict[str, Any]] = []
    for role_name, parent, rel, expected in KEY_ARTIFACTS:
        p = verify_exact(project, rel, expected, role_name, rehash)
        if str(rel) not in locks[parent].get('hashes', {}) or locks[parent]['hashes'][str(rel)] != expected:
            raise RuntimeError(f'{role_name} not fixed by expected parent {parent}')
        source_rows.append({'source_role': role_name, 'parent_lock': parent, 'path': str(rel), 'sha256': expected, 'parsed_in_physiological_characterization': False, 'planned_use': 'discovery_physiological_reference_atlas/heldout_physiological_transfer 14D fingerprint execution'})
    for split, rel, expected in HEAVY_SOURCES:
        source_rows.append({'source_role': f'{split.lower()}_compact_169', 'parent_lock': 'historical_fixed_source', 'path': str(rel), 'sha256': expected, 'parsed_in_physiological_characterization': False, 'planned_use': 'signal_space_feature_decomposition only if the pre-prespecified replicated-prominent-block gate allows source decomposition'})
    policy = policy_object()
    policy_path = project / POLICY_REL
    writej(policy_path, policy)
    coord_rows = []
    for i, block, metric, transform, interpretation in COORDINATES:
        coord_rows.append({'coordinate_0based': i, 'standardized_name': f'standardized__{block}__{metric}', 'block': block, 'metric': metric, 'raw_transform': transform, 'prespecified_signal_interpretation': interpretation, 'semantic_interpretation_permitted': False})
    block_rows = []
    for name, start, stop, n, interp, rms_i, cos_i in BLOCKS:
        block_rows.append({'block': name, 'feature_start_0based_inclusive': start, 'feature_stop_0based_exclusive': stop, 'n_features': n, 'rms_coordinate_0based': rms_i, 'cosine_coordinate_0based': cos_i, 'prespecified_interpretation': interp, 'uniform_block_contribution_reference': 1 / 6, 'topographic_inference_permitted': False})
    stage_rows = [{'phase': 'physiological_characterization', 'role': 'analysis specification', 'values_opened': False, 'selection_permitted': False, 'next': 'discovery_physiological_reference_atlas'}, {'phase': 'discovery_physiological_reference_atlas', 'role': 'Discovery 14D physiological reference atlas', 'values_opened': True, 'selection_permitted': False, 'next': 'heldout_physiological_transfer'}, {'phase': 'heldout_physiological_transfer', 'role': 'Validation+Confirmation fixed fingerprint transfer', 'values_opened': True, 'selection_permitted': False, 'next': 'signal_space_feature_decomposition'}, {'phase': 'signal_space_feature_decomposition', 'role': '169-feature decomposition of pre-prespecified replicated-prominent blocks', 'values_opened': True, 'selection_permitted': 'only by physiological_characterization prespecified gate', 'next': 'physiological_synthesis'}, {'phase': 'physiological_synthesis', 'role': 'cross-split physiological synthesis', 'values_opened': False, 'selection_permitted': False, 'next': 'evidence refresh'}]
    claim_rows = [{'claim': 'replicated families have a reproducible 14D EEG feature-space fingerprint', 'allowed_if': 'heldout_physiological_transfer prespecified fingerprint-direction gate passes', 'boundary': 'describes transferred signal geometry, not a distinct biological mechanism'}, {'claim': 'a named non-global block is a replicated prominent contributor', 'allowed_if': 'physiological_characterization replicated-prominent-block rule passes', 'boundary': 'contribution to clustering geometry, not causal generator'}, {'claim': 'specific 169-feature components contribute to a replicated block', 'allowed_if': 'signal_space_feature_decomposition stable-subfeature descriptive rule passes', 'boundary': 'no scalp/source localization unless independently justified'}, {'claim': 'families are specific emotions or true emotion kinds', 'allowed_if': 'never in signal_space_characterization', 'boundary': 'forbidden by structural_replication closure'}, {'claim': 'report-transition structure is distortion/dishonesty/suppression/exaggeration', 'allowed_if': 'never in signal_space_characterization', 'boundary': 'forbidden; neutral report/evaluation-stage transformation only'}, {'claim': 'topographic extension branch is rescued', 'allowed_if': 'never in signal_space_characterization', 'boundary': 'topographic_transfer_synthesis closure remains binding'}]
    files: list[Path] = []
    t1 = tables / 'physiological_characterization_prespecified_coordinate_schema.csv'
    write_csv(t1, coord_rows)
    files.append(t1)
    t2 = tables / 'physiological_characterization_prespecified_feature_block_schema.csv'
    write_csv(t2, block_rows)
    files.append(t2)
    t3 = tables / 'physiological_characterization_prespecified_family_contrast_registry.csv'
    write_csv(t3, canonical_contrasts())
    files.append(t3)
    t4 = tables / 'physiological_characterization_prespecified_source_reference_registry.csv'
    write_csv(t4, source_rows)
    files.append(t4)
    t5 = tables / 'physiological_characterization_program_stages.csv'
    write_csv(t5, stage_rows)
    files.append(t5)
    t6 = tables / 'physiological_characterization_claims_boundaries.csv'
    write_csv(t6, claim_rows)
    files.append(t6)
    qreh = qc / 'physiological_characterization_source_rehash.csv'
    write_csv(qreh, rehash)
    files.append(qreh)
    contract = {'phase': PHASE, 'status': 'design_prespecified', 'next_phase': NEXT_PHASE, 'structural_replication_analysis_concluded_and_not_reopened': True, 'topographic_extension_analysis_concluded_and_not_reopened': True, 'behavioral_values_read': False, 'emotion_category_values_read': False, 'subjective_report_values_read': False, 'structural_signature_or_assignment_values_parsed': False, 'heavy_compact_array_values_parsed': False, 'new_inference_run': False, 'K_reselected': False, 'families_refit': False, 'semantic_relabeling_permitted': False, 'experience_K': [2, 3, 4], 'report_transition_K': [2, 3], 'n_prespecified_coordinates': 14, 'n_non_global_feature_blocks': 6, 'n_compact_features': 169, 'n_canonical_pairwise_contrasts': 14}
    qcon = qc / 'physiological_characterization_contract.json'
    writej(qcon, contract)
    files.append(qcon)
    summary = {**contract, 'generated_at': generated, 'design_principle': 'characterize the already-replicated family geometry first in the exact 14D clustering space, then decompose only pre-prespecified replicated prominent blocks to their 169-feature sources', 'primary_gate': policy['heldout_fingerprint_metrics']['fingerprint_direction_replication_gate'], 'prominent_block_rule': policy['heldout_fingerprint_metrics']['replicated_prominent_block_rule'], 'signal_space_feature_decomposition_gate': 'only physiological_characterization-predefined replicated-prominent blocks from heldout_physiological_transfer may enter 169-feature decomposition'}
    qsum = qc / 'physiological_characterization_summary.json'
    writej(qsum, summary)
    files.append(qsum)
    report = reports / 'physiological_characterization_report.md'
    report.write_text(f'# physiological_characterization — replicated-family physiological characterization analysis specification\n\nC10A specifications an outcome-free neurophysiological characterization program for the already-replicated structural_replication families. It does not reopen structural_replication semantics or C8 topographic rescue.\n\n## Primary 14D fingerprint\nEvery replicated family contrast is characterized using all 14 standardized structural_replication signature coordinates. The six non-global blocks are time-frequency, morphology, channel PLV, waveform/DCT, covariance, and global PLV; the final global RMS/cosine pair is an overlapping aggregate rather than a seventh independent source.\n\nRMS is interpreted only as magnitude of stage-to-stage feature change. Cosine similarity is interpreted only as direction/profile preservation. Neither is emotional intensity or psychological similarity.\n\n## Held-out replication\nDiscovery fixes the reference orientation for every pairwise family contrast. Validation and Confirmation are evaluated with the same participant-balanced geometry, 2000 participant bootstrap resamples, contrast-vector cosine, all-coordinate sign concordance, and six-block contribution profiles. No K, coordinate, contrast, or threshold can be selected after viewing results.\n\n## 169-feature decomposition\nOnly blocks meeting the pre-prespecified replicated-prominent-block rule may be decomposed in signal_space_feature_decomposition. All features in an allowed block must be emitted. Topographic/source-localization inference is forbidden; the failed C8 held-out topographic gate remains closed.\n\n## Interpretation boundary\nA successful signal_space_characterization program can say what signal transformations characterize the replicated family geometry. It cannot name the families as emotions, establish causal neural mechanisms, resurrect failed semantic mappings, or interpret report-transition families as distortion/dishonesty/suppression/exaggeration.\n\nNext phase: **{NEXT_PHASE}** — Discovery 14D physiological reference atlas.', encoding='utf-8')
    files.append(report)
    files.append(policy_path)
    hashes = {}
    for p in files:
        rel = p.resolve().relative_to(project.resolve())
        hashes[str(rel)] = sha256_path(p)
    final_lock = {'phase': PHASE, 'status': 'physiological_characterization_design_prespecified', 'generated_at': generated, 'next_phase': NEXT_PHASE, 'hashes': hashes, 'structural_replication_analysis_reopened': False, 'topographic_extension_analysis_reopened': False, 'behavioral_values_read': False, 'structural_values_parsed': False, 'heavy_compact_values_parsed': False, 'new_inference_run': False, 'K_reselected': False, 'families_refit': False, 'experience_K': [2, 3, 4], 'report_transition_K': [2, 3], 'runtime_seconds': round(time.time() - started, 3)}
    qlock = qc / 'physiological_characterization_final_output_lock_hashes.json'
    writej(qlock, final_lock)
    files.append(qlock)
    print('physiological_characterization PASS')
    print('structural_replication branch reopened: False')
    print('topographic extension branch reopened: False')
    print('Behavior/semantic values opened: False')
    print('prespecified experience K: 2,3,4')
    print('prespecified report-transition K: 2,3')
    print('prespecified coordinates: 14; non-global blocks: 6; compact features: 169')
    print('Canonical pairwise family contrasts: 14')
    print('Next phase:', NEXT_PHASE)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())