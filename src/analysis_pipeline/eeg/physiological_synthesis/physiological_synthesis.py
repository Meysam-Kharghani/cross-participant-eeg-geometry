# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'physiological_synthesis'
signal_space_feature_decomposition_LOCK_SHA = 'f4a026f7241ea45e7431238f5d48ef6b9c1501fe23d525a411b473ccf63e855e'
signal_space_feature_decomposition_POLICY_SHA = 'f20618a9bb03199e82b2d5e55fb58b2ee52e8faf7281cf7ff8cee306ad0d8821'
signal_space_feature_decomposition_SOURCE_SHA = 'd621510c31af00b07891def0d1869aad5f4fd105427e01b3145bbb9ea4167c61'
heldout_physiological_transfer_LOCK_SHA = '190b9226fc272d901b30ae52cde8666c035262bf5adc653ef562614fecc3b1df'
heldout_physiological_transfer_AUTH_SHA = '873d08f61663d15f79829f1921f43e1c21dc04ed70bc98f50e4130d8a4d800ba'
signal_space_feature_decomposition = Path('outputs/interpretability/signal_space_feature_decomposition_feature_source_decomposition')
heldout_physiological_transfer = Path('outputs/interpretability/heldout_physiological_transfer_heldout_physiological_fingerprint_transfer')
OUT = Path('outputs/interpretability/physiological_synthesis_physiological_synthesis')
LOCK = signal_space_feature_decomposition / 'qc/signal_space_feature_decomposition_final_output_lock_hashes.json'
POLICY = Path('analysis_specifications/S064_signal_space_feature_decomposition_physiological_synthesis.json')
signal_space_feature_decomposition_SOURCE = Path('src/analysis_pipeline/eeg/signal_space_feature_decomposition/feature_source_decomposition.py')
heldout_physiological_transfer_LOCK = heldout_physiological_transfer / 'qc/heldout_physiological_transfer_final_output_lock_hashes.json'
heldout_physiological_transfer_AUTH = heldout_physiological_transfer / 'tables/heldout_physiological_transfer_replicated_prominent_block_registry.csv'
BLOCK = signal_space_feature_decomposition / 'tables/signal_space_feature_decomposition_authorized_block_summary.csv'
PHYS = signal_space_feature_decomposition / 'tables/signal_space_feature_decomposition_physiological_block_summary.csv'
STABLE = signal_space_feature_decomposition / 'tables/signal_space_feature_decomposition_crosssplit_stable_subfeature_registry.csv'
LONG = signal_space_feature_decomposition / 'tables/signal_space_feature_decomposition_authorized_feature_decomposition_long.csv'
IDENTITY = signal_space_feature_decomposition / 'qc/signal_space_feature_decomposition_preprocessing_identity_audit.csv'
COHORT = signal_space_feature_decomposition / 'qc/signal_space_feature_decomposition_cohort_audit.csv'
REHASH = signal_space_feature_decomposition / 'qc/signal_space_feature_decomposition_source_rehash.csv'
SUMMARY = signal_space_feature_decomposition / 'qc/signal_space_feature_decomposition_summary.json'
ALL_BLOCKS = ['time_frequency', 'morphology', 'plv_channel', 'waveform_dct', 'covariance', 'plv_global']
POSSIBLE = {'experience': 10, 'report_transition': 4}

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def require(project, rel, expected):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'missing required file: {rel}')
    obs = sha(p)
    if obs != expected:
        raise RuntimeError(f'SHA drift: {rel}\nexpected {expected}\nobserved {obs}')

def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + '')

def verify_parent(project):
    require(project, LOCK, signal_space_feature_decomposition_LOCK_SHA)
    require(project, POLICY, signal_space_feature_decomposition_POLICY_SHA)
    require(project, signal_space_feature_decomposition_SOURCE, signal_space_feature_decomposition_SOURCE_SHA)
    require(project, heldout_physiological_transfer_LOCK, heldout_physiological_transfer_LOCK_SHA)
    require(project, heldout_physiological_transfer_AUTH, heldout_physiological_transfer_AUTH_SHA)
    lock = json.loads((project / LOCK).read_text())
    expected = {'phase': 'signal_space_feature_decomposition', 'status': 'feature_source_decomposition_fixed', 'next_phase': 'physiological_synthesis', 'n_heldout_physiological_transfer_authorized_contrast_blocks': 31, 'n_crosssplit_feature_rows': 938, 'n_stable_subfeature_rows': 277, 'behavioral_or_semantic_values_opened': False, 'new_inferential_p_values_run': False, 'new_feature_or_block_selection_run': False, 'topographic_or_source_localization_inference_run': False, 'all_authorized_features_emitted': True, 'assignment_conditioned_limitation_remains_binding': True}
    for k, v in expected.items():
        if lock.get(k) != v:
            raise RuntimeError(f'signal_space_feature_decomposition lock field drift: {k}')
    if len(lock.get('hashes', {})) != 15:
        raise RuntimeError('signal_space_feature_decomposition lock must contain 15 child hashes')
    for rel, exp in lock['hashes'].items():
        p = project / rel
        if not p.is_file() or sha(p) != exp:
            raise RuntimeError(f'signal_space_feature_decomposition child hash failure: {rel}')
    pol = json.loads((project / POLICY).read_text())
    checks = {'phase': 'signal_space_feature_decomposition', 'next_phase': 'physiological_synthesis', 'behavioral_or_semantic_values_permitted': False, 'new_feature_or_block_selection_permitted': False, 'new_inferential_tests_permitted': False, 'family_semantic_naming_permitted': False, 'topographic_or_source_localization_inference_permitted': False, 'topographic_extension_or_structural_replication_reopening_permitted': False, 'physiological_synthesis_must_preserve_assignment_conditioned_limitation': True, 'physiological_synthesis_must_report_all_authorized_blocks': True}
    for k, v in checks.items():
        if pol.get(k) != v:
            raise RuntimeError(f'signal_space_feature_decomposition->physiological_synthesis policy drift: {k}')
    return lock

def audit_tables(project):
    block = pd.read_csv(project / BLOCK)
    phys = pd.read_csv(project / PHYS)
    stable = pd.read_csv(project / STABLE)
    long = pd.read_csv(project / LONG)
    ident = pd.read_csv(project / IDENTITY)
    cohort = pd.read_csv(project / COHORT)
    reh = pd.read_csv(project / REHASH)
    summ = json.loads((project / SUMMARY).read_text())
    if len(block) != 31 or int(block.n_stable_subfeatures.sum()) != 277:
        raise RuntimeError('block summary count drift')
    if not block.any_stable_subfeature.astype(bool).all():
        raise RuntimeError('not all 31 blocks have stable subfeatures')
    if not block.all_features_emitted.astype(bool).all():
        raise RuntimeError('not all authorized features emitted')
    if len(stable) != 938 or int(stable.stable_subfeature.astype(bool).sum()) != 277:
        raise RuntimeError('stable registry count drift')
    if len(long) != 2814 or set(long.split.unique()) != {'Discovery', 'Validation', 'Confirmation'}:
        raise RuntimeError('long-table split expansion drift')
    if len(ident) != 6 or not ident.passed.astype(bool).all():
        raise RuntimeError('preprocessing identity failed')
    if float(ident.max_abs_raw_signature_error.max()) > 1e-09:
        raise RuntimeError('identity error exceeds tolerance')
    if len(reh) != 25 or not reh.verified.astype(bool).all():
        raise RuntimeError('signal_space_feature_decomposition source rehash is not 25/25')
    if cohort.behavioral_or_semantic_values_opened.astype(bool).any():
        raise RuntimeError('behavior/semantic values opened')
    d = stable.discovery_contrast_rms_share.to_numpy(float)
    v = stable.validation_contrast_rms_share.to_numpy(float)
    c = stable.confirmation_contrast_rms_share.to_numpy(float)
    same = (np.sign(d) == np.sign(v)) & (np.sign(d) == np.sign(c)) & (np.sign(d) != 0)
    dd = np.where(d > 0, stable.discovery_high_family_rms_share, stable.discovery_low_family_rms_share)
    vd = np.where(d > 0, stable.validation_high_family_rms_share, stable.validation_low_family_rms_share)
    cd = np.where(d > 0, stable.confirmation_high_family_rms_share, stable.confirmation_low_family_rms_share)
    u = stable.uniform_share_reference.to_numpy(float)
    recalc = same & (dd > u) & (vd > u) & (cd > u)
    if not np.array_equal(recalc, stable.stable_subfeature.astype(bool).to_numpy()):
        raise RuntimeError('independent stable-subfeature recomputation disagrees')
    if summ.get('stable_subfeature_rows') != 277 or summ.get('authorized_blocks_with_at_least_one_stable_subfeature') != 31:
        raise RuntimeError('signal_space_feature_decomposition summary drift')
    return (block, phys, stable, long, ident, cohort, reh)

def synthesize(block, phys, stable):
    domain = []
    for dom, g in block.groupby('domain', sort=False):
        nf = int(g.n_features.sum())
        ns = int(g.n_stable_subfeatures.sum())
        domain.append({'domain': dom, 'n_possible_family_contrasts': POSSIBLE[dom], 'n_authorized_contrast_block_pairs': len(g), 'n_authorized_feature_instances': nf, 'n_stable_subfeature_instances': ns, 'fraction_stable_within_authorized_feature_instances': ns / nf, 'n_distinct_authorized_blocks': int(g.block.nunique()), 'all_authorized_blocks_have_stable_subfeature': bool(g.any_stable_subfeature.astype(bool).all())})
    domain = pd.DataFrame(domain)
    cov = []
    for dom in ['experience', 'report_transition']:
        p = phys[phys.domain == dom].set_index('block')
        for b in ALL_BLOCKS:
            if b in p.index:
                r = p.loc[b]
                na = int(r.n_authorized_contrasts)
                nr = int(r.n_authorized_contrast_feature_rows)
                ns = int(r.n_stable_subfeature_rows)
                nu = int(r.n_unique_feature_indices_stable_at_least_once)
            else:
                na = nr = ns = nu = 0
            cov.append({'domain': dom, 'block': b, 'n_possible_family_contrasts': POSSIBLE[dom], 'n_authorized_contrasts': na, 'authorized_contrast_coverage': na / POSSIBLE[dom], 'n_authorized_contrast_feature_rows': nr, 'n_stable_subfeature_rows': ns, 'stable_fraction_within_authorized_feature_rows': ns / nr if nr else np.nan, 'n_unique_feature_indices_stable_at_least_once': nu})
    coverage = pd.DataFrame(cov)
    ledger = block.copy()
    ledger['stable_fraction_within_block'] = ledger.n_stable_subfeatures / ledger.n_features
    stable_only = stable[stable.stable_subfeature.astype(bool)].copy()
    return (domain, coverage, ledger, stable_only)

def make_claims():
    return pd.DataFrame([['physiological_synthesis_01', 'SUPPORTED_DESCRIPTIVE', 'The 169-feature decomposition is numerically faithful to the historical structural_replication 14D signatures.', 'All six identity audits pass; max reconstruction error <1e-9.', 'Implementation identity, not biological validation.'], ['physiological_synthesis_02', 'SUPPORTED_DESCRIPTIVE', 'Every pre-authorized contrast×block pair contains cross-split stable RMS-share subfeatures.', '31/31 authorized blocks contain >=1 stable subfeature; 277/938 feature instances satisfy the prespecified rule.', 'Assignment-conditioned descriptive stability; no new p-value.'], ['physiological_synthesis_03', 'SUPPORTED_DESCRIPTIVE', 'Experience-family physiology is multicomponent in the compact representation.', 'Authorized experience contrasts span all six prespecified blocks; time-frequency and morphology each occur in 6/10 contrasts.', 'Does not establish six independent mechanisms.'], ['physiological_synthesis_04', 'SUPPORTED_DESCRIPTIVE', 'Report-transition characterization emphasizes time-frequency, morphology and PLV/synchrony-related blocks.', 'Time-frequency and morphology each occur in 3/4 transition contrasts; channel/global PLV each in 2/4.', 'Does not prove a distinct report-stage biological mechanism.'], ['physiological_synthesis_05', 'SUPPORTED_ABSENCE_WITHIN_prespecified_GATE', 'Covariance and waveform/DCT were not authorized for report-transition contrasts.', '0/4 transition contrasts were authorized for either block.', 'Not evidence that those processes are biologically absent.'], ['physiological_synthesis_06', 'NOT_SUPPORTED', 'The replicated families have specific emotion identities.', 'structural_replication semantic/report branch closed without a corrected replicated semantic mapping.', 'Family semantic naming remains forbidden.'], ['physiological_synthesis_07', 'NOT_SUPPORTED', 'The fingerprints identify causal generators or source-localized mechanisms.', 'signal_space_characterization uses compact scalp-EEG summaries and performs no source-localization/topographic rescue.', 'C8 topographic closure remains binding.'], ['physiological_synthesis_08', 'NOT_SUPPORTED', 'Experience and report-transition families are biologically independent systems.', 'Different fingerprints and low assignment overlap are descriptive only.', 'No mechanistic independence claim.']], columns=['claim_id', 'status', 'claim', 'evidence', 'boundary'])

def self_test():
    d = np.array([0.1, -0.1, 0.1])
    v = np.array([0.2, -0.2, -0.2])
    c = np.array([0.3, -0.3, 0.3])
    if ((np.sign(d) == np.sign(v)) & (np.sign(d) == np.sign(c)) & (np.sign(d) != 0)).tolist() != [True, True, False]:
        raise RuntimeError('self-test failed')
    print('physiological_synthesis self-test PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.project:
        raise SystemExit('--project required')
    project = Path(a.project).resolve()
    print(f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] verifying signal_space_feature_decomposition lock/policy/ancestry")
    lock = verify_parent(project)
    print(f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] independently auditing signal_space_feature_decomposition synthesis inputs")
    block, phys, stable, long, ident, cohort, reh = audit_tables(project)
    out = project / OUT
    if out.exists() and any(out.rglob('*')) and (not a.force):
        raise RuntimeError(f'nonempty physiological_synthesis output: {out}')
    if a.force and out.exists():
        shutil.rmtree(out)
    tabs = out / 'tables'
    qc = out / 'qc'
    reps = out / 'reports'
    for x in [tabs, qc, reps]:
        x.mkdir(parents=True, exist_ok=True)
    domain, coverage, ledger, stable_only = synthesize(block, phys, stable)
    claims = make_claims()
    generated = []

    def save_csv(df, name):
        p = tabs / name
        df.to_csv(p, index=False)
        generated.append(p)
        return p
    save_csv(domain, 'physiological_synthesis_domain_physiological_summary.csv')
    save_csv(coverage, 'physiological_synthesis_block_coverage_summary.csv')
    save_csv(ledger, 'physiological_synthesis_contrast_block_ledger.csv')
    save_csv(stable_only, 'physiological_synthesis_stable_subfeature_ledger.csv')
    save_csv(claims, 'physiological_synthesis_claims_ledger.csv')
    maxerr = float(ident.max_abs_raw_signature_error.max())
    report = f'# physiological_synthesis — final physiological synthesis\n\n## fixed state\n- heldout_physiological_transfer-authorized contrast×block pairs: **31**\n- Authorized compact-feature instances: **938**\n- prespecified stable-subfeature instances: **277**\n- Authorized blocks with ≥1 stable subfeature: **31/31**\n- Max historical 14D reconstruction error: **{maxerr:.3e}**\n- New tests/data/feature selection in physiological_synthesis: **none**\n\n## Experience-family characterization\nAcross 10 pre-registered experience contrasts, 21 contrast×block pairs were authorized.\nThey contain 650 feature instances, with 175 satisfying the prespecified cross-split stable rule (26.9%).\nAuthorized contrasts span all six blocks. Time-frequency and morphology are the most recurrent,\neach occurring in 6/10 contrasts. Covariance, waveform/DCT and PLV-related contributions distinguish\nadditional subsets of contrasts. This supports a multicomponent compact-EEG fingerprint rather than\na single physiological axis.\n\n## Report-transition characterization\nAcross 4 pre-registered transition contrasts, 10 contrast×block pairs were authorized.\nThey contain 288 feature instances, with 102 stable instances (35.4%).\nTime-frequency and morphology occur in 3/4 contrasts; channel PLV and global PLV each in 2/4.\nCovariance and waveform/DCT were not authorized in any transition contrast under the prespecified gate.\nThis is compatible with a compact-EEG transition fingerprint emphasizing spectral/morphological\nchange plus synchrony-related structure, but it does not establish a distinct report-stage mechanism.\n\n## Interpretation boundary\nAll results are assignment-conditioned characterization of the prespecified compact EEG representation.\nThey do not establish emotion identities, causal generators, source localization, scalp-topographic\nreplication, or biological independence between experience and report-transition systems.\nTopographic-extension and structural semantic-mapping closures remain binding.\n\n## Branch decision\nThe physiological characterization branch is closed. No post-hoc feature rescue or within-signal_space_characterization selection is permitted.\nThe defensible subsequent analysis is evidence-ledger refresh using fixed structural_replication+signal_space_characterization claims.'
    rp = reps / 'physiological_synthesis_report.md'
    rp.write_text(report)
    generated.append(rp)
    summ = {'phase': PHASE, 'status': 'physiological_synthesis_complete', 'generated_at': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'signal_space_feature_decomposition_final_lock_sha256': signal_space_feature_decomposition_LOCK_SHA, 'signal_space_feature_decomposition_physiological_synthesis_policy_sha256': signal_space_feature_decomposition_POLICY_SHA, 'n_authorized_contrast_blocks': 31, 'n_authorized_feature_instances': 938, 'n_stable_subfeature_instances': 277, 'authorized_blocks_with_stable_subfeature': 31, 'max_preprocessing_identity_error': maxerr, 'new_data_opened': False, 'new_inferential_tests_run': False, 'new_feature_or_block_selection_run': False, 'family_semantic_naming_run': False, 'topographic_or_source_localization_inference_run': False, 'assignment_conditioned_limitation_remains_binding': True, 'topographic_extension_analysis_reopened': False, 'structural_replication_semantic_analysis_reopened': False, 'signal_space_characterization_analysis_concluded': True, 'next_phase': 'evidence-ledger refresh'}
    sp = qc / 'physiological_synthesis_summary.json'
    write_json(sp, summ)
    generated.append(sp)
    cp = qc / 'physiological_synthesis_contract.json'
    write_json(cp, {'phase': PHASE, 'role': 'synthesis_and_claims_ledger_only', 'parents': {'signal_space_feature_decomposition_final_lock': {'path': str(LOCK), 'sha256': signal_space_feature_decomposition_LOCK_SHA}, 'signal_space_feature_decomposition_physiological_synthesis_policy': {'path': str(POLICY), 'sha256': signal_space_feature_decomposition_POLICY_SHA}, 'signal_space_feature_decomposition_source': {'path': str(signal_space_feature_decomposition_SOURCE), 'sha256': signal_space_feature_decomposition_SOURCE_SHA}, 'heldout_physiological_transfer_final_lock': {'path': str(heldout_physiological_transfer_LOCK), 'sha256': heldout_physiological_transfer_LOCK_SHA}, 'heldout_physiological_transfer_authorization_registry': {'path': str(heldout_physiological_transfer_AUTH), 'sha256': heldout_physiological_transfer_AUTH_SHA}}, 'forbidden': ['new_data_opening', 'new_inferential_tests', 'new_feature_or_block_selection', 'family_semantic_naming', 'topographic_or_source_localization_inference', 'topographic_extension_or_structural_replication_reopening']})
    generated.append(cp)
    refs = [(LOCK, signal_space_feature_decomposition_LOCK_SHA, False), (POLICY, signal_space_feature_decomposition_POLICY_SHA, True), (signal_space_feature_decomposition_SOURCE, signal_space_feature_decomposition_SOURCE_SHA, False), (heldout_physiological_transfer_LOCK, heldout_physiological_transfer_LOCK_SHA, False), (heldout_physiological_transfer_AUTH, heldout_physiological_transfer_AUTH_SHA, False)]
    for rel in [BLOCK, PHYS, STABLE, LONG, IDENTITY, COHORT, REHASH, SUMMARY]:
        refs.append((rel, lock['hashes'][str(rel)], True))
    rr = []
    for rel, exp, parsed in refs:
        obs = sha(project / rel)
        rr.append({'path': str(rel), 'expected_sha256': exp, 'observed_sha256': obs, 'verified': obs == exp, 'values_parsed': parsed})
    sr = qc / 'physiological_synthesis_source_rehash.csv'
    pd.DataFrame(rr).to_csv(sr, index=False)
    generated.append(sr)
    closure = project / 'analysis_specifications/S049_physiological_synthesis_signal_space_characterization_scope.json'
    write_json(closure, {'phase': PHASE, 'status': 'signal_space_characterization_analysis_concluded', 'authorization': 'evidence-ledger refresh only unless a separately pre-registered new analysis is specified', 'new_within_signal_space_characterization_analysis_permitted': False, 'posthoc_feature_rescue_permitted': False, 'semantic_family_naming_permitted': False, 'topographic_rescue_permitted': False, 'assignment_conditioned_limitation_remains_binding': True, 'topographic_and_structural_semantic_closures_remain_binding': True})
    generated.append(closure)
    lockout = qc / 'physiological_synthesis_final_output_lock_hashes.json'
    hashes = {str(p.relative_to(project)): sha(p) for p in generated}
    write_json(lockout, {'phase': PHASE, 'status': 'physiological_synthesis_fixed', 'generated_at': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'signal_space_feature_decomposition_final_lock_sha256': signal_space_feature_decomposition_LOCK_SHA, 'signal_space_feature_decomposition_physiological_synthesis_policy_sha256': signal_space_feature_decomposition_POLICY_SHA, 'n_authorized_contrast_blocks': 31, 'n_authorized_feature_instances': 938, 'n_stable_subfeature_instances': 277, 'new_data_opened': False, 'new_inferential_tests_run': False, 'new_feature_or_block_selection_run': False, 'family_semantic_naming_run': False, 'topographic_or_source_localization_inference_run': False, 'assignment_conditioned_limitation_remains_binding': True, 'signal_space_characterization_analysis_concluded': True, 'next_phase': 'evidence-ledger refresh', 'hashes': hashes})
    generated.append(lockout)
    source = Path(__file__).resolve()
    print('physiological_synthesis PASS')
    print('signal_space_feature_decomposition child hashes verified: 15/15')
    print('signal_space_feature_decomposition source references verified: 25/25')
    print('Authorized contrast-block pairs: 31')
    print('Authorized feature instances: 938')
    print('Stable subfeature instances: 277')
    print('Authorized blocks with >=1 stable subfeature: 31/31')
    print(f'Preprocessing identity max abs error: {maxerr:.3e}')
    print('New data opened: False')
    print('New inferential tests run: False')
    print('New feature/block selection run: False')
    print('signal_space_characterization branch: CLOSED')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())