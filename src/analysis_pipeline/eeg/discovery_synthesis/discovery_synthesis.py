# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase discovery_synthesis: synthesize fixed Discovery evidence and specification robustness/validation gates.

This phase is deliberately non-inferential. It verifies the fixed mapping_inference mapping
outputs and selected upstream evidence, creates a cross-phase evidence matrix,
locks permitted/prohibited interpretations, pre-registers discovery_robustness_audit robustness
checks, and defines the only allowable future Validation gate. It does not refit
alignment or mapping models and does not read Validation or Confirmation data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'discovery_synthesis'
SOURCE_PHASE = 'mapping_inference'
NEXT_PHASE = 'discovery_robustness_audit'
EXPECTED_mapping_inference_MAPPING_POLICY_SHA = '9e277b8127e38268480b56ee1e89e8e7907076aea7b479e28aa203e714ed97fe'
EXPECTED_cross_person_alignment_fit_ALIGNMENT_POLICY_SHA = 'bfc10e1e646406359b989e03b36d50058c51e311c1df23cdb28c97f0712ee7e1'
EXPECTED_secondary_models_EXECUTION_POLICY_SHA = 'cae19363e1404a0add7a806524647465ccde7d3f3f27aa8be5f5229c7288328c'
ALPHA = 0.05

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def require(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Missing prerequisite(s):\n  ' + '\n  '.join(missing))

def bval(x: Any) -> bool:
    return bool(x) and str(x).strip().lower() not in {'false', '0', 'none', 'nan', ''}

def input_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    r = project / 'outputs/reports'
    return {'mapping_inference_contract': c / 'mapping_inference_mapping_inference_contract.json', 'mapping_inference_next_policy': project / 'analysis_specifications/S042_mapping_inference_discovery_synthesis.json', 'mapping_inference_lock': q / 'mapping_inference_final_output_lock_hashes.json', 'mapping_inference_summary': q / 'mapping_inference_mapping_inference_summary.json', 'mapping_inference_input_rehash': q / 'mapping_inference_input_rehash.csv', 'mapping_inference_implementation': c / 'mapping_inference_fixed_implementation_resolution.json', 'mapping_inference_implementation_lock': q / 'mapping_inference_implementation_lock.json', 'mapping_inference_global': t / 'mapping_inference_global_hypothesis_tests.csv', 'mapping_inference_followups': t / 'mapping_inference_report_dimension_followups.csv', 'mapping_inference_h1_contrasts': t / 'mapping_inference_h1_participant_family_contrasts.csv', 'mapping_inference_h1_descriptives': t / 'mapping_inference_h1_raw_family_descriptives.csv', 'mapping_inference_h1_null': t / 'mapping_inference_h1_null_replicates.csv', 'mapping_inference_h1_boot': t / 'mapping_inference_h1_bootstrap_summary.csv', 'mapping_inference_h2_predictions': t / 'mapping_inference_h2_oof_predictions.csv', 'mapping_inference_h2_selection': t / 'mapping_inference_h2_outer_alpha_selection.csv', 'mapping_inference_h2_null': t / 'mapping_inference_h2_null_replicates.csv', 'mapping_inference_h2_boot': t / 'mapping_inference_h2_bootstrap_summary.csv', 'mapping_inference_h3_predictions': t / 'mapping_inference_h3_oof_state_profiles_predictions.csv', 'mapping_inference_h3_null': t / 'mapping_inference_h3_null_replicates.csv', 'mapping_inference_h3_boot': t / 'mapping_inference_h3_bootstrap_summary.csv', 'mapping_inference_categorical': t / 'mapping_inference_h4_h6_categorical_tests.csv', 'mapping_inference_categorical_crosstabs': t / 'mapping_inference_h4_h6_descriptive_crosstabs.csv', 'mapping_inference_categorical_null': t / 'mapping_inference_h4_h6_null_replicates.csv', 'mapping_inference_report': r / 'mapping_inference_mapping_inference_report.md', 'mapping_inference_log': project / 'outputs/logs/mapping_inference_mapping_inference.log', 'cross_person_alignment_fit_lock': q / 'cross_person_alignment_fit_final_output_lock_hashes.json', 'cross_person_alignment_fit_summary': q / 'cross_person_alignment_fit_cross_person_alignment_summary.json', 'cross_person_alignment_fit_state_selection': t / 'cross_person_alignment_fit_state_family_selection.csv', 'cross_person_alignment_fit_trajectory_selection': t / 'cross_person_alignment_fit_trajectory_family_selection.csv', 'cross_person_alignment_fit_trajectory_candidates': t / 'cross_person_alignment_fit_trajectory_family_candidate_metrics.csv', 'cross_person_alignment_fit_trajectory_assignments': t / 'cross_person_alignment_fit_trajectory_family_assignments.csv', 'cross_person_alignment_fit_trajectory_lopo': t / 'cross_person_alignment_fit_trajectory_family_lopo_diagnostics.csv', 'mapping_design_lock': q / 'mapping_design_final_output_lock_hashes.json', 'mapping_design_trial_manifest': t / 'mapping_design_trial_mapping_manifest.csv', 'secondary_models_contract': c / 'secondary_models_secondary_models_contract.json', 'secondary_models_summary': q / 'secondary_models_secondary_models_summary.json', 'secondary_models_class_metrics': t / 'secondary_models_diagnostic_classification_metrics.csv', 'secondary_models_p2_sign': t / 'secondary_models_person_specific_sign_tests.csv', 'secondary_models_u1_intervals': t / 'secondary_models_unseen_person_bootstrap_intervals.csv'}

def output_paths(project: Path) -> dict[str, Path]:
    c = project / 'config'
    q = project / 'outputs/qc'
    t = project / 'outputs/tables'
    r = project / 'outputs/reports'
    l = project / 'outputs/logs'
    return {'log': l / 'discovery_synthesis_discovery_synthesis.log', 'mapping_inference_rehash': q / 'discovery_synthesis_mapping_inference_final_rehash.csv', 'upstream_rehash': q / 'discovery_synthesis_upstream_evidence_rehash.csv', 'evidence_matrix': t / 'discovery_synthesis_cross_phase_evidence_matrix.csv', 'claims': t / 'discovery_synthesis_fixed_interpretation_claims.csv', 'robustness_manifest': t / 'discovery_synthesis_prespecified_robustness_check_manifest.csv', 'validation_manifest': t / 'discovery_synthesis_prespecified_validation_gate_manifest.csv', 'interpretation_policy': project / 'analysis_specifications/S027_discovery_synthesis_fixed_interpretation.json', 'robustness_policy': project / 'analysis_specifications/S029_discovery_synthesis_robustness_audit.json', 'validation_policy': project / 'analysis_specifications/S030_discovery_synthesis_validation_gate.json', 'summary': q / 'discovery_synthesis_discovery_synthesis_summary.json', 'report': r / 'discovery_synthesis_discovery_synthesis_report.md', 'contract': c / 'discovery_synthesis_discovery_synthesis_contract.json', 'next_policy': project / 'analysis_specifications/S028_discovery_synthesis_robustness_audit.json', 'final_lock': q / 'discovery_synthesis_final_output_lock_hashes.json'}
mapping_inference_KEY_TO_INPUT = {'implementation': 'mapping_inference_contract', 'h1_contrasts': 'mapping_inference_h1_contrasts', 'h1_null': 'mapping_inference_h1_null', 'h2_predictions': 'mapping_inference_h2_predictions', 'h2_null': 'mapping_inference_h2_null', 'h3_profiles': 'mapping_inference_h3_predictions', 'h3_null': 'mapping_inference_h3_null', 'categorical_tests': 'mapping_inference_categorical', 'categorical_null': 'mapping_inference_categorical_null', 'global_tests': 'mapping_inference_global', 'report_followups': 'mapping_inference_followups', 'summary': 'mapping_inference_summary', 'report': 'mapping_inference_report', 'contract': 'mapping_inference_contract', 'next_policy': 'mapping_inference_next_policy'}

def verify_mapping_inference(inputs: dict[str, Path]) -> pd.DataFrame:
    lock = json.loads(inputs['mapping_inference_lock'].read_text())
    contract = json.loads(inputs['mapping_inference_contract'].read_text())
    policy = json.loads(inputs['mapping_inference_next_policy'].read_text())
    summary = json.loads(inputs['mapping_inference_summary'].read_text())
    if lock.get('phase') != SOURCE_PHASE or contract.get('phase') != SOURCE_PHASE:
        raise RuntimeError('mapping_inference phase mismatch')
    if contract.get('mapping_policy_sha256') != EXPECTED_mapping_inference_MAPPING_POLICY_SHA:
        raise RuntimeError('mapping_inference mapping-policy SHA mismatch')
    if policy.get('next_phase') != PHASE or not bval(policy.get('next_phase_permitted')):
        raise RuntimeError('mapping_inference did not authorize discovery_synthesis')
    if bval(lock.get('validation_opened')) or bval(lock.get('confirmation_opened')):
        raise RuntimeError('mapping_inference boundary violation: later cohort was opened')
    if summary.get('decision', {}).get('n_global_supported') != 0:
        raise RuntimeError('mapping_inference summary unexpectedly reports supported global hypotheses')
    hashes = lock.get('hashes', {})
    file_by_hash = {sha256(p): (k, p) for k, p in inputs.items() if p.exists()}
    rows = []
    for logical, expected in hashes.items():
        found = file_by_hash.get(expected)
        rows.append({'source_phase': SOURCE_PHASE, 'object': logical, 'expected_sha256': expected, 'observed_sha256': expected if found else '', 'path': str(found[1]) if found else '', 'match': bool(found)})
        if not found:
            raise RuntimeError(f'mapping_inference fixed object not found or hash mismatch: {logical}')
    return pd.DataFrame(rows)

def verify_upstream(inputs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dlock = json.loads(inputs['cross_person_alignment_fit_lock'].read_text())
    dsummary = json.loads(inputs['cross_person_alignment_fit_summary'].read_text())
    if dlock.get('alignment_policy_sha256') != EXPECTED_cross_person_alignment_fit_ALIGNMENT_POLICY_SHA:
        raise RuntimeError('cross_person_alignment_fit alignment-policy SHA mismatch')
    if bval(dlock.get('validation_opened')) or bval(dlock.get('confirmation_opened')):
        raise RuntimeError('cross_person_alignment_fit boundary violation')
    dmap = {'state_selection': 'cross_person_alignment_fit_state_selection', 'trajectory_selection': 'cross_person_alignment_fit_trajectory_selection', 'trajectory_candidates': 'cross_person_alignment_fit_trajectory_candidates', 'trajectory_selected_assignments': 'cross_person_alignment_fit_trajectory_assignments', 'trajectory_lopo': 'cross_person_alignment_fit_trajectory_lopo'}
    for logical, key in dmap.items():
        exp = dlock.get('hashes', {}).get(logical)
        obs = sha256(inputs[key])
        rows.append({'source_phase': 'cross_person_alignment_fit', 'object': logical, 'expected_sha256': exp, 'observed_sha256': obs, 'path': str(inputs[key]), 'match': exp == obs})
        if not exp or exp != obs:
            raise RuntimeError(f'cross_person_alignment_fit hash mismatch: {logical}')
    rows.append({'source_phase': 'cross_person_alignment_fit', 'object': 'summary_unfixed_but_recorded', 'expected_sha256': '', 'observed_sha256': sha256(inputs['cross_person_alignment_fit_summary']), 'path': str(inputs['cross_person_alignment_fit_summary']), 'match': True})
    if dsummary.get('trajectory_branch', {}).get('selected_shared_k') != 2 or not bval(dsummary.get('trajectory_branch', {}).get('stable_discrete_family_solution')):
        raise RuntimeError('cross_person_alignment_fit trajectory structural result mismatch')
    if bval(dsummary.get('state_branch', {}).get('stable_discrete_family_solution')):
        raise RuntimeError('cross_person_alignment_fit unexpectedly selected discrete shared state families')
    elock = json.loads(inputs['mapping_design_lock'].read_text())
    exp = elock.get('hashes', {}).get('trial_manifest')
    obs = sha256(inputs['mapping_design_trial_manifest'])
    rows.append({'source_phase': 'mapping_design', 'object': 'trial_manifest', 'expected_sha256': exp, 'observed_sha256': obs, 'path': str(inputs['mapping_design_trial_manifest']), 'match': exp == obs})
    if not exp or exp != obs:
        raise RuntimeError('mapping_design trial-manifest hash mismatch')
    ccontract = json.loads(inputs['secondary_models_contract'].read_text())
    if ccontract.get('execution_policy_sha256') != EXPECTED_secondary_models_EXECUTION_POLICY_SHA:
        raise RuntimeError('secondary_models execution-policy SHA mismatch')
    cmap = {'summary': 'secondary_models_summary', 'class_metrics': 'secondary_models_class_metrics', 'p2_sign_tests': 'secondary_models_p2_sign', 'u1_bootstrap_intervals': 'secondary_models_u1_intervals'}
    for logical, key in cmap.items():
        exp = ccontract.get('output_hashes', {}).get(logical)
        obs = sha256(inputs[key])
        rows.append({'source_phase': 'secondary_models', 'object': logical, 'expected_sha256': exp, 'observed_sha256': obs, 'path': str(inputs[key]), 'match': exp == obs})
        if not exp or exp != obs:
            raise RuntimeError(f'secondary_models hash mismatch: {logical}')
    return pd.DataFrame(rows)

def validate_counts(inputs: dict[str, Path]) -> dict[str, Any]:
    global_df = pd.read_csv(inputs['mapping_inference_global'])
    follow = pd.read_csv(inputs['mapping_inference_followups'])
    h1n = pd.read_csv(inputs['mapping_inference_h1_null'])
    h2n = pd.read_csv(inputs['mapping_inference_h2_null'])
    h3n = pd.read_csv(inputs['mapping_inference_h3_null'])
    catn = pd.read_csv(inputs['mapping_inference_categorical_null'])
    h1c = pd.read_csv(inputs['mapping_inference_h1_contrasts'])
    h2p = pd.read_csv(inputs['mapping_inference_h2_predictions'])
    h3p = pd.read_csv(inputs['mapping_inference_h3_predictions'])
    trial = pd.read_csv(inputs['mapping_design_trial_manifest'], low_memory=False)
    lopo = pd.read_csv(inputs['cross_person_alignment_fit_trajectory_lopo'])
    checks = {'global_hypotheses': len(global_df) == 6 and set(global_df['hypothesis']) == {'H1', 'H2', 'H3', 'H4', 'H5', 'H6'}, 'global_supported_zero': int(global_df['global_supported'].astype(bool).sum()) == 0, 'followup_rows': len(follow) == 30, 'h1_null_rows': len(h1n) == 10000, 'h2_null_rows': len(h2n) == 1000, 'h3_null_rows': len(h3n) == 10000, 'categorical_null_rows': len(catn) == 30000, 'h1_participants': len(h1c) == 31 and h1c['subject_id'].nunique() == 31, 'h2_trials': len(h2p) == 1237 and h2p['subject_id'].nunique() == 31, 'h3_states': len(h3p) == 40 and h3p['subject_id'].nunique() == 19, 'trial_manifest': len(trial) == 1237 and trial['subject_id'].nunique() == 31, 'lopo_rows': len(lopo) == 1237 and lopo['held_out_subject_id'].nunique() == 31}
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError('Count/schema boundary failed: ' + ', '.join(failed))
    return checks

def build_evidence(inputs: dict[str, Path]) -> pd.DataFrame:
    c2 = json.loads(inputs['secondary_models_summary'].read_text())
    classm = pd.read_csv(inputs['secondary_models_class_metrics'])
    p2 = pd.read_csv(inputs['secondary_models_p2_sign'])
    u1 = pd.read_csv(inputs['secondary_models_u1_intervals'])
    dsum = json.loads(inputs['cross_person_alignment_fit_summary'].read_text())
    glob = pd.read_csv(inputs['mapping_inference_global'])
    follow = pd.read_csv(inputs['mapping_inference_followups'])
    rows: list[dict[str, Any]] = []
    for item in c2.get('primary_support_decisions', []):
        rows.append({'evidence_id': f"E_PRIMARY_{item['outcome'].upper()}", 'phase': 'primary_inference/C', 'analysis': 'primary pooled signal-specific prediction', 'endpoint': item['outcome'], 'estimate': item['delta_P1_minus_B0'], 'ci_low': item['delta_P1_minus_B0_ci_lower'], 'ci_high': item['delta_P1_minus_B0_ci_upper'], 'p_value': item['primary_permutation_p_holm'], 'decision': 'not_supported', 'interpretation_boundary': 'No signal-specific predictive support versus all prespecified gates.'})
    for _, row in p2.iterrows():
        rows.append({'evidence_id': f'E_PERSON_{str(row.outcome).upper()}', 'phase': 'secondary_models', 'analysis': 'person-specific compact EEG', 'endpoint': row.outcome, 'estimate': row.median_subject_delta_r2, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': row.exact_sign_p_holm, 'decision': 'not_supported', 'interpretation_boundary': 'No group-level support for person-specific implementation.'})
    for outcome in ['valence', 'liking']:
        r = u1[(u1.outcome == outcome) & (u1.model_id == 'DELTA_U1_MINUS_REFERENCE')].iloc[0]
        rows.append({'evidence_id': f'E_UNSEEN_{outcome.upper()}', 'phase': 'secondary_models', 'analysis': 'unseen-person universal decoding', 'endpoint': outcome, 'estimate': r.bootstrap_mean, 'ci_low': r.ci_lower_95, 'ci_high': r.ci_upper_95, 'p_value': np.nan, 'decision': 'not_supported', 'interpretation_boundary': 'No cross-person universal decoding support.'})
    identity = classm[(classm.dataset_id == 'identity_all_three') & (classm.representation == 'compact_signal_507')].iloc[0]
    rows.append({'evidence_id': 'E_IDENTITY', 'phase': 'secondary_models', 'analysis': 'subject identity probe', 'endpoint': 'balanced_accuracy', 'estimate': identity.balanced_accuracy, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'decision': 'strong_diagnostic_signal', 'interpretation_boundary': 'Diagnostic, not a hedonic outcome claim.'})
    for ds, eid in [('context_stimulus_early', 'E_CONTEXT_EARLY'), ('context_stimulus_late', 'E_CONTEXT_LATE'), ('context_rating_early', 'E_CONTEXT_RATING')]:
        r = classm[classm.dataset_id == ds].iloc[0]
        rows.append({'evidence_id': eid, 'phase': 'secondary_models', 'analysis': 'context probe', 'endpoint': ds, 'estimate': r.roc_auc, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'decision': 'diagnostic_structure_present', 'interpretation_boundary': 'Shows contextual information in representation; not outcome mapping.'})
    rows.append({'evidence_id': 'E_SHARED_STATE_TAXONOMY', 'phase': 'cross_person_alignment_fit', 'analysis': 'cross-person local-state alignment', 'endpoint': 'selected_shared_k', 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'decision': 'no_discrete_solution_forced', 'interpretation_boundary': 'Retain continuous state embedding; no universal discrete state taxonomy.'})
    rows.append({'evidence_id': 'E_TRAJECTORY_GEOMETRY', 'phase': 'cross_person_alignment_fit', 'analysis': 'cross-person trajectory geometry', 'endpoint': 'selected_shared_k', 'estimate': float(dsum['trajectory_branch']['selected_shared_k']), 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'decision': 'stable_two_family_geometry', 'interpretation_boundary': f"Outcome-free geometric structure only; LOPO agreement={dsum['trajectory_branch']['lopo_agreement']:.6f}."})
    for _, r in glob.iterrows():
        rows.append({'evidence_id': f'E_MAP_{r.hypothesis}', 'phase': 'mapping_inference', 'analysis': r.description, 'endpoint': r.statistic_name, 'estimate': r.observed_statistic, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': r.holm_p, 'decision': 'not_supported', 'interpretation_boundary': 'Discovery-only post-hoc mapping; family labels remain geometric.'})
    disgust = follow[(follow.parent_hypothesis == 'H1') & (follow.report == 'disgust')].iloc[0]
    rows.append({'evidence_id': 'E_DISGUST_UNGATED', 'phase': 'mapping_inference', 'analysis': 'H1 child follow-up', 'endpoint': 'disgust family-2 minus family-1 adjusted contrast', 'estimate': disgust.effect_or_delta_r2, 'ci_low': disgust.ci_low, 'ci_high': disgust.ci_high, 'p_value': disgust.within_parent_holm_p, 'decision': 'not_interpretable_due_parent_gate', 'interpretation_boundary': 'Cannot support renaming, mechanistic interpretation, or validation target because H1 global Holm p=1.'})
    return pd.DataFrame(rows)

def build_claims() -> pd.DataFrame:
    claims = [('outcome_model_integrity', 'No prespecified primary analysis supports a signal-specific cross-person EEG representation of Valence or Liking.', 'allowed', 'Do not claim absence of all EEG information or prove a universal null.', 'E_PRIMARY_VALENCE;E_PRIMARY_LIKING'), ('C2', 'Person-specific and unseen-person implementations did not rescue Valence/Liking prediction under the prespecified procedures.', 'allowed', 'Do not generalize beyond the tested compact representation, sample, windows, and models.', 'E_PERSON_VALENCE;E_PERSON_LIKING;E_UNSEEN_VALENCE;E_UNSEEN_LIKING'), ('C3', 'The EEG representation strongly contains participant identity and stimulus context information.', 'allowed', 'Do not interpret diagnostic identity/context decoding as hedonic decoding.', 'E_IDENTITY;E_CONTEXT_EARLY;E_CONTEXT_LATE;E_CONTEXT_RATING'), ('C4', 'No stable shared discrete taxonomy of local states was supported; the aligned state representation remains continuous.', 'allowed', 'Do not invent common discrete state families post hoc.', 'E_SHARED_STATE_TAXONOMY'), ('C5', 'A reproducible two-family structure exists in the prespecified three-stage trajectory geometry.', 'allowed', 'Call them Family 1 and Family 2 or geometric trajectory families only.', 'E_TRAJECTORY_GEOMETRY'), ('C6', 'The two trajectory families have no supported global mapping to report profile, context, emotion category, or stimulus identity in Discovery.', 'allowed', 'Do not name either family as pleasure, suffering, disgust, arousal, context, or stimulus class.', 'E_MAP_H1;E_MAP_H4;E_MAP_H5;E_MAP_H6'), ('C7', 'Continuous trajectory geometry and continuous aligned-state coordinates did not add supported report prediction.', 'allowed', 'Do not reinterpret negative delta R2 as biological absence of state structure.', 'E_MAP_H2;E_MAP_H3'), ('C8', 'The raw/child disgust contrast is an ungated Discovery observation only.', 'allowed_with_explicit_ungated_label', 'It cannot motivate family renaming, a confirmatory claim, or selective Validation testing.', 'E_DISGUST_UNGATED'), ('structural_replication', 'All results remain Discovery-only and neither Validation nor Confirmation has been opened.', 'allowed', 'Do not inspect later-cohort outcomes until a separately hashed analysis specification explicitly permits it.', 'E_TRAJECTORY_GEOMETRY;E_MAP_H1;E_MAP_H2;E_MAP_H3;E_MAP_H4;E_MAP_H5;E_MAP_H6')]
    return pd.DataFrame(claims, columns=['claim_id', 'fixed_interpretation', 'status', 'prohibited_overreach', 'source_evidence_ids'])

def build_robustness_manifest() -> pd.DataFrame:
    rows = [('R1', 'integrity_and_reproduction', 'Recompute every mapping_inference hash, row count, empirical p-value and Holm decision from fixed outputs.', 'all hashes/counts exact; empirical p differs by at most one Monte-Carlo unit; Holm decisions exact', 'deterministic', 'all'), ('R2', 'h1_fixed_contrast_jackknife', 'Drop one participant at a time from the 31 fixed H1 contrast vectors; recompute mean vector and shrinkage Hotelling statistic only.', 'descriptive influence bounds only; cannot create a supported claim when prespecified H1 is null', '31 leave-one-participant-out runs', 'H1'), ('R3', 'h2_fixed_oof_jackknife', 'Drop one participant at a time from fixed H2 OOF predictions; recompute global and dimension delta R2 without refitting.', 'report min/max and number positive; prespecified H2 decision remains primary', '31 leave-one-participant-out runs', 'H2'), ('R4', 'h3_fixed_oof_jackknife', 'Drop one participant at a time from fixed H3 OOF state-profile predictions; recompute occupancy-weighted delta R2 without refitting.', 'report min/max and number positive; prespecified H3 decision remains primary', '19 leave-one-participant-out runs', 'H3'), ('R5', 'categorical_fixed_assignment_jackknife', 'Drop one participant at a time; recompute participant-balanced NMI for H4-H6 using fixed families and targets.', 'report influence ranges; no new claim gate', '31 x 3 runs', 'H4-H6'), ('R6', 'lopo_agreeing_subset_sensitivity', 'Restrict H1 and H4-H6 to trajectories whose cross_person_alignment_fit LOPO assignment agrees with full-fit family; rerun exactly the prespecified permutation schemes.', '10,000 permutations per eligible test; labels and family count fixed; results sensitivity-only', '40,000 permutations', 'H1,H4,H5,H6'), ('R7', 'null_calibration_audit', 'Audit finite nulls, null variance, observed percentile, plus-one p-value, Monte-Carlo standard error and resolution.', 'all null arrays finite; nonzero null SD; reported p reproducible', 'deterministic', 'H1-H6'), ('R8', 'trajectory_structural_gate_audit', 'Recompute overall and subject-level LOPO agreement and full-fit family occupancy from fixed cross_person_alignment_fit files.', 'overall agreement >=0.99; minimum subject agreement >=0.90; at least 30/31 subjects >=0.95; both families represented in all 31 subjects', 'deterministic', 'trajectory geometry'), ('R9', 'label_swap_invariance', 'Verify all conclusions/statistics are invariant to swapping numeric Family 1/2 labels, except signed child contrasts which must flip sign exactly.', 'exact invariance or exact sign reversal as specified', 'deterministic', 'H1,H4-H6')]
    return pd.DataFrame(rows, columns=['check_id', 'check_name', 'prespecified_operation', 'interpretation_or_pass_rule', 'planned_work', 'scope'])

def build_validation_manifest() -> pd.DataFrame:
    rows = [('G1', 'integrity_precondition', 'discovery_robustness_audit R1 and R7 pass', 'required', 'No later-cohort access if any source/hash/null-calibration failure occurs.'), ('G2', 'trajectory_structure_precondition', 'discovery_robustness_audit R8 passes all fixed thresholds', 'required', 'This gate concerns only outcome-free trajectory geometry.'), ('G3', 'allowed_validation_scope', 'prespecified transfer of Discovery trajectory preprocessing/centroids and Family 1/2 assignments to Validation without refitting', 'conditionally_permitted_after_new_validation_geometry_transfer_design_specification', 'Validation outcomes/reports remain unread during geometry transfer.'), ('G4', 'mapping_validation_scope', 'Confirmatory testing of H1-H6 in Validation', 'prohibited', 'No mapping_inference global hypothesis was supported in Discovery.'), ('G5', 'state_taxonomy_scope', 'Transfer of a shared discrete local-state taxonomy', 'prohibited', 'cross_person_alignment_fit selected no stable shared discrete state-family solution.'), ('G6', 'family_naming', 'Rename Family 1/2 using any report, context, category, or stimulus label', 'prohibited', 'Families remain geometric regardless of sensitivity results.'), ('G7', 'confirmation_cohort', 'Open Confirmation', 'prohibited', 'Confirmation requires a later explicit specification after a successful, scope-matched Validation result.'), ('G8', 'next_specification', 'validation_geometry_transfer_design must hash all Validation geometry-transfer code, preprocessing, centroids, coverage metrics and failure rules before any Validation file is read', 'required', 'discovery_robustness_audit may recommend validation_geometry_transfer_design but cannot itself open Validation.')]
    return pd.DataFrame(rows, columns=['gate_id', 'gate_name', 'rule', 'status', 'boundary'])

def write_report(path: Path, summary: dict[str, Any], evidence: pd.DataFrame, claims: pd.DataFrame) -> None:
    h = evidence[evidence.evidence_id == 'E_TRAJECTORY_GEOMETRY'].iloc[0]
    lines = ['# Phase discovery_synthesis — fixed Discovery synthesis and gate specification', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Validation opened: **no**', '- Confirmation opened: **no**', '', '## fixed synthesis', '', '- Primary pooled, person-specific, and unseen-person Valence/Liking support gates remain negative.', '- EEG contains strong participant-identity and context information, but this is diagnostic structure rather than hedonic decoding.', '- No shared discrete local-state taxonomy was forced; the continuous aligned state representation is retained.', f'- The two-family trajectory geometry remains structurally strong (selected K={int(h.estimate)}), but all six mapping_inference mappings are unsupported after Holm correction.', '- The disgust child contrast remains ungated and cannot be used to rename families or define a confirmatory target.', '', '## prespecified subsequent analysis', '', '- discovery_robustness_audit executes only the pre-registered fixed-output robustness and influence checks.', '- It may not refit alignment, change assignments, tune models, rename families, or inspect Validation/Confirmation.', '- A later validation_geometry_transfer_design specification may permit outcome-free geometry transfer to Validation only if the structural/integrity gates pass.', '- Confirmatory Validation of H1-H6 is prohibited because no Discovery parent hypothesis was supported.', '', '## fixed interpretation claims', '']
    for _, r in claims.iterrows():
        lines.append(f'- **{r.claim_id}:** {r.fixed_interpretation}')
    atomic_text(path, ''.join(lines) + '')

def self_test() -> None:
    x = {'b': 2, 'a': [1, 3]}
    assert canonical_sha(x) == canonical_sha({'a': [1, 3], 'b': 2})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x.txt'
        atomic_text(p, 'abc')
        assert sha256(p) == hashlib.sha256(b'abc\n').hexdigest()
    claims = build_claims()
    robust = build_robustness_manifest()
    gate = build_validation_manifest()
    assert len(claims) == 9 and len(robust) == 9 and (len(gate) == 8)
    assert (gate.status == 'prohibited').sum() >= 4
    print('discovery_synthesis self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project.resolve()
    inputs = input_paths(project)
    outputs = output_paths(project)
    require(inputs.values())
    if outputs['final_lock'].exists() and (not args.force):
        raise RuntimeError(f"Output already exists: {outputs['final_lock']} (use --force)")
    log_lines: list[str] = []

    def note(msg: str) -> None:
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        log_lines.append(line)
        atomic_text(outputs['log'], ''.join(log_lines) + '')
    note('Verifying every fixed mapping_inference object and authorization for discovery_synthesis.')
    c4f_rehash = verify_mapping_inference(inputs)
    atomic_csv(outputs['mapping_inference_rehash'], c4f_rehash)
    note('Verifying required cross_person_alignment_fit, mapping_design, and secondary_models upstream evidence objects.')
    upstream_rehash = verify_upstream(inputs)
    atomic_csv(outputs['upstream_rehash'], upstream_rehash)
    note('Auditing fixed row counts, null replicate counts, cohorts, and global decisions.')
    count_checks = validate_counts(inputs)
    note('Building cross-phase evidence matrix and locking interpretation boundaries.')
    evidence = build_evidence(inputs)
    claims = build_claims()
    atomic_csv(outputs['evidence_matrix'], evidence)
    atomic_csv(outputs['claims'], claims)
    note('Freezing discovery_robustness_audit robustness checks and the future Validation/Confirmation gate.')
    robustness = build_robustness_manifest()
    validation = build_validation_manifest()
    atomic_csv(outputs['robustness_manifest'], robustness)
    atomic_csv(outputs['validation_manifest'], validation)
    source_hashes = {'mapping_inference_final_lock': sha256(inputs['mapping_inference_lock']), 'mapping_inference_contract': sha256(inputs['mapping_inference_contract']), 'cross_person_alignment_fit_final_lock': sha256(inputs['cross_person_alignment_fit_lock']), 'mapping_design_final_lock': sha256(inputs['mapping_design_lock']), 'secondary_models_contract': sha256(inputs['secondary_models_contract']), 'trial_manifest': sha256(inputs['mapping_design_trial_manifest']), 'trajectory_lopo': sha256(inputs['cross_person_alignment_fit_trajectory_lopo'])}
    interpretation_spec = {'phase': PHASE, 'fixed_before_robustness_execution': True, 'family_labels': ['Family 1', 'Family 2'], 'renaming_permitted': False, 'global_mapping_support': [], 'disgust_child_status': 'ungated_nonconfirmatory_observation', 'discrete_shared_state_taxonomy_supported': False, 'continuous_state_embedding_retained': True, 'trajectory_geometry_supported_as_structure_only': True, 'validation_opened': False, 'confirmation_opened': False, 'claims_file_sha256': sha256(outputs['claims']), 'evidence_matrix_sha256': sha256(outputs['evidence_matrix'])}
    atomic_json(outputs['interpretation_policy'], interpretation_spec)
    interpretation_sha = canonical_sha(interpretation_spec)
    robustness_spec = {'phase': PHASE, 'next_phase': NEXT_PHASE, 'source_hashes': source_hashes, 'interpretation_policy_canonical_sha256': interpretation_sha, 'check_manifest_sha256': sha256(outputs['robustness_manifest']), 'seed': 540, 'permutations_lopo_subset': 10000, 'no_refit_alignment': True, 'no_change_family_count_or_assignment': True, 'no_model_retuning': True, 'validation_opened': False, 'confirmation_opened': False, 'structural_gate_thresholds': {'overall_lopo_agreement_minimum': 0.99, 'minimum_subject_lopo_agreement': 0.9, 'subjects_at_or_above_0_95_minimum': 30, 'participants_with_both_full_fit_families_minimum': 31}, 'sensitivity_results_cannot_override_prespecified_mapping_inference_primary_decisions': True}
    atomic_json(outputs['robustness_policy'], robustness_spec)
    robustness_sha = canonical_sha(robustness_spec)
    validation_spec = {'phase': PHASE, 'fixed_before_any_validation_access': True, 'validation_currently_open': False, 'confirmation_currently_open': False, 'conditional_future_scope': 'outcome-free transfer of the fixed two-family trajectory geometry only', 'required_preconditions': ['discovery_robustness_audit_R1_pass', 'discovery_robustness_audit_R7_pass', 'discovery_robustness_audit_R8_pass', 'new_validation_geometry_transfer_design_specification_hashed_before_validation_read'], 'prohibited_scopes': ['confirmatory H1-H6 mapping validation', 'shared discrete state taxonomy transfer', 'family renaming', 'confirmation access'], 'validation_gate_manifest_sha256': sha256(outputs['validation_manifest']), 'robustness_policy_canonical_sha256': robustness_sha}
    atomic_json(outputs['validation_policy'], validation_spec)
    validation_sha = canonical_sha(validation_spec)
    summary = {'generated_at': now(), 'phase': PHASE, 'status': 'discovery_synthesis_interpretation_robustness_and_validation_gate_prespecified_candidate' if not args.test_mode else 'test_mode_complete', 'counts': {'mapping_inference_fixed_objects_verified': len(c4f_rehash), 'upstream_objects_verified_or_recorded': len(upstream_rehash), 'evidence_rows': len(evidence), 'fixed_claims': len(claims), 'robustness_checks': len(robustness), 'validation_gate_rules': len(validation), **{f'check_{k}': bool(v) for k, v in count_checks.items()}}, 'decision': {'global_mapping_hypotheses_supported': [], 'trajectory_geometry_status': 'stable_two_family_outcome_free_structure', 'shared_discrete_state_taxonomy_status': 'not_supported_continuous_only', 'family_renaming_permitted': False, 'discovery_robustness_audit_robustness_execution_permitted': not args.test_mode, 'validation_opened': False, 'confirmation_opened': False, 'future_validation_scope_if_all_gates_pass': 'outcome_free_trajectory_geometry_transfer_only', 'confirmatory_mapping_validation_permitted': False}, 'policy_canonical_sha256': {'interpretation': interpretation_sha, 'robustness': robustness_sha, 'validation_gate': validation_sha}}
    atomic_json(outputs['summary'], summary)
    write_report(outputs['report'], summary, evidence, claims)
    primary_keys = [k for k in outputs if k not in {'contract', 'next_policy', 'final_lock', 'log'}]
    out_hashes = {k: sha256(outputs[k]) for k in primary_keys}
    contract = {'generated_at': now(), 'phase': PHASE, 'source_phase': SOURCE_PHASE, 'status': summary['status'], 'source_hashes': source_hashes, 'policy_canonical_sha256': summary['policy_canonical_sha256'], 'counts': summary['counts'], 'decision': summary['decision'], 'output_hashes': out_hashes, 'alignment_or_mapping_refit': False, 'family_count_or_assignment_modified': False, 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['contract'], contract)
    next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': not args.test_mode, 'next_phase_role': 'execute only the prespecified fixed-output robustness, influence, LOPO-stable-subset, null-calibration, and structural-gate checks', 'required_boundary': ['verify every discovery_synthesis and source hash', 'use fixed cross_person_alignment_fit family assignments and mapping_inference predictions/statistics without refitting', 'do not rename families or create a new mapping claim', 'sensitivity results cannot override prespecified mapping_inference primary decisions', 'Validation and Confirmation remain closed'], 'robustness_policy_canonical_sha256': robustness_sha, 'validation_gate_policy_canonical_sha256': validation_sha, 'validation_opened': False, 'confirmation_opened': False}
    atomic_json(outputs['next_policy'], next_policy)
    note('discovery_synthesis synthesis and specifications complete; locking outputs without opening later cohorts.')
    out_hashes['contract'] = sha256(outputs['contract'])
    out_hashes['next_policy'] = sha256(outputs['next_policy'])
    out_hashes['log'] = sha256(outputs['log'])
    atomic_json(outputs['final_lock'], {'generated_at': now(), 'phase': PHASE, 'policy_canonical_sha256': summary['policy_canonical_sha256'], 'validation_opened': False, 'confirmation_opened': False, 'hashes': out_hashes})
    if not args.test_mode:
        pass
    print(pd.read_csv(inputs['mapping_inference_global'])[['hypothesis', 'observed_statistic', 'holm_p', 'global_supported']].to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())