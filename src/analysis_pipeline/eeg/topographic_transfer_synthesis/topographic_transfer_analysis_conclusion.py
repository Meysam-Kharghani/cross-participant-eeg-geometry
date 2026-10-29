# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
PHASE = 'topographic_transfer_synthesis'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
OUT_REL = Path('outputs/interpretability/topographic_transfer_synthesis_topographic_transfer_analysis_conclusion')
UPSTREAM = {'family_interpretability_extension': {'lock': Path('outputs/interpretability/family_interpretability_extension_family_interpretability_extension/qc/family_interpretability_extension_final_output_lock_hashes.json'), 'lock_sha': '0744d4ec50c7e2cee73e96bdabafa60587534940759b519e307b0606862f0dd6', 'summary': Path('outputs/interpretability/family_interpretability_extension_family_interpretability_extension/qc/family_interpretability_extension_summary.json')}, 'outcome_free_family_source_decomposition': {'lock': Path('outputs/interpretability/outcome_free_family_source_decomposition_outcome_free_family_source_decomposition/qc/outcome_free_family_source_decomposition_final_output_lock_hashes.json'), 'lock_sha': 'b9aa42899e30282746526b933688b72bb3b97d680f9bfaaf23c3b5efb671d765', 'summary': Path('outputs/interpretability/outcome_free_family_source_decomposition_outcome_free_family_source_decomposition/qc/outcome_free_family_source_decomposition_summary.json')}, 'full_channel_topology_association': {'lock': Path('outputs/interpretability/full_channel_topology_association_full_channel_topology_association/qc/full_channel_topology_association_final_output_lock_hashes.json'), 'lock_sha': 'fc66f67febff1bc8876ad0871e4823c3aad495589d0cfc17f8e2c448435dd0f1', 'summary': Path('outputs/interpretability/full_channel_topology_association_full_channel_topology_association/qc/full_channel_topology_association_summary.json')}, 'heldout_topography_transfer_design': {'lock': Path('outputs/interpretability/heldout_topography_transfer_design_heldout_topography_transfer_design_specification/qc/heldout_topography_transfer_design_final_output_lock_hashes.json'), 'lock_sha': 'a5590556ee9e4a32ff709a2abb3e19cbee45a698fb85f440f0ae5cf768abbbf0', 'summary': Path('outputs/interpretability/heldout_topography_transfer_design_heldout_topography_transfer_design_specification/qc/heldout_topography_transfer_design_summary.json')}, 'validation_full_channel_transfer': {'lock': Path('outputs/interpretability/validation_full_channel_transfer_validation_full_channel_transfer/qc/validation_full_channel_transfer_final_output_lock_hashes.json'), 'lock_sha': '288ecca209353f24bb979244a0576191c59865173c9ab0dcdc8c34213f60d767', 'summary': Path('outputs/interpretability/validation_full_channel_transfer_validation_full_channel_transfer/qc/validation_full_channel_transfer_summary.json')}}
validation_full_channel_transfer_REPLICATION = Path('outputs/interpretability/validation_full_channel_transfer_validation_full_channel_transfer/tables/validation_full_channel_transfer_validation_parent_replication.csv')

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

def readj(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def verify_lock_and_children(project: Path, phase: str, spec: dict):
    lock_path = project / spec['lock']
    if not lock_path.is_file():
        raise RuntimeError(f'Missing prespecified {phase} final lock: {lock_path}')
    got = sha256(lock_path)
    if got != spec['lock_sha']:
        raise RuntimeError(f"{phase} final-lock SHA drift: {got} != {spec['lock_sha']}")
    lock = readj(lock_path)
    if lock.get('phase') != phase:
        raise RuntimeError(f"Wrong phase inside {lock_path}: {lock.get('phase')}")
    for rel, expected in lock.get('hashes', {}).items():
        p = project / rel
        if not p.is_file():
            raise RuntimeError(f'Missing {phase} fixed artifact: {p}')
        observed = sha256(p)
        if observed != expected:
            raise RuntimeError(f'{phase} fixed artifact drift: {rel}: {observed} != {expected}')
    summary_path = project / spec['summary']
    if not summary_path.is_file():
        raise RuntimeError(f'Missing {phase} summary: {summary_path}')
    summary = readj(summary_path)
    if summary.get('phase') != phase:
        raise RuntimeError(f'Wrong phase inside summary {summary_path}')
    return (lock, summary)

def validate_boundaries(chain: dict) -> None:
    a = chain['family_interpretability_extension'][1]
    d = chain['outcome_free_family_source_decomposition'][1]
    e = chain['full_channel_topology_association'][1]
    f = chain['heldout_topography_transfer_design'][1]
    g = chain['validation_full_channel_transfer'][1]
    if a.get('supported_domains') != [] or a.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('family_interpretability_extension semantic boundary drift')
    d_lock = chain['outcome_free_family_source_decomposition'][0]
    if d.get('behavior_outcomes_read') is not False or d_lock.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('outcome_free_family_source_decomposition outcome/semantic boundary drift')
    if e.get('scientific_boundaries', {}).get('behavioral_report_values_read') is not False:
        raise RuntimeError('full_channel_topology_association behavioral boundary drift')
    if e.get('scientific_boundaries', {}).get('Validation_opened') is not False:
        raise RuntimeError('full_channel_topology_association Validation boundary drift')
    if f.get('heldout_signal_values_opened') is not False:
        raise RuntimeError('heldout_topography_transfer_design held-out boundary drift')
    if f.get('Confirmation_transfer_values_may_open_now') is not False:
        raise RuntimeError('heldout_topography_transfer_design Confirmation boundary drift')
    if g.get('Validation_gate_passed') is not False:
        raise RuntimeError('validation_full_channel_transfer Validation gate is not the prespecified FAIL')
    if g.get('Confirmation_transfer_values_may_open_now') is not False:
        raise RuntimeError('validation_full_channel_transfer unexpectedly permits Confirmation')
    sci = g.get('scientific_boundaries', {})
    if sci.get('Confirmation_signal_or_feature_values_read') is not False:
        raise RuntimeError('validation_full_channel_transfer Confirmation signal boundary drift')
    if sci.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('validation_full_channel_transfer semantic boundary drift')
    if g.get('next_phase') != PHASE:
        raise RuntimeError(f"validation_full_channel_transfer no longer points to {PHASE}: {g.get('next_phase')}")

def self_test() -> None:
    fake = {'family_interpretability_extension': ({}, {'supported_domains': [], 'semantic_relabeling_permitted': False}), 'outcome_free_family_source_decomposition': ({'semantic_relabeling_permitted': False}, {'behavior_outcomes_read': False}), 'full_channel_topology_association': ({}, {'scientific_boundaries': {'behavioral_report_values_read': False, 'Validation_opened': False}}), 'heldout_topography_transfer_design': ({}, {'heldout_signal_values_opened': False, 'Confirmation_transfer_values_may_open_now': False}), 'validation_full_channel_transfer': ({}, {'Validation_gate_passed': False, 'Confirmation_transfer_values_may_open_now': False, 'scientific_boundaries': {'Confirmation_signal_or_feature_values_read': False, 'semantic_relabeling_permitted': False}, 'next_phase': PHASE})}
    validate_boundaries(fake)
    print('topographic_transfer_synthesis self-test: PASS')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    chain = {}
    for phase, spec in UPSTREAM.items():
        print(f'[{now()}] verifying {phase}', flush=True)
        chain[phase] = verify_lock_and_children(project, phase, spec)
    validate_boundaries(chain)
    family_interpretability_extension = chain['family_interpretability_extension'][1]
    outcome_free_family_source_decomposition = chain['outcome_free_family_source_decomposition'][1]
    full_channel_topology_association = chain['full_channel_topology_association'][1]
    heldout_topography_transfer_design = chain['heldout_topography_transfer_design'][1]
    validation_full_channel_transfer = chain['validation_full_channel_transfer'][1]
    rep_path = project / validation_full_channel_transfer_REPLICATION
    if not rep_path.is_file():
        raise RuntimeError(f'Missing validation_full_channel_transfer parent replication table: {rep_path}')
    reps = pd.read_csv(rep_path)
    required = {'parent_test', 'validation_energy_Holm_p', 'validation_mean_template_cosine', 'validation_median_template_cosine', 'validation_concordance_Holm_p', 'validation_parent_replicated'}
    if not required.issubset(reps.columns):
        raise RuntimeError(f'validation_full_channel_transfer replication table columns drift: missing {sorted(required - set(reps.columns))}')
    if len(reps) != 3:
        raise RuntimeError(f'Expected 3 prespecified validation_full_channel_transfer parents, got {len(reps)}')
    replicated = reps['validation_parent_replicated'].astype(str).str.lower().isin(['true', '1', 'yes'])
    if int(replicated.sum()) != 1:
        raise RuntimeError(f'prespecified validation_full_channel_transfer replication count drift: {int(replicated.sum())}')
    out = project / OUT_REL
    final = out / 'qc/topographic_transfer_synthesis_final_output_lock_hashes.json'
    if out.exists():
        if final.is_file() and (not a.force):
            raise RuntimeError(f'Completed topographic_transfer_synthesis output exists: {out}')
        shutil.rmtree(out)
    (out / 'qc').mkdir(parents=True)
    (out / 'tables').mkdir()
    (out / 'reports').mkdir()
    stage_rows = [{'phase': 'family_interpretability_extension', 'role': 'post-hoc family interpretability extension', 'result': '0 supported primary domains after Holm; semantic relabeling remains forbidden', 'heldout_signal_opened_here': False}, {'phase': 'outcome_free_family_source_decomposition', 'role': 'outcome-free compact-source decomposition', 'result': 'F1 shorter/more direct/more aligned; F2 larger/longer/less direct/more opposed', 'heldout_signal_opened_here': False}, {'phase': 'full_channel_topology_association', 'role': 'Discovery full-channel exploratory topology association', 'result': f"{len(full_channel_topology_association.get('supported_exploratory_parents', []))} of 15 parents survived Discovery Holm", 'heldout_signal_opened_here': False}, {'phase': 'heldout_topography_transfer_design', 'role': 'held-out topography transfer analysis specification', 'result': 'Validation-first; parent requires energy+directional concordance; gate >=2/3', 'heldout_signal_opened_here': False}, {'phase': 'validation_full_channel_transfer', 'role': 'one-shot Validation structural transfer', 'result': '1/3 parents replicated; Validation gate FAIL; Confirmation remains closed', 'heldout_signal_opened_here': True}]
    pd.DataFrame(stage_rows).to_csv(out / 'tables/topographic_transfer_synthesis_topographic_branch_synthesis.csv', index=False)
    rep_out = reps[['parent_test', 'validation_energy_Holm_p', 'validation_mean_template_cosine', 'validation_median_template_cosine', 'validation_concordance_Holm_p', 'validation_parent_replicated']].copy()
    rep_out.to_csv(out / 'tables/topographic_transfer_synthesis_validation_parent_replication_prespecified.csv', index=False)
    kres = pd.DataFrame(family_interpretability_extension.get('K_selection_context', []))
    if not kres.empty:
        kres['heldout_transfer_completed'] = kres['k'].eq(2)
        kres['interpretation'] = kres['k'].map(lambda k: 'coarse K=2 solution transferred structurally across held-out cohorts' if int(k) == 2 else 'Discovery-only candidate resolution; not yet granted held-out family status')
        kres.to_csv(out / 'tables/topographic_transfer_synthesis_multiresolution_context.csv', index=False)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'topographic_transfer_analysis_concluded', 'upstream_chain_verified': list(UPSTREAM.keys()), 'topographic_branch_conclusion': {'Discovery_supported_exploratory_parents': full_channel_topology_association.get('supported_exploratory_parents', []), 'Validation_parents_replicated': int(replicated.sum()), 'Validation_parents_total': 3, 'Validation_gate_required': 'at least 2 of 3 parents replicate', 'Validation_gate_passed': False, 'Confirmation_topographic_transfer_opened': False, 'strongest_allowed_reading': 'partial Validation structural support restricted to covariance topology; branch-level transfer criterion failed'}, 'family_interpretation_boundary': {'geometric_reading': outcome_free_family_source_decomposition.get('fixed_geometric_reading', {}), 'semantic_relabeling_permitted': False, 'behavioral_semantic_explanation_supported_in_family_interpretability_extension': False}, 'multiresolution_context': {'Discovery_eligible_K': [int(x['k']) for x in family_interpretability_extension.get('K_selection_context', []) if x.get('eligible')], 'heldout_transferred_resolution': 2, 'finer_K_heldout_family_status_granted': False}, 'scientific_boundaries': {'raw_EEG_or_feature_arrays_opened_in_topographic_transfer_synthesis': False, 'physiology_values_opened_in_topographic_transfer_synthesis': False, 'behavioral_report_values_opened_in_topographic_transfer_synthesis': False, 'emotion_category_values_opened_in_topographic_transfer_synthesis': False, 'Validation_signal_values_opened_in_topographic_transfer_synthesis': False, 'Confirmation_signal_or_feature_values_opened_in_topographic_transfer_synthesis': False, 'family_refit_or_rename': False, 'semantic_relabeling_permitted': False, 'posthoc_rescue_of_failed_validation_full_channel_transfer_gate': False}, 'next_phase': 'multiresolution_experience_report_design', 'next_phase_role': 'Design-specification an outcome-free multi-resolution experience/report dissociation program before opening any new analysis values.'}
    writej(out / 'qc/topographic_transfer_synthesis_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'role': 'closure/synthesis only', 'prespecified_upstream_final_lock_sha256': {phase: spec['lock_sha'] for phase, spec in UPSTREAM.items()}, 'validation_full_channel_transfer_branch_gate': {'rule': '>=2 of 3 prespecified Discovery parents replicate in Validation', 'observed': '1 of 3', 'result': 'FAIL', 'Confirmation_may_open': False}, 'semantic_relabeling_permitted': False, 'posthoc_parent_selection_permitted': False, 'Confirmation_signal_or_feature_values_may_open': False, 'new_inference_performed': False, 'next_phase': 'multiresolution_experience_report_design'}
    writej(out / 'qc/topographic_transfer_synthesis_contract.json', contract)
    lines = ['# topographic_transfer_synthesis — Topographic transfer analysis conclusion', '', 'This phase performs **no new signal or behavioral inference**. It verifies and closes the fixed family_interpretability_extension→validation_full_channel_transfer branch.', '', '## prespecified branch result', '', '- full_channel_topology_association Discovery exploratory full-channel screen: 3/15 parent tests survived Holm.', '- heldout_topography_transfer_design fixed the Validation-first transfer rule before held-out signal values were opened.', '- validation_full_channel_transfer Validation: only 1/3 prespecified parents replicated under the required energy + directional-concordance rule.', '- Required branch gate: at least 2/3 parents.', '- **Branch gate: FAIL.**', '- **Confirmation topographic/signal-derived transfer remains closed.**', '', '## prespecified parent outcomes', '', '| Parent | Energy Holm p | Mean cosine | Median cosine | Concordance Holm p | Replicated |', '|---|---:|---:|---:|---:|---|']
    for r in rep_out.itertuples(index=False):
        lines.append(f"| {r.parent_test} | {float(r.validation_energy_Holm_p):.6g} | {float(r.validation_mean_template_cosine):.6f} | {float(r.validation_median_template_cosine):.6f} | {float(r.validation_concordance_Holm_p):.6g} | {bool(str(r.validation_parent_replicated).lower() in ['true', '1', 'yes'])} |")
    lines += ['', '## Allowed interpretation', '', 'F1 remains a shorter/more-direct/more-segment-aligned trajectory family; F2 remains a larger/longer/less-direct/more-segment-opposed trajectory family. No emotion, valence, liking, context, or other psychological name is licensed.', '', 'The isolated covariance-topology Validation replication may be reported as partial structural support, but it cannot rescue the failed pre-registered branch gate.', '', '## Multi-resolution context', '', 'Discovery K=2, K=3, K=4, and K=6 passed the earlier stability eligibility screen. Only the coarse K=2 solution has completed the fixed held-out structural transfer program; finer resolutions remain Discovery candidates rather than validated family taxonomies.', '', '## Next phase', '', '**multiresolution_experience_report_design** — design-specification an outcome-free multi-resolution experience/report dissociation program before opening new analysis values.']
    (out / 'reports/topographic_transfer_synthesis_report.md').write_text(''.join(lines) + '', encoding='utf-8')
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    hashes = {str(p.relative_to(project)): sha256(p) for p in files}
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': 'topographic_transfer_analysis_concluded', 'hashes': hashes, 'validation_full_channel_transfer_Validation_gate_passed': False, 'parents_replicated': 1, 'Confirmation_signal_or_feature_values_read_in_topographic_transfer_synthesis': False, 'Confirmation_transfer_values_may_open_now': False, 'behavioral_report_values_read_in_topographic_transfer_synthesis': False, 'semantic_relabeling_permitted': False, 'new_inference_performed': False, 'next_phase': 'multiresolution_experience_report_design'})
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    hz = project / '' / f'topographic_transfer_synthesis_topographic_transfer_analysis_conclusion_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    hs = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{hs}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] topographic_transfer_synthesis PASS')
    print('C8 branch gate: FAIL (1/3; required >=2/3)')
    print('Confirmation signal/features opened in topographic_transfer_synthesis: False')
    print('Semantic relabeling permitted: False')
    print('New inference performed: False')
    print('Next phase: multiresolution_experience_report_design')
    print('', hz)
    print('SHA-256:', hs)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())