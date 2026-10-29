# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
PHASE = 'source_gsr_dynamic_design'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
F_LOCK = 'outputs/qc/exact_physio_eeg_sync_recovery_final_output_lock_hashes.json'
F_SUM = 'outputs/qc/exact_physio_eeg_sync_recovery_exact_sync_recovery_summary.json'
F_ROUTE = 'analysis_specifications/S031_exact_physio_eeg_sync_recovery.json'
E_LOCK = 'outputs/qc/mapping_design_final_output_lock_hashes.json'
E_MAP = 'outputs/tables/mapping_design_trial_mapping_manifest.csv'
E_MAP_SHA = '6f130a2431e3f0c4fc881c2e8e20ea99290459fc97b6c6695da746ac38a82cab'
CANONICAL = ['sad4', 'sad5', 'sad8', 'dis4', 'dis5', 'dis8', 'fear4', 'fear5', 'fear8', 'neu4', 'neu5', 'neu8', 'joy4', 'joy5', 'joy8', 'ten4', 'ten5', 'ten8', 'ins4', 'ins5', 'ins8']
EXPECTED_COLS = ['Trial', 'SCR', 'SCL', 'iSCR']
MIN_PARTICIPANTS = 18
MIN_MAPPED_FRACTION = 0.7
MIN_TRIALS_PER_PARTICIPANT = 16
MIN_TRIALS_PER_CONTEXT = 4
SEGMENTS_PER_TRIAL = 10
SEGMENT_SECONDS = 3

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def writecsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict('records')
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open('w', newline='', encoding='utf-8') as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

def boolish(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

def verify_lock(project: Path, rel: str, phase: str):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'Missing upstream lock: {rel}')
    lock = readj(p)
    if lock.get('phase') != phase:
        raise RuntimeError(f"Wrong upstream phase in {rel}: {lock.get('phase')}")
    for r, h in lock.get('hashes', {}).items():
        q = project / r
        if not q.is_file():
            raise RuntimeError(f'Missing fixed artifact: {r}')
        g = sha256(q)
        if g != h:
            raise RuntimeError(f'Upstream hash mismatch {r}: {g} != {h}')
    return lock

def normalize_stimulus_id(x: Any):
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s in {'nan', 'n/a', 'na', 'none'}:
        return None
    s = s.replace('_', '').replace('-', '').replace(' ', '')
    aliases = {'sadness': 'sad', 'sad': 'sad', 'disgust': 'dis', 'dis': 'dis', 'fear': 'fear', 'neutral': 'neu', 'neu': 'neu', 'joy': 'joy', 'tenderness': 'ten', 'tender': 'ten', 'ten': 'ten', 'inspiration': 'ins', 'inspire': 'ins', 'ins': 'ins'}
    m = re.search('(sad|dis|fear|neu|joy|ten|ins)(4|5|8)', s)
    if m:
        return m.group(1) + m.group(2)
    for long, short in aliases.items():
        if long in s:
            m = re.search('(4|5|8)', s)
            if m:
                return short + m.group(1)
    return None

def source_path(dataset: Path, sid: str, context: str):
    ses = 'ses-ima' if context == 'imagery' else 'ses-vid'
    return dataset / 'derivatives' / sid / ses / 'physio' / f'{sid}_{ses}_task-emotion_GSRfeatures.csv'

def audit_source_file(path: Path, sid: str, context: str):
    base = {'subject_id': sid, 'context': context, 'source_path': str(path), 'found': False, 'source_sha256': '', 'header_columns': '', 'expected_columns_present': False, 'n_rows_trial_column': 0, 'n_normalized_trial_rows': 0, 'n_unique_normalized_stimuli': 0, 'all_21_canonical_stimuli_present': False, 'all_canonical_stimuli_exactly_10_rows': False, 'source_valid_for_dynamic_route': False, 'unparsed_trial_label_examples': '', 'gsr_derivative_amplitude_read': False}
    if not path.is_file():
        return (base, {})
    base['found'] = True
    base['source_sha256'] = sha256(path)
    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    base['header_columns'] = '|'.join(map(str, hdr))
    base['expected_columns_present'] = all((c in hdr for c in EXPECTED_COLS))
    if 'Trial' not in hdr:
        return (base, {})
    d = pd.read_csv(path, usecols=['Trial'])
    base['n_rows_trial_column'] = int(len(d))
    norm = d['Trial'].map(normalize_stimulus_id)
    base['n_normalized_trial_rows'] = int(norm.notna().sum())
    bad = d.loc[norm.isna(), 'Trial'].dropna().astype(str).unique()[:8]
    base['unparsed_trial_label_examples'] = '|'.join(bad)
    cnt = Counter(norm.dropna())
    base['n_unique_normalized_stimuli'] = int(len(cnt))
    base['all_21_canonical_stimuli_present'] = set(cnt) == set(CANONICAL)
    base['all_canonical_stimuli_exactly_10_rows'] = all((cnt.get(s, 0) == SEGMENTS_PER_TRIAL for s in CANONICAL))
    base['source_valid_for_dynamic_route'] = bool(base['expected_columns_present'] and len(d) == len(CANONICAL) * SEGMENTS_PER_TRIAL and base['all_21_canonical_stimuli_present'] and base['all_canonical_stimuli_exactly_10_rows'])
    rowmap = {}
    if base['source_valid_for_dynamic_route']:
        for stim in CANONICAL:
            idx = [int(i) for i, v in enumerate(norm.tolist()) if v == stim]
            rowmap[stim] = idx
    return (base, rowmap)

def self_test():
    vals = ['sad4', 'Sadness_4', 'DIS-5', 'fear8', 'neutral 4', 'joy_5', 'tenderness8', 'inspiration-4']
    exp = ['sad4', 'sad4', 'dis5', 'fear8', 'neu4', 'joy5', 'ten8', 'ins4']
    got = [normalize_stimulus_id(x) for x in vals]
    if got != exp:
        raise RuntimeError(f'stimulus parser self-test failed: {got}')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x.csv'
        rows = []
        for stim in CANONICAL:
            for _ in range(10):
                rows.append({'Trial': stim, 'SCR': 999, 'SCL': 888, 'iSCR': 777})
        pd.DataFrame(rows).to_csv(p, index=False)
        a, m = audit_source_file(p, 'sub-00', 'imagery')
        if not a['source_valid_for_dynamic_route'] or len(m) != 21:
            raise RuntimeError('source audit self-test failed')
    print('source_gsr_dynamic_design self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--method-snapshot', type=Path, default=None)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    project = args.project
    dataset = args.dataset
    if not project.is_dir():
        raise RuntimeError(f'Project not found: {project}')
    if not dataset.is_dir():
        raise RuntimeError(f'Dataset not found: {dataset}')
    flock = verify_lock(project, F_LOCK, 'exact_physio_eeg_sync_recovery')
    if flock.get('exact_sync_gate_pass') is not False or flock.get('next_phase') != 'source_gsr_dynamic_design':
        raise RuntimeError('exact_physio_eeg_sync_recovery does not authorize source_gsr_dynamic_design')
    fsum = readj(project / F_SUM)
    froute = readj(project / F_ROUTE)
    if fsum.get('raw_amplitude_analysis_permitted') is not False:
        raise RuntimeError('exact_physio_eeg_sync_recovery raw amplitude boundary mismatch')
    if not fsum.get('fallback_author_GSR_stimulus_dynamic_audit_permitted'):
        raise RuntimeError('exact_physio_eeg_sync_recovery did not authorize author-GSR fallback audit')
    if froute.get('selected_next_phase') != 'source_gsr_dynamic_design':
        raise RuntimeError('exact_physio_eeg_sync_recovery route-policy mismatch')
    ep = project / E_LOCK
    if not ep.is_file():
        raise RuntimeError(f'Missing upstream lock: {E_LOCK}')
    elock = readj(ep)
    if elock.get('phase') != 'mapping_design':
        raise RuntimeError(f"Wrong upstream phase in {E_LOCK}: {elock.get('phase')}")
    if elock.get('hashes', {}).get('trial_manifest') != E_MAP_SHA:
        raise RuntimeError('mapping_design final lock does not carry the prespecified trial-manifest digest')
    mp = project / E_MAP
    if not mp.is_file():
        raise RuntimeError(f'Missing mapping_design trial mapping: {E_MAP}')
    if sha256(mp) != E_MAP_SHA:
        raise RuntimeError(f'mapping_design trial mapping SHA mismatch: {sha256(mp)}')
    usecols = ['subject_id', 'trial_number', 'context', 'stimulus_mapping_available', 'stimulus_id', 'acquisition_fraction']
    m = pd.read_csv(mp, usecols=usecols)
    m['subject_id'] = m.subject_id.astype(str)
    m['context'] = m.context.astype(str)
    m['stimulus_mapping_available'] = m.stimulus_mapping_available.map(boolish)
    discovery_subjects = sorted(m.subject_id.unique())
    source_rows = []
    row_maps = {}
    for sid in discovery_subjects:
        for ctx in ['imagery', 'video']:
            p = source_path(dataset, sid, ctx)
            a, rm = audit_source_file(p, sid, ctx)
            source_rows.append(a)
            if a['source_valid_for_dynamic_route']:
                row_maps[sid, ctx] = rm
    sdef = pd.DataFrame(source_rows)
    valid_files = int(sdef.source_valid_for_dynamic_route.sum()) if len(sdef) else 0
    valid_subjects = sorted(set(sdef.loc[sdef.source_valid_for_dynamic_route, 'subject_id'])) if len(sdef) else []
    map_rows = []
    for _, r in m.sort_values(['subject_id', 'trial_number']).iterrows():
        sid = str(r.subject_id)
        ctx = str(r.context)
        stim = normalize_stimulus_id(r.stimulus_id) if bool(r.stimulus_mapping_available) else None
        rm = row_maps.get((sid, ctx), {})
        idx = rm.get(stim, []) if stim else []
        mapped = len(idx) == SEGMENTS_PER_TRIAL
        p = source_path(dataset, sid, ctx)
        map_rows.append({'subject_id': sid, 'trial_number': int(r.trial_number), 'context': ctx, 'acquisition_fraction': float(r.acquisition_fraction) if pd.notna(r.acquisition_fraction) else math.nan, 'stimulus_mapping_available': bool(r.stimulus_mapping_available), 'stimulus_id': stim or '', 'author_gsr_source_path': str(p) if p.is_file() else '', 'author_gsr_source_sha256': sha256(p) if p.is_file() else '', 'source_row_indices': '|'.join(map(str, idx)), 'n_author_gsr_segments': len(idx), 'mapped_to_exact_10x3s_author_gsr': mapped, 'gsr_derivative_amplitude_read': False})
    tdf = pd.DataFrame(map_rows)
    part = []
    for sid in discovery_subjects:
        q = tdf[tdf.subject_id == sid]
        mapped = q[q.mapped_to_exact_10x3s_author_gsr]
        ni = int((mapped.context == 'imagery').sum())
        nv = int((mapped.context == 'video').sum())
        nt = int(len(mapped))
        eligible = bool(nt >= MIN_TRIALS_PER_PARTICIPANT and ni >= MIN_TRIALS_PER_CONTEXT and (nv >= MIN_TRIALS_PER_CONTEXT))
        part.append({'subject_id': sid, 'fixed_discovery_trials': int(len(q)), 'mapped_author_gsr_trials': nt, 'mapped_imagery_trials': ni, 'mapped_video_trials': nv, 'min16_and_both_contexts_pass': eligible, 'gsr_derivative_amplitude_read': False})
    pdf = pd.DataFrame(part)
    n_eligible = int(pdf.min16_and_both_contexts_pass.sum()) if len(pdf) else 0
    denom = int(tdf.stimulus_mapping_available.sum()) if len(tdf) else 0
    numer = int((tdf.stimulus_mapping_available & tdf.mapped_to_exact_10x3s_author_gsr).sum()) if len(tdf) else 0
    frac = numer / denom if denom else 0.0
    gate = bool(n_eligible >= MIN_PARTICIPANTS and frac >= MIN_MAPPED_FRACTION)
    design = {'phase': PHASE, 'generated_at': now(), 'scope': 'Discovery-only author-provided stimulus-period dynamic GSR derivatives; rating-period physiology remains untested.', 'author_derivative_interpretation': {'source_pattern': 'derivatives/{subject_id}/{ses}/physio/{subject_id}_{ses}_task-emotion_GSRfeatures.csv', 'segments_per_trial': SEGMENTS_PER_TRIAL, 'seconds_per_segment': SEGMENT_SECONDS, 'primary_channels': ['SCR', 'SCL'], 'sensitivity_channel': ['iSCR'], 'primary_representation': '20D ordered dynamic profile = 10 SCR bins + 10 SCL bins; preserve bin order; no early/late relabeling.', 'rating_period_claim_permitted': False}, 'preprocessing_prespecified': {'feature_selection': 'none', 'numeric_coercion': 'values opened only in discovery_source_gsr_materialization_qc after H source gate passes', 'imputation': 'training-fold median per bin/channel', 'scaling': 'training-fold median/IQR; pooled fallback for zero IQR', 'winsorization': 'after robust scaling, clip to [-6,6]', 'primary_iSCR_use': False, 'iSCR_role': 'prespecified sensitivity only; not added to primary 20D family'}, 'nuisance_covariates': ['context', 'acquisition_fraction'], 'primary_tests': {'DG1': {'question': 'Does the 20D stimulus-period GSR dynamic profile differ between fixed EEG trajectory families beyond context and acquisition order?', 'statistic': 'participant-balanced within-person multivariate profile contrast after nuisance adjustment', 'permutation': '10000 within-participant label permutations preserving participant trial counts', 'family_labels_opened_in_phase': 'discovery_dynamic_gsr_inference, not source_gsr_dynamic_design/I'}, 'DG2': {'question': 'Can dynamic GSR predict fixed trajectory family in an unseen person beyond context + acquisition_fraction?', 'model': 'nested leave-one-participant-out ridge logistic regression; baseline nuisance-only vs full nuisance+20D GSR', 'metric': 'balanced accuracy gain full-minus-baseline', 'permutation': '1000 participant-stratified label permutations with model selection repeated', 'bootstrap': '5000 participant bootstraps'}, 'DG3': {'question': 'Does the 20D stimulus-period GSR dynamic profile couple to the fixed continuous 6D EEG trajectory geometry beyond nuisance?', 'statistic': 'participant-balanced nuisance-adjusted multivariate RV/cross-covariance association', 'permutation': '10000 within-participant permutations', 'geometry_columns': ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']}}, 'multiplicity': 'Holm correction jointly across DG1, DG2, DG3.', 'robustness_prespecified': ['complete-case sensitivity where feasible', 'SCR-only and SCL-only follow-ups only if their parent primary test is supported', 'iSCR-augmented sensitivity only after primary results are fixed', 'leave-one-participant-out influence audit', 'family numerical-label swap invariance for DG1/DG2'], 'heldout_policy': {'Validation_derivative_values_opened': False, 'Confirmation_derivative_values_opened': False, 'opening_condition': 'No held-out GSR derivative values may be opened unless >=1 DG1-DG3 primary is Holm-supported and its prespecified robustness gate passes in Discovery.'}, 'forbidden': ['using Valence, Liking, other reports, emotion category, or stimulus semantics to choose features/windows/tests', 'claiming these author 10x3-s stimulus derivatives measure rating-period physiology', 'opening raw GSR/PPG amplitude after exact_physio_eeg_sync_recovery exact-sync failure', 'opening Validation or Confirmation derivative values during source_gsr_dynamic_design/I/J Discovery work', 'post-hoc selection of individual GSR bins because they correlate with an outcome']}
    paths = {'sources': project / 'outputs/qc/source_gsr_dynamic_design_author_gsr_source_schema_audit.csv', 'mapping': project / 'outputs/tables/source_gsr_dynamic_design_author_gsr_trial_row_mapping.csv', 'elig': project / 'outputs/tables/source_gsr_dynamic_design_discovery_author_gsr_eligibility.csv', 'design': project / 'analysis_specifications/S066_source_gsr_dynamic.json', 'summary': project / 'outputs/qc/source_gsr_dynamic_design_author_gsr_dynamic_design_specification_summary.json', 'next': project / 'analysis_specifications/S065_source_gsr_dynamic.json', 'report': project / 'outputs/reports/source_gsr_dynamic_design_author_gsr_dynamic_design_specification_report.md', 'contract': project / 'config/source_gsr_dynamic_design_author_gsr_dynamic_design_specification_contract.json'}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    writecsv(paths['sources'], source_rows)
    writecsv(paths['mapping'], map_rows)
    writecsv(paths['elig'], part)
    writej(paths['design'], design)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'source_gate_passed' if gate else 'source_gate_failed', 'counts': {'fixed_discovery_participants': len(discovery_subjects), 'fixed_discovery_trials': int(len(tdf)), 'stimulus_mappable_fixed_trials': denom, 'valid_author_gsr_source_files': valid_files, 'participants_with_any_valid_author_gsr_source': len(valid_subjects), 'mapped_exact_10x3s_author_gsr_trials': numer, 'eligible_participants_min16_both_contexts': n_eligible}, 'fractions': {'mapped_fraction_among_stimulus_mappable_trials': frac}, 'gates': {'minimum_participants_required': MIN_PARTICIPANTS, 'minimum_mapped_fraction_required': MIN_MAPPED_FRACTION, 'minimum_trials_per_participant': MIN_TRIALS_PER_PARTICIPANT, 'minimum_trials_per_context': MIN_TRIALS_PER_CONTEXT, 'source_gate_pass': gate}, 'boundaries': {'GSR_derivative_amplitude_read': False, 'raw_GSR_amplitude_read': False, 'raw_PPG_amplitude_read': False, 'behavioral_outcomes_read': False, 'family_assignments_read': False, 'trajectory_geometry_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False}}
    writej(paths['summary'], summary)
    nextpol = {'phase': PHASE, 'generated_at': now(), 'selected_next_phase': 'discovery_source_gsr_materialization_qc' if gate else 'dynamic_gsr_synthesis', 'if_source_gate_passes': {'next_phase': 'discovery_source_gsr_materialization_qc', 'role': 'Discovery-only materialization/QC of prespecified 10x3-s author GSR dynamic values under exact source-row mapping; no family/geometry/outcome inference.'}, 'if_source_gate_fails': {'next_phase': 'dynamic_gsr_synthesis', 'role': 'Close stimulus-only author-GSR fallback without opening derivative amplitudes.'}, 'raw_event_aligned_GSR_PPG_reopened': False, 'rating_period_physio_tested': False}
    writej(paths['next'], nextpol)
    rep = f"# source_gsr_dynamic_design — author-provided stimulus-only dynamic GSR analysis specification\n\nStatus: **{('PASS' if gate else 'FAIL')}**.\n\nexact_physio_eeg_sync_recovery did not recover <=1 s exact EEG↔raw-physiology synchronization, so raw event-aligned GSR/PPG amplitude remains closed.\n\nThis fallback uses the dataset authors' already-derived GSR files only as a **stimulus-period dynamic sequence**. The published method describes 3-s windows for SCR/SCL extraction; the released files are audited here as 21 trials × 10 ordered rows per trial.\n\n## Discovery source audit\n- fixed Discovery participants: **{len(discovery_subjects)}**\n- fixed Discovery trials: **{len(tdf)}**\n- Stimulus-mappable fixed trials: **{denom}**\n- Valid author GSR source files: **{valid_files}**\n- Exact mapped 10×3-s trial sequences: **{numer}** / {denom} ({frac:.4f})\n- Eligible participants (>=16 mapped trials and >=4/context): **{n_eligible}**\n- prespecified source gate: **{('PASS' if gate else 'FAIL')}**\n\n## Boundary\nNo SCR/SCL/iSCR numeric values were read in this phase. Only file headers, the `Trial` column, row counts, source hashes, and outcome-free mapping_design lineage fields were materialized.\n\nThe fallback cannot test physiology during rating and cannot replace the failed raw exact-synchronization route."
    paths['report'].write_text(rep, encoding='utf-8')
    method_sha = sha256(args.method_snapshot) if args.method_snapshot and args.method_snapshot.is_file() else ''
    contract = {'phase': PHASE, 'generated_at': now(), 'inputs': {'exact_physio_eeg_sync_recovery_final_lock_sha256': sha256(project / F_LOCK), 'mapping_design_final_lock_sha256': sha256(project / E_LOCK), 'mapping_design_trial_mapping_sha256': sha256(project / E_MAP), 'method_snapshot_sha256': method_sha}, 'source_gate_pass': gate, 'GSR_derivative_amplitude_read': False, 'raw_signal_amplitude_read': False, 'behavioral_outcomes_read': False, 'family_assignments_read': False, 'trajectory_geometry_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'outputs': {}}
    lockhash = {}
    for k, p in paths.items():
        if k == 'contract':
            continue
        lockhash[str(p.relative_to(project))] = sha256(p)
    contract['outputs'] = lockhash
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/source_gsr_dynamic_design_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': lockhash, 'source_gate_pass': gate, 'next_phase': nextpol['selected_next_phase'], 'GSR_derivative_amplitude_read': False, 'raw_signal_amplitude_read': False, 'behavioral_outcomes_read': False, 'family_assignments_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'source_gsr_dynamic_design_author_gsr_dynamic_design_specification_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print(f"[{now()}] source_gsr_dynamic_design {('PASS' if gate else 'STOP')}: eligible={n_eligible}; mapped={numer}/{denom} ({frac:.4f}); valid_files={valid_files}", flush=True)
    print('Derivative amplitude read: False', flush=True)
    print('Next phase:', nextpol['selected_next_phase'], flush=True)
    print('', hz, flush=True)
    print('SHA-256:', sh, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())