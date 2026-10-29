# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
PHASE = 'discovery_source_gsr_materialization_qc'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
DEFAULT_DATASET = Path('<DATASET_ROOT>')
H_LOCK = 'outputs/qc/source_gsr_dynamic_design_final_output_lock_hashes.json'
H_SUM = 'outputs/qc/source_gsr_dynamic_design_author_gsr_dynamic_design_specification_summary.json'
H_DESIGN = 'analysis_specifications/S066_source_gsr_dynamic.json'
H_ROUTE = 'analysis_specifications/S065_source_gsr_dynamic.json'
H_SOURCE = 'outputs/qc/source_gsr_dynamic_design_author_gsr_source_schema_audit.csv'
H_MAP = 'outputs/tables/source_gsr_dynamic_design_author_gsr_trial_row_mapping.csv'
H_ELIG = 'outputs/tables/source_gsr_dynamic_design_discovery_author_gsr_eligibility.csv'
PRIMARY_CHANNELS = ['SCR', 'SCL']
SENSITIVITY_CHANNELS = ['iSCR']
ALL_CHANNELS = PRIMARY_CHANNELS + SENSITIVITY_CHANNELS
N_BINS = 10
MIN_PRIMARY_FINITE_CELLS_PER_TRIAL = 2
MIN_PRIMARY_BIN_GLOBAL_FINITE_FRACTION = 0.5
MIN_TRIALS_PER_PARTICIPANT = 16
MIN_TRIALS_PER_CONTEXT = 4
MIN_ELIGIBLE_PARTICIPANTS = 18

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

def writecsv(path, df_or_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(path, index=False)
    else:
        pd.DataFrame(df_or_rows).to_csv(path, index=False)

def boolish(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes', 'y'}

def verify_lock(project: Path, rel: str, phase: str):
    p = project / rel
    if not p.is_file():
        raise RuntimeError(f'Missing upstream lock: {p}')
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

def feature_col(ch, b):
    return f'{ch.lower()}_bin{b:02d}'

def parse_indices(x):
    vals = [v for v in str(x).split('|') if v != '']
    out = [int(v) for v in vals]
    if len(out) != N_BINS or len(set(out)) != N_BINS:
        raise RuntimeError(f'Expected {N_BINS} unique source row indices, got {out}')
    return out

def exact_source_audit_lookup(source_audit: pd.DataFrame):
    d = {}
    for _, r in source_audit.iterrows():
        p = str(r['source_path'])
        if boolish(r['source_valid_for_dynamic_route']):
            d[p] = {'sha256': str(r['source_sha256']), 'subject_id': str(r['subject_id']), 'context': str(r['context'])}
    return d

def materialize_one_source(path: Path):
    cols = ['Trial', 'SCR', 'SCL', 'iSCR']
    d = pd.read_csv(path, usecols=cols)
    if len(d) == 0:
        raise RuntimeError(f'Empty source: {path}')
    for c in ['SCR', 'SCL', 'iSCR']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['Trial'] = d['Trial'].astype(str)
    return d

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x.csv'
        rows = []
        for k in range(21):
            trial = f'stim{k:02d}'
            for b in range(10):
                rows.append({'Trial': trial, 'SCR': k + b / 10, 'SCL': 100 + k + b / 10, 'iSCR': b})
        pd.DataFrame(rows).to_csv(p, index=False)
        d = materialize_one_source(p)
        idx = list(range(20, 30))
        q = d.iloc[idx]
        if len(q) != 10 or not np.isfinite(q['SCR']).all() or (not np.isfinite(q['SCL']).all()):
            raise RuntimeError('materialization self-test failed')
        vals = {}
        for ch in ALL_CHANNELS:
            for b, v in enumerate(q[ch].tolist(), 1):
                vals[feature_col(ch, b)] = float(v)
        if len(vals) != 30:
            raise RuntimeError('feature vector self-test failed')
    print('discovery_source_gsr_materialization_qc self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
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
    hlock = verify_lock(project, H_LOCK, 'source_gsr_dynamic_design')
    hsum = readj(project / H_SUM)
    hdesign = readj(project / H_DESIGN)
    hroute = readj(project / H_ROUTE)
    if hlock.get('source_gate_pass') is not True or hlock.get('next_phase') != 'discovery_source_gsr_materialization_qc':
        raise RuntimeError('source_gsr_dynamic_design final lock does not authorize discovery_source_gsr_materialization_qc')
    if hsum.get('gates', {}).get('source_gate_pass') is not True:
        raise RuntimeError('source_gsr_dynamic_design source gate did not pass')
    if hroute.get('selected_next_phase') != 'discovery_source_gsr_materialization_qc':
        raise RuntimeError('source_gsr_dynamic_design next-route policy mismatch')
    if hdesign.get('scope') != 'Discovery-only author-provided stimulus-period dynamic GSR derivatives; rating-period physiology remains untested.':
        raise RuntimeError('Unexpected source_gsr_dynamic_design design scope')
    if hdesign.get('author_derivative_interpretation', {}).get('segments_per_trial') != 10:
        raise RuntimeError('Unexpected prespecified segment count')
    if hdesign.get('author_derivative_interpretation', {}).get('primary_channels') != ['SCR', 'SCL']:
        raise RuntimeError('Unexpected prespecified primary channels')
    source_audit = pd.read_csv(project / H_SOURCE)
    mapping = pd.read_csv(project / H_MAP)
    helig = pd.read_csv(project / H_ELIG)
    for c in ['mapped_to_exact_10x3s_author_gsr', 'stimulus_mapping_available']:
        mapping[c] = mapping[c].map(boolish)
    helig['min16_and_both_contexts_pass'] = helig['min16_and_both_contexts_pass'].map(boolish)
    mm = mapping[mapping['mapped_to_exact_10x3s_author_gsr']].copy()
    if len(mm) != int(hsum['counts']['mapped_exact_10x3s_author_gsr_trials']):
        raise RuntimeError('Mapped-trial count drift from source_gsr_dynamic_design')
    if mm['author_gsr_source_path'].isna().any():
        raise RuntimeError('Mapped row lacks source path')
    if mm.duplicated(['subject_id', 'trial_number']).any():
        raise RuntimeError('Duplicate fixed trial keys in source_gsr_dynamic_design mapping')
    valid_lookup = exact_source_audit_lookup(source_audit)
    unique_sources = sorted(mm['author_gsr_source_path'].astype(str).unique())
    discovery_subjects = set(mapping['subject_id'].astype(str))
    for pstr in unique_sources:
        rec = valid_lookup.get(pstr)
        if rec is None:
            raise RuntimeError(f'Mapped source not valid in source_gsr_dynamic_design audit: {pstr}')
        if rec['subject_id'] not in discovery_subjects:
            raise RuntimeError(f'Held-out/non-Discovery source requested: {pstr}')
        p = Path(pstr)
        if not p.is_file():
            raise RuntimeError(f'Source missing: {p}')
        if sha256(p) != rec['sha256']:
            raise RuntimeError(f'Source SHA drift: {p}')
    impl = {'phase': PHASE, 'generated_at': now(), 'role': 'Discovery-only materialization/QC of prespecified author-provided 10x3-s GSR dynamic values; no family/geometry/outcome inference.', 'source_rows': 'Only source_gsr_dynamic_design rows with mapped_to_exact_10x3s_author_gsr=True.', 'feature_materialization': {'primary': ['scr_bin01..scr_bin10', 'scl_bin01..scl_bin10'], 'sensitivity': ['iscr_bin01..iscr_bin10'], 'bin_order': 'exact released row order within the source_gsr_dynamic_design-fixed stimulus row block', 'scaling_or_imputation_performed_here': False}, 'pre_inference_qc_rules': {'trial_primary_viable': 'at least one finite SCR bin AND at least one finite SCL bin', 'global_primary_bin_viable': f'finite fraction >= {MIN_PRIMARY_BIN_GLOBAL_FINITE_FRACTION:.2f} for each of 20 primary bins', 'participant_viable': f'>={MIN_TRIALS_PER_PARTICIPANT} viable mapped trials and >={MIN_TRIALS_PER_CONTEXT} viable trials/context', 'cohort_viable': f'>={MIN_ELIGIBLE_PARTICIPANTS} participants pass participant viability'}, 'boundaries': {'family_assignments_read': False, 'trajectory_geometry_read': False, 'behavioral_outcomes_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False, 'rating_period_physio_tested': False}}
    impl_path = project / 'config/discovery_source_gsr_materialization_qc_pre_amplitude_materialization_contract.json'
    writej(impl_path, impl)
    cache = {}
    opened_sources = []
    for pstr in unique_sources:
        p = Path(pstr)
        cache[pstr] = materialize_one_source(p)
        opened_sources.append({'source_path': pstr, 'source_sha256': sha256(p), 'subject_id': valid_lookup[pstr]['subject_id'], 'context': valid_lookup[pstr]['context'], 'rows': int(len(cache[pstr])), 'columns_materialized': 'Trial|SCR|SCL|iSCR'})
    rows = []
    for _, r in mm.sort_values(['subject_id', 'trial_number']).iterrows():
        pstr = str(r['author_gsr_source_path'])
        d = cache[pstr]
        idx = parse_indices(r['source_row_indices'])
        if min(idx) < 0 or max(idx) >= len(d):
            raise RuntimeError(f'Source row out of range for {r.subject_id} trial {r.trial_number}')
        q = d.iloc[idx]
        if len(q) != N_BINS:
            raise RuntimeError('Unexpected row block length')
        vals = {}
        finite_by = {}
        for ch in ALL_CHANNELS:
            vv = pd.to_numeric(q[ch], errors='coerce').to_numpy(dtype=float)
            finite_by[ch] = int(np.isfinite(vv).sum())
            for b, v in enumerate(vv, 1):
                vals[feature_col(ch, b)] = float(v) if np.isfinite(v) else math.nan
        viable = finite_by['SCR'] >= 1 and finite_by['SCL'] >= 1
        complete_primary = finite_by['SCR'] == N_BINS and finite_by['SCL'] == N_BINS
        complete_sens = finite_by['iSCR'] == N_BINS
        row = {'subject_id': str(r['subject_id']), 'trial_number': int(r['trial_number']), 'context': str(r['context']), 'acquisition_fraction': float(r['acquisition_fraction']) if pd.notna(r['acquisition_fraction']) else math.nan, 'stimulus_id': str(r['stimulus_id']), 'source_path': pstr, 'source_sha256': str(r['author_gsr_source_sha256']), 'source_row_indices': str(r['source_row_indices']), 'scr_finite_bins': finite_by['SCR'], 'scl_finite_bins': finite_by['SCL'], 'iscr_finite_bins': finite_by['iSCR'], 'primary_trial_viable': viable, 'primary_complete_case': complete_primary, 'iscr_complete_case': complete_sens, **vals}
        rows.append(row)
    fdf = pd.DataFrame(rows)
    if len(fdf) != len(mm):
        raise RuntimeError('Materialized trial count mismatch')
    primary_cols = [feature_col(ch, b) for ch in PRIMARY_CHANNELS for b in range(1, N_BINS + 1)]
    sens_cols = [feature_col('iSCR', b) for b in range(1, N_BINS + 1)]
    fq = []
    for c in primary_cols + sens_cols:
        x = pd.to_numeric(fdf[c], errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(x)
        vals = x[finite]
        fq.append({'feature': c, 'channel': 'SCR' if c.startswith('scr_') else 'SCL' if c.startswith('scl_') else 'iSCR', 'role': 'primary' if c in primary_cols else 'sensitivity', 'n_trials': len(x), 'n_finite': int(finite.sum()), 'finite_fraction': float(finite.mean()) if len(x) else 0.0, 'n_nonfinite': int((~finite).sum()), 'min': float(np.min(vals)) if len(vals) else math.nan, 'median': float(np.median(vals)) if len(vals) else math.nan, 'max': float(np.max(vals)) if len(vals) else math.nan, 'global_primary_bin_viable': bool(c not in primary_cols or finite.mean() >= MIN_PRIMARY_BIN_GLOBAL_FINITE_FRACTION)})
    fqdf = pd.DataFrame(fq)
    part = []
    for sid in sorted(fdf['subject_id'].unique()):
        q = fdf[fdf.subject_id == sid]
        v = q[q.primary_trial_viable]
        ni = int((v.context == 'imagery').sum())
        nv = int((v.context == 'video').sum())
        n = int(len(v))
        passv = bool(n >= MIN_TRIALS_PER_PARTICIPANT and ni >= MIN_TRIALS_PER_CONTEXT and (nv >= MIN_TRIALS_PER_CONTEXT))
        part.append({'subject_id': sid, 'materialized_mapped_trials': int(len(q)), 'primary_viable_trials': n, 'primary_complete_case_trials': int(q.primary_complete_case.sum()), 'iscr_complete_case_trials': int(q.iscr_complete_case.sum()), 'viable_imagery_trials': ni, 'viable_video_trials': nv, 'discovery_source_gsr_materialization_qc_min16_both_contexts_pass': passv})
    pdf = pd.DataFrame(part)
    primary_bins_pass = bool(fqdf.loc[fqdf.role == 'primary', 'global_primary_bin_viable'].all())
    n_part = int(pdf.discovery_source_gsr_materialization_qc_min16_both_contexts_pass.sum())
    gate = bool(primary_bins_pass and n_part >= MIN_ELIGIBLE_PARTICIPANTS)
    paths = {'impl': impl_path, 'opened': project / 'outputs/qc/discovery_source_gsr_materialization_qc_opened_discovery_gsr_sources.csv', 'features': project / 'outputs/tables/discovery_source_gsr_materialization_qc_discovery_author_gsr_dynamic_values.csv', 'feature_qc': project / 'outputs/qc/discovery_source_gsr_materialization_qc_author_gsr_feature_qc.csv', 'elig': project / 'outputs/tables/discovery_source_gsr_materialization_qc_discovery_author_gsr_amplitude_qc_eligibility.csv', 'summary': project / 'outputs/qc/discovery_source_gsr_materialization_qc_discovery_author_gsr_materialization_qc_summary.json', 'next': project / 'analysis_specifications/S026_discovery_source_gsr_materialization_qc.json', 'report': project / 'outputs/reports/discovery_source_gsr_materialization_qc_discovery_author_gsr_materialization_qc_report.md', 'contract': project / 'config/discovery_source_gsr_materialization_qc_discovery_author_gsr_materialization_qc_contract.json'}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    writecsv(paths['opened'], opened_sources)
    writecsv(paths['features'], fdf)
    writecsv(paths['feature_qc'], fqdf)
    writecsv(paths['elig'], pdf)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'materialization_gate_passed' if gate else 'materialization_gate_failed', 'counts': {'discovery_sources_opened': len(opened_sources), 'mapped_trials_materialized': int(len(fdf)), 'primary_viable_trials': int(fdf.primary_trial_viable.sum()), 'primary_complete_case_trials': int(fdf.primary_complete_case.sum()), 'iscr_complete_case_trials': int(fdf.iscr_complete_case.sum()), 'participants_with_materialized_trials': int(fdf.subject_id.nunique()), 'eligible_participants_after_amplitude_qc': n_part, 'primary_feature_bins': len(primary_cols), 'sensitivity_feature_bins': len(sens_cols)}, 'fractions': {'primary_viable_trial_fraction': float(fdf.primary_trial_viable.mean()) if len(fdf) else 0.0, 'primary_complete_case_fraction': float(fdf.primary_complete_case.mean()) if len(fdf) else 0.0, 'iscr_complete_case_fraction': float(fdf.iscr_complete_case.mean()) if len(fdf) else 0.0}, 'gates': {'all_20_primary_bins_global_finite_fraction_ge_0p50': primary_bins_pass, 'eligible_participants_required': MIN_ELIGIBLE_PARTICIPANTS, 'eligible_participants_observed': n_part, 'materialization_gate_pass': gate}, 'boundaries': {'Discovery_GSR_derivative_values_opened': True, 'raw_GSR_amplitude_read': False, 'raw_PPG_amplitude_read': False, 'family_assignments_read': False, 'trajectory_geometry_read': False, 'behavioral_outcomes_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False, 'rating_period_physio_tested': False}}
    writej(paths['summary'], summary)
    nextp = {'phase': PHASE, 'generated_at': now(), 'selected_next_phase': 'discovery_dynamic_gsr_inference' if gate else 'dynamic_gsr_synthesis', 'if_materialization_gate_passes': {'next_phase': 'discovery_dynamic_gsr_inference', 'role': 'Discovery-only prespecified DG1-DG3 inference using discovery_source_gsr_materialization_qc materialized GSR dynamics plus fixed cross_person_alignment_fit/mapping_design family/geometry, with Holm and prespecified robustness.'}, 'if_materialization_gate_fails': {'next_phase': 'dynamic_gsr_synthesis', 'role': 'Close author-GSR stimulus-dynamic fallback without DG1-DG3 inference.'}, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'rating_period_physio_tested': False}
    writej(paths['next'], nextp)
    report = f"# discovery_source_gsr_materialization_qc — Discovery author-GSR dynamic materialization/QC\n\nStatus: **{('PASS' if gate else 'FAIL')}**.\n\nThis is the first phase that reads numeric values from the dataset authors' released Discovery GSR derivative files. It materializes the prespecified 20D primary profile (10 SCR + 10 SCL ordered bins) and the 10D iSCR sensitivity profile.\n\n## Materialization\n- Discovery source files opened: **{len(opened_sources)}**\n- Exact source_gsr_dynamic_design-mapped trials materialized: **{len(fdf)}**\n- Primary-viable trials: **{int(fdf.primary_trial_viable.sum())}**\n- Primary complete-case trials: **{int(fdf.primary_complete_case.sum())}**\n- Eligible participants after amplitude QC: **{n_part}**\n- All 20 primary bins have >=50% finite values: **{primary_bins_pass}**\n- discovery_dynamic_gsr_inference inference permitted: **{gate}**\n\n## Hard boundary\nNo EEG-family assignments, continuous EEG trajectory geometry, behavioral outcomes, Valence/Liking, emotion category, Validation GSR derivative values, or Confirmation GSR derivative values were read in discovery_source_gsr_materialization_qc.\n\nNo imputation, scaling, winsorization, model fitting, permutation test, bootstrap, or inferential feature selection was performed.\n\nThese 10×3-s author GSR derivatives are interpreted only as stimulus-period dynamics. They do not test rating-period physiology."
    paths['report'].write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'inputs': {'source_gsr_dynamic_design_final_lock_sha256': sha256(project / H_LOCK), 'source_gsr_dynamic_design_sha256': sha256(project / H_DESIGN), 'source_gsr_dynamic_design_mapping_sha256': sha256(project / H_MAP), 'source_gsr_dynamic_design_source_audit_sha256': sha256(project / H_SOURCE)}, 'materialization_gate_pass': gate, 'Discovery_GSR_derivative_values_opened': True, 'raw_signal_amplitude_read': False, 'family_assignments_read': False, 'trajectory_geometry_read': False, 'behavioral_outcomes_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False, 'outputs': {}}
    lockhash = {}
    for k, p in paths.items():
        if k == 'contract':
            continue
        lockhash[str(p.relative_to(project))] = sha256(p)
    contract['outputs'] = dict(lockhash)
    writej(paths['contract'], contract)
    lockhash[str(paths['contract'].relative_to(project))] = sha256(paths['contract'])
    final = project / 'outputs/qc/discovery_source_gsr_materialization_qc_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': summary['status'], 'hashes': lockhash, 'materialization_gate_pass': gate, 'next_phase': nextp['selected_next_phase'], 'Discovery_GSR_derivative_values_opened': True, 'raw_signal_amplitude_read': False, 'family_assignments_read': False, 'trajectory_geometry_read': False, 'behavioral_outcomes_read': False, 'Validation_GSR_derivative_values_opened': False, 'Confirmation_GSR_derivative_values_opened': False, 'inference_performed': False})
    rels = list(lockhash.keys()) + [str(final.relative_to(project))]
    hz = project / '' / f'discovery_source_gsr_materialization_qc_discovery_author_gsr_materialization_qc_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    sh = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{sh}  {hz.name}', encoding='utf-8')
    print(f"[{now()}] discovery_source_gsr_materialization_qc {('PASS' if gate else 'STOP')}", flush=True)
    print(f'Sources opened: {len(opened_sources)}', flush=True)
    print(f'Trials materialized: {len(fdf)}; primary viable={int(fdf.primary_trial_viable.sum())}; complete-case={int(fdf.primary_complete_case.sum())}', flush=True)
    print(f'Eligible participants after amplitude QC: {n_part}', flush=True)
    print('Family/geometry/outcome inference performed: False', flush=True)
    print('Validation/Confirmation derivative values opened: False', flush=True)
    print('Next phase:', nextp['selected_next_phase'], flush=True)
    print('', hz, flush=True)
    print('SHA-256:', sh, flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())