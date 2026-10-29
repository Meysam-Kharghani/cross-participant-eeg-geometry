# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase validation_semantic_report_execution — one-shot Validation semantic/report execution.

Executes every semantic_report_execution_design-prespecified A1/A2/B2/B3 test in Validation regardless of the
Discovery discovery_semantic_report_execution results. Numerical modeling/permutation helpers are imported
from the exact fixed discovery_semantic_report_execution source. Validation target materialization obeys the
semantic_report_execution_design lineage correction: reports only from the prespecified canonical report columns,
source-audited against raw behavior TSVs; context from the authoritative validation_source_epoch_lineage
event-timing lineage; category from authoritative context + recomputed
context-specific schedule index + the exact prespecified README sequence parser.

No new EEG signal array is opened. Confirmation behavioral values remain closed."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
PHASE = 'validation_semantic_report_execution'
NEXT_PHASE = 'validation_confirmation_semantic_design'
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
EXPERIENCE_K = [2, 3, 4]
TRANSITION_K = [2, 3]
REPORTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'valence', 'familiarity', 'liking']
CATEGORIES = ['sadness', 'disgust', 'fear', 'neutral', 'joy', 'tenderness', 'inspiration']
STD_SIG = ['standardized__time_frequency_channel_median_iqr__rms_displacement', 'standardized__time_frequency_channel_median_iqr__cosine_similarity', 'standardized__morphology_channel_median_iqr__rms_displacement', 'standardized__morphology_channel_median_iqr__cosine_similarity', 'standardized__plv_channel_median_iqr__rms_displacement', 'standardized__plv_channel_median_iqr__cosine_similarity', 'standardized__waveform_dct_channel_median_iqr__rms_displacement', 'standardized__waveform_dct_channel_median_iqr__cosine_similarity', 'standardized__covariance_summary__rms_displacement', 'standardized__covariance_summary__cosine_similarity', 'standardized__plv_global__rms_displacement', 'standardized__plv_global__cosine_similarity', 'standardized__global__rms_displacement', 'standardized__global__cosine_similarity']
A1_PERMUTATIONS = 10000
MODEL_PERMUTATIONS = 1000
ALPHA = 0.05
discovery_semantic_report_execution_OUT = Path('outputs/interpretability/discovery_semantic_report_execution_discovery_semantic_report_execution')
discovery_semantic_report_execution_LOCK = discovery_semantic_report_execution_OUT / 'qc/discovery_semantic_report_execution_final_output_lock_hashes.json'
discovery_semantic_report_execution_LOCK_SHA256 = '4ace7a17facf24471572a6a05adc2e6ce55082e1f0890b882c938150f2ed4b59'
discovery_semantic_report_execution_NEXT = Path('analysis_specifications/S025_discovery_semantic_report_validation.json')
discovery_semantic_report_execution_NEXT_SHA256 = 'e0d3869a0098545390c7a94918f11238a73a1ab505fe668d46c6b1a5e002abf1'
discovery_semantic_report_execution_NUMERIC_specification = discovery_semantic_report_execution_OUT / 'qc/discovery_semantic_report_execution_numeric_implementation_specification.json'
discovery_semantic_report_execution_NUMERIC_specification_SHA256 = 'f7747503f9e03b80560d20481e76be7301a9b1f703a986f9d244d7f308bd9d57'
discovery_semantic_report_execution_SOURCE = Path('src/analysis_pipeline/eeg/discovery_semantic_report_execution/discovery_semantic_report_execution.py')
discovery_semantic_report_execution_SOURCE_SHA256 = 'caa62ecc3d5dfe825cd166d8e68078fd1fe5c185c81eb1caef8701f1cacbf29e'
semantic_report_execution_design_PROGRAM = Path('analysis_specifications/S063_semantic_report.json')
semantic_report_execution_design_PROGRAM_SHA256 = 'b9b6800c4ee6bf22033b81c41031d5d805dc57c5020b6eda21d6d931cc5b28eb'
semantic_report_execution_design_RAW_INVENTORY = Path('outputs/interpretability/semantic_report_execution_design_semantic_report_execution_design_specification/tables/semantic_report_execution_design_heldout_raw_source_inventory.csv')
semantic_report_execution_design_RAW_INVENTORY_SHA256 = 'effa5dd59a8d32e5a335db18ebb33df0790f2fce282b23d622e9d3e91c57d704'
CANONICAL = Path('data/canonical_behavioral_table.csv')
CANONICAL_SHA256 = 'e31f96f918e806d466fd2bccc85b94d5c9dc96002bf97a97a442e8059a0365ed'
VAL_TIMING = Path('outputs/tables/validation_source_epoch_lineage_validation_authoritative_trial_timing_manifest.csv')
VAL_TIMING_SHA256 = '08a5f19fefb0b7371592f7d9ab796774168a05b64295bfa8019a8cdf1338d145'
README_REL = Path('README.md')
README_SHA256 = '379cfc97fce08b4c54d3dc26e95c35d504ca130666bbcf02eff59638f0ff7e3a'
README_SEQUENCE_CORE = Path('src/analysis_pipeline/data_preparation/event_grammar_reconstruction.py')
README_SEQUENCE_CORE_SHA256 = '39b690bdbd692cd5a8df6f370e7ad96ddff135ba9f96e6817c6e7e77c23474ac'
EXP_ASSIGN = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_family_assignments_all_k.csv')
EXP_ASSIGN_SHA256 = 'c83ad33c6e1485de814b6653087c5f7b929674f46c0edff8247b540fa0813fe1'
EXP_SIG = Path('outputs/interpretability/validation_stimulus_experience_multiresolution_transfer_validation_stimulus_experience_multiresolution_transfer/tables/validation_stimulus_experience_multiresolution_transfer_validation_experience_signatures.csv')
EXP_SIG_SHA256 = '212539287c54395ea8474f3a24a6d4c99d6edc99423b27afa836c60a698ca298'
TR_ASSIGN = Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_transition_family_assignments_all_k.csv')
TR_ASSIGN_SHA256 = '5915daa79e6a56d685ad9d418ea104efafa56f6c3800382537c308683c285c53'
TR_SIG = Path('outputs/interpretability/validation_report_transition_multiresolution_transfer_validation_report_transition_multiresolution_transfer/tables/validation_report_transition_multiresolution_transfer_validation_report_transition_signatures.csv')
TR_SIG_SHA256 = 'd81bd9749045a9126a57d067d0164b63020d2e1aca2acbab1d2ca832c7e2c067'
OUT = Path('outputs/interpretability/validation_semantic_report_execution_validation_semantic_report_execution')
CACHE = Path('outputs/large/validation_semantic_report_execution_validation_semantic_report_execution_cache_v1')
NEXT_CONFIG = Path('analysis_specifications/S088_validation_semantic_report_confirmation.json')

def now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def verify(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(Path(path))
    if got != expected:
        raise RuntimeError(f'{label} SHA drift: expected {expected}, got {got}')
    return got

def readj(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '', encoding='utf-8')
    os.replace(tmp, path)

def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def bval(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def norm_subject(x: Any) -> str:
    m = re.fullmatch('(?:sub[-_ ]?)?(\\d+)', str(x).strip(), re.I)
    return f'sub-{int(m.group(1)):02d}' if m else str(x).strip()

def norm_name(x: Any) -> str:
    return re.sub('[^a-z0-9]+', '', str(x).strip().lower())

def import_exact(path: Path, expected: str, name: str):
    verify(path, expected, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import {name}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod

def verify_discovery_semantic_report_execution(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify(project / discovery_semantic_report_execution_LOCK, discovery_semantic_report_execution_LOCK_SHA256, 'discovery_semantic_report_execution final lock')
    lock = readj(project / discovery_semantic_report_execution_LOCK)
    if lock.get('phase') != 'discovery_semantic_report_execution' or lock.get('next_phase') != PHASE:
        raise RuntimeError('discovery_semantic_report_execution authorization drift')
    if lock.get('next_phase_runs_all_prespecified_tests_regardless_Discovery_p') is not True:
        raise RuntimeError('discovery_semantic_report_execution lockbox rule drift')
    if lock.get('Validation_behavioral_values_read') is not False:
        raise RuntimeError('Validation behavior was already marked open in discovery_semantic_report_execution')
    if lock.get('Confirmation_behavioral_values_read') is not False:
        raise RuntimeError('Confirmation behavior boundary drift')
    if lock.get('new_EEG_signal_values_opened') is not False:
        raise RuntimeError('discovery_semantic_report_execution EEG boundary drift')
    for rel, expected in lock.get('hashes', {}).items():
        verify(project / rel, expected, f'discovery_semantic_report_execution fixed child {rel}')
    verify(project / discovery_semantic_report_execution_NEXT, discovery_semantic_report_execution_NEXT_SHA256, 'discovery_semantic_report_execution→validation_semantic_report_execution policy')
    nxt = readj(project / discovery_semantic_report_execution_NEXT)
    if nxt.get('Validation_behavior_opening_permitted_in_validation_semantic_report_execution') is not True:
        raise RuntimeError('discovery_semantic_report_execution does not permit Validation behavior opening')
    if nxt.get('Confirmation_behavior_opening_permitted_in_validation_semantic_report_execution') is not False:
        raise RuntimeError('discovery_semantic_report_execution unexpectedly permits Confirmation behavior')
    if nxt.get('new_EEG_signal_opening_permitted') is not False:
        raise RuntimeError('discovery_semantic_report_execution unexpectedly permits new EEG')
    required = ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']
    if nxt.get('Validation_tests_required') != required:
        raise RuntimeError('discovery_semantic_report_execution Validation test registry drift')
    if nxt.get('Discovery_results_select_Validation_tests') is not False:
        raise RuntimeError('discovery_semantic_report_execution result-selection boundary drift')
    verify(project / discovery_semantic_report_execution_NUMERIC_specification, discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'discovery_semantic_report_execution numerical specification')
    verify(project / discovery_semantic_report_execution_SOURCE, discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution numerical source')
    verify(project / semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256, 'semantic_report_execution_design program policy')
    verify(project / semantic_report_execution_design_RAW_INVENTORY, semantic_report_execution_design_RAW_INVENTORY_SHA256, 'semantic_report_execution_design held-out source inventory')
    return (lock, nxt)

def resolve_report_columns(columns: Iterable[str]) -> dict[str, str]:
    cols = list(columns)
    by_norm = {norm_name(c): c for c in cols}
    out: dict[str, str] = {}
    for i, report in enumerate(REPORTS, start=1):
        candidates = [norm_name(report), f'score{i}', f'score0{i}', f'rating{i}', f'rating0{i}']
        hits = [by_norm[c] for c in candidates if c in by_norm]
        hits = list(dict.fromkeys(hits))
        if len(hits) != 1:
            raise RuntimeError(f'Could not resolve exactly one raw report column for {report}; candidates={hits}; columns={cols}')
        out[report] = hits[0]
    return out

def resolve_trial_column(columns: Iterable[str]) -> str | None:
    by_norm = {norm_name(c): c for c in columns}
    for key in ['trialnumberglobal', 'trialnumber', 'globaltrial', 'trial', 'trialindex', 'trialid']:
        if key in by_norm:
            return by_norm[key]
    return None

def compare_report_values(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    if len(a) != len(b):
        raise RuntimeError(f'{label}: row-count mismatch {len(a)} vs {len(b)}')
    for r in REPORTS:
        x = pd.to_numeric(a[r], errors='coerce').to_numpy(float)
        y = pd.to_numeric(b[r], errors='coerce').to_numpy(float)
        same = np.isnan(x) & np.isnan(y) | np.isclose(x, y, rtol=0.0, atol=1e-12, equal_nan=True)
        if not bool(np.all(same)):
            bad = np.flatnonzero(~same)[:5].tolist()
            raise RuntimeError(f'{label}: raw/canonical report mismatch for {r} at rows {bad}')

def audit_raw_reports(dataset: Path, canonical_reports: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, can in canonical_reports.groupby('subject_id', sort=True):
        can = can.sort_values('trial_number').reset_index(drop=True)
        inv = inventory[(inventory.split == 'validation') & (inventory.subject_id == subject) & (inventory.role == 'raw_behavior_tsv')]
        if len(inv) != 1:
            raise RuntimeError(f'{subject}: semantic_report_execution_design raw behavior inventory row count {len(inv)}')
        p = dataset / subject / 'beh' / f'{subject}_task-emotion_beh.tsv'
        expected = str(inv.iloc[0].sha256)
        verify(p, expected, f'{subject} raw behavior TSV')
        header = pd.read_csv(p, sep='\t', nrows=0)
        report_map = resolve_report_columns(header.columns)
        trial_col = resolve_trial_column(header.columns)
        usecols = list(dict.fromkeys(([trial_col] if trial_col else []) + list(report_map.values())))
        raw = pd.read_csv(p, sep='\t', usecols=usecols, low_memory=False)
        raw = raw.rename(columns={v: k for k, v in report_map.items()})
        method = ''
        if trial_col is not None:
            raw['trial_number'] = pd.to_numeric(raw[trial_col], errors='coerce')
            usable = raw['trial_number'].notna() & ~raw['trial_number'].duplicated(keep=False)
            rr = raw.loc[usable, ['trial_number'] + REPORTS].copy()
            rr['trial_number'] = rr['trial_number'].astype(int)
            rr['_raw_present'] = True
            joined = can[['trial_number'] + REPORTS].merge(rr, on='trial_number', how='left', suffixes=('_canonical', '_raw'), validate='one_to_one')
            if joined['_raw_present'].fillna(False).all():
                a = joined[[f'{r}_canonical' for r in REPORTS]].rename(columns={f'{r}_canonical': r for r in REPORTS})
                b = joined[[f'{r}_raw' for r in REPORTS]].rename(columns={f'{r}_raw': r for r in REPORTS})
                compare_report_values(a, b, f'{subject} keyed raw report audit')
                method = f'raw_column_{trial_col}_keyed'
            else:
                trial_col = None
        if trial_col is None:
            if len(raw) == len(can):
                rr = raw[REPORTS].reset_index(drop=True)
                compare_report_values(can[REPORTS].reset_index(drop=True), rr, f'{subject} ordered-retained raw report audit')
                method = 'ordered_retained_roster_alignment'
            elif int(can.trial_number.max()) <= len(raw):
                idx = can.trial_number.astype(int).to_numpy() - 1
                rr = raw.iloc[idx][REPORTS].reset_index(drop=True)
                compare_report_values(can[REPORTS].reset_index(drop=True), rr, f'{subject} raw-row-index report audit')
                method = 'one_based_raw_row_index_equals_trial_number'
            else:
                raise RuntimeError(f'{subject}: cannot source-audit canonical reports against raw behavior; raw rows={len(raw)}, canonical rows={len(can)}')
        rows.append({'subject_id': subject, 'canonical_trials': int(len(can)), 'raw_behavior_rows': int(len(raw)), 'audit_method': method, 'raw_behavior_sha256': expected, 'all_ten_reports_match': True})
    return pd.DataFrame(rows)

def materialize_validation_targets(dataset: Path, project: Path, seqcore) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify(project / CANONICAL, CANONICAL_SHA256, 'prespecified canonical report source')
    verify(project / VAL_TIMING, VAL_TIMING_SHA256, 'Validation authoritative timing lineage')
    verify(dataset / README_REL, README_SHA256, 'dataset README')
    verify(project / README_SEQUENCE_CORE, README_SEQUENCE_CORE_SHA256, 'README sequence parser core')
    inv = pd.read_csv(project / semantic_report_execution_design_RAW_INVENTORY, low_memory=False)
    inv['subject_id'] = inv.subject_id.map(norm_subject)
    inv['split'] = inv.split.astype(str).str.lower()
    vinv = inv[inv.split.eq('validation')].copy()
    if len(vinv) != 24 or vinv.subject_id.nunique() != 12:
        raise RuntimeError(f'semantic_report_execution_design Validation raw-source inventory drift: rows={len(vinv)}, subjects={vinv.subject_id.nunique()}')
    h = pd.read_csv(project / CANONICAL, nrows=0)
    required = {'subject_id', 'trial_number_global', *REPORTS}
    missing = required - set(h.columns)
    if missing:
        raise RuntimeError(f'Canonical report source missing required columns: {sorted(missing)}')
    can = pd.read_csv(project / CANONICAL, usecols=['subject_id', 'trial_number_global'] + REPORTS, low_memory=False)
    can['subject_id'] = can.subject_id.map(norm_subject)
    can['trial_number'] = pd.to_numeric(can.trial_number_global, errors='raise').astype(int)
    th = pd.read_csv(project / VAL_TIMING, nrows=0)
    tneed = {'raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'subject_acquisition_order_1based'}
    if not tneed.issubset(set(th.columns)):
        raise RuntimeError(f'Validation timing manifest schema drift; missing {sorted(tneed - set(th.columns))}')
    timing = pd.read_csv(project / VAL_TIMING, usecols=sorted(tneed), low_memory=False)
    timing['subject_id'] = timing.subject_id.map(norm_subject)
    timing['raw_behavior_row_id'] = timing.raw_behavior_row_id.astype(str)
    timing['trial_number'] = pd.to_numeric(timing.trial_number, errors='raise').astype(int)
    timing['context'] = timing.context.map(seqcore.normalize_context)
    timing['subject_acquisition_order_1based'] = pd.to_numeric(timing.subject_acquisition_order_1based, errors='raise').astype(int)
    if len(timing) != 494 or timing.subject_id.nunique() != 12:
        raise RuntimeError(f'Validation authoritative roster drift: n={len(timing)}, subjects={timing.subject_id.nunique()}')
    if timing.duplicated(['subject_id', 'trial_number']).any() or timing.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Validation timing key duplication')
    subjects = sorted(timing.subject_id.unique())
    can = can[can.subject_id.isin(subjects)].copy()
    if len(can) != 494 or can.duplicated(['subject_id', 'trial_number']).any():
        raise RuntimeError(f'Validation canonical report roster drift: n={len(can)}')
    canonical_reports = can[['subject_id', 'trial_number'] + REPORTS].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    raw_audit = audit_raw_reports(dataset, canonical_reports, vinv)
    seqs = seqcore.parse_readme_sequences(dataset / README_REL)
    seqman = seqcore.build_readme_manifest(seqs)
    seqman['subject_id'] = seqman.subject_id.map(norm_subject)
    seqman['context'] = seqman.context.map(seqcore.normalize_context)
    seqman = seqman.rename(columns={'available_sequence_index': 'recomputed_context_trial_index'})
    if not set(subjects).issubset(set(seqman.subject_id)):
        raise RuntimeError(f'README parser missing Validation subjects: {sorted(set(subjects) - set(seqman.subject_id))}')
    md_rows = []
    md_audit_rows = []
    for subject in subjects:
        ir = vinv[(vinv.subject_id == subject) & (vinv.role == 'safe_metadata_schedule')]
        if len(ir) != 1:
            raise RuntimeError(f'{subject}: semantic_report_execution_design metadata inventory row count {len(ir)}')
        mp = dataset / 'derivatives' / 'metadata' / f'{subject}_trials_with_context.csv'
        expected = str(ir.iloc[0].sha256)
        verify(mp, expected, f'{subject} safe metadata schedule')
        mh = pd.read_csv(mp, nrows=0)
        need = {'trial_number', 'context'}
        if not need.issubset(set(mh.columns)):
            raise RuntimeError(f'{subject}: metadata missing {sorted(need - set(mh.columns))}')
        optional = [c for c in ['global_trial', 'context_trial_idx'] if c in mh.columns]
        m = pd.read_csv(mp, usecols=['trial_number', 'context'] + optional, low_memory=False)
        m['trial_number'] = pd.to_numeric(m.trial_number, errors='raise').astype(int)
        m['context'] = m.context.map(seqcore.normalize_context)
        if m.duplicated('trial_number').any():
            raise RuntimeError(f'{subject}: duplicate metadata trial_number')
        if not m.context.isin(['imagery', 'video']).all():
            raise RuntimeError(f'{subject}: unrecognized metadata context')
        m = m.sort_values('trial_number').reset_index(drop=True)
        m['recomputed_context_trial_index'] = m.groupby('context').cumcount() + 1
        m['subject_id'] = subject
        md_rows.append(m[['subject_id', 'trial_number', 'context', 'recomputed_context_trial_index']])
        md_audit_rows.append({'subject_id': subject, 'metadata_rows': int(len(m)), 'imagery_rows': int((m.context == 'imagery').sum()), 'video_rows': int((m.context == 'video').sum()), 'metadata_sha256': expected})
    metadata = pd.concat(md_rows, ignore_index=True)
    max_order = timing.groupby('subject_id')['subject_acquisition_order_1based'].transform('max').astype(float)
    timing['acquisition_fraction'] = timing.subject_acquisition_order_1based.astype(float) / max_order
    targets = timing.merge(canonical_reports, on=['subject_id', 'trial_number'], how='left', validate='one_to_one')
    metadata = metadata.rename(columns={'context': 'metadata_context'})
    targets = targets.merge(metadata, on=['subject_id', 'trial_number'], how='left', validate='one_to_one')
    targets['metadata_event_context_agreement'] = targets.context.astype(str) == targets.metadata_context.astype(str)
    targets['context_authoritative_for_analysis'] = targets.context.isin(['imagery', 'video']) & targets.metadata_event_context_agreement
    seqjoin = seqman[['subject_id', 'context', 'recomputed_context_trial_index', 'stimulus_id', 'emotion_category', 'sequence_source', 'available_sequence_length']].copy()
    targets = targets.merge(seqjoin, on=['subject_id', 'context', 'recomputed_context_trial_index'], how='left', validate='many_to_one')
    targets['stimulus_authoritative_for_analysis'] = targets.context_authoritative_for_analysis & targets.stimulus_id.notna() & targets.emotion_category.astype(str).isin(CATEGORIES)
    targets['report_source_attachment_exact'] = True
    targets['report_complete'] = np.isfinite(targets[REPORTS].apply(pd.to_numeric, errors='coerce').to_numpy(float)).all(axis=1)
    targets.loc[~targets.stimulus_authoritative_for_analysis, ['stimulus_id', 'emotion_category']] = np.nan
    cols = ['raw_behavior_row_id', 'subject_id', 'trial_number', 'subject_acquisition_order_1based', 'acquisition_fraction', 'context', 'metadata_context', 'metadata_event_context_agreement', 'recomputed_context_trial_index', 'context_authoritative_for_analysis', 'stimulus_id', 'emotion_category', 'sequence_source', 'available_sequence_length', 'stimulus_authoritative_for_analysis', 'report_source_attachment_exact', 'report_complete'] + REPORTS
    targets = targets[cols].sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    lineage_audit = pd.DataFrame(md_audit_rows).merge(raw_audit, on='subject_id', how='inner', validate='one_to_one')
    subject_context = targets.groupby('subject_id', as_index=False).agg(authoritative_trials=('trial_number', 'size'), context_agree_trials=('metadata_event_context_agreement', 'sum'), context_analyzable_trials=('context_authoritative_for_analysis', 'sum'), stimulus_analyzable_trials=('stimulus_authoritative_for_analysis', 'sum'), complete_report_trials=('report_complete', 'sum'))
    lineage_audit = lineage_audit.merge(subject_context, on='subject_id', how='inner', validate='one_to_one')
    return (targets, raw_audit, lineage_audit)

def load_validation_structural(project: Path) -> dict[str, pd.DataFrame]:
    refs = [(EXP_ASSIGN, EXP_ASSIGN_SHA256, 'Validation experience assignments'), (EXP_SIG, EXP_SIG_SHA256, 'Validation experience signature'), (TR_ASSIGN, TR_ASSIGN_SHA256, 'Validation transition assignments'), (TR_SIG, TR_SIG_SHA256, 'Validation transition signature')]
    for rel, h, label in refs:
        verify(project / rel, h, label)
    ea = pd.read_csv(project / EXP_ASSIGN, low_memory=False)
    es = pd.read_csv(project / EXP_SIG, low_memory=False)
    ta = pd.read_csv(project / TR_ASSIGN, low_memory=False)
    ts = pd.read_csv(project / TR_SIG, low_memory=False)
    for d in [ea, es, ta, ts]:
        d['raw_behavior_row_id'] = d.raw_behavior_row_id.astype(str)
        d['subject_id'] = d.subject_id.map(norm_subject)
    for c in STD_SIG:
        if c not in es.columns or c not in ts.columns:
            raise RuntimeError(f'Missing prespecified Validation signature coordinate: {c}')
    return {'exp_assign': ea, 'exp_sig': es, 'tr_assign': ta, 'tr_sig': ts}

def finite_rows(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.isfinite(df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)).all(axis=1)

def prepare_a1(targets: pd.DataFrame, structural: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    a = structural['exp_assign']
    a = a[a.k.astype(int).eq(k)].copy()
    if 'participant_eligible' in a.columns:
        a = a[a.participant_eligible.map(bval)]
    a = a[['raw_behavior_row_id', 'subject_id', 'family_1based']]
    d = a.merge(targets, on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    m = d.stimulus_authoritative_for_analysis.map(bval) & d.emotion_category.astype(str).isin(CATEGORIES)
    d = d[m].copy()
    if d.subject_id.nunique() < 9 or len(d) < 250:
        raise RuntimeError(f'Validation A1 K={k} unexpectedly small: n={len(d)}, subjects={d.subject_id.nunique()}')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def prepare_logistic(targets: pd.DataFrame, structural: dict[str, pd.DataFrame], role: str, k: int) -> pd.DataFrame:
    a = structural['exp_assign'] if role == 'experience' else structural['tr_assign']
    a = a[a.k.astype(int).eq(k)].copy()
    if 'participant_eligible' in a.columns:
        a = a[a.participant_eligible.map(bval)]
    a = a[['raw_behavior_row_id', 'subject_id', 'family_1based']]
    d = a.merge(targets, on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    m = d.context_authoritative_for_analysis.map(bval)
    m &= d.context.astype(str).isin(['video', 'imagery'])
    m &= np.isfinite(pd.to_numeric(d.acquisition_fraction, errors='coerce'))
    m &= finite_rows(d, REPORTS)
    d = d[m].copy()
    if d.subject_id.nunique() < 9 or len(d) < 250:
        raise RuntimeError(f'Validation {role} K={k} logistic cohort unexpectedly small: n={len(d)}, subjects={d.subject_id.nunique()}')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def prepare_b2(targets: pd.DataFrame, structural: dict[str, pd.DataFrame]) -> pd.DataFrame:
    es = structural['exp_sig'][['raw_behavior_row_id', 'subject_id'] + STD_SIG].copy()
    ts = structural['tr_sig'][['raw_behavior_row_id', 'subject_id'] + STD_SIG].copy()
    es = es.rename(columns={c: f'exp__{c}' for c in STD_SIG})
    ts = ts.rename(columns={c: f'tr__{c}' for c in STD_SIG})
    d = es.merge(ts, on=['raw_behavior_row_id', 'subject_id'], how='inner', validate='one_to_one')
    d = d.merge(targets, on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    m = d.context_authoritative_for_analysis.map(bval)
    m &= d.context.astype(str).isin(['video', 'imagery'])
    m &= np.isfinite(pd.to_numeric(d.acquisition_fraction, errors='coerce'))
    m &= finite_rows(d, REPORTS)
    m &= finite_rows(d, [f'exp__{c}' for c in STD_SIG])
    m &= finite_rows(d, [f'tr__{c}' for c in STD_SIG])
    d = d[m].copy()
    if d.subject_id.nunique() < 9 or len(d) < 250:
        raise RuntimeError(f'Validation B2 cohort unexpectedly small: n={len(d)}, subjects={d.subject_id.nunique()}')
    return d.sort_values(['subject_id', 'raw_behavior_row_id']).reset_index(drop=True)

def ensure_cache(cache: Path, meta: dict[str, Any]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    mp = cache / 'cache_meta.json'
    if mp.is_file():
        if readj(mp) != meta:
            raise RuntimeError(f'validation_semantic_report_execution cache metadata mismatch at {cache}; remove cache before changed-code rerun')
    else:
        writej(mp, meta)

def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict('records') if path.is_file() else []

def save_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    d = pd.DataFrame(rows)
    if len(d) and 'permutation' in d.columns:
        d = d.sort_values('permutation').drop_duplicates('permutation', keep='last')
    atomic_csv(path, d)

def permutation_p(observed: float, null: np.ndarray) -> float:
    z = np.asarray(null, float)
    z = z[np.isfinite(z)]
    return float((1 + np.sum(z >= observed)) / (len(z) + 1))

def run_a1_one(core, df: pd.DataFrame, k: int, p: int) -> dict[str, Any]:
    fam = df.family_1based.to_numpy(int)
    subj = df.subject_id.astype(str).to_numpy()
    cat = df.emotion_category.astype(str).to_numpy()
    fp = core.permute_rows_within_subject(fam[:, None], subj, p, 'A1_family').ravel()
    tab = core.weighted_contingency(cat, fp, subj, CATEGORIES, list(range(1, k + 1)))
    m = core.contingency_metrics(tab)
    return {'permutation': p, 'NMI': m['NMI'], 'Cramers_V': m['Cramers_V']}

def self_test() -> int:
    assert PHASE == 'validation_semantic_report_execution'
    assert NEXT_PHASE == 'validation_confirmation_semantic_design'
    assert EXPERIENCE_K == [2, 3, 4]
    assert TRANSITION_K == [2, 3]
    assert len(REPORTS) == 10
    assert len(STD_SIG) == 14
    m = resolve_report_columns(['Score_1', 'Score_2', 'Score_3', 'Score_4', 'Score_5', 'Score_6', 'Score_7', 'Score_8', 'Score_9', 'Score_10', 'trial_number'])
    assert [m[r] for r in REPORTS] == [f'Score_{i}' for i in range(1, 11)]
    assert resolve_trial_column(['foo', 'trial_number', 'Joy']) == 'trial_number'
    print('validation_semantic_report_execution self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, int(os.environ.get('validation_semantic_report_execution_WORKERS', '1'))))
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    dataset = args.dataset.resolve()
    project = args.project.resolve()
    workers = max(1, int(args.workers))
    if not dataset.is_dir():
        raise RuntimeError(f'Dataset root not found: {dataset}')
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    print(f'[{now()}] validation_semantic_report_execution permutation workers: {workers}', flush=True)
    print(f'[{now()}] verifying discovery_semantic_report_execution lockbox and exact numerical implementation', flush=True)
    _, discovery_semantic_report_execution_next = verify_discovery_semantic_report_execution(project)
    core = import_exact(project / discovery_semantic_report_execution_SOURCE, discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_fixed_semantic_numeric_core')
    seqcore = import_exact(project / README_SEQUENCE_CORE, README_SEQUENCE_CORE_SHA256, 'semantic_report_execution_design_fixed_readme_sequence_core')
    if core.REPORTS != REPORTS or core.EXPERIENCE_K != EXPERIENCE_K or core.TRANSITION_K != TRANSITION_K:
        raise RuntimeError('discovery_semantic_report_execution numerical helper constant drift')
    if core.STD_SIG != STD_SIG:
        raise RuntimeError('discovery_semantic_report_execution signature coordinate drift')
    print(f'[{now()}] one-shot opening Validation target lineage and source-auditing report values', flush=True)
    targets, raw_report_audit, lineage_audit = materialize_validation_targets(dataset, project, seqcore)
    structural = load_validation_structural(project)
    out = project / OUT
    final_lock = out / 'qc/validation_semantic_report_execution_final_output_lock_hashes.json'
    if final_lock.is_file() and (not args.force):
        raise RuntimeError(f'Completed validation_semantic_report_execution output already exists: {final_lock}')
    if out.exists() and args.force:
        shutil.rmtree(out)
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    script_sha = sha256(Path(__file__).resolve())
    cache = project / CACHE
    cache_meta = {'phase': PHASE, 'source_code_sha256': script_sha, 'discovery_semantic_report_execution_source_sha256': discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_numeric_specification_sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'discovery_semantic_report_execution_next_policy_sha256': discovery_semantic_report_execution_NEXT_SHA256, 'canonical_sha256': CANONICAL_SHA256, 'validation_timing_sha256': VAL_TIMING_SHA256, 'experience_assign_sha256': EXP_ASSIGN_SHA256, 'experience_signature_sha256': EXP_SIG_SHA256, 'transition_assign_sha256': TR_ASSIGN_SHA256, 'transition_signature_sha256': TR_SIG_SHA256, 'A1_permutations': A1_PERMUTATIONS, 'model_permutations': MODEL_PERMUTATIONS}
    ensure_cache(cache, cache_meta)
    atomic_csv(out / 'tables/validation_semantic_report_execution_validation_semantic_target_manifest.csv', targets)
    atomic_csv(out / 'qc/validation_semantic_report_execution_raw_report_source_audit.csv', raw_report_audit)
    atomic_csv(out / 'qc/validation_semantic_report_execution_target_lineage_subject_audit.csv', lineage_audit)
    print(f'[{now()}] A1 category ↔ experience-family mapping/mixing', flush=True)
    a1_obs_rows, a1_cells, a1_conc, a1_nulls, a1_perm_frames = ([], [], [], [], [])
    for k in EXPERIENCE_K:
        df = prepare_a1(targets, structural, k)
        obs, cells, conc = core.a1_observed_and_mixing(df, k)
        cp = cache / f'A1_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(A1_PERMUTATIONS) if p not in done]
        chunk = 25
        for i in range(0, len(pending), chunk):
            inds = pending[i:i + chunk]

            def fn(pp):
                return run_a1_one(core, df, k, pp)
            rows.extend(core.parallel_map(fn, inds, workers))
            save_checkpoint(cp, rows)
            print(f'  A1 K={k}: {min(i + chunk, len(pending)) + len(done)}/{A1_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        if len(perm) != A1_PERMUTATIONS:
            raise RuntimeError(f'A1 K={k} permutation count drift: {len(perm)}')
        p_nmi = permutation_p(obs['NMI'], perm.NMI.to_numpy(float))
        p_v = permutation_p(obs['Cramers_V'], perm.Cramers_V.to_numpy(float))
        parent_p = min(1.0, 2.0 * min(p_nmi, p_v))
        obs.update({'test_id': f'A1_K{k}', 'k': k, 'trials': len(df), 'participants': df.subject_id.nunique(), 'p_NMI': p_nmi, 'p_Cramers_V': p_v, 'parent_p_raw': parent_p})
        a1_obs_rows.append(obs)
        a1_cells.append(cells)
        a1_conc.append(conc)
        a1_nulls.append({'test_id': f'A1_K{k}', 'permutations': A1_PERMUTATIONS, 'NMI_null_mean': float(perm.NMI.mean()), 'NMI_null_sd': float(perm.NMI.std(ddof=1)), 'Cramers_V_null_mean': float(perm.Cramers_V.mean()), 'Cramers_V_null_sd': float(perm.Cramers_V.std(ddof=1))})
        q = perm.copy()
        q.insert(0, 'k', k)
        a1_perm_frames.append(q)
    a1_metrics = pd.DataFrame(a1_obs_rows)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A1_parent_metrics.csv', a1_metrics)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A1_mixing_cells.csv', pd.concat(a1_cells, ignore_index=True))
    atomic_csv(out / 'tables/validation_semantic_report_execution_A1_concentration.csv', pd.concat(a1_conc, ignore_index=True))
    atomic_csv(out / 'tables/validation_semantic_report_execution_A1_null_summary.csv', pd.DataFrame(a1_nulls))
    atomic_csv(out / 'tables/validation_semantic_report_execution_A1_permutation_values.csv', pd.concat(a1_perm_frames, ignore_index=True))
    print(f'[{now()}] A2 joint 10-report prediction of experience family', flush=True)
    a2_rows, a2_fold_tables, a2_perm_frames = ([], [], [])
    a2_obs_by_k: dict[int, Any] = {}
    a2_df_by_k: dict[int, pd.DataFrame] = {}
    for k in EXPERIENCE_K:
        df = prepare_logistic(targets, structural, 'experience', k)
        a2_df_by_k[k] = df
        obs = core.nested_lopo_logistic(df, 'family_1based', REPORTS, salt=f'A2_K{k}')
        a2_obs_by_k[k] = obs
        ft = pd.DataFrame(obs.fold_rows)
        ft.insert(0, 'k', k)
        a2_fold_tables.append(ft)
        cp = cache / f'A2_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        for i in range(0, len(pending), 5):
            inds = pending[i:i + 5]

            def fn(pp):
                return {'permutation': pp, 'Delta_BA': core.permutation_logistic_stat(df, 'family_1based', REPORTS, pp, f'A2_K{k}', obs.baseline_ba)}
            rows.extend(core.parallel_map(fn, inds, workers))
            save_checkpoint(cp, rows)
            print(f'  A2 K={k}: {min(i + 5, len(pending)) + len(done)}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        if len(perm) != MODEL_PERMUTATIONS:
            raise RuntimeError(f'A2 K={k} permutation count drift')
        p = permutation_p(obs.statistic, perm.Delta_BA.to_numpy(float))
        a2_rows.append({'test_id': f'A2_K{k}', 'k': k, 'trials': len(df), 'participants': df.subject_id.nunique(), 'baseline_BA': obs.baseline_ba, 'full_10report_BA': obs.full_ba, 'Delta_BA': obs.statistic, 'parent_p_raw': p, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(perm.Delta_BA.mean()), 'null_sd': float(perm.Delta_BA.std(ddof=1))})
        q = perm.copy()
        q.insert(0, 'k', k)
        a2_perm_frames.append(q)
    a2_metrics = pd.DataFrame(a2_rows)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A2_parent_metrics.csv', a2_metrics)
    atomic_csv(out / 'qc/validation_semantic_report_execution_A2_outer_fold_audit.csv', pd.concat(a2_fold_tables, ignore_index=True))
    atomic_csv(out / 'tables/validation_semantic_report_execution_A2_permutation_values.csv', pd.concat(a2_perm_frames, ignore_index=True))
    parent_ids, parent_p = ([], [])
    for k in EXPERIENCE_K:
        parent_ids.append(f'A1_K{k}')
        parent_p.append(float(a1_metrics.loc[a1_metrics.k.eq(k), 'parent_p_raw'].iloc[0]))
    for k in EXPERIENCE_K:
        parent_ids.append(f'A2_K{k}')
        parent_p.append(float(a2_metrics.loc[a2_metrics.k.eq(k), 'parent_p_raw'].iloc[0]))
    a_holm = core.holm_table(parent_ids, parent_p, ALPHA)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A_family_holm.csv', a_holm)
    a2_child_rows, a2_child_perm_frames = ([], [])
    for k in EXPERIENCE_K:
        parent = a_holm[a_holm.test_id.eq(f'A2_K{k}')].iloc[0]
        df = a2_df_by_k[k]
        obs = a2_obs_by_k[k]
        if not bool(parent.reject_holm_0p05):
            for report in REPORTS:
                a2_child_rows.append({'k': k, 'report': report, 'status': 'NOT_RUN_PARENT_GATE', 'Delta_BA': np.nan, 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
            continue
        child_obs = {r: core.nested_lopo_logistic_full_ba(df, 'family_1based', [r], salt=f'A2_K{k}') - obs.baseline_ba for r in REPORTS}
        cp = cache / f'A2_K{k}_single_report_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        for i in range(0, len(pending), 2):
            inds = pending[i:i + 2]

            def fn(pp):
                return core.single_report_permutation_row(df, 'family_1based', pp, f'A2_K{k}', 'A2_reports', obs.baseline_ba)
            rows.extend(core.parallel_map(fn, inds, workers))
            save_checkpoint(cp, rows)
            print(f'  A2 K={k} child reports: {min(i + 2, len(pending)) + len(done)}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        rawp = [permutation_p(child_obs[r], perm[r].to_numpy(float)) for r in REPORTS]
        h = core.holm_table(REPORTS, rawp, ALPHA)
        for r, p0 in zip(REPORTS, rawp):
            hr = h[h.test_id.eq(r)].iloc[0]
            a2_child_rows.append({'k': k, 'report': r, 'status': 'RUN_PARENT_PASSED', 'Delta_BA': child_obs[r], 'p_raw': p0, 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
        q = perm.copy()
        q.insert(0, 'k', k)
        a2_child_perm_frames.append(q)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A2_single_report_followups.csv', pd.DataFrame(a2_child_rows))
    a2_child_perm = pd.concat(a2_child_perm_frames, ignore_index=True) if a2_child_perm_frames else pd.DataFrame(columns=['k', 'permutation'] + REPORTS)
    atomic_csv(out / 'tables/validation_semantic_report_execution_A2_single_report_permutation_values.csv', a2_child_perm)
    print(f'[{now()}] B2 incremental report information from report-transition signature', flush=True)
    b2df = prepare_b2(targets, structural)
    b2obs = core.nested_lopo_ridge_b2(b2df)
    atomic_csv(out / 'qc/validation_semantic_report_execution_B2_outer_fold_audit.csv', pd.DataFrame(b2obs.fold_rows))
    cp = cache / 'B2_permutations.csv'
    rows = load_checkpoint(cp)
    done = {int(r['permutation']) for r in rows}
    pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
    for i in range(0, len(pending), 5):
        inds = pending[i:i + 5]

        def fn(pp):
            rr = core.permutation_b2(b2df, pp)
            row = {'permutation': pp, 'Delta_R2': rr.primary_delta_r2}
            for j, r in enumerate(REPORTS):
                row[f'Delta_R2_{r}'] = float(rr.dimension_delta_r2[j])
            return row
        rows.extend(core.parallel_map(fn, inds, workers))
        save_checkpoint(cp, rows)
        print(f'  B2: {min(i + 5, len(pending)) + len(done)}/{MODEL_PERMUTATIONS}', flush=True)
    b2perm = pd.DataFrame(rows).sort_values('permutation')
    b2p = permutation_p(b2obs.primary_delta_r2, b2perm.Delta_R2.to_numpy(float))
    atomic_csv(out / 'tables/validation_semantic_report_execution_B2_permutation_values.csv', b2perm)
    b2parent = pd.DataFrame([{'test_id': 'B2', 'trials': len(b2df), 'participants': b2df.subject_id.nunique(), 'R2_M0': b2obs.m0_r2, 'R2_M1': b2obs.m1_r2, 'R2_M2': b2obs.m2_r2, 'Delta_R2_M2_minus_M1': b2obs.primary_delta_r2, 'p_raw': b2p, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(b2perm.Delta_R2.mean()), 'null_sd': float(b2perm.Delta_R2.std(ddof=1))}])
    atomic_csv(out / 'tables/validation_semantic_report_execution_B2_parent_metric.csv', b2parent)
    b2_children = []
    if b2p < ALPHA:
        rawp = [permutation_p(float(b2obs.dimension_delta_r2[j]), b2perm[f'Delta_R2_{r}'].to_numpy(float)) for j, r in enumerate(REPORTS)]
        h = core.holm_table(REPORTS, rawp, ALPHA)
        for j, (r, p0) in enumerate(zip(REPORTS, rawp)):
            hr = h[h.test_id.eq(r)].iloc[0]
            b2_children.append({'report': r, 'status': 'RUN_PARENT_PASSED', 'Delta_R2': float(b2obs.dimension_delta_r2[j]), 'p_raw': p0, 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
    else:
        for j, r in enumerate(REPORTS):
            b2_children.append({'report': r, 'status': 'NOT_INTERPRETED_PARENT_GATE', 'Delta_R2': float(b2obs.dimension_delta_r2[j]), 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
    atomic_csv(out / 'tables/validation_semantic_report_execution_B2_dimension_followups.csv', pd.DataFrame(b2_children))
    print(f'[{now()}] B3 joint 10-report prediction of report-transition family', flush=True)
    b3_rows, b3_fold_tables, b3_perm_frames = ([], [], [])
    b3_obs_by_k, b3_df_by_k = ({}, {})
    for k in TRANSITION_K:
        df = prepare_logistic(targets, structural, 'transition', k)
        b3_df_by_k[k] = df
        obs = core.nested_lopo_logistic(df, 'family_1based', REPORTS, salt=f'B3_K{k}')
        b3_obs_by_k[k] = obs
        ft = pd.DataFrame(obs.fold_rows)
        ft.insert(0, 'k', k)
        b3_fold_tables.append(ft)
        cp = cache / f'B3_K{k}_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        for i in range(0, len(pending), 5):
            inds = pending[i:i + 5]

            def fn(pp):
                return {'permutation': pp, 'Delta_BA': core.permutation_logistic_stat(df, 'family_1based', REPORTS, pp, f'B3_K{k}', obs.baseline_ba)}
            rows.extend(core.parallel_map(fn, inds, workers))
            save_checkpoint(cp, rows)
            print(f'  B3 K={k}: {min(i + 5, len(pending)) + len(done)}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        p = permutation_p(obs.statistic, perm.Delta_BA.to_numpy(float))
        b3_rows.append({'test_id': f'B3_K{k}', 'k': k, 'trials': len(df), 'participants': df.subject_id.nunique(), 'baseline_BA': obs.baseline_ba, 'full_10report_BA': obs.full_ba, 'Delta_BA': obs.statistic, 'parent_p_raw': p, 'permutations': MODEL_PERMUTATIONS, 'null_mean': float(perm.Delta_BA.mean()), 'null_sd': float(perm.Delta_BA.std(ddof=1))})
        q = perm.copy()
        q.insert(0, 'k', k)
        b3_perm_frames.append(q)
    b3_metrics = pd.DataFrame(b3_rows)
    atomic_csv(out / 'tables/validation_semantic_report_execution_B3_parent_metrics.csv', b3_metrics)
    atomic_csv(out / 'qc/validation_semantic_report_execution_B3_outer_fold_audit.csv', pd.concat(b3_fold_tables, ignore_index=True))
    atomic_csv(out / 'tables/validation_semantic_report_execution_B3_permutation_values.csv', pd.concat(b3_perm_frames, ignore_index=True))
    b3_holm = core.holm_table([f'B3_K{k}' for k in TRANSITION_K], [float(b3_metrics.loc[b3_metrics.k.eq(k), 'parent_p_raw'].iloc[0]) for k in TRANSITION_K], ALPHA)
    atomic_csv(out / 'tables/validation_semantic_report_execution_B3_holm.csv', b3_holm)
    b3_child_rows = []
    b3_child_perm_frames = []
    for k in TRANSITION_K:
        parent = b3_holm[b3_holm.test_id.eq(f'B3_K{k}')].iloc[0]
        df = b3_df_by_k[k]
        obs = b3_obs_by_k[k]
        if not bool(parent.reject_holm_0p05):
            for r in REPORTS:
                b3_child_rows.append({'k': k, 'report': r, 'status': 'NOT_RUN_PARENT_GATE', 'Delta_BA': np.nan, 'p_raw': np.nan, 'p_holm_within_10': np.nan, 'reject_holm_0p05': False})
            continue
        child_obs = {r: core.nested_lopo_logistic_full_ba(df, 'family_1based', [r], salt=f'B3_K{k}') - obs.baseline_ba for r in REPORTS}
        cp = cache / f'B3_K{k}_single_report_permutations.csv'
        rows = load_checkpoint(cp)
        done = {int(r['permutation']) for r in rows}
        pending = [p for p in range(MODEL_PERMUTATIONS) if p not in done]
        for i in range(0, len(pending), 2):
            inds = pending[i:i + 2]

            def fn(pp):
                return core.single_report_permutation_row(df, 'family_1based', pp, f'B3_K{k}', 'B3_reports', obs.baseline_ba)
            rows.extend(core.parallel_map(fn, inds, workers))
            save_checkpoint(cp, rows)
            print(f'  B3 K={k} child reports: {min(i + 2, len(pending)) + len(done)}/{MODEL_PERMUTATIONS}', flush=True)
        perm = pd.DataFrame(rows).sort_values('permutation')
        rawp = [permutation_p(child_obs[r], perm[r].to_numpy(float)) for r in REPORTS]
        h = core.holm_table(REPORTS, rawp, ALPHA)
        for r, p0 in zip(REPORTS, rawp):
            hr = h[h.test_id.eq(r)].iloc[0]
            b3_child_rows.append({'k': k, 'report': r, 'status': 'RUN_PARENT_PASSED', 'Delta_BA': child_obs[r], 'p_raw': p0, 'p_holm_within_10': float(hr.p_holm), 'reject_holm_0p05': bool(hr.reject_holm_0p05)})
        q = perm.copy()
        q.insert(0, 'k', k)
        b3_child_perm_frames.append(q)
    atomic_csv(out / 'tables/validation_semantic_report_execution_B3_single_report_followups.csv', pd.DataFrame(b3_child_rows))
    b3_child_perm = pd.concat(b3_child_perm_frames, ignore_index=True) if b3_child_perm_frames else pd.DataFrame(columns=['k', 'permutation'] + REPORTS)
    atomic_csv(out / 'tables/validation_semantic_report_execution_B3_single_report_permutation_values.csv', b3_child_perm)
    parent_rows = []
    for _, r in a_holm.iterrows():
        parent_rows.append({'family': 'A', 'test_id': r.test_id, 'p_raw': float(r.p_raw), 'p_adjusted': float(r.p_holm), 'reject_adjusted_0p05': bool(r.reject_holm_0p05)})
    parent_rows.append({'family': 'B2', 'test_id': 'B2', 'p_raw': b2p, 'p_adjusted': b2p, 'reject_adjusted_0p05': bool(b2p < ALPHA)})
    for _, r in b3_holm.iterrows():
        parent_rows.append({'family': 'B3', 'test_id': r.test_id, 'p_raw': float(r.p_raw), 'p_adjusted': float(r.p_holm), 'reject_adjusted_0p05': bool(r.reject_holm_0p05)})
    parent_table = pd.DataFrame(parent_rows)
    atomic_csv(out / 'tables/validation_semantic_report_execution_validation_parent_test_registry.csv', parent_table)
    numeric_identity = {'phase': PHASE, 'generated_at': now(), 'discovery_semantic_report_execution_numeric_source_sha256': discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_numeric_implementation_specification_sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'discovery_semantic_report_execution_helpers_reused_exactly': True, 'permutation_seed_namespace_inherited_from_fixed_discovery_semantic_report_execution_helper': True, 'Discovery_results_used_for_test_selection': False, 'tests_run': ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']}
    writej(out / 'qc/validation_semantic_report_execution_numeric_implementation_identity.json', numeric_identity)
    source_rows = []
    for role, rel, expected in [('discovery_semantic_report_execution_final_lock', discovery_semantic_report_execution_LOCK, discovery_semantic_report_execution_LOCK_SHA256), ('discovery_semantic_report_execution_next_policy', discovery_semantic_report_execution_NEXT, discovery_semantic_report_execution_NEXT_SHA256), ('discovery_semantic_report_execution_numeric_specification', discovery_semantic_report_execution_NUMERIC_specification, discovery_semantic_report_execution_NUMERIC_specification_SHA256), ('discovery_semantic_report_execution_numeric_source', discovery_semantic_report_execution_SOURCE, discovery_semantic_report_execution_SOURCE_SHA256), ('semantic_report_execution_design_program', semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256), ('semantic_report_execution_design_raw_inventory', semantic_report_execution_design_RAW_INVENTORY, semantic_report_execution_design_RAW_INVENTORY_SHA256), ('canonical_reports', CANONICAL, CANONICAL_SHA256), ('Validation_timing', VAL_TIMING, VAL_TIMING_SHA256), ('README_parser', README_SEQUENCE_CORE, README_SEQUENCE_CORE_SHA256), ('Validation_experience_assignments', EXP_ASSIGN, EXP_ASSIGN_SHA256), ('Validation_experience_signature', EXP_SIG, EXP_SIG_SHA256), ('Validation_transition_assignments', TR_ASSIGN, TR_ASSIGN_SHA256), ('Validation_transition_signature', TR_SIG, TR_SIG_SHA256)]:
        got = sha256(project / rel)
        if got != expected:
            raise RuntimeError(f'Unexpected source rehash drift: {rel}')
        source_rows.append({'role': role, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': True})
    source_rows.append({'role': 'dataset_README', 'path': str(dataset / README_REL), 'expected_sha256': README_SHA256, 'observed_sha256': sha256(dataset / README_REL), 'verified': sha256(dataset / README_REL) == README_SHA256})
    atomic_csv(out / 'qc/validation_semantic_report_execution_source_rehash.csv', pd.DataFrame(source_rows))
    cohort_rows = []
    for k in EXPERIENCE_K:
        d1 = prepare_a1(targets, structural, k)
        d2 = prepare_logistic(targets, structural, 'experience', k)
        cohort_rows += [{'test_id': f'A1_K{k}', 'trials': len(d1), 'participants': d1.subject_id.nunique()}, {'test_id': f'A2_K{k}', 'trials': len(d2), 'participants': d2.subject_id.nunique()}]
    cohort_rows.append({'test_id': 'B2', 'trials': len(b2df), 'participants': b2df.subject_id.nunique()})
    for k in TRANSITION_K:
        d = prepare_logistic(targets, structural, 'transition', k)
        cohort_rows.append({'test_id': f'B3_K{k}', 'trials': len(d), 'participants': d.subject_id.nunique()})
    atomic_csv(out / 'qc/validation_semantic_report_execution_test_cohort_audit.csv', pd.DataFrame(cohort_rows))
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'role': 'Validation semantic/report synthesis plus Confirmation opening/analysis specification; validation_confirmation_semantic_design itself opens no Confirmation values', 'Validation_results_for_audit_only_not_test_selection': parent_table.to_dict('records'), 'all_prespecified_tests_required_in_Confirmation': ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3'], 'Validation_results_select_Confirmation_tests': False, 'Confirmation_behavior_opening_permitted_in_validation_confirmation_semantic_design': False, 'Confirmation_behavior_opening_may_be_prespecified_for_confirmation_semantic_report_execution': True, 'new_EEG_signal_opening_permitted': False, 'canonical_context_stimulus_category_permitted': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'discovery_semantic_report_execution_numeric_source_sha256': discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_numeric_implementation_specification_sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256}
    writej(project / NEXT_CONFIG, next_policy)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'Validation_semantic_report_execution_complete', 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'A_family_parent_results': parent_table[parent_table.family.eq('A')].to_dict('records'), 'B2_parent_result': parent_table[parent_table.test_id.eq('B2')].to_dict('records')[0], 'B3_parent_results': parent_table[parent_table.family.eq('B3')].to_dict('records'), 'Validation_behavioral_report_values_read': True, 'Validation_emotion_category_values_read': True, 'Validation_context_values_read': True, 'Discovery_behavioral_values_reopened': False, 'Confirmation_behavioral_values_read': False, 'new_EEG_signal_values_opened': False, 'canonical_report_columns_used': True, 'canonical_context_category_used': False, 'raw_behavior_reports_source_audited': True, 'Discovery_results_selected_Validation_tests': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_opens_Confirmation_values': False, 'runtime_seconds': time.time() - started}
    writej(out / 'qc/validation_semantic_report_execution_summary.json', summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'source_code_sha256': script_sha, 'discovery_semantic_report_execution_final_lock_sha256': discovery_semantic_report_execution_LOCK_SHA256, 'discovery_semantic_report_execution_next_policy_sha256': discovery_semantic_report_execution_NEXT_SHA256, 'discovery_semantic_report_execution_numeric_source_sha256': discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_numeric_specification_sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'boundary': {'Validation_behavioral_report_values_read': True, 'Validation_emotion_category_values_read': True, 'Validation_context_values_read': True, 'Discovery_behavioral_values_reopened': False, 'Confirmation_behavioral_values_read': False, 'new_EEG_signal_values_opened': False, 'canonical_context_category_used': False, 'behavior_changed_structural_K_or_family_ids': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE}
    writej(out / 'qc/validation_semantic_report_execution_contract.json', contract)
    report_lines = ['# validation_semantic_report_execution — Validation semantic/report execution', '', '**One-shot Validation target opening. All semantic_report_execution_design-prespecified tests were executed regardless of Discovery results. No new EEG signal arrays or Confirmation behavior were opened.**', '', '## Validation parent results', '', '```text', parent_table.to_string(index=False), '```', '', '## Target lineage', '', 'Reports came only from the prespecified canonical report columns and were source-audited against the raw Validation behavior TSVs. Canonical context/stimulus/category were not used.', 'Context came from the authoritative validation_source_epoch_lineage event-timing lineage and was analyzable only when it agreed with the safe metadata schedule. Category was reconstructed with the exact prespecified README sequence parser using the recomputed within-context schedule index; no fallback was permitted.', '', '## Interpretation discipline', '', 'A1/A2 address alignment between human labels/reports and the replicated experience partitions. Null results alone do not prove that human labels are wrong.', 'B2/B3 address report/evaluation-stage neural transformation. A positive association cannot be described as distortion, dishonesty, suppression, exaggeration, or conscious editing.', '', '## Lockbox', '', 'Validation results do not select Confirmation tests. validation_confirmation_semantic_design is synthesis/design-specification only and must not open Confirmation values; the identical prespecified test set can only be opened one-shot in confirmation_semantic_report_execution after validation_confirmation_semantic_design locks.', '', f'## Next\n\n**{NEXT_PHASE}** — Validation synthesis + Confirmation semantic opening/analysis specification (no Confirmation values).', '']
    (out / 'reports/validation_semantic_report_execution_report.md').write_text(''.join(report_lines), encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, out / 'qc/validation_semantic_report_execution_contract.json', out / 'qc/validation_semantic_report_execution_summary.json', out / 'qc/validation_semantic_report_execution_source_rehash.csv', out / 'qc/validation_semantic_report_execution_test_cohort_audit.csv', out / 'qc/validation_semantic_report_execution_numeric_implementation_identity.json', out / 'qc/validation_semantic_report_execution_raw_report_source_audit.csv', out / 'qc/validation_semantic_report_execution_target_lineage_subject_audit.csv', out / 'qc/validation_semantic_report_execution_A2_outer_fold_audit.csv', out / 'qc/validation_semantic_report_execution_B2_outer_fold_audit.csv', out / 'qc/validation_semantic_report_execution_B3_outer_fold_audit.csv', out / 'tables/validation_semantic_report_execution_validation_semantic_target_manifest.csv', out / 'tables/validation_semantic_report_execution_A1_parent_metrics.csv', out / 'tables/validation_semantic_report_execution_A1_mixing_cells.csv', out / 'tables/validation_semantic_report_execution_A1_concentration.csv', out / 'tables/validation_semantic_report_execution_A1_null_summary.csv', out / 'tables/validation_semantic_report_execution_A1_permutation_values.csv', out / 'tables/validation_semantic_report_execution_A2_parent_metrics.csv', out / 'tables/validation_semantic_report_execution_A2_permutation_values.csv', out / 'tables/validation_semantic_report_execution_A_family_holm.csv', out / 'tables/validation_semantic_report_execution_A2_single_report_followups.csv', out / 'tables/validation_semantic_report_execution_A2_single_report_permutation_values.csv', out / 'tables/validation_semantic_report_execution_B2_parent_metric.csv', out / 'tables/validation_semantic_report_execution_B2_permutation_values.csv', out / 'tables/validation_semantic_report_execution_B2_dimension_followups.csv', out / 'tables/validation_semantic_report_execution_B3_parent_metrics.csv', out / 'tables/validation_semantic_report_execution_B3_permutation_values.csv', out / 'tables/validation_semantic_report_execution_B3_holm.csv', out / 'tables/validation_semantic_report_execution_B3_single_report_followups.csv', out / 'tables/validation_semantic_report_execution_B3_single_report_permutation_values.csv', out / 'tables/validation_semantic_report_execution_validation_parent_test_registry.csv', out / 'reports/validation_semantic_report_execution_report.md']
    final = {'phase': PHASE, 'generated_at': now(), 'status': 'Validation_semantic_report_execution_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'Validation_behavioral_values_read': True, 'Confirmation_behavioral_values_read': False, 'new_EEG_signal_values_opened': False, 'canonical_context_category_used': False, 'raw_behavior_reports_source_audited': True, 'Discovery_results_selected_Validation_tests': False, 'Validation_results_select_Confirmation_tests': False, 'behavior_changed_structural_K_or_family_ids': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_opens_Confirmation_values': False}
    writej(final_lock, final)
    phase_dir = Path(__file__).resolve().parent
    print('')
    print('validation_semantic_report_execution PASS')
    for _, r in parent_table.iterrows():
        print(f'{r.test_id}: p_raw={r.p_raw:.6g}, p_adjusted={r.p_adjusted:.6g}, reject={bool(r.reject_adjusted_0p05)}')
    print('Validation behavioral values opened: True')
    print('Confirmation behavioral values opened: False')
    print('New EEG signal values opened: False')
    print('Canonical context/category used: False')
    print('Raw behavior reports source-audited: True')
    print('Discovery results selected Validation tests: False')
    print('Validation results select Confirmation tests: False')
    print('Semantic relabeling permitted: False')
    print('Distortion interpretation permitted: False')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())