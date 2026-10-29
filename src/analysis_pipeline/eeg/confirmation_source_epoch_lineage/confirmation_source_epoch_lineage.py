# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase confirmation_source_epoch_lineage — Confirmation source/header/epoch lineage specification.

This phase opens Confirmation only for safe roster keys, source EDF headers,
annotations, and event timing. It reuses the exact prespecified the validated source-epoch attachment rule event
lineage implementation and raw_eeg_epoch_folds epoch-window implementation. No EEG sample
amplitude, report, outcome, emotion-category, or stimulus-identity value is read."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, re, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'confirmation_source_epoch_lineage'
NEXT_PHASE = 'confirmation_preprocessing_qc'
ROSTER = {'sub-03': 42, 'sub-13': 42, 'sub-19': 42, 'sub-23': 42, 'sub-27': 42, 'sub-31': 42, 'sub-35': 42, 'sub-46': 42, 'sub-47': 42, 'sub-48': 42, 'sub-49': 42, 'sub-59': 32}
FULL42 = {s for s, n in ROSTER.items() if n == 42}
SUB59_TRIALS = list(range(1, 16)) + list(range(22, 39))
EXPECTED = {'n_lock': 'd2f06565fa0e458af159025f12b6c17df97ecf35f7d6fba3ff2be94589f709a4', 'n_contract': 'b4d3f7d8c335ada6e48217c46a5b1818449253119e073f0923453a553745f8a3', 'n_next': '22a10a93a64d1a53039649761f9eb13f43e2f98f5e76e59f03b624a6efc50806', 'n_policy': '3b7cd41357c55da0a1e464806a66741349464934472faf18bc60b4ecc8152d52', 'n_attachment': 'd4b05598dde8cc351eb7da217aa3ea455c221c6ed1c453f942a11bd94a4ba638', 'n_reference': '383c3528514944e8f76eb03a6b8ee39c22fa1ad5e6f53d6278e4691285380afa', 'n_roster': '7dca92433e0259d14ed5b391c8207e06b67ed932eef4a6a25ed31fb724fe1ab8', 'n_core': '15dde8b52c7d7938c8cd4b3db19b6ca03cf73ca13416fab0d8736c0b77e8ee5b', 'n_lineage': 'c088554da2135e5f397edc4301913f2fc2b6f15932ffbbc9384471719d669bf7', 'n_executor': '81264dd342d8ed4ef2c7f32493d009c66f736d3a831a51040900fe67a064842e', 'raw_eeg_epoch_folds': '93ca3cc67f7fb72f008845941a16f8a802e4b0aee601d53fbeb5c84c8e25aa8b', 'raw_eeg_preprocessing_qc': 'bd6a50669f0a6ba17965a4497433f1d4583d4a98f840dff29a73866d801e2ecb', 'outcome_free_features': 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba', 'canonical': 'e31f96f918e806d466fd2bccc85b94d5c9dc96002bf97a97a442e8059a0365ed', 'roster_specification': '344cc931adb99a727a77f5a03c2b8a77d6af8563fd5218a91b4517d7f568b0dc'}
TRANSFER_CANONICAL = '73df540947a247e541f20766de290b3c7cf6878ec2e1cc3bb17684f642c286c9'
ATTACHMENT_CANONICAL = 'a8b705d974af41d073f2e3e58f702fbb3e70efd5c8c0dbdbf1d2798e7f135278'
FORBIDDEN = ('score_', 'valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion', 'stimulus_id', 'video_name')
WITHIN_PERSON_FOLDS = 5

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def require(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise RuntimeError('Missing required files:\n  - ' + '\n  - '.join(missing))

def safe_csv(path, allowed, required):
    cols = list(pd.read_csv(path, nrows=0).columns)
    use = [c for c in allowed if c in cols]
    if any((any((t in c.lower() for t in FORBIDDEN)) for c in use)):
        raise RuntimeError(f'Forbidden column requested from {Path(path).name}')
    missing = [c for c in required if c not in use]
    if missing:
        raise RuntimeError(f'{Path(path).name} missing safe columns {missing}; available={cols}')
    return pd.read_csv(path, usecols=use, low_memory=False)

def normalize_context(value):
    t = str(value).strip().lower()
    if t in {'ima', 'imagery', 'ses-ima'}:
        return 'imagery'
    if t in {'vid', 'video', 'ses-vid'}:
        return 'video'
    return t

def bool_value(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y'}

def zippack(project, files):
    root = f'confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_{tag()}'
    z = project / '' / f'{root}'
    z.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(z)
    Path(str(z) + '.sha256').write_text(f'{digest}  {z.name}')
    return (z, digest)

def source_paths(project):
    return {'n_lock': project / 'outputs/qc/validation_confirmation_synthesis_final_output_lock_hashes.json', 'n_contract': project / 'config/validation_confirmation_synthesis_validation_confirmation_synthesis_contract.json', 'n_next': project / 'analysis_specifications/S076_validation_confirmation_synthesis_source_epoch.json', 'n_policy': project / 'analysis_specifications/S075_validation_confirmation_synthesis_geometry_transfer.json', 'n_attachment': project / 'analysis_specifications/S073_validation_confirmation_synthesis_event_attachment.json', 'n_reference': project / 'analysis_specifications/S074_validation_confirmation_synthesis_geometry_reference.json', 'n_roster': project / 'outputs/tables/validation_confirmation_synthesis_confirmation_roster_manifest.csv', 'n_core': project / 'analysis_specifications/reference_code/validation_confirmation_transfer_core.py', 'n_lineage': project / 'analysis_specifications/reference_code/validation_lineage_reference.py', 'n_executor': project / 'src/analysis_pipeline/eeg/validation_confirmation_synthesis/validation_confirmation_synthesis.py', 'raw_eeg_epoch_folds': project / 'src/analysis_pipeline/eeg/raw_eeg_epoch_folds/raw_eeg_epoch_folds.py', 'raw_eeg_preprocessing_qc': project / 'src/analysis_pipeline/eeg/raw_eeg_preprocessing_qc/raw_eeg_preprocessing_qc.py', 'outcome_free_features': project / 'src/analysis_pipeline/eeg/outcome_free_features/outcome_free_features.py', 'canonical': project / 'data/canonical_behavioral_table.csv', 'roster_specification': project / 'analysis_specifications/S009_confirmation_source_epoch_lineage_trial.json'}

def verify_validation_confirmation_synthesis(project, paths):
    require(paths.values())
    for key, expected in EXPECTED.items():
        observed = sha(paths[key])
        if observed != expected:
            raise RuntimeError(f'{key} hash mismatch: expected {expected}, observed {observed}')
    lock = json.loads(paths['n_lock'].read_text())
    contract = json.loads(paths['n_contract'].read_text())
    next_policy = json.loads(paths['n_next'].read_text())
    policy = json.loads(paths['n_policy'].read_text())
    attachment = json.loads(paths['n_attachment'].read_text())
    roster = pd.read_csv(paths['n_roster'], low_memory=False)
    roster_specification = json.loads(paths['roster_specification'].read_text())
    if canonical_sha(policy) != TRANSFER_CANONICAL:
        raise RuntimeError('validation_confirmation_synthesis transfer policy canonical hash mismatch')
    if canonical_sha(attachment) != ATTACHMENT_CANONICAL:
        raise RuntimeError('validation_confirmation_synthesis attachment policy canonical hash mismatch')
    if next_policy.get('next_phase') != PHASE or not next_policy.get('next_phase_permitted'):
        raise RuntimeError('validation_confirmation_synthesis did not authorize confirmation_source_epoch_lineage')
    if contract.get('boundary', {}).get('confirmation_files_read') or contract.get('boundary', {}).get('confirmation_opened'):
        raise RuntimeError('validation_confirmation_synthesis Confirmation boundary was violated')
    if contract.get('executor_sha256') != EXPECTED['n_executor']:
        raise RuntimeError('Unexpected validation_confirmation_synthesis executor')
    expected_subjects = list(ROSTER)
    if roster.subject_id.astype(str).tolist() != expected_subjects or roster.n_trials.astype(int).tolist() != [ROSTER[s] for s in expected_subjects] or int(roster.n_trials.sum()) != 494:
        raise RuntimeError('prespecified validation_confirmation_synthesis roster changed')
    if roster.confirmation_accessed_in_validation_confirmation_synthesis.map(bool_value).any():
        raise RuntimeError('validation_confirmation_synthesis says Confirmation was accessed')
    if roster_specification.get('subjects') != {k: list(range(1, 43)) if k != 'sub-59' else SUB59_TRIALS for k in ROSTER}:
        raise RuntimeError('confirmation_source_epoch_lineage exact trial-key specification changed')
    rel = {'source_rehash': 'outputs/qc/validation_confirmation_synthesis_source_rehash.csv', 'evidence': 'outputs/tables/validation_confirmation_synthesis_validation_confirmation_evidence_matrix.csv', 'roster': 'outputs/tables/validation_confirmation_synthesis_confirmation_roster_manifest.csv', 'preprocessing': 'outputs/tables/validation_confirmation_synthesis_discovery_reference_preprocessing.csv', 'centroids': 'outputs/tables/validation_confirmation_synthesis_discovery_reference_centroids.csv', 'schema': 'outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_output_schema.csv', 'gates': 'outputs/tables/validation_confirmation_synthesis_prespecified_confirmation_gate_manifest.csv', 'reference': 'analysis_specifications/S074_validation_confirmation_synthesis_geometry_reference.json', 'policy': 'analysis_specifications/S075_validation_confirmation_synthesis_geometry_transfer.json', 'attachment_policy': 'analysis_specifications/S073_validation_confirmation_synthesis_event_attachment.json', 'next': 'analysis_specifications/S076_validation_confirmation_synthesis_source_epoch.json', 'core': 'analysis_specifications/reference_code/validation_confirmation_transfer_core.py', 'lineage_reference': 'analysis_specifications/reference_code/validation_lineage_reference.py', 'summary': 'outputs/qc/validation_confirmation_synthesis_validation_confirmation_synthesis_summary.json', 'report': 'outputs/reports/validation_confirmation_synthesis_validation_confirmation_synthesis_report.md', 'contract': 'config/validation_confirmation_synthesis_validation_confirmation_synthesis_contract.json'}
    for key, expected in lock.get('hashes', {}).items():
        if key not in rel:
            raise RuntimeError(f'Unknown validation_confirmation_synthesis fixed object: {key}')
        p = project / rel[key]
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError(f'validation_confirmation_synthesis fixed object mismatch: {key}')
    return (policy, attachment, roster_specification)

def build_schedule(project, dataset, roster_specification):
    canonical = project / 'data/canonical_behavioral_table.csv'
    c = safe_csv(canonical, ('subject_id', 'trial_number_global'), ('subject_id', 'trial_number_global'))
    c = c[c.subject_id.astype(str).isin(ROSTER)].copy()
    c.trial_number_global = pd.to_numeric(c.trial_number_global, errors='raise').astype(int)
    expected = roster_specification['subjects']
    observed = {s: g.trial_number_global.sort_values().astype(int).tolist() for s, g in c.groupby('subject_id', sort=True)}
    if observed != expected or len(c) != 494 or c.duplicated(['subject_id', 'trial_number_global']).any():
        raise RuntimeError(f'Canonical Confirmation trial-key gate failed: {observed}')
    rows = []
    metadata_paths = []
    for sub, g in c.groupby('subject_id', sort=True):
        mp = dataset / 'derivatives/metadata' / f'{sub}_trials_with_context.csv'
        require([mp])
        metadata_paths.append(mp)
        m = safe_csv(mp, ('sub', 'global_trial', 'context', 'context_trial_idx', 'rating_time_sec', 'trial_number'), ('global_trial', 'trial_number', 'context', 'rating_time_sec'))
        m.global_trial = pd.to_numeric(m.global_trial, errors='raise').astype(int)
        m.trial_number = pd.to_numeric(m.trial_number, errors='raise').astype(int)
        m.rating_time_sec = pd.to_numeric(m.rating_time_sec, errors='coerce')
        m.context = m.context.map(normalize_context)
        if m.duplicated('trial_number').any():
            raise RuntimeError(f'{sub}: duplicate metadata trial_number')
        x = g.merge(m, left_on='trial_number_global', right_on='trial_number', how='left', validate='one_to_one')
        if x.rating_time_sec.isna().any() or (~x.context.isin(['imagery', 'video'])).any():
            raise RuntimeError(f'{sub}: safe metadata fields incomplete')
        for r in x.sort_values('trial_number_global').itertuples():
            rows.append({'raw_behavior_row_id': f'confirmation|{sub}|trial-{int(r.trial_number_global):03d}', 'subject_id': sub, 'trial_number': int(r.trial_number_global), 'metadata_global_trial': int(r.global_trial), 'metadata_context': str(r.context), 'metadata_rating_time_sec': float(r.rating_time_sec), 'metadata_fields_role': 'diagnostic_non_authoritative_except_sub59_exact_rating_anchor', 'reports_outcomes_read': False, 'stimulus_identity_read': False, 'canonical_context_read': False, 'metadata_global_trial_used_as_join_key': False})
    out = pd.DataFrame(rows).sort_values(['subject_id', 'trial_number'], kind='mergesort').reset_index(drop=True)
    return (out, canonical, metadata_paths)

def attach_partial_subject(subject, g, events):
    e = events[events.subject_id.eq(subject)].sort_values(['global_onset_sec', 'run', 'annotation_index'], kind='mergesort').reset_index(drop=True)
    ratings = e[e.event_class.eq('rating')].reset_index(drop=True)
    rows = []
    used_stim = set()
    used_rating = set()
    for r in g.sort_values('trial_number', kind='mergesort').itertuples():
        delta = np.abs(ratings.global_onset_sec.to_numpy(float) - float(r.metadata_rating_time_sec))
        hit = np.flatnonzero(delta <= 0.001)
        if len(hit) != 1:
            raise RuntimeError(f'{subject} trial {r.trial_number}: exact retained rating anchor count {len(hit)}')
        pos = int(hit[0])
        rr = ratings.iloc[pos]
        previous = float(ratings.iloc[pos - 1].global_onset_sec) if pos > 0 else -math.inf
        stimuli = e[(e.global_onset_sec > previous + 1e-09) & (e.global_onset_sec < float(rr.global_onset_sec) - 1e-09) & e.event_class.isin(['imagery', 'video'])].sort_values(['global_onset_sec', 'run', 'annotation_index'], kind='mergesort')
        if len(stimuli) != 1:
            raise RuntimeError(f'{subject} trial {r.trial_number}: expected one exact source stimulus between adjacent raw ratings; observed {len(stimuli)}')
        ss = stimuli.iloc[0]
        sk = (str(ss.source_path), int(ss.annotation_index))
        rk = (str(rr.source_path), int(rr.annotation_index))
        if sk in used_stim or rk in used_rating:
            raise RuntimeError(f'{subject}: source event reuse')
        used_stim.add(sk)
        used_rating.add(rk)
        if int(ss.run) != int(rr.run) or str(ss.source_path) != str(rr.source_path):
            raise RuntimeError(f'{subject}: cross-run retained pair')
        evctx = str(ss.event_class)
        rows.append({**r._asdict(), 'context': evctx, 'context_source': 'partial_roster_exact_retained_rating_anchor_unique_adjacent_source_stimulus', 'metadata_event_context_agreement': bool(str(r.metadata_context) == evctx), 'metadata_rating_delta_from_selected_sec': float(rr.global_onset_sec - float(r.metadata_rating_time_sec)), 'source_previous_rating_global_onset_sec': previous if math.isfinite(previous) else np.nan, 'source_stimulus_candidate_count': 1, 'source_stimulus_candidate_contexts': evctx, 'source_stimulus_mixed_context_candidates': False, 'source_competing_stimulus_before_selected_count': 0, 'source_stimulus_selection_rule': 'partial_roster_exact_retained_rating_anchor_unique_adjacent_source_stimulus', 'event_grammar_pair_cost': np.nan, 'event_grammar_selection_stability_count': 3, 'event_grammar_selected_in_tight': True, 'event_grammar_selected_in_wide': True, 'source_stimulus_run': int(ss.run), 'source_stimulus_path': str(ss.source_path), 'source_stimulus_annotation_index': int(ss.annotation_index), 'source_stimulus_local_onset_sec': float(ss.onset_sec), 'source_stimulus_global_onset_sec': float(ss.global_onset_sec), 'source_rating_run': int(rr.run), 'source_rating_path': str(rr.source_path), 'source_rating_annotation_index': int(rr.annotation_index), 'source_rating_local_onset_sec': float(rr.onset_sec), 'source_rating_global_onset_sec': float(rr.global_onset_sec), 'continuous_event_authoritative': True})
    out = pd.DataFrame(rows)
    if len(out) != ROSTER[subject] or out.trial_number.astype(int).tolist() != SUB59_TRIALS:
        raise RuntimeError(f'{subject}: retained trial-ID gate failed')
    return out

def attach_all(schedule, events, lineage):
    rows = []
    candidate_rows = []
    selected_rows = []
    qc = []
    lookup = lineage.event_row_lookup(events)
    for sub, g in schedule.groupby('subject_id', sort=True):
        if sub not in FULL42:
            part = attach_partial_subject(sub, g, events)
            rows.append(part)
            qc.append({'subject_id': sub, 'attachment_method': 'exact_retained_anchor', 'roster_trials': len(part), 'target_imagery': np.nan, 'target_video': np.nan, 'primary_selected': len(part), 'tight_selected': len(part), 'wide_selected': len(part), 'primary_pairs_in_tight': len(part), 'primary_pairs_in_wide': len(part), 'primary_total_cost': np.nan, 'attachment_complete': True})
            continue
        variants = {}
        stats = {}
        for name in lineage.VARIANTS:
            cand = lineage.generate_candidates(sub, events, name)
            sel, st = lineage.select_nonoverlap(cand, 21, 21)
            candidate_rows.append(cand)
            variants[name] = sel
            stats[name] = st
            diagnostic = sel.copy()
            diagnostic['selected_variant'] = name
            selected_rows.append(diagnostic)
        primary = variants['primary'].sort_values('rating_onset_global_sec', kind='mergesort').reset_index(drop=True)
        if not stats['primary']['feasible'] or len(primary) != 42:
            raise RuntimeError(f"{sub}: prespecified primary 21/21 event grammar did not reach 42 trials: {stats['primary']}")
        if (primary.stimulus_run != primary.rating_run).any() or (primary.stimulus_source_path != primary.rating_source_path).any():
            raise RuntimeError(f'{sub}: cross-run primary pair')
        tight_set = lineage.signature(variants['tight'])
        wide_set = lineage.signature(variants['wide'])
        primary_set = lineage.signature(primary)
        ordered = g.sort_values('trial_number', kind='mergesort').reset_index(drop=True)
        for i in range(42):
            pair = primary.iloc[i]
            signature = (round(float(pair.stimulus_onset_global_sec), 4), round(float(pair.rating_onset_global_sec), 4), str(pair.context))
            stability = 1 + int(signature in tight_set) + int(signature in wide_set)
            in_tight = bool(signature in tight_set)
            in_wide = bool(signature in wide_set)
            rows.append(pd.DataFrame([lineage.pair_to_trial(ordered.iloc[i], pair, lookup, 'primary_event_grammar_prespecified_21_imagery_21_video_chronological_sequence', in_tight, in_wide)]))
        qc.append({'subject_id': sub, 'attachment_method': 'primary_event_grammar_sequence', 'roster_trials': 42, 'target_imagery': 21, 'target_video': 21, 'primary_selected': len(primary), 'tight_selected': len(variants['tight']), 'wide_selected': len(variants['wide']), 'primary_pairs_in_tight': len(primary_set & tight_set), 'primary_pairs_in_wide': len(primary_set & wide_set), 'primary_total_cost': float(stats['primary']['total_cost']), 'attachment_complete': True})
    out = pd.concat(rows, ignore_index=True, sort=False).drop(columns=['Index'], errors='ignore')
    out = out.sort_values(['subject_id', 'source_stimulus_global_onset_sec', 'raw_behavior_row_id'], kind='mergesort').reset_index(drop=True)
    out['subject_acquisition_order_1based'] = out.groupby('subject_id').cumcount() + 1
    out['within_subject_block_fold'] = out.groupby('subject_id')['subject_acquisition_order_1based'].transform(lambda s: np.floor((s.to_numpy(float) - 1.0) * WITHIN_PERSON_FOLDS / float(len(s))).astype(int) + 1)
    if len(out) != 494 or out.subject_id.nunique() != 12 or out.groupby('subject_id').size().to_dict() != ROSTER or out.duplicated('raw_behavior_row_id').any():
        raise RuntimeError('Final Confirmation attached-roster gate failed')
    if out.duplicated(['source_rating_path', 'source_rating_annotation_index']).any() or out.duplicated(['source_stimulus_path', 'source_stimulus_annotation_index']).any():
        raise RuntimeError('Final source-event reuse gate failed')
    if not out.source_stimulus_path.eq(out.source_rating_path).all() or not out.source_stimulus_run.eq(out.source_rating_run).all():
        raise RuntimeError('Final source pair run/path mismatch')
    candidates = pd.concat(candidate_rows, ignore_index=True, sort=False) if candidate_rows else pd.DataFrame()
    selected = pd.concat(selected_rows, ignore_index=True, sort=False) if selected_rows else pd.DataFrame()
    return (out, candidates, selected, pd.DataFrame(qc))

def output_paths(project):
    return {'headers': project / 'outputs/tables/confirmation_source_epoch_lineage_confirmation_raw_header_manifest.csv', 'events': project / 'outputs/tables/confirmation_source_epoch_lineage_confirmation_annotation_event_inventory.csv', 'grammar_candidates': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_candidates.csv', 'grammar_selected': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_selected_pairs.csv', 'grammar_qc': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_subject_qc.csv', 'trials': project / 'outputs/tables/confirmation_source_epoch_lineage_confirmation_authoritative_trial_timing_manifest.csv', 'epochs': project / 'outputs/tables/confirmation_source_epoch_lineage_confirmation_prespecified_epoch_manifest.csv', 'coverage': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_epoch_coverage_summary.csv', 'subject_qc': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_subject_lineage_qc.csv', 'source_rehash': project / 'outputs/qc/confirmation_source_epoch_lineage_source_rehash.csv', 'summary': project / 'outputs/qc/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_summary.json', 'contract': project / 'config/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_contract.json', 'next': project / 'analysis_specifications/S008_confirmation_source_epoch_lineage_preprocessing.json', 'report': project / 'outputs/reports/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_report.md', 'lock': project / 'outputs/qc/confirmation_source_epoch_lineage_final_output_lock_hashes.json', 'log': project / 'outputs/logs/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage.log'}

def self_test():
    assert sum(ROSTER.values()) == 494 and SUB59_TRIALS == list(range(1, 16)) + list(range(22, 39))
    order = np.arange(1, 43)
    folds = np.floor((order - 1) * 5 / 42).astype(int) + 1
    assert set(folds) == {1, 2, 3, 4, 5}
    schedule = pd.DataFrame([{'raw_behavior_row_id': 'confirmation|sub-59|trial-001', 'subject_id': 'sub-59', 'trial_number': 1, 'metadata_global_trial': 1, 'metadata_context': 'imagery', 'metadata_rating_time_sec': 80.0}, {'raw_behavior_row_id': 'confirmation|sub-59|trial-022', 'subject_id': 'sub-59', 'trial_number': 22, 'metadata_global_trial': 22, 'metadata_context': 'video', 'metadata_rating_time_sec': 280.0}])
    events = pd.DataFrame([{'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 0, 'onset_sec': 10.0, 'global_onset_sec': 10.0, 'event_class': 'imagery'}, {'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 1, 'onset_sec': 80.0, 'global_onset_sec': 80.0, 'event_class': 'rating'}, {'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 2, 'onset_sec': 120.0, 'global_onset_sec': 120.0, 'event_class': 'video'}, {'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 3, 'onset_sec': 180.0, 'global_onset_sec': 180.0, 'event_class': 'rating'}, {'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 4, 'onset_sec': 210.0, 'global_onset_sec': 210.0, 'event_class': 'video'}, {'subject_id': 'sub-59', 'run': 1, 'source_path': 'x.edf', 'annotation_index': 5, 'onset_sec': 280.0, 'global_onset_sec': 280.0, 'event_class': 'rating'}])
    try:
        attach_partial_subject('sub-59', schedule, events)
    except RuntimeError as exc:
        assert 'retained trial-ID gate' in str(exc)
    print('confirmation_source_epoch_lineage self-test: PASS')
    return 0

def main():
    args = parse_args()
    if args.self_test:
        return self_test()
    project = args.project.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    started = time.time()
    paths = source_paths(project)
    outputs = output_paths(project)
    for p in outputs.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if outputs['lock'].exists() and (not args.force):
        raise RuntimeError('confirmation_source_epoch_lineage outputs exist; use --force after review')
    with outputs['log'].open('w') as log:

        def note(message):
            line = f'[{now()}] {message}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        policy, attachment, roster_specification = verify_validation_confirmation_synthesis(project, paths)
        note('Verified every validation_confirmation_synthesis fixed output, canonical policy, exact Confirmation roster, and prespecified raw_eeg_epoch_folds/raw_eeg_preprocessing_qc/outcome_free_features lineage.')
        raw_eeg_epoch_folds = load_module(paths['raw_eeg_epoch_folds'], 'prespecified_raw_eeg_epoch_folds_confirmation')
        lineage = load_module(paths['n_lineage'], 'prespecified_validation_source_epoch_lineage_lineage_reference')
        lineage.ROSTER = ROSTER.copy()
        lineage.FULL42 = FULL42.copy()
        trigger = lineage.trigger_map(dataset / 'README.md', set(ROSTER))
        schedule, canonical, metadata_paths = build_schedule(project, dataset, roster_specification)
        note('Reading Confirmation EDF headers and annotations only; no sample amplitude.')
        headers, events = lineage.headers_events(dataset, trigger)
        if len(headers) < 12 or not headers.channel_count.eq(64).all() or headers.sample_amplitude_read.map(bool_value).any():
            raise RuntimeError('Confirmation header/channel/amplitude gate failed')
        note('Applying the prespecified two-branch attachment rule: 21/21 global event grammar for complete rosters; exact retained anchors for sub-59.')
        trials, candidates, selected, grammar_qc = attach_all(schedule, events, lineage)
        trials = raw_eeg_epoch_folds.derive_trial_boundaries(trials, events, headers)
        epochs = raw_eeg_epoch_folds.build_epoch_manifest(trials, headers)
        coverage, subject_coverage, unavailable = raw_eeg_epoch_folds.coverage_tables(epochs)
        coverage = coverage.rename(columns={'branch_permitted_for_raw_eeg_preprocessing_qc': 'branch_permitted_for_confirmation_preprocessing_qc'})
        headers.to_csv(outputs['headers'], index=False)
        events.to_csv(outputs['events'], index=False)
        candidates.to_csv(outputs['grammar_candidates'], index=False)
        selected.to_csv(outputs['grammar_selected'], index=False)
        grammar_qc.to_csv(outputs['grammar_qc'], index=False)
        trials.to_csv(outputs['trials'], index=False)
        epochs.to_csv(outputs['epochs'], index=False)
        coverage.to_csv(outputs['coverage'], index=False)
        subject_rows = []
        for sub, g in trials.groupby('subject_id', sort=True):
            eg = epochs[epochs.subject_id.eq(sub)]
            q = grammar_qc[grammar_qc.subject_id.eq(sub)].iloc[0]
            subject_rows.append({'subject_id': sub, 'prespecified_roster_trials': ROSTER[sub], 'attached_trials': len(g), 'attachment_method': q.attachment_method, 'target_imagery': q.target_imagery, 'target_video': q.target_video, 'attached_imagery': int((g.context == 'imagery').sum()), 'attached_video': int((g.context == 'video').sum()), 'primary_pairs_in_tight': int(q.primary_pairs_in_tight), 'primary_pairs_in_wide': int(q.primary_pairs_in_wide), 'raw_edf_files': g.source_stimulus_path.nunique(), 'unique_stimulus_events': g[['source_stimulus_path', 'source_stimulus_annotation_index']].drop_duplicates().shape[0], 'unique_rating_events': g[['source_rating_path', 'source_rating_annotation_index']].drop_duplicates().shape[0], 'stimulus_early_available': int(eg.query("epoch_family=='stimulus_early'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'stimulus_late_available': int(eg.query("epoch_family=='stimulus_late'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'rating_early_available': int(eg.query("epoch_family=='rating_early'").epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'metadata_event_context_agreement_rows': int(g.metadata_event_context_agreement.sum()), 'metadata_event_context_disagreement_rows': int((~g.metadata_event_context_agreement).sum()), 'source_lineage_complete': True, 'reports_outcomes_read': False, 'eeg_amplitude_read': False})
        pd.DataFrame(subject_rows).to_csv(outputs['subject_qc'], index=False)
        source_rows = []
        semantic = {'n_contract', 'n_next', 'n_policy', 'n_attachment', 'n_reference', 'n_roster', 'roster_specification'}
        for key, p in paths.items():
            source_rows.append({'source_id': key, 'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size, 'semantic_values_read': key in semantic, 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        source_rows.append({'source_id': 'canonical_confirmation_trial_keys', 'path': str(canonical), 'sha256': sha(canonical), 'bytes': canonical.stat().st_size, 'semantic_values_read': True, 'semantic_values_scope': 'subject_id and trial_number_global only; canonical context not read', 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        for p in metadata_paths:
            source_rows.append({'source_id': 'safe_confirmation_metadata', 'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size, 'semantic_values_read': True, 'semantic_values_scope': 'trial_number/global_trial/context/rating_time_sec only; authoritative only for sub-59 retained rating anchors; otherwise diagnostic', 'eeg_amplitude_read': False, 'reports_outcomes_read': False})
        pd.DataFrame(source_rows).to_csv(outputs['source_rehash'], index=False)
        branches = bool(coverage.branch_permitted_for_confirmation_preprocessing_qc.map(raw_eeg_epoch_folds.python_bool).all())
        exact = len(trials) == 494 and trials.subject_id.nunique() == 12
        summary = {'generated_at': now(), 'phase': PHASE, 'status': 'confirmation_source_header_epoch_lineage_specification_complete_candidate', 'counts': {'confirmation_subjects': 12, 'authoritative_trials': len(trials), 'raw_edf_files': len(headers), 'annotation_events': len(events), 'event_grammar_candidate_rows': len(candidates), 'event_grammar_selected_diagnostic_rows': len(selected), 'epoch_manifest_rows': len(epochs), 'available_epoch_rows': int(epochs.epoch_available.map(raw_eeg_epoch_folds.python_bool).sum()), 'metadata_event_context_agreement_rows': int(trials.metadata_event_context_agreement.sum()), 'metadata_event_context_disagreement_rows': int((~trials.metadata_event_context_agreement).sum())}, 'coverage': coverage.to_dict('records'), 'decision': {'validation_confirmation_synthesis_integrity_passed': True, 'exact_prespecified_confirmation_trial_keys_passed': True, 'exact_prespecified_roster_attached': exact, 'all_complete_rosters_reconstructed_with_21_21_event_grammar': bool(grammar_qc[grammar_qc.subject_id != 'sub-59'].primary_selected.eq(42).all()), 'sub59_exact_retained_anchor_gate_passed': bool(trials[trials.subject_id.eq('sub-59')].trial_number.astype(int).tolist() == SUB59_TRIALS), 'raw_headers_and_annotations_read': True, 'eeg_amplitude_read': False, 'reports_outcomes_read': False, 'stimulus_identity_read': False, 'context_derived_from_exact_source_trigger': True, 'canonical_context_read': False, 'metadata_global_trial_used_as_join_key': False, 'all_three_epoch_branches_permitted_for_confirmation_preprocessing_qc': branches, 'confirmation_preprocessing_qc_preprocessing_permitted': bool(exact and branches), 'confirmation_opened_for_source_lineage_only': True, 'validation_targets_read': False, 'alternate_window_selected': False}, 'runtime_seconds': round(time.time() - started, 3)}
        outputs['summary'].write_text(json.dumps(summary, indent=2) + '')
        required = {k: sha(outputs[k]) for k in ('headers', 'trials', 'epochs', 'coverage', 'subject_qc', 'grammar_qc')}
        next_policy = {'generated_at': now(), 'phase': PHASE, 'next_phase': NEXT_PHASE, 'next_phase_permitted': summary['decision']['confirmation_preprocessing_qc_preprocessing_permitted'], 'next_phase_role': 'apply exact prespecified raw_eeg_preprocessing_qc preprocessing/QC to available Confirmation epochs; do not read reports, outcomes, emotion category, stimulus identity, or Validation targets', 'required_inputs': required, 'required_source_code_sha256': EXPECTED['raw_eeg_preprocessing_qc'], 'required_boundary': ['verify every confirmation_source_epoch_lineage hash', 'use only epoch_available rows', 'do not alter windows or preprocessing/QC', 'do not read reports/outcomes/stimulus identity', 'do not use context as a target', 'Validation targets remain unread'], 'confirmation_opened_for_preprocessing_only': False}
        outputs['next'].write_text(json.dumps(next_policy, indent=2) + '')
        core = {k: sha(outputs[k]) for k in ('headers', 'events', 'grammar_candidates', 'grammar_selected', 'grammar_qc', 'trials', 'epochs', 'coverage', 'subject_qc', 'source_rehash', 'summary', 'next')}
        contract = {**summary, 'transfer_policy_canonical_sha256': TRANSFER_CANONICAL, 'attachment_policy_canonical_sha256': ATTACHMENT_CANONICAL, 'exact_trial_key_specification_sha256': EXPECTED['roster_specification'], 'source_code_lineage': {'raw_eeg_epoch_folds': EXPECTED['raw_eeg_epoch_folds'], 'raw_eeg_preprocessing_qc': EXPECTED['raw_eeg_preprocessing_qc'], 'outcome_free_features': EXPECTED['outcome_free_features'], 'validation_source_epoch_lineage_reference': EXPECTED['n_lineage']}, 'epoch_specs': raw_eeg_epoch_folds.EPOCH_SPECS, 'prespecified_roster': ROSTER, 'prespecified_trial_ids': roster_specification['subjects'], 'attachment_rule': '42-trial subjects: prespecified primary global non-overlapping event grammar with 21 imagery/21 video and chronological behavioral mapping; sub-59: exact retained rating anchors with one exact adjacent source stimulus, preserving original trial IDs', 'output_hashes': core, 'source_dataset_modified': False}
        outputs['contract'].write_text(json.dumps(contract, indent=2) + '')
        outputs['report'].write_text(f'# Phase confirmation_source_epoch_lineage — Confirmation source/epoch lineage\n\n- Confirmation participants: **12**\n- Authoritative trials: **{len(trials)}**\n- Raw EDF files: **{len(headers)}**\n- Epoch rows: **{len(epochs)}**\n- EEG amplitude read: **False**\n- Reports/outcomes read: **False**\n- Complete 42-trial rosters: **primary event grammar, 21 imagery + 21 video, chronological behavioral mapping**\n- sub-59: **32 exact retained rating anchors with original IDs 1–15 and 22–38**\n- Metadata context/rating for complete rosters: **diagnostic only**\n- Canonical context read: **False**\n- All epoch branches permit confirmation_preprocessing_qc: **{branches}**')
        fixed = {**core, 'contract': sha(outputs['contract']), 'report': sha(outputs['report'])}
        outputs['lock'].write_text(json.dumps({'generated_at': now(), 'phase': PHASE, 'hashes': fixed, 'transfer_policy_canonical_sha256': TRANSFER_CANONICAL, 'attachment_policy_canonical_sha256': ATTACHMENT_CANONICAL, 'exact_trial_key_specification_sha256': EXPECTED['roster_specification'], 'eeg_amplitude_read': False, 'reports_outcomes_read': False, 'validation_targets_read': False}, indent=2) + '')
        note(f"confirmation_source_epoch_lineage complete. confirmation_preprocessing_qc permitted: {summary['decision']['confirmation_preprocessing_qc_preprocessing_permitted']}.")
    archive, digest = zippack(project, list(outputs.values()))
    print(f'{archive}\nSHA-256: {digest}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())