# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase confirmation_preprocessing_qc — Confirmation raw-EEG preprocessing and amplitude QC.

Scientific boundary
-------------------
This cohort adapter applies the exact prespecified raw_eeg_preprocessing_qc signal engine to the confirmation_source_epoch_lineage-
prespecified Confirmation epoch manifest. It changes only cohort selection, output
namespace, expected counts, and the already-prespecified Confirmation continuation
gates. The following remain exact raw_eeg_preprocessing_qc behavior:

- continuous 0.5–45 Hz fourth-order Butterworth SOS filtering, zero phase;
- exact confirmation_source_epoch_lineage source-sample extraction;
- scipy.signal.resample_poly to 200 Hz, padtype="line";
- deterministic recording-channel QC;
- iterative 10%-trimmed robust average reference;
- prespecified epoch-channel QC;
- 64-channel arrays with failed channels retained as NaN;
- no interpolation and no automatic ICA removal.

No report, outcome, emotion category, stimulus identity, released derivative
amplitude, Validation target file, or Confirmation semantic target is read. Context is used only as operational
metadata for already-prespecified retention diagnostics and is never a target.

outputs/large/confirmation_preprocessing_qc_confirmation_raw_eeg_epochs_v1/

Per-source checkpoints are resumable. A completed checkpoint is reused only
when the raw_eeg_preprocessing_qc policy, confirmation_source_epoch_lineage epoch-manifest hash, source size/mtime, stored file
sizes, and stored file SHA-256 values all match."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
DEFAULT_DATASET = Path('<DATASET_ROOT>')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
PHASE = 'confirmation_preprocessing_qc'
PREFIX = 'confirmation_preprocessing_qc'
EXPECTED_raw_eeg_preprocessing_qc_SHA = 'bd6a50669f0a6ba17965a4497433f1d4583d4a98f840dff29a73866d801e2ecb'
EXPECTED_O_COUNTS = {'subjects': 12, 'trials': 494, 'epoch_rows': 1482, 'available_rows': 1457, 'raw_sources': 23}
EXPECTED_O_HASHES = {'headers': '2bcc3c06bd27445dcb621595a4571c79047d11ac8b1ada6b1ada525f98d966b3', 'events': '3674f6acb0a7e2724a423009adbf3b88d86a5dfe2a9a749d472eae67d4d1e0fa', 'grammar_candidates': 'b3d3767c1d1e3859b9eb4a6b12145c02cb224fde5a0f098dc474f673453932da', 'grammar_selected': '73b2c1499667c85fef6eff975a3fde90ab18812685812a68e35be08141f55818', 'trials': 'dcb34cdd525faffb9fc38d865589d598d436a8429f225554290ae5c731dd693f', 'epochs': '7e4d98c19dad3e7997a56e74da424d8290e4d22a7f453a76e8ff8a514c2261bf', 'coverage': '8bf939bb4e50386190b3270427fb8437c195e4452bcd9a42a6f7f347a1639f32', 'subject_qc': '354913c12f61ddff22c6b013cf3c0255dcbcaf63337a44b9e69f35483d49f4f7', 'source_rehash': 'a7fe77c21cc26faa34b38bea9ddc645afe25d0c816e9098dfd60c1d3ac1f116a', 'grammar_qc': '22500960f8a985bf69ca7b58071f4be543c5fa74494c7538cf662667f4ba6a84', 'summary': 'eef16534bf6b0233e490699fbd8ebdf42d45f6ab7aa00357cebb3ac3a77aa765', 'next': '16b0b2cb6215d8da3a97caa3baa6fb249d2ef67718940a9ee38960afb7884c7e', 'contract': '40047315f5f4c2388a52fd84fb30325ca52ff396c2dff4b203f438c4347ca36d', 'report': '12262dca68c501830884ff4436e32be6a232ab6780b170f691ae51101658b9f1'}
validation_confirmation_synthesis_TRANSFER_POLICY_SHA = '73df540947a247e541f20766de290b3c7cf6878ec2e1cc3bb17684f642c286c9'
validation_confirmation_synthesis_ATTACHMENT_POLICY_SHA = 'a8b705d974af41d073f2e3e58f702fbb3e70efd5c8c0dbdbf1d2798e7f135278'
validation_confirmation_synthesis_EXACT_TRIAL_KEY_SHA = '344cc931adb99a727a77f5a03c2b8a77d6af8563fd5218a91b4517d7f568b0dc'
STORE_REL = Path('outputs/large/confirmation_preprocessing_qc_confirmation_raw_eeg_epochs_v1')
TRANSFER_GATES = {'minimum_eligible_participants': 9, 'minimum_complete_trajectory_fraction_of_494': 0.75, 'minimum_complete_trajectories_per_participant': 16}
O_REL = {'headers': Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_raw_header_manifest.csv'), 'events': Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_annotation_event_inventory.csv'), 'grammar_candidates': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_candidates.csv'), 'grammar_selected': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_selected_pairs.csv'), 'trials': Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_authoritative_trial_timing_manifest.csv'), 'epochs': Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_prespecified_epoch_manifest.csv'), 'coverage': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_epoch_coverage_summary.csv'), 'subject_qc': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_subject_lineage_qc.csv'), 'source_rehash': Path('outputs/qc/confirmation_source_epoch_lineage_source_rehash.csv'), 'grammar_qc': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_event_grammar_subject_qc.csv'), 'summary': Path('outputs/qc/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_summary.json'), 'next': Path('analysis_specifications/S008_confirmation_source_epoch_lineage_preprocessing.json'), 'contract': Path('config/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_contract.json'), 'report': Path('outputs/reports/confirmation_source_epoch_lineage_confirmation_source_epoch_lineage_report.md'), 'final_lock': Path('outputs/qc/confirmation_source_epoch_lineage_final_output_lock_hashes.json')}
raw_eeg_preprocessing_qc_REL = Path('src/analysis_pipeline/eeg/raw_eeg_preprocessing_qc/raw_eeg_preprocessing_qc.py')

def parse_args():
    p = argparse.ArgumentParser(description='Apply exact prespecified raw_eeg_preprocessing_qc preprocessing/QC to confirmation_source_epoch_lineage Confirmation epochs.')
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true', help='Replace phase-level metadata; valid per-source checkpoints are reused.')
    p.add_argument('--recompute-sources', action='store_true', help='Recompute completed large per-source stores after explicit review.')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()

def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path: Path, value: Any):
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + '', encoding='utf-8')
    os.replace(tmp, path)

def json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)

def pybool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {'true', '1', 'yes', 'y', 't'}

def require(paths: Iterable[Path]):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n  - ' + '\n  - '.join(missing))

def load_raw_eeg_preprocessing_qc(project: Path):
    path = project / raw_eeg_preprocessing_qc_REL
    require([path])
    observed = sha256_file(path)
    if observed != EXPECTED_raw_eeg_preprocessing_qc_SHA:
        raise RuntimeError(f'prespecified raw_eeg_preprocessing_qc source hash mismatch: expected {EXPECTED_raw_eeg_preprocessing_qc_SHA}, observed {observed}')
    spec = importlib.util.spec_from_file_location('prespecified_raw_eeg_preprocessing_qc_engine', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load prespecified raw_eeg_preprocessing_qc engine.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, 'POLICY_SHA256', None) != '1895a78035f536e73afd85e15e4b52b40fdc5b03bfd65048089e3b6ba87fa017':
        raise RuntimeError('prespecified raw_eeg_preprocessing_qc policy hash changed.')
    return (module, path)

def verify_checkpoint_full(source_dir: Path) -> bool:
    sp = source_dir / 'source_summary.json'
    if not sp.is_file():
        return False
    try:
        s = read_json(sp)
    except Exception:
        return False
    for row in s.get('stored_files', []):
        p = source_dir / row['relative_path']
        if not p.is_file() or int(p.stat().st_size) != int(row['size_bytes']) or sha256_file(p) != row['sha256']:
            return False
    return True

def load_and_verify_inputs(project: Path):
    paths = {k: project / v for k, v in O_REL.items()}
    require(paths.values())
    for key, expected in EXPECTED_O_HASHES.items():
        observed = sha256_file(paths[key])
        if observed != expected:
            raise RuntimeError(f'confirmation_source_epoch_lineage hash mismatch for {key}: expected {expected}, observed {observed}')
    summary = read_json(paths['summary'])
    policy = read_json(paths['next'])
    contract = read_json(paths['contract'])
    final_lock = read_json(paths['final_lock'])
    if final_lock.get('phase') != 'confirmation_source_epoch_lineage' or final_lock.get('hashes') != EXPECTED_O_HASHES:
        raise RuntimeError('confirmation_source_epoch_lineage final lock does not match the exact 14 prespecified output hashes.')
    if final_lock.get('eeg_amplitude_read') is not False or final_lock.get('reports_outcomes_read') is not False or final_lock.get('validation_targets_read') is not False:
        raise RuntimeError('confirmation_source_epoch_lineage boundary flags changed in final lock.')
    if summary.get('phase') != 'confirmation_source_epoch_lineage' or not summary.get('decision', {}).get('confirmation_preprocessing_qc_preprocessing_permitted'):
        raise RuntimeError('confirmation_source_epoch_lineage did not permit confirmation_preprocessing_qc.')
    if policy.get('next_phase') != 'confirmation_preprocessing_qc' or not policy.get('next_phase_permitted'):
        raise RuntimeError('prespecified confirmation_source_epoch_lineage next-phase policy is not valid for confirmation_preprocessing_qc.')
    if contract.get('transfer_policy_canonical_sha256') != validation_confirmation_synthesis_TRANSFER_POLICY_SHA:
        raise RuntimeError('validation_confirmation_synthesis transfer-policy lineage mismatch.')
    if contract.get('attachment_policy_canonical_sha256') != validation_confirmation_synthesis_ATTACHMENT_POLICY_SHA:
        raise RuntimeError('validation_confirmation_synthesis attachment-policy lineage mismatch.')
    if contract.get('exact_trial_key_specification_sha256') != validation_confirmation_synthesis_EXACT_TRIAL_KEY_SHA:
        raise RuntimeError('validation_confirmation_synthesis exact trial-key lineage mismatch.')
    if contract.get('source_code_lineage', {}).get('raw_eeg_preprocessing_qc') != EXPECTED_raw_eeg_preprocessing_qc_SHA:
        raise RuntimeError('confirmation_source_epoch_lineage contract expected a different raw_eeg_preprocessing_qc engine.')
    epochs = pd.read_csv(paths['epochs'], low_memory=False)
    headers = pd.read_csv(paths['headers'], low_memory=False)
    trials = pd.read_csv(paths['trials'], low_memory=False)
    coverage = pd.read_csv(paths['coverage'], low_memory=False)
    if len(epochs) != EXPECTED_O_COUNTS['epoch_rows'] or len(trials) != EXPECTED_O_COUNTS['trials']:
        raise RuntimeError('confirmation_source_epoch_lineage prespecified row counts changed.')
    if epochs['subject_id'].nunique() != EXPECTED_O_COUNTS['subjects'] or headers['raw_source_path'].nunique() != EXPECTED_O_COUNTS['raw_sources']:
        raise RuntimeError('confirmation_source_epoch_lineage subject/source counts changed.')
    available = epochs.loc[epochs['epoch_available'].map(pybool)].copy()
    if len(available) != EXPECTED_O_COUNTS['available_rows']:
        raise RuntimeError('confirmation_source_epoch_lineage available epoch count changed.')
    if available['epoch_id'].duplicated().any():
        raise RuntimeError('Available epoch IDs are not unique.')
    if not available['behavioral_values_used'].map(lambda x: not pybool(x)).all():
        raise RuntimeError('Behavioral values were marked used.')
    if not available['eeg_amplitude_read'].map(lambda x: not pybool(x)).all():
        raise RuntimeError('confirmation_source_epoch_lineage unexpectedly marked amplitude read.')
    if set(available['epoch_family']) != {'stimulus_early', 'stimulus_late', 'rating_early'}:
        raise RuntimeError('Epoch families changed.')
    if not coverage['branch_permitted_for_confirmation_preprocessing_qc'].map(pybool).all():
        raise RuntimeError('Not all prespecified branches were permitted.')
    forbidden_fragments = ('valence', 'liking', 'arousal', 'joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'familiarity', 'emotion_category', 'stimulus_id')
    bad = [c for c in available.columns if any((x in c.lower() for x in forbidden_fragments))]
    if bad:
        raise RuntimeError(f'Forbidden report/outcome columns in confirmation_source_epoch_lineage epoch manifest: {bad}')
    available['raw_run'] = available['raw_run'].astype(int)
    available['start_sample_nearest'] = available['start_sample_nearest'].astype(int)
    available['stop_sample_nearest_exclusive'] = available['stop_sample_nearest_exclusive'].astype(int)
    available['expected_output_samples'] = available['expected_output_samples'].astype(int)
    return (paths, summary, policy, contract, available, headers, trials)

def confirmation_retention(epoch_qc: pd.DataFrame):
    rows = []
    subrows = []
    for family, g in epoch_qc.groupby('epoch_family', sort=True):
        eligible = g['modeling_eligible'].map(pybool)
        context_ret = {}
        for context, cg in g.groupby('context', sort=True):
            context_ret[str(context)] = float(cg['modeling_eligible'].map(pybool).mean())
        for subject, sg in g.groupby('subject_id', sort=True):
            ok = sg['modeling_eligible'].map(pybool)
            imagery = int((ok & sg['context'].eq('imagery')).sum())
            video = int((ok & sg['context'].eq('video')).sum())
            total = int(ok.sum())
            subrows.append({'subject_id': subject, 'epoch_family': family, 'available_before_qc': int(len(sg)), 'modeling_eligible': total, 'imagery_eligible': imagery, 'video_eligible': video, 'raw_eeg_preprocessing_qc_within_person_minimum_met': bool(total >= 16 and imagery >= 6 and (video >= 6))})
        rows.append({'epoch_family': family, 'available_before_amplitude_qc': int(len(g)), 'modeling_eligible_rows': int(eligible.sum()), 'overall_retention': float(eligible.mean()), 'raw_eeg_preprocessing_qc_minimum_overall_retention': 0.75, 'imagery_retention': float(context_ret.get('imagery', math.nan)), 'video_retention': float(context_ret.get('video', math.nan)), 'raw_eeg_preprocessing_qc_minimum_context_retention': 0.65, 'retention_thresholds_passed': bool(float(eligible.mean()) >= 0.75 and set(context_ret) == {'imagery', 'video'} and (min(context_ret.values()) >= 0.65)), 'threshold_changed_after_amplitude': False})
    return (pd.DataFrame(rows), pd.DataFrame(subrows))

def trajectory_availability(epoch_qc: pd.DataFrame, trials: pd.DataFrame):
    x = epoch_qc[['raw_behavior_row_id', 'subject_id', 'epoch_family', 'modeling_eligible']].copy()
    x['modeling_eligible'] = x['modeling_eligible'].map(pybool)
    wide = x.pivot(index=['raw_behavior_row_id', 'subject_id'], columns='epoch_family', values='modeling_eligible').reset_index()
    for fam in ['stimulus_early', 'stimulus_late', 'rating_early']:
        if fam not in wide:
            wide[fam] = False
        wide[fam] = wide[fam].fillna(False).astype(bool)
    wide['complete_three_epoch_trajectory'] = wide[['stimulus_early', 'stimulus_late', 'rating_early']].all(axis=1)
    base = trials[['raw_behavior_row_id', 'subject_id', 'trial_number']].copy()
    out = base.merge(wide, on=['raw_behavior_row_id', 'subject_id'], how='left', validate='one_to_one')
    for c in ['stimulus_early', 'stimulus_late', 'rating_early', 'complete_three_epoch_trajectory']:
        out[c] = out[c].fillna(False).astype(bool)
    subject = out.groupby('subject_id', sort=True).agg(authoritative_trials=('raw_behavior_row_id', 'size'), complete_three_epoch_trajectories=('complete_three_epoch_trajectory', 'sum')).reset_index()
    subject['minimum_required'] = TRANSFER_GATES['minimum_complete_trajectories_per_participant']
    subject['eligible_for_confirmation_feature_trajectory'] = subject['complete_three_epoch_trajectories'] >= subject['minimum_required']
    complete = int(out['complete_three_epoch_trajectory'].sum())
    fraction = complete / EXPECTED_O_COUNTS['trials']
    eligible_subjects = int(subject['eligible_for_confirmation_feature_trajectory'].sum())
    gate = pd.DataFrame([{'gate_id': 'K1', 'metric': 'eligible_participants', 'observed': eligible_subjects, 'threshold': 9, 'comparison': '>=', 'pass': eligible_subjects >= 9}, {'gate_id': 'K2', 'metric': 'complete_trajectory_fraction_of_494', 'observed': fraction, 'threshold': 0.75, 'comparison': '>=', 'pass': fraction >= 0.75}, {'gate_id': 'K3', 'metric': 'source_and_algorithm_hash_gates', 'observed': 1, 'threshold': 1, 'comparison': '==', 'pass': True}, {'gate_id': 'K4', 'metric': 'reports_outcomes_read', 'observed': 0, 'threshold': 0, 'comparison': '==', 'pass': True}])
    return (out, subject, gate)

def self_test(project: Path | None=None):
    q = pd.DataFrame([{'raw_behavior_row_id': 'a', 'subject_id': 's1', 'epoch_family': f, 'modeling_eligible': True, 'context': 'imagery'} for f in ['stimulus_early', 'stimulus_late', 'rating_early']] + [{'raw_behavior_row_id': 'b', 'subject_id': 's1', 'epoch_family': f, 'modeling_eligible': True, 'context': 'video'} for f in ['stimulus_early', 'stimulus_late', 'rating_early']])
    t = pd.DataFrame({'raw_behavior_row_id': ['a', 'b'], 'subject_id': ['s1', 's1'], 'trial_number': [1, 2]})
    o, s, g = trajectory_availability(q, t)
    assert int(o.complete_three_epoch_trajectory.sum()) == 2 and int(s.complete_three_epoch_trajectories.iloc[0]) == 2
    if project is not None and (project / raw_eeg_preprocessing_qc_REL).is_file():
        m, _ = load_raw_eeg_preprocessing_qc(project)
        rng = np.random.default_rng(540)
        x = rng.normal(0, 2e-06, size=(64, 1000))
        rec = np.ones(64, dtype=bool)
        y, mask, metrics = m.iterative_reference_and_qc(x, rec)
        assert y.shape == x.shape and mask.shape == (64,) and ('epoch_passed' in metrics)
    print('confirmation_preprocessing_qc self-test: PASS')

def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    started = time.time()
    if args.self_test:
        self_test(project if project.exists() else None)
        return 0
    paths, j_summary, j_policy, j_contract, available, headers, trials = load_and_verify_inputs(project)
    raw_eeg_preprocessing_qc, raw_eeg_preprocessing_qc_path = load_raw_eeg_preprocessing_qc(project)
    if raw_eeg_preprocessing_qc.POLICY_SHA256 != '1895a78035f536e73afd85e15e4b52b40fdc5b03bfd65048089e3b6ba87fa017':
        raise RuntimeError('raw_eeg_preprocessing_qc policy mismatch.')
    epoch_hash = sha256_file(paths['epochs'])
    store_root = project / STORE_REL
    store_root.mkdir(parents=True, exist_ok=True)
    tables = project / 'outputs/tables'
    qc = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config = project / 'config'
    logs = project / 'outputs/logs'
    out = {'recording_qc': qc / f'{PREFIX}_confirmation_recording_channel_qc.csv', 'epoch_qc': tables / f'{PREFIX}_confirmation_epoch_amplitude_qc.csv', 'inventory': tables / f'{PREFIX}_confirmation_large_array_inventory.csv', 'branch_qc': qc / f'{PREFIX}_confirmation_branch_retention_qc.csv', 'subject_branch_qc': qc / f'{PREFIX}_confirmation_subject_branch_retention_qc.csv', 'trajectory_manifest': tables / f'{PREFIX}_confirmation_post_qc_trajectory_availability.csv', 'subject_trajectory_qc': qc / f'{PREFIX}_confirmation_subject_trajectory_availability.csv', 'transfer_gate': qc / f'{PREFIX}_confirmation_pre_feature_transfer_gate.csv', 'source_rehash': qc / f'{PREFIX}_confirmation_source_rehash.csv', 'summary': qc / f'{PREFIX}_confirmation_preprocessing_qc_summary.json', 'contract': config / f'{PREFIX}_confirmation_preprocessing_qc_contract.json', 'next': config / f'{PREFIX}_prespecified_confirmation_feature_trajectory_feature_extraction_policy.json', 'final_lock': qc / f'{PREFIX}_final_output_lock_hashes.json', 'report': reports / f'{PREFIX}_confirmation_preprocessing_qc_report.md', 'log': logs / f'{PREFIX}_confirmation_preprocessing_qc.log'}
    existing = [p for p in out.values() if p.exists()]
    if existing and (not args.force):
        raise RuntimeError('Phase outputs exist; use --force after review:\n  - ' + '\n  - '.join(map(str, existing)))
    header_lookup = headers.set_index('raw_source_path')
    source_dirs = []
    with out['log'].open('w', encoding='utf-8') as log:

        def note(msg):
            line = f'[{now_iso()}] {msg}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        groups = list(available.groupby('raw_source_path', sort=True))
        note(f'Verified confirmation_source_epoch_lineage, all prespecified hashes, and exact raw_eeg_preprocessing_qc engine {EXPECTED_raw_eeg_preprocessing_qc_SHA}.')
        note(f'Processing or reusing {len(groups)} Confirmation source checkpoints; reports/outcomes, semantic targets, and Validation targets remain unread.')
        for i, (source_text, group) in enumerate(groups, 1):
            if source_text not in header_lookup.index:
                raise RuntimeError(f'No prespecified header row for {source_text}')
            note(f'[{i}/{len(groups)}] {source_text} ({len(group)} available epochs).')
            d = raw_eeg_preprocessing_qc.process_source(group, header_lookup.loc[source_text], store_root=store_root, epoch_manifest_sha256=epoch_hash, recompute=args.recompute_sources)
            if not verify_checkpoint_full(d):
                raise RuntimeError(f'Completed source checkpoint failed full stored-file SHA audit: {d}')
            source_dirs.append(d)
        recording, epoch_qc, inventory = raw_eeg_preprocessing_qc.collect_source_outputs(source_dirs)
        if len(recording) != len(groups) * raw_eeg_preprocessing_qc.CHANNEL_COUNT:
            raise RuntimeError('Recording QC row count incomplete.')
        if len(epoch_qc) != len(available) or set(epoch_qc.epoch_id) != set(available.epoch_id):
            raise RuntimeError('Epoch QC set differs from prespecified available epochs.')
        recording.to_csv(out['recording_qc'], index=False)
        epoch_qc.to_csv(out['epoch_qc'], index=False)
        inventory.to_csv(out['inventory'], index=False)
        branch, subbranch = confirmation_retention(epoch_qc)
        branch.to_csv(out['branch_qc'], index=False)
        subbranch.to_csv(out['subject_branch_qc'], index=False)
        traj, subtraj, gate = trajectory_availability(epoch_qc, trials)
        traj.to_csv(out['trajectory_manifest'], index=False)
        subtraj.to_csv(out['subject_trajectory_qc'], index=False)
        gate.to_csv(out['transfer_gate'], index=False)
        rehash = []
        for source_dir in source_dirs:
            ss = read_json(source_dir / 'source_summary.json')
            p = Path(ss['raw_source_path'])
            rehash.append({'raw_source_path': ss['raw_source_path'], 'size_bytes': int(p.stat().st_size), 'mtime_ns': int(p.stat().st_mtime_ns), 'sha256': ss['raw_source_sha256'], 'hash_source': 'exact_prespecified_raw_eeg_preprocessing_qc_engine_during_source_processing'})
        pd.DataFrame(rehash).sort_values('raw_source_path').to_csv(out['source_rehash'], index=False)
        complete = int(traj.complete_three_epoch_trajectory.sum())
        eligible_sub = int(subtraj.eligible_for_confirmation_feature_trajectory.sum())
        permitted = bool(gate['pass'].map(pybool).all())
        decision = {'all_source_checkpoints_complete': len(source_dirs) == len(groups), 'exact_prespecified_raw_eeg_preprocessing_qc_engine_used': True, 'raw_eeg_preprocessing_qc_policy_sha256': raw_eeg_preprocessing_qc.POLICY_SHA256, 'all_stored_files_sha256_verified': True, 'threshold_changed_after_amplitude': False, 'epoch_window_changed_after_amplitude': False, 'automatic_ica_removal_used': False, 'channel_interpolation_used': False, 'reports_outcomes_read': False, 'stimulus_identity_read': False, 'released_derivative_amplitude_read': False, 'confirmation_opened_for_preprocessing_only': True, 'confirmation_reports_outcomes_read': False, 'validation_targets_read': False, 'confirmation_feature_trajectory_feature_extraction_permitted': permitted}
        summary = {'generated_at': now_iso(), 'phase': PHASE, 'status': 'confirmation_exact_raw_eeg_preprocessing_qc_preprocessing_and_amplitude_qc_complete_candidate', 'validation_confirmation_synthesis_attachment_policy_canonical_sha256': validation_confirmation_synthesis_ATTACHMENT_POLICY_SHA, 'validation_confirmation_synthesis_exact_trial_key_specification_sha256': validation_confirmation_synthesis_EXACT_TRIAL_KEY_SHA, 'validation_confirmation_synthesis_transfer_policy_canonical_sha256': validation_confirmation_synthesis_TRANSFER_POLICY_SHA, 'raw_eeg_preprocessing_qc_source_code_sha256': EXPECTED_raw_eeg_preprocessing_qc_SHA, 'raw_eeg_preprocessing_qc_policy_sha256': raw_eeg_preprocessing_qc.POLICY_SHA256, 'counts': {'confirmation_subjects': 12, 'authoritative_trials': 494, 'raw_edf_sources': len(groups), 'available_epochs_before_amplitude_qc': len(available), 'modeling_eligible_epochs': int(epoch_qc.modeling_eligible.map(pybool).sum()), 'complete_three_epoch_trajectories': complete, 'complete_trajectory_fraction_of_494': complete / 494, 'eligible_participants_for_confirmation_feature_trajectory': eligible_sub, 'recording_channel_qc_rows': len(recording), 'large_store_files': len(inventory), 'large_store_bytes': int(inventory.size_bytes.sum())}, 'branch_retention': branch.to_dict('records'), 'transfer_gate': gate.to_dict('records'), 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        atomic_json(out['summary'], summary)
        next_policy = {'generated_at': now_iso(), 'phase': PHASE, 'next_phase': 'confirmation_feature_trajectory', 'next_phase_permitted': permitted, 'next_phase_role': 'extract exact prespecified outcome_free_features outcome-free compact features from confirmation_preprocessing_qc modeling-eligible Confirmation epochs and construct three-epoch trajectories; do not read reports/outcomes/context targets, stimulus identity, or Validation targets', 'required_raw_eeg_preprocessing_qc_source_code_sha256': EXPECTED_raw_eeg_preprocessing_qc_SHA, 'required_outcome_free_features_source_code_sha256': 'd12ca5779f8c285d8194abe5609c0aa509278d58699b8f1e20d8586c4d01bcba', 'required_inputs': {'epoch_qc': sha256_file(out['epoch_qc']), 'inventory': sha256_file(out['inventory']), 'trajectory_manifest': sha256_file(out['trajectory_manifest']), 'subject_trajectory_qc': sha256_file(out['subject_trajectory_qc']), 'transfer_gate': sha256_file(out['transfer_gate'])}, 'large_store_root': str(store_root), 'required_boundary': ['verify every confirmation_preprocessing_qc hash', 'use only modeling_eligible epoch rows', 'read arrays only through fixed inventory', 'apply exact prespecified outcome_free_features feature schema', 'do not read reports/outcomes/stimulus identity', 'context must not be used as a target', 'Validation targets remain unread'], 'confirmation_opened_for_feature_extraction_only': False, 'confirmation_reports_outcomes_read': False, 'validation_targets_read': False}
        atomic_json(out['next'], next_policy)
        primary = ['recording_qc', 'epoch_qc', 'inventory', 'branch_qc', 'subject_branch_qc', 'trajectory_manifest', 'subject_trajectory_qc', 'transfer_gate', 'source_rehash', 'summary', 'next']
        hashes = {k: sha256_file(out[k]) for k in primary}
        contract = {**summary, 'exact_engine_adapter': {'source_script': str(raw_eeg_preprocessing_qc_path), 'source_code_sha256': EXPECTED_raw_eeg_preprocessing_qc_SHA, 'algorithm_changes': False, 'allowed_adapter_changes': ['Confirmation cohort selection', 'confirmation_preprocessing_qc output namespace', 'prespecified expected counts', 'validation_confirmation_synthesis transfer-availability continuation gate']}, 'fixed_raw_eeg_preprocessing_qc_policy': raw_eeg_preprocessing_qc.FIXED_POLICY, 'confirmation_transfer_gates': TRANSFER_GATES, 'large_store_root': str(store_root), 'large_arrays_embedded_in_release': False, 'resume_policy': 'valid per-source checkpoints reused only after full stored-file SHA audit; --recompute-sources rebuilds them', 'input_hashes': {k: sha256_file(v) for k, v in paths.items()}, 'output_hashes': hashes, 'source_dataset_modified': False, 'confirmation_reports_outcomes_read': False, 'validation_targets_read': False}
        atomic_json(out['contract'], contract)
        report = ['# Phase confirmation_preprocessing_qc — Confirmation preprocessing and amplitude QC', '', f"- Generated: `{summary['generated_at']}`", f'- Exact raw_eeg_preprocessing_qc engine SHA-256: `{EXPECTED_raw_eeg_preprocessing_qc_SHA}`', f'- Raw EDF sources: **{len(groups)}**', f'- Available epochs before amplitude QC: **{len(available)}**', f"- Modeling-eligible epochs: **{summary['counts']['modeling_eligible_epochs']}**", f'- Complete three-epoch trajectories: **{complete}/494 ({complete / 494:.4f})**', f'- Eligible participants with at least 16 complete trajectories: **{eligible_sub}/12**', f'- confirmation_feature_trajectory permitted: **{permitted}**', '', '## Branch retention', '', '```text', branch.to_string(index=False), '```', '', '## Transfer-availability gate', '', '```text', gate.to_string(index=False), '```', '', '## Boundary', '', '- Reports/outcomes read: **no**', '- Stimulus identity read: **no**', '- Context used as a target: **no**', '- Confirmation opened for amplitude preprocessing only: **yes**', '- Confirmation reports/outcomes read: **no**', '- Validation targets read: **no**', '- Thresholds/windows changed after amplitude: **no**', '']
        out['report'].write_text(''.join(report) + '', encoding='utf-8')
        final_hashes = {**hashes, 'contract': sha256_file(out['contract']), 'report': sha256_file(out['report'])}
        atomic_json(out['final_lock'], {'generated_at': now_iso(), 'phase': PHASE, 'hashes': final_hashes, 'raw_eeg_preprocessing_qc_source_code_sha256': EXPECTED_raw_eeg_preprocessing_qc_SHA, 'raw_eeg_preprocessing_qc_policy_sha256': raw_eeg_preprocessing_qc.POLICY_SHA256, 'reports_outcomes_read': False, 'confirmation_reports_outcomes_read': False, 'validation_targets_read': False})
        note(f'confirmation_preprocessing_qc complete. Complete trajectories={complete}/494; eligible participants={eligible_sub}/12; confirmation_feature_trajectory permitted={permitted}.')
    files = list(out.values()) + [raw_eeg_preprocessing_qc_path]
    return 0
if __name__ == '__main__':
    raise SystemExit(main())