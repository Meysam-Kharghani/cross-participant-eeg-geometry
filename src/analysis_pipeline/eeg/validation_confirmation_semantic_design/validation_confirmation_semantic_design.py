# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
Phase validation_confirmation_semantic_design — Validation synthesis + Confirmation semantic/report analysis specification.

Recovery build after invalid summary-only validation_confirmation_semantic_design/confirmation_semantic_report_execution artifacts.

This phase:
- verifies the exact validation_semantic_report_execution final lock and every fixed child;
- verifies the validation_semantic_report_execution→validation_confirmation_semantic_design authorization policy;
- verifies the prespecified semantic_report_execution_design semantic/report program;
- verifies the exact discovery_semantic_report_execution numerical implementation identity;
- verifies the immutable Confirmation structural families/signatures (confirmation_stimulus_experience_multiresolution_transfer/confirmation_report_transition_multiresolution_transfer);
- specifications Confirmation target-lineage references from semantic_report_execution_design inventories;
- synthesizes Discovery + Validation parent-test results for audit only;
- requires all nine pre-prespecified tests in confirmation_semantic_report_execution regardless of prior p-values;
- opens/parses NO Confirmation behavioral/report/category/context values;
- opens no new EEG signal arrays.

It creates a real validation_confirmation_semantic_design final lock and a validation_confirmation_semantic_design→confirmation_semantic_report_execution execution policy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
PHASE = 'validation_confirmation_semantic_design'
NEXT_PHASE = 'confirmation_semantic_report_execution'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
REQUIRED_TESTS = ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']
EXPERIENCE_K = [2, 3, 4]
TRANSITION_K = [2, 3]
validation_semantic_report_execution_OUT = Path('outputs/interpretability/validation_semantic_report_execution_validation_semantic_report_execution')
validation_semantic_report_execution_LOCK = validation_semantic_report_execution_OUT / 'qc/validation_semantic_report_execution_final_output_lock_hashes.json'
validation_semantic_report_execution_LOCK_SHA256 = '51ba5dfa1cd304211cea3093a0a3ab68621852d942920e254adfe34c3410230c'
validation_semantic_report_execution_NEXT = Path('analysis_specifications/S088_validation_semantic_report_confirmation.json')
validation_semantic_report_execution_NEXT_SHA256 = '2816a1442da45806e270f4cca179014e2114c751e2233f218698d116337eaace'
validation_semantic_report_execution_REGISTRY = validation_semantic_report_execution_OUT / 'tables/validation_semantic_report_execution_validation_parent_test_registry.csv'
validation_semantic_report_execution_REGISTRY_SHA256 = '2e707cbfbbd560b63fb971709b7270a5d09ebf7e7995d1c08a5c61458ef1a9b4'
validation_semantic_report_execution_NUMERIC_ID = validation_semantic_report_execution_OUT / 'qc/validation_semantic_report_execution_numeric_implementation_identity.json'
validation_semantic_report_execution_NUMERIC_ID_SHA256 = '3fe3077a8e1c1c82d62db8dd3954790839fa75efc59917d4f928eaa18769d793'
discovery_semantic_report_execution_OUT = Path('outputs/interpretability/discovery_semantic_report_execution_discovery_semantic_report_execution')
discovery_semantic_report_execution_LOCK = discovery_semantic_report_execution_OUT / 'qc/discovery_semantic_report_execution_final_output_lock_hashes.json'
discovery_semantic_report_execution_LOCK_SHA256 = '4ace7a17facf24471572a6a05adc2e6ce55082e1f0890b882c938150f2ed4b59'
discovery_semantic_report_execution_NEXT = Path('analysis_specifications/S025_discovery_semantic_report_validation.json')
discovery_semantic_report_execution_NEXT_SHA256 = 'e0d3869a0098545390c7a94918f11238a73a1ab505fe668d46c6b1a5e002abf1'
discovery_semantic_report_execution_REGISTRY = discovery_semantic_report_execution_OUT / 'tables/discovery_semantic_report_execution_discovery_parent_test_registry.csv'
discovery_semantic_report_execution_REGISTRY_SHA256 = '926db0fde8c7da05d4b925b7d819051a2635ca088fe40448f623ad1478df7c26'
discovery_semantic_report_execution_NUMERIC_specification = discovery_semantic_report_execution_OUT / 'qc/discovery_semantic_report_execution_numeric_implementation_specification.json'
discovery_semantic_report_execution_NUMERIC_specification_SHA256 = 'f7747503f9e03b80560d20481e76be7301a9b1f703a986f9d244d7f308bd9d57'
discovery_semantic_report_execution_SOURCE = Path('src/analysis_pipeline/eeg/discovery_semantic_report_execution/discovery_semantic_report_execution.py')
discovery_semantic_report_execution_SOURCE_SHA256 = 'caa62ecc3d5dfe825cd166d8e68078fd1fe5c185c81eb1caef8701f1cacbf29e'
semantic_report_execution_design_OUT = Path('outputs/interpretability/semantic_report_execution_design_semantic_report_execution_design_specification')
semantic_report_execution_design_LOCK = semantic_report_execution_design_OUT / 'qc/semantic_report_execution_design_final_output_lock_hashes.json'
semantic_report_execution_design_LOCK_SHA256 = '63b1a0f3ea555b0e03067a6ad324d8ec733baa9a34b700b05305ba72967fce25'
semantic_report_execution_design_PROGRAM = Path('analysis_specifications/S063_semantic_report.json')
semantic_report_execution_design_PROGRAM_SHA256 = 'b9b6800c4ee6bf22033b81c41031d5d805dc57c5020b6eda21d6d931cc5b28eb'
semantic_report_execution_design_RAW_INVENTORY = semantic_report_execution_design_OUT / 'tables/semantic_report_execution_design_heldout_raw_source_inventory.csv'
semantic_report_execution_design_RAW_INVENTORY_SHA256 = 'effa5dd59a8d32e5a335db18ebb33df0790f2fce282b23d622e9d3e91c57d704'
semantic_report_execution_design_REF_INVENTORY = semantic_report_execution_design_OUT / 'tables/semantic_report_execution_design_source_reference_inventory.csv'
semantic_report_execution_design_REF_INVENTORY_SHA256 = '911a4f501f3dce6831ecba28da1c00fac65535083cded416d8a2bc9893202318'
confirmation_stimulus_experience_multiresolution_transfer_OUT = Path('outputs/interpretability/confirmation_stimulus_experience_multiresolution_transfer_confirmation_stimulus_experience_multiresolution_transfer')
confirmation_stimulus_experience_multiresolution_transfer_LOCK = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'qc/confirmation_stimulus_experience_multiresolution_transfer_final_output_lock_hashes.json'
confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256 = '3b6f03c0d44903345871470cc40e35b0d9d0aaa9d836954e1c0009a7485b6b8b'
CONF_EXP_ASSIGN = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_family_assignments_all_k.csv'
CONF_EXP_ASSIGN_SHA256 = '2159d0e73b3cc4333a733b1eed28634a80269420d7397f73b38ebd868f0736ed'
CONF_EXP_SIG = confirmation_stimulus_experience_multiresolution_transfer_OUT / 'tables/confirmation_stimulus_experience_multiresolution_transfer_confirmation_experience_signatures.csv'
CONF_EXP_SIG_SHA256 = '99f9a171c75fdd6680e96cf08c91e73ca9abba4ee2c9b71a09fac1837d962c5d'
confirmation_report_transition_multiresolution_transfer_OUT = Path('outputs/interpretability/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_multiresolution_transfer')
confirmation_report_transition_multiresolution_transfer_LOCK = confirmation_report_transition_multiresolution_transfer_OUT / 'qc/confirmation_report_transition_multiresolution_transfer_final_output_lock_hashes.json'
confirmation_report_transition_multiresolution_transfer_LOCK_SHA256 = '14bb0cb021ae071204c90e5be71959bacbc6fc2259e409ed111fc299578d9375'
CONF_TR_ASSIGN = confirmation_report_transition_multiresolution_transfer_OUT / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_transition_family_assignments_all_k.csv'
CONF_TR_ASSIGN_SHA256 = 'a21018d812c13912ddd7feeebf1902bed6f71e57ebf508158114c3f475feda0b'
CONF_TR_SIG = confirmation_report_transition_multiresolution_transfer_OUT / 'tables/confirmation_report_transition_multiresolution_transfer_confirmation_report_transition_signatures.csv'
CONF_TR_SIG_SHA256 = '52e7967d864bd2a3d21fb7a59a59fe83bde10fd9f5530da747bbf3faddfe1fb6'
confirmation_source_epoch_lineage_LOCK = Path('outputs/qc/confirmation_source_epoch_lineage_final_output_lock_hashes.json')
confirmation_source_epoch_lineage_LOCK_SHA256 = 'd502dbc7eb5557d33a6e8e23f54127271b136180100704de477b5951b5e91f82'
CONF_TIMING = Path('outputs/tables/confirmation_source_epoch_lineage_confirmation_authoritative_trial_timing_manifest.csv')
CONF_TIMING_SHA256 = 'dcb34cdd525faffb9fc38d865589d598d436a8429f225554290ae5c731dd693f'
CANONICAL = Path('data/canonical_behavioral_table.csv')
CANONICAL_SHA256 = 'e31f96f918e806d466fd2bccc85b94d5c9dc96002bf97a97a442e8059a0365ed'
ANALYSIS_specification = Path('config/analysis_specification.v1.json')
ANALYSIS_specification_SHA256 = 'cfd175b2ef3de731355d7378650ffa1b5d31a06171f4600a379c504fe2a85dfa'
PARTICIPANT_SPLIT = Path('config/participant_split_manifest.v1.csv')
PARTICIPANT_SPLIT_SHA256 = '0f6156e2a0de5af59dd3a7cd15b2a4a914dc0a0540b9d941fa6676a697f8d453'
README_SEQUENCE_CORE = Path('src/analysis_pipeline/data_preparation/event_grammar_reconstruction.py')
README_SEQUENCE_CORE_SHA256 = '39b690bdbd692cd5a8df6f370e7ad96ddff135ba9f96e6817c6e7e77c23474ac'
DATASET_README_SHA256 = '379cfc97fce08b4c54d3dc26e95c35d504ca130666bbcf02eff59638f0ff7e3a'
OUT = Path('outputs/interpretability/validation_confirmation_semantic_design_validation_synthesis_confirmation_design_specification')
NEXT_CONFIG = Path('analysis_specifications/S072_validation_confirmation_semantic_report.json')

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
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing {label}: {path}')
    got = sha256(path)
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

def verify_path_lock_children(project: Path, rel: Path, expected_sha: str, expected_phase: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p = project / rel
    verify(p, expected_sha, f'{expected_phase} final lock')
    obj = readj(p)
    if obj.get('phase') != expected_phase:
        raise RuntimeError(f'{expected_phase} final-lock phase drift')
    hashes = obj.get('hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f'{expected_phase} final lock has no child hashes')
    rows: list[dict[str, Any]] = []
    for child_rel, expected in sorted(hashes.items()):
        if '/' not in str(child_rel):
            raise RuntimeError(f'{expected_phase} uses non-path child key unexpectedly: {child_rel}')
        child = project / child_rel
        got = verify(child, str(expected), f'{expected_phase} fixed child {child_rel}')
        rows.append({'source_phase': expected_phase, 'path': str(child_rel), 'expected_sha256': str(expected), 'observed_sha256': got, 'verified': True, 'content_values_parsed_in_validation_confirmation_semantic_design': False})
    return (obj, rows)

def audit_policy_and_boundaries(validation_semantic_report_execution_lock: dict[str, Any], validation_semantic_report_execution_next: dict[str, Any], discovery_semantic_report_execution_lock: dict[str, Any], semantic_report_execution_design_lock: dict[str, Any], confirmation_stimulus_experience_multiresolution_transfer_lock: dict[str, Any], confirmation_report_transition_multiresolution_transfer_lock: dict[str, Any], confirmation_source_epoch_lineage_lock: dict[str, Any]) -> None:
    if validation_semantic_report_execution_lock.get('next_phase') != PHASE:
        raise RuntimeError('validation_semantic_report_execution does not authorize validation_confirmation_semantic_design')
    if validation_semantic_report_execution_lock.get('Confirmation_behavioral_values_read') is not False:
        raise RuntimeError('validation_semantic_report_execution Confirmation boundary drift')
    if validation_semantic_report_execution_lock.get('Validation_results_select_Confirmation_tests') is not False:
        raise RuntimeError('validation_semantic_report_execution result-selection boundary drift')
    if validation_semantic_report_execution_lock.get('new_EEG_signal_values_opened') is not False:
        raise RuntimeError('validation_semantic_report_execution new-EEG boundary drift')
    if validation_semantic_report_execution_next.get('phase') != 'validation_semantic_report_execution' or validation_semantic_report_execution_next.get('next_phase') != PHASE:
        raise RuntimeError('validation_semantic_report_execution→validation_confirmation_semantic_design policy authorization drift')
    if validation_semantic_report_execution_next.get('Confirmation_behavior_opening_permitted_in_validation_confirmation_semantic_design') is not False:
        raise RuntimeError('validation_semantic_report_execution policy unexpectedly permits Confirmation behavior in validation_confirmation_semantic_design')
    if validation_semantic_report_execution_next.get('Confirmation_behavior_opening_may_be_prespecified_for_confirmation_semantic_report_execution') is not True:
        raise RuntimeError('validation_semantic_report_execution policy does not permit freezing confirmation_semantic_report_execution opening')
    if validation_semantic_report_execution_next.get('Validation_results_select_Confirmation_tests') is not False:
        raise RuntimeError('validation_semantic_report_execution policy permits Validation-driven test selection')
    if validation_semantic_report_execution_next.get('all_prespecified_tests_required_in_Confirmation') != REQUIRED_TESTS:
        raise RuntimeError('validation_semantic_report_execution Confirmation test registry drift')
    if validation_semantic_report_execution_next.get('new_EEG_signal_opening_permitted') is not False:
        raise RuntimeError('validation_semantic_report_execution policy unexpectedly permits new EEG')
    if validation_semantic_report_execution_next.get('canonical_context_stimulus_category_permitted') is not False:
        raise RuntimeError('validation_semantic_report_execution policy unexpectedly permits canonical context/category')
    if validation_semantic_report_execution_next.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('validation_semantic_report_execution policy unexpectedly permits semantic relabeling')
    if validation_semantic_report_execution_next.get('distortion_interpretation_permitted') is not False:
        raise RuntimeError('validation_semantic_report_execution policy unexpectedly permits distortion interpretation')
    if discovery_semantic_report_execution_lock.get('phase') != 'discovery_semantic_report_execution' or discovery_semantic_report_execution_lock.get('next_phase') != 'validation_semantic_report_execution':
        raise RuntimeError('discovery_semantic_report_execution chain drift')
    if semantic_report_execution_design_lock.get('phase') != 'semantic_report_execution_design' or semantic_report_execution_design_lock.get('next_phase') != 'discovery_semantic_report_execution':
        raise RuntimeError('semantic_report_execution_design chain drift')
    if confirmation_stimulus_experience_multiresolution_transfer_lock.get('passing_K') != EXPERIENCE_K:
        raise RuntimeError('confirmation_stimulus_experience_multiresolution_transfer Confirmation experience K drift')
    for k in ['Confirmation_behavioral_values_read', 'K_reselected', 'centroids_refit', 'thresholds_tuned', 'semantic_relabeling_permitted']:
        if bool(confirmation_stimulus_experience_multiresolution_transfer_lock.get(k)):
            raise RuntimeError(f'confirmation_stimulus_experience_multiresolution_transfer structural boundary violation: {k}=True')
    if confirmation_report_transition_multiresolution_transfer_lock.get('passing_K') != TRANSITION_K:
        raise RuntimeError('confirmation_report_transition_multiresolution_transfer Confirmation transition K drift')
    for k in ['Confirmation_behavioral_values_read', 'K_reselected', 'centroids_refit', 'thresholds_tuned', 'semantic_relabeling_permitted', 'distortion_interpretation_permitted']:
        if bool(confirmation_report_transition_multiresolution_transfer_lock.get(k)):
            raise RuntimeError(f'confirmation_report_transition_multiresolution_transfer structural boundary violation: {k}=True')
    if confirmation_source_epoch_lineage_lock.get('phase') != 'confirmation_source_epoch_lineage':
        raise RuntimeError('confirmation_source_epoch_lineage lineage phase drift')
    if confirmation_source_epoch_lineage_lock.get('reports_outcomes_read') is not False:
        raise RuntimeError('confirmation_source_epoch_lineage unexpectedly read reports/outcomes')
    if confirmation_source_epoch_lineage_lock.get('hashes', {}).get('trials') != CONF_TIMING_SHA256:
        raise RuntimeError('confirmation_source_epoch_lineage Confirmation timing hash drift')

def validate_semantic_report_execution_design_program(program: dict[str, Any]) -> None:
    if program.get('phase') != 'semantic_report_execution_design':
        raise RuntimeError('semantic_report_execution_design program phase drift')
    order = program.get('execution_order', [])
    expected = ['discovery_semantic_report_execution', 'validation_semantic_report_execution', 'validation_confirmation_semantic_design', 'confirmation_semantic_report_execution', 'cross_split_semantic_report_synthesis']
    got = [x.get('phase') for x in order]
    if got != expected:
        raise RuntimeError(f'semantic_report_execution_design execution order drift: {got}')
    absolute = program.get('absolute_lineage_rules', {})
    if 'validation_source_epoch_lineage/confirmation_source_epoch_lineage authoritative timing manifest' not in str(absolute.get('heldout_context', '')):
        raise RuntimeError('semantic_report_execution_design held-out context lineage drift')
    if 'Never use canonical stimulus_id/emotion_category' not in str(absolute.get('heldout_stimulus_category', '')):
        raise RuntimeError('semantic_report_execution_design held-out category prohibition drift')
    reports_rule = str(absolute.get('reports', ''))
    if 'Validation/Confirmation' not in reports_rule or 'ten report columns' not in reports_rule:
        raise RuntimeError('semantic_report_execution_design report-source rule drift')
    forbidden = ' | '.join(map(str, program.get('forbidden', []))).lower()
    for token in ['canonical context', 'post-hoc report subset', 'selecting one k', 'changing structural family ids', 'split-specific family relabeling', 'distortion']:
        if token not in forbidden:
            raise RuntimeError(f'semantic_report_execution_design forbidden registry missing token: {token}')

def load_parent_registry(project: Path, path: Path, expected_sha: str, split: str) -> pd.DataFrame:
    verify(project / path, expected_sha, f'{split} parent-test registry')
    df = pd.read_csv(project / path)
    need = ['family', 'test_id', 'p_raw', 'p_adjusted', 'reject_adjusted_0p05']
    if df.columns.tolist() != need:
        raise RuntimeError(f'{split} registry schema drift: {df.columns.tolist()}')
    if df['test_id'].tolist() != REQUIRED_TESTS:
        raise RuntimeError(f'{split} registry test order/content drift')
    df = df.copy()
    df.insert(0, 'split', split)
    df['required_in_confirmation'] = True
    df['selected_by_prior_results'] = False
    return df

def confirmation_source_specification(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    verify(project / semantic_report_execution_design_RAW_INVENTORY, semantic_report_execution_design_RAW_INVENTORY_SHA256, 'semantic_report_execution_design raw heldout inventory')
    verify(project / semantic_report_execution_design_REF_INVENTORY, semantic_report_execution_design_REF_INVENTORY_SHA256, 'semantic_report_execution_design source-reference inventory')
    raw = pd.read_csv(project / semantic_report_execution_design_RAW_INVENTORY)
    required_raw_cols = ['split', 'subject_id', 'role', 'path', 'size_bytes', 'sha256', 'values_parsed_in_semantic_report_execution_design']
    if raw.columns.tolist() != required_raw_cols:
        raise RuntimeError('semantic_report_execution_design raw inventory schema drift')
    conf_raw = raw[raw['split'].astype(str).str.lower().eq('confirmation')].copy()
    if len(conf_raw) != 24:
        raise RuntimeError(f'Expected 24 Confirmation raw/safe-metadata source rows, got {len(conf_raw)}')
    if conf_raw['subject_id'].nunique() != 12:
        raise RuntimeError('Expected 12 Confirmation subjects in raw source inventory')
    if conf_raw['values_parsed_in_semantic_report_execution_design'].astype(str).str.lower().isin(['true', '1', 'yes']).any():
        raise RuntimeError('semantic_report_execution_design inventory says Confirmation source values were parsed')
    counts = conf_raw.groupby('subject_id')['role'].nunique()
    if not (counts == 2).all():
        raise RuntimeError('Each Confirmation subject must have raw behavior + safe metadata source')
    conf_raw['values_parsed_in_validation_confirmation_semantic_design'] = False
    conf_raw['file_content_rehashed_in_validation_confirmation_semantic_design'] = False
    refs = pd.read_csv(project / semantic_report_execution_design_REF_INVENTORY)
    needed_roles = {'experience_assignments': CONF_EXP_ASSIGN_SHA256, 'experience_signature': CONF_EXP_SIG_SHA256, 'transition_assignments': CONF_TR_ASSIGN_SHA256, 'transition_signature': CONF_TR_SIG_SHA256, 'Confirmation_event_timing': CONF_TIMING_SHA256, 'Confirmation_lineage_lock': confirmation_source_epoch_lineage_LOCK_SHA256, 'canonical_trial_table_REPORTS_ONLY_FUTURE': CANONICAL_SHA256, 'analysis_specification': ANALYSIS_specification_SHA256, 'participant_split': PARTICIPANT_SPLIT_SHA256, 'README_sequence_core': README_SEQUENCE_CORE_SHA256, 'dataset_README_sequence_source': DATASET_README_SHA256}
    prespecified_rows = []
    for role, expected in needed_roles.items():
        sub = refs[refs['role'].astype(str).eq(role)]
        if role in {'experience_assignments', 'experience_signature', 'transition_assignments', 'transition_signature'}:
            sub = sub[sub['split'].astype(str).eq('Confirmation')]
        if len(sub) != 1:
            raise RuntimeError(f'semantic_report_execution_design source inventory expected exactly one row for {role}, got {len(sub)}')
        r = sub.iloc[0].to_dict()
        if str(r['expected_sha256']) != expected or str(r['observed_sha256']) != expected:
            raise RuntimeError(f'semantic_report_execution_design source inventory hash drift for {role}')
        if str(r.get('values_parsed_in_semantic_report_execution_design', '')).lower() in {'true', '1', 'yes'}:
            raise RuntimeError(f'semantic_report_execution_design source inventory unexpectedly parsed {role}')
        r['values_parsed_in_validation_confirmation_semantic_design'] = False
        prespecified_rows.append(r)
    return (conf_raw, pd.DataFrame(prespecified_rows))

def self_test() -> int:
    assert PHASE == 'validation_confirmation_semantic_design'
    assert NEXT_PHASE == 'confirmation_semantic_report_execution'
    assert REQUIRED_TESTS == ['A1_K2', 'A1_K3', 'A1_K4', 'A2_K2', 'A2_K3', 'A2_K4', 'B2', 'B3_K2', 'B3_K3']
    assert EXPERIENCE_K == [2, 3, 4]
    assert TRANSITION_K == [2, 3]
    assert len(validation_semantic_report_execution_LOCK_SHA256) == 64
    assert len(semantic_report_execution_design_LOCK_SHA256) == 64
    assert len(confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256) == 64
    assert len(confirmation_report_transition_multiresolution_transfer_LOCK_SHA256) == 64
    print('validation_confirmation_semantic_design recovery self-test PASS')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    started = time.time()
    project = args.project.resolve()
    if not project.is_dir():
        raise RuntimeError(f'Project root not found: {project}')
    out = project / OUT
    historical_stub = out / 'qc/validation_confirmation_semantic_design_summary.json'
    if historical_stub.exists():
        raise RuntimeError(f'historical invalid summary-only validation_confirmation_semantic_design artifact detected: {historical_stub}. Quarantine the old validation_confirmation_semantic_design output directory before running recovery validation_confirmation_semantic_design.')
    for sub in ['qc', 'tables', 'reports']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] verifying exact validation_semantic_report_execution recovery point and fixed children', flush=True)
    validation_semantic_report_execution_lock, rows_s = verify_path_lock_children(project, validation_semantic_report_execution_LOCK, validation_semantic_report_execution_LOCK_SHA256, 'validation_semantic_report_execution')
    verify(project / validation_semantic_report_execution_NEXT, validation_semantic_report_execution_NEXT_SHA256, 'validation_semantic_report_execution→validation_confirmation_semantic_design policy')
    validation_semantic_report_execution_next = readj(project / validation_semantic_report_execution_NEXT)
    print(f'[{now()}] verifying discovery_semantic_report_execution numerical identity and semantic_report_execution_design program lineage', flush=True)
    discovery_semantic_report_execution_lock, rows_r = verify_path_lock_children(project, discovery_semantic_report_execution_LOCK, discovery_semantic_report_execution_LOCK_SHA256, 'discovery_semantic_report_execution')
    verify(project / discovery_semantic_report_execution_NEXT, discovery_semantic_report_execution_NEXT_SHA256, 'discovery_semantic_report_execution→validation_semantic_report_execution policy')
    verify(project / discovery_semantic_report_execution_NUMERIC_specification, discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'discovery_semantic_report_execution numeric specification')
    verify(project / discovery_semantic_report_execution_SOURCE, discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution numeric source')
    verify(project / validation_semantic_report_execution_NUMERIC_ID, validation_semantic_report_execution_NUMERIC_ID_SHA256, 'validation_semantic_report_execution numeric identity')
    semantic_report_execution_design_lock, rows_q = verify_path_lock_children(project, semantic_report_execution_design_LOCK, semantic_report_execution_design_LOCK_SHA256, 'semantic_report_execution_design')
    verify(project / semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256, 'semantic_report_execution_design prespecified semantic/report program')
    semantic_report_execution_design_program = readj(project / semantic_report_execution_design_PROGRAM)
    validate_semantic_report_execution_design_program(semantic_report_execution_design_program)
    print(f'[{now()}] verifying immutable Confirmation structural state without parsing values', flush=True)
    confirmation_stimulus_experience_multiresolution_transfer_lock, rows_g = verify_path_lock_children(project, confirmation_stimulus_experience_multiresolution_transfer_LOCK, confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer')
    confirmation_report_transition_multiresolution_transfer_lock, rows_o = verify_path_lock_children(project, confirmation_report_transition_multiresolution_transfer_LOCK, confirmation_report_transition_multiresolution_transfer_LOCK_SHA256, 'confirmation_report_transition_multiresolution_transfer')
    verify(project / CONF_EXP_ASSIGN, CONF_EXP_ASSIGN_SHA256, 'Confirmation experience assignments')
    verify(project / CONF_EXP_SIG, CONF_EXP_SIG_SHA256, 'Confirmation experience signatures')
    verify(project / CONF_TR_ASSIGN, CONF_TR_ASSIGN_SHA256, 'Confirmation transition assignments')
    verify(project / CONF_TR_SIG, CONF_TR_SIG_SHA256, 'Confirmation transition signatures')
    verify(project / confirmation_source_epoch_lineage_LOCK, confirmation_source_epoch_lineage_LOCK_SHA256, 'confirmation_source_epoch_lineage Confirmation lineage lock')
    confirmation_source_epoch_lineage_lock = readj(project / confirmation_source_epoch_lineage_LOCK)
    audit_policy_and_boundaries(validation_semantic_report_execution_lock, validation_semantic_report_execution_next, discovery_semantic_report_execution_lock, semantic_report_execution_design_lock, confirmation_stimulus_experience_multiresolution_transfer_lock, confirmation_report_transition_multiresolution_transfer_lock, confirmation_source_epoch_lineage_lock)
    print(f'[{now()}] synthesizing Discovery + Validation test results for audit only', flush=True)
    discovery = load_parent_registry(project, discovery_semantic_report_execution_REGISTRY, discovery_semantic_report_execution_REGISTRY_SHA256, 'Discovery')
    validation = load_parent_registry(project, validation_semantic_report_execution_REGISTRY, validation_semantic_report_execution_REGISTRY_SHA256, 'Validation')
    audit = pd.concat([discovery, validation], ignore_index=True)
    if audit['reject_adjusted_0p05'].astype(bool).any():
        raise RuntimeError('Parent-test recovery state drift: a corrected parent unexpectedly rejects')
    audit_path = out / 'tables/validation_confirmation_semantic_design_discovery_validation_parent_test_audit.csv'
    atomic_csv(audit_path, audit)
    registry = pd.DataFrame({'order': range(1, len(REQUIRED_TESTS) + 1), 'test_id': REQUIRED_TESTS, 'required_in_confirmation': True, 'selected_by_Discovery_results': False, 'selected_by_Validation_results': False, 'K_reselection_permitted': False, 'semantic_relabeling_permitted': False})
    registry_path = out / 'tables/validation_confirmation_semantic_design_confirmation_prespecified_test_registry.csv'
    atomic_csv(registry_path, registry)
    print(f'[{now()}] freezing Confirmation source references; no Confirmation source values parsed', flush=True)
    conf_raw, conf_refs = confirmation_source_specification(project)
    raw_path = out / 'tables/validation_confirmation_semantic_design_confirmation_raw_source_inventory_prespecified.csv'
    refs_path = out / 'tables/validation_confirmation_semantic_design_confirmation_source_reference_specification.csv'
    atomic_csv(raw_path, conf_raw)
    atomic_csv(refs_path, conf_refs)
    rehash_rows = rows_s + rows_r + rows_q + rows_g + rows_o
    for label, rel, expected in [('validation_semantic_report_execution_to_validation_confirmation_semantic_design_policy', validation_semantic_report_execution_NEXT, validation_semantic_report_execution_NEXT_SHA256), ('discovery_semantic_report_execution_to_validation_semantic_report_execution_policy', discovery_semantic_report_execution_NEXT, discovery_semantic_report_execution_NEXT_SHA256), ('discovery_semantic_report_execution_numeric_specification', discovery_semantic_report_execution_NUMERIC_specification, discovery_semantic_report_execution_NUMERIC_specification_SHA256), ('discovery_semantic_report_execution_numeric_source', discovery_semantic_report_execution_SOURCE, discovery_semantic_report_execution_SOURCE_SHA256), ('validation_semantic_report_execution_numeric_identity', validation_semantic_report_execution_NUMERIC_ID, validation_semantic_report_execution_NUMERIC_ID_SHA256), ('semantic_report_execution_design_program', semantic_report_execution_design_PROGRAM, semantic_report_execution_design_PROGRAM_SHA256), ('semantic_report_execution_design_raw_inventory', semantic_report_execution_design_RAW_INVENTORY, semantic_report_execution_design_RAW_INVENTORY_SHA256), ('semantic_report_execution_design_reference_inventory', semantic_report_execution_design_REF_INVENTORY, semantic_report_execution_design_REF_INVENTORY_SHA256), ('confirmation_source_epoch_lineage_lineage_lock', confirmation_source_epoch_lineage_LOCK, confirmation_source_epoch_lineage_LOCK_SHA256), ('Confirmation_experience_assignments', CONF_EXP_ASSIGN, CONF_EXP_ASSIGN_SHA256), ('Confirmation_experience_signatures', CONF_EXP_SIG, CONF_EXP_SIG_SHA256), ('Confirmation_transition_assignments', CONF_TR_ASSIGN, CONF_TR_ASSIGN_SHA256), ('Confirmation_transition_signatures', CONF_TR_SIG, CONF_TR_SIG_SHA256)]:
        got = verify(project / rel, expected, label)
        rehash_rows.append({'source_phase': label, 'path': str(rel), 'expected_sha256': expected, 'observed_sha256': got, 'verified': True, 'content_values_parsed_in_validation_confirmation_semantic_design': False})
    rehash = pd.DataFrame(rehash_rows).drop_duplicates(subset=['path', 'expected_sha256'], keep='first')
    rehash_path = out / 'qc/validation_confirmation_semantic_design_source_rehash.csv'
    atomic_csv(rehash_path, rehash.sort_values(['source_phase', 'path']))
    next_policy = {'phase': PHASE, 'generated_at': now(), 'next_phase': NEXT_PHASE, 'next_phase_permitted': True, 'role': 'one-shot Confirmation target materialization + all prespecified semantic/report tests', 'authorization_chain': ['semantic_report_execution_design->discovery_semantic_report_execution', 'discovery_semantic_report_execution->validation_semantic_report_execution', 'validation_semantic_report_execution->validation_confirmation_semantic_design', 'validation_confirmation_semantic_design->confirmation_semantic_report_execution'], 'all_prespecified_tests_required': REQUIRED_TESTS, 'prior_results_select_Confirmation_tests': False, 'Confirmation_behavior_opening_permitted_in_confirmation_semantic_report_execution': True, 'Confirmation_behavioral_values_read_in_validation_confirmation_semantic_design': False, 'Confirmation_context_category_values_read_in_validation_confirmation_semantic_design': False, 'new_EEG_signal_opening_permitted': False, 'confirmation_semantic_report_execution_must_reuse_exact_discovery_semantic_report_execution_numeric_helpers': True, 'discovery_semantic_report_execution_numeric_source': {'path': str(discovery_semantic_report_execution_SOURCE), 'sha256': discovery_semantic_report_execution_SOURCE_SHA256}, 'discovery_semantic_report_execution_numeric_specification': {'path': str(discovery_semantic_report_execution_NUMERIC_specification), 'sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256}, 'prespecified_structural_inputs': {'experience_assignments': {'path': str(CONF_EXP_ASSIGN), 'sha256': CONF_EXP_ASSIGN_SHA256}, 'experience_signatures': {'path': str(CONF_EXP_SIG), 'sha256': CONF_EXP_SIG_SHA256}, 'transition_assignments': {'path': str(CONF_TR_ASSIGN), 'sha256': CONF_TR_ASSIGN_SHA256}, 'transition_signatures': {'path': str(CONF_TR_SIG), 'sha256': CONF_TR_SIG_SHA256}, 'experience_K': EXPERIENCE_K, 'transition_K': TRANSITION_K}, 'prespecified_target_lineage': {'reports': {'source': 'canonical report columns only, keyed by subject_id + trial_number_global', 'path': str(CANONICAL), 'sha256': CANONICAL_SHA256, 'must_source_audit_against_raw_behavior_tsv': True}, 'context': {'source': 'confirmation_source_epoch_lineage authoritative event-timing context', 'path': str(CONF_TIMING), 'sha256': CONF_TIMING_SHA256, 'lineage_lock_path': str(confirmation_source_epoch_lineage_LOCK), 'lineage_lock_sha256': confirmation_source_epoch_lineage_LOCK_SHA256, 'must_agree_safe_metadata_schedule': True}, 'category': {'source': 'authoritative context + complete safe metadata within-context index + exact README sequence parser', 'README_sequence_core_path': str(README_SEQUENCE_CORE), 'README_sequence_core_sha256': README_SEQUENCE_CORE_SHA256, 'dataset_README_sha256': DATASET_README_SHA256, 'canonical_stimulus_or_category_fallback_permitted': False, 'ambiguous_or_missing_sequence_item_policy': 'unavailable_no_fallback'}, 'analysis_specification': {'path': str(ANALYSIS_specification), 'sha256': ANALYSIS_specification_SHA256}, 'participant_split': {'path': str(PARTICIPANT_SPLIT), 'sha256': PARTICIPANT_SPLIT_SHA256}, 'raw_source_inventory_specification': str(raw_path.relative_to(project)), 'source_reference_specification': str(refs_path.relative_to(project))}, 'prespecified_boundaries': {'canonical_context_stimulus_category_permitted': False, 'posthoc_report_subset_selection_permitted': False, 'K_reselection_permitted': False, 'family_ID_changes_permitted': False, 'semantic_relabeling_permitted': False, 'split_specific_family_relabeling_permitted': False, 'behavior_may_change_structural_decisions': False, 'distortion_dishonesty_suppression_exaggeration_interpretation_permitted': False, 'external_replication_claim_permitted': False}, 'followup_gate': 'single-report/dimension followups only according to the already-prespecified semantic_report_execution_design parent gates and Holm rules', 'next_phase_after_success': 'cross_split_semantic_report_synthesis'}
    writej(project / NEXT_CONFIG, next_policy)
    summary = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_synthesis_confirmation_design_specification_complete', 'recovery_point_validation_semantic_report_execution_verified': True, 'validation_semantic_report_execution_final_lock_sha256': validation_semantic_report_execution_LOCK_SHA256, 'semantic_report_execution_design_program_verified': True, 'discovery_semantic_report_execution_numeric_identity_verified': True, 'Confirmation_structural_inputs_verified': True, 'Discovery_parent_rejections_after_prespecified_correction': int(discovery['reject_adjusted_0p05'].astype(bool).sum()), 'Validation_parent_rejections_after_prespecified_correction': int(validation['reject_adjusted_0p05'].astype(bool).sum()), 'all_nine_tests_required_in_Confirmation': True, 'required_tests': REQUIRED_TESTS, 'experience_K': EXPERIENCE_K, 'report_transition_K': TRANSITION_K, 'Confirmation_behavioral_values_read': False, 'Confirmation_context_values_read': False, 'Confirmation_emotion_category_values_read': False, 'Confirmation_raw_source_file_contents_rehashed': False, 'new_EEG_signal_values_opened': False, 'canonical_context_category_used': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_opens_Confirmation_values': True, 'runtime_seconds': time.time() - started}
    summary_path = out / 'qc/validation_confirmation_semantic_design_summary.json'
    writej(summary_path, summary)
    contract = {'phase': PHASE, 'generated_at': now(), 'status': 'synthesis_and_design_specification_only_no_Confirmation_target_values', 'source_code_sha256': sha256(Path(__file__).resolve()), 'verified_parent_locks': {'validation_semantic_report_execution': validation_semantic_report_execution_LOCK_SHA256, 'discovery_semantic_report_execution': discovery_semantic_report_execution_LOCK_SHA256, 'semantic_report_execution_design': semantic_report_execution_design_LOCK_SHA256, 'confirmation_stimulus_experience_multiresolution_transfer': confirmation_stimulus_experience_multiresolution_transfer_LOCK_SHA256, 'confirmation_report_transition_multiresolution_transfer': confirmation_report_transition_multiresolution_transfer_LOCK_SHA256, 'confirmation_source_epoch_lineage': confirmation_source_epoch_lineage_LOCK_SHA256}, 'validation_semantic_report_execution_to_validation_confirmation_semantic_design_policy_sha256': validation_semantic_report_execution_NEXT_SHA256, 'discovery_semantic_report_execution_numeric_source_sha256': discovery_semantic_report_execution_SOURCE_SHA256, 'discovery_semantic_report_execution_numeric_specification_sha256': discovery_semantic_report_execution_NUMERIC_specification_SHA256, 'Confirmation_test_registry': REQUIRED_TESTS, 'prior_results_select_Confirmation_tests': False, 'boundary': {'Confirmation_behavioral_values_read': False, 'Confirmation_context_values_read': False, 'Confirmation_emotion_category_values_read': False, 'new_EEG_signal_values_opened': False, 'canonical_context_category_used': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False}, 'next_phase': NEXT_PHASE}
    contract_path = out / 'qc/validation_confirmation_semantic_design_contract.json'
    writej(contract_path, contract)
    report_path = out / 'reports/validation_confirmation_semantic_design_report.md'
    report = f"# validation_confirmation_semantic_design — Validation synthesis + Confirmation analysis specification (recovery build)\n\n**This phase opened no Confirmation behavioral/report/category/context values and no new EEG signal arrays.**\n\n## Recovery point\n\n- Exact validation_semantic_report_execution final lock verified: `{validation_semantic_report_execution_LOCK_SHA256}`\n- Exact validation_semantic_report_execution→validation_confirmation_semantic_design authorization policy verified: `{validation_semantic_report_execution_NEXT_SHA256}`\n- Exact semantic_report_execution_design prespecified semantic/report program verified.\n- Exact discovery_semantic_report_execution numerical implementation identity verified.\n- Immutable Confirmation experience K=2/3/4 and transition K=2/3 structural inputs verified.\n\n## Discovery + Validation audit\n\nThe parent-test registries from discovery_semantic_report_execution and validation_semantic_report_execution are copied into a synthesis table for audit only.\nNo prior p-value selects, drops, ranks, or changes a Confirmation test.\n\nAll nine prespecified Confirmation parent tests remain mandatory:\n\n`{', '.join(REQUIRED_TESTS)}`\n\n## Confirmation opening boundary\n\nC9T does **not** parse Confirmation report, context, category, or raw-behavior values.\nIt only specifications pre-existing provenance references from semantic_report_execution_design and verifies already-open structural\nartifacts from confirmation_stimulus_experience_multiresolution_transfer/confirmation_report_transition_multiresolution_transfer.\n\nC9U may perform the one-shot Confirmation target materialization and execute all prespecified tests.\n\n## Interpretation boundary\n\nNo family semantic naming is permitted.\nA non-significant mapping does not prove that human emotion labels are wrong.\nReport-transition activity must not be described as distortion, dishonesty, suppression,\nexaggeration, or conscious editing.\n\n## Next\n\n**{NEXT_PHASE}** — one-shot Confirmation target materialization + all prespecified A1/A2/B2/B3 tests."
    report_path.write_text(report, encoding='utf-8')
    lock_targets = [project / NEXT_CONFIG, contract_path, summary_path, rehash_path, audit_path, registry_path, raw_path, refs_path, report_path]
    final_lock = out / 'qc/validation_confirmation_semantic_design_final_output_lock_hashes.json'
    final_obj = {'phase': PHASE, 'generated_at': now(), 'status': 'validation_synthesis_confirmation_design_specification_complete', 'hashes': {str(p.relative_to(project)): sha256(p) for p in lock_targets}, 'recovery_point_validation_semantic_report_execution_verified': True, 'Confirmation_behavioral_values_read': False, 'Confirmation_context_values_read': False, 'Confirmation_emotion_category_values_read': False, 'new_EEG_signal_values_opened': False, 'all_prespecified_tests_required_in_Confirmation': REQUIRED_TESTS, 'prior_results_select_Confirmation_tests': False, 'semantic_relabeling_permitted': False, 'distortion_interpretation_permitted': False, 'next_phase': NEXT_PHASE, 'next_phase_opens_Confirmation_values': True}
    writej(final_lock, final_obj)
    script_dir = Path(__file__).resolve().parent
    for p in sorted(script_dir.iterdir()):
        if p.is_file():
            pass
    print('')
    print('validation_confirmation_semantic_design RECOVERY PASS')
    print(f'validation_semantic_report_execution recovery lock verified: {validation_semantic_report_execution_LOCK_SHA256}')
    print(f"Discovery corrected parent rejections: {summary['Discovery_parent_rejections_after_prespecified_correction']}")
    print(f"Validation corrected parent rejections: {summary['Validation_parent_rejections_after_prespecified_correction']}")
    print('Confirmation values opened in validation_confirmation_semantic_design: False')
    print(f'All prespecified Confirmation tests required: {REQUIRED_TESTS}')
    print(f'Next phase: {NEXT_PHASE}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())