# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
subjective discovery analysis — discovery-only subjective architecture.

Scientific purpose
------------------
Test whether pleasure/aversion reports are adequately described by one scalar
axis or require multiple person-relative subjective configurations.

The analysis is strictly restricted to discovery participants. Validation and
confirmation outcomes are not used for model selection, preprocessing,
dimensionality selection, clustering, or hypothesis revision.

Primary roles
-------------
Latent subjective features:
    joy, inspiration, tenderness, sadness, fear, disgust, arousal

Auxiliary appraisal:
    familiarity

Hedonic outcomes:
    valence, liking

Contextual descriptors, never used for unsupervised state discovery:
    context, emotion_category, stimulus_id

Main analyses
-------------
1. prespecified-input and lockbox validation.
2. Discovery-only report-quality and category-overlap audit.
3. Person-relative PCA, parallel analysis, reconstruction CV, and stability.
4. Continuous-versus-discrete comparison using Gaussian mixtures.
5. Participant-held-out prediction of valence and liking.
6. Explicit label-based calibration curves.
7. Proposed validation specification, without opening validation outcomes.

Default project
---------------
<PROJECT_ROOT>"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
try:
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, normalized_mutual_info_score, r2_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge
except ImportError as exc:
    raise SystemExit(f'Missing scientific dependency: {exc}\nInstall the project environment from env/environment.yml or install numpy pandas scipy scikit-learn matplotlib joblib.')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
CORE_PROFILE = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal']
AUXILIARY_REPORTS = ['familiarity']
HEDONIC_OUTCOMES = ['valence', 'liking']
CATEGORICAL_DESCRIPTORS = ['context', 'emotion_category']
PRIMARY_KEY = ['subject_id', 'trial_number_global']
MIDPOINT = 3.5
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
CALIBRATION_TRIALS = [0, 5, 10, 20, 30]
RANDOM_SEED = 1540

@dataclass
class Paths:
    project: Path

    @property
    def canonical(self) -> Path:
        return self.project / 'data/canonical_behavioral_table.csv'

    @property
    def split(self) -> Path:
        return self.project / 'config/participant_split_manifest.v1.csv'

    @property
    def specification(self) -> Path:
        return self.project / 'config/analysis_specification.v1.json'

    @property
    def contract(self) -> Path:
        return self.project / 'config/canonical_trial_contract.v1.yaml'

    @property
    def tables(self) -> Path:
        return self.project / 'outputs/tables'

    @property
    def qc(self) -> Path:
        return self.project / 'outputs/qc'

    @property
    def reports(self) -> Path:
        return self.project / 'outputs/reports'

    @property
    def figures(self) -> Path:
        return self.project / 'outputs/figures'

    @property
    def models(self) -> Path:
        return self.project / 'outputs/models'

    @property
    def logs(self) -> Path:
        return self.project / 'outputs/logs'

    @property
    def config(self) -> Path:
        return self.project / 'config'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run discovery-only subjective architecture subjective discovery analysis.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--parallel-iterations', type=int, default=300)
    parser.add_argument('--pca-bootstrap', type=int, default=150)
    parser.add_argument('--gmm-bootstrap', type=int, default=60)
    parser.add_argument('--metric-bootstrap', type=int, default=300)
    parser.add_argument('--force', action='store_true', help='Replace existing subjective discovery analysis outputs.')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def log(message: str, handle=None) -> None:
    line = f'[{now_iso()}] {message}'
    print(line)
    if handle is not None:
        handle.write(line + '')
        handle.flush()

def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float('nan')
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float('nan')
    return float(pearsonr(y_true, y_pred).statistic)

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {'r2': float(r2_score(y_true, y_pred)), 'mae': float(mean_absolute_error(y_true, y_pred)), 'rmse': float(math.sqrt(mean_squared_error(y_true, y_pred))), 'pearson_r': safe_corr(y_true, y_pred)}

def within_person_z(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        grouped = output.groupby('subject_id')[column]
        means = grouped.transform('mean')
        stds = grouped.transform(lambda values: values.std(ddof=0)).replace(0, 1.0)
        output[f'rel_{column}'] = (output[column] - means) / stds
    return output

def validate_inputs(paths: Paths) -> dict:
    required = [paths.canonical, paths.split, paths.specification, paths.contract]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Required prespecified inputs are missing:\n  - ' + '\n  - '.join((str(path) for path in missing)))
    specification = json.loads(paths.specification.read_text(encoding='utf-8'))
    expected_canonical = specification['hashes']['canonical_trial_table']
    expected_split = specification['hashes']['participant_split_manifest']
    observed_canonical = sha256_file(paths.canonical)
    observed_split = sha256_file(paths.split)
    if observed_canonical != expected_canonical:
        raise RuntimeError(f'Canonical trial table hash changed after specification.\nExpected: {expected_canonical}\nObserved: {observed_canonical}')
    if observed_split != expected_split:
        raise RuntimeError(f'Participant split hash changed after specification.\nExpected: {expected_split}\nObserved: {observed_split}')
    if specification.get('version') != 'v1':
        raise RuntimeError(f"Unexpected specification version: {specification.get('version')}")
    return {'specification': specification, 'canonical_sha256': observed_canonical, 'split_sha256': observed_split}

def prepare_discovery(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    canonical = pd.read_csv(paths.canonical)
    split = pd.read_csv(paths.split)
    required_columns = PRIMARY_KEY + CORE_PROFILE + AUXILIARY_REPORTS + HEDONIC_OUTCOMES + CATEGORICAL_DESCRIPTORS + ['stimulus_id', 'age', 'gender']
    missing = [column for column in required_columns if column not in canonical.columns]
    if missing:
        raise RuntimeError(f'Canonical table is missing columns: {missing}')
    if canonical.duplicated(PRIMARY_KEY).any():
        raise RuntimeError('Canonical primary key is not unique.')
    if split['subject_id'].duplicated().any():
        raise RuntimeError('Split manifest contains duplicate subjects.')
    merged = canonical.merge(split[['subject_id', 'split']], on='subject_id', how='left', validate='many_to_one')
    if merged['split'].isna().any():
        raise RuntimeError('Some canonical participants are missing from split manifest.')
    discovery = merged.loc[merged['split'].eq('discovery')].copy()
    discovery_subjects = set(split.loc[split['split'].eq('discovery'), 'subject_id'])
    observed_subjects = set(discovery['subject_id'])
    if observed_subjects != discovery_subjects:
        raise RuntimeError('Discovery subject set does not match split manifest.')
    numeric_columns = CORE_PROFILE + AUXILIARY_REPORTS + HEDONIC_OUTCOMES
    discovery[numeric_columns] = discovery[numeric_columns].apply(pd.to_numeric, errors='coerce')
    if discovery[numeric_columns].isna().any().any():
        missing_counts = discovery[numeric_columns].isna().sum().loc[lambda values: values > 0].to_dict()
        raise RuntimeError(f'Unexpected missing discovery ratings: {missing_counts}')
    discovery = within_person_z(discovery, CORE_PROFILE + AUXILIARY_REPORTS + HEDONIC_OUTCOMES)
    audit = {'canonical_rows': int(len(canonical)), 'canonical_subjects': int(canonical['subject_id'].nunique()), 'discovery_rows': int(len(discovery)), 'discovery_subjects': int(discovery['subject_id'].nunique()), 'validation_subjects': int(split['split'].eq('validation').sum()), 'confirmation_subjects': int(split['split'].eq('confirmation').sum()), 'discovery_only': bool(discovery['split'].eq('discovery').all())}
    return (discovery, split, audit)

def input_audit_rows(discovery: pd.DataFrame, audit: dict) -> list[dict]:
    rows = [{'check': 'canonical_rows', 'value': audit['canonical_rows'], 'expected': 2457, 'passed': audit['canonical_rows'] == 2457}, {'check': 'canonical_subjects', 'value': audit['canonical_subjects'], 'expected': 60, 'passed': audit['canonical_subjects'] == 60}, {'check': 'discovery_subjects', 'value': audit['discovery_subjects'], 'expected': 36, 'passed': audit['discovery_subjects'] == 36}, {'check': 'discovery_only', 'value': audit['discovery_only'], 'expected': True, 'passed': audit['discovery_only']}, {'check': 'primary_key_unique', 'value': int(discovery.duplicated(PRIMARY_KEY).sum()), 'expected': 0, 'passed': not discovery.duplicated(PRIMARY_KEY).any()}]
    for column in CORE_PROFILE + AUXILIARY_REPORTS + HEDONIC_OUTCOMES:
        values = discovery[column]
        rows.append({'check': f'{column}_range', 'value': f'{values.min():.4g}..{values.max():.4g}', 'expected': '0..7', 'passed': bool(values.min() >= 0 and values.max() <= 7)})
    return rows

def category_overlap(discovery: pd.DataFrame) -> pd.DataFrame:
    person_means = discovery.groupby('subject_id')[HEDONIC_OUTCOMES].mean().rename(columns={'valence': 'person_valence_mean', 'liking': 'person_liking_mean'})
    frame = discovery.join(person_means, on='subject_id')
    frame['valence_above_person_mean'] = frame['valence'] > frame['person_valence_mean']
    frame['liking_above_person_mean'] = frame['liking'] > frame['person_liking_mean']
    output = []
    for group_columns in [['emotion_category'], ['context', 'emotion_category']]:
        for key, group in frame.groupby(group_columns, observed=True):
            if not isinstance(key, tuple):
                key = (key,)
            record = {column: value for column, value in zip(group_columns, key)}
            record.update({'aggregation': '+'.join(group_columns), 'n_trials': len(group), 'n_subjects': group['subject_id'].nunique(), 'valence_mean': group['valence'].mean(), 'valence_sd': group['valence'].std(), 'valence_min': group['valence'].min(), 'valence_max': group['valence'].max(), 'valence_below_midpoint_prop': (group['valence'] < MIDPOINT).mean(), 'valence_above_midpoint_prop': (group['valence'] > MIDPOINT).mean(), 'valence_above_person_mean_prop': group['valence_above_person_mean'].mean(), 'liking_mean': group['liking'].mean(), 'liking_sd': group['liking'].std(), 'liking_min': group['liking'].min(), 'liking_max': group['liking'].max(), 'liking_below_midpoint_prop': (group['liking'] < MIDPOINT).mean(), 'liking_above_midpoint_prop': (group['liking'] > MIDPOINT).mean(), 'liking_above_person_mean_prop': group['liking_above_person_mean'].mean()})
            output.append(record)
    return pd.DataFrame(output)

def person_context_summary(discovery: pd.DataFrame) -> pd.DataFrame:
    summaries = discovery.groupby(['subject_id', 'context'], observed=True).agg(n_trials=('trial_number_global', 'size'), valence_mean=('valence', 'mean'), valence_sd=('valence', 'std'), liking_mean=('liking', 'mean'), liking_sd=('liking', 'std'), arousal_mean=('arousal', 'mean'), familiarity_mean=('familiarity', 'mean')).reset_index()
    return summaries

def correlation_summary(discovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({'level': 'pooled_raw', 'subject_id': '', 'n': len(discovery), 'valence_liking_r': safe_corr(discovery['valence'].to_numpy(), discovery['liking'].to_numpy())})
    rows.append({'level': 'pooled_person_relative', 'subject_id': '', 'n': len(discovery), 'valence_liking_r': safe_corr(discovery['rel_valence'].to_numpy(), discovery['rel_liking'].to_numpy())})
    for subject, group in discovery.groupby('subject_id'):
        rows.append({'level': 'within_subject', 'subject_id': subject, 'n': len(group), 'valence_liking_r': safe_corr(group['valence'].to_numpy(), group['liking'].to_numpy())})
    return pd.DataFrame(rows)

def reconstruction_cv(X: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    splitter = GroupKFold(n_splits=6)
    for n_components in range(1, X.shape[1] + 1):
        fold_mse = []
        for fold, (train, test) in enumerate(splitter.split(X, groups=groups), start=1):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            pca.fit(X_train)
            reconstructed = pca.inverse_transform(pca.transform(X_test))
            mse = mean_squared_error(X_test, reconstructed)
            fold_mse.append(float(mse))
            rows.append({'n_components': n_components, 'fold': fold, 'mse': float(mse)})
    frame = pd.DataFrame(rows)
    summary = frame.groupby('n_components').agg(mean_mse=('mse', 'mean'), sd_mse=('mse', 'std'), n_folds=('mse', 'size')).reset_index()
    summary['se_mse'] = summary['sd_mse'] / np.sqrt(summary['n_folds'])
    return summary

def parallel_analysis(X_relative: np.ndarray, subject_ids: np.ndarray, iterations: int) -> tuple[pd.DataFrame, StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_relative)
    observed_pca = PCA(random_state=RANDOM_SEED)
    scores = observed_pca.fit_transform(X_scaled)
    observed = observed_pca.explained_variance_
    rng = np.random.default_rng(RANDOM_SEED)
    null_eigenvalues = np.zeros((iterations, X_scaled.shape[1]), dtype=float)
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    for iteration in range(iterations):
        null = X_relative.copy()
        for feature_index in range(null.shape[1]):
            for indices in subject_indices.values():
                null[indices, feature_index] = rng.permutation(null[indices, feature_index])
        null_scaled = StandardScaler().fit_transform(null)
        null_pca = PCA(random_state=RANDOM_SEED)
        null_pca.fit(null_scaled)
        null_eigenvalues[iteration] = null_pca.explained_variance_
    null_95 = np.quantile(null_eigenvalues, 0.95, axis=0)
    selected = observed > null_95
    table = pd.DataFrame({'component': np.arange(1, len(observed) + 1), 'observed_eigenvalue': observed, 'null_eigenvalue_95': null_95, 'observed_explained_variance_ratio': observed_pca.explained_variance_ratio_, 'selected_by_parallel_analysis': selected})
    return (table, scaler, observed_pca, scores)

def choose_dimensions(parallel: pd.DataFrame, reconstruction: pd.DataFrame) -> dict:
    parallel_n = int(parallel['selected_by_parallel_analysis'].sum())
    parallel_n = max(1, parallel_n)
    best_index = reconstruction['mean_mse'].idxmin()
    best_row = reconstruction.loc[best_index]
    threshold = best_row['mean_mse'] + best_row['se_mse']
    eligible = reconstruction.loc[reconstruction['mean_mse'] <= threshold, 'n_components']
    one_se_n = int(eligible.min())
    selected_n = max(1, min(parallel_n, one_se_n))
    one_dimensional_mse = float(reconstruction.loc[reconstruction['n_components'].eq(1), 'mean_mse'].iloc[0])
    selected_mse = float(reconstruction.loc[reconstruction['n_components'].eq(selected_n), 'mean_mse'].iloc[0])
    return {'parallel_dimensions': parallel_n, 'cv_one_se_dimensions': one_se_n, 'selected_dimensions': selected_n, 'pc1_explained_variance_ratio': float(parallel.loc[parallel['component'].eq(1), 'observed_explained_variance_ratio'].iloc[0]), 'one_component_cv_mse': one_dimensional_mse, 'selected_dimension_cv_mse': selected_mse, 'relative_reconstruction_improvement': (one_dimensional_mse - selected_mse) / one_dimensional_mse if one_dimensional_mse > 0 else 0.0}

def align_components(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    similarity = np.abs(reference.T @ candidate)
    row_index, column_index = linear_sum_assignment(-similarity)
    aligned = np.zeros_like(reference)
    for ref_component, candidate_component in zip(row_index, column_index):
        vector = candidate[:, candidate_component].copy()
        sign = np.sign(np.dot(reference[:, ref_component], vector))
        if sign == 0:
            sign = 1
        aligned[:, ref_component] = vector * sign
    return aligned

def pca_stability(discovery: pd.DataFrame, selected_n: int, reference_loadings: np.ndarray, iterations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    subjects = discovery['subject_id'].unique()
    rel_columns = [f'rel_{column}' for column in CORE_PROFILE]
    rows = []
    for iteration in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        boot_parts = [discovery.loc[discovery['subject_id'].eq(subject), rel_columns] for subject in sampled]
        X_boot = pd.concat(boot_parts, ignore_index=True).to_numpy()
        X_boot = StandardScaler().fit_transform(X_boot)
        pca = PCA(n_components=selected_n, random_state=RANDOM_SEED + iteration)
        pca.fit(X_boot)
        candidate = pca.components_.T
        aligned = align_components(reference_loadings, candidate)
        for component in range(selected_n):
            congruence = float(np.dot(reference_loadings[:, component], aligned[:, component]))
            rows.append({'bootstrap_iteration': iteration + 1, 'component': component + 1, 'congruence': congruence})
    frame = pd.DataFrame(rows)
    return frame.groupby('component').agg(median_congruence=('congruence', 'median'), q05_congruence=('congruence', lambda values: values.quantile(0.05)), q95_congruence=('congruence', lambda values: values.quantile(0.95))).reset_index()

def gmm_analysis(discovery: pd.DataFrame, selected_n: int, full_scaler: StandardScaler, full_pca: PCA, gmm_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame, GaussianMixture, dict]:
    rel_columns = [f'rel_{column}' for column in CORE_PROFILE]
    X = discovery[rel_columns].to_numpy()
    groups = discovery['subject_id'].to_numpy()
    categories = discovery['emotion_category'].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=6)
    rng = np.random.default_rng(RANDOM_SEED + 2)
    full_scores = full_pca.transform(full_scaler.transform(X))[:, :selected_n]
    selection_rows = []
    full_models: dict[int, GaussianMixture] = {}
    full_labels: dict[int, np.ndarray] = {}
    max_clusters = min(8, max(2, len(discovery) // 100))
    for clusters in range(1, max_clusters + 1):
        fold_scores = []
        for fold, (train, test) in enumerate(splitter.split(X, groups=groups), start=1):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
            pca = PCA(n_components=selected_n, random_state=RANDOM_SEED + fold)
            train_scores = pca.fit_transform(X_train)
            test_scores = pca.transform(X_test)
            model = GaussianMixture(n_components=clusters, covariance_type='full', n_init=5, reg_covar=1e-06, random_state=RANDOM_SEED + fold)
            model.fit(train_scores)
            fold_scores.append(float(model.score(test_scores)))
        full_model = GaussianMixture(n_components=clusters, covariance_type='full', n_init=10, reg_covar=1e-06, random_state=RANDOM_SEED)
        full_model.fit(full_scores)
        labels = full_model.predict(full_scores)
        full_models[clusters] = full_model
        full_labels[clusters] = labels
        cluster_fractions = np.bincount(labels, minlength=clusters) / len(labels)
        stability_values = []
        if clusters == 1:
            stability_values = [1.0]
        else:
            subjects = discovery['subject_id'].unique()
            subject_row_indices = {subject: np.flatnonzero(discovery['subject_id'].to_numpy() == subject) for subject in subjects}
            reference = labels
            for iteration in range(gmm_bootstrap):
                sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
                indices = np.concatenate([subject_row_indices[subject] for subject in sampled_subjects])
                bootstrap_model = GaussianMixture(n_components=clusters, covariance_type='full', n_init=5, reg_covar=1e-06, random_state=RANDOM_SEED + clusters * 1000 + iteration)
                try:
                    bootstrap_model.fit(full_scores[indices])
                    bootstrap_labels = bootstrap_model.predict(full_scores)
                    stability_values.append(adjusted_rand_score(reference, bootstrap_labels))
                except Exception:
                    stability_values.append(float('nan'))
        stability_array = np.asarray(stability_values, dtype=float)
        stability_array = stability_array[np.isfinite(stability_array)]
        selection_rows.append({'clusters': clusters, 'cv_log_likelihood_mean': float(np.mean(fold_scores)), 'cv_log_likelihood_sd': float(np.std(fold_scores, ddof=1)), 'bic': float(full_model.bic(full_scores)), 'aic': float(full_model.aic(full_scores)), 'min_cluster_fraction': float(cluster_fractions.min()), 'stability_ari_median': float(np.median(stability_array)), 'stability_ari_q05': float(np.quantile(stability_array, 0.05)), 'category_nmi': float(normalized_mutual_info_score(categories, labels))})
    selection = pd.DataFrame(selection_rows)
    best_row = selection.loc[selection['cv_log_likelihood_mean'].idxmax()]
    best_k = int(best_row['clusters'])
    baseline_ll = float(selection.loc[selection['clusters'].eq(1), 'cv_log_likelihood_mean'].iloc[0])
    delta_ll = float(best_row['cv_log_likelihood_mean'] - baseline_ll)
    discrete_supported = bool(best_k > 1 and delta_ll >= 0.02 and (best_row['stability_ari_median'] >= 0.6) and (best_row['min_cluster_fraction'] >= 0.05))
    best_model = full_models[best_k]
    best_labels = full_labels[best_k]
    assignments = discovery[PRIMARY_KEY + ['context', 'emotion_category', 'valence', 'liking']].copy()
    assignments['candidate_configuration'] = best_labels + 1
    summary_rows = []
    for configuration, group in assignments.groupby('candidate_configuration'):
        summary_rows.append({'candidate_configuration': int(configuration), 'n_trials': int(len(group)), 'fraction': float(len(group) / len(assignments)), 'n_subjects': int(group['subject_id'].nunique()), 'valence_mean': float(group['valence'].mean()), 'liking_mean': float(group['liking'].mean()), 'dominant_category': str(group['emotion_category'].value_counts().index[0])})
    summary = {'best_candidate_clusters': best_k, 'delta_cv_log_likelihood_vs_k1': delta_ll, 'best_stability_ari_median': float(best_row['stability_ari_median']), 'best_min_cluster_fraction': float(best_row['min_cluster_fraction']), 'best_category_nmi': float(best_row['category_nmi']), 'discrete_supported': discrete_supported, 'preferred_representation': 'discrete_candidate' if discrete_supported else 'continuous_latent_geometry'}
    return (selection, pd.DataFrame(summary_rows), best_model, {**summary, 'assignments': assignments, 'full_scores': full_scores})

def make_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(('numeric', StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([('preprocessor', preprocessor), ('model', Ridge())])

def tuned_ridge_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame, y_train: np.ndarray, train_groups: np.ndarray, numeric_columns: list[str], categorical_columns: list[str]) -> tuple[np.ndarray, float]:
    pipeline = make_pipeline(numeric_columns, categorical_columns)
    unique_groups = np.unique(train_groups)
    n_inner = min(3, len(unique_groups))
    if n_inner < 2:
        raise RuntimeError('Too few groups for inner CV.')
    search = GridSearchCV(pipeline, param_grid={'model__alpha': RIDGE_ALPHAS}, scoring='neg_mean_squared_error', cv=GroupKFold(n_splits=n_inner), n_jobs=-1, refit=True)
    search.fit(train_frame, y_train, groups=train_groups)
    predictions = search.predict(test_frame)
    return (predictions, float(search.best_params_['model__alpha']))

def prediction_benchmark(discovery: pd.DataFrame, selected_n: int, metric_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    outer = GroupKFold(n_splits=6)
    groups = discovery['subject_id'].to_numpy()
    rel_core = [f'rel_{column}' for column in CORE_PROFILE]
    rel_extended = rel_core + ['rel_familiarity']
    raw_extended = CORE_PROFILE + AUXILIARY_REPORTS
    model_specs = {'mean': ([], []), 'category_context': ([], CATEGORICAL_DESCRIPTORS), 'profile_raw': (raw_extended, []), 'profile_person_relative': (rel_extended, []), 'combined_raw': (raw_extended, CATEGORICAL_DESCRIPTORS), 'combined_person_relative': (rel_extended, CATEGORICAL_DESCRIPTORS)}
    prediction_rows = []
    fold_metric_rows = []
    for outcome in HEDONIC_OUTCOMES:
        y = discovery[outcome].to_numpy()
        for fold, (train, test) in enumerate(outer.split(discovery, y, groups), start=1):
            train_frame = discovery.iloc[train].copy()
            test_frame = discovery.iloc[test].copy()
            y_train = y[train]
            y_test = y[test]
            train_groups = groups[train]
            fold_scaler = StandardScaler()
            X_train_core = fold_scaler.fit_transform(train_frame[rel_core])
            X_test_core = fold_scaler.transform(test_frame[rel_core])
            fold_pca = PCA(n_components=selected_n, random_state=RANDOM_SEED + fold)
            train_pcs = fold_pca.fit_transform(X_train_core)
            test_pcs = fold_pca.transform(X_test_core)
            for component in range(selected_n):
                column = f'fold_pc{component + 1}'
                train_frame[column] = train_pcs[:, component]
                test_frame[column] = test_pcs[:, component]
            pca_specs = {'pc1_person_relative': (['fold_pc1'], []), 'pca_k_person_relative': ([f'fold_pc{component + 1}' for component in range(selected_n)], []), 'pca_k_plus_context': ([f'fold_pc{component + 1}' for component in range(selected_n)], CATEGORICAL_DESCRIPTORS)}
            combined_specs = {**model_specs, **pca_specs}
            for model_name, (numeric_columns, categorical_columns) in combined_specs.items():
                if model_name == 'mean':
                    predictions = np.full(len(test), y_train.mean())
                    alpha = float('nan')
                else:
                    predictions, alpha = tuned_ridge_predict(train_frame, test_frame, y_train, train_groups, numeric_columns, categorical_columns)
                metrics = metric_dict(y_test, predictions)
                fold_metric_rows.append({'outcome': outcome, 'model': model_name, 'fold': fold, 'n_test': len(test), 'alpha': alpha, **metrics})
                for row_index, prediction in zip(test, predictions):
                    source_row = discovery.iloc[row_index]
                    prediction_rows.append({'subject_id': source_row['subject_id'], 'trial_number_global': source_row['trial_number_global'], 'context': source_row['context'], 'emotion_category': source_row['emotion_category'], 'outcome': outcome, 'model': model_name, 'fold': fold, 'observed': source_row[outcome], 'predicted': float(prediction)})
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    rng = np.random.default_rng(RANDOM_SEED + 3)
    aggregate_rows = []
    for (outcome, model), group in predictions.groupby(['outcome', 'model']):
        observed = group['observed'].to_numpy()
        predicted = group['predicted'].to_numpy()
        point = metric_dict(observed, predicted)
        subjects = group['subject_id'].unique()
        subject_parts = {subject: group.loc[group['subject_id'].eq(subject)] for subject in subjects}
        bootstrap_values = {metric: [] for metric in point}
        for _ in range(metric_bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            boot = pd.concat([subject_parts[subject] for subject in sampled], ignore_index=True)
            values = metric_dict(boot['observed'].to_numpy(), boot['predicted'].to_numpy())
            for metric, value in values.items():
                if np.isfinite(value):
                    bootstrap_values[metric].append(value)
        record = {'outcome': outcome, 'model': model, 'n_trials': len(group), 'n_subjects': len(subjects)}
        for metric, value in point.items():
            samples = np.asarray(bootstrap_values[metric], dtype=float)
            record[metric] = value
            record[f'{metric}_ci_low'] = float(np.quantile(samples, 0.025)) if len(samples) else float('nan')
            record[f'{metric}_ci_high'] = float(np.quantile(samples, 0.975)) if len(samples) else float('nan')
        aggregate_rows.append(record)
    aggregate = pd.DataFrame(aggregate_rows)
    model_selection = aggregate.groupby('model')['r2'].mean().sort_values(ascending=False)
    eligible = model_selection.drop(labels=['mean'], errors='ignore')
    selected_model = str(eligible.index[0])
    raw_candidates = [model for model in ['category_context', 'profile_raw', 'combined_raw'] if model in model_selection.index]
    calibration_base_model = str(model_selection.loc[raw_candidates].sort_values(ascending=False).index[0])
    summary = {'selected_validation_model': selected_model, 'selected_model_mean_discovery_r2': float(model_selection[selected_model]), 'calibration_base_model': calibration_base_model, 'model_mean_r2': {model: float(value) for model, value in model_selection.items()}}
    return (aggregate, fold_metrics, predictions, summary)

def calibration_curve(predictions: pd.DataFrame, base_model: str) -> pd.DataFrame:
    rows = []
    for outcome in HEDONIC_OUTCOMES:
        subset = predictions.loc[predictions['outcome'].eq(outcome) & predictions['model'].eq(base_model)].copy()
        for n_calibration in CALIBRATION_TRIALS:
            evaluation_parts = []
            for subject, group in subset.groupby('subject_id'):
                group = group.sort_values('trial_number_global').copy()
                if len(group) <= n_calibration + 5:
                    continue
                if n_calibration == 0:
                    offset = 0.0
                    evaluation = group
                else:
                    calibration = group.iloc[:n_calibration]
                    offset = float((calibration['observed'] - calibration['predicted']).mean())
                    evaluation = group.iloc[n_calibration:].copy()
                evaluation['calibrated_prediction'] = evaluation['predicted'] + offset
                evaluation_parts.append(evaluation)
            if not evaluation_parts:
                continue
            evaluation = pd.concat(evaluation_parts, ignore_index=True)
            metrics = metric_dict(evaluation['observed'].to_numpy(), evaluation['calibrated_prediction'].to_numpy())
            rows.append({'outcome': outcome, 'base_model': base_model, 'calibration_trials': n_calibration, 'evaluation_trials': len(evaluation), 'evaluation_subjects': evaluation['subject_id'].nunique(), **metrics})
    return pd.DataFrame(rows)

def save_figures(paths: Paths, parallel: pd.DataFrame, reconstruction: pd.DataFrame, loadings: pd.DataFrame, gmm_selection: pd.DataFrame, prediction_metrics: pd.DataFrame, calibration: pd.DataFrame) -> list[Path]:
    paths.figures.mkdir(parents=True, exist_ok=True)
    outputs = []
    path = paths.figures / 'subjective_discovery_parallel_analysis.png'
    plt.figure(figsize=(8, 5))
    plt.plot(parallel['component'], parallel['observed_eigenvalue'], marker='o', label='Observed')
    plt.plot(parallel['component'], parallel['null_eigenvalue_95'], marker='o', label='95% permuted null')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Discovery subjective profile: parallel analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    path = paths.figures / 'subjective_discovery_reconstruction_cv.png'
    plt.figure(figsize=(8, 5))
    plt.errorbar(reconstruction['n_components'], reconstruction['mean_mse'], yerr=reconstruction['se_mse'], marker='o', capsize=3)
    plt.xlabel('PCA components')
    plt.ylabel('Participant-held-out reconstruction MSE')
    plt.title('Continuous dimensionality')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    path = paths.figures / 'subjective_discovery_pca_loadings.png'
    loading_matrix = loadings.pivot(index='feature', columns='component', values='loading')
    plt.figure(figsize=(8, 5))
    plt.imshow(loading_matrix.to_numpy(), aspect='auto')
    plt.xticks(range(len(loading_matrix.columns)), loading_matrix.columns)
    plt.yticks(range(len(loading_matrix.index)), loading_matrix.index)
    plt.colorbar(label='Loading')
    plt.title('Person-relative subjective PCA loadings')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    path = paths.figures / 'subjective_discovery_gmm_selection.png'
    plt.figure(figsize=(8, 5))
    plt.errorbar(gmm_selection['clusters'], gmm_selection['cv_log_likelihood_mean'], yerr=gmm_selection['cv_log_likelihood_sd'], marker='o', capsize=3)
    plt.xlabel('Gaussian mixture components')
    plt.ylabel('Held-out log likelihood per trial')
    plt.title('Continuous versus discrete subjective structure')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    path = paths.figures / 'subjective_discovery_prediction_r2.png'
    plot_data = prediction_metrics.pivot(index='model', columns='outcome', values='r2').sort_values(by=HEDONIC_OUTCOMES, ascending=False)
    plot_data.plot(kind='bar', figsize=(11, 6))
    plt.ylabel('Participant-held-out R²')
    plt.title('Discovery prediction of hedonic reports')
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    path = paths.figures / 'subjective_discovery_calibration_curve.png'
    plt.figure(figsize=(8, 5))
    for outcome, group in calibration.groupby('outcome'):
        plt.plot(group['calibration_trials'], group['r2'], marker='o', label=outcome)
    plt.xlabel('Labeled calibration trials per participant')
    plt.ylabel('R² on remaining trials')
    plt.title('Explicit person-specific calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    return outputs

def write_validation_spec(path: Path, dimensions: dict, gmm_summary: dict, prediction_summary: dict, correlations: pd.DataFrame) -> None:
    pooled_raw = correlations.loc[correlations['level'].eq('pooled_raw'), 'valence_liking_r'].iloc[0]
    pooled_relative = correlations.loc[correlations['level'].eq('pooled_person_relative'), 'valence_liking_r'].iloc[0]
    text = f"schema_version: 1\nstatus: proposed_after_discovery_before_validation\n\nphase:\n  name: subjective_architecture_validation\n  discovery_completed: true\n  validation_opened: false\n  confirmation_opened: false\n\nprespecified_inputs:\n  latent_features:{chr(10).join((f'    - {feature}' for feature in CORE_PROFILE))}\n  auxiliary_appraisal:\n    - familiarity\n  hedonic_outcomes:\n    - valence\n    - liking\n  contextual_descriptors:\n    - context\n    - emotion_category\n\ncontinuous_structure:\n  selected_dimensions: {dimensions['selected_dimensions']}\n  parallel_analysis_dimensions: {dimensions['parallel_dimensions']}\n  reconstruction_one_se_dimensions: {dimensions['cv_one_se_dimensions']}\n  pc1_explained_variance_ratio: {dimensions['pc1_explained_variance_ratio']:.8f}\n  relative_reconstruction_improvement_over_pc1: {dimensions['relative_reconstruction_improvement']:.8f}\n\ndiscrete_structure:\n  best_candidate_components: {gmm_summary['best_candidate_clusters']}\n  discrete_supported_in_discovery: {str(gmm_summary['discrete_supported']).lower()}\n  preferred_representation: {gmm_summary['preferred_representation']}\n  validation_rule: >\n    A discrete interpretation requires held-out log-likelihood improvement,\n    median participant-bootstrap ARI at least 0.60, and minimum component\n    prevalence at least 0.05.\n\nprediction:\n  selected_model: {prediction_summary['selected_validation_model']}\n  selected_model_mean_discovery_r2: {prediction_summary['selected_model_mean_discovery_r2']:.8f}\n  calibration_base_model: {prediction_summary['calibration_base_model']}\n  primary_metrics:\n    - r2\n    - mae\n    - pearson_r\n  participant_held_out: true\n\ndescriptive_reference:\n  pooled_valence_liking_r: {pooled_raw:.8f}\n  person_relative_valence_liking_r: {pooled_relative:.8f}\n\nvalidation_rules:\n  - do_not_change_selected_dimensions\n  - do_not_change_model_feature_sets\n  - do_not_change_ridge_alpha_grid\n  - do_not_use_confirmation_outcomes\n  - report_all_negative_results\n  - preserve_raw_and_person_relative_analyses"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def build_report(audit: dict, dimensions: dict, stability: pd.DataFrame, gmm_summary: dict, prediction_metrics: pd.DataFrame, prediction_summary: dict, calibration: pd.DataFrame, correlations: pd.DataFrame, category: pd.DataFrame, runtime: float) -> str:
    pooled_raw = correlations.loc[correlations['level'].eq('pooled_raw'), 'valence_liking_r'].iloc[0]
    pooled_relative = correlations.loc[correlations['level'].eq('pooled_person_relative'), 'valence_liking_r'].iloc[0]
    category_only = category.loc[category['aggregation'].eq('emotion_category')]
    categories_crossing_valence = int((category_only['valence_min'].lt(MIDPOINT) & category_only['valence_max'].gt(MIDPOINT)).sum())
    categories_crossing_liking = int((category_only['liking_min'].lt(MIDPOINT) & category_only['liking_max'].gt(MIDPOINT)).sum())
    lines = ['# subjective discovery analysis — discovery subjective architecture', '', f'- Generated: `{now_iso()}`', f'- Runtime: `{runtime:.1f} seconds`', f"- Discovery participants: **{audit['discovery_subjects']}**", f"- Discovery trials: **{audit['discovery_rows']}**", '- Validation outcomes used: **no**', '- Confirmation outcomes used: **no**', '', '## Scientific question', '', 'Is hedonic experience adequately described by one scalar report axis, or do multiple person-relative subjective configurations carry reproducible information about valence and liking?', '', '## Continuous subjective geometry', '', f"- PC1 explained variance: **{dimensions['pc1_explained_variance_ratio']:.3f}**", f"- Parallel-analysis dimensions: **{dimensions['parallel_dimensions']}**", f"- Reconstruction one-SE dimensions: **{dimensions['cv_one_se_dimensions']}**", f"- prespecified discovery dimensionality: **{dimensions['selected_dimensions']}**", f"- Reconstruction improvement over one component: **{dimensions['relative_reconstruction_improvement']:.1%}**", '', 'Participant-bootstrap component congruence:', '', '| Component | Median | 5th percentile | 95th percentile |', '|---:|---:|---:|---:|']
    for _, row in stability.iterrows():
        lines.append(f"| {int(row['component'])} | {row['median_congruence']:.3f} | {row['q05_congruence']:.3f} | {row['q95_congruence']:.3f} |")
    lines.extend(['', '## Continuous versus discrete organization', '', f"- Best GMM candidate: **K={gmm_summary['best_candidate_clusters']}**", f"- Held-out Δ log-likelihood versus K=1: **{gmm_summary['delta_cv_log_likelihood_vs_k1']:.4f} per trial**", f"- Median bootstrap ARI: **{gmm_summary['best_stability_ari_median']:.3f}**", f"- Smallest component fraction: **{gmm_summary['best_min_cluster_fraction']:.3f}**", f"- NMI with nominal emotion category: **{gmm_summary['best_category_nmi']:.3f}**", f"- Discrete-support criterion met: **{gmm_summary['discrete_supported']}**", f"- Preferred discovery representation: **{gmm_summary['preferred_representation']}**", '', '## Hedonic report relation', '', f'- Pooled valence–liking correlation: **{pooled_raw:.3f}**', f'- Person-relative valence–liking correlation: **{pooled_relative:.3f}**', f'- Categories spanning both sides of the valence midpoint: **{categories_crossing_valence}/{len(category_only)}**', f'- Categories spanning both sides of the liking midpoint: **{categories_crossing_liking}/{len(category_only)}**', '', '## Participant-held-out prediction', '', '| Outcome | Model | R² | MAE | Pearson r |', '|---|---|---:|---:|---:|'])
    ordered = prediction_metrics.sort_values(['outcome', 'r2'], ascending=[True, False])
    for _, row in ordered.iterrows():
        lines.append(f"| `{row['outcome']}` | `{row['model']}` | {row['r2']:.3f} | {row['mae']:.3f} | {row['pearson_r']:.3f} |")
    lines.extend(['', f"Selected model for validation: **`{prediction_summary['selected_validation_model']}`**.", '', '## Explicit person calibration', '', f"Calibration base model: **`{prediction_summary['calibration_base_model']}`**.", '', '| Outcome | Calibration trials | Evaluation trials | R² | MAE |', '|---|---:|---:|---:|---:|'])
    for _, row in calibration.sort_values(['outcome', 'calibration_trials']).iterrows():
        lines.append(f"| `{row['outcome']}` | {int(row['calibration_trials'])} | {int(row['evaluation_trials'])} | {row['r2']:.3f} | {row['mae']:.3f} |")
    lines.extend(['', '## Interpretation boundary', '', 'These results characterize subjective report organization. They do not establish neural or physiological mechanisms. The latent features are reports, not biological states; category labels were not used for unsupervised PCA or GMM fitting.', '', '## Next gate', '', 'The proposed validation specification is now prespecified. subjective validation analysis may evaluate the selected dimensionality and prediction model on validation participants exactly once. Confirmation remains closed.', ''])
    return ''.join(lines)

def main() -> int:
    args = parse_args()
    started = time.time()
    paths = Paths(args.project.expanduser().resolve())
    output_targets = [paths.tables / 'subjective_discovery_discovery_subjective_table.csv', paths.tables / 'subjective_discovery_category_overlap.csv', paths.tables / 'subjective_discovery_person_context_summary.csv', paths.tables / 'subjective_discovery_correlations.csv', paths.tables / 'subjective_discovery_parallel_analysis.csv', paths.tables / 'subjective_discovery_reconstruction_cv.csv', paths.tables / 'subjective_discovery_pca_loadings.csv', paths.tables / 'subjective_discovery_pca_scores.csv', paths.tables / 'subjective_discovery_pca_stability.csv', paths.tables / 'subjective_discovery_gmm_selection.csv', paths.tables / 'subjective_discovery_gmm_configuration_summary.csv', paths.tables / 'subjective_discovery_gmm_assignments.csv', paths.tables / 'subjective_discovery_prediction_metrics.csv', paths.tables / 'subjective_discovery_prediction_fold_metrics.csv', paths.tables / 'subjective_discovery_oof_predictions.csv', paths.tables / 'subjective_discovery_calibration_curve.csv', paths.qc / 'subjective_discovery_input_audit.csv', paths.qc / 'subjective_discovery_lockbox_audit.csv', paths.reports / 'subjective_discovery_subjective_architecture_report.md', paths.config / 'subjective_validation_specification.proposed.yaml', paths.models / 'subjective_discovery_subjective_models.joblib']
    existing = [path for path in output_targets if path.exists()]
    if existing and (not args.force):
        print('ERROR: subjective discovery analysis outputs already exist. Refusing to overwrite.\nUse --force only before validation has been opened.\n  - ' + '\n  - '.join((str(path) for path in existing)), file=sys.stderr)
        return 1
    log_path = paths.logs / 'subjective_discovery_subjective_architecture.log'
    with log_path.open('w', encoding='utf-8') as log_handle:
        log('Validating prespecified inputs.', log_handle)
        prespecified = validate_inputs(paths)
        log('Preparing discovery-only analysis table.', log_handle)
        discovery, split, audit = prepare_discovery(paths)
        discovery.to_csv(paths.tables / 'subjective_discovery_discovery_subjective_table.csv', index=False)
        audit_rows = input_audit_rows(discovery, audit)
        write_csv(paths.qc / 'subjective_discovery_input_audit.csv', audit_rows, ['check', 'value', 'expected', 'passed'])
        if not all((bool(row['passed']) for row in audit_rows)):
            raise RuntimeError('subjective discovery analysis input audit failed.')
        log('Computing descriptive subjective architecture.', log_handle)
        category = category_overlap(discovery)
        person_context = person_context_summary(discovery)
        correlations = correlation_summary(discovery)
        category.to_csv(paths.tables / 'subjective_discovery_category_overlap.csv', index=False)
        person_context.to_csv(paths.tables / 'subjective_discovery_person_context_summary.csv', index=False)
        correlations.to_csv(paths.tables / 'subjective_discovery_correlations.csv', index=False)
        rel_columns = [f'rel_{column}' for column in CORE_PROFILE]
        X_relative = discovery[rel_columns].to_numpy()
        subject_ids = discovery['subject_id'].to_numpy()
        log('Running parallel analysis.', log_handle)
        parallel, full_scaler, full_pca, full_scores_all = parallel_analysis(X_relative, subject_ids, args.parallel_iterations)
        reconstruction = reconstruction_cv(X_relative, subject_ids)
        dimensions = choose_dimensions(parallel, reconstruction)
        selected_n = dimensions['selected_dimensions']
        parallel.to_csv(paths.tables / 'subjective_discovery_parallel_analysis.csv', index=False)
        reconstruction.to_csv(paths.tables / 'subjective_discovery_reconstruction_cv.csv', index=False)
        selected_pca = PCA(n_components=selected_n, random_state=RANDOM_SEED)
        selected_scores = selected_pca.fit_transform(full_scaler.transform(X_relative))
        loadings_matrix = selected_pca.components_.T
        loading_rows = []
        for feature_index, feature in enumerate(CORE_PROFILE):
            for component in range(selected_n):
                loading_rows.append({'feature': feature, 'component': component + 1, 'loading': float(loadings_matrix[feature_index, component])})
        loadings = pd.DataFrame(loading_rows)
        loadings.to_csv(paths.tables / 'subjective_discovery_pca_loadings.csv', index=False)
        score_table = discovery[PRIMARY_KEY + ['context', 'emotion_category', 'valence', 'liking']].copy()
        for component in range(selected_n):
            score_table[f'subjective_pc{component + 1}'] = selected_scores[:, component]
        score_table.to_csv(paths.tables / 'subjective_discovery_pca_scores.csv', index=False)
        log('Estimating PCA stability.', log_handle)
        stability = pca_stability(discovery, selected_n, loadings_matrix, args.pca_bootstrap)
        stability.to_csv(paths.tables / 'subjective_discovery_pca_stability.csv', index=False)
        log('Comparing continuous and discrete structure.', log_handle)
        gmm_selection, gmm_configuration_summary, best_gmm, gmm_output = gmm_analysis(discovery, selected_n, full_scaler, selected_pca, args.gmm_bootstrap)
        gmm_summary = {key: value for key, value in gmm_output.items() if key not in {'assignments', 'full_scores'}}
        gmm_selection.to_csv(paths.tables / 'subjective_discovery_gmm_selection.csv', index=False)
        gmm_configuration_summary.to_csv(paths.tables / 'subjective_discovery_gmm_configuration_summary.csv', index=False)
        gmm_output['assignments'].to_csv(paths.tables / 'subjective_discovery_gmm_assignments.csv', index=False)
        log('Running participant-held-out hedonic prediction.', log_handle)
        prediction_metrics, fold_metrics, predictions, prediction_summary = prediction_benchmark(discovery, selected_n, args.metric_bootstrap)
        prediction_metrics.to_csv(paths.tables / 'subjective_discovery_prediction_metrics.csv', index=False)
        fold_metrics.to_csv(paths.tables / 'subjective_discovery_prediction_fold_metrics.csv', index=False)
        predictions.to_csv(paths.tables / 'subjective_discovery_oof_predictions.csv', index=False)
        calibration = calibration_curve(predictions, prediction_summary['calibration_base_model'])
        calibration.to_csv(paths.tables / 'subjective_discovery_calibration_curve.csv', index=False)
        log('Writing validation specification.', log_handle)
        validation_spec = paths.config / 'subjective_validation_specification.proposed.yaml'
        write_validation_spec(validation_spec, dimensions, gmm_summary, prediction_summary, correlations)
        model_bundle = {'created_at': now_iso(), 'core_profile': CORE_PROFILE, 'auxiliary_reports': AUXILIARY_REPORTS, 'selected_dimensions': selected_n, 'scaler': full_scaler, 'pca': selected_pca, 'gmm': best_gmm, 'gmm_summary': gmm_summary, 'prediction_summary': prediction_summary, 'canonical_sha256': prespecified['canonical_sha256'], 'split_sha256': prespecified['split_sha256']}
        joblib.dump(model_bundle, paths.models / 'subjective_discovery_subjective_models.joblib')
        figure_paths = save_figures(paths, parallel, reconstruction, loadings, gmm_selection, prediction_metrics, calibration)
        analyzed_subjects = set(discovery['subject_id'])
        validation_subjects = set(split.loc[split['split'].eq('validation'), 'subject_id'])
        confirmation_subjects = set(split.loc[split['split'].eq('confirmation'), 'subject_id'])
        lockbox_rows = [{'check': 'analyzed_subjects_are_discovery_only', 'value': len(analyzed_subjects), 'expected': 36, 'passed': analyzed_subjects.isdisjoint(validation_subjects | confirmation_subjects)}, {'check': 'validation_subjects_in_analysis', 'value': len(analyzed_subjects & validation_subjects), 'expected': 0, 'passed': not analyzed_subjects & validation_subjects}, {'check': 'confirmation_subjects_in_analysis', 'value': len(analyzed_subjects & confirmation_subjects), 'expected': 0, 'passed': not analyzed_subjects & confirmation_subjects}, {'check': 'validation_spec_status_proposed', 'value': validation_spec.exists(), 'expected': True, 'passed': validation_spec.exists()}]
        write_csv(paths.qc / 'subjective_discovery_lockbox_audit.csv', lockbox_rows, ['check', 'value', 'expected', 'passed'])
        if not all((bool(row['passed']) for row in lockbox_rows)):
            raise RuntimeError('Lockbox audit failed.')
        report_path = paths.reports / 'subjective_discovery_subjective_architecture_report.md'
        report_path.write_text(build_report(audit, dimensions, stability, gmm_summary, prediction_metrics, prediction_summary, calibration, correlations, category, time.time() - started), encoding='utf-8')
        summary_path = paths.qc / 'subjective_discovery_summary.json'
        summary_path.write_text(json.dumps({'generated_at': now_iso(), 'audit': audit, 'dimensions': dimensions, 'gmm': gmm_summary, 'prediction': prediction_summary, 'validation_spec': str(validation_spec), 'confirmation_opened': False, 'validation_opened': False, 'runtime_seconds': round(time.time() - started, 2)}, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        log(f'Report written: {report_path}', log_handle)
    print()
    print('=' * 78)
    print('PHASE 1A SUBJECTIVE ARCHITECTURE COMPLETED')
    print('=' * 78)
    print(f"Discovery trials:       {audit['discovery_rows']}")
    print(f"Discovery participants: {audit['discovery_subjects']}")
    print(f'Selected dimensions:    {selected_n}')
    print(f"Preferred structure:    {gmm_summary['preferred_representation']}")
    print(f"Validation model:       {prediction_summary['selected_validation_model']}")
    print(f"Report:                 {paths.reports / 'subjective_discovery_subjective_architecture_report.md'}")
    print(f'Runtime:                {time.time() - started:.1f} seconds')
    print()
    print('Validation and confirmation remain unopened.')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    raise SystemExit(main())