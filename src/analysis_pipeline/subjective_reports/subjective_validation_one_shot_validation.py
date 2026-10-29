# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
subjective validation analysis — one-shot validation of subjective architecture.

This script opens the 12 validation participants exactly once under the prespecified
Phase 1 validation specification. Confirmation participants remain closed.

prespecified primary tests
--------------------
1. Apply the discovery-fitted, two-dimensional person-relative geometry.
2. Evaluate the prespecified profile_raw8 ridge model for valence and liking.
3. Compare it against mean and category/context baselines.
4. Treat the five-component GMM only as a secondary density summary.
5. Evaluate 20-trial person-intercept calibration only as exploratory.

The script does not tune features, dimensions, thresholds, cluster number, or
the alpha grid using validation outcomes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
try:
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.linalg import subspace_angles
    from scipy.stats import pearsonr
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(f'Missing scientific dependency: {exc}\nInstall numpy, pandas, scipy, scikit-learn, matplotlib, and joblib.')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
CORE7 = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal']
PROFILE_RAW8 = CORE7 + ['familiarity']
OUTCOMES = ['valence', 'liking']
CONTEXT_COLUMNS = ['context', 'emotion_category']
PRIMARY_KEY = ['subject_id', 'trial_number_global']
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
prespecified_DIMENSIONS = 2
prespecified_GMM_K = 5
prespecified_CALIBRATION_TRIALS = 20
GEOMETRY_MAX_ANGLE_DEGREES = 30.0
GEOMETRY_MIN_RECONSTRUCTION_IMPROVEMENT = 0.15
PREDICTION_MIN_MEAN_R2 = 0.5
PREDICTION_MIN_OUTCOME_R2 = 0.3
DISCRETE_MAX_AMBIGUITY = 0.25
DISCRETE_MIN_COMPONENT_FRACTION = 0.05
DISCRETE_MIN_CONTEXT_ARI = 0.5
SEED = 1542

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the one-shot subjective validation analysis validation.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--parallel-iterations', type=int, default=500)
    parser.add_argument('--metric-bootstrap', type=int, default=1000)
    parser.add_argument('--calibration-repeats', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted run only when the validation-open state matches the current script and prespecified inputs.')
    parser.add_argument('--rerun-identical', action='store_true', help='Reproduce a completed validation run without changing the specification. The original validation result remains primary.')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def log(message: str, handle=None) -> None:
    line = f'[{now_iso()}] {message}'
    print(line, flush=True)
    if handle is not None:
        handle.write(line + '')
        handle.flush()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def safe_corr(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3:
        return float('nan')
    if np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return float('nan')
    return float(pearsonr(observed, predicted).statistic)

def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {'r2': float(r2_score(observed, predicted)), 'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(math.sqrt(mean_squared_error(observed, predicted))), 'pearson_r': safe_corr(observed, predicted)}

def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Required files are missing:\n  - ' + '\n  - '.join((str(path) for path in missing)))

def load_split_map(split_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    split = pd.read_csv(split_path)
    if split['subject_id'].duplicated().any():
        raise RuntimeError('Split manifest contains duplicate participant IDs.')
    split_map = dict(zip(split['subject_id'].astype(str), split['split'].astype(str)))
    return (split, split_map)

def load_selected_rows(canonical_path: Path, split_map: dict[str, str], allowed_splits: set[str]) -> pd.DataFrame:
    """
    Read rows sequentially and retain only discovery/validation participants.

    Confirmation values are never added to the analysis frame.
    """
    rows = []
    with canonical_path.open('r', encoding='utf-8-sig', errors='strict', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        required = set(PRIMARY_KEY + CORE7 + ['familiarity'] + OUTCOMES + CONTEXT_COLUMNS + ['stimulus_id'])
        missing = sorted(required - set(columns))
        if missing:
            raise RuntimeError(f'Canonical table is missing columns: {missing}')
        for row in reader:
            subject = str(row.get('subject_id', '')).strip()
            split = split_map.get(subject)
            if split not in allowed_splits:
                continue
            selected = {column: row.get(column, '') for column in columns}
            selected['split'] = split
            rows.append(selected)
    frame = pd.DataFrame(rows)
    numeric = CORE7 + ['familiarity'] + OUTCOMES
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if frame[numeric].isna().any().any():
        counts = frame[numeric].isna().sum().loc[lambda values: values > 0].to_dict()
        raise RuntimeError(f'Missing or nonnumeric ratings: {counts}')
    frame['trial_number_global'] = pd.to_numeric(frame['trial_number_global'], errors='raise')
    if frame.duplicated(PRIMARY_KEY).any():
        raise RuntimeError('Selected canonical rows have duplicate primary keys.')
    return frame

def within_person_z(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    output = np.zeros((len(frame), len(columns)), dtype=float)
    for index, column in enumerate(columns):
        values = frame[column].astype(float)
        grouped = values.groupby(frame['subject_id'])
        means = grouped.transform('mean')
        scales = grouped.transform(lambda x: x.std(ddof=0))
        scales = scales.where(scales > 1e-12, 1.0)
        output[:, index] = ((values - means) / scales).to_numpy()
    return output

def parallel_analysis_count(X: np.ndarray, subject_ids: np.ndarray, iterations: int) -> tuple[pd.DataFrame, int]:
    scaler = StandardScaler()
    observed_scaled = scaler.fit_transform(X)
    observed_pca = PCA(random_state=SEED)
    observed_pca.fit(observed_scaled)
    observed = observed_pca.explained_variance_
    rng = np.random.default_rng(SEED + 100)
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    null_values = np.zeros((iterations, X.shape[1]), dtype=float)
    for iteration in range(iterations):
        null = X.copy()
        for feature_index in range(X.shape[1]):
            for indices in subject_indices.values():
                null[indices, feature_index] = rng.permutation(null[indices, feature_index])
        null_pca = PCA(random_state=SEED)
        null_pca.fit(StandardScaler().fit_transform(null))
        null_values[iteration] = null_pca.explained_variance_
    null_95 = np.quantile(null_values, 0.95, axis=0)
    selected = observed > null_95
    count = max(1, int(selected.sum()))
    table = pd.DataFrame({'component': np.arange(1, len(observed) + 1), 'validation_eigenvalue': observed, 'permuted_null_95': null_95, 'validation_explained_variance_ratio': observed_pca.explained_variance_ratio_, 'selected': selected})
    return (table, count)

def geometry_validation(discovery: pd.DataFrame, validation: pd.DataFrame, parallel_iterations: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, StandardScaler, PCA, np.ndarray, np.ndarray]:
    X_discovery = within_person_z(discovery, CORE7)
    X_validation = within_person_z(validation, CORE7)
    discovery_scaler = StandardScaler()
    discovery_scaled = discovery_scaler.fit_transform(X_discovery)
    validation_scaled = discovery_scaler.transform(X_validation)
    discovery_pca = PCA(n_components=len(CORE7), random_state=SEED)
    discovery_pca.fit(discovery_scaled)
    discovery_scores = discovery_pca.transform(discovery_scaled)
    validation_scores = discovery_pca.transform(validation_scaled)
    one_component = np.zeros_like(validation_scores)
    one_component[:, 0] = validation_scores[:, 0]
    two_component = np.zeros_like(validation_scores)
    two_component[:, :2] = validation_scores[:, :2]
    reconstruction_1 = discovery_pca.inverse_transform(one_component)
    reconstruction_2 = discovery_pca.inverse_transform(two_component)
    mse_1 = float(mean_squared_error(validation_scaled, reconstruction_1))
    mse_2 = float(mean_squared_error(validation_scaled, reconstruction_2))
    improvement = (mse_1 - mse_2) / mse_1 if mse_1 > 0 else 0.0
    validation_pca = PCA(n_components=prespecified_DIMENSIONS, random_state=SEED)
    validation_pca.fit(validation_scaled)
    angles = np.degrees(subspace_angles(discovery_pca.components_[:2].T, validation_pca.components_[:2].T))
    total_variance = float(np.var(validation_scaled, axis=0, ddof=1).sum())
    captured_variance = float(np.var(validation_scores[:, :2], axis=0, ddof=1).sum())
    prespecified_variance_fraction = captured_variance / total_variance if total_variance > 0 else float('nan')
    parallel_table, parallel_count = parallel_analysis_count(X_validation, validation['subject_id'].to_numpy(), parallel_iterations)
    geometry_passed = bool(parallel_count == prespecified_DIMENSIONS and improvement >= GEOMETRY_MIN_RECONSTRUCTION_IMPROVEMENT and (float(np.max(angles)) <= GEOMETRY_MAX_ANGLE_DEGREES))
    validation_score_table = validation[PRIMARY_KEY + ['subject_id', 'context', 'emotion_category', 'valence', 'liking']].copy()
    validation_score_table['subjective_pc1'] = validation_scores[:, 0]
    validation_score_table['subjective_pc2'] = validation_scores[:, 1]
    summary_rows = [{'metric': 'parallel_dimensions', 'value': parallel_count, 'threshold': prespecified_DIMENSIONS, 'passed': parallel_count == prespecified_DIMENSIONS}, {'metric': 'one_component_reconstruction_mse', 'value': mse_1, 'threshold': '', 'passed': True}, {'metric': 'two_component_reconstruction_mse', 'value': mse_2, 'threshold': '', 'passed': True}, {'metric': 'relative_reconstruction_improvement', 'value': improvement, 'threshold': GEOMETRY_MIN_RECONSTRUCTION_IMPROVEMENT, 'passed': improvement >= GEOMETRY_MIN_RECONSTRUCTION_IMPROVEMENT}, {'metric': 'max_subspace_angle_degrees', 'value': float(np.max(angles)), 'threshold': GEOMETRY_MAX_ANGLE_DEGREES, 'passed': float(np.max(angles)) <= GEOMETRY_MAX_ANGLE_DEGREES}, {'metric': 'mean_subspace_angle_degrees', 'value': float(np.mean(angles)), 'threshold': '', 'passed': True}, {'metric': 'prespecified_two_axis_variance_fraction', 'value': prespecified_variance_fraction, 'threshold': '', 'passed': True}, {'metric': 'geometry_replication', 'value': geometry_passed, 'threshold': True, 'passed': geometry_passed}]
    return (pd.DataFrame(summary_rows), parallel_table, {'parallel_dimensions': parallel_count, 'reconstruction_mse_1d': mse_1, 'reconstruction_mse_2d': mse_2, 'relative_reconstruction_improvement': improvement, 'max_subspace_angle_degrees': float(np.max(angles)), 'mean_subspace_angle_degrees': float(np.mean(angles)), 'prespecified_two_axis_variance_fraction': prespecified_variance_fraction, 'replicated': geometry_passed}, discovery_scaler, discovery_pca, discovery_scores[:, :2], validation_scores[:, :2])

def make_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(('numeric', StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns))
    return Pipeline([('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop')), ('model', Ridge())])

def discovery_tuned_model(discovery: pd.DataFrame, outcome: str, numeric_columns: list[str], categorical_columns: list[str]) -> tuple[Pipeline, float]:
    pipeline = make_pipeline(numeric_columns, categorical_columns)
    search = GridSearchCV(pipeline, param_grid={'model__alpha': RIDGE_ALPHAS}, scoring='neg_mean_squared_error', cv=GroupKFold(n_splits=6), n_jobs=1, refit=True)
    search.fit(discovery, discovery[outcome].to_numpy(), groups=discovery['subject_id'].to_numpy())
    return (search.best_estimator_, float(search.best_params_['model__alpha']))

def participant_bootstrap_metrics(frame: pd.DataFrame, iterations: int) -> dict:
    rng = np.random.default_rng(SEED + 200)
    subjects = frame['subject_id'].unique()
    parts = {subject: frame.loc[frame['subject_id'].eq(subject)] for subject in subjects}
    values = defaultdict(list)
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        bootstrap = pd.concat([parts[subject] for subject in sampled], ignore_index=True)
        result = metrics(bootstrap['observed'].to_numpy(), bootstrap['predicted'].to_numpy())
        for metric, value in result.items():
            if np.isfinite(value):
                values[metric].append(value)
    output = {}
    for metric, samples in values.items():
        array = np.asarray(samples, dtype=float)
        output[f'{metric}_ci_low'] = float(np.quantile(array, 0.025))
        output[f'{metric}_ci_high'] = float(np.quantile(array, 0.975))
    return output

def prediction_validation(discovery: pd.DataFrame, validation: pd.DataFrame, bootstrap_iterations: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    specifications = {'mean': ([], []), 'category_context': ([], CONTEXT_COLUMNS), 'profile_raw8': (PROFILE_RAW8, [])}
    prediction_rows = []
    trained_models = {}
    selected_alphas = {}
    for outcome in OUTCOMES:
        for model_name, (numeric_columns, categorical_columns) in specifications.items():
            if model_name == 'mean':
                predicted = np.full(len(validation), discovery[outcome].mean())
                alpha = float('nan')
                model = None
            else:
                model, alpha = discovery_tuned_model(discovery, outcome, numeric_columns, categorical_columns)
                predicted = model.predict(validation)
                trained_models[f'{outcome}__{model_name}'] = model
            selected_alphas[f'{outcome}__{model_name}'] = alpha
            for row_index, prediction in enumerate(predicted):
                source = validation.iloc[row_index]
                prediction_rows.append({'subject_id': source['subject_id'], 'trial_number_global': source['trial_number_global'], 'context': source['context'], 'emotion_category': source['emotion_category'], 'outcome': outcome, 'model': model_name, 'alpha_selected_in_discovery': alpha, 'observed': source[outcome], 'predicted': float(prediction)})
    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for (outcome, model_name), group in predictions.groupby(['outcome', 'model']):
        point = metrics(group['observed'].to_numpy(), group['predicted'].to_numpy())
        confidence = participant_bootstrap_metrics(group, bootstrap_iterations)
        metric_rows.append({'scope': 'overall', 'context': 'all', 'outcome': outcome, 'model': model_name, 'n_trials': len(group), 'n_subjects': group['subject_id'].nunique(), **point, **confidence})
        for context, context_group in group.groupby('context'):
            metric_rows.append({'scope': 'context', 'context': context, 'outcome': outcome, 'model': model_name, 'n_trials': len(context_group), 'n_subjects': context_group['subject_id'].nunique(), **metrics(context_group['observed'].to_numpy(), context_group['predicted'].to_numpy()), 'r2_ci_low': float('nan'), 'r2_ci_high': float('nan'), 'mae_ci_low': float('nan'), 'mae_ci_high': float('nan'), 'rmse_ci_low': float('nan'), 'rmse_ci_high': float('nan'), 'pearson_r_ci_low': float('nan'), 'pearson_r_ci_high': float('nan')})
    metric_table = pd.DataFrame(metric_rows)
    selected = metric_table.loc[metric_table['scope'].eq('overall') & metric_table['model'].eq('profile_raw8')]
    baseline = metric_table.loc[metric_table['scope'].eq('overall') & metric_table['model'].eq('category_context')]
    selected_r2 = dict(zip(selected['outcome'], selected['r2']))
    baseline_r2 = dict(zip(baseline['outcome'], baseline['r2']))
    mean_r2 = float(np.mean([selected_r2[outcome] for outcome in OUTCOMES]))
    beats_baseline = all((selected_r2[outcome] > baseline_r2[outcome] for outcome in OUTCOMES))
    prediction_replicated = bool(mean_r2 >= PREDICTION_MIN_MEAN_R2 and all((selected_r2[outcome] >= PREDICTION_MIN_OUTCOME_R2 for outcome in OUTCOMES)) and beats_baseline)
    summary = {'selected_model': 'profile_raw8', 'selected_r2': {outcome: float(selected_r2[outcome]) for outcome in OUTCOMES}, 'category_context_r2': {outcome: float(baseline_r2[outcome]) for outcome in OUTCOMES}, 'mean_selected_r2': mean_r2, 'beats_category_context_both_outcomes': beats_baseline, 'replicated': prediction_replicated, 'thresholds': {'minimum_mean_r2': PREDICTION_MIN_MEAN_R2, 'minimum_each_outcome_r2': PREDICTION_MIN_OUTCOME_R2}, 'selected_alphas': selected_alphas}
    return (metric_table, predictions, summary, trained_models)

def discrete_validation(discovery: pd.DataFrame, validation: pd.DataFrame, discovery_scores: np.ndarray, validation_scores: np.ndarray, discovery_null_q95: float) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    one_component = GaussianMixture(n_components=1, covariance_type='full', n_init=10, reg_covar=1e-06, random_state=SEED)
    five_component = GaussianMixture(n_components=prespecified_GMM_K, covariance_type='full', n_init=30, reg_covar=1e-06, random_state=SEED)
    one_component.fit(discovery_scores)
    five_component.fit(discovery_scores)
    ll_k1 = float(one_component.score(validation_scores))
    ll_k5 = float(five_component.score(validation_scores))
    delta_ll = ll_k5 - ll_k1
    probabilities = five_component.predict_proba(validation_scores)
    labels = probabilities.argmax(axis=1)
    max_probability = probabilities.max(axis=1)
    fractions = np.bincount(labels, minlength=prespecified_GMM_K) / len(labels)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1) / math.log(prespecified_GMM_K)
    assignment_table = validation[PRIMARY_KEY + ['subject_id', 'context', 'emotion_category', 'valence', 'liking']].copy()
    assignment_table['candidate_configuration'] = labels + 1
    assignment_table['maximum_posterior_probability'] = max_probability
    assignment_table['normalized_posterior_entropy'] = entropy
    configuration_rows = []
    for configuration, group in assignment_table.groupby('candidate_configuration'):
        configuration_rows.append({'candidate_configuration': int(configuration), 'n_trials': len(group), 'fraction': len(group) / len(assignment_table), 'n_subjects': group['subject_id'].nunique(), 'valence_mean': group['valence'].mean(), 'liking_mean': group['liking'].mean(), 'mean_maximum_posterior': group['maximum_posterior_probability'].mean()})
    context_models = {}
    context_labels = {}
    contexts = sorted(discovery['context'].astype(str).unique())
    for index, context in enumerate(contexts):
        mask = discovery['context'].astype(str).eq(context).to_numpy()
        model = GaussianMixture(n_components=prespecified_GMM_K, covariance_type='full', n_init=20, reg_covar=1e-06, random_state=SEED + 100 + index)
        model.fit(discovery_scores[mask])
        context_models[context] = model
        context_labels[context] = model.predict(validation_scores)
    context_ari = float(adjusted_rand_score(context_labels[contexts[0]], context_labels[contexts[1]])) if len(contexts) == 2 else float('nan')
    ambiguity_fraction = float(np.mean(max_probability < 0.8))
    minimum_fraction = float(fractions.min())
    secondary_criteria_met = bool(delta_ll > discovery_null_q95 and minimum_fraction >= DISCRETE_MIN_COMPONENT_FRACTION and (ambiguity_fraction <= DISCRETE_MAX_AMBIGUITY) and (math.isnan(context_ari) or context_ari >= DISCRETE_MIN_CONTEXT_ARI))
    final_role = 'exploratory_density_summary'
    summary = {'prespecified_k': prespecified_GMM_K, 'validation_log_likelihood_k1': ll_k1, 'validation_log_likelihood_k5': ll_k5, 'validation_delta_log_likelihood': delta_ll, 'discovery_continuous_null_q95': discovery_null_q95, 'minimum_component_fraction': minimum_fraction, 'median_maximum_posterior': float(np.median(max_probability)), 'ambiguity_fraction_below_0_8': ambiguity_fraction, 'mean_normalized_entropy': float(np.mean(entropy)), 'context_assignment_ari': context_ari, 'validation_secondary_criteria_met': secondary_criteria_met, 'discovery_robust_discrete_support': False, 'final_role': final_role}
    models = {'gmm_k1': one_component, 'gmm_k5': five_component, 'context_gmms': context_models}
    return (assignment_table, pd.DataFrame(configuration_rows), summary, models)

def calibration_validation(predictions: pd.DataFrame, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = predictions.loc[predictions['model'].eq('profile_raw8')].copy()
    chronological_rows = []
    random_rows = []
    rng = np.random.default_rng(SEED + 300)
    for outcome in OUTCOMES:
        outcome_frame = base.loc[base['outcome'].eq(outcome)].copy()
        evaluation_parts = []
        for subject, group in outcome_frame.groupby('subject_id'):
            group = group.sort_values('trial_number_global')
            if len(group) <= prespecified_CALIBRATION_TRIALS + 5:
                continue
            calibration = group.iloc[:prespecified_CALIBRATION_TRIALS]
            evaluation = group.iloc[prespecified_CALIBRATION_TRIALS:].copy()
            offset = float((calibration['observed'] - calibration['predicted']).mean())
            evaluation['calibrated_prediction'] = evaluation['predicted'] + offset
            evaluation_parts.append(evaluation)
        chronological = pd.concat(evaluation_parts, ignore_index=True)
        base_metrics = metrics(chronological['observed'].to_numpy(), chronological['predicted'].to_numpy())
        calibrated_metrics = metrics(chronological['observed'].to_numpy(), chronological['calibrated_prediction'].to_numpy())
        chronological_rows.append({'outcome': outcome, 'calibration_trials': prespecified_CALIBRATION_TRIALS, 'evaluation_trials': len(chronological), 'evaluation_subjects': chronological['subject_id'].nunique(), 'base_r2_same_remainder': base_metrics['r2'], 'calibrated_r2': calibrated_metrics['r2'], 'delta_r2': calibrated_metrics['r2'] - base_metrics['r2'], 'base_mae_same_remainder': base_metrics['mae'], 'calibrated_mae': calibrated_metrics['mae'], 'delta_mae': calibrated_metrics['mae'] - base_metrics['mae']})
        for repeat in range(1, repeats + 1):
            random_parts = []
            for subject, group in outcome_frame.groupby('subject_id'):
                if len(group) <= prespecified_CALIBRATION_TRIALS + 5:
                    continue
                indices = group.index.to_numpy()
                calibration_indices = rng.choice(indices, size=prespecified_CALIBRATION_TRIALS, replace=False)
                calibration = group.loc[calibration_indices]
                evaluation = group.drop(index=calibration_indices).copy()
                offset = float((calibration['observed'] - calibration['predicted']).mean())
                evaluation['calibrated_prediction'] = evaluation['predicted'] + offset
                random_parts.append(evaluation)
            random_evaluation = pd.concat(random_parts, ignore_index=True)
            base_metrics = metrics(random_evaluation['observed'].to_numpy(), random_evaluation['predicted'].to_numpy())
            calibrated_metrics = metrics(random_evaluation['observed'].to_numpy(), random_evaluation['calibrated_prediction'].to_numpy())
            random_rows.append({'repeat': repeat, 'outcome': outcome, 'calibration_trials': prespecified_CALIBRATION_TRIALS, 'evaluation_trials': len(random_evaluation), 'delta_r2': calibrated_metrics['r2'] - base_metrics['r2'], 'delta_mae': calibrated_metrics['mae'] - base_metrics['mae']})
    chronological_table = pd.DataFrame(chronological_rows)
    random_frame = pd.DataFrame(random_rows)
    summary_rows = []
    for outcome, group in random_frame.groupby('outcome'):
        summary_rows.append({'outcome': outcome, 'calibration_trials': prespecified_CALIBRATION_TRIALS, 'repeats': len(group), 'delta_r2_median': group['delta_r2'].median(), 'delta_r2_q025': group['delta_r2'].quantile(0.025), 'delta_r2_q975': group['delta_r2'].quantile(0.975), 'delta_mae_median': group['delta_mae'].median(), 'delta_mae_q025': group['delta_mae'].quantile(0.025), 'delta_mae_q975': group['delta_mae'].quantile(0.975), 'probability_mae_improves': float(np.mean(group['delta_mae'] < 0))})
    summary_table = pd.DataFrame(summary_rows)
    return (chronological_table, summary_table, {'role': 'exploratory_only', 'calibration_trials': prespecified_CALIBRATION_TRIALS, 'outcomes_with_q975_mae_improvement': [row['outcome'] for row in summary_rows if row['delta_mae_q975'] < 0]})

def category_overlap_validation(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in validation.groupby('emotion_category'):
        rows.append({'emotion_category': category, 'n_trials': len(group), 'n_subjects': group['subject_id'].nunique(), 'valence_min': group['valence'].min(), 'valence_max': group['valence'].max(), 'valence_below_midpoint_fraction': float(np.mean(group['valence'] < 3.5)), 'valence_above_midpoint_fraction': float(np.mean(group['valence'] > 3.5)), 'liking_min': group['liking'].min(), 'liking_max': group['liking'].max(), 'liking_below_midpoint_fraction': float(np.mean(group['liking'] < 3.5)), 'liking_above_midpoint_fraction': float(np.mean(group['liking'] > 3.5))})
    return pd.DataFrame(rows)

def save_figures(figures_dir: Path, parallel: pd.DataFrame, geometry: dict, prediction_metrics: pd.DataFrame, calibration: pd.DataFrame, configurations: pd.DataFrame) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    path = figures_dir / 'subjective_validation_parallel_analysis.png'
    plt.figure(figsize=(8, 5))
    plt.plot(parallel['component'], parallel['validation_eigenvalue'], marker='o', label='Validation')
    plt.plot(parallel['component'], parallel['permuted_null_95'], marker='o', label='95% within-person null')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Validation dimensionality')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_validation_geometry_reconstruction.png'
    plt.figure(figsize=(7, 5))
    plt.bar(['1 prespecified axis', '2 prespecified axes'], [geometry['reconstruction_mse_1d'], geometry['reconstruction_mse_2d']])
    plt.ylabel('Validation reconstruction MSE')
    plt.title('prespecified subjective geometry')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    overall = prediction_metrics.loc[prediction_metrics['scope'].eq('overall')]
    path = figures_dir / 'subjective_validation_prediction_r2.png'
    pivot = overall.pivot(index='model', columns='outcome', values='r2')
    pivot.plot(kind='bar', figsize=(9, 5))
    plt.axhline(0, linewidth=1)
    plt.ylabel('Validation R²')
    plt.title('prespecified hedonic-report prediction')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_validation_calibration.png'
    plt.figure(figsize=(7, 5))
    plt.bar(calibration['outcome'], calibration['delta_mae_median'])
    plt.axhline(0, linewidth=1)
    plt.ylabel('Median ΔMAE')
    plt.title('Exploratory 20-trial calibration')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_validation_configuration_prevalence.png'
    plt.figure(figsize=(8, 5))
    plt.bar(configurations['candidate_configuration'].astype(str), configurations['fraction'])
    plt.xlabel('Exploratory GMM component')
    plt.ylabel('Validation fraction')
    plt.title('Five-component density summary')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths

def build_report(audit: dict, geometry: dict, prediction: dict, discrete: dict, calibration: dict, prediction_metrics: pd.DataFrame, calibration_table: pd.DataFrame, category_overlap: pd.DataFrame, runtime_seconds: float) -> str:
    selected = prediction_metrics.loc[prediction_metrics['scope'].eq('overall') & prediction_metrics['model'].eq('profile_raw8')].sort_values('outcome')
    categories_crossing_valence = int((category_overlap['valence_min'].lt(3.5) & category_overlap['valence_max'].gt(3.5)).sum())
    categories_crossing_liking = int((category_overlap['liking_min'].lt(3.5) & category_overlap['liking_max'].gt(3.5)).sum())
    lines = ['# subjective validation analysis — one-shot validation', '', f'- Generated: `{now_iso()}`', f'- Runtime: `{runtime_seconds:.1f} seconds`', f"- Validation participants: **{audit['validation_subjects']}**", f"- Validation trials: **{audit['validation_trials']}**", '- Confirmation participants analyzed: **0**', '', '## prespecified decisions', '', '- Primary subjective representation: **continuous, two-dimensional, person-relative**', '- prespecified hedonic-report predictor: **`profile_raw8`**', '- Five-component GMM: **secondary exploratory density summary**', '- Person calibration: **exploratory only**', '', '## Geometry validation', '', f"- Validation parallel-analysis dimensions: **{geometry['parallel_dimensions']}**", f"- prespecified two-axis variance fraction: **{geometry['prespecified_two_axis_variance_fraction']:.3f}**", f"- 1D reconstruction MSE: **{geometry['reconstruction_mse_1d']:.3f}**", f"- 2D reconstruction MSE: **{geometry['reconstruction_mse_2d']:.3f}**", f"- Relative 2D improvement: **{geometry['relative_reconstruction_improvement']:.1%}**", f"- Maximum subspace angle: **{geometry['max_subspace_angle_degrees']:.2f}°**", f"- prespecified geometry replicated: **{geometry['replicated']}**", '', '## prespecified prediction', '', '| Outcome | Validation R² | 95% CI | MAE | Pearson r |', '|---|---:|---:|---:|---:|']
    for _, row in selected.iterrows():
        lines.append(f"| `{row['outcome']}` | {row['r2']:.3f} | [{row['r2_ci_low']:.3f}, {row['r2_ci_high']:.3f}] | {row['mae']:.3f} | {row['pearson_r']:.3f} |")
    lines.extend(['', f"- Mean validation R²: **{prediction['mean_selected_r2']:.3f}**", f"- Beats category/context for both outcomes: **{prediction['beats_category_context_both_outcomes']}**", f"- prespecified prediction replicated: **{prediction['replicated']}**", '', '## Exploratory five-component density summary', '', f"- Validation Δ log likelihood, K=5 versus K=1: **{discrete['validation_delta_log_likelihood']:.4f}**", f"- Discovery continuous-null 95th percentile: **{discrete['discovery_continuous_null_q95']:.4f}**", f"- Minimum component fraction: **{discrete['minimum_component_fraction']:.3f}**", f"- Posterior ambiguity fraction: **{discrete['ambiguity_fraction_below_0_8']:.3f}**", f"- Context assignment ARI: **{discrete['context_assignment_ari']:.3f}**", f"- Secondary validation criteria met: **{discrete['validation_secondary_criteria_met']}**", '', 'Regardless of these secondary results, the GMM cannot be promoted to a biological-state taxonomy because the prespecified discovery robustness criterion was not met.', '', '## Exploratory person calibration', '', '| Outcome | Median ΔR² | 95% interval | Median ΔMAE | P(MAE improves) |', '|---|---:|---:|---:|---:|'])
    for _, row in calibration_table.sort_values('outcome').iterrows():
        lines.append(f"| `{row['outcome']}` | {row['delta_r2_median']:.3f} | [{row['delta_r2_q025']:.3f}, {row['delta_r2_q975']:.3f}] | {row['delta_mae_median']:.3f} | {row['probability_mae_improves']:.3f} |")
    lines.extend(['', '## Category overlap', '', f'- Categories spanning both sides of the valence midpoint: **{categories_crossing_valence}/{len(category_overlap)}**', f'- Categories spanning both sides of the liking midpoint: **{categories_crossing_liking}/{len(category_overlap)}**', '', '## Interpretation boundary', '', 'This validates organization and predictability among subjective reports. It does not establish neural or physiological mechanisms. Confirmation remains untouched for the later multimodal model chosen after the EEG/physiology phases.', ''])
    return ''.join(lines)

def main() -> int:
    args = parse_args()
    project = args.project.expanduser().resolve()
    started = time.time()
    canonical_path = project / 'data/canonical_behavioral_table.csv'
    split_path = project / 'config/participant_split_manifest.v1.csv'
    specification_path = project / 'config/analysis_specification.v1.json'
    validation_spec_path = project / 'analysis_specifications/S067_subjective_validation.yaml'
    sensitivity_summary_path = project / 'outputs/qc/subjective_discovery_summary.json'
    discovery_table_path = project / 'outputs/tables/subjective_discovery_discovery_subjective_table.csv'
    script_path = Path(__file__).resolve()
    require_files([canonical_path, split_path, specification_path, validation_spec_path, sensitivity_summary_path, discovery_table_path])
    tables_dir = project / 'outputs/tables'
    qc_dir = project / 'outputs/qc'
    reports_dir = project / 'outputs/reports'
    figures_dir = project / 'outputs/figures'
    models_dir = project / 'outputs/models'
    logs_dir = project / 'outputs/logs'
    config_dir = project / 'config'
    state_path = config_dir / 'S068_subjective_validation_state.json'
    summary_path = qc_dir / 'subjective_validation_summary.json'
    specification = json.loads(specification_path.read_text(encoding='utf-8'))
    sensitivity = json.loads(sensitivity_summary_path.read_text(encoding='utf-8'))
    canonical_hash = sha256_file(canonical_path)
    split_hash = sha256_file(split_path)
    specification_hash = sha256_file(validation_spec_path)
    script_hash = sha256_file(script_path)
    if canonical_hash != specification['hashes']['canonical_trial_table']:
        raise RuntimeError('Canonical table hash changed after specification.')
    if split_hash != specification['hashes']['participant_split_manifest']:
        raise RuntimeError('Participant split hash changed after specification.')
    if sensitivity.get('validation_opened'):
        raise RuntimeError('Sensitivity summary says validation was already opened.')
    if sensitivity.get('confirmation_opened'):
        raise RuntimeError('Confirmation was already opened.')
    if state_path.exists():
        existing_state = json.loads(state_path.read_text(encoding='utf-8'))
        same_inputs = all([existing_state.get('canonical_sha256') == canonical_hash, existing_state.get('split_sha256') == split_hash, existing_state.get('validation_spec_sha256') == specification_hash, existing_state.get('script_sha256') == script_hash])
        if not same_inputs:
            raise RuntimeError('A validation state exists with different inputs or code.')
        status = existing_state.get('status')
        if status == 'completed' and (not args.rerun_identical):
            raise RuntimeError('Primary validation is already completed. Use --rerun-identical only for exact reproduction.')
        if status == 'opened_running' and (not args.resume):
            raise RuntimeError('An interrupted validation state exists. Use --resume with unchanged code and inputs.')
    split, split_map = load_split_map(split_path)
    selected = load_selected_rows(canonical_path, split_map, {'discovery', 'validation'})
    discovery = selected.loc[selected['split'].eq('discovery')].copy()
    validation = selected.loc[selected['split'].eq('validation')].copy()
    discovery_subjects = set(discovery['subject_id'])
    validation_subjects = set(validation['subject_id'])
    confirmation_subjects = set(split.loc[split['split'].eq('confirmation'), 'subject_id'].astype(str))
    if len(discovery_subjects) != 36:
        raise RuntimeError(f'Expected 36 discovery participants; found {len(discovery_subjects)}.')
    if len(validation_subjects) != 12:
        raise RuntimeError(f'Expected 12 validation participants; found {len(validation_subjects)}.')
    if discovery_subjects & validation_subjects or validation_subjects & confirmation_subjects:
        raise RuntimeError('Participant split overlap detected.')
    state = {'opened_at': now_iso(), 'status': 'opened_running', 'primary_run': not args.rerun_identical, 'rerun_identical': bool(args.rerun_identical), 'canonical_sha256': canonical_hash, 'split_sha256': split_hash, 'validation_spec_sha256': specification_hash, 'script_sha256': script_hash, 'discovery_participants': 36, 'validation_participants': 12, 'confirmation_participants_analyzed': 0}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    log_path = logs_dir / 'subjective_validation_validation.log'
    with log_path.open('w', encoding='utf-8') as log_handle:
        log('Validation opened under prespecified Phase 1 specification.', log_handle)
        audit = {'discovery_subjects': len(discovery_subjects), 'discovery_trials': len(discovery), 'validation_subjects': len(validation_subjects), 'validation_trials': len(validation), 'confirmation_subjects_analyzed': 0, 'confirmation_rows_retained': 0, 'canonical_sha256': canonical_hash, 'split_sha256': split_hash, 'validation_spec_sha256': specification_hash, 'script_sha256': script_hash}
        audit_rows = [{'check': 'discovery_subjects', 'value': audit['discovery_subjects'], 'expected': 36, 'passed': audit['discovery_subjects'] == 36}, {'check': 'validation_subjects', 'value': audit['validation_subjects'], 'expected': 12, 'passed': audit['validation_subjects'] == 12}, {'check': 'confirmation_subjects_analyzed', 'value': 0, 'expected': 0, 'passed': True}, {'check': 'confirmation_rows_retained', 'value': 0, 'expected': 0, 'passed': True}, {'check': 'discovery_validation_disjoint', 'value': True, 'expected': True, 'passed': True}]
        write_csv(qc_dir / 'subjective_validation_lockbox_audit.csv', audit_rows, ['check', 'value', 'expected', 'passed'])
        log('Applying prespecified two-dimensional geometry.', log_handle)
        geometry_table, parallel_table, geometry_summary, discovery_scaler, discovery_pca, discovery_scores, validation_scores = geometry_validation(discovery, validation, args.parallel_iterations)
        geometry_table.to_csv(tables_dir / 'subjective_validation_geometry_validation.csv', index=False)
        parallel_table.to_csv(tables_dir / 'subjective_validation_validation_parallel_analysis.csv', index=False)
        validation_score_table = validation[PRIMARY_KEY + ['subject_id', 'context', 'emotion_category', 'valence', 'liking']].copy()
        validation_score_table['subjective_pc1'] = validation_scores[:, 0]
        validation_score_table['subjective_pc2'] = validation_scores[:, 1]
        validation_score_table.to_csv(tables_dir / 'subjective_validation_validation_subjective_scores.csv', index=False)
        log('Evaluating prespecified profile_raw8 prediction.', log_handle)
        prediction_metrics, predictions, prediction_summary, prediction_models = prediction_validation(discovery, validation, args.metric_bootstrap)
        prediction_metrics.to_csv(tables_dir / 'subjective_validation_prediction_metrics.csv', index=False)
        predictions.to_csv(tables_dir / 'subjective_validation_validation_predictions.csv', index=False)
        log('Evaluating exploratory five-component density summary.', log_handle)
        discovery_null_q95 = float(sensitivity['continuous_null']['null_delta_q95'])
        assignments, configuration_summary, discrete_summary, discrete_models = discrete_validation(discovery, validation, discovery_scores, validation_scores, discovery_null_q95)
        assignments.to_csv(tables_dir / 'subjective_validation_gmm_assignments.csv', index=False)
        configuration_summary.to_csv(tables_dir / 'subjective_validation_gmm_configuration_summary.csv', index=False)
        log('Evaluating exploratory 20-trial calibration.', log_handle)
        calibration_chronological, calibration_random, calibration_summary = calibration_validation(predictions, args.calibration_repeats)
        calibration_chronological.to_csv(tables_dir / 'subjective_validation_calibration_chronological.csv', index=False)
        calibration_random.to_csv(tables_dir / 'subjective_validation_calibration_random_summary.csv', index=False)
        category_overlap = category_overlap_validation(validation)
        category_overlap.to_csv(tables_dir / 'subjective_validation_category_overlap.csv', index=False)
        model_bundle = {'created_at': now_iso(), 'validation_spec_sha256': specification_hash, 'geometry': {'features': CORE7, 'dimensions': prespecified_DIMENSIONS, 'scaler': discovery_scaler, 'pca': discovery_pca}, 'prediction_models': prediction_models, 'prediction_summary': prediction_summary, 'discrete_models': discrete_models, 'discrete_summary': discrete_summary, 'confirmation_opened': False}
        model_path = models_dir / 'subjective_validation_prespecified_validation_models.joblib'
        joblib.dump(model_bundle, model_path)
        overall_primary_replicated = bool(geometry_summary['replicated'] and prediction_summary['replicated'])
        summary = {'generated_at': now_iso(), 'audit': audit, 'prespecified_specification': {'dimensions': prespecified_DIMENSIONS, 'geometry_features': CORE7, 'prediction_model': 'profile_raw8', 'prediction_features': PROFILE_RAW8, 'gmm_k': prespecified_GMM_K, 'calibration_trials': prespecified_CALIBRATION_TRIALS}, 'geometry': geometry_summary, 'prediction': prediction_summary, 'discrete_secondary': discrete_summary, 'calibration_exploratory': calibration_summary, 'primary_subjective_architecture_replicated': overall_primary_replicated, 'validation_opened': True, 'validation_completed': True, 'confirmation_opened': False, 'runtime_seconds': round(time.time() - started, 2)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '', encoding='utf-8')
        figure_paths = save_figures(figures_dir, parallel_table, geometry_summary, prediction_metrics, calibration_random, configuration_summary)
        report_path = reports_dir / 'subjective_validation_validation_report.md'
        report_path.write_text(build_report(audit, geometry_summary, prediction_summary, discrete_summary, calibration_summary, prediction_metrics, calibration_random, category_overlap, time.time() - started), encoding='utf-8')
        log(f'Validation report written: {report_path}', log_handle)
    state.update({'completed_at': now_iso(), 'status': 'completed', 'summary_sha256': sha256_file(summary_path), 'confirmation_opened': False})
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '', encoding='utf-8')
    files = [state_path, qc_dir / 'subjective_validation_lockbox_audit.csv', summary_path, tables_dir / 'subjective_validation_geometry_validation.csv', tables_dir / 'subjective_validation_validation_parallel_analysis.csv', tables_dir / 'subjective_validation_validation_subjective_scores.csv', tables_dir / 'subjective_validation_prediction_metrics.csv', tables_dir / 'subjective_validation_validation_predictions.csv', tables_dir / 'subjective_validation_gmm_assignments.csv', tables_dir / 'subjective_validation_gmm_configuration_summary.csv', tables_dir / 'subjective_validation_calibration_chronological.csv', tables_dir / 'subjective_validation_calibration_random_summary.csv', tables_dir / 'subjective_validation_category_overlap.csv', reports_dir / 'subjective_validation_validation_report.md', log_path, *figure_paths]
    print()
    print('=' * 78)
    print('PHASE 1B ONE-SHOT VALIDATION COMPLETED')
    print('=' * 78)
    print(f"Geometry replicated:      {geometry_summary['replicated']}")
    print(f"Prediction replicated:    {prediction_summary['replicated']}")
    print(f"Primary result replicated: {summary['primary_subjective_architecture_replicated']}")
    print(f"Discrete final role:      {discrete_summary['final_role']}")
    print('Confirmation opened:      False')
    print(f"Report:                   {reports_dir / 'subjective_validation_validation_report.md'}")
    print(f'Runtime:                  {time.time() - started:.1f} seconds')
    print()
    print('Do not rerun with changed code or thresholds. Confirmation remains closed.')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    raise SystemExit(main())