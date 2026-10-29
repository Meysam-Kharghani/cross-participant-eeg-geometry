# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""
subjective sensitivity analysis — discovery sensitivity and specification specification.

This stage remains strictly discovery-only. It resolves four issues before
validation is opened:

1. Test whether the two-dimensional person-relative geometry survives
   alternative within-person normalizations and feature definitions.
2. Replace single-split GMM selection with repeated participant-held-out
   density evaluation, a one-standard-error rule, posterior ambiguity,
   context replication, and a continuous Gaussian-copula null.
3. Recalculate person calibration fairly: calibrated and uncalibrated
   predictions are scored on the same remaining trials.
4. Run interpretable subjective-profile ablations before freezing the exact
   validation specification.

It does not read validation or confirmation outcomes."""
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
from datetime import datetime
from pathlib import Path
from typing import Iterable
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.linalg import subspace_angles
    from scipy.stats import norm, rankdata, pearsonr
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(f'Missing scientific dependency: {exc}\nInstall numpy, pandas, scipy, scikit-learn, and matplotlib.')
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
CORE7 = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal']
POSITIVE3 = ['joy', 'inspiration', 'tenderness']
NEGATIVE3 = ['sadness', 'fear', 'disgust']
OUTCOMES = ['valence', 'liking']
CONTEXT_COLUMNS = ['context', 'emotion_category']
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CALIBRATION_TRIALS = [0, 5, 10, 20, 30]
SEED = 1541

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run discovery-only subjective sensitivity analysis sensitivity audit.')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--gmm-repeats', type=int, default=40)
    parser.add_argument('--gmm-null-iterations', type=int, default=60)
    parser.add_argument('--parallel-iterations', type=int, default=200)
    parser.add_argument('--calibration-repeats', type=int, default=300)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def log(message: str, handle=None) -> None:
    """Write a timestamped message to stdout and optionally to a log file."""
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

def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float('nan')
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float('nan')
    return float(pearsonr(y_true, y_pred).statistic)

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {'r2': float(r2_score(y_true, y_pred)), 'mae': float(mean_absolute_error(y_true, y_pred)), 'rmse': float(math.sqrt(mean_squared_error(y_true, y_pred))), 'pearson_r': safe_corr(y_true, y_pred)}

def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Required files are missing:\n  - ' + '\n  - '.join((str(path) for path in missing)))

def validate_lockbox(project: Path, discovery: pd.DataFrame) -> dict:
    split_path = project / 'config/participant_split_manifest.v1.csv'
    specification_path = project / 'config/analysis_specification.v1.json'
    canonical_path = project / 'data/canonical_behavioral_table.csv'
    subjective_summary_path = project / 'outputs/qc/subjective_discovery_summary.json'
    require_files([split_path, specification_path, canonical_path, subjective_summary_path])
    split = pd.read_csv(split_path)
    specification = json.loads(specification_path.read_text(encoding='utf-8'))
    subjective_summary = json.loads(subjective_summary_path.read_text(encoding='utf-8'))
    if subjective_summary.get('validation_opened'):
        raise RuntimeError('Validation has already been opened. Discovery specification may no longer be changed.')
    if subjective_summary.get('confirmation_opened'):
        raise RuntimeError('Confirmation has already been opened.')
    expected_canonical = specification['hashes']['canonical_trial_table']
    expected_split = specification['hashes']['participant_split_manifest']
    if sha256_file(canonical_path) != expected_canonical:
        raise RuntimeError('Canonical table hash changed.')
    if sha256_file(split_path) != expected_split:
        raise RuntimeError('Split manifest hash changed.')
    discovery_subjects = set(discovery['subject_id'])
    manifest_discovery = set(split.loc[split['split'].eq('discovery'), 'subject_id'])
    validation = set(split.loc[split['split'].eq('validation'), 'subject_id'])
    confirmation = set(split.loc[split['split'].eq('confirmation'), 'subject_id'])
    if discovery_subjects != manifest_discovery:
        raise RuntimeError('Discovery table subject set differs from prespecified split.')
    if discovery_subjects & (validation | confirmation):
        raise RuntimeError('Lockbox contamination detected.')
    return {'discovery_subjects': len(discovery_subjects), 'validation_subjects_read': 0, 'confirmation_subjects_read': 0, 'canonical_sha256': expected_canonical, 'split_sha256': expected_split}

def within_person_transform(frame: pd.DataFrame, columns: list[str], method: str) -> np.ndarray:
    output = np.zeros((len(frame), len(columns)), dtype=float)
    for feature_index, column in enumerate(columns):
        values = frame[column].astype(float)
        grouped = values.groupby(frame['subject_id'])
        if method == 'z':
            center = grouped.transform('mean')
            scale = grouped.transform(lambda x: x.std(ddof=0))
        elif method == 'center':
            center = grouped.transform('mean')
            scale = pd.Series(np.ones(len(frame)), index=frame.index)
        elif method == 'robust':
            center = grouped.transform('median')
            scale = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            fallback = grouped.transform(lambda x: x.std(ddof=0))
            scale = scale.where(scale > 1e-12, fallback)
        else:
            raise ValueError(method)
        scale = scale.where(scale > 1e-12, 1.0)
        output[:, feature_index] = ((values - center) / scale).to_numpy()
    return output

def parallel_count(X: np.ndarray, subject_ids: np.ndarray, iterations: int, seed: int) -> tuple[int, np.ndarray, np.ndarray, PCA, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(random_state=seed)
    pca.fit(X_scaled)
    observed = pca.explained_variance_
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    null_values = np.zeros((iterations, X.shape[1]), dtype=float)
    for iteration in range(iterations):
        null = X.copy()
        for feature in range(X.shape[1]):
            for indices in subject_indices.values():
                null[indices, feature] = rng.permutation(null[indices, feature])
        null_pca = PCA(random_state=seed)
        null_pca.fit(StandardScaler().fit_transform(null))
        null_values[iteration] = null_pca.explained_variance_
    null_95 = np.quantile(null_values, 0.95, axis=0)
    n_selected = max(1, int(np.sum(observed > null_95)))
    return (n_selected, observed, null_95, pca, scaler)

def dimensional_sensitivity(discovery: pd.DataFrame, iterations: int) -> tuple[pd.DataFrame, dict]:
    subject_ids = discovery['subject_id'].to_numpy()
    representations = [('person_z_core7', CORE7, 'z'), ('person_centered_core7', CORE7, 'center'), ('person_robust_core7', CORE7, 'robust'), ('person_z_core6_no_arousal', POSITIVE3 + NEGATIVE3, 'z'), ('person_z_core8_plus_familiarity', CORE7 + ['familiarity'], 'z')]
    fitted = {}
    rows = []
    for index, (name, columns, method) in enumerate(representations):
        X = within_person_transform(discovery, columns, method)
        n_selected, observed, null_95, pca, scaler = parallel_count(X, subject_ids, iterations, SEED + index * 100)
        fitted[name] = {'columns': columns, 'method': method, 'pca': pca, 'scaler': scaler, 'X': X}
        rows.append({'representation': name, 'features': '|'.join(columns), 'normalization': method, 'n_features': len(columns), 'parallel_dimensions': n_selected, 'pc1_explained_variance_ratio': float(pca.explained_variance_ratio_[0]), 'pc2_explained_variance_ratio': float(pca.explained_variance_ratio_[1]) if len(columns) >= 2 else float('nan'), 'first_two_cumulative_variance': float(pca.explained_variance_ratio_[:2].sum()), 'max_principal_angle_degrees_vs_reference': float('nan'), 'mean_principal_angle_degrees_vs_reference': float('nan')})
    reference = fitted['person_z_core7']['pca'].components_[:2].T
    for row in rows:
        name = row['representation']
        item = fitted[name]
        if item['columns'] == CORE7:
            candidate = item['pca'].components_[:2].T
            angles = np.degrees(subspace_angles(reference, candidate))
            row['max_principal_angle_degrees_vs_reference'] = float(np.max(angles))
            row['mean_principal_angle_degrees_vs_reference'] = float(np.mean(angles))
    summary = {'reference_dimensions': int(next((row['parallel_dimensions'] for row in rows if row['representation'] == 'person_z_core7'))), 'core7_all_normalizations_select_two': all((row['parallel_dimensions'] == 2 for row in rows if row['n_features'] == 7)), 'core7_max_subspace_angle': max((row['max_principal_angle_degrees_vs_reference'] for row in rows if row['n_features'] == 7)), 'reference_pca': fitted['person_z_core7']['pca'], 'reference_scaler': fitted['person_z_core7']['scaler'], 'reference_X': fitted['person_z_core7']['X']}
    return (pd.DataFrame(rows), summary)

def repeated_subject_folds(subjects: np.ndarray, repeats: int, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(subjects))
    for repeat in range(1, repeats + 1):
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        fold_subjects = np.array_split(shuffled, n_folds)
        yield (repeat, fold_subjects)

def repeated_gmm_cv(X: np.ndarray, groups: np.ndarray, selected_dimensions: int, repeats: int, max_k: int=8) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    unique_subjects = np.unique(groups)
    rows = []
    for repeat, fold_subjects in repeated_subject_folds(unique_subjects, repeats, 6, SEED + 1000):
        repeat_scores = defaultdict(list)
        for fold, test_subjects in enumerate(fold_subjects, start=1):
            test_mask = np.isin(groups, test_subjects)
            train_mask = ~test_mask
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_test = scaler.transform(X[test_mask])
            pca = PCA(n_components=selected_dimensions, random_state=SEED + repeat + fold)
            train_scores = pca.fit_transform(X_train)
            test_scores = pca.transform(X_test)
            for k in range(1, max_k + 1):
                model = GaussianMixture(n_components=k, covariance_type='full', n_init=4, reg_covar=1e-06, random_state=SEED + repeat * 100 + fold * 10 + k)
                model.fit(train_scores)
                score = float(model.score(test_scores))
                repeat_scores[k].append(score)
                rows.append({'repeat': repeat, 'fold': fold, 'clusters': k, 'test_log_likelihood_per_trial': score, 'n_test_subjects': len(test_subjects), 'n_test_trials': int(test_mask.sum())})
        for k, scores in repeat_scores.items():
            rows.append({'repeat': repeat, 'fold': 0, 'clusters': k, 'test_log_likelihood_per_trial': float(np.mean(scores)), 'n_test_subjects': len(unique_subjects), 'n_test_trials': len(X)})
    detailed = pd.DataFrame(rows)
    repeat_level = detailed.loc[detailed['fold'].eq(0)].copy()
    summary = repeat_level.groupby('clusters').agg(mean_cv_log_likelihood=('test_log_likelihood_per_trial', 'mean'), sd_cv_log_likelihood=('test_log_likelihood_per_trial', 'std'), q05_cv_log_likelihood=('test_log_likelihood_per_trial', lambda x: x.quantile(0.05)), q95_cv_log_likelihood=('test_log_likelihood_per_trial', lambda x: x.quantile(0.95)), repeats=('test_log_likelihood_per_trial', 'size')).reset_index()
    summary['se_cv_log_likelihood'] = summary['sd_cv_log_likelihood'] / np.sqrt(summary['repeats'])
    best = summary.loc[summary['mean_cv_log_likelihood'].idxmax()]
    threshold = best['mean_cv_log_likelihood'] - best['se_cv_log_likelihood']
    eligible = summary.loc[summary['mean_cv_log_likelihood'] >= threshold]
    one_se_k = int(eligible['clusters'].min())
    best_k = int(best['clusters'])
    result = {'best_k': best_k, 'one_se_k': one_se_k, 'best_mean_log_likelihood': float(best['mean_cv_log_likelihood']), 'one_se_threshold': float(threshold), 'delta_best_vs_k1': float(best['mean_cv_log_likelihood'] - summary.loc[summary['clusters'].eq(1), 'mean_cv_log_likelihood'].iloc[0]), 'delta_one_se_vs_k1': float(summary.loc[summary['clusters'].eq(one_se_k), 'mean_cv_log_likelihood'].iloc[0] - summary.loc[summary['clusters'].eq(1), 'mean_cv_log_likelihood'].iloc[0])}
    return (detailed, summary, result)

def empirical_gaussian_copula_sample(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, p = scores.shape
    normal_scores = np.zeros_like(scores, dtype=float)
    for column in range(p):
        ranks = rankdata(scores[:, column], method='average')
        probabilities = (ranks - 0.5) / n
        normal_scores[:, column] = norm.ppf(probabilities)
    correlation = np.corrcoef(normal_scores, rowvar=False)
    correlation = np.atleast_2d(correlation)
    correlation += np.eye(p) * 1e-08
    sampled_normal = rng.multivariate_normal(mean=np.zeros(p), cov=correlation, size=n)
    sampled = np.zeros_like(scores, dtype=float)
    quantile_grid = (np.arange(n) + 0.5) / n
    for column in range(p):
        sorted_values = np.sort(scores[:, column])
        probabilities = norm.cdf(sampled_normal[:, column])
        sampled[:, column] = np.interp(probabilities, quantile_grid, sorted_values, left=sorted_values[0], right=sorted_values[-1])
    return sampled

def fixed_group_cv_delta(scores: np.ndarray, groups: np.ndarray, max_k: int, seed: int) -> float:
    unique_subjects = np.array(sorted(np.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    fold_subjects = np.array_split(unique_subjects, 6)
    means = defaultdict(list)
    for fold, test_subjects in enumerate(fold_subjects, start=1):
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        for k in range(1, max_k + 1):
            model = GaussianMixture(n_components=k, covariance_type='full', n_init=3, reg_covar=1e-06, random_state=seed + fold * 10 + k)
            model.fit(scores[train_mask])
            means[k].append(float(model.score(scores[test_mask])))
    average = {k: float(np.mean(values)) for k, values in means.items()}
    return max((average[k] for k in range(2, max_k + 1))) - average[1]

def continuous_null_test(scores: np.ndarray, groups: np.ndarray, observed_delta: float, iterations: int, max_k: int=8) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED + 2000)
    rows = []
    for iteration in range(1, iterations + 1):
        null_scores = empirical_gaussian_copula_sample(scores, rng)
        delta = fixed_group_cv_delta(null_scores, groups, max_k, SEED + 3000 + iteration)
        rows.append({'iteration': iteration, 'null_max_delta_log_likelihood': delta, 'observed_delta_log_likelihood': observed_delta, 'null_at_least_observed': delta >= observed_delta})
    frame = pd.DataFrame(rows)
    p_value = (1 + int(frame['null_at_least_observed'].sum())) / (iterations + 1)
    return (frame, {'continuous_null_p_value': float(p_value), 'null_delta_median': float(frame['null_max_delta_log_likelihood'].median()), 'null_delta_q95': float(frame['null_max_delta_log_likelihood'].quantile(0.95))})

def full_gmm_diagnostics(scores: np.ndarray, k: int) -> tuple[GaussianMixture, dict]:
    model = GaussianMixture(n_components=k, covariance_type='full', n_init=20, reg_covar=1e-06, random_state=SEED + 4000)
    model.fit(scores)
    probabilities = model.predict_proba(scores)
    labels = probabilities.argmax(axis=1)
    max_probability = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / math.log(k) if k > 1 else np.zeros(len(scores))
    fractions = np.bincount(labels, minlength=k) / len(labels)
    return (model, {'k': k, 'min_cluster_fraction': float(fractions.min()), 'median_max_posterior': float(np.median(max_probability)), 'fraction_max_posterior_below_0_8': float(np.mean(max_probability < 0.8)), 'mean_normalized_entropy': float(np.mean(normalized_entropy))})

def context_replication(discovery: pd.DataFrame, scores: np.ndarray, k: int) -> tuple[pd.DataFrame, dict]:
    contexts = sorted(discovery['context'].dropna().astype(str).unique())
    rows = []
    if len(contexts) != 2:
        return (pd.DataFrame(), {'context_replication_available': False})
    models = {}
    labels_all = {}
    for index, context in enumerate(contexts):
        mask = discovery['context'].astype(str).eq(context).to_numpy()
        model = GaussianMixture(n_components=k, covariance_type='full', n_init=20, reg_covar=1e-06, random_state=SEED + 5000 + index)
        model.fit(scores[mask])
        models[context] = model
        labels_all[context] = model.predict(scores)
        probabilities = model.predict_proba(scores[mask])
        rows.append({'context': context, 'n_trials': int(mask.sum()), 'n_subjects': int(discovery.loc[mask, 'subject_id'].nunique()), 'mean_training_log_likelihood': float(model.score(scores[mask])), 'mean_max_posterior': float(probabilities.max(axis=1).mean())})
    ari = float(adjusted_rand_score(labels_all[contexts[0]], labels_all[contexts[1]]))
    means_a = models[contexts[0]].means_
    means_b = models[contexts[1]].means_
    distance = np.linalg.norm(means_a[:, None, :] - means_b[None, :, :], axis=2)
    from scipy.optimize import linear_sum_assignment
    row_index, col_index = linear_sum_assignment(distance)
    matched_distances = distance[row_index, col_index]
    return (pd.DataFrame(rows), {'context_replication_available': True, 'contexts': contexts, 'all_trial_assignment_ari_between_context_models': ari, 'mean_matched_centroid_distance': float(matched_distances.mean()), 'max_matched_centroid_distance': float(matched_distances.max())})

def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(('numeric', StandardScaler(), numeric))
    if categorical:
        transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([('preprocessor', preprocessor), ('model', Ridge())])

def prediction_ablation(discovery: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    specs = {'mean': ([], []), 'category_context': ([], CONTEXT_COLUMNS), 'arousal_only': (['arousal'], []), 'familiarity_only': (['familiarity'], []), 'positive_only': (POSITIVE3, []), 'negative_only': (NEGATIVE3, []), 'six_emotions': (POSITIVE3 + NEGATIVE3, []), 'profile_core7': (CORE7, []), 'profile_raw8': (CORE7 + ['familiarity'], []), 'profile_raw8_plus_context': (CORE7 + ['familiarity'], CONTEXT_COLUMNS)}
    groups = discovery['subject_id'].to_numpy()
    outer = GroupKFold(n_splits=6)
    prediction_rows = []
    for outcome in OUTCOMES:
        y = discovery[outcome].to_numpy(dtype=float)
        for fold, (train, test) in enumerate(outer.split(discovery, y, groups), start=1):
            train_frame = discovery.iloc[train]
            test_frame = discovery.iloc[test]
            y_train = y[train]
            train_groups = groups[train]
            for model_name, (numeric, categorical) in specs.items():
                if model_name == 'mean':
                    predicted = np.full(len(test), y_train.mean())
                    alpha = float('nan')
                else:
                    pipeline = make_pipeline(numeric, categorical)
                    search = GridSearchCV(pipeline, param_grid={'model__alpha': RIDGE_ALPHAS}, scoring='neg_mean_squared_error', cv=GroupKFold(n_splits=5), n_jobs=1, refit=True)
                    search.fit(train_frame, y_train, groups=train_groups)
                    predicted = search.predict(test_frame)
                    alpha = float(search.best_params_['model__alpha'])
                for source_index, prediction in zip(test, predicted):
                    prediction_rows.append({'subject_id': discovery.iloc[source_index]['subject_id'], 'trial_number_global': discovery.iloc[source_index]['trial_number_global'], 'outcome': outcome, 'model': model_name, 'fold': fold, 'alpha': alpha, 'observed': y[source_index], 'predicted': float(prediction)})
    predictions = pd.DataFrame(prediction_rows)
    rows = []
    for (outcome, model), group in predictions.groupby(['outcome', 'model']):
        rows.append({'outcome': outcome, 'model': model, 'n_trials': len(group), 'n_subjects': group['subject_id'].nunique(), **metrics(group['observed'].to_numpy(), group['predicted'].to_numpy())})
    summary = pd.DataFrame(rows)
    mean_r2 = summary.groupby('model')['r2'].mean().sort_values(ascending=False)
    selected = str(mean_r2.drop(labels=['mean'], errors='ignore').index[0])
    return (summary, predictions, {'selected_model': selected, 'mean_r2_by_model': {model: float(value) for model, value in mean_r2.items()}})

def fair_calibration(oof: pd.DataFrame, base_model: str, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = oof.loc[oof['model'].eq(base_model)].copy()
    chronological_rows = []
    random_rows = []
    rng = np.random.default_rng(SEED + 6000)
    for outcome in OUTCOMES:
        outcome_frame = base.loc[base['outcome'].eq(outcome)].copy()
        for n_calibration in CALIBRATION_TRIALS:
            evaluation_parts = []
            for subject, group in outcome_frame.groupby('subject_id'):
                group = group.sort_values('trial_number_global').copy()
                if n_calibration == 0:
                    evaluation = group.copy()
                    offset = 0.0
                else:
                    if len(group) <= n_calibration + 5:
                        continue
                    calibration = group.iloc[:n_calibration]
                    evaluation = group.iloc[n_calibration:].copy()
                    offset = float((calibration['observed'] - calibration['predicted']).mean())
                evaluation['calibrated_prediction'] = evaluation['predicted'] + offset
                evaluation_parts.append(evaluation)
            evaluation = pd.concat(evaluation_parts, ignore_index=True)
            base_metrics = metrics(evaluation['observed'].to_numpy(), evaluation['predicted'].to_numpy())
            calibrated_metrics = metrics(evaluation['observed'].to_numpy(), evaluation['calibrated_prediction'].to_numpy())
            chronological_rows.append({'scheme': 'chronological_first_n', 'outcome': outcome, 'base_model': base_model, 'calibration_trials': n_calibration, 'evaluation_trials': len(evaluation), 'evaluation_subjects': evaluation['subject_id'].nunique(), 'base_r2_same_remainder': base_metrics['r2'], 'calibrated_r2': calibrated_metrics['r2'], 'delta_r2': calibrated_metrics['r2'] - base_metrics['r2'], 'base_mae_same_remainder': base_metrics['mae'], 'calibrated_mae': calibrated_metrics['mae'], 'delta_mae': calibrated_metrics['mae'] - base_metrics['mae']})
            if n_calibration == 0:
                continue
            for repeat in range(1, repeats + 1):
                evaluation_parts = []
                for subject, group in outcome_frame.groupby('subject_id'):
                    if len(group) <= n_calibration + 5:
                        continue
                    indices = group.index.to_numpy()
                    calibration_indices = rng.choice(indices, size=n_calibration, replace=False)
                    calibration = group.loc[calibration_indices]
                    evaluation = group.drop(index=calibration_indices).copy()
                    offset = float((calibration['observed'] - calibration['predicted']).mean())
                    evaluation['calibrated_prediction'] = evaluation['predicted'] + offset
                    evaluation_parts.append(evaluation)
                evaluation = pd.concat(evaluation_parts, ignore_index=True)
                base_metrics = metrics(evaluation['observed'].to_numpy(), evaluation['predicted'].to_numpy())
                calibrated_metrics = metrics(evaluation['observed'].to_numpy(), evaluation['calibrated_prediction'].to_numpy())
                random_rows.append({'scheme': 'random_n', 'repeat': repeat, 'outcome': outcome, 'base_model': base_model, 'calibration_trials': n_calibration, 'evaluation_trials': len(evaluation), 'delta_r2': calibrated_metrics['r2'] - base_metrics['r2'], 'delta_mae': calibrated_metrics['mae'] - base_metrics['mae']})
    chronological = pd.DataFrame(chronological_rows)
    random_frame = pd.DataFrame(random_rows)
    random_summary_rows = []
    for (outcome, n), group in random_frame.groupby(['outcome', 'calibration_trials']):
        random_summary_rows.append({'outcome': outcome, 'base_model': base_model, 'calibration_trials': int(n), 'repeats': len(group), 'delta_r2_median': float(group['delta_r2'].median()), 'delta_r2_q025': float(group['delta_r2'].quantile(0.025)), 'delta_r2_q975': float(group['delta_r2'].quantile(0.975)), 'delta_mae_median': float(group['delta_mae'].median()), 'delta_mae_q025': float(group['delta_mae'].quantile(0.025)), 'delta_mae_q975': float(group['delta_mae'].quantile(0.975)), 'probability_mae_improves': float(np.mean(group['delta_mae'] < 0))})
    random_summary = pd.DataFrame(random_summary_rows)
    n20 = random_summary.loc[random_summary['calibration_trials'].eq(20)]
    calibration_confirmed = bool(len(n20) == 2 and (n20['delta_mae_q975'] < 0).all())
    return (chronological, random_summary, {'calibration_confirmed_at_20': calibration_confirmed, 'calibration_role': 'validation_secondary' if calibration_confirmed else 'exploratory_only'})

def write_validation_spec(path: Path, dimension_summary: dict, gmm_cv: dict, null_summary: dict, posterior: dict, context: dict, old_gmm: pd.DataFrame, prediction: dict, calibration: dict) -> dict:
    one_se_k = gmm_cv['one_se_k']
    old_row = old_gmm.loc[old_gmm['clusters'].eq(one_se_k)]
    stability_q05 = float(old_row['stability_ari_q05'].iloc[0]) if len(old_row) else float('nan')
    context_ari = float(context.get('all_trial_assignment_ari_between_context_models', float('nan')))
    discrete_robust = bool(one_se_k > 1 and null_summary['continuous_null_p_value'] <= 0.05 and (stability_q05 >= 0.6) and (posterior['min_cluster_fraction'] >= 0.05) and (posterior['fraction_max_posterior_below_0_8'] <= 0.25) and (math.isnan(context_ari) or context_ari >= 0.5))
    representation = 'continuous_2d_primary_with_discrete_secondary' if discrete_robust else 'continuous_2d_primary_discrete_exploratory'
    text = f'''schema_version: 1\nstatus: prespecified_before_validation\nprespecified_at: "{now_iso()}"\n\nphase:\n  discovery_completed: true\n  sensitivity_completed: true\n  validation_opened: false\n  confirmation_opened: false\n\nsubjective_geometry:\n  primary_representation: {representation}\n  dimensions: 2\n  normalization: within_person_z\n  features:{chr(10).join((f'    - {feature}' for feature in CORE7))}\n  dimensional_sensitivity_passed: {str(dimension_summary['core7_all_normalizations_select_two']).lower()}\n\ndiscrete_secondary:\n  repeated_cv_best_k: {gmm_cv['best_k']}\n  repeated_cv_one_se_k: {one_se_k}\n  continuous_null_p_value: {null_summary['continuous_null_p_value']:.8f}\n  stability_ari_q05: {stability_q05:.8f}\n  minimum_component_fraction: {posterior['min_cluster_fraction']:.8f}\n  posterior_ambiguity_fraction_below_0_8: {posterior['fraction_max_posterior_below_0_8']:.8f}\n  context_assignment_ari: {context_ari:.8f}\n  robust_discrete_support: {str(discrete_robust).lower()}\n  interpretation_rule: >\n    Gaussian-mixture components are density summaries, not biological states.\n    A discrete claim requires all prespecified robustness criteria and independent\n    validation.\n\nhedonic_prediction:\n  selected_model: {prediction['selected_model']}\n  outcomes:\n    - valence\n    - liking\n  participant_held_out: true\n  ridge_alpha_grid:\n    - 0.01\n    - 0.1\n    - 1.0\n    - 10.0\n    - 100.0\n  interpretation: subjective_report_mapping_not_mechanism\n\nperson_calibration:\n  role: {calibration['calibration_role']}\n  primary_calibration_trials: 20\n  calibration_method: participant_intercept_offset\n  comparison_uses_same_remaining_trials: true\n\nvalidation_rules:\n  - apply_the_prespecified_two_dimensional_transform\n  - evaluate_the_prespecified_prediction_model\n  - evaluate_discrete_structure_only_as_secondary\n  - do_not_change_gmm_thresholds\n  - do_not_change_feature_sets\n  - do_not_change_alpha_grid\n  - do_not_use_confirmation_outcomes\n  - report_negative_and_ambiguous_results'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return {'primary_representation': representation, 'discrete_robust': discrete_robust, 'stability_q05': stability_q05, 'selected_prediction_model': prediction['selected_model'], 'calibration_role': calibration['calibration_role']}

def save_figures(figures_dir: Path, dimensions: pd.DataFrame, gmm_summary: pd.DataFrame, null_frame: pd.DataFrame, ablation: pd.DataFrame, calibration: pd.DataFrame) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    path = figures_dir / 'subjective_discovery_dimension_sensitivity.png'
    plt.figure(figsize=(10, 5))
    plt.bar(dimensions['representation'], dimensions['first_two_cumulative_variance'])
    plt.xticks(rotation=35, ha='right')
    plt.ylabel('Variance explained by first two PCs')
    plt.title('Sensitivity of two-dimensional subjective geometry')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_discovery_repeated_gmm_cv.png'
    plt.figure(figsize=(8, 5))
    plt.errorbar(gmm_summary['clusters'], gmm_summary['mean_cv_log_likelihood'], yerr=gmm_summary['se_cv_log_likelihood'], marker='o', capsize=3)
    plt.xlabel('Mixture components')
    plt.ylabel('Repeated participant-held-out log likelihood')
    plt.title('GMM complexity sensitivity')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_discovery_continuous_null.png'
    plt.figure(figsize=(8, 5))
    plt.hist(null_frame['null_max_delta_log_likelihood'], bins=20)
    plt.axvline(null_frame['observed_delta_log_likelihood'].iloc[0], linewidth=2, label='Observed')
    plt.xlabel('Best Δ held-out log likelihood over K=1')
    plt.ylabel('Continuous-null iterations')
    plt.title('Continuous copula null for apparent mixture structure')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_discovery_prediction_ablation.png'
    pivot = ablation.pivot(index='model', columns='outcome', values='r2').sort_values(by=OUTCOMES, ascending=False)
    pivot.plot(kind='bar', figsize=(11, 6))
    plt.axhline(0, linewidth=1)
    plt.ylabel('Participant-held-out R²')
    plt.title('Subjective-profile prediction ablation')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    path = figures_dir / 'subjective_discovery_fair_calibration.png'
    plt.figure(figsize=(8, 5))
    for outcome, group in calibration.groupby('outcome'):
        plt.plot(group['calibration_trials'], group['delta_mae_median'], marker='o', label=outcome)
        plt.fill_between(group['calibration_trials'], group['delta_mae_q025'], group['delta_mae_q975'], alpha=0.2)
    plt.axhline(0, linewidth=1)
    plt.xlabel('Random labeled calibration trials')
    plt.ylabel('ΔMAE: calibrated minus uncalibrated')
    plt.title('Fair calibration on identical remaining trials')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths

def build_report(audit: dict, dimensions: pd.DataFrame, dimension_summary: dict, gmm_cv: dict, null_summary: dict, posterior: dict, context: dict, spec: dict, ablation: pd.DataFrame, calibration_chronological: pd.DataFrame, calibration_random: pd.DataFrame, runtime: float) -> str:
    lines = ['# subjective sensitivity analysis — discovery sensitivity audit', '', f'- Generated: `{now_iso()}`', f'- Runtime: `{runtime:.1f} seconds`', f"- Discovery participants: **{audit['discovery_subjects']}**", '- Validation outcomes used: **no**', '- Confirmation outcomes used: **no**', '', '## Decision', '', f"- prespecified primary representation: **`{spec['primary_representation']}`**", f"- Selected prediction model: **`{spec['selected_prediction_model']}`**", f"- Calibration role: **`{spec['calibration_role']}`**", '', '## Dimensional sensitivity', '', '| Representation | Parallel dimensions | PC1 | First two PCs | Max angle |', '|---|---:|---:|---:|---:|']
    for _, row in dimensions.iterrows():
        angle = row['max_principal_angle_degrees_vs_reference']
        angle_text = f'{angle:.2f}°' if np.isfinite(angle) else 'NA'
        lines.append(f"| `{row['representation']}` | {int(row['parallel_dimensions'])} | {row['pc1_explained_variance_ratio']:.3f} | {row['first_two_cumulative_variance']:.3f} | {angle_text} |")
    lines.extend(['', f"- All core-seven normalizations selected two dimensions: **{dimension_summary['core7_all_normalizations_select_two']}**", '', '## Mixture sensitivity', '', f"- Repeated-CV best K: **{gmm_cv['best_k']}**", f"- One-standard-error K: **{gmm_cv['one_se_k']}**", f"- Observed best Δ log likelihood over K=1: **{gmm_cv['delta_best_vs_k1']:.4f}**", f"- Continuous-copula null p-value: **{null_summary['continuous_null_p_value']:.4f}**", f"- Minimum component fraction: **{posterior['min_cluster_fraction']:.3f}**", f"- Median maximum posterior: **{posterior['median_max_posterior']:.3f}**", f"- Ambiguous posterior fraction, max posterior < .80: **{posterior['fraction_max_posterior_below_0_8']:.3f}**", f"- Context-model assignment ARI: **{context.get('all_trial_assignment_ari_between_context_models', float('nan')):.3f}**", f"- Robust discrete support: **{spec['discrete_robust']}**", '', 'A Gaussian mixture can approximate a curved or non-Gaussian continuous density. Therefore its components remain secondary density summaries even when reproducible.', '', '## Prediction ablation', '', '| Outcome | Model | R² | MAE | Pearson r |', '|---|---|---:|---:|---:|'])
    for _, row in ablation.sort_values(['outcome', 'r2'], ascending=[True, False]).iterrows():
        lines.append(f"| `{row['outcome']}` | `{row['model']}` | {row['r2']:.3f} | {row['mae']:.3f} | {row['pearson_r']:.3f} |")
    lines.extend(['', '## Fair calibration', '', 'Calibrated and uncalibrated predictions are now evaluated on exactly the same remaining trials.', '', '| Outcome | Trials | Median ΔR² | Median ΔMAE | P(MAE improves) |', '|---|---:|---:|---:|---:|'])
    for _, row in calibration_random.sort_values(['outcome', 'calibration_trials']).iterrows():
        lines.append(f"| `{row['outcome']}` | {int(row['calibration_trials'])} | {row['delta_r2_median']:.3f} | {row['delta_mae_median']:.3f} | {row['probability_mae_improves']:.3f} |")
    lines.extend(['', '## Interpretation boundary', '', 'This stage concerns the geometry and predictability of subjective reports. Strong prediction from other reports is not evidence of a neural or physiological mechanism.', '', '## Next gate', '', 'The exact Phase 1 validation specification is now prespecified. subjective validation analysis may open validation participants once. Confirmation remains closed.', ''])
    return ''.join(lines)

def main() -> int:
    args = parse_args()
    project = args.project.expanduser().resolve()
    started = time.time()
    tables_dir = project / 'outputs/tables'
    qc_dir = project / 'outputs/qc'
    reports_dir = project / 'outputs/reports'
    figures_dir = project / 'outputs/figures'
    logs_dir = project / 'outputs/logs'
    config_dir = project / 'config'
    models_dir = project / 'outputs/models'
    discovery_path = tables_dir / 'subjective_discovery_discovery_subjective_table.csv'
    oof_path = tables_dir / 'subjective_discovery_oof_predictions.csv'
    old_gmm_path = tables_dir / 'subjective_discovery_gmm_selection.csv'
    subjective_summary_path = qc_dir / 'subjective_discovery_summary.json'
    require_files([discovery_path, oof_path, old_gmm_path, subjective_summary_path])
    output_paths = [tables_dir / 'subjective_discovery_dimension_sensitivity.csv', tables_dir / 'subjective_discovery_gmm_repeated_cv_folds.csv', tables_dir / 'subjective_discovery_gmm_repeated_cv_summary.csv', tables_dir / 'subjective_discovery_continuous_null.csv', tables_dir / 'subjective_discovery_context_replication.csv', tables_dir / 'subjective_discovery_prediction_ablation.csv', tables_dir / 'subjective_discovery_prediction_ablation_oof.csv', tables_dir / 'subjective_discovery_calibration_chronological.csv', tables_dir / 'subjective_discovery_calibration_random_summary.csv', qc_dir / 'subjective_discovery_summary.json', qc_dir / 'subjective_discovery_lockbox_audit.csv', reports_dir / 'subjective_discovery_sensitivity_report.md', config_dir / 'S067_subjective_validation.yaml']
    existing = [path for path in output_paths if path.exists()]
    if existing and (not args.force):
        print('ERROR: subjective sensitivity analysis outputs already exist. Refusing to overwrite.\n  - ' + '\n  - '.join((str(path) for path in existing)), file=sys.stderr)
        return 1
    for directory in [tables_dir, qc_dir, reports_dir, figures_dir, logs_dir, config_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / 'subjective_discovery_sensitivity.log'
    with log_path.open('w', encoding='utf-8') as log_handle:
        discovery = pd.read_csv(discovery_path)
        oof = pd.read_csv(oof_path)
        old_gmm = pd.read_csv(old_gmm_path)
        subjective_summary = json.loads(subjective_summary_path.read_text(encoding='utf-8'))
        audit = validate_lockbox(project, discovery)
        audit_rows = [{'check': 'discovery_subjects', 'value': audit['discovery_subjects'], 'expected': 36, 'passed': audit['discovery_subjects'] == 36}, {'check': 'validation_subjects_read', 'value': 0, 'expected': 0, 'passed': True}, {'check': 'confirmation_subjects_read', 'value': 0, 'expected': 0, 'passed': True}, {'check': 'subjective_discovery_validation_opened', 'value': subjective_summary.get('validation_opened'), 'expected': False, 'passed': not subjective_summary.get('validation_opened')}]
        write_csv(qc_dir / 'subjective_discovery_lockbox_audit.csv', audit_rows, ['check', 'value', 'expected', 'passed'])
        log('Running dimensional sensitivity.', log_handle)
        dimension_table, dimension_summary = dimensional_sensitivity(discovery, args.parallel_iterations)
        dimension_table.to_csv(tables_dir / 'subjective_discovery_dimension_sensitivity.csv', index=False)
        selected_dimensions = int(subjective_summary['dimensions']['selected_dimensions'])
        X = dimension_summary['reference_X']
        groups = discovery['subject_id'].to_numpy()
        log('Running repeated participant GMM CV.', log_handle)
        gmm_folds, gmm_summary_table, gmm_cv = repeated_gmm_cv(X, groups, selected_dimensions, args.gmm_repeats)
        gmm_folds.to_csv(tables_dir / 'subjective_discovery_gmm_repeated_cv_folds.csv', index=False)
        gmm_summary_table.to_csv(tables_dir / 'subjective_discovery_gmm_repeated_cv_summary.csv', index=False)
        reference_scaler = dimension_summary['reference_scaler']
        reference_pca = dimension_summary['reference_pca']
        scores = reference_pca.transform(reference_scaler.transform(X))[:, :selected_dimensions]
        log('Running continuous-copula null.', log_handle)
        null_frame, null_summary = continuous_null_test(scores, groups, gmm_cv['delta_best_vs_k1'], args.gmm_null_iterations)
        null_frame.to_csv(tables_dir / 'subjective_discovery_continuous_null.csv', index=False)
        full_model, posterior = full_gmm_diagnostics(scores, gmm_cv['one_se_k'])
        context_table, context_summary = context_replication(discovery, scores, gmm_cv['one_se_k'])
        context_table.to_csv(tables_dir / 'subjective_discovery_context_replication.csv', index=False)
        log('Running prediction ablations.', log_handle)
        ablation, ablation_oof, prediction_summary = prediction_ablation(discovery)
        ablation.to_csv(tables_dir / 'subjective_discovery_prediction_ablation.csv', index=False)
        ablation_oof.to_csv(tables_dir / 'subjective_discovery_prediction_ablation_oof.csv', index=False)
        log('Recalculating calibration fairly.', log_handle)
        calibration_chronological, calibration_random, calibration_summary = fair_calibration(oof, subjective_summary['prediction']['calibration_base_model'], args.calibration_repeats)
        calibration_chronological.to_csv(tables_dir / 'subjective_discovery_calibration_chronological.csv', index=False)
        calibration_random.to_csv(tables_dir / 'subjective_discovery_calibration_random_summary.csv', index=False)
        spec_path = config_dir / 'S067_subjective_validation.yaml'
        spec_summary = write_validation_spec(spec_path, dimension_summary, gmm_cv, null_summary, posterior, context_summary, old_gmm, prediction_summary, calibration_summary)
        summary = {'generated_at': now_iso(), 'discovery_only': True, 'audit': audit, 'dimensions': {key: value for key, value in dimension_summary.items() if key not in {'reference_pca', 'reference_scaler', 'reference_X'}}, 'gmm_repeated_cv': gmm_cv, 'continuous_null': null_summary, 'posterior': posterior, 'context_replication': context_summary, 'prediction': prediction_summary, 'calibration': calibration_summary, 'validation_spec': spec_summary, 'validation_opened': False, 'confirmation_opened': False, 'runtime_seconds': round(time.time() - started, 2)}
        summary_path = qc_dir / 'subjective_discovery_summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '', encoding='utf-8')
        figure_paths = save_figures(figures_dir, dimension_table, gmm_summary_table, null_frame, ablation, calibration_random)
        report_path = reports_dir / 'subjective_discovery_sensitivity_report.md'
        report_path.write_text(build_report(audit, dimension_table, dimension_summary, gmm_cv, null_summary, posterior, context_summary, spec_summary, ablation, calibration_chronological, calibration_random, time.time() - started), encoding='utf-8')
        log(f'Report written: {report_path}', log_handle)
    print()
    print('=' * 78)
    print('PHASE 1A.1 SENSITIVITY AUDIT COMPLETED')
    print('=' * 78)
    print(f"Primary representation: {spec_summary['primary_representation']}")
    print(f"Discrete robust:        {spec_summary['discrete_robust']}")
    print(f"Prediction model:       {spec_summary['selected_prediction_model']}")
    print(f"Calibration role:       {spec_summary['calibration_role']}")
    print(f'Report:                 {report_path}')
    print(f'Validation spec:        {spec_path}')
    print(f'Runtime:                {time.time() - started:.1f} seconds')
    print()
    print('Validation and confirmation remain unopened.')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    raise SystemExit(main())