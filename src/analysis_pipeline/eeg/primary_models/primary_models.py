# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
"""Phase primary_models: fit prespecified primary nested models and emit outer predictions.

Runs B0 covariate baseline, B1 QC-only negative control, and P1 all-three
full-signal for valence and liking on 1,237 prespecified discovery rows. It creates
outcome-free outer/inner transform caches for exact primary_inference permutation reuse.
No bootstrap, permutation, Holm correction, secondary model, validation, or
confirmation analysis is performed here."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
REL = {'summary': Path('outputs/qc/outcome_model_integrity_outcome_model_integrity_summary.json'), 'contract': Path('config/outcome_model_integrity_outcome_model_integrity_contract.json'), 'execution': Path('config/outcome_model_integrity_prespecified_primary_models_execution_policy.json'), 'file_integrity': Path('outputs/tables/outcome_model_integrity_feature_store_file_integrity.csv'), 'feature_rows': Path('outputs/tables/outcome_model_integrity_unified_feature_row_manifest.csv'), 'trial_outcomes': Path('outputs/tables/outcome_model_integrity_discovery_trial_outcome_manifest.csv'), 'analysis_rows': Path('outputs/tables/outcome_model_integrity_analysis_set_row_manifest.csv'), 'model_spec': Path('config/outcome_model_integrity_prespecified_model_specification.csv'), 'target_policy': Path('config/outcome_model_integrity_prespecified_target_policy.json'), 'nested_policy': Path('config/outcome_model_integrity_prespecified_nested_model_policy.json')}
CACHE_REL = Path('outputs/large/primary_models_primary_nested_cache_v1')
EXPECTED = {'files': 1425, 'bytes': 126050925, 'feature_rows': 3853, 'trials': 1237, 'subjects': 31, 'epoch_rows': 3711}
SEED = 540
OUTCOMES = ['valence', 'liking']
MODELS = ['B0_seen_covariate_baseline', 'B1_seen_qc_only', 'P1_seen_all3_full_signal']
FAMILIES = ['stimulus_early', 'stimulus_late', 'rating_early']
ALPHAS = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000.0], float)
COMPONENTS = [8, 16, 32, 64, 128]
SD_FLOOR = 1e-08
TIE = 1e-12
BLOCKS = [('time_frequency_path', 'tf', 1920), ('morphology_path', 'morph', 384), ('covariance_path', 'cov', 2080), ('covariance_summary_path', 'covsum', 21), ('plv_channel_path', 'plvch', 384), ('plv_global_path', 'plvgl', 24), ('waveform_dct_path', 'dct', 1280)]
PER_EPOCH = 6093
SIGNAL_CELLS = 18279
QC_CELLS = 195
UPPER = np.triu_indices(64)
POLICY = {'phase': 'primary_models', 'seed': SEED, 'models': MODELS, 'outcomes': OUTCOMES, 'family_order': FAMILIES, 'rows': 'all_three_epoch_families and within_person_eligible=true', 'target': 'participant training mean and sample SD ddof=1; SD<=1e-8 fatal', 'covariates': '31 subject indicators + video + training-z raw_run/acquisition_fraction/squared', 'qc': '3*(64 availability+good count); training zero-variance removal and z-score', 'signal': '6093 per epoch; training all-missing drop, median impute, zero-variance drop, z-score, randomized PCA', 'pca': {'components': COMPONENTS, 'solver': 'randomized', 'n_oversamples': 10, 'iterated_power': 4, 'whiten': False}, 'ridge': {'alphas': ALPHAS.tolist(), 'intercept': 'unpenalized by centering', 'solver': 'compact SVD'}, 'inner_metric': 'participant-equal mean squared person-relative error', 'tie': 'fewer components then larger alpha within 1e-12', 'inference': False, 'secondary_models': False, 'validation': False, 'confirmation': False}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--force', action='store_true')
    p.add_argument('--recompute-cache', action='store_true')
    return p.parse_args()

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def tag():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def csha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def jread(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))

def bval(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {'true', '1', 'yes'}:
        return True
    if s in {'false', '0', 'no'}:
        return False
    raise RuntimeError(f'bad bool {x!r}')

def atom_json(p: Path, x):
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v) if isinstance(v, Path) else v) + '', encoding='utf-8')
    os.replace(q, p)

def required(ps: Iterable[Path]):
    m = [p for p in ps if not p.is_file()]
    if m:
        raise RuntimeError('Missing:\n  - ' + '\n  - '.join(map(str, m)))

def stored_manifest(root: Path, exclude: set[str]):
    return [{'relative_path': str(p.relative_to(root)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and str(p.relative_to(root)) not in exclude]

def checkpoint_ok(root: Path, name: str, input_sig: str):
    p = root / name
    if not p.is_file():
        return False
    try:
        s = jread(p)
    except Exception:
        return False
    if s.get('complete') is not True or s.get('policy_sha256') != POLICY_SHA or s.get('input_signature_sha256') != input_sig:
        return False
    for r in s.get('stored_files', []):
        f = root / r['relative_path']
        if not f.is_file() or f.stat().st_size != int(r['size_bytes']) or sha(f) != r['sha256']:
            return False
    return True

def load_inputs(project: Path):
    paths = {k: project / v for k, v in REL.items()}
    required(paths.values())
    summary = jread(paths['summary'])
    contract = jread(paths['contract'])
    execution = jread(paths['execution'])
    if summary.get('phase') != 'outcome_model_integrity':
        raise RuntimeError('Unexpected prerequisite phase')
    for k in ['feature_store_integrity_passed', 'discovery_outcome_attachment_exact', 'primary_analysis_rows_prespecified', 'nested_model_design_prespecified', 'primary_models_nested_model_fitting_permitted']:
        if not summary['decision'].get(k):
            raise RuntimeError(f'Prerequisite failed: {k}')
    if summary['decision'].get('any_model_fitted') or summary['decision'].get('validation_opened') or summary['decision'].get('confirmation_opened'):
        raise RuntimeError('Prerequisite state invalid')
    if execution.get('next_phase') != 'primary_inference_design':
        raise RuntimeError('Unexpected successor')
    omap = {'summary': 'summary', 'file_integrity': 'file_integrity', 'feature_rows': 'feature_row_manifest', 'trial_outcomes': 'trial_outcomes', 'analysis_rows': 'analysis_rows', 'model_spec': 'model_table', 'target_policy': 'target_policy', 'nested_policy': 'model_policy'}
    for k, hk in omap.items():
        if sha(paths[k]) != contract['output_hashes'][hk]:
            raise RuntimeError(f'{k} hash changed')
    fi = pd.read_csv(paths['file_integrity'], low_memory=False)
    fr = pd.read_csv(paths['feature_rows'], low_memory=False)
    to = pd.read_csv(paths['trial_outcomes'], low_memory=False)
    ar = pd.read_csv(paths['analysis_rows'], low_memory=False)
    if len(fi) != EXPECTED['files'] or int(fi.observed_size_bytes.sum()) != EXPECTED['bytes'] or len(fr) != EXPECTED['feature_rows']:
        raise RuntimeError('Prerequisite counts changed')
    return (paths, contract, fi, fr, to, ar)

def rehash(fi: pd.DataFrame):
    out = []
    for i, r in fi.iterrows():
        p = Path(str(r.stored_path))
        if not p.is_file() or p.stat().st_size != int(r.expected_size_bytes) or sha(p) != r.expected_sha256:
            raise RuntimeError(f'Feature integrity failed: {p}')
        out.append({'inventory_row_1based': i + 1, 'stored_path': str(p), 'size_bytes': p.stat().st_size, 'sha256': r.expected_sha256, 'passed': True})
        if (i + 1) % 200 == 0 or i + 1 == len(fi):
            print(f'[{now()}] Rehashed {i + 1}/{len(fi)} files.', flush=True)
    return pd.DataFrame(out)

def primary_metadata(to: pd.DataFrame, ar: pd.DataFrame):
    p = ar[(ar.analysis_set == 'all_three_epoch_families') & ar.within_person_eligible.map(bval)].copy()
    if len(p) != EXPECTED['trials'] or p.subject_id.nunique() != EXPECTED['subjects']:
        raise RuntimeError('Primary set changed')
    cols = ['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'raw_run', 'raw_source_path', 'subject_acquisition_order_1based', 'within_subject_block_fold', 'cross_person_fold', 'acquisition_fraction', 'valence', 'liking']
    o = to[cols].copy()
    m = p[['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'within_subject_block_fold', 'cross_person_fold', 'subject_acquisition_order_1based']].merge(o, on=['raw_behavior_row_id', 'subject_id', 'trial_number', 'context', 'within_subject_block_fold', 'cross_person_fold', 'subject_acquisition_order_1based'], how='left', validate='one_to_one')
    if m[OUTCOMES].isna().any().any():
        raise RuntimeError('Missing outcome')
    m = m.sort_values(['subject_id', 'subject_acquisition_order_1based', 'raw_behavior_row_id']).reset_index(drop=True)
    m['primary_row_0based'] = np.arange(len(m))
    return m

def flat_rows(a, kind):
    v = np.asarray(a)
    return v[:, UPPER[0], UPPER[1]] if kind == 'cov' else v.reshape(v.shape[0], -1)

def assemble(cache: Path, meta: pd.DataFrame, fr: pd.DataFrame, input_sig: str, recompute: bool):
    final = cache / 'assembled_primary'
    name = 'assembled_summary.json'
    if not recompute and checkpoint_ok(final, name, input_sig):
        return final
    tmp = cache / '.assembled.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        n = len(meta)
        sig = np.lib.format.open_memmap(tmp / 'signal_float32.npy', mode='w+', dtype=np.float32, shape=(n, SIGNAL_CELLS))
        qc = np.lib.format.open_memmap(tmp / 'qc_float32.npy', mode='w+', dtype=np.float32, shape=(n, QC_CELLS))
        sig[:] = np.nan
        qc[:] = np.nan
        f = fr[fr.raw_behavior_row_id.astype(str).isin(set(meta.raw_behavior_row_id.astype(str)))].copy()
        if len(f) != EXPECTED['epoch_rows']:
            raise RuntimeError('Primary epoch count changed')
        for fi, fam in enumerate(FAMILIES):
            ff = f[f.epoch_family == fam].copy()
            rows = meta[['raw_behavior_row_id', 'primary_row_0based']].merge(ff, on='raw_behavior_row_id', how='left', validate='one_to_one')
            if rows.epoch_id.isna().any():
                raise RuntimeError(f'Missing {fam}')
            off = fi * PER_EPOCH
            bo = 0
            for pcol, kind, width in BLOCKS:
                for pt, g in rows.groupby(pcol, sort=True):
                    a = np.load(Path(str(pt)), mmap_mode='r', allow_pickle=False)
                    vals = flat_rows(a[g.feature_row_0based.astype(int).to_numpy()], kind)
                    if vals.shape[1] != width:
                        raise RuntimeError(f'Width {fam} {kind}')
                    ridx = g.primary_row_0based.astype(int).to_numpy()
                    sig[ridx, off + bo:off + bo + width] = vals.astype(np.float32)
                bo += width
            if bo != PER_EPOCH:
                raise RuntimeError('Per-epoch width')
            qo = fi * 65
            for pt, g in rows.groupby('time_frequency_path', sort=True):
                a = np.load(Path(str(pt)), mmap_mode='r', allow_pickle=False)
                mask = np.isfinite(a[g.feature_row_0based.astype(int).to_numpy(), :, 0, 0])
                good = g.final_good_channels.astype(int).to_numpy()
                if not np.array_equal(mask.sum(1), good):
                    raise RuntimeError('QC mask mismatch')
                ridx = g.primary_row_0based.astype(int).to_numpy()
                qc[ridx, qo:qo + 64] = mask.astype(np.float32)
                qc[ridx, qo + 64] = good.astype(np.float32)
        sig.flush()
        qc.flush()
        meta.to_csv(tmp / 'metadata.csv', index=False)
        if np.isinf(np.asarray(sig)).any() or not np.isfinite(np.asarray(qc)).all():
            raise RuntimeError('Assembled nonfinite error')
        atom_json(tmp / name, {'complete': True, 'phase': 'primary_models', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': input_sig, 'stored_files': stored_manifest(tmp, {name})})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        if tmp.exists():
            (tmp / 'FAILED.txt').write_text(f'Failed {now()}')
        raise

def cov_transform(meta, train_idx, eval_idx, levels):
    tr = meta.iloc[train_idx]
    ev = meta.iloc[eval_idx]

    def binary(df):
        s = np.column_stack([df.subject_id.eq(x).to_numpy(float) for x in levels])
        c = df.context.eq('video').to_numpy(float)[:, None]
        return np.concatenate([s, c], axis=1)
    bt, be = (binary(tr), binary(ev))
    ct = np.column_stack([tr.raw_run.to_numpy(float), tr.acquisition_fraction.to_numpy(float), tr.acquisition_fraction.to_numpy(float) ** 2])
    ce = np.column_stack([ev.raw_run.to_numpy(float), ev.acquisition_fraction.to_numpy(float), ev.acquisition_fraction.to_numpy(float) ** 2])
    mu = ct.mean(0)
    sd = ct.std(0)
    nz = sd > 0
    zt = np.zeros_like(ct)
    ze = np.zeros_like(ce)
    zt[:, nz] = (ct[:, nz] - mu[nz]) / sd[nz]
    ze[:, nz] = (ce[:, nz] - mu[nz]) / sd[nz]
    return (np.concatenate([bt, zt], 1).astype(np.float32), np.concatenate([be, ze], 1).astype(np.float32), {'output_columns': len(levels) + 4, 'zero_continuous': int((~nz).sum())})

def qc_transform(qc, train_idx, eval_idx):
    tr = np.asarray(qc[train_idx], np.float64)
    ev = np.asarray(qc[eval_idx], np.float64)
    var = tr.var(0)
    keep = np.isfinite(var) & (var > 0)
    tr = tr[:, keep]
    ev = ev[:, keep]
    mu = tr.mean(0)
    sd = tr.std(0)
    if not (sd > 0).all():
        raise RuntimeError('QC variance failure')
    return (((tr - mu) / sd).astype(np.float32), ((ev - mu) / sd).astype(np.float32), {'input_columns': qc.shape[1], 'dropped': int((~keep).sum()), 'output_columns': int(keep.sum())})

def signal_transform(sig, train_idx, eval_idx, seed):
    tr = np.asarray(sig[train_idx], np.float32)
    ev = np.asarray(sig[eval_idx], np.float32)
    allnan = np.isnan(tr).all(0)
    tr = tr[:, ~allnan]
    ev = ev[:, ~allnan]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(tr, axis=0)
    if not np.isfinite(med).all():
        raise RuntimeError('Nonfinite median')
    tr = np.where(np.isnan(tr), med[None, :], tr)
    ev = np.where(np.isnan(ev), med[None, :], ev)
    var = tr.var(0)
    keep = np.isfinite(var) & (var > 0)
    tr = tr[:, keep]
    ev = ev[:, keep]
    mu = tr.mean(0)
    sd = tr.std(0)
    if not (sd > 0).all():
        raise RuntimeError('Signal variance failure')
    tr = (tr - mu) / sd
    ev = (ev - mu) / sd
    nc = min(max(COMPONENTS), tr.shape[0] - 1, tr.shape[1])
    if nc < min(COMPONENTS):
        raise RuntimeError('Too few PCA components')
    p = PCA(n_components=nc, svd_solver='randomized', random_state=int(seed), n_oversamples=10, iterated_power=4, whiten=False)
    zt = p.fit_transform(tr)
    ze = p.transform(ev)
    return (zt.astype(np.float32), ze.astype(np.float32), {'input_columns': sig.shape[1], 'all_missing_dropped': int(allnan.sum()), 'zero_variance_dropped': int((~keep).sum()), 'retained_columns': tr.shape[1], 'pca_components': nc, 'explained_ratio_sum': float(p.explained_variance_ratio_.sum()), 'seed': int(seed)})

def split_cache(root: Path, split_id: str, meta, sig, qc, train_idx, eval_idx, levels, input_sig, seed, recompute):
    final = root / split_id
    name = 'split_summary.json'
    ssig = csha({'base': input_sig, 'id': split_id, 'train': train_idx.astype(int).tolist(), 'eval': eval_idx.astype(int).tolist(), 'seed': int(seed)})
    if not recompute and checkpoint_ok(final, name, ssig):
        return final
    tmp = root / ('.' + split_id + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        ct, ce, cd = cov_transform(meta, train_idx, eval_idx, levels)
        qt, qe, qd = qc_transform(qc, train_idx, eval_idx)
        pt, pe, pd = signal_transform(sig, train_idx, eval_idx, seed)
        arrays = {'train_indices.npy': train_idx.astype(np.int64), 'eval_indices.npy': eval_idx.astype(np.int64), 'cov_train.npy': ct, 'cov_eval.npy': ce, 'qc_train.npy': qt, 'qc_eval.npy': qe, 'pca_train.npy': pt, 'pca_eval.npy': pe}
        for n, a in arrays.items():
            np.save(tmp / n, a, allow_pickle=False)
        atom_json(tmp / 'diagnostics.json', {'split_id': split_id, 'train_rows': len(train_idx), 'eval_rows': len(eval_idx), 'covariates': cd, 'qc': qd, 'signal': pd})
        atom_json(tmp / name, {'complete': True, 'phase': 'primary_models', 'policy_sha256': POLICY_SHA, 'input_signature_sha256': ssig, 'stored_files': stored_manifest(tmp, {name})})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        return final
    except Exception:
        if tmp.exists():
            (tmp / 'FAILED.txt').write_text(f'Failed {now()}')
        raise

def load_split(d: Path):
    return {k: np.load(d / f, mmap_mode='r', allow_pickle=False) for k, f in {'train': 'train_indices.npy', 'eval': 'eval_indices.npy', 'ct': 'cov_train.npy', 'ce': 'cov_eval.npy', 'qt': 'qc_train.npy', 'qe': 'qc_eval.npy', 'pt': 'pca_train.npy', 'pe': 'pca_eval.npy'}.items()}

def target_scale(meta, outcome, train_idx, eval_idx):
    tr = meta.iloc[train_idx]
    ev = meta.iloc[eval_idx]
    st = tr.groupby('subject_id')[outcome].agg(['mean', 'std'])
    bad = st.index[st['std'].isna() | (st['std'] <= SD_FLOOR)].tolist()
    if bad:
        raise RuntimeError(f'{outcome} bad SD: {bad}')
    if not set(ev.subject_id).issubset(set(st.index)):
        raise RuntimeError('Eval subject absent from train')
    tm = tr.subject_id.map(st['mean']).to_numpy(float)
    ts = tr.subject_id.map(st['std']).to_numpy(float)
    em = ev.subject_id.map(st['mean']).to_numpy(float)
    es = ev.subject_id.map(st['std']).to_numpy(float)
    return ((tr[outcome].to_numpy(float) - tm) / ts, (ev[outcome].to_numpy(float) - em) / es, em, es)

def ridge_grid(xtr, ytr, xev, alphas):
    x = np.asarray(xtr, np.float64)
    e = np.asarray(xev, np.float64)
    y = np.asarray(ytr, np.float64)
    xm = x.mean(0)
    ym = float(y.mean())
    xc = x - xm
    ec = e - xm
    yc = y - ym
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    uy = u.T @ yc
    out = np.empty((len(e), len(alphas)))
    for j, a in enumerate(alphas):
        out[:, j] = ym + ec @ (vt.T @ (s / (s * s + float(a)) * uy))
    return out

def pe_mse(y, p, subjects):
    return float(pd.DataFrame({'s': subjects, 'e': (np.asarray(y) - np.asarray(p)) ** 2}).groupby('s').e.mean().mean())

def candidates(model):
    return [(c, float(a)) for c in COMPONENTS for a in ALPHAS] if model == MODELS[2] else [(None, float(a)) for a in ALPHAS]

def choose(df):
    m = float(df.inner_mse.min())
    z = df[df.inner_mse <= m + TIE].copy()
    z['cs'] = pd.to_numeric(z.pca_components, errors='coerce').fillna(0).astype(int)
    return z.sort_values(['cs', 'ridge_alpha'], ascending=[True, False]).iloc[0]

def matrices(s, model, comp):
    ct = np.asarray(s['ct'], np.float64)
    ce = np.asarray(s['ce'], np.float64)
    if model == MODELS[0]:
        return (ct, ce)
    if model == MODELS[1]:
        return (np.concatenate([ct, np.asarray(s['qt'], np.float64)], 1), np.concatenate([ce, np.asarray(s['qe'], np.float64)], 1))
    if comp is None or comp > s['pt'].shape[1]:
        raise RuntimeError('Bad component count')
    return (np.concatenate([ct, np.asarray(s['pt'][:, :comp], np.float64)], 1), np.concatenate([ce, np.asarray(s['pe'][:, :comp], np.float64)], 1))

def fit_models(meta: pd.DataFrame, fold_root: Path):
    preds = []
    selected = []
    scores = []
    folds = [1, 2, 3, 4, 5]
    for outcome in OUTCOMES:
        for outer in folds:
            osplit = load_split(fold_root / f'outer-{outer}')
            otr = np.asarray(osplit['train'], int)
            ote = np.asarray(osplit['eval'], int)
            lookup = {int(g): i for i, g in enumerate(otr)}
            for model in MODELS:
                cand = candidates(model)
                oof = {c: np.full(len(otr), np.nan) for c in cand}
                truth = np.full(len(otr), np.nan)
                for inner in folds:
                    if inner == outer:
                        continue
                    sp = load_split(fold_root / f'outer-{outer}_inner-{inner}')
                    itr = np.asarray(sp['train'], int)
                    iev = np.asarray(sp['eval'], int)
                    yztr, yzv, _, _ = target_scale(meta, outcome, itr, iev)
                    loc = np.array([lookup[int(g)] for g in iev], int)
                    truth[loc] = yzv
                    if model == MODELS[2]:
                        for c in COMPONENTS:
                            xt, xe = matrices(sp, model, c)
                            gp = ridge_grid(xt, yztr, xe, ALPHAS)
                            for j, a in enumerate(ALPHAS):
                                oof[c, float(a)][loc] = gp[:, j]
                    else:
                        xt, xe = matrices(sp, model, None)
                        gp = ridge_grid(xt, yztr, xe, ALPHAS)
                        for j, a in enumerate(ALPHAS):
                            oof[None, float(a)][loc] = gp[:, j]
                if np.isnan(truth).any():
                    raise RuntimeError('Incomplete inner truth')
                subs = meta.iloc[otr].subject_id.to_numpy()
                rows = []
                for (c, a), p in oof.items():
                    if np.isnan(p).any():
                        raise RuntimeError('Incomplete inner prediction')
                    r = {'outcome': outcome, 'outer_fold': outer, 'model_id': model, 'pca_components': c, 'ridge_alpha': a, 'inner_mse': pe_mse(truth, p, subs), 'inner_rows': len(p), 'inner_subjects': len(set(subs))}
                    rows.append(r)
                    scores.append(r)
                sf = pd.DataFrame(rows)
                ch = choose(sf)
                comp = None if pd.isna(ch.pca_components) else int(ch.pca_components)
                alpha = float(ch.ridge_alpha)
                yztr, yzte, emean, esd = target_scale(meta, outcome, otr, ote)
                xt, xe = matrices(osplit, model, comp)
                pz = ridge_grid(xt, yztr, xe, np.array([alpha]))[:, 0]
                praw = pz * esd + emean
                te = meta.iloc[ote].copy()
                yraw = te[outcome].to_numpy(float)
                selected.append({'outcome': outcome, 'outer_fold': outer, 'model_id': model, 'pca_components': comp, 'ridge_alpha': alpha, 'selected_inner_mse': float(ch.inner_mse), 'outer_train_rows': len(otr), 'outer_test_rows': len(ote), 'design_columns': xt.shape[1]})
                for k, (gi, z, pzr, pr, mn, sd) in enumerate(zip(ote, yzte, pz, praw, emean, esd)):
                    r = meta.iloc[int(gi)]
                    preds.append({'raw_behavior_row_id': r.raw_behavior_row_id, 'primary_row_0based': int(r.primary_row_0based), 'subject_id': r.subject_id, 'trial_number': int(r.trial_number), 'context': r.context, 'raw_run': int(r.raw_run), 'subject_acquisition_order_1based': int(r.subject_acquisition_order_1based), 'within_subject_block_fold': int(r.within_subject_block_fold), 'cross_person_fold': int(r.cross_person_fold), 'outcome': outcome, 'model_id': model, 'outer_fold': outer, 'observed_raw': float(yraw[k]), 'observed_person_relative': float(z), 'predicted_person_relative': float(pzr), 'predicted_raw': float(pr), 'outer_training_subject_mean': float(mn), 'outer_training_subject_sd': float(sd), 'person_relative_error': float(z - pzr), 'raw_error': float(yraw[k] - pr), 'pca_components': comp, 'ridge_alpha': alpha})
                print(f'[{now()}] Completed {outcome} outer {outer} {model}', flush=True)
    return (pd.DataFrame(preds), pd.DataFrame(selected), pd.DataFrame(scores))

def diagnostics(pred):
    rows = []
    for (o, m, f), g in pred.groupby(['outcome', 'model_id', 'outer_fold'], sort=True):
        y = g.observed_person_relative.to_numpy(float)
        p = g.predicted_person_relative.to_numpy(float)
        den = float(np.sum(y * y))
        r2 = float(1 - np.sum((y - p) ** 2) / den) if den > 0 else math.nan
        rows.append({'outcome': o, 'model_id': m, 'outer_fold': int(f), 'test_rows': len(g), 'test_subjects': g.subject_id.nunique(), 'participant_equal_mse': pe_mse(y, p, g.subject_id.to_numpy()), 'training_referenced_r2_diagnostic': r2, 'raw_mae': float(np.mean(np.abs(g.raw_error.to_numpy(float)))), 'inference_or_claim_gate': False})
    return pd.DataFrame(rows)

def cache_inventory(root):
    return pd.DataFrame([{'stored_path': str(p), 'relative_path': str(p.relative_to(root)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(root.rglob('*')) if p.is_file()])

def write_report(path, summary, selected, diag):
    lines = ['# Phase primary_models — Primary nested model fitting', '', f"- Generated: `{summary['generated_at']}`", f"- Status: **`{summary['status']}`**", '- Discovery-only primary models fitted: **yes**', '- Bootstrap/permutation inference: **no**', '- Secondary models fitted: **no**', '- Validation/confirmation opened: **no**', '', '## Execution', '', f"- Primary trials: **{summary['counts']['primary_trial_rows']}**", f"- Participants: **{summary['counts']['primary_subjects']}**", f"- Outer prediction rows: **{summary['counts']['outer_prediction_rows']}**", f"- Inner candidate score rows: **{summary['counts']['inner_candidate_score_rows']}**", '', '## Selected hyperparameters', '', '| Outcome | Fold | Model | PCA | Alpha | Inner participant-equal MSE |', '|---|---:|---|---:|---:|---:|']
    for _, r in selected.sort_values(['outcome', 'outer_fold', 'model_id']).iterrows():
        lines.append(f"| `{r.outcome}` | {int(r.outer_fold)} | `{r.model_id}` | {('' if pd.isna(r.pca_components) else int(r.pca_components))} | {r.ridge_alpha:g} | {r.selected_inner_mse:.6f} |")
    lines += ['', '## Outer-fold diagnostics', '', 'Diagnostics below are not confidence intervals, permutation tests, Holm-adjusted results, or support decisions.', '', '| Outcome | Model | Mean fold MSE | Mean fold raw MAE |', '|---|---|---:|---:|']
    a = diag.groupby(['outcome', 'model_id']).agg(mse=('participant_equal_mse', 'mean'), mae=('raw_mae', 'mean')).reset_index()
    for _, r in a.iterrows():
        lines.append(f'| `{r.outcome}` | `{r.model_id}` | {r.mse:.6f} | {r.mae:.6f} |')
    lines += ['', '## Decision', '', f"- Complete outer prediction coverage: **{summary['decision']['complete_outer_prediction_coverage']}**", f"- All transforms training-only: **{summary['decision']['all_transforms_training_only']}**", f"- primary_inference inference permitted: **{summary['decision']['primary_inference_inference_permitted']}**", f"- Scientific signal support decided: **{summary['decision']['scientific_signal_support_decided']}**", '', 'primary_inference must reuse the prespecified outcome-free split matrices, fully retune hyperparameters for all 1,000 circular-shift permutations, then perform the 5,000-participant bootstrap and Holm correction.', '']
    path.write_text(''.join(lines), encoding='utf-8')

def main():
    a = args()
    started = time.time()
    project = a.project.expanduser().resolve()
    paths, contract, fi, fr, to, ar = load_inputs(project)
    tables = project / 'outputs/tables'
    qcd = project / 'outputs/qc'
    reports = project / 'outputs/reports'
    config = project / 'config'
    logs = project / 'outputs/logs'
    cache = project / CACHE_REL
    out = {'rehash': qcd / 'primary_models_feature_store_rehash.csv', 'predictions': tables / 'primary_models_primary_outer_predictions.csv', 'selected': tables / 'primary_models_selected_hyperparameters.csv', 'inner_scores': tables / 'primary_models_inner_candidate_scores.csv', 'diagnostics': qcd / 'primary_models_outer_fold_diagnostics.csv', 'cache_inventory': tables / 'primary_models_cache_inventory.csv', 'summary': qcd / 'primary_models_primary_nested_model_summary.json', 'contract': config / 'primary_models_primary_nested_model_contract.json', 'next_policy': config / 'S051_primary_models_inference.json', 'report': reports / 'primary_models_primary_nested_model_report.md', 'log': logs / 'primary_models_primary_nested_model.log'}
    existing = [p for p in out.values() if p.exists()]
    if existing and (not a.force):
        raise RuntimeError('Outputs exist; use --force:\n  - ' + '\n  - '.join(map(str, existing)))
    with out['log'].open('w', encoding='utf-8') as log:

        def note(s):
            line = f'[{now()}] {s}'
            print(line, flush=True)
            log.write(line + '')
            log.flush()
        note('Rehashing complete outcome_model_integrity.1 feature store.')
        rh = rehash(fi)
        rh.to_csv(out['rehash'], index=False)
        meta = primary_metadata(to, ar)
        input_sig = csha({'contract': sha(paths['contract']), 'feature_rows': sha(paths['feature_rows']), 'trial_outcomes': sha(paths['trial_outcomes']), 'analysis_rows': sha(paths['analysis_rows']), 'policy': POLICY_SHA})
        note('Assembling prespecified signal and QC matrices.')
        assembled = assemble(cache, meta, fr, input_sig, a.recompute_cache)
        sig = np.load(assembled / 'signal_float32.npy', mmap_mode='r', allow_pickle=False)
        qc = np.load(assembled / 'qc_float32.npy', mmap_mode='r', allow_pickle=False)
        cm = pd.read_csv(assembled / 'metadata.csv', low_memory=False)
        if cm.raw_behavior_row_id.astype(str).tolist() != meta.raw_behavior_row_id.astype(str).tolist():
            raise RuntimeError('Cached metadata order changed')
        foldroot = cache / 'fold_transforms'
        foldroot.mkdir(parents=True, exist_ok=True)
        levels = sorted(meta.subject_id.astype(str).unique())
        labels = [1, 2, 3, 4, 5]
        fl = meta.within_subject_block_fold.astype(int).to_numpy()
        count = 0
        for outer in labels:
            tr = np.flatnonzero(fl != outer)
            te = np.flatnonzero(fl == outer)
            note(f'Outer transform {outer}/5')
            split_cache(foldroot, f'outer-{outer}', meta, sig, qc, tr, te, levels, input_sig, SEED + 100 * outer, a.recompute_cache)
            count += 1
            for inner in labels:
                if inner == outer:
                    continue
                itr = np.flatnonzero((fl != outer) & (fl != inner))
                iev = np.flatnonzero(fl == inner)
                note(f'Outer {outer}, inner validation {inner}')
                split_cache(foldroot, f'outer-{outer}_inner-{inner}', meta, sig, qc, itr, iev, levels, input_sig, SEED + 100 * outer + inner, a.recompute_cache)
                count += 1
        if count != 25:
            raise RuntimeError('Transform count incomplete')
        note('Fitting B0, B1, and P1 with nested selection.')
        pred, sel, sc = fit_models(meta, foldroot)
        expected = len(meta) * len(OUTCOMES) * len(MODELS)
        if len(pred) != expected or not pred.groupby(['raw_behavior_row_id', 'outcome', 'model_id']).size().eq(1).all():
            raise RuntimeError('Prediction coverage incomplete')
        if len(sel) != 30 or len(sc) != 630:
            raise RuntimeError(f'Score counts incorrect: {len(sel)}, {len(sc)}')
        dg = diagnostics(pred)
        pred.to_csv(out['predictions'], index=False)
        sel.to_csv(out['selected'], index=False)
        sc.to_csv(out['inner_scores'], index=False)
        dg.to_csv(out['diagnostics'], index=False)
        ci = cache_inventory(cache)
        ci.to_csv(out['cache_inventory'], index=False)
        decision = {'complete_outer_prediction_coverage': True, 'all_transforms_training_only': True, 'complete_inner_hyperparameter_selection': True, 'primary_inference_inference_permitted': True, 'bootstrap_inference_performed': False, 'permutation_inference_performed': False, 'multiplicity_correction_performed': False, 'scientific_signal_support_decided': False, 'secondary_models_fitted': False, 'secondary_models_remain_mandatory': True, 'validation_opened': False, 'confirmation_opened': False}
        summary = {'generated_at': now(), 'phase': 'primary_models', 'status': 'primary_nested_model_fitting_and_outer_predictions_complete_candidate', 'fitting_policy_sha256': POLICY_SHA, 'counts': {'feature_files_rehashed': len(rh), 'feature_bytes_rehashed': int(rh.size_bytes.sum()), 'primary_trial_rows': len(meta), 'primary_subjects': meta.subject_id.nunique(), 'outer_transform_checkpoints': 5, 'inner_transform_checkpoints': 20, 'outer_prediction_rows': len(pred), 'selected_hyperparameter_rows': len(sel), 'inner_candidate_score_rows': len(sc), 'cache_files': len(ci), 'cache_bytes': int(ci.size_bytes.sum())}, 'primary_models': MODELS, 'outcomes': OUTCOMES, 'decision': decision, 'runtime_seconds': round(time.time() - started, 3)}
        out['summary'].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        con = {**summary, 'fitting_policy': POLICY, 'cache_root': str(cache), 'large_cache_embedded_in_release': False, 'cache_reuse_policy': 'full SHA-256 verification', 'input_hashes': {k: sha(v) for k, v in paths.items()}, 'output_hashes': {k: sha(v) for k, v in out.items() if k not in {'contract', 'next_policy', 'report', 'log'} and v.is_file()}, 'feature_store_modified': False, 'canonical_master_modified': False, 'validation_opened': False, 'confirmation_opened': False}
        out['contract'].write_text(json.dumps(con, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        nxt = {'generated_at': now(), 'phase': 'primary_models', 'next_phase': 'primary_inference', 'next_phase_permitted': True, 'prediction_manifest': str(out['predictions']), 'selected_hyperparameters': str(out['selected']), 'inner_candidate_scores': str(out['inner_scores']), 'cache_inventory': str(out['cache_inventory']), 'cache_root': str(cache), 'required_inference': {'participant_bootstrap_replicates': 5000, 'bootstrap_seed': 540, 'circular_shift_permutations': 1000, 'permutation_seed': 540, 'permutation_recipe': 'within participant and context circular shift by nonzero offset at least 3 rows', 'permutation_fitting': 'complete inner PCA/ridge retuning using prespecified outcome-free matrices', 'multiplicity': 'Holm across valence and liking', 'support_gate': ['lower 95% participant-bootstrap CI P1-B0 R2 > 0', 'lower 95% participant-bootstrap CI P1-B1 R2 > 0', 'Holm-adjusted permutation p < 0.05']}, 'secondary_model_commitment': 'all prespecified secondary and diagnostic families remain mandatory regardless of primary_models outer results', 'validation_opened': False, 'confirmation_opened': False}
        out['next_policy'].write_text(json.dumps(nxt, ensure_ascii=False, indent=2) + '', encoding='utf-8')
        write_report(out['report'], summary, sel, dg)
        note('primary_models complete; no inference or support decision made.')
    print('' + '=' * 100)
    print('PHASE primary_models PRIMARY NESTED MODELS COMPLETED')
    print('=' * 100)
    print(f'Primary trials: {len(meta)}')
    print(f'Prediction rows: {len(pred)}')
    print(f'Inner candidate rows: {len(sc)}')
    print('Scientific support decided: False')
    print(f'Cache: {cache}')
    print(f'Runtime: {time.time() - started:.2f} seconds')
    return 0
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    raise SystemExit(main())