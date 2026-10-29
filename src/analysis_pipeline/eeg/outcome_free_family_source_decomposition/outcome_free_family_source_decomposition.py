# Public scientific analysis source. Non-analytic packaging/export routines and machine-specific paths are omitted.
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
PHASE = 'outcome_free_family_source_decomposition'
DEFAULT_PROJECT = Path('<PROJECT_ROOT>')
SEED = 20260812
family_interpretability_extension_LOCK = Path('outputs/interpretability/family_interpretability_extension_family_interpretability_extension/qc/family_interpretability_extension_final_output_lock_hashes.json')
family_interpretability_extension_SUM = Path('outputs/interpretability/family_interpretability_extension_family_interpretability_extension/qc/family_interpretability_extension_summary.json')
latent_state_fit_LOCK = Path('outputs/qc/latent_state_fit_primary_lock_hashes.json')
latent_state_fit_LOAD = Path('outputs/tables/latent_state_fit_pca_loadings.csv')
latent_state_fit_TRAJ = Path('outputs/tables/latent_state_fit_continuous_trajectories.csv')
cross_person_alignment_fit_LOCK = Path('outputs/qc/cross_person_alignment_fit_final_output_lock_hashes.json')
cross_person_alignment_fit_ASSIGN = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_assignments.csv')
cross_person_alignment_fit_CENT = Path('outputs/tables/cross_person_alignment_fit_trajectory_family_centroids.csv')
mapping_design_LOCK = Path('outputs/qc/mapping_design_final_output_lock_hashes.json')
mapping_design_MAP = Path('outputs/tables/mapping_design_trial_mapping_manifest.csv')
latent_state_design_SCHEMA = Path('outputs/tables/latent_state_design_compact_feature_schema.csv')
latent_state_design_CONTRACT = Path('config/latent_state_design_latent_state_design_contract.json')
BANDS = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma']
TF_BINS = ['whole_epoch', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']
MORPH = ['winsorized_rms_uv', 'line_length_uv_per_sample', 'hjorth_mobility', 'hjorth_complexity', 'lag1_autocorrelation', 'normalized_spectral_entropy']
COVSUM = [*[f'normalized_eigenvalue_{i:02d}' for i in range(1, 17)], 'effective_rank', 'participation_ratio', 'regularized_log_determinant', 'global_field_power_median_uv', 'global_field_power_iqr_uv']
PLVSTAT = ['global_mean_plv', 'global_sd_plv', 'global_q25_plv', 'global_q75_plv']
GEOM = ['standardized_early_to_late_norm', 'standardized_late_to_rating_norm', 'standardized_early_to_rating_direct_norm', 'standardized_total_path_length', 'standardized_directness_ratio', 'standardized_segment_cosine']

def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def stamp():
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')

def sha256(p: Path):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def writej(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '', encoding='utf-8')

def bval(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {'true', '1', 'yes'}

def verify_inputs(project: Path):
    req = [family_interpretability_extension_LOCK, family_interpretability_extension_SUM, latent_state_fit_LOCK, latent_state_fit_LOAD, latent_state_fit_TRAJ, cross_person_alignment_fit_LOCK, cross_person_alignment_fit_ASSIGN, cross_person_alignment_fit_CENT, mapping_design_LOCK, mapping_design_MAP, latent_state_design_SCHEMA, latent_state_design_CONTRACT]
    miss = [str(p) for p in req if not (project / p).is_file()]
    if miss:
        raise RuntimeError('Missing prerequisites:\n  ' + '\n  '.join(miss))
    a = readj(project / family_interpretability_extension_LOCK)
    if a.get('phase') != 'family_interpretability_extension' or a.get('next_phase') != PHASE:
        raise RuntimeError('family_interpretability_extension does not authorize outcome_free_family_source_decomposition')
    if a.get('any_primary_domain_supported_after_Holm') is not False:
        raise RuntimeError('family_interpretability_extension semantic support unexpectedly present')
    if a.get('semantic_relabeling_permitted') is not False:
        raise RuntimeError('family_interpretability_extension semantic boundary mismatch')
    for rel, exp in a.get('hashes', {}).items():
        p = project / rel
        if not p.is_file() or sha256(p) != exp:
            raise RuntimeError(f'family_interpretability_extension lock mismatch: {rel}')
    b = readj(project / latent_state_fit_LOCK)
    if b.get('phase') != 'latent_state_fit' or b.get('context_or_reports_read_before_lock') is not False:
        raise RuntimeError('latent_state_fit outcome-free boundary mismatch')
    for rel, key in [(latent_state_fit_LOAD, 'pca_loadings'), (latent_state_fit_TRAJ, 'continuous_trajectories')]:
        if sha256(project / rel) != b['hashes'][key]:
            raise RuntimeError(f'latent_state_fit hash mismatch: {rel}')
    d = readj(project / cross_person_alignment_fit_LOCK)
    if d.get('phase') != 'cross_person_alignment_fit' or bval(d.get('reports_outcomes_context_qc_read')):
        raise RuntimeError('cross_person_alignment_fit outcome-free boundary mismatch')
    if sha256(project / cross_person_alignment_fit_ASSIGN) != d['hashes']['trajectory_selected_assignments']:
        raise RuntimeError('cross_person_alignment_fit assignment hash mismatch')
    if sha256(project / cross_person_alignment_fit_CENT) != d['hashes']['trajectory_selected_centroids']:
        raise RuntimeError('cross_person_alignment_fit centroid hash mismatch')
    e = readj(project / mapping_design_LOCK)
    if e.get('phase') != 'mapping_design' or e.get('associations_calculated') is not False:
        raise RuntimeError('mapping_design boundary mismatch')
    if sha256(project / mapping_design_MAP) != e['hashes']['trial_manifest']:
        raise RuntimeError('mapping_design manifest hash mismatch')
    c = readj(project / latent_state_design_CONTRACT)
    if c.get('phase') != 'latent_state_design' or c['counts']['compact_cells_per_epoch'] != 169:
        raise RuntimeError('latent_state_design compact representation drift')
    return (a, b, d, e, c)

def source_schema(schema: pd.DataFrame) -> pd.DataFrame:
    if len(schema) != 169 or schema['compact_cell_0based'].tolist() != list(range(169)):
        raise RuntimeError('Unexpected compact schema')
    rows = []
    for j in range(169):
        block = str(schema.loc[j, 'feature_block'])
        fine = ''
        neuro = ''
        summary = ''
        if 0 <= j < 60:
            local = j
            summary = 'channel_median' if local < 30 else 'channel_iqr'
            k = local if local < 30 else local - 30
            band = BANDS[k // 5]
            tb = TF_BINS[k % 5]
            fine = f'TF|{summary}|{band}|{tb}'
            neuro = f'TF_band|{band}'
        elif 60 <= j < 72:
            local = j - 60
            summary = 'channel_median' if local < 6 else 'channel_iqr'
            k = local if local < 6 else local - 6
            fine = f'MORPH|{summary}|{MORPH[k]}'
            neuro = f'MORPH_metric|{MORPH[k]}'
        elif 72 <= j < 84:
            local = j - 72
            summary = 'channel_median' if local < 6 else 'channel_iqr'
            k = local if local < 6 else local - 6
            fine = f'PLV_channel|{summary}|{BANDS[k]}'
            neuro = f'PLV_channel_band|{BANDS[k]}'
        elif 84 <= j < 124:
            local = j - 84
            summary = 'channel_median' if local < 20 else 'channel_iqr'
            k = local if local < 20 else local - 20
            fine = f'DCT|{summary}|coef_{k + 1:02d}'
            neuro = f'DCT_coef|{k + 1:02d}'
        elif 124 <= j < 145:
            k = j - 124
            fine = f'COVSUM|{COVSUM[k]}'
            neuro = f'COVSUM|{COVSUM[k]}'
            summary = 'direct'
        else:
            k = j - 145
            band = BANDS[k // 4]
            st = PLVSTAT[k % 4]
            fine = f'PLV_global|{band}|{st}'
            neuro = f'PLV_global_band|{band}'
            summary = 'direct'
        rows.append({'compact_cell_0based': j, 'feature_name': schema.loc[j, 'feature_name'], 'feature_block': block, 'fine_source': fine, 'neuro_source': neuro, 'summary_type': summary})
    return pd.DataFrame(rows)

def reconstruct_trial_sources(load: pd.DataFrame, traj: pd.DataFrame, fam: pd.DataFrame):
    key = ['subject_id', 'raw_behavior_row_id', 'trial_number']
    f = fam[key + ['trajectory_family_1based']].copy()
    d = traj.merge(f, on=key, how='inner', validate='one_to_one')
    if len(d) != 1237 or d['subject_id'].nunique() != 31:
        raise RuntimeError(f'Trajectory/family merge drift: {len(d)}, {d.subject_id.nunique()}')
    rec_ab = np.zeros((len(d), 169), float)
    rec_bc = np.zeros((len(d), 169), float)
    for sid, idx in d.groupby('subject_id').groups.items():
        ii = np.array(list(idx), int)
        l = load[load.subject_id.eq(sid)].copy()
        pcs = sorted(l.pc_1based.unique())
        if pcs != list(range(1, len(pcs) + 1)) or len(pcs) != 8:
            raise RuntimeError(f'{sid}: unexpected PCA roster {pcs}')
        comp = np.zeros((8, 169), float)
        var = np.zeros(8, float)
        for pc in pcs:
            q = l[l.pc_1based.eq(pc)].sort_values('compact_cell_0based')
            if q.compact_cell_0based.tolist() != list(range(169)):
                raise RuntimeError(f'{sid} pc{pc}: loading cells drift')
            comp[pc - 1] = q.pca_component_loading.to_numpy(float)
            var[pc - 1] = float(q.explained_variance.iloc[0])
        ab = np.column_stack([d.loc[ii, f'late_minus_early_pc{k}'].to_numpy(float) for k in range(1, 9)])
        bc = np.column_stack([d.loc[ii, f'rating_minus_late_pc{k}'].to_numpy(float) for k in range(1, 9)])
        rec_ab[ii] = ab * np.sqrt(var)[None, :] @ comp
        rec_bc[ii] = bc * np.sqrt(var)[None, :] @ comp
    return (d, rec_ab, rec_bc)

def participant_balanced_contrast(d, values, metric_name, group_name, groups, B, seed):
    rows = []
    rng = np.random.default_rng(seed)
    subj = sorted(d.subject_id.unique())
    for gname, cols in groups.items():
        cols = np.asarray(cols, int)
        x = np.nanmean(values[:, cols], axis=1)
        c = []
        for sid in subj:
            q = np.where(d.subject_id.to_numpy() == sid)[0]
            y = d.iloc[q].trajectory_family_1based.to_numpy(int)
            a = x[q][y == 1]
            b = x[q][y == 2]
            c.append(np.nanmean(b) - np.nanmean(a) if len(a) and len(b) else np.nan)
        c = np.asarray(c, float)
        c = c[np.isfinite(c)]
        obs = float(np.mean(c))
        sd = float(np.std(c, ddof=1)) if len(c) > 1 else math.nan
        dz = obs / sd if np.isfinite(sd) and sd > 1e-12 else math.nan
        boots = np.empty(B, float)
        for b in range(B):
            take = rng.integers(0, len(c), size=len(c))
            boots[b] = float(np.mean(c[take]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({'metric': metric_name, 'grouping': group_name, 'source_group': gname, 'n_cells': len(cols), 'n_participants': len(c), 'family2_minus_family1_participant_mean': obs, 'participant_standardized_contrast': dz, 'participant_bootstrap_ci_low': float(lo), 'participant_bootstrap_ci_high': float(hi), 'ci_role': 'descriptive_stability_not_independent_hypothesis_test'})
    return pd.DataFrame(rows)

def corrected_partition_null(d, B, seed):
    rng = np.random.default_rng(seed)
    d = d.sort_values(['subject_id', 'trial_number']).reset_index(drop=True)
    subj = []
    for sid, q in d.groupby('subject_id', sort=True):
        y = q.trajectory_family_1based.eq(2).to_numpy(int)
        half = q.acquisition_fraction.to_numpy(float) <= 0.5
        imag = q.context.eq('imagery').to_numpy(bool)
        subj.append((sid, y, half, imag))

    def rho_for(labels_list, partition):
        aa = []
        bb = []
        for (sid, y, half, imag), yp in zip(subj, labels_list):
            m = half if partition == 'half' else imag
            if m.any() and (~m).any():
                aa.append(float(np.mean(yp[m])))
                bb.append(float(np.mean(yp[~m])))
        return float(spearmanr(aa, bb).statistic)
    orig = [x[1] for x in subj]
    out = []
    for part, name in [('half', 'first_half_vs_second_half'), ('context', 'imagery_vs_video')]:
        obs = rho_for(orig, part)
        null = np.empty(B, float)
        for b in range(B):
            labs = [rng.permutation(y) for _, y, _, _ in subj]
            null[b] = rho_for(labs, part)
        p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (B + 1))
        out.append({'partition': name, 'observed_spearman_rho': obs, 'null_mean_rho': float(np.mean(null)), 'null_sd_rho': float(np.std(null, ddof=1)), 'two_sided_composition_preserving_p': p, 'null_scheme': 'within-participant family-label shuffle preserving participant F1/F2 totals and fixed partition membership'})
    return pd.DataFrame(out)

def geometry_axis(cent):
    required = {'trajectory_family_1based', 'standardized_geometry_feature', 'centroid_value'}
    if not required.issubset(cent.columns):
        raise RuntimeError(f'Unexpected centroid schema: {cent.columns.tolist()}')
    c = cent.copy()
    if 'selected_shared_k' in c.columns:
        c = c[c.selected_shared_k.eq(2)].copy()
    wide = c.pivot(index='trajectory_family_1based', columns='standardized_geometry_feature', values='centroid_value')
    if wide.index.tolist() != [1, 2]:
        raise RuntimeError(f'Unexpected family centroid roster: {wide.index.tolist()}')
    rows = []
    for short, stdcol in [('early_to_late_norm', 'standardized_early_to_late_norm'), ('late_to_rating_norm', 'standardized_late_to_rating_norm'), ('early_to_rating_direct_norm', 'standardized_early_to_rating_direct_norm'), ('total_path_length', 'standardized_total_path_length'), ('directness_ratio', 'standardized_directness_ratio'), ('segment_cosine', 'standardized_segment_cosine')]:
        if short not in wide.columns:
            raise RuntimeError(f'Missing centroid geometry feature: {short}')
        a = float(wide.loc[1, short])
        b = float(wide.loc[2, short])
        diff = b - a
        rows.append({'geometry_dimension': stdcol, 'family1_centroid': a, 'family2_centroid': b, 'family2_minus_family1': diff})
    z = pd.DataFrame(rows)
    z['absolute_contrast_share'] = np.abs(z.family2_minus_family1) / np.abs(z.family2_minus_family1).sum()
    return z.sort_values('absolute_contrast_share', ascending=False)

def self_test():
    rng = np.random.default_rng(4)
    comp = np.linalg.qr(rng.normal(size=(6, 6)))[0].T[:3]
    var = np.array([4.0, 2.0, 0.5])
    dw = rng.normal(size=(5, 3))
    rec = dw * np.sqrt(var) @ comp
    if rec.shape != (5, 6) or not np.isfinite(rec).all():
        raise RuntimeError('reconstruction self-test failed')
    print('outcome_free_family_source_decomposition self-test: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--bootstraps', type=int, default=5000)
    ap.add_argument('--partition-permutations', type=int, default=10000)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    project = a.project.resolve()
    verify_inputs(project)
    out = project / 'outputs/interpretability/outcome_free_family_source_decomposition_outcome_free_family_source_decomposition'
    if out.exists():
        if a.force:
            import shutil
            shutil.rmtree(out)
        else:
            raise RuntimeError(f'Output exists: {out}; use --force to reproduce')
    (out / 'tables').mkdir(parents=True)
    (out / 'qc').mkdir()
    (out / 'reports').mkdir()
    schema = pd.read_csv(project / latent_state_design_SCHEMA)
    ss = source_schema(schema)
    load = pd.read_csv(project / latent_state_fit_LOAD)
    traj = pd.read_csv(project / latent_state_fit_TRAJ)
    fam = pd.read_csv(project / cross_person_alignment_fit_ASSIGN)
    cent = pd.read_csv(project / cross_person_alignment_fit_CENT)
    map4 = pd.read_csv(project / mapping_design_MAP, usecols=['subject_id', 'raw_behavior_row_id', 'trial_number', 'context', 'acquisition_fraction'])
    d, ab, bc = reconstruct_trial_sources(load, traj, fam)
    d = d.merge(map4, on=['subject_id', 'raw_behavior_row_id', 'trial_number'], how='left', validate='one_to_one')
    if d[['context', 'acquisition_fraction']].isna().any().any():
        raise RuntimeError('Outcome-free task-design lineage merge incomplete')
    ac = ab + bc
    quantities = {'early_to_late_reconstructed_energy': ab ** 2, 'late_to_rating_reconstructed_energy': bc ** 2, 'early_to_rating_reconstructed_energy': ac ** 2, 'segment_cross_product': ab * bc}
    block_groups = {k: v.compact_cell_0based.tolist() for k, v in ss.groupby('feature_block', sort=False)}
    neuro_groups = {k: v.compact_cell_0based.tolist() for k, v in ss.groupby('neuro_source', sort=False)}
    fine_groups = {k: v.compact_cell_0based.tolist() for k, v in ss.groupby('fine_source', sort=False)}
    pieces = []
    for mi, (m, x) in enumerate(quantities.items()):
        pieces.append(participant_balanced_contrast(d, x, m, 'feature_block', block_groups, a.bootstraps, SEED + mi * 100))
        pieces.append(participant_balanced_contrast(d, x, m, 'neuro_source', neuro_groups, a.bootstraps, SEED + mi * 100 + 1))
        pieces.append(participant_balanced_contrast(d, x, m, 'fine_source', fine_groups, a.bootstraps, SEED + mi * 100 + 2))
    attrib = pd.concat(pieces, ignore_index=True)
    attrib.to_csv(out / 'tables/outcome_free_family_source_decomposition_reconstructed_source_attribution.csv', index=False)
    rank = attrib.copy()
    rank['rank_abs_contrast'] = rank.groupby(['metric', 'grouping'])['participant_standardized_contrast'].transform(lambda s: s.abs().rank(method='min', ascending=False))
    rank = rank.sort_values(['metric', 'grouping', 'rank_abs_contrast'])
    rank.to_csv(out / 'tables/outcome_free_family_source_decomposition_source_attribution_ranked.csv', index=False)
    ga = geometry_axis(cent)
    ga.to_csv(out / 'tables/outcome_free_family_source_decomposition_family_geometry_axis_decomposition.csv', index=False)
    occ = corrected_partition_null(d, a.partition_permutations, SEED + 911)
    occ.to_csv(out / 'tables/outcome_free_family_source_decomposition_corrected_occupancy_partition_null.csv', index=False)
    ss.to_csv(out / 'tables/outcome_free_family_source_decomposition_compact_source_schema_interpreted.csv', index=False)
    pcrows = []
    for seg, prefix in [('early_to_late', 'late_minus_early'), ('late_to_rating', 'rating_minus_late')]:
        for k in range(1, 9):
            x = d[f'{prefix}_pc{k}'].to_numpy(float) ** 2
            cs = []
            for sid, q in d.groupby('subject_id'):
                ii = q.index.to_numpy(int)
                y = q.trajectory_family_1based.to_numpy(int)
                cs.append(float(np.mean(x[ii][y == 2]) - np.mean(x[ii][y == 1])))
            pcrows.append({'segment': seg, 'pc_variance_order': k, 'family2_minus_family1_whitened_pc_energy': float(np.mean(cs)), 'participant_standardized_contrast': float(np.mean(cs) / np.std(cs, ddof=1)) if np.std(cs, ddof=1) > 1e-12 else np.nan, 'interpretation_boundary': 'PC index is participant-specific variance order; not a shared anatomical component'})
    pd.DataFrame(pcrows).to_csv(out / 'tables/outcome_free_family_source_decomposition_pc_energy_attribution.csv', index=False)
    top = rank[(rank.grouping == 'feature_block') & (rank.rank_abs_contrast <= 6)].copy()
    top.to_csv(out / 'tables/outcome_free_family_source_decomposition_block_level_summary.csv', index=False)
    topfine = rank[(rank.grouping == 'neuro_source') & (rank.rank_abs_contrast <= 12)].copy()
    topfine.to_csv(out / 'tables/outcome_free_family_source_decomposition_top_neuro_source_summary.csv', index=False)
    gdict = {r.geometry_dimension: r.family2_minus_family1 for _, r in ga.iterrows()}
    interpretation = {'phase': PHASE, 'generated_at': now(), 'status': 'outcome_free_source_decomposition_complete', 'counts': {'trials': len(d), 'participants': d.subject_id.nunique(), 'compact_cells': 169}, 'fixed_geometric_reading': {'family1': 'shorter / more direct / more segment-aligned trajectory', 'family2': 'larger / longer / less direct / more segment-opposed trajectory', 'semantic_name_permitted': False}, 'occupancy_correction': {r.partition: {'observed_rho': float(r.observed_spearman_rho), 'composition_preserving_p': float(r.two_sided_composition_preserving_p)} for _, r in occ.iterrows()}, 'source_attribution_boundary': 'Attribution decomposes the already-discovered family geometry into the compact EEG representation from which it was built. Bootstrap CIs measure participant stability; they are not independent hypothesis tests.', 'topography_boundary': 'The 169-cell primary family representation collapsed channel-wise TF/morphology/PLV/DCT features to channel median+IQR. Therefore scalp topography was not part of the family-defining representation and cannot be claimed as the source of the K=2 split from outcome_free_family_source_decomposition.', 'behavior_outcomes_read': False, 'Validation_opened': False, 'Confirmation_opened': False, 'next_phase': 'full_channel_topology_association', 'next_phase_role': 'Optional outcome-free full-signal/topographic association design using fixed families, explicitly exploratory because channel-specific information was not used to define the families.'}
    writej(out / 'qc/outcome_free_family_source_decomposition_summary.json', interpretation)
    report = f"# outcome_free_family_source_decomposition — Outcome-free source decomposition of the two EEG trajectory families\n\nStatus: **complete**.\n\nNo behavioral report value, valence, liking, emotion score, Validation, or Confirmation data were read. Task-design columns used only for the corrected occupancy diagnostic were `context` and `acquisition_fraction`.\n\n## What is being decomposed\nThe two fixed families were defined from six geometric quantities derived from participant-specific whitened PCA trajectories over:\n`stimulus_early -> stimulus_late -> rating_early`.\n\noutcome_free_family_source_decomposition reconstructs, from the original fixed PCA loadings and explained variances, the portion of standardized 169-cell compact-feature displacement represented by the retained eight PCs. It then attributes:\n- early-to-late displacement energy;\n- late-to-rating displacement energy;\n- early-to-rating displacement energy;\n- segment cross-product (alignment vs reversal)\nto the original outcome-free feature blocks and interpretable frequency/summary subgroups.\n\nThis is **source attribution, not independent evidence**: the families were built from this representation.\n\n## Critical topography boundary\nThe primary 169-cell representation used channel median/IQR summaries for time-frequency, morphology, channel-PLV and waveform-DCT blocks. Therefore channel-specific scalp topography was *not* present in the family-defining representation. A later full-signal topographic analysis can only be an exploratory association with the fixed families, not a claim that a scalp map caused the K=2 solution.\n\n## Corrected occupancy diagnostic\nThe family_interpretability_extension split-occupancy correlation used an across-participant permutation that did not preserve within-participant compositional coupling. outcome_free_family_source_decomposition replaces it with within-participant label shuffles preserving each participant's total F1/F2 count and fixed partition memberships.\n\n## Claim boundary\nFamily 1/2 remain geometric labels. No positive/negative, imagery/video, appraisal/reporting, pleasure/pain, conflict, or other semantic relabeling is permitted here."
    (out / 'reports/outcome_free_family_source_decomposition_report.md').write_text(report, encoding='utf-8')
    contract = {'phase': PHASE, 'generated_at': now(), 'scientific_role': 'outcome-free decomposition/attribution of already-fixed K=2 trajectory geometry', 'new_family_discovery': False, 'new_semantic_inference': False, 'behavioral_report_values_read': False, 'validation_opened': False, 'confirmation_opened': False, 'source_attribution_is_independent_hypothesis_test': False, 'topography_used_in_family_definition': False, 'next_phase': 'full_channel_topology_association'}
    writej(out / 'qc/outcome_free_family_source_decomposition_contract.json', contract)
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    hashes = {str(p.relative_to(project)): sha256(p) for p in files}
    final = out / 'qc/outcome_free_family_source_decomposition_final_output_lock_hashes.json'
    writej(final, {'phase': PHASE, 'generated_at': now(), 'status': 'outcome_free_source_decomposition_complete', 'hashes': hashes, 'semantic_relabeling_permitted': False, 'behavioral_report_values_read': False, 'Validation_opened': False, 'Confirmation_opened': False, 'topography_used_in_family_definition': False, 'next_phase': 'full_channel_topology_association'})
    rels = [str(p.relative_to(project)) for p in sorted(out.rglob('*')) if p.is_file()]
    hz = project / '' / f'outcome_free_family_source_decomposition_outcome_free_family_source_decomposition_{stamp()}'
    hz.parent.mkdir(parents=True, exist_ok=True)
    h = sha256(hz)
    Path(str(hz) + '.sha256').write_text(f'{h}  {hz.name}', encoding='utf-8')
    print(f'[{now()}] outcome_free_family_source_decomposition PASS')
    print(f'trials={len(d)} participants={d.subject_id.nunique()} compact_cells=169')
    for _, r in occ.iterrows():
        print(f'{r.partition}: rho={r.observed_spearman_rho:.6f}; corrected_p={r.two_sided_composition_preserving_p:.6g}')
    print('Semantic relabeling permitted: False')
    print('Next phase: full_channel_topology_association')
    print('', hz)
    print('SHA-256:', h)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())