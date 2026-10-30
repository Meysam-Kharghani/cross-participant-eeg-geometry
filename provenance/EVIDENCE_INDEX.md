# Evidence index

This index maps the major scientific evidence branches to repository artifacts.

## Frozen scientific core

- Subjective-report organization and held-out hedonic reconstruction: `results/source_data/subjective_discovery_parallel_analysis.csv`, `results/source_data/subjective_validation_geometry.csv`, and `results/source_data/subjective_validation_prediction_summary.csv`.
- Direct EEG hedonic tests and participant/context diagnostics: `results/source_data/direct_eeg_and_diagnostic_metrics.csv`.
- Outcome-free local-state alignment: `results/source_data/local_state_alignment_candidates.csv` and `results/source_data/local_state_alignment_decision.csv`.
- Three-epoch trajectory geometry and fixed internal transfer: `results/source_data/trajectory_transfer_summary.csv` plus `results/reproducibility_inputs/trajectory_geometry/`.
- Core evidence ledger: `results/supplementary_tables/TableS1_core_evidence_ledger.csv`.

## Held-out extensions

- Multiresolution experience and report-transition structural transfer: `results/source_data/multiresolution_structural_transfer_summary.csv`.
- Splitwise semantic/report parent tests: `results/source_data/semantic_report_parent_tests.csv`.
- Semantic-target integrity summary: `results/supplementary_tables/TableS3_data_integrity_summary.csv` and `DATA_INTEGRITY.md`.
- Compact EEG signal-space characterization: `results/main_tables/Table2_signal_space_summary.csv`, `results/source_data/signal_space_block_summary.csv`, and `results/source_data/signal_space_domain_summary.csv`.
- Topographic extension boundary: `results/source_data/topography_extension_summary.csv`.
- Peripheral-physiology boundary analyses: `results/source_data/physiology_boundary_summary.csv`.
- Scientific design records: `analysis_specifications/` with `analysis_specifications/SPECIFICATION_INDEX.csv`.

## Post-hoc sensitivity analyses

- Quadrant semantic evaluation: `supplementary_analyses/quadrant_semantic_evaluation/`.
- Experience-to-report relational analysis: `supplementary_analyses/experience_transition_relational_analysis/`.
- Corrected within-trial geometry continuity analysis: `supplementary_analyses/within_trial_geometry_continuity/`.
- Trajectory discreteness, redundancy, alternative-algorithm, continuous-null, and temporal-pairing audits: `supplementary_analyses/trajectory_geometry_sensitivity/`.
- Cluster-independent cross-split PCA-axis audit: `supplementary_analyses/cluster_independent_axis_sensitivity/`.
- Multiresolution continuous-null calibration: `supplementary_analyses/multiresolution_continuous_null_sensitivity/` with curated rerun inputs in `results/reproducibility_inputs/multiresolution_null/`.
- Existing-QC exclusion sensitivity: `supplementary_analyses/existing_qc_exclusion_sensitivity/` with curated rerun inputs in `results/reproducibility_inputs/existing_qc_exclusion/`.
- Summary of post-hoc branches: `results/supplementary_tables/TableS2_supplementary_analysis_summary.csv`.

## Figure source coverage

All seven figures are regenerated from repository tables using `scripts/render_figures.py`. Figure 4A is schematic; its transfer and family-balance panels use repository data. Raw EEG, GSR, and PPG recordings remain external to the repository.
