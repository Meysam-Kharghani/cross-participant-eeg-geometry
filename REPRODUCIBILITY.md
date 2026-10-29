# Reproducibility

Run the repository verification suite:

```bash
bash scripts/verify_repository.sh
```

The suite checks SHA-256 integrity, source syntax, scientific-result gates, sensitivity summaries, and documentation paths.

## Figure reproduction

```bash
python3 -m pip install -r environment/requirements-figures.txt
python3 scripts/render_figures.py --output generated_figures
```

All seven figures are regenerated from repository result/source-data tables. Figure 4A is explicitly schematic; its transfer and family-balance panels are data-driven.

## Semantic target lineage

The semantic branch requires the corrected event-grammar lineage described in `DATA_INTEGRITY.md`. Discovery stimulus/category targets are derived only after event-based context reconstruction and context-specific sequence indexing. Validation and Confirmation specifications independently reconstruct held-out targets and prohibit canonical category/stimulus columns.

## Trajectory-geometry sensitivity

`trajectory_geometry_sensitivity/` evaluates algebraic redundancy, alternative representations and scaling, alternative clustering algorithms, matched continuous nulls, context-preserving temporal-destruction nulls, and centroid-derived continuous-axis stability.

`cluster_independent_axis_sensitivity/` estimates participant-balanced PCA axes without using cluster assignments and evaluates cross-split alignment and context-preserving nulls.

## Multiresolution continuous-null sensitivity

`multiresolution_continuous_null_sensitivity/` is rerunnable from curated derived inputs under `results/reproducibility_inputs/multiresolution_null/`. It evaluates Discovery compactness and the specificity of the existing held-out support criterion against a covariance-matched single-Gaussian reference.

## Existing-QC exclusion sensitivity

`existing_qc_exclusion_sensitivity/` is rerunnable from the consolidated QC table and assignment inputs. It applies stricter Discovery-derived historical QC thresholds unchanged to Validation and Confirmation.

Stages requiring reconstruction from raw EEG, GSR, or PPG recordings require the public EmoEEG-MC dataset. The repository emphasizes scientific auditability and reproducible derived calculations rather than machine-specific execution history.
