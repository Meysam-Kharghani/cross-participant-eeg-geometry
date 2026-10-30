# Trajectory-geometry sensitivity audit

**Status: POST HOC.** This analysis was conducted after the prespecified primary analysis sequence. It is an interpretive robustness audit, not preregistered evidence and not independent confirmation.

## Scientific question

Does the highly stable `K=2` trajectory solution reflect a representation-robust low-dimensional organization, and should the partition be interpreted as evidence for two privileged discrete states?

## Main findings

- The six trajectory descriptors are mathematically redundant; three primitive quantities determine the remaining geometric summaries.
- The nonredundant three-coordinate representation recovers about 96.3% of the prespecified six-coordinate assignments and has LOPO agreement of about 0.996.
- Exact `K=2` membership varies under reasonable feature ablations and alternative clustering algorithms even when each alternative remains internally stable.
- Covariance-matched unimodal Gaussian data can show comparable `K=2` compactness; separation-axis mixture diagnostics do not establish clear bimodality.
- Participant-by-context-preserving temporal-destruction nulls reproduce much of the global `K=2` compactness, support, and balance while not preserving the observed trial-level family identities.
- The continuous centroid-separation direction is highly stable under participant deletion in both the six-coordinate and primitive three-coordinate representations.

## Interpretation boundary

The preferred interpretation is a **highly reproducible low-dimensional trajectory geometry** for which `K=2` provides a stable, parsimonious coarse-graining. This sensitivity analysis does not establish two privileged discrete biological or psychological states.

## Reproducibility

Portable rerun scripts are provided in `supplementary_analyses/trajectory_geometry_sensitivity/code/`. Required analysis-ready inputs are in `results/reproducibility_inputs/trajectory_geometry/`.

From the repository root, install the sensitivity-analysis environment:

```bash
python3 -m pip install -r environment/requirements-sensitivity-analysis.txt
```

Run the portable core audit with:

```bash
python3 supplementary_analyses/trajectory_geometry_sensitivity/code/posthoc_geometry_sensitivity_portable.py \
  --project . \
  --out generated_sensitivity/core \
  --gaussian-reps 500 \
  --gmm-reps 300 \
  --destruction-reps 80 \
  --context-csv results/reproducibility_inputs/trajectory_geometry/trial_mapping_manifest.csv
```

Run the participant-by-context temporal-destruction null and centroid-derived continuous-axis analysis with:

```bash
python3 supplementary_analyses/trajectory_geometry_sensitivity/code/posthoc_geometry_context_preserving.py \
  --trajectories results/reproducibility_inputs/trajectory_geometry/continuous_trajectories.csv \
  --assignments results/reproducibility_inputs/trajectory_geometry/trajectory_family_assignments.csv \
  --lopo results/reproducibility_inputs/trajectory_geometry/trajectory_family_lopo_diagnostics.csv \
  --manifest results/reproducibility_inputs/trajectory_geometry/trial_mapping_manifest.csv \
  --out generated_sensitivity/context_preserving \
  --seed 540 \
  --null-reps 400 \
  --detailed-reps 100 \
  --null-lopo-reps 20
```

These reruns are computationally non-trivial because participant-deletion clustering is repeated many times. Released result tables and integrity checks permit direct numerical verification without recomputing every stochastic replicate.
