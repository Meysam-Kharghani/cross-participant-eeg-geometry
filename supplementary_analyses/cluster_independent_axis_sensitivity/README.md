# Cluster-independent continuous-axis sensitivity

This post-hoc analysis asks whether the dominant trajectory-geometry direction reproduces across the internal participant splits without defining the direction from a clustering solution.

## Method

For both the prespecified six-coordinate trajectory representation and the nonredundant three-coordinate representation, the analysis:

1. applies the same coordinate transformations, participant-wise median/IQR scaling, and winsorization used by the trajectory sensitivity program;
2. computes a participant-balanced covariance matrix within each split;
3. extracts the leading principal-component direction without using family assignments or K-means centroids;
4. compares the acute angle between Discovery, Validation, and Confirmation axes;
5. destroys same-trial temporal pairing within participant × elicitation context and recomputes the Discovery axis over 400 seeded null replicates for each of three null types.

## Cross-split result

The leading six-coordinate axis explained 50.7%, 52.9%, and 51.0% of participant-balanced variance in Discovery, Validation, and Confirmation. Acute Discovery-to-Validation and Discovery-to-Confirmation angles were 5.34° and 8.29°.

For the nonredundant three-coordinate representation, the corresponding variance fractions were 56.7%, 58.0%, and 56.2%, with Discovery-to-Validation and Discovery-to-Confirmation angles of 2.96° and 9.22°.

## Null boundary

Context-preserving temporal-destruction nulls also generated leading axes close to the observed Discovery direction. Mean six-coordinate null angles were approximately 4.49° for rating-only shuffling, 7.39° for late-plus-rating pair shuffling, and 7.19° for independent late/rating shuffling. The three-coordinate nulls showed the same qualitative pattern.

The result therefore supports a reproducible leading direction in the analyzed coordinate system but does **not** show that axis orientation is specific to intact same-trial temporal correspondence. This is consistent with the broader conclusion that geometric reproducibility, privileged discreteness, semantic identification, and temporal-pairing specificity are distinct claims.

## Reproduction

From the repository root, run:

```bash
python3 supplementary_analyses/cluster_independent_axis_sensitivity/code/cluster_independent_axis_sensitivity.py \
  --discovery-trajectories results/reproducibility_inputs/trajectory_geometry/continuous_trajectories.csv \
  --discovery-manifest results/reproducibility_inputs/trajectory_geometry/trial_mapping_manifest.csv \
  --validation-trajectories results/reproducibility_inputs/trajectory_geometry/heldout/validation_raw_trajectory_geometry.csv \
  --validation-keys results/reproducibility_inputs/trajectory_geometry/heldout/validation_eligible_trajectory_keys.csv \
  --confirmation-trajectories results/reproducibility_inputs/trajectory_geometry/heldout/confirmation_raw_trajectory_geometry.csv \
  --confirmation-keys results/reproducibility_inputs/trajectory_geometry/heldout/confirmation_eligible_trajectory_keys.csv \
  --output-dir generated_sensitivity/cluster_independent_axis \
  --null-reps 400 \
  --seed 540
```

This analysis is post hoc and does not redefine any frozen primary support criterion.
