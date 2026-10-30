# primary analysis — Updated trajectory-geometry robustness audit
## Context-preserving null + continuous-axis analysis

**Status:** post-hoc sensitivity analysis. These analyses do not alter the fixed primary results.

## Why this update matters

The newly supplied trial-mapping manifest provides the elicitation context for every Discovery trajectory.
This permits a stricter permutation null that preserves each participant's imagery/video context while
destroying same-trial temporal pairing.

The key question is:

> Does the observed K=2 compactness/stability require intact early→late→rating trajectories, or can a
> similarly stable K=2 organization arise from the context-conditioned marginal distributions plus the
> trajectory representation itself?

## Context-preserving null design

Permutations were performed **within participant × context**. A no-fixed-point permutation was used when
an epoch or epoch-pair was shuffled.

Three nulls were tested, with 400 replicates each:

1. `rating_only`: retain the original early–late pair; shuffle rating within participant × context.
2. `late_rating_pair`: retain the original late–rating pair; shuffle that pair relative to early.
3. `late_rating_independent`: shuffle late and rating independently within participant × context.

The observed K=2 relative SSE gain was **0.311376**.

## Main context-preserving null results

              null_type  reps  observed_relative_SSE_gain  mean_gain  sd_gain  p95_gain  p99_gain  empirical_p_gain_ge_observed  mean_pb_support  sd_pb_support  mean_pb_family1  mean_min_sq_dist  mean_ARI_vs_observed  mean_aligned_agreement_vs_observed
       late_rating_pair   400                    0.311376   0.303700 0.006636  0.313326  0.316102                      0.089776         0.989888       0.002921         0.607257          2.563283             -0.000876                            0.518664
late_rating_independent   400                    0.311376   0.305149 0.007557  0.315904  0.320089                      0.179551         0.990452       0.003239         0.603184          2.546872             -0.000673                            0.519198
            rating_only   400                    0.311376   0.308374 0.007863  0.317972  0.322618                      0.319202         0.989254       0.003205         0.605272          2.598582              0.006653                            0.547490

### Interpretation

The context-preserving null materially changes the earlier trajectory-destruction result.

- `late_rating_pair`: mean null gain = 0.303700; empirical p = 0.0898
- `late_rating_independent`: mean null gain = 0.305149; empirical p = 0.1796
- `rating_only`: mean null gain = 0.308374; empirical p = 0.3192

Thus the observed K=2 compactness is **not clearly stronger** than a null that preserves participant and
elicitation context while destroying same-trial temporal correspondence.

The earlier participant-only shuffle had produced p≈0.012 for the stronger destruction nulls. The new
result shows that this apparent difference was partly caused by mixing imagery and video during
permutation. Once context is preserved, the K=2 compactness difference is no longer conventionally
significant.

## prespecified-support and family-balance diagnostics

Observed participant-balanced prespecified-support fraction is approximately 0.9943. In the context-preserving
nulls, the mean prespecified-support fraction remains approximately 0.989–0.990, and the participant-balanced
Family-1 proportion remains approximately 0.60–0.61.

Therefore:

> High in-support fraction and the ~60:40 family balance are not specific signatures of intact same-trial
> trajectory structure.

However, the null trajectories do **not** preserve the identity of the observed trial assignments.
Mean ARI between the null prespecified-centroid assignments and the real observed family labels is approximately
zero, and aligned agreement is only about 0.52–0.55.

So two facts coexist:

1. the global K=2 compactness/support/balance can be reproduced without intact same-trial pairing;
2. the exact trial-by-trial family membership cannot.

## Diagnostic null LOPO stability

A smaller diagnostic analysis (20 null realizations per null type) asked whether each null dataset could
itself produce a stable K=2 solution under participant deletion.

              null_type  mean_lopo  sd_lopo  p10_lopo  p90_lopo  mean_min_subject  mean_n95  mean_both  gate_pass_rate
late_rating_independent   0.983832 0.016361  0.956831  0.996847          0.881124     28.55       31.0            0.45
       late_rating_pair   0.985449 0.012565  0.972676  0.996766          0.892466     28.75       31.0            0.50
            rating_only   0.982781 0.013591  0.966936  0.996039          0.886831     27.85       31.0            0.30

Many context-preserving null realizations remain highly LOPO-stable, and a non-trivial fraction pass an
R8-like gate. This diagnostic used fewer repetitions than the main null and should not be treated as a
formal primary test, but it makes an important interpretive point:

> Very high participant-deletion stability is not, by itself, evidence that K=2 corresponds to two
> privileged biological or psychological states.

## Continuous separation-axis stability

The strongest new positive result is obtained by treating the geometry continuously rather than
taxonomically.

For each leave-one-participant-out fit, the centroid-separation direction was aligned to the full-fit
direction and the angle between them was calculated.

representation  mean_axis_cosine  min_axis_cosine  mean_axis_angle_deg  median_axis_angle_deg  max_axis_angle_deg  mean_LOPO_assignment_agreement
      prespecified6D          0.999059         0.995141             1.522927               0.407279            5.650448                        0.993952
   primitive3D          0.999926         0.999657             0.569285               0.440966            1.500504                        0.995847

For the prespecified 6D representation:
- mean axis cosine to full fit ≈ 0.99906
- mean angular deviation ≈ 1.52°
- maximum angular deviation ≈ 5.65°

For the non-redundant primitive 3D representation:
- mean axis cosine ≈ 0.99993
- mean angular deviation ≈ 0.57°
- maximum angular deviation ≈ 1.50°

This is very strong evidence that a **continuous organizing direction in trajectory-geometry space is
highly reproducible under participant deletion**.

Combined with the earlier findings that:
- the primitive 3D representation recovers ~96.3% of the prespecified 6D assignments,
- a covariance-matched unimodal Gaussian can produce comparable K=2 clusterability,
- and 1D mixture diagnostics do not consistently favor two components,

the most defensible interpretation is:

> EEG trajectories contain a highly reproducible low-dimensional geometric organization. A K=2
> partition is a stable and parsimonious quantization of that organization, but the available evidence
> does not establish two privileged discrete neural or psychological states.

## Recommended reporting wording

Preferred:
> “A highly reproducible low-dimensional trajectory geometry was recovered, with a stable K=2
> partition providing a parsimonious summary of the organization.”

or:
> “A reproducible continuous trajectory organization was observed, for which a two-family partition
> provided a stable coarse-grained description.”

Avoid wording that implies:
- exactly two natural neural states,
- an intrinsically bimodal distribution,
- or that high prespecified-support fraction alone demonstrates preserved same-trial temporal dynamics.

## Recommended supplement additions

The highest-value post-hoc sensitivity results are:

1. primitive 3D reparameterization;
2. matched single-Gaussian clusterability null;
3. continuous separation-axis LOPO stability;
4. participant × context trajectory-destruction null.

These should be explicitly labeled **post hoc sensitivity analyses**, not independent confirmation.

## Bottom line

The new context-preserving test weakens the interpretation of the **K=2 partition as a special discrete
structure**, but it does not erase the main structural finding. It shifts the strongest claim toward a
**highly reproducible continuous trajectory geometry / organizing axis**, while K=2 should be treated as
a convenient coarse-graining rather than an ontology.
