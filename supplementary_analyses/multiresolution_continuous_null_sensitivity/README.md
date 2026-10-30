# Multiresolution continuous-null calibration

This post-hoc sensitivity evaluates whether multiresolution EEG structure is unusually compact relative to a covariance-matched single-Gaussian reference in the analyzed standardized signature space, and whether the existing held-out support criterion is specific under that reference.

Across experience-related K=2,3,4 and report-transition K=2,3, the observed Discovery participant-weighted silhouette exceeded the Gaussian reference distribution. In contrast, the frozen Validation/Confirmation in-support criterion was highly permissive under the same reference: null support fractions were near one and every null replicate satisfied the existing structural-transfer gate.

The result therefore separates two claims. Discovery geometry is more compact than the matched Gaussian reference at the tested resolutions, but high held-out in-support is best interpreted as compatibility with the frozen coordinate geometry rather than as null-specific evidence for discrete latent states or a unique natural K.

## Reproduce

```bash
python supplementary_analyses/multiresolution_continuous_null_sensitivity/code/multiresolution_continuous_null_sensitivity.py \
  --out generated_multiresolution_null \
  --null-reps 500 \
  --null-n-init 20
```

The required curated derived inputs are released under `results/reproducibility_inputs/multiresolution_null/`.
