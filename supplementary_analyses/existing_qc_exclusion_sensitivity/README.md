# Existing-QC exclusion sensitivity

This post-hoc sensitivity asks whether the reported trajectory and multiresolution transfer summaries depend on epochs with relatively extreme values under the historical EEG quality-control metrics.

Stricter thresholds were defined only from Discovery at the 95%, 97.5%, and 99% quantiles of maximum absolute amplitude, peak-to-peak amplitude, and robust scale, with a corresponding lower-tail threshold for retained channel count. The thresholds were then applied unchanged to Validation and Confirmation.

Even at the most stringent 95% setting, substantial trial fractions were removed while participant-balanced in-support remained approximately 0.993-1.000 across the tested held-out trajectory, experience, and report-transition structures. The result shows that the reported transfer compatibility is not driven by a small subset of visibly poor-QC epochs under the historical metrics. It does not establish absence of ocular or muscular contamination and is not a new preprocessing pipeline.

## Reproduce

```bash
python supplementary_analyses/existing_qc_exclusion_sensitivity/code/existing_qc_exclusion_sensitivity.py \
  --out generated_existing_qc_sensitivity
```

The curated QC and assignment inputs are released under `results/reproducibility_inputs/existing_qc_exclusion/` and `results/reproducibility_inputs/multiresolution_null/`.
