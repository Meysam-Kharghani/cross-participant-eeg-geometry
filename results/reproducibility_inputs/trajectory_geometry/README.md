# Trajectory-geometry reproducibility inputs

Derived, analysis-ready files support direct audit and sensitivity analysis without redistribution of raw EEG. Files include continuous trajectories, fixed family assignments, candidate metrics, leave-one-participant-out diagnostics, stability replicates, and the minimal trial mapping required for context-preserving null analyses.

`trial_mapping_manifest.csv` is intentionally data-minimized: it contains only participant/trial keys and elicitation context. Hedonic outcomes, affective ratings, stimulus labels, and emotion-category labels are not redistributed in this reproducibility-input table because they are not required by the trajectory-geometry sensitivity analyses.

The `heldout/` directory contains derived Validation and Confirmation raw trajectory geometry and eligible-row keys used only for the later post-hoc cluster-independent cross-split axis audit. It contains derived geometry rather than raw EEG.
