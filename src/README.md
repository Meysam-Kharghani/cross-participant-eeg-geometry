# Scientific analysis source

`analysis_pipeline/subjective_reports/` contains the subjective-report analyses. `analysis_pipeline/eeg/` contains implementation snapshots for EEG preprocessing, direct outcome tests, outcome-free geometry, held-out transfer, multiresolution structure, semantic/report analyses, signal-space characterization, topographic testing, and physiology boundaries. `supplementary_analyses/` contains the post-hoc experience-to-report and sensitivity analyses.

Source snapshots preserve scientific stage boundaries and analysis logic. Machine-specific defaults are represented by portable placeholders where necessary. Full raw-data execution requires the external EmoEEG-MC dataset and the upstream derived arrays described in `REPRODUCIBILITY.md`.
