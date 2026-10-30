# Environment scope

This repository separates three reproducibility environments.

## Figure rendering
`requirements-figures.txt` contains the minimal dependencies used to regenerate the table-driven publication figures.

## Historical primary analysis
`requirements-analysis-unpinned.txt` records the package families used by the historical analysis pipeline. Exact historical package versions were not recoverable from the curated repository bundle, so this file is intentionally **not** presented as a bitwise environment lock.

## Post-hoc sensitivity analyses
`requirements-sensitivity-analysis.txt` provides a tested reference environment for the repository post-hoc trajectory-geometry, multiresolution continuous-null, and existing-QC exclusion sensitivity code. It is a reproducibility environment rather than a claim about the exact environment used at every historical or later execution stage.

This sensitivity environment must not be retroactively described as the environment used for the prespecified primary analysis.
