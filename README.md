# Cross-Participant Reproducible EEG Geometry During Emotional Experience

This repository contains a secondary analysis of the public EmoEEG-MC dataset (OpenNeuro `ds005540`). The primary objective is to test whether EEG recorded during emotional experience exhibits reproducible cross-participant geometry that can be distinguished from direct hedonic reconstruction, privileged discreteness, and semantic identification.

## Main findings

- Subjective reports show a reproducible low-dimensional organization and strong held-out reconstruction of valence and liking; this is a behavioral benchmark rather than the main EEG result.
- Direct EEG reconstruction of valence and liking is not supported in the tested signal-specific models.
- Outcome-free three-epoch EEG trajectory geometry transfers across non-overlapping participant splits within the same dataset.
- The original two-family partition is best interpreted as a stable coarse-graining of trajectory geometry, not as evidence for two privileged neural states.
- Experience-related structures at `K=2,3,4` and report-transition structures at `K=2,3` satisfy the frozen cross-split structural-transfer criteria.
- A matched continuous-Gaussian calibration shows that the held-out in-support criterion is highly permissive under the null reference. High in-support is therefore treated as compatibility with the frozen coordinate geometry, not as null-specific evidence for discrete latent structure.
- None of the 27 corrected split-level semantic/report parent tests passes multiplicity-adjusted significance.
- Compact EEG signal-space characterization shows cross-split stable, multicomponent patterns without identifying source-localized generators or causal mechanisms.
- Peripheral-physiology tests are unsupported in the analyzed representations, and exact raw EEG–peripheral event synchronization is not recovered at the required precision.
- A cluster-independent PCA audit finds closely aligned leading trajectory-geometry axes across splits, but context-preserving temporal-destruction nulls produce similarly oriented axes. Axis orientation alone is therefore not specific evidence for intact same-trial temporal pairing.

These results support reproducible **within-dataset EEG geometry** across held-out participants. They do not establish external replication, a unique natural number of emotional states, an emotion-specific neural code, causal source localization, or a direct neural readout of hedonic value.

## Participant splits

The fixed participant split contains 60 participants:

| Split | Participants |
|---|---:|
| Discovery | 36 |
| Validation | 12 |
| Confirmation | 12 |

Trajectory-based structural analyses use eligibility rules defined before held-out evaluation. The resulting eligible trajectory cohorts are 31 Discovery participants, 12 Validation participants, and 11 Confirmation participants. Other analyses may use different row counts according to their prespecified completeness rules; those counts are reported in the corresponding result tables.

## Meaning of “prespecified” in this repository

Within this secondary analysis, *prespecified* means that an analysis rule, feature set, threshold, or test was frozen **before the relevant held-out split or target values were opened for that stage**. It does not mean that the analysis was prospectively preregistered before the original EmoEEG-MC data were collected.

The repository distinguishes three evidence layers:

1. a frozen scientific core;
2. later held-out extensions whose rules were fixed before the relevant split was evaluated; and
3. explicitly labeled post-hoc sensitivity analyses.

## Semantic-target integrity

Stimulus and emotion-category targets used by the semantic analyses do not rely on the invalid raw-row assumption that trials 1–21 are imagery and trials 22–42 are video. The data-preparation audit explicitly detects that defect. Discovery context is reconstructed from event annotations, a context-specific trial index is recomputed in acquisition order, and stimulus/category candidates are mapped through the official README sequence. For Validation and Confirmation, the semantic specifications explicitly prohibit use of canonical stimulus/category targets and reconstruct held-out targets from authoritative context plus context-specific sequence position.

See `DATA_INTEGRITY.md` and `results/supplementary_tables/TableS3_data_integrity_summary.csv`.

## Relation to the original EmoEEG-MC decoding results

The original EmoEEG-MC study evaluates supervised emotion decoding from EEG using known emotion labels. The present semantic analyses ask a different question: whether **fixed outcome-free geometric families or signatures**, defined without using those labels, map reliably onto emotion categories or report structure across held-out participants. Failure of the present semantic mapping tests therefore does not contradict the possibility of supervised emotion decoding from EEG.

## Repository contents

- `src/` — scientific analysis source organized by analysis stage.
- `analysis_specifications/` — frozen and stage-specific analysis specifications retained for scientific provenance.
- `config/` — participant split and canonical analysis contracts.
- `results/` — main tables, source data, supplementary tables, and curated reproducibility inputs.
- `supplementary_analyses/` — explicitly post-hoc sensitivity analyses.
- `figures/` — table-driven figures.
- `environment/` — dependency scopes.
- `scripts/` — repository verification and figure-regeneration utilities.
- `provenance/` — scientific lineage, evidence index, upstream source hashes, and repository checksums.

The numbered specification and schema identifiers retained in `analysis_specifications/` and `config/` are scientific provenance identifiers used to preserve frozen analysis contracts.

## Verification

```bash
bash scripts/verify_repository.sh
```

The verification suite checks repository integrity, Python syntax, key scientific results, sensitivity summaries, and documentation paths.

## Figure regeneration

```bash
python3 -m pip install -r environment/requirements-figures.txt
python3 scripts/render_figures.py --output generated_figures
```

All seven figures are regenerated from repository result/source-data tables. Figure 4A is schematic; the transfer and family-balance panels are data-driven. Raw EEG arrays are not redistributed.

## Data availability

Raw EEG, GSR, and PPG recordings are not redistributed. Retrieval information for the source EmoEEG-MC dataset is provided in `data/README.md`.

## Author

**Meysam Kharghani**  
Department of Cognitive Neuroscience, Faculty of Education and Psychology, University of Tabriz, Tabriz, Iran  
ORCID: 0009-0004-9527-756X

## License

Code and repository documentation are available under the MIT License; see `LICENSE`. The source EmoEEG-MC dataset is not redistributed here and remains subject to the terms of its authoritative data source.
