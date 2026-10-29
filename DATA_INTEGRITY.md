# Data integrity and semantic-target provenance

## Behavioral row-order hazard

The data-preparation audit identifies a source-code defect in which behavioral trial number was used to assign trials 1–21 to imagery and 22–42 to video. The experimental acquisition order is block-interleaved, so that rule is not valid for reconstructing context.

## Discovery correction

`src/analysis_pipeline/data_preparation/event_grammar_reconstruction.py` reconstructs trial context from event annotations and then:

1. maps reconstructed event trials back to raw behavioral trial numbers;
2. recomputes the context-specific trial index in acquisition order;
3. parses participant stimulus sequences directly from the source dataset README;
4. maps `(subject, context, context-specific trial index)` to stimulus identity and emotion category only when sequence position is structurally identifiable.

The semantic-analysis Discovery master is explicitly the corrected derivative-closure master produced after this audit; semantic analyses do not use the invalid row-based context rule.

## Held-out target reconstruction

Validation and Confirmation semantic specifications impose a stricter boundary: canonical `stimulus_id` and `emotion_category` are forbidden as held-out targets. Instead, held-out stimulus/category values are reconstructed from authoritative event-derived context and context-specific sequence position using the frozen README sequence parser. Missing or ambiguous mappings remain unavailable; there is no fallback to behavioral row order.

## Semantic-result consistency

`results/source_data/semantic_report_parent_tests.csv` contains 27 split-level parent tests across Discovery, Validation, and Confirmation. None passes multiplicity-adjusted significance. `results/supplementary_tables/TableS1_core_evidence_ledger.csv` and the main evidence summary are aligned to this final corrected target lineage.

## Scope

These corrections concern semantic target provenance. The outcome-free EEG geometry, fixed structural-transfer analysis, continuous-null calibrations, and signal-space characterizations do not depend on semantic category labels for defining their geometric structures.
