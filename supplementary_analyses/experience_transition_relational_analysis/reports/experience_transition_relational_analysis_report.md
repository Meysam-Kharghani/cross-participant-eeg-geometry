# Experience↔transition relational architecture

**Post-hoc branch; prespecified primary analysis/structural_replication/signal_space_characterization/primary_analysis_reference were not modified.**

Discovery-selected K4→K2 mapping: **E1->T1;E2->T1;E3->T2;E4->T2**

## H1 — fixed coarsening

| split        | mapping                     |   common_trials |   common_participants |   participant_balanced_BA |   BA_boot95_lo |   BA_boot95_hi |   fixed_mapping_permutation_p |   selection_corrected_permutation_p |   ARI_fixed_coarsening_vs_transition |   NMI_unweighted_fixed_coarsening_vs_transition |   heldout_parent_holm_p |
|:-------------|:----------------------------|----------------:|----------------------:|--------------------------:|---------------:|---------------:|------------------------------:|------------------------------------:|-------------------------------------:|------------------------------------------------:|------------------------:|
| Discovery    | E1->T1;E2->T1;E3->T2;E4->T2 |            1237 |                    31 |                  0.578955 |       0.55271  |       0.60628  |                    0.00019996 |                          0.00019996 |                           0.0413674  |                                      0.0195467  |             nan         |
| Validation   | E1->T1;E2->T1;E3->T2;E4->T2 |             437 |                    12 |                  0.551183 |       0.510503 |       0.594329 |                    0.0183963  |                        nan          |                           0.0238833  |                                      0.00853682 |               0.0735853 |
| Confirmation | E1->T1;E2->T1;E3->T2;E4->T2 |             459 |                    12 |                  0.522004 |       0.463686 |       0.580452 |                    0.182364   |                        nan          |                           0.00756612 |                                      0.0024432  |               0.547091  |


## H2 — conditional association

| split        |   common_trials |   common_participants |   participant_balanced_weighted_NMI |   NMI_boot95_lo |   NMI_boot95_hi |   mutual_information_nats |   H_experience_nats |   H_transition_nats |   transition_uncertainty_reduction_MI_over_HT |   uncertainty_reduction_boot95_lo |   uncertainty_reduction_boot95_hi |   within_participant_permutation_p |   heldout_parent_holm_p |
|:-------------|----------------:|----------------------:|------------------------------------:|----------------:|----------------:|--------------------------:|--------------------:|--------------------:|----------------------------------------------:|----------------------------------:|----------------------------------:|-----------------------------------:|------------------------:|
| Discovery    |            1237 |                    31 |                          0.0159424  |     0.0078678   |       0.0291482 |                0.0150644  |             1.33758 |            0.66754  |                                    0.022567   |                       0.0111312   |                         0.0413311 |                         0.00019996 |              nan        |
| Validation   |             437 |                    12 |                          0.00595575 |     0.00126215  |       0.0230216 |                0.00557289 |             1.32629 |            0.660161 |                                    0.00844172 |                       0.00179763  |                         0.0323652 |                         0.19816    |                0.547091 |
| Confirmation |             459 |                    12 |                          0.00118854 |     0.000428924 |       0.0205922 |                0.00112518 |             1.33153 |            0.673077 |                                    0.0016717  |                       0.000604753 |                         0.0288686 |                         0.824035   |                0.824035 |


## Decisions

| hypothesis                                       | status           | gate                                                                                                                           | max_claim_if_pass                                                                                                       |
|:-------------------------------------------------|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| H1_K4_experience_to_K2_transition_coarsening     | NOT_REPLICATED   | Both heldout: BA>0.5, bootstrap lower95>0.5, Holm p<0.05                                                                       | Internally replicated statistical coarsening alignment under the fixed Discovery map; not family identity or mechanism. |
| H2_experience_transition_conditional_association | NOT_REPLICATED   | Both heldout: Holm-adjusted within-participant permutation p<0.05; effect size reported as weighted NMI/uncertainty reduction. | Internally replicated conditional association; not determinism, independence, or biological mechanism.                  |
| H3_shared_14D_geometry_correspondence            | DESCRIPTIVE_ONLY | No inferential pass gate. Fixed-map contrast cosine, bootstrap CI, and rank among 14 mappings are descriptive.                 | Shared-coordinate profile resemblance only; not metric identity, source identity, or mechanism.                         |


## Interpretation boundary

A positive result can establish only internally replicated statistical alignment/association under the prespecified representations. It does not establish that experience and transition families are the same biological states, independent mechanisms, causal stages, or source-localized generators. H3 is descriptive only.
