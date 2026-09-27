# Claude determinism check: run1 (imperfect) vs. run2 (enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg)

## Cleaning: enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg (enm)

- **baseline_unit_count**: 1968
- **local_unit_count**: 1724
- **exact_overlap_units**: 860
- **overlap_precision**: 0.4988
- **overlap_recall**: 0.4370
- **baseline_coverage_ratio**: 0.4748
- **local_coverage_ratio**: 0.4473

## Cleaning: eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg (eng)

- **baseline_unit_count**: 866
- **local_unit_count**: 881
- **exact_overlap_units**: 726
- **overlap_precision**: 0.8241
- **overlap_recall**: 0.8383
- **baseline_coverage_ratio**: 0.9363
- **local_coverage_ratio**: 0.9423

## Alignment: enm-eng (coverage relative to run2's cleaned units)

- **baseline_pair_count**: 704
- **local_pair_count**: 752
- **full_pair_overlap**: 256
- **full_pair_precision**: 0.3413
- **full_pair_recall**: 0.3652
- **source_span_overlap**: 286
- **source_span_precision**: 0.3813
- **source_span_recall**: 0.4092
- **baseline_source_coverage**: 0.7447
- **local_source_coverage**: 0.8520
- **baseline_target_coverage**: 0.7555
- **local_target_coverage**: 0.8608
- **baseline**: {'avg_confidence': 0.8901937240091237, 'avg_links_per_pair': 2.7869318181818183, 'method_split': {'embedding': 22, 'llm': 682}}
- **local**: {'avg_confidence': 0.8934278572334888, 'avg_links_per_pair': 2.723404255319149, 'method_split': {'embedding': 25, 'llm': 727}}
- **local_verification_drop_rate**: 0.0105
- **run1_cleaned_unit_counts**: {'enm': 1968, 'eng': 866}
- **run2_cleaned_unit_counts**: {'enm': 1724, 'eng': 881}
