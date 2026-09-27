# Run comparison: run1-imperfect vs. run2 (enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg)

## Run metadata

- **compared_at_utc**: 2026-09-27T17:43:14Z
- **work**: enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg
- **run1-imperfect model**: anthropic/claude-sonnet-5
- **run2 model**: anthropic/claude-sonnet-5
- **files_compared**:
  - source: data/raw/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.txt
  - run1-imperfect cleaned: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.baseline_claude-sonnet-5-api-run1-imperfect_20260924T000348Z.cleaned.json
  - run2 cleaned: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.cleaned.json
  - source: data/raw/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.pdf
  - run1-imperfect cleaned: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.baseline_claude-sonnet-5-api-run1-imperfect_20260924T000348Z.cleaned.json
  - run2 cleaned: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.cleaned.json
  - run1-imperfect bitext: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-eng.baseline_claude-sonnet-5-api-run1-imperfect_20260924T000348Z.bitext.jsonl
  - run2 bitext: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-eng.bitext.jsonl
- **caveat**: Cleaning/alignment overlap below is a strict exact-string match after whitespace/case normalization (see eval/metrics.py compare_cleaning). A legitimate but differently-split-or-merged unit (e.g. one run keeping two lines as a single unit where the other splits them -- a real risk on irregularly-punctuated source text) counts as a full mismatch here even when no content was lost or changed, only re-segmented. Low overlap alongside similar *_coverage_ratio values (fraction of raw text kept) points at segmentation differences rather than content differences; low overlap alongside diverging coverage ratios points at real content differences.

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
- **run_a_cleaned_unit_counts**: {'enm': 1968, 'eng': 866}
- **run_b_cleaned_unit_counts**: {'enm': 1724, 'eng': 881}
