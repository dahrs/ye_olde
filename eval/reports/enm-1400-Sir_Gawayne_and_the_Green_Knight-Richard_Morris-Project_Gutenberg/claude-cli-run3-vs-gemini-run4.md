# Run comparison: claude-cli-run3 vs. gemini-run4 (enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg)

## Run metadata

- **compared_at_utc**: 2026-10-02T09:03:27Z
- **work**: enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg
- **claude-cli-run3 model**: claude_code_cli/sonnet
- **gemini-run4 model**: gemini/gemini-3.8-flash
- **files_compared**:
  - source: data/raw/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.txt
  - claude-cli-run3 cleaned: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.baseline_claude-code-cli-sonnet-run3_20261002T011745Z.cleaned.json
  - gemini-run4 cleaned: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.cleaned.json
  - source: data/raw/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.pdf
  - claude-cli-run3 cleaned: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.baseline_claude-code-cli-sonnet-run3_20261002T011745Z.cleaned.json
  - gemini-run4 cleaned: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.cleaned.json
  - claude-cli-run3 bitext: eval/baselines/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-eng.baseline_claude-code-cli-sonnet-run3_20261002T011745Z.bitext.jsonl
  - gemini-run4 bitext: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-eng.bitext.jsonl
- **caveat**: Cleaning/alignment overlap below is a strict exact-string match after whitespace/case normalization (see eval/metrics.py compare_cleaning). A legitimate but differently-split-or-merged unit (e.g. one run keeping two lines as a single unit where the other splits them -- a real risk on irregularly-punctuated source text) counts as a full mismatch here even when no content was lost or changed, only re-segmented. Low overlap alongside similar *_coverage_ratio values (fraction of raw text kept) points at segmentation differences rather than content differences; low overlap alongside diverging coverage ratios points at real content differences.

## Cleaning: enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg (enm)

- **baseline_unit_count**: 1724
- **local_unit_count**: 1724
- **exact_overlap_units**: 1724
- **overlap_precision**: 1.0000
- **overlap_recall**: 1.0000
- **baseline_coverage_ratio**: 0.4473
- **local_coverage_ratio**: 0.4473

## Cleaning: eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg (eng)

- **baseline_unit_count**: 881
- **local_unit_count**: 881
- **exact_overlap_units**: 881
- **overlap_precision**: 1.0000
- **overlap_recall**: 1.0000
- **baseline_coverage_ratio**: 0.9423
- **local_coverage_ratio**: 0.9423

## Alignment: enm-eng (coverage relative to gemini-run4's cleaned units)

- **baseline_pair_count**: 877
- **local_pair_count**: 557
- **full_pair_overlap**: 380
- **full_pair_precision**: 0.6835
- **full_pair_recall**: 0.4333
- **source_span_overlap**: 391
- **source_span_precision**: 0.7032
- **source_span_recall**: 0.4458
- **baseline_source_coverage**: 0.8058
- **local_source_coverage**: 0.6416
- **baseline_target_coverage**: 0.8173
- **local_target_coverage**: 0.6488
- **baseline**: {'avg_confidence': 0.8646610588820793, 'avg_links_per_pair': 3.5370581527936147, 'method_split': {'embedding': 25, 'llm': 852}}
- **local**: {'avg_confidence': 0.9478774661392882, 'avg_links_per_pair': 4.215439856373429, 'method_split': {'embedding': 25, 'llm': 532}}
- **local_verification_drop_rate**: 0.0054
- **run_a_verification_drop_rate**: 0.0570
- **run_a_cleaned_unit_counts**: {'enm': 1724, 'eng': 881}
- **run_b_cleaned_unit_counts**: {'enm': 1724, 'eng': 881}
