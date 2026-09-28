# Alignment comparison: Claude (hybrid) vs. algorithmic (trigram, no LLM)

## enm-eng

- **baseline_pair_count**: 752
- **local_pair_count**: 344
- **full_pair_overlap**: 146
- **full_pair_precision**: 0.4244
- **full_pair_recall**: 0.1947
- **source_span_overlap**: 182
- **source_span_precision**: 0.5291
- **source_span_recall**: 0.2427
- **baseline_source_coverage**: 0.8520
- **local_source_coverage**: 0.3011
- **baseline_target_coverage**: 0.8608
- **local_target_coverage**: 0.3496
- **baseline**: {'avg_confidence': 0.8934278572334888, 'avg_links_per_pair': 2.723404255319149, 'method_split': {'embedding': 25, 'llm': 727}}
- **local**: {'avg_confidence': 0.33295189990999446, 'avg_links_per_pair': 0.0, 'method_split': {'lexical': 344}}
- **claude_cleaned_source**: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.cleaned.json
- **claude_cleaned_target**: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/eng-1999-Sir_Gawayne_and_the_Green_Knight-w_A_Neilson-Project_Gutenberg.cleaned.json
- **claude_bitext**: data/processed/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg/enm-eng.bitext.jsonl
- **note**: Both runs use the IDENTICAL cleaned units above -- only the alignment method differs (Claude hybrid embedding+LLM vs. mode='algorithmic' character-trigram matching, no model of any kind).
