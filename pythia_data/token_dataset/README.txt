# Stage 0 Pipeline Outputs (With Validations)

Generated: 2026-02-16

## Files Description

### pile_statistics.json
- Comprehensive statistics for the Pile corpus
- Total tokens, unique tokens, frequency distributions
- Percentiles (p1, p5, p10, ..., p99)
- Coverage analysis (top tokens covering X% of corpus)

### all_tokens_complete.json
- **COMPLETE dataset**: ALL 50063 tokens with full information
- Each token includes:
  - token_id, token_string, raw_count
  - freq_per_million, log_frequency, percentile
  - is_single_token_no_space, is_single_token_with_space
  - string_length
- Nested JSON structure for programmatic access
- For token classification, use token_categories/token_categories.csv (script 08)

### all_tokens_complete.csv
- Same as above in CSV format (flat structure)
- Sorted by frequency (descending)
- Easy to load in pandas, Excel, or other tools

### frequency_bands.json
- Decile boundaries in log-frequency space
- Token counts per decile
- Sample token_ids from each decile
- Mapping to HF/MF/LF modes for stratification

### validation_report.json
- Basic pipeline validation results
- Sanity checks (data loaded, sizes reasonable, etc.)

### critical_validations.json
- Tokenization consistency checks
- Zipf's law verification with R² score
- Band separation validation (no overlap)

### statistical_validations.json
- Power analysis: Sample size adequacy for Cohen's d = 0.5
- Inter-band tests: T-tests and ANOVA confirming statistical distinctness
- Heteroscedasticity check: Variance equality across bands
- Recommendations for appropriate statistical methods

## Next Steps

All validations should show PASS status before proceeding to Stage 1.
Review statistical validations to determine appropriate analysis methods.
