# Token ID vs Frequency Correlation Analysis

Generated: 2026-02-16

## Summary

This analysis examines the relationship between token IDs (vocabulary indices) 
and their actual frequencies in the Pile corpus.

## Key Findings

### Overall Correlation (Log Frequency)
- **Pearson r**: -0.7088 (very strong negative)
  - p-value: 0.00e+00
  - Significant: Yes

- **Spearman ρ**: -0.8058 (very strong negative)
  - p-value: 0.00e+00
  - Significant: Yes

### Interpretation

The correlation coefficient of -0.7088 indicates a 
very strong negative correlation between token ID and frequency.


This suggests that the tokenizer vocabulary IS somewhat ordered by frequency -
lower token IDs tend to 
have higher frequencies.


### Correlation by Frequency Band


**Very Low** (log freq: -5.00 to 0.23)
- Tokens: 10,013
- Pearson r: 0.4488 (moderate positive)
- Average Token ID: 38460.4

**Low** (log freq: 0.23 to 0.40)
- Tokens: 10,012
- Pearson r: -0.3469 (moderate negative)
- Average Token ID: 36392.6

**Medium** (log freq: 0.40 to 0.63)
- Tokens: 10,013
- Pearson r: -0.3861 (moderate negative)
- Average Token ID: 27042.3

**High** (log freq: 0.63 to 1.00)
- Tokens: 10,012
- Pearson r: -0.5306 (strong negative)
- Average Token ID: 17167.4

**Very High** (log freq: 1.00 to 4.58)
- Tokens: 10,013
- Pearson r: -0.5901 (strong negative)
- Average Token ID: 6694.9


## Files

1. **correlation_results.json**: Complete numerical results with all metrics
2. **correlation_scatter.png**: Scatter plots showing token_id vs frequency
3. **binned_analysis.png**: Average frequency per token ID bin
4. **correlation_by_bands.png**: Correlation coefficients by frequency band

## Data Source

- Input: <PROJECT_ROOT>/pythia_data/token_dataset/all_tokens_complete.csv
- Tokens analyzed: 50,063
- Analysis date: 2026-02-16T19:18:53.111542

## Implications


The negative correlation suggests that:
- Lower token IDs tend to have HIGHER frequencies
- The tokenizer may have prioritized common tokens in vocabulary construction
- This is consistent with BPE/tokenization algorithms that build from common patterns
