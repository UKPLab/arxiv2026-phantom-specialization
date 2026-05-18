#!/usr/bin/env python3
"""
Build the full token dataset (Stage 0).

Pipeline step 06.  Loads all tokens from merged_token_frequencies.tsv,
computes log-frequencies and percentiles, exports per-token records, and
runs validations (tokenization, distribution, band separation, power
analysis, inter-band tests).

Usage:
    python 06_build_token_dataset.py

Outputs:
    token_dataset/
    ├── pile_statistics.json             # Full Pile corpus stats
    ├── all_tokens_complete.json         # ALL 50k tokens with full info
    ├── all_tokens_complete.csv          # Same as CSV for analysis
    ├── frequency_bands.json             # Decile boundaries and counts
    ├── validation_report.json           # Pipeline validation results
    ├── critical_validations.json        # Tokenization & distribution checks
    ├── statistical_validations.json     # Power analysis & statistical tests
    └── README.txt                       # Description of outputs

Note: Use token_categories/token_categories.csv (from script 08) for
      authoritative BPE-aware token classification.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats as scipy_stats

# Import Stage 0 modules
from utils import FrequencyDataLoader, PythiaTokenizer, FrequencyTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            str(Path(__file__).resolve().parent / "token_dataset" / "pipeline.log")
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "token_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def export_pile_statistics(loader: FrequencyDataLoader, output_path: Path):
    """Export comprehensive Pile corpus statistics"""
    logger.info("Exporting Pile corpus statistics...")

    stats = loader.compute_frequency_stats()

    # Add metadata
    stats["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(loader.file_path),
        "pipeline_version": "Stage_0_v1.1_with_validations",
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved: {output_path}")
    return stats


def export_all_tokens_complete(
    loader: FrequencyDataLoader,
    transformer: FrequencyTransformer,
    tokenizer: PythiaTokenizer,
    output_json: Path,
    output_csv: Path,
):
    """
    Export COMPLETE information for ALL tokens with whitespace preservation
    """
    logger.info("Exporting complete information for ALL tokens...")
    logger.info(f"Processing {len(loader.token_to_freq)} tokens...")

    all_tokens_data = {}
    csv_rows = []

    total = len(loader.token_to_freq)
    for idx, token_id in enumerate(loader.token_to_freq.keys(), 1):
        if idx % 5000 == 0:
            logger.info(f"  Processed {idx}/{total} tokens...")

        # Get comprehensive info
        token_string = loader.token_to_string[token_id]
        raw_count = loader.token_to_freq[token_id]
        freq_per_million = transformer.normalized_frequencies[token_id]
        log_freq = transformer.log_frequencies[token_id]
        percentile = transformer.percentiles.get(token_id, None)

        # Test single-token status using round-trip test
        # (Ġ-prefixed tokens from get_vocab() need decode->encode round-trip, not lstrip)
        is_single_no_space = False
        is_single_with_space = False
        try:
            # Round-trip test: decode token_id -> text -> re-encode
            decoded_text = tokenizer.tokenizer.decode([token_id])
            ids_roundtrip = tokenizer.tokenizer.encode(
                decoded_text, add_special_tokens=False
            )
            is_single_no_space = (
                len(ids_roundtrip) == 1 and ids_roundtrip[0] == token_id
            )

            # With-space test: encode " " + stripped text
            ids_with_space = tokenizer.tokenizer.encode(
                " " + decoded_text.lstrip(), add_special_tokens=False
            )
            is_single_with_space = len(ids_with_space) == 1
        except Exception as e:
            logger.debug(f"Token {token_id} single-token test failed: {e}")

        token_data = {
            "token_id": token_id,
            "token_string": token_string,
            "raw_count": int(raw_count),
            "freq_per_million": float(freq_per_million),
            "log_frequency": float(log_freq),
            "percentile": float(percentile) if percentile is not None else None,
            "is_single_token_no_space": is_single_no_space,
            "is_single_token_with_space": is_single_with_space,
            "string_length": len(token_string),
        }

        all_tokens_data[str(token_id)] = token_data
        csv_rows.append(token_data)

    # Export JSON (nested)
    logger.info(f"Writing JSON with {len(all_tokens_data)} tokens...")
    with open(output_json, "w") as f:
        json.dump(all_tokens_data, f, indent=2)
    logger.info(f"Saved: {output_json}")

    # Export CSV (flat)
    logger.info(f"Writing CSV with {len(csv_rows)} rows...")
    df = pd.DataFrame(csv_rows)
    df = df.sort_values("raw_count", ascending=False)  # Sort by frequency
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved: {output_csv}")

    return all_tokens_data


def export_frequency_bands(transformer: FrequencyTransformer, output_path: Path):
    """Export frequency band boundaries and token counts"""
    logger.info("Computing frequency bands (deciles)...")

    log_freqs = list(transformer.log_frequencies.values())

    # Compute decile boundaries
    deciles = np.percentile(log_freqs, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    bands = {
        "decile_boundaries": {
            f"decile_{i + 1}": float(deciles[i]) for i in range(len(deciles))
        },
        "range": {
            "min_log_freq": float(min(log_freqs)),
            "max_log_freq": float(max(log_freqs)),
            "total_range": float(max(log_freqs) - min(log_freqs)),
        },
    }

    # Count tokens in each decile
    decile_ranges = [(min(log_freqs), deciles[0])]
    for i in range(len(deciles) - 1):
        decile_ranges.append((deciles[i], deciles[i + 1]))
    decile_ranges.append((deciles[-1], max(log_freqs)))

    bands["decile_counts"] = {}
    for idx, (min_log, max_log) in enumerate(decile_ranges):
        tokens_in_range = transformer.get_tokens_in_log_range(min_log, max_log)
        bands["decile_counts"][f"decile_{idx + 1}"] = {
            "min_log_freq": float(min_log),
            "max_log_freq": float(max_log),
            "num_tokens": len(tokens_in_range),
            "token_ids": tokens_in_range[:100],  # Sample for inspection
        }

    # Map to HF/MF/LF modes
    bands["mode_mapping"] = {
        "HF": {"deciles": [9, 10], "description": "Top 20% (9th and 10th decile)"},
        "MF": {"deciles": [5, 6], "description": "Middle 20% (5th and 6th decile)"},
        "LF": {"deciles": [1, 2], "description": "Bottom 20% (1st and 2nd decile)"},
    }

    with open(output_path, "w") as f:
        json.dump(bands, f, indent=2)

    logger.info(f"Saved: {output_path}")
    return bands


def run_critical_validations(
    loader: FrequencyDataLoader,
    transformer: FrequencyTransformer,
    tokenizer: PythiaTokenizer,
    output_path: Path,
) -> Dict:
    logger.info("Running validations...")

    validations = {"timestamp": datetime.now().isoformat(), "validations": {}}

    # Validation 1: Tokenization consistency check
    # Uses decode->encode round-trip (not raw vocab keys which contain Ġ prefix)
    logger.info("  1/3: Tokenization consistency...")
    test_token_ids = list(loader.token_to_string.keys())[:100]

    tokenization_issues = []
    for token_id in test_token_ids:
        try:
            # Correct round-trip: decode token_id -> text -> re-encode
            decoded_text = tokenizer.tokenizer.decode([token_id])
            encoded = tokenizer.tokenizer.encode(decoded_text, add_special_tokens=False)
            if len(encoded) != 1 or encoded[0] != token_id:
                tokenization_issues.append(
                    {
                        "token_id": token_id,
                        "token_string": loader.token_to_string[token_id],
                        "decoded_text": decoded_text,
                        "encoded_length": len(encoded),
                        "encoded_ids": encoded,
                    }
                )
        except Exception as e:
            tokenization_issues.append(
                {
                    "token_id": token_id,
                    "token_string": loader.token_to_string[token_id],
                    "error": str(e),
                }
            )

    validations["validations"]["tokenization_consistency"] = {
        "tokens_tested": len(test_token_ids),
        "issues_found": len(tokenization_issues),
        "sample_issues": tokenization_issues[:10],
        "status": "PASS" if len(tokenization_issues) == 0 else "WARNING",
    }

    # Validation 2: Frequency distribution (Zipf's law)
    logger.info("  2/3: Frequency distribution (Zipf's law)...")
    log_freqs = np.array(list(transformer.log_frequencies.values()))
    sorted_log_freqs = np.sort(log_freqs)[::-1]  # Descending
    ranks = np.arange(1, len(sorted_log_freqs) + 1)

    # Zipf's law: freq ∝ 1/rank^α  ->  log(freq) = -α·log(rank) + c
    # We fit log10(freq_per_million) vs log10(rank). Using freq_per_million
    # instead of raw counts only shifts the intercept (c), not the slope (α).
    # This is the standard log-log regression for Zipf analysis.
    log_ranks = np.log10(ranks)
    coeffs = np.polyfit(log_ranks, sorted_log_freqs, 1)
    alpha = -coeffs[0]  # Slope (should be ~1 for Zipf's law)
    r_squared = np.corrcoef(log_ranks, sorted_log_freqs)[0, 1] ** 2

    zipf_passed = bool(0.8 <= alpha <= 1.5 and r_squared > 0.95)
    validations["validations"]["zipf_distribution"] = {
        "alpha_exponent": float(alpha),
        "expected_alpha": 1.0,
        "r_squared": float(r_squared),
        "log_freq_range": float(log_freqs.max() - log_freqs.min()),
        "expected_range_min": 5.0,
        "status": "PASS" if zipf_passed else "WARNING",
    }

    # Validation 3: Band separation
    logger.info("  3/3: Band separation...")
    log_freqs_list = list(transformer.log_frequencies.values())
    deciles = np.percentile(log_freqs_list, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Check HF, MF, LF bands
    hf_tokens = transformer.get_tokens_in_log_range(deciles[7], max(log_freqs_list))
    mf_tokens = transformer.get_tokens_in_log_range(deciles[3], deciles[5])
    lf_tokens = transformer.get_tokens_in_log_range(min(log_freqs_list), deciles[1])

    hf_freqs = [transformer.log_frequencies[t] for t in hf_tokens]
    mf_freqs = [transformer.log_frequencies[t] for t in mf_tokens]
    lf_freqs = [transformer.log_frequencies[t] for t in lf_tokens]

    # Check for overlap
    hf_min, hf_max = min(hf_freqs), max(hf_freqs)
    mf_min, mf_max = min(mf_freqs), max(mf_freqs)
    lf_min, lf_max = min(lf_freqs), max(lf_freqs)

    overlap_hf_mf = bool(hf_min < mf_max)
    overlap_mf_lf = bool(mf_min < lf_max)

    validations["validations"]["band_separation"] = {
        "HF": {"min": float(hf_min), "max": float(hf_max), "n_tokens": len(hf_tokens)},
        "MF": {"min": float(mf_min), "max": float(mf_max), "n_tokens": len(mf_tokens)},
        "LF": {"min": float(lf_min), "max": float(lf_max), "n_tokens": len(lf_tokens)},
        "overlap_HF_MF": overlap_hf_mf,
        "overlap_MF_LF": overlap_mf_lf,
        "status": "PASS" if not (overlap_hf_mf or overlap_mf_lf) else "WARNING",
    }

    # Overall status
    all_pass = all(v["status"] == "PASS" for v in validations["validations"].values())
    validations["overall_status"] = "PASS" if all_pass else "WARNING"

    with open(output_path, "w") as f:
        json.dump(validations, f, indent=2)

    logger.info(f"Saved: {output_path}")
    return validations


def run_statistical_validations(
    transformer: FrequencyTransformer, output_path: Path
) -> Dict:

    logger.info("Running statistical validations...")

    validations = {"timestamp": datetime.now().isoformat(), "validations": {}}

    # Get log-frequencies for bands
    log_freqs_list = list(transformer.log_frequencies.values())
    deciles = np.percentile(log_freqs_list, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    hf_tokens = transformer.get_tokens_in_log_range(deciles[7], max(log_freqs_list))
    mf_tokens = transformer.get_tokens_in_log_range(deciles[3], deciles[5])
    lf_tokens = transformer.get_tokens_in_log_range(min(log_freqs_list), deciles[1])

    hf_freqs = np.array([transformer.log_frequencies[t] for t in hf_tokens])
    mf_freqs = np.array([transformer.log_frequencies[t] for t in mf_tokens])
    lf_freqs = np.array([transformer.log_frequencies[t] for t in lf_tokens])

    # Validation 1: Power analysis
    logger.info("  1/3: Power analysis...")
    # For detecting Cohen's d = 0.5 effect with α = 0.003125 (Bonferroni for 16 comparisons)
    # Using standard power analysis formulas
    alpha_bonferroni = 0.05 / 16
    effect_size = 0.5
    power = 0.80

    # Approximate sample size for t-test (two-tailed)
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha_bonferroni / 2)
    z_beta = norm.ppf(power)
    n_required = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    # Check if we have enough tokens
    min_tokens_available = min(len(hf_tokens), len(mf_tokens), len(lf_tokens))
    adequate_power = bool(min_tokens_available >= n_required)

    validations["validations"]["power_analysis"] = {
        "effect_size_target": effect_size,
        "alpha_bonferroni": alpha_bonferroni,
        "power_target": power,
        "n_required_per_band": int(np.ceil(n_required)),
        "n_available_HF": len(hf_tokens),
        "n_available_MF": len(mf_tokens),
        "n_available_LF": len(lf_tokens),
        "min_available": min_tokens_available,
        "adequate_power": adequate_power,
        "status": "PASS" if adequate_power else "WARNING",
    }

    # Validation 2: Inter-band statistical tests
    logger.info("  2/3: Inter-band validation (t-tests and ANOVA)...")

    # T-tests between pairs
    t_hf_mf, p_hf_mf = scipy_stats.ttest_ind(hf_freqs, mf_freqs)
    t_mf_lf, p_mf_lf = scipy_stats.ttest_ind(mf_freqs, lf_freqs)
    t_hf_lf, p_hf_lf = scipy_stats.ttest_ind(hf_freqs, lf_freqs)

    # ANOVA across all three bands
    f_stat, p_anova = scipy_stats.f_oneway(hf_freqs, mf_freqs, lf_freqs)

    # Convert to Python bool
    sig_hf_mf = bool(p_hf_mf < alpha_bonferroni)
    sig_mf_lf = bool(p_mf_lf < alpha_bonferroni)
    sig_hf_lf = bool(p_hf_lf < alpha_bonferroni)
    sig_anova = bool(p_anova < 0.05)
    all_sig = bool(sig_hf_mf and sig_mf_lf and sig_hf_lf)

    validations["validations"]["inter_band_tests"] = {
        "HF_vs_MF": {
            "t_statistic": float(t_hf_mf),
            "p_value": float(p_hf_mf),
            "significant": sig_hf_mf,
        },
        "MF_vs_LF": {
            "t_statistic": float(t_mf_lf),
            "p_value": float(p_mf_lf),
            "significant": sig_mf_lf,
        },
        "HF_vs_LF": {
            "t_statistic": float(t_hf_lf),
            "p_value": float(p_hf_lf),
            "significant": sig_hf_lf,
        },
        "ANOVA": {
            "f_statistic": float(f_stat),
            "p_value": float(p_anova),
            "significant": sig_anova,
        },
        "status": "PASS" if all_sig else "FAIL",
    }

    # Validation 3: Heteroscedasticity check
    logger.info("  3/3: Heteroscedasticity check...")

    # Levene's test for equality of variances
    levene_stat, levene_p = scipy_stats.levene(hf_freqs, mf_freqs, lf_freqs)

    # Variance ratios
    var_hf = np.var(hf_freqs)
    var_mf = np.var(mf_freqs)
    var_lf = np.var(lf_freqs)

    heteroscedastic = bool(levene_p < 0.05)
    var_ratio = float(var_lf / var_hf) if var_hf > 0 else None

    validations["validations"]["heteroscedasticity"] = {
        "variance_HF": float(var_hf),
        "variance_MF": float(var_mf),
        "variance_LF": float(var_lf),
        "std_HF": float(np.std(hf_freqs)),
        "std_MF": float(np.std(mf_freqs)),
        "std_LF": float(np.std(lf_freqs)),
        "variance_ratio_LF_HF": var_ratio,
        "levene_statistic": float(levene_stat),
        "levene_p_value": float(levene_p),
        "heteroscedastic": heteroscedastic,
        "recommendation": "Use robust standard errors or Welch t-test"
        if heteroscedastic
        else "Standard methods OK",
        "status": "INFO",  # informational
    }

    # Overall assessment
    validations["overall_assessment"] = {
        "power_adequate": adequate_power,
        "bands_statistically_distinct": all_sig,
        "variance_heterogeneity_detected": heteroscedastic,
        "recommended_methods": [
            "Mixed-effects models to account for variance differences"
            if heteroscedastic
            else "Standard parametric tests OK",
            "Bonferroni correction for multiple comparisons (α = 0.003125)",
            "Bootstrap confidence intervals for LF band due to higher variance"
            if var_lf > 2 * var_hf
            else "Standard CI OK",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(validations, f, indent=2)

    logger.info(f"Saved: {output_path}")
    return validations


def create_validation_report(
    loader: FrequencyDataLoader,
    transformer: FrequencyTransformer,
    tokenizer: PythiaTokenizer,
    output_path: Path,
):
    """Create validation report for pipeline"""
    logger.info("Generating validation report...")

    report = {
        "pipeline_name": "Stage 0 - Core Infrastructure",
        "execution_timestamp": datetime.now().isoformat(),
        "validation_checks": {},
        "summary": {},
    }

    # Check 1: Data loaded
    passed_1 = bool(45000 <= len(loader.token_to_freq) <= 55000)
    report["validation_checks"]["data_loaded"] = {
        "status": "PASS" if loader._loaded else "FAIL",
        "num_tokens": len(loader.token_to_freq),
        "expected_range": [45000, 55000],
        "passed": passed_1,
    }

    # Check 2: Total tokens reasonable (~300B for full Pile corpus)
    passed_2 = bool(280e9 < loader.total_tokens < 320e9)
    report["validation_checks"]["total_tokens"] = {
        "status": "PASS" if passed_2 else "FAIL",
        "total_tokens": int(loader.total_tokens),
        "expected_range": [280e9, 320e9],
        "passed": passed_2,
    }

    # Check 3: Log-frequencies computed
    passed_3 = bool(len(transformer.log_frequencies) == len(loader.token_to_freq))
    report["validation_checks"]["log_frequencies"] = {
        "status": "PASS" if len(transformer.log_frequencies) > 0 else "FAIL",
        "num_computed": len(transformer.log_frequencies),
        "passed": passed_3,
    }

    # Check 4: Percentiles computed
    passed_4 = bool(len(transformer.percentiles) == len(loader.token_to_freq))
    report["validation_checks"]["percentiles"] = {
        "status": "PASS" if len(transformer.percentiles) > 0 else "FAIL",
        "num_computed": len(transformer.percentiles),
        "passed": passed_4,
    }

    # Check 5: Zipf's law approximate verification
    log_freqs = list(transformer.log_frequencies.values())
    log_freq_std = np.std(log_freqs)
    passed_5 = bool((max(log_freqs) - min(log_freqs)) > 5.0)
    report["validation_checks"]["zipf_distribution"] = {
        "log_freq_range": float(max(log_freqs) - min(log_freqs)),
        "log_freq_std": float(log_freq_std),
        "expected_range_min": 5.0,
        "passed": passed_5,
        "status": "PASS" if passed_5 else "FAIL",
    }

    # Check 6: Tokenizer loaded
    report["validation_checks"]["tokenizer"] = {
        "status": "PASS" if tokenizer._loaded else "FAIL",
        "vocab_size": tokenizer.vocab_size,
        "model": tokenizer.model_name,
        "passed": tokenizer._loaded,
    }

    all_passed = all(check["passed"] for check in report["validation_checks"].values())

    report["summary"] = {
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(
            1 for check in report["validation_checks"].values() if check["passed"]
        ),
        "total_checks": len(report["validation_checks"]),
        "pipeline_ready": all_passed,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved: {output_path}")
    return report


def create_readme(output_dir: Path):
    """Create README describing outputs"""
    readme_content = """# Stage 0 Pipeline Outputs (With Validations)

Generated: {timestamp}

## Files Description

### pile_statistics.json
- Comprehensive statistics for the Pile corpus
- Total tokens, unique tokens, frequency distributions
- Percentiles (p1, p5, p10, ..., p99)
- Coverage analysis (top tokens covering X% of corpus)

### all_tokens_complete.json
- **COMPLETE dataset**: ALL {num_tokens} tokens with full information
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
"""

    # Get actual number of tokens
    with open(output_dir / "pile_statistics.json") as f:
        stats = json.load(f)
        num_tokens = stats["total_unique_tokens"]

    readme_content = readme_content.format(
        timestamp=datetime.now().isoformat(), num_tokens=num_tokens
    )

    with open(output_dir / "README.txt", "w") as f:
        f.write(readme_content)

    logger.info(f"Saved: {output_dir / 'README.txt'}")


def main():
    """Main pipeline execution"""
    print_section("STAGE 0: CORE INFRASTRUCTURE WITH VALIDATIONS")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Started at: {datetime.now().isoformat()}")

    start_time = datetime.now()

    try:
        # Step 1: Load frequency data
        print_section("loading pile token frequencies")
        loader = FrequencyDataLoader(
            str(SCRIPT_DIR / "pile_frequencies" / "merged_token_frequencies.tsv")
        )
        loader.load()
        print(f"Loaded {len(loader.token_to_freq):,} tokens")
        print(f"Total occurrences: {loader.total_tokens:,}")

        # Step 2: Load tokenizer
        print_section("loading pythia tokenizer")
        tokenizer = PythiaTokenizer("EleutherAI/pythia-70m")
        tokenizer.load()
        print(f"Loaded tokenizer: {tokenizer.model_name}")
        print(f"Vocabulary size: {tokenizer.vocab_size:,}")

        # Step 3: Compute log-frequencies
        print_section("computing log-frequencies")
        transformer = FrequencyTransformer(loader)
        transformer.compute_log_frequencies()
        print(
            f"Computed log-frequencies for {len(transformer.log_frequencies):,} tokens"
        )

        # Step 4: Compute percentiles
        print_section("computing percentiles")
        transformer.compute_percentiles()
        print(f"Computed percentiles for {len(transformer.percentiles):,} tokens")

        # Step 5: Export data
        print_section("exporting dataset")

        pile_stats = export_pile_statistics(loader, OUTPUT_DIR / "pile_statistics.json")
        all_tokens = export_all_tokens_complete(
            loader,
            transformer,
            tokenizer,
            OUTPUT_DIR / "all_tokens_complete.json",
            OUTPUT_DIR / "all_tokens_complete.csv",
        )
        bands = export_frequency_bands(transformer, OUTPUT_DIR / "frequency_bands.json")

        # Step 6: Basic validation
        print_section("basic validation")
        validation = create_validation_report(
            loader, transformer, tokenizer, OUTPUT_DIR / "validation_report.json"
        )

        print_section("Validations: Tokenization, Distribution, Bands")
        critical_val = run_critical_validations(
            loader, transformer, tokenizer, OUTPUT_DIR / "critical_validations.json"
        )

        print_section("Statistical Validations: Power, Tests, Heteroscedasticity")
        stat_val = run_statistical_validations(
            transformer, OUTPUT_DIR / "statistical_validations.json"
        )

        # Create README
        create_readme(OUTPUT_DIR)

        # Print all validation summaries
        print("\n" + "=" * 80)
        print("  VALIDATION SUMMARY")
        print("=" * 80)

        print("\nBasic Validation:")
        for check_name, check_data in validation["validation_checks"].items():
            status = "PASS" if check_data["status"] == "PASS" else "FAIL"
            print(f"  {status} - {check_name}")

        print("\nValidation:")
        for check_name, check_data in critical_val["validations"].items():
            status = f"{check_data['status']}"
            print(f"  {status} - {check_name}")

        print("\nStatistical Validation:")
        for check_name, check_data in stat_val["validations"].items():
            status = f"{check_data['status']}"
            print(f"  {status} - {check_name}")

        print("\n" + "=" * 80)
        print(f"  PIPELINE STATUS: {validation['summary']['overall_status']}")
        print(f"  STATUS: {critical_val['overall_status']}")
        print("=" * 80)

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print_section("PIPELINE COMPLETE")
        print(f"Processed {len(loader.token_to_freq):,} tokens")
        print(f"Generated 8 output files in: {OUTPUT_DIR}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"All validations completed")
        print("\nNext: Review validation reports before proceeding to Stage 1")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nPIPELINE FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
