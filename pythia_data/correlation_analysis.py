#!/usr/bin/env python3
"""
Token ID vs Frequency Correlation Analysis
==========================================
Optional analysis script (not part of core pipeline).

Analyzes the correlation between token IDs (vocabulary index) and their
actual frequencies in the Pile corpus. This helps understand if the tokenizer's
vocabulary ordering has any relationship with token usage patterns. (following Niu et al. 2025)

Key Questions:
- Is there a correlation between token_id and frequency?
- Does the tokenizer prioritize common tokens with lower IDs?
- How does this correlation vary across frequency bands?

Usage:
    python correlation_analysis.py

Outputs:
    correlation_analysis/
    ├── correlation_results.json      # Correlation coefficients & statistics
    ├── correlation_scatter.png       # Scatter plot: token_id vs frequency
    ├── correlation_by_bands.png      # Correlation within frequency bands
    ├── binned_analysis.png           # Binned averages
    ├── correlation_analysis.log      # Execution log
    └── README.txt                    # Description of findings
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats


logger = logging.getLogger(__name__)

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "token_dataset" / "all_tokens_complete.csv"
OUTPUT_DIR = SCRIPT_DIR / "correlation_analysis"

try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Test write permissions
    test_file = OUTPUT_DIR / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
except Exception as e:
    print(f"ERROR: Cannot write to {OUTPUT_DIR}")
    print(f"Details: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "correlation_analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger.info(f"Output directory ready: {OUTPUT_DIR}")
logger.info(f"Log file: {OUTPUT_DIR / 'correlation_analysis.log'}")


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_token_data(data_path: Path) -> pd.DataFrame:
    """
    Load complete token dataset

    Returns:
        DataFrame with all token information
    """
    logger.info(f"Loading token data from {data_path}...")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} tokens")

    return df


def compute_correlations(df: pd.DataFrame) -> Dict:
    """
    Compute various correlation metrics between token_id and frequencies

    Args:
        df: DataFrame with token_id and frequency columns

    Returns:
        Dict with correlation results
    """
    logger.info("Computing correlation coefficients...")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_tokens": len(df),
        "correlations": {},
    }

    # For each frequency measure
    freq_columns = ["raw_count", "freq_per_million", "log_frequency"]

    for freq_col in freq_columns:
        logger.info(f"  Computing correlations for {freq_col}...")

        token_ids = df["token_id"].values
        frequencies = df[freq_col].values

        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = scipy_stats.pearsonr(token_ids, frequencies)

        # Spearman correlation (monotonic relationship, rank-based)
        spearman_r, spearman_p = scipy_stats.spearmanr(token_ids, frequencies)

        # Kendall's tau (another rank-based correlation)
        kendall_tau, kendall_p = scipy_stats.kendalltau(token_ids, frequencies)

        results["correlations"][freq_col] = {
            "pearson": {
                "coefficient": float(pearson_r),
                "p_value": float(pearson_p),
                "interpretation": interpret_correlation(pearson_r),
                "significant": bool(pearson_p < 0.05),
            },
            "spearman": {
                "coefficient": float(spearman_r),
                "p_value": float(spearman_p),
                "interpretation": interpret_correlation(spearman_r),
                "significant": bool(spearman_p < 0.05),
            },
            "kendall_tau": {
                "coefficient": float(kendall_tau),
                "p_value": float(kendall_p),
                "interpretation": interpret_correlation(kendall_tau),
                "significant": bool(kendall_p < 0.05),
            },
        }

    logger.info("Correlations computed")
    return results


def interpret_correlation(r: float) -> str:
    """
    Interpret correlation coefficient magnitude

    Args:
        r: Correlation coefficient

    Returns:
        String interpretation
    """
    abs_r = abs(r)

    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if r > 0 else "negative"

    return f"{strength} {direction}"


def analyze_by_frequency_bands(df: pd.DataFrame) -> Dict:
    """
    Analyze correlation within different frequency bands

    Args:
        df: DataFrame with token data

    Returns:
        Dict with band-specific correlations
    """
    logger.info("Analyzing correlations by frequency bands...")

    # Define bands based on log-frequency deciles
    log_freqs = df["log_frequency"].values
    deciles = np.percentile(log_freqs, [20, 40, 60, 80])

    bands = {
        "very_low": (log_freqs.min(), deciles[0]),
        "low": (deciles[0], deciles[1]),
        "medium": (deciles[1], deciles[2]),
        "high": (deciles[2], deciles[3]),
        "very_high": (deciles[3], log_freqs.max()),
    }

    results = {}

    for band_name, (min_log, max_log) in bands.items():
        band_df = df[
            (df["log_frequency"] >= min_log) & (df["log_frequency"] <= max_log)
        ]

        if len(band_df) < 10:
            continue

        pearson_r, pearson_p = scipy_stats.pearsonr(
            band_df["token_id"].values, band_df["log_frequency"].values
        )

        spearman_r, spearman_p = scipy_stats.spearmanr(
            band_df["token_id"].values, band_df["log_frequency"].values
        )

        results[band_name] = {
            "log_freq_range": [float(min_log), float(max_log)],
            "n_tokens": len(band_df),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "avg_token_id": float(band_df["token_id"].mean()),
            "interpretation": interpret_correlation(pearson_r),
        }

    logger.info(f"Analyzed {len(results)} frequency bands")
    return results


def create_scatter_plot(df: pd.DataFrame, output_path: Path):
    """
    Create scatter plot of token_id vs log_frequency

    Args:
        df: DataFrame with token data
        output_path: Path to save figure
    """
    logger.info("Creating scatter plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Token ID vs Frequency Correlation Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Full scatter plot (log frequency)
    ax = axes[0, 0]
    ax.scatter(df["token_id"], df["log_frequency"], alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Log Frequency")
    ax.set_title("Token ID vs Log Frequency (All Tokens)")
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(df["token_id"], df["log_frequency"], 1)
    p = np.poly1d(z)
    ax.plot(
        df["token_id"],
        p(df["token_id"]),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z[0]:.2e}x+{z[1]:.2f}",
    )
    ax.legend()

    # 2. Hexbin plot for density
    ax = axes[0, 1]
    hb = ax.hexbin(
        df["token_id"], df["log_frequency"], gridsize=50, cmap="Blues", mincnt=1
    )
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Log Frequency")
    ax.set_title("Token ID vs Log Frequency (Density)")
    plt.colorbar(hb, ax=ax, label="Count")

    # 3. Raw count (log scale)
    ax = axes[1, 0]
    ax.scatter(df["token_id"], df["raw_count"], alpha=0.3, s=10, c="coral")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Raw Count")
    ax.set_yscale("log")
    ax.set_title("Token ID vs Raw Count (Log Scale)")
    ax.grid(True, alpha=0.3, which="both")

    # 4. Distribution of token IDs by frequency quantile
    ax = axes[1, 1]

    # Divide into quartiles by log_frequency
    df_sorted = df.sort_values("log_frequency")
    n_per_quartile = len(df) // 4
    quartiles = []
    labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]

    for i in range(4):
        start_idx = i * n_per_quartile
        end_idx = (i + 1) * n_per_quartile if i < 3 else len(df)
        quartile_ids = df_sorted.iloc[start_idx:end_idx]["token_id"].values
        quartiles.append(quartile_ids)

    bp = ax.boxplot(quartiles, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_xlabel("Frequency Quartile")
    ax.set_ylabel("Token ID")
    ax.set_title("Distribution of Token IDs by Frequency Quartile")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved scatter plot: {output_path}")


def create_binned_analysis_plot(df: pd.DataFrame, output_path: Path):
    """
    Create binned analysis showing average frequency per token ID bin

    Args:
        df: DataFrame with token data
        output_path: Path to save figure
    """
    logger.info("Creating binned analysis plot...")

    # Create bins for token IDs
    n_bins = 50
    df_sorted = df.sort_values("token_id")
    df_sorted["token_id_bin"] = pd.cut(df_sorted["token_id"], bins=n_bins)

    # Compute statistics per bin
    binned_stats = (
        df_sorted.groupby("token_id_bin")
        .agg(
            {
                "token_id": ["mean", "count"],
                "log_frequency": ["mean", "std"],
                "raw_count": "mean",
            }
        )
        .reset_index()
    )

    binned_stats.columns = [
        "bin",
        "avg_token_id",
        "count",
        "avg_log_freq",
        "std_log_freq",
        "avg_raw_count",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "Binned Analysis: Average Frequency by Token ID Range",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Average log frequency per bin with error bars
    ax = axes[0]
    ax.errorbar(
        binned_stats["avg_token_id"],
        binned_stats["avg_log_freq"],
        yerr=binned_stats["std_log_freq"],
        fmt="o-",
        capsize=5,
        capthick=2,
        color="steelblue",
        ecolor="lightblue",
        markersize=6,
        linewidth=2,
        alpha=0.8,
    )
    ax.set_xlabel("Token ID (Bin Center)", fontsize=12)
    ax.set_ylabel("Average Log Frequency", fontsize=12)
    ax.set_title(f"Average Log Frequency per Token ID Bin (n={n_bins} bins)")
    ax.grid(True, alpha=0.3)

    # 2. Token count per bin
    ax = axes[1]
    ax.bar(
        binned_stats["avg_token_id"],
        binned_stats["count"],
        width=(binned_stats["avg_token_id"].max() - binned_stats["avg_token_id"].min())
        / n_bins
        * 0.9,
        color="coral",
        alpha=0.7,
        edgecolor="darkred",
    )
    ax.set_xlabel("Token ID (Bin Center)", fontsize=12)
    ax.set_ylabel("Number of Tokens", fontsize=12)
    ax.set_title("Token Distribution Across ID Bins")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved binned analysis: {output_path}")


def create_band_correlation_plot(band_results: Dict, output_path: Path):
    """
    Create plot showing correlations by frequency band

    Args:
        band_results: Dict with band-specific correlations
        output_path: Path to save figure
    """
    logger.info("Creating band correlation plot...")

    # Extract data
    bands = list(band_results.keys())
    pearson_rs = [band_results[b]["pearson_r"] for b in bands]
    spearman_rs = [band_results[b]["spearman_r"] for b in bands]
    n_tokens = [band_results[b]["n_tokens"] for b in bands]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Correlation by Frequency Band", fontsize=16, fontweight="bold")

    # 1. Correlation coefficients
    ax = axes[0]
    x = np.arange(len(bands))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        pearson_rs,
        width,
        label="Pearson r",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, spearman_rs, width, label="Spearman ρ", color="coral", alpha=0.8
    )

    ax.set_xlabel("Frequency Band", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title("Pearson and Spearman Correlations by Band")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace("_", " ").title() for b in bands], rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

    # 2. Number of tokens per band
    ax = axes[1]
    ax.bar(x, n_tokens, color="lightgreen", alpha=0.7, edgecolor="darkgreen")
    ax.set_xlabel("Frequency Band", fontsize=12)
    ax.set_ylabel("Number of Tokens", fontsize=12)
    ax.set_title("Token Count by Frequency Band")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace("_", " ").title() for b in bands], rotation=45, ha="right"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, v in enumerate(n_tokens):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved band correlation plot: {output_path}")


def create_readme(results: Dict, band_results: Dict, output_dir: Path):
    """
    Create README describing findings

    Args:
        results: Overall correlation results
        band_results: Band-specific results
        output_dir: Output directory
    """
    # Extract key findings
    log_pearson = results["correlations"]["log_frequency"]["pearson"]
    log_spearman = results["correlations"]["log_frequency"]["spearman"]

    readme_content = f"""# Token ID vs Frequency Correlation Analysis

Generated: {datetime.now().isoformat()}

## Summary

This analysis examines the relationship between token IDs (vocabulary indices) 
and their actual frequencies in the Pile corpus.

## Key Findings

### Overall Correlation (Log Frequency)
- **Pearson r**: {log_pearson["coefficient"]:.4f} ({log_pearson["interpretation"]})
  - p-value: {log_pearson["p_value"]:.2e}
  - Significant: {"Yes" if log_pearson["significant"] else "No"}

- **Spearman ρ**: {log_spearman["coefficient"]:.4f} ({log_spearman["interpretation"]})
  - p-value: {log_spearman["p_value"]:.2e}
  - Significant: {"Yes" if log_spearman["significant"] else "No"}

### Interpretation

The correlation coefficient of {log_pearson["coefficient"]:.4f} indicates a 
{log_pearson["interpretation"]} correlation between token ID and frequency.

"""

    if abs(log_pearson["coefficient"]) > 0.3:
        readme_content += f"""
This suggests that the tokenizer vocabulary IS somewhat ordered by frequency -
{"lower" if log_pearson["coefficient"] < 0 else "higher"} token IDs tend to 
have {"higher" if log_pearson["coefficient"] < 0 else "lower"} frequencies.
"""
    else:
        readme_content += """
This suggests that the tokenizer vocabulary is NOT strongly ordered by frequency.
Token IDs appear to be relatively independent of actual usage patterns in the corpus.
"""

    readme_content += f"""

### Correlation by Frequency Band

"""
    for band_name, band_data in band_results.items():
        readme_content += f"""
**{band_name.replace("_", " ").title()}** (log freq: {band_data["log_freq_range"][0]:.2f} to {band_data["log_freq_range"][1]:.2f})
- Tokens: {band_data["n_tokens"]:,}
- Pearson r: {band_data["pearson_r"]:.4f} ({band_data["interpretation"]})
- Average Token ID: {band_data["avg_token_id"]:.1f}
"""

    readme_content += f"""

## Files

1. **correlation_results.json**: Complete numerical results with all metrics
2. **correlation_scatter.png**: Scatter plots showing token_id vs frequency
3. **binned_analysis.png**: Average frequency per token ID bin
4. **correlation_by_bands.png**: Correlation coefficients by frequency band

## Data Source

- Input: {DATA_PATH}
- Tokens analyzed: {results["n_tokens"]:,}
- Analysis date: {results["timestamp"]}

## Implications

"""

    if abs(log_pearson["coefficient"]) < 0.1:
        readme_content += """
The negligible correlation suggests that:
- Token IDs are essentially random with respect to frequency
- Tokenizer design did not prioritize common tokens with lower IDs
- Token stratification by frequency must use actual frequency data, not token IDs
"""
    elif log_pearson["coefficient"] < -0.3:
        readme_content += """
The negative correlation suggests that:
- Lower token IDs tend to have HIGHER frequencies
- The tokenizer may have prioritized common tokens in vocabulary construction
- This is consistent with BPE/tokenization algorithms that build from common patterns
"""
    else:
        readme_content += """
The positive correlation suggests that:
- Higher token IDs tend to have HIGHER frequencies
- This is unusual and may warrant further investigation
- Could indicate vocabulary was sorted or constructed in a non-standard way
"""

    with open(output_dir / "README.txt", "w") as f:
        f.write(readme_content)

    logger.info(f"Created README: {output_dir / 'README.txt'}")


def main():
    """Main analysis execution"""
    print_section("TOKEN ID vs FREQUENCY CORRELATION ANALYSIS")
    print(f"Data source: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Started at: {datetime.now().isoformat()}")

    # Verify input file exists
    if not DATA_PATH.exists():
        print(f"\nERROR: Input file not found: {DATA_PATH}")
        print("Please run 06_build_token_dataset.py first to generate this file.")
        return 1

    start_time = datetime.now()

    try:
        # Load data
        print_section("loading token data")
        df = load_token_data(DATA_PATH)
        print(f"Loaded {len(df):,} tokens")
        print(f"  Token ID range: {df['token_id'].min()} - {df['token_id'].max()}")
        print(
            f"  Log freq range: {df['log_frequency'].min():.2f} - {df['log_frequency'].max():.2f}"
        )

        # Compute overall correlations
        print_section("overall correlations")
        results = compute_correlations(df)

        # Display results
        for freq_type, corr_data in results["correlations"].items():
            print(f"\n{freq_type}:")
            print(
                f"  Pearson r:  {corr_data['pearson']['coefficient']:7.4f} "
                f"({corr_data['pearson']['interpretation']})"
            )
            print(
                f"  Spearman ρ: {corr_data['spearman']['coefficient']:7.4f} "
                f"({corr_data['spearman']['interpretation']})"
            )
            print(
                f"  Kendall τ:  {corr_data['kendall_tau']['coefficient']:7.4f} "
                f"({corr_data['kendall_tau']['interpretation']})"
            )

        # Analyze by frequency bands
        print_section("correlations by band")
        band_results = analyze_by_frequency_bands(df)
        for band_name, band_data in band_results.items():
            print(f"\n{band_name.replace('_', ' ').title()}:")
            print(f"  Tokens: {band_data['n_tokens']:,}")
            print(
                f"  Pearson r: {band_data['pearson_r']:7.4f} ({band_data['interpretation']})"
            )

        # Create visualizations
        print_section("scatter plots")
        scatter_path = OUTPUT_DIR / "correlation_scatter.png"
        create_scatter_plot(df, scatter_path)
        if scatter_path.exists():
            print(f"Saved: {scatter_path}")
        else:
            print(f"WARNING: Failed to save {scatter_path}")

        print_section("additional plots")
        binned_path = OUTPUT_DIR / "binned_analysis.png"
        create_binned_analysis_plot(df, binned_path)
        if binned_path.exists():
            print(f"Saved: {binned_path}")
        else:
            print(f"WARNING: Failed to save {binned_path}")

        band_path = OUTPUT_DIR / "correlation_by_bands.png"
        create_band_correlation_plot(band_results, band_path)
        if band_path.exists():
            print(f"Saved: {band_path}")
        else:
            print(f"WARNING: Failed to save {band_path}")

        # Save results
        print_section("saving results")

        # Add band results to main results
        results["band_analysis"] = band_results

        # Save JSON
        results_path = OUTPUT_DIR / "correlation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if results_path.exists():
            size_kb = results_path.stat().st_size / 1024
            print(f"Saved: {results_path} ({size_kb:.1f} KB)")
        else:
            print(f"WARNING: Failed to save {results_path}")

        # Create README
        readme_path = OUTPUT_DIR / "README.txt"
        create_readme(results, band_results, OUTPUT_DIR)
        if readme_path.exists():
            print(f"Saved: {readme_path}")
        else:
            print(f"WARNING: Failed to save {readme_path}")

        # List all created files
        print("\n" + "=" * 80)
        print("  CREATED FILES")
        print("=" * 80)
        all_files = sorted(OUTPUT_DIR.glob("*"))
        for f in all_files:
            if f.is_file():
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  {f.name:<40} {size_str:>12}")

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print_section("ANALYSIS COMPLETE")
        print(f"Analyzed {len(df):,} tokens")
        print(f"Generated {len(list(OUTPUT_DIR.glob('*')))} output files")
        print(f"Location: {OUTPUT_DIR}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"\nKey Finding:")
        log_corr = results["correlations"]["log_frequency"]["pearson"]
        print(
            f"  Pearson r = {log_corr['coefficient']:.4f} ({log_corr['interpretation']})"
        )
        print(f"  p-value = {log_corr['p_value']:.2e}")
        print(f"\nNext: Review plots and README.txt in {OUTPUT_DIR}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nANALYSIS FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
