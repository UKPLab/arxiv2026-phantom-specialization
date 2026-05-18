"""
Utils package for Phase Structural analysis.

Provides constants, extraction, data loading, edge analysis,
graph analysis, and plotting.
"""

from .constants import (
    # Phase 1 re-exports
    MODELS,
    MODEL_DIR_NAMES,
    MODEL_LAYERS,
    MODEL_HEADS,
    MODEL_CAPACITY,
    MODEL_TOTAL_EDGES,
    MODEL_THRESHOLDS,
    BANDS,
    FREQUENCY_BANDS,
    CONTROL_BAND,
    FREQUENCY_RANK,
    BAND_NAMES,
    LOW_FREQ_BANDS,
    HIGH_FREQ_BANDS,
    ADJACENT_PAIRS,
    CROSS_SPECTRUM_PAIRS,
    DRAWS,
    N_DRAWS,
    BAND_COLORS,
    MODEL_COLORS,
    ALPHA,
    N_BOOTSTRAP,
    RANDOM_SEED,
    # Phase 2 specific
    PHASE2_BASE,
    CIRCUITS_BASE,
    OUTPUT_DIR,
    EXTRACTION_DIR,
    ANALYSIS_DIR,
    VIZ_DIR,
    ALL_CIRCUITS_JSON,
    CIRCUITS_SUMMARY_CSV,
    MODEL_INFO,
    SOURCE_TYPES,
    DESTINATION_TYPES,
    COMPONENT_TYPES,
    EDGE_CATEGORIES,
    COMPONENT_COLORS,
    EDGE_CATEGORY_COLORS,
    get_output_dirs,
)

from .data_loading import (
    load_extracted_data,
    build_structure_df,
    load_functional_data,
)

from .edge_analysis import (
    # Original
    get_edge_set,
    compute_jaccard,
    compute_containment,
    compute_overlap,
    compute_universal_edges_per_draw,
    compute_universal_edges_across_draws,
    compute_band_specific_edges,
    compute_edge_sharing_spectrum,
    compute_jaccard_matrix,
    compute_band_jaccard_summary,
    compute_within_between_jaccard,
    # Deep decomposition (new)
    build_edge_index,
    get_filtered_edge_set,
    compute_component_jaccard,
    compute_edge_sharing_with_properties,
    compute_band_affinity_matrix,
    compute_band_affinity_summary,
    compute_layer_band_sensitivity,
    compute_per_layer_universal_fraction,
    compute_head_band_presence,
    compute_head_band_entropy,
    compute_directed_containment,
    # Draw stability (S-G3)
    compute_edge_draw_stability,
    compute_draw_stability_vs_sharing,
)

from .graph_analysis import (
    build_circuit_graph,
    compute_graph_metrics,
    compute_degree_stats,
    compute_hub_nodes,
    compute_all_graph_metrics,
    classify_hub_universality,
    # Universal core connectivity (S-G4)
    compute_universal_core_connectivity,
)

from .plotting import (
    setup_plotting,
    save_figure,
    plot_layer_flow_heatmap,
    plot_head_participation_heatmap,
    plot_jaccard_heatmap,
    plot_band_jaccard_heatmap,
    # Deep decomposition (new)
    plot_component_jaccard_comparison,
    plot_layer_sensitivity_profile,
    plot_edge_sharing_profile,
    plot_band_affinity_heatmap,
    plot_head_universality_map,
    plot_graph_metrics_comparison,
)
