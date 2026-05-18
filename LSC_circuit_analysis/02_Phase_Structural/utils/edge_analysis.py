"""
Edge set analysis for Phase Structural analysis.

Circuit comparison operations: Jaccard similarity, containment,
universal/band-specific edges, edge sharing spectrum, and deep
decomposition (component-level, layer-level, head-level).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

from .constants import BANDS, DRAWS, FREQUENCY_BANDS, FREQUENCY_RANK, MODEL_INFO


# =============================================================================
# EDGE SET OPERATIONS
# =============================================================================


def get_edge_set(circuit: Dict) -> Set[str]:
    """Get set of edge strings from a circuit structure dict."""
    return set(circuit.get("edge_list", []))


def compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity: |A n B| / |A u B|."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_containment(small_set: Set[str], large_set: Set[str]) -> float:
    """Containment ratio: |small n large| / |small|."""
    if len(small_set) == 0:
        return 1.0
    return len(small_set & large_set) / len(small_set)


def compute_overlap(set1: Set[str], set2: Set[str]) -> Dict[str, Any]:
    """Overlap statistics between two edge sets."""
    intersection = set1 & set2
    union = set1 | set2
    return {
        "intersection_size": len(intersection),
        "union_size": len(union),
        "only_in_1": len(set1 - set2),
        "only_in_2": len(set2 - set1),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "containment_1_in_2": len(intersection) / len(set1) if set1 else 1.0,
        "containment_2_in_1": len(intersection) / len(set2) if set2 else 1.0,
    }


# =============================================================================
# UNIVERSAL & BAND-SPECIFIC EDGES
# =============================================================================


def compute_universal_edges_per_draw(
    circuits: Dict[str, Dict],
    model: str,
    bands: List[str] = None,
) -> Dict[str, Set[str]]:
    """
    Compute universal edges per draw (present in ALL bands for that draw).

    Returns dict mapping draw -> set of universal edges.
    """
    if bands is None:
        bands = BANDS

    result = {}
    for draw in DRAWS:
        band_edge_sets = {}
        for band in bands:
            for c in circuits.values():
                if c["model"] == model and c["draw"] == draw and c["band"] == band:
                    band_edge_sets[band] = get_edge_set(c)
                    break

        if len(band_edge_sets) == len(bands):
            result[draw] = set.intersection(*band_edge_sets.values())
        else:
            result[draw] = set()

    return result


def compute_universal_edges_across_draws(
    circuits: Dict[str, Dict],
    model: str,
    bands: List[str] = None,
) -> Set[str]:
    """
    Compute universal edges across ALL draws (in all bands AND all draws).
    """
    per_draw = compute_universal_edges_per_draw(circuits, model, bands)
    if not per_draw:
        return set()
    return set.intersection(*per_draw.values())


def compute_band_specific_edges(
    circuits: Dict[str, Dict],
    model: str,
    draw: str,
    bands: List[str] = None,
) -> Dict[str, Set[str]]:
    """
    Compute band-specific edges (present in only one band).

    Returns dict mapping band -> set of band-specific edges.
    """
    if bands is None:
        bands = BANDS

    band_edge_sets = {}
    for band in bands:
        for c in circuits.values():
            if c["model"] == model and c["draw"] == draw and c["band"] == band:
                band_edge_sets[band] = get_edge_set(c)
                break

    if len(band_edge_sets) != len(bands):
        return {}

    result = {}
    for band in bands:
        other_edges = set.union(*[band_edge_sets[b] for b in bands if b != band])
        result[band] = band_edge_sets[band] - other_edges

    return result


# =============================================================================
# EDGE SHARING SPECTRUM
# =============================================================================


def compute_edge_sharing_spectrum(
    circuits: Dict[str, Dict],
    model: str,
    draw: str,
    bands: List[str] = None,
) -> Dict[int, int]:
    """
    Count how many bands each edge appears in.

    Returns dict mapping n_bands -> count of edges appearing in exactly n_bands.
    """
    if bands is None:
        bands = BANDS

    band_edge_sets = {}
    for band in bands:
        for c in circuits.values():
            if c["model"] == model and c["draw"] == draw and c["band"] == band:
                band_edge_sets[band] = get_edge_set(c)
                break

    if len(band_edge_sets) != len(bands):
        return {}

    # Count appearances for each edge
    all_edges = set.union(*band_edge_sets.values())
    spectrum = defaultdict(int)
    for edge in all_edges:
        n = sum(1 for b in bands if edge in band_edge_sets[b])
        spectrum[n] += 1

    return dict(sorted(spectrum.items()))


# =============================================================================
# JACCARD MATRICES
# =============================================================================


def compute_jaccard_matrix(
    circuits: Dict[str, Dict],
    model: str,
    exclude_universal: bool = False,
    bands: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Jaccard similarity matrix for all circuits of a given model.

    Matrix entries: Jaccard(circuit_i, circuit_j) for all band x draw combos.

    Returns (similarity_matrix, labels).
    """
    if bands is None:
        bands = BANDS

    # Collect relevant circuits sorted by band then draw
    relevant = []
    for c in circuits.values():
        if c["model"] == model and c["band"] in bands:
            relevant.append(c)

    band_order = {b: i for i, b in enumerate(bands)}
    draw_order = {d: i for i, d in enumerate(DRAWS)}
    relevant.sort(
        key=lambda c: (band_order.get(c["band"], 99), draw_order.get(c["draw"], 99))
    )

    n = len(relevant)
    matrix = np.zeros((n, n))
    labels = [f"{c['band']}_{c['draw']}" for c in relevant]

    # Get universal edges if excluding
    universal = set()
    if exclude_universal:
        universal = compute_universal_edges_across_draws(circuits, model, bands)

    for i in range(n):
        edges_i = get_edge_set(relevant[i]) - universal
        for j in range(n):
            edges_j = get_edge_set(relevant[j]) - universal
            matrix[i, j] = compute_jaccard(edges_i, edges_j)

    return matrix, labels


def compute_band_jaccard_summary(
    circuits: Dict[str, Dict],
    model: str,
    exclude_universal: bool = False,
    bands: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean Jaccard similarity between all band pairs.

    Returns nested dict: result[band1][band2] = mean_jaccard.
    """
    if bands is None:
        bands = BANDS

    universal = set()
    if exclude_universal:
        universal = compute_universal_edges_across_draws(circuits, model, bands)

    # Collect edges by band and draw
    band_draw_edges = defaultdict(dict)
    for c in circuits.values():
        if c["model"] == model and c["band"] in bands:
            edges = get_edge_set(c) - universal
            band_draw_edges[c["band"]][c["draw"]] = edges

    result = {}
    for band1 in bands:
        result[band1] = {}
        for band2 in bands:
            jaccards = []
            for draw1, edges1 in band_draw_edges.get(band1, {}).items():
                for draw2, edges2 in band_draw_edges.get(band2, {}).items():
                    if band1 == band2 and draw1 == draw2:
                        continue
                    jaccards.append(compute_jaccard(edges1, edges2))
            result[band1][band2] = np.mean(jaccards) if jaccards else 0.0

    return result


def compute_within_between_jaccard(
    circuits: Dict[str, Dict],
    model: str,
    bands: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Compute within-band and between-band Jaccard values.

    Returns dict with 'within' and 'between' lists of Jaccard values.
    """
    if bands is None:
        bands = BANDS

    band_draw_edges = defaultdict(dict)
    for c in circuits.values():
        if c["model"] == model and c["band"] in bands:
            band_draw_edges[c["band"]][c["draw"]] = get_edge_set(c)

    within = []
    between = []

    band_list = [b for b in bands if b in band_draw_edges]
    for i, band1 in enumerate(band_list):
        for j, band2 in enumerate(band_list):
            for draw1, edges1 in band_draw_edges[band1].items():
                for draw2, edges2 in band_draw_edges[band2].items():
                    if band1 == band2 and draw1 == draw2:
                        continue
                    jac = compute_jaccard(edges1, edges2)
                    if band1 == band2:
                        within.append(jac)
                    elif i < j:
                        between.append(jac)

    return {"within": within, "between": between}


# =============================================================================
# EDGE FILTERING (for deep decomposition)
# =============================================================================


def build_edge_index(circuit: Dict) -> Dict[str, Dict]:
    """
    Map raw edge string -> edge properties dict.

    Built once per circuit for fast property-based filtering of edge_list sets.
    """
    index = {}
    for edge in circuit.get("edges", []):
        index[edge["raw"]] = edge
    return index


def get_filtered_edge_set(
    circuit: Dict,
    index: Dict[str, Dict],
    component: str = None,
    layer: int = None,
    src_type: str = None,
    edge_cat: str = None,
    is_input: bool = None,
    is_output: bool = None,
) -> Set[str]:
    """
    Filter edge_list by properties using the pre-built index.

    Args:
        component: Filter by destination component: 'attn'|'mlp'|'resid'
                   Maps to dst_type: attn_in, mlp_in, resid_post
        layer: Filter by destination layer
        src_type: Filter by source type: 'embed'|'attn'|'mlp'
        edge_cat: Filter by edge_category string (e.g., 'attn_to_mlp')
        is_input: Filter by is_input flag
        is_output: Filter by is_output flag

    Returns:
        Set of raw edge strings matching all filters.
    """
    dst_type_map = {"attn": "attn_in", "mlp": "mlp_in", "resid": "resid_post"}

    result = set()
    for raw in circuit.get("edge_list", []):
        props = index.get(raw)
        if props is None:
            continue
        if component is not None and props["dst_type"] != dst_type_map.get(
            component, component
        ):
            continue
        if layer is not None and props["dst_layer"] != layer:
            continue
        if src_type is not None and props["src_type"] != src_type:
            continue
        if edge_cat is not None and props["edge_category"] != edge_cat:
            continue
        if is_input is not None and props["is_input"] != is_input:
            continue
        if is_output is not None and props["is_output"] != is_output:
            continue
        result.add(raw)
    return result


def _get_circuit(circuits: Dict, model: str, band: str, draw: str) -> Optional[Dict]:
    """Get a specific circuit from the circuits dict."""
    for c in circuits.values():
        if c["model"] == model and c["band"] == band and c["draw"] == draw:
            return c
    return None


# =============================================================================
# COMPONENT-LEVEL JACCARD (Section 1a)
# =============================================================================


def compute_component_jaccard(
    circuits: Dict,
    model: str,
    component: str,
    bands: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Compute within-band and between-band Jaccard for a specific component type.

    Args:
        component: 'attn'|'mlp'|'resid' (filters by destination type)

    Returns:
        {'within': [float, ...], 'between': [float, ...]}
    """
    if bands is None:
        bands = BANDS

    # Build indices and filtered edge sets
    band_draw_edges = defaultdict(dict)
    for c in circuits.values():
        if c["model"] == model and c["band"] in bands:
            idx = build_edge_index(c)
            filtered = get_filtered_edge_set(c, idx, component=component)
            band_draw_edges[c["band"]][c["draw"]] = filtered

    within = []
    between = []
    band_list = [b for b in bands if b in band_draw_edges]

    for i, band1 in enumerate(band_list):
        for j, band2 in enumerate(band_list):
            for draw1, edges1 in band_draw_edges[band1].items():
                for draw2, edges2 in band_draw_edges[band2].items():
                    if band1 == band2 and draw1 == draw2:
                        continue
                    jac = compute_jaccard(edges1, edges2)
                    if band1 == band2:
                        within.append(jac)
                    elif i < j:
                        between.append(jac)

    return {"within": within, "between": between}


# =============================================================================
# EDGE SHARING WITH PROPERTIES (Section 1b, 2a)
# =============================================================================


def compute_edge_sharing_with_properties(
    circuits: Dict,
    model: str,
    draw: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    For each edge in the union, determine sharing level and edge properties.

    Returns DataFrame with columns:
        raw, sharing_level, dst_type, dst_layer, src_type, src_layer,
        edge_category, layer_distance, is_skip, is_input, is_output,
        bands_present (comma-separated)
    """
    if bands is None:
        bands = BANDS

    # Collect edge sets and indices
    band_edges = {}
    band_indices = {}
    for band in bands:
        c = _get_circuit(circuits, model, band, draw)
        if c is None:
            continue
        band_edges[band] = get_edge_set(c)
        band_indices[band] = build_edge_index(c)

    if len(band_edges) != len(bands):
        return pd.DataFrame()

    all_edges = set.union(*band_edges.values())
    rows = []
    for raw in all_edges:
        present_in = [b for b in bands if raw in band_edges[b]]
        sharing_level = len(present_in)

        # Get properties from the first band that has this edge
        props = None
        for b in present_in:
            props = band_indices[b].get(raw)
            if props is not None:
                break

        if props is None:
            continue

        rows.append(
            {
                "raw": raw,
                "sharing_level": sharing_level,
                "dst_type": props["dst_type"],
                "dst_layer": props["dst_layer"],
                "src_type": props["src_type"],
                "src_layer": props.get("src_layer", -1),
                "edge_category": props["edge_category"],
                "layer_distance": props["layer_distance"],
                "is_skip": props["is_skip"],
                "is_input": props["is_input"],
                "is_output": props["is_output"],
                "bands_present": ",".join(present_in),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# BAND AFFINITY MATRIX (Section 2b)
# =============================================================================


def compute_band_affinity_matrix(
    circuits: Dict,
    model: str,
    draw: str,
    exclude_universal: bool = True,
    bands: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Jaccard similarity between band pairs on non-universal edges.

    Returns (n_bands x n_bands matrix, band_labels).
    """
    if bands is None:
        bands = BANDS

    band_edges = {}
    for band in bands:
        c = _get_circuit(circuits, model, band, draw)
        if c is not None:
            band_edges[band] = get_edge_set(c)

    if len(band_edges) != len(bands):
        return np.zeros((len(bands), len(bands))), bands

    universal = set()
    if exclude_universal:
        universal = set.intersection(*band_edges.values())

    n = len(bands)
    matrix = np.zeros((n, n))
    for i in range(n):
        edges_i = band_edges[bands[i]] - universal
        for j in range(n):
            edges_j = band_edges[bands[j]] - universal
            matrix[i, j] = compute_jaccard(edges_i, edges_j)

    return matrix, bands


def compute_band_affinity_summary(
    circuits: Dict,
    model: str,
    exclude_universal: bool = True,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    Average affinity matrix across draws. Returns DataFrame with band pairs.
    """
    if bands is None:
        bands = BANDS

    matrices = []
    for draw in DRAWS:
        mat, _ = compute_band_affinity_matrix(
            circuits, model, draw, exclude_universal, bands
        )
        matrices.append(mat)

    avg_matrix = np.mean(matrices, axis=0)

    rows = []
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            rank_i = FREQUENCY_RANK.get(bands[i])
            rank_j = FREQUENCY_RANK.get(bands[j])
            freq_dist = (
                abs(rank_i - rank_j)
                if (rank_i is not None and rank_j is not None)
                else None
            )
            rows.append(
                {
                    "model": model,
                    "band_1": bands[i],
                    "band_2": bands[j],
                    "affinity": avg_matrix[i, j],
                    "freq_distance": freq_dist,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# LAYER-LEVEL DECOMPOSITION (Section 3a, 3b)
# =============================================================================


def compute_layer_band_sensitivity(
    circuits: Dict,
    model: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    For each destination layer, compute mean pairwise Jaccard between bands.

    Low Jaccard = high sensitivity (bands use different edges at this layer).
    High Jaccard = low sensitivity (bands share edges at this layer).

    Returns DataFrame: model, layer, mean_jaccard, std_jaccard, n_pairs
    """
    if bands is None:
        bands = BANDS

    n_layers = MODEL_INFO[model]["n_layers"]
    n_heads = MODEL_INFO[model]["n_heads"]

    # Build filtered edge sets by layer
    # For each (band, draw, layer): set of raw edge strings at that layer
    layer_edges = defaultdict(lambda: defaultdict(dict))
    for c in circuits.values():
        if c["model"] == model and c["band"] in bands:
            idx = build_edge_index(c)
            for layer in range(n_layers):
                filtered = get_filtered_edge_set(c, idx, layer=layer)
                layer_edges[layer][c["band"]][c["draw"]] = filtered

    rows = []
    for layer in range(n_layers):
        jaccards = []
        band_list = [b for b in bands if b in layer_edges[layer]]
        for i, b1 in enumerate(band_list):
            for j, b2 in enumerate(band_list):
                if i >= j:
                    continue
                for d1, e1 in layer_edges[layer][b1].items():
                    for d2, e2 in layer_edges[layer][b2].items():
                        jac = compute_jaccard(e1, e2)
                        jaccards.append(jac)

        rows.append(
            {
                "model": model,
                "layer": layer,
                "mean_jaccard": np.mean(jaccards) if jaccards else 0.0,
                "std_jaccard": np.std(jaccards) if jaccards else 0.0,
                "n_pairs": len(jaccards),
            }
        )

    return pd.DataFrame(rows)


def compute_per_layer_universal_fraction(
    circuits: Dict,
    model: str,
    draw: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    What fraction of edges at each destination layer are universal?

    Returns DataFrame: model, draw, layer, total_edges, universal_count, universal_fraction
    """
    if bands is None:
        bands = BANDS

    n_layers = MODEL_INFO[model]["n_layers"]

    # Get edge sets by layer and band
    layer_band_edges = defaultdict(dict)
    for band in bands:
        c = _get_circuit(circuits, model, band, draw)
        if c is None:
            continue
        idx = build_edge_index(c)
        for layer in range(n_layers):
            layer_band_edges[layer][band] = get_filtered_edge_set(c, idx, layer=layer)

    rows = []
    for layer in range(n_layers):
        band_sets = layer_band_edges[layer]
        if len(band_sets) != len(bands):
            continue

        all_edges = set.union(*band_sets.values())
        universal = set.intersection(*band_sets.values())

        rows.append(
            {
                "model": model,
                "draw": draw,
                "layer": layer,
                "total_edges": len(all_edges),
                "universal_count": len(universal),
                "universal_fraction": len(universal) / len(all_edges)
                if all_edges
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# HEAD-LEVEL DECOMPOSITION (Section 4a, 4b)
# =============================================================================


def compute_head_band_presence(
    circuits: Dict,
    model: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    For each attention head, count how many bands it appears in (has >=1 incoming edge).

    Averaged across draws. Returns DataFrame per (model, draw, head).
    """
    if bands is None:
        bands = BANDS

    n_layers = MODEL_INFO[model]["n_layers"]
    n_heads = MODEL_INFO[model]["n_heads"]

    rows = []
    for draw in DRAWS:
        # For each band, get set of active heads
        band_active_heads = {}
        for band in bands:
            c = _get_circuit(circuits, model, band, draw)
            if c is None:
                continue
            active = set()
            for edge in c.get("edges", []):
                if edge["dst_type"] == "attn_in" and edge.get("dst_head") is not None:
                    active.add(f"A{edge['dst_layer']}.{edge['dst_head']}")
            band_active_heads[band] = active

        # For each head, count bands
        for layer in range(n_layers):
            for head in range(n_heads):
                head_key = f"A{layer}.{head}"
                present_bands = [
                    b for b in bands if head_key in band_active_heads.get(b, set())
                ]
                rows.append(
                    {
                        "model": model,
                        "draw": draw,
                        "head": head_key,
                        "layer": layer,
                        "head_idx": head,
                        "n_bands": len(present_bands),
                        "bands_present": ",".join(present_bands),
                        "universality_score": len(present_bands) / len(bands),
                    }
                )

    return pd.DataFrame(rows)


def compute_head_band_entropy(
    circuits: Dict,
    model: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    For each head, compute entropy of its edge-count distribution across bands.

    High normalized entropy = uniform across bands (universal head).
    Low normalized entropy = concentrated in specific bands (discriminative head).

    Averaged across draws.
    """
    if bands is None:
        bands = BANDS

    n_layers = MODEL_INFO[model]["n_layers"]
    n_heads = MODEL_INFO[model]["n_heads"]

    rows = []
    for draw in DRAWS:
        # For each band, get edge counts per head
        band_head_counts = defaultdict(lambda: defaultdict(int))
        for band in bands:
            c = _get_circuit(circuits, model, band, draw)
            if c is None:
                continue
            by_head = c.get("by_head", {})
            for head_key, count in by_head.items():
                band_head_counts[head_key][band] = count

        for layer in range(n_layers):
            for head in range(n_heads):
                head_key = f"A{layer}.{head}"
                counts = [band_head_counts[head_key].get(b, 0) for b in bands]
                total = sum(counts)

                if total == 0:
                    entropy = 0.0
                    norm_entropy = 0.0
                    dominant = None
                else:
                    probs = np.array(counts, dtype=float) / total
                    probs = probs[probs > 0]
                    entropy = -np.sum(probs * np.log2(probs))
                    max_entropy = np.log2(len(bands))
                    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                    dominant = bands[np.argmax(counts)]

                rows.append(
                    {
                        "model": model,
                        "draw": draw,
                        "head": head_key,
                        "layer": layer,
                        "head_idx": head,
                        "entropy": entropy,
                        "max_entropy": np.log2(len(bands)),
                        "normalized_entropy": norm_entropy,
                        "dominant_band": dominant,
                        "total_edges": total,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# DIRECTED CONTAINMENT (Section 2d: tests H5 subset hypothesis)
# =============================================================================


def compute_directed_containment(
    circuits: Dict,
    model: str,
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    For each ordered band pair (A, B), compute |A  and  B| / |B|.

    This measures how much of B's edges are contained in A.
    If H5 (subset hypothesis) holds: containment(low, high) > containment(high, low)
    meaning low-freq circuits contain most of high-freq circuit edges.
    """
    if bands is None:
        bands = BANDS

    rows = []
    for draw in DRAWS:
        band_edges = {}
        for band in bands:
            c = _get_circuit(circuits, model, band, draw)
            if c is not None:
                band_edges[band] = get_edge_set(c)

        for b1 in bands:
            for b2 in bands:
                if b1 == b2:
                    continue
                if b1 not in band_edges or b2 not in band_edges:
                    continue
                # How much of b2 is in b1?
                containment = compute_containment(band_edges[b2], band_edges[b1])
                rows.append(
                    {
                        "model": model,
                        "draw": draw,
                        "source_band": b1,
                        "target_band": b2,
                        "containment": containment,
                        "source_size": len(band_edges[b1]),
                        "target_size": len(band_edges[b2]),
                        "intersection": len(band_edges[b1] & band_edges[b2]),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# DRAW STABILITY (S-G3)
# =============================================================================


def compute_edge_draw_stability(
    circuits: Dict,
    model: str,
    band: str,
) -> pd.DataFrame:
    """
    For a given model and band, compute draw stability of every edge.

    Loads the 3 draws' edge sets, and for each edge in their union,
    records how many draws it appears in (1, 2, or 3).

    Returns DataFrame with columns:
        raw, model, band, n_draws, draws_present
    """
    edge_draw_presence = defaultdict(set)

    for draw in DRAWS:
        c = _get_circuit(circuits, model, band, draw)
        if c is None:
            continue
        for raw_edge in get_edge_set(c):
            edge_draw_presence[raw_edge].add(draw)

    rows = []
    for raw_edge, draws_present in edge_draw_presence.items():
        rows.append(
            {
                "raw": raw_edge,
                "model": model,
                "band": band,
                "n_draws": len(draws_present),
                "draws_present": ",".join(sorted(draws_present)),
            }
        )

    return pd.DataFrame(rows)


def compute_draw_stability_vs_sharing(
    circuits: Dict,
    model: str,
    reference_draw: str = "draw_1",
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    Cross-tabulate draw stability with band sharing level for a model.

    For each edge:
    - sharing_level: how many bands it appears in (for the reference draw)
    - n_draws: how many draws it appears in (for its band)

    Edges appearing only in non-reference draws get sharing_level=NaN.

    Returns DataFrame with columns:
        raw, model, band, n_draws, sharing_level
    """
    if bands is None:
        bands = BANDS

    # Step 1: Get sharing level for each edge (using reference draw)
    df_sharing = compute_edge_sharing_with_properties(
        circuits, model, reference_draw, bands
    )
    if df_sharing.empty:
        return pd.DataFrame()

    # Build edge -> sharing_level lookup
    sharing_lookup = dict(zip(df_sharing["raw"], df_sharing["sharing_level"]))

    # Step 2: For each band, compute draw stability
    rows = []
    for band in bands:
        df_stab = compute_edge_draw_stability(circuits, model, band)
        if df_stab.empty:
            continue
        for _, row in df_stab.iterrows():
            sl = sharing_lookup.get(row["raw"])
            rows.append(
                {
                    "raw": row["raw"],
                    "model": model,
                    "band": band,
                    "n_draws": row["n_draws"],
                    "sharing_level": sl if sl is not None else np.nan,
                }
            )

    return pd.DataFrame(rows)
