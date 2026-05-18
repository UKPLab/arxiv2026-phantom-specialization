"""
Graph-theoretic analysis of circuit structure using networkx.

Treats each circuit as a directed graph and computes topological
metrics: diameter, clustering, degree distributions, hub nodes,
connected components.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from .constants import MODELS, BANDS, DRAWS, MODEL_INFO


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def build_circuit_graph(circuit: Dict) -> nx.DiGraph:
    """
    Build a directed graph from circuit edges.

    Nodes represent components:
        'embed': embedding layer
        'A{layer}.{head}': attention head output
        'M{layer}': MLP output
        'R{layer}': residual stream at layer (destination only)

    Edge attributes: edge_category, layer_distance, src_type, dst_type.
    """
    G = nx.DiGraph()

    for edge in circuit.get("edges", []):
        # Build source node name
        if edge["src_type"] == "embed":
            src_node = "embed"
        elif edge["src_type"] == "attn":
            src_node = f"A{edge['src_layer']}.{edge['src_head']}"
        else:  # mlp
            src_node = f"M{edge['src_layer']}"

        # Build destination node name
        if edge["dst_type"] == "attn_in":
            dst_node = f"A{edge['dst_layer']}.{edge['dst_head']}"
        elif edge["dst_type"] == "mlp_in":
            dst_node = f"M{edge['dst_layer']}"
        else:  # resid_post
            dst_node = f"R{edge['dst_layer']}"

        # Add nodes with type attributes
        G.add_node(src_node, node_type=edge["src_type"], layer=edge["src_layer"])
        G.add_node(
            dst_node,
            node_type=edge["dst_type"].replace("_in", "").replace("_post", ""),
            layer=edge["dst_layer"],
        )

        # Add edge (or increment weight if parallel edges exist)
        if G.has_edge(src_node, dst_node):
            G[src_node][dst_node]["weight"] += 1
        else:
            G.add_edge(
                src_node,
                dst_node,
                weight=1,
                edge_category=edge["edge_category"],
                layer_distance=edge["layer_distance"],
            )

    return G


# =============================================================================
# GRAPH METRICS
# =============================================================================


def compute_graph_metrics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute graph-theoretic metrics.

    Returns dict with:
        n_nodes, n_edges, density,
        diameter (of largest weakly connected component),
        avg_path_length (of largest WCC),
        clustering_coefficient (on undirected projection),
        n_weakly_connected, largest_wcc_fraction,
        n_strongly_connected, largest_scc_fraction,
        avg_in_degree, avg_out_degree, max_in_degree, max_out_degree
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph) if n_nodes > 0 else 0.0

    # Weakly connected components
    wccs = list(nx.weakly_connected_components(graph))
    n_wcc = len(wccs)
    largest_wcc = max(wccs, key=len) if wccs else set()
    largest_wcc_frac = len(largest_wcc) / n_nodes if n_nodes > 0 else 0.0

    # Diameter and avg path length on largest WCC
    if len(largest_wcc) > 1:
        sub = graph.subgraph(largest_wcc)
        try:
            diameter = nx.diameter(sub.to_undirected())
        except nx.NetworkXError:
            diameter = -1
        try:
            avg_path = nx.average_shortest_path_length(sub.to_undirected())
        except nx.NetworkXError:
            avg_path = -1.0
    else:
        diameter = 0
        avg_path = 0.0

    # Strongly connected components
    sccs = list(nx.strongly_connected_components(graph))
    n_scc = len(sccs)
    largest_scc = max(sccs, key=len) if sccs else set()
    largest_scc_frac = len(largest_scc) / n_nodes if n_nodes > 0 else 0.0

    # Clustering coefficient on undirected projection
    undirected = graph.to_undirected()
    clustering = nx.average_clustering(undirected) if n_nodes > 0 else 0.0

    # Degree statistics
    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "diameter": diameter,
        "avg_path_length": avg_path,
        "clustering_coefficient": clustering,
        "n_weakly_connected": n_wcc,
        "largest_wcc_fraction": largest_wcc_frac,
        "n_strongly_connected": n_scc,
        "largest_scc_fraction": largest_scc_frac,
        "avg_in_degree": np.mean(in_degrees) if in_degrees else 0.0,
        "avg_out_degree": np.mean(out_degrees) if out_degrees else 0.0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
    }


# =============================================================================
# DEGREE AND HUB ANALYSIS
# =============================================================================


def compute_degree_stats(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Compute in-degree and out-degree for each node.

    Returns DataFrame: node, node_type, layer, in_degree, out_degree, total_degree
    """
    rows = []
    for node in graph.nodes():
        data = graph.nodes[node]
        rows.append(
            {
                "node": node,
                "node_type": data.get("node_type", "unknown"),
                "layer": data.get("layer", -1),
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "total_degree": graph.in_degree(node) + graph.out_degree(node),
            }
        )
    return pd.DataFrame(rows)


def compute_hub_nodes(graph: nx.DiGraph, top_k: int = 10) -> pd.DataFrame:
    """
    Identify top hub nodes by betweenness and degree centrality.

    Returns DataFrame: node, node_type, layer, centrality_type, score
    """
    rows = []

    # Betweenness centrality
    bc = nx.betweenness_centrality(graph)
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for node, score in top_bc:
        data = graph.nodes[node]
        rows.append(
            {
                "node": node,
                "node_type": data.get("node_type", "unknown"),
                "layer": data.get("layer", -1),
                "centrality_type": "betweenness",
                "score": score,
            }
        )

    # Degree centrality
    dc = nx.degree_centrality(graph)
    top_dc = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for node, score in top_dc:
        data = graph.nodes[node]
        rows.append(
            {
                "node": node,
                "node_type": data.get("node_type", "unknown"),
                "layer": data.get("layer", -1),
                "centrality_type": "degree",
                "score": score,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def compute_all_graph_metrics(
    circuits: Dict,
    models: List[str] = None,
    bands: List[str] = None,
    draws: List[str] = None,
) -> pd.DataFrame:
    """
    Compute graph metrics for all circuits.

    Returns DataFrame: one row per circuit with all graph metrics.
    """
    if models is None:
        models = MODELS
    if bands is None:
        bands = BANDS
    if draws is None:
        draws = DRAWS

    rows = []
    for c in circuits.values():
        if c["model"] not in models or c["band"] not in bands or c["draw"] not in draws:
            continue

        graph = build_circuit_graph(c)
        metrics = compute_graph_metrics(graph)
        metrics["circuit_id"] = c["circuit_id"]
        metrics["model"] = c["model"]
        metrics["band"] = c["band"]
        metrics["draw"] = c["draw"]
        rows.append(metrics)

    return pd.DataFrame(rows)


def classify_hub_universality(
    circuits: Dict,
    model: str,
    top_k: int = 10,
    bands: List[str] = None,
    draws: List[str] = None,
) -> pd.DataFrame:
    """
    For top hub nodes across all circuits of a model, check how many bands
    they appear as hubs in.

    Returns DataFrame: node, model, centrality_type, n_bands_as_hub,
                       bands_as_hub, is_universal_hub
    """
    if bands is None:
        bands = BANDS
    if draws is None:
        draws = DRAWS

    # Collect hub sets per (band, draw)
    hub_records = defaultdict(
        lambda: defaultdict(set)
    )  # {centrality_type: {node: set(bands)}}

    for c in circuits.values():
        if c["model"] != model or c["band"] not in bands or c["draw"] not in draws:
            continue

        graph = build_circuit_graph(c)
        hubs_df = compute_hub_nodes(graph, top_k=top_k)

        for _, row in hubs_df.iterrows():
            hub_records[row["centrality_type"]][row["node"]].add(c["band"])

    rows = []
    for ctype, node_bands in hub_records.items():
        for node, band_set in node_bands.items():
            rows.append(
                {
                    "node": node,
                    "model": model,
                    "centrality_type": ctype,
                    "n_bands_as_hub": len(band_set),
                    "bands_as_hub": ",".join(sorted(band_set)),
                    "is_universal_hub": len(band_set) == len(bands),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# UNIVERSAL CORE CONNECTIVITY (S-G4)
# =============================================================================


def compute_universal_core_connectivity(
    circuits: Dict,
    model: str,
    bands: List[str] = None,
    draws: List[str] = None,
) -> Dict[str, Any]:
    """
    Check connectivity of the universal edge core for a model.

    Builds a graph from only universal edges (present in all bands),
    then checks:
    - Does a path exist from 'embed' to the final residual node?
    - How many weakly connected components exist?
    - What fraction of the full circuit's nodes does the core cover?

    Returns dict with:
        model, path_exists, path_exists_any_draw,
        path_per_draw, n_components_mean, n_components_per_draw,
        n_nodes_core, n_nodes_full, node_coverage,
        n_edges_core_mean, n_edges_full_mean
    """
    from .edge_analysis import (
        compute_universal_edges_per_draw,
        get_edge_set,
        _get_circuit,
    )

    if bands is None:
        bands = BANDS
    if draws is None:
        draws = DRAWS

    n_layers = MODEL_INFO[model]["n_layers"]
    final_resid = f"R{n_layers - 1}"

    universal_per_draw = compute_universal_edges_per_draw(circuits, model, bands)

    path_per_draw = {}
    comp_per_draw = {}
    core_nodes_all = set()
    core_edges_total = 0
    full_nodes_all = set()
    full_edges_total = 0
    n_draws_checked = 0

    for draw in draws:
        universal_edges = universal_per_draw.get(draw, set())

        # Collect edge properties for universal edges from all band circuits
        all_edge_props_by_raw = {}
        full_edge_props = {}
        for band in bands:
            c = _get_circuit(circuits, model, band, draw)
            if c is None:
                continue
            for e in c.get("edges", []):
                if e["raw"] in universal_edges:
                    all_edge_props_by_raw[e["raw"]] = e
                full_edge_props[e["raw"]] = e

        if not all_edge_props_by_raw:
            continue

        n_draws_checked += 1

        # Build universal core graph
        synthetic_core = {"edges": list(all_edge_props_by_raw.values())}
        G_core = build_circuit_graph(synthetic_core)

        # Build full circuit graph (union of all bands)
        synthetic_full = {"edges": list(full_edge_props.values())}
        G_full = build_circuit_graph(synthetic_full)

        # Check path from embed to final resid
        has_path = False
        if "embed" in G_core and final_resid in G_core:
            has_path = nx.has_path(G_core, "embed", final_resid)
        path_per_draw[draw] = has_path

        # Weakly connected components
        wccs = list(nx.weakly_connected_components(G_core))
        comp_per_draw[draw] = len(wccs)

        core_nodes_all.update(G_core.nodes())
        core_edges_total += G_core.number_of_edges()
        full_nodes_all.update(G_full.nodes())
        full_edges_total += G_full.number_of_edges()

    return {
        "model": model,
        "path_exists": all(path_per_draw.values()) if path_per_draw else False,
        "path_exists_any_draw": any(path_per_draw.values()) if path_per_draw else False,
        "path_per_draw": path_per_draw,
        "n_components_mean": np.mean(list(comp_per_draw.values()))
        if comp_per_draw
        else 0,
        "n_components_per_draw": comp_per_draw,
        "n_nodes_core": len(core_nodes_all),
        "n_nodes_full": len(full_nodes_all),
        "node_coverage": len(core_nodes_all) / len(full_nodes_all)
        if full_nodes_all
        else 0,
        "n_edges_core_mean": core_edges_total / max(n_draws_checked, 1),
        "n_edges_full_mean": full_edges_total / max(n_draws_checked, 1),
    }
