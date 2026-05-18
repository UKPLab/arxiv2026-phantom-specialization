"""
Edge parsing and structure extraction for Phase Structural analysis.

Contains the core logic for parsing AutoCircuit/TransformerLens edge
representations and extracting structural information from
prune_scores.pkl files.
"""

import re
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from .constants import (
    CIRCUITS_BASE,
    MODEL_DIR_NAMES,
    MODEL_INFO,
    MODEL_THRESHOLDS,
)


# =============================================================================
# PARSED EDGE REPRESENTATION
# =============================================================================


@dataclass
class ParsedEdge:
    """
    Fully parsed edge with source and destination information.

    Source types: 'embed' (layer=-1), 'attn' (with head), 'mlp'
    Destination types: 'attn_in' (with head), 'mlp_in', 'resid_post'
    """

    src_type: str
    src_layer: int
    src_head: Optional[int]
    dst_type: str
    dst_layer: int
    dst_head: Optional[int]
    raw: str = ""

    @property
    def is_input_edge(self) -> bool:
        return self.src_type == "embed"

    @property
    def is_output_edge(self) -> bool:
        return self.dst_type == "resid_post"

    @property
    def layer_distance(self) -> int:
        if self.src_layer < 0:
            return self.dst_layer + 1
        return self.dst_layer - self.src_layer

    @property
    def is_skip_connection(self) -> bool:
        return self.layer_distance > 1

    @property
    def is_local(self) -> bool:
        return self.src_layer >= 0 and self.src_layer == self.dst_layer

    @property
    def edge_category(self) -> str:
        src = "attn" if self.src_type == "attn" else self.src_type
        dst = (
            "attn"
            if self.dst_type == "attn_in"
            else ("mlp" if self.dst_type == "mlp_in" else "resid")
        )
        return f"{src}_to_{dst}"

    @property
    def src_key(self) -> str:
        if self.src_type == "embed":
            return "embed"
        elif self.src_type == "attn":
            return f"A{self.src_layer}.{self.src_head}"
        else:
            return f"M{self.src_layer}"

    @property
    def dst_key(self) -> str:
        if self.dst_type == "attn_in":
            return f"A{self.dst_layer}.{self.dst_head}"
        elif self.dst_type == "mlp_in":
            return f"M{self.dst_layer}"
        else:
            return f"R{self.dst_layer}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_type": self.src_type,
            "src_layer": self.src_layer,
            "src_head": self.src_head,
            "dst_type": self.dst_type,
            "dst_layer": self.dst_layer,
            "dst_head": self.dst_head,
            "raw": self.raw,
            "is_input": self.is_input_edge,
            "is_output": self.is_output_edge,
            "is_skip": self.is_skip_connection,
            "layer_distance": self.layer_distance,
            "edge_category": self.edge_category,
        }


# =============================================================================
# EDGE PARSER
# =============================================================================


class EdgeParser:
    """
    Parse AutoCircuit/TransformerLens edge names to structural information.

    Prune scores keys are DESTINATION hooks. Tensor indices encode SOURCES.

    Destination hooks:
    - blocks.{layer}.hook_attn_in  -> shape [n_heads, n_src]
    - blocks.{layer}.hook_mlp_in   -> shape [n_src]
    - blocks.{layer}.hook_resid_post -> shape [n_src]

    Source index encoding (for model with n_heads per layer):
    - 0 = embedding
    - 1 to n_heads = layer 0 attention heads
    - n_heads + 1 = layer 0 MLP
    - n_heads + 2 to 2*n_heads + 1 = layer 1 attention heads
    - 2*n_heads + 2 = layer 1 MLP
    - etc.
    """

    ATTN_IN_PATTERN = re.compile(r"^blocks\.(\d+)\.hook_attn_in$")
    MLP_IN_PATTERN = re.compile(r"^blocks\.(\d+)\.hook_mlp_in$")
    RESID_POST_PATTERN = re.compile(r"^blocks\.(\d+)\.hook_resid_post$")
    RESID_PRE_PATTERN = re.compile(r"^blocks\.(\d+)\.hook_resid_pre$")
    EMBED_PATTERN = re.compile(r"^hook_(pos_)?embed$")
    EDGE_WITH_INDEX_PATTERN = re.compile(r"^(.+?)\[([^\]]+)\]$")

    @classmethod
    def decode_src_idx(cls, src_idx: int, n_heads: int) -> Dict[str, Any]:
        """Decode source index to component info."""
        if src_idx == 0:
            return {"type": "embed", "layer": -1, "head": None}

        components_per_layer = n_heads + 1
        adjusted = src_idx - 1
        layer = adjusted // components_per_layer
        position = adjusted % components_per_layer

        if position < n_heads:
            return {"type": "attn", "layer": layer, "head": position}
        else:
            return {"type": "mlp", "layer": layer, "head": None}

    @classmethod
    def parse_destination(cls, hook_name: str) -> Dict[str, Any]:
        """Parse destination hook name."""
        match = cls.ATTN_IN_PATTERN.match(hook_name)
        if match:
            return {"type": "attn_in", "layer": int(match.group(1))}

        match = cls.MLP_IN_PATTERN.match(hook_name)
        if match:
            return {"type": "mlp_in", "layer": int(match.group(1))}

        match = cls.RESID_POST_PATTERN.match(hook_name)
        if match:
            return {"type": "resid_post", "layer": int(match.group(1))}

        match = cls.RESID_PRE_PATTERN.match(hook_name)
        if match:
            return {"type": "resid_pre", "layer": int(match.group(1))}

        if cls.EMBED_PATTERN.match(hook_name):
            return {"type": "embed", "layer": -1}

        return {"type": "unknown", "layer": None}

    @classmethod
    def parse_edge(
        cls, hook_name: str, head_idx: Optional[int], src_idx: int, n_heads: int
    ) -> ParsedEdge:
        """Parse a single edge from prune_scores tensor."""
        src = cls.decode_src_idx(src_idx, n_heads)
        dst = cls.parse_destination(hook_name)

        if head_idx is not None:
            raw = f"{hook_name}[{head_idx},{src_idx}]"
        else:
            raw = f"{hook_name}[{src_idx}]"

        return ParsedEdge(
            src_type=src["type"],
            src_layer=src["layer"],
            src_head=src["head"],
            dst_type=dst["type"],
            dst_layer=dst["layer"],
            dst_head=head_idx,
            raw=raw,
        )

    @classmethod
    def parse_edge_string(cls, edge_str: str, n_heads: int) -> Optional[ParsedEdge]:
        """Parse edge string like 'blocks.2.hook_attn_in[3,15]'."""
        match = cls.EDGE_WITH_INDEX_PATTERN.match(edge_str)
        if not match:
            return None

        hook_name = match.group(1)
        indices = [int(x.strip()) for x in match.group(2).split(",")]
        dst = cls.parse_destination(hook_name)

        if dst["type"] == "attn_in" and len(indices) >= 2:
            head_idx, src_idx = indices[0], indices[1]
        elif len(indices) >= 1:
            head_idx, src_idx = None, indices[0]
        else:
            return None

        return cls.parse_edge(hook_name, head_idx, src_idx, n_heads)


# =============================================================================
# STRUCTURE EXTRACTION
# =============================================================================


def extract_circuit_structure(
    prune_scores: Dict,
    model_name: str,
    band: str,
    draw: str,
) -> Dict[str, Any]:
    """
    Extract structural information from prune_scores.

    Single pass over all edges, computing:
    - Parsed edges and raw edge list (for Jaccard)
    - Layer flow matrix (src_layer -> dst_layer)
    - Component flow matrix (src_type -> dst_type)
    - Head connectivity (incoming/outgoing/head-to-head)
    - Edge categories (skip, local, forward, input, output)
    - Per-layer and per-head statistics

    Args:
        prune_scores: Dict mapping hook names to score tensors (inf = in circuit)
        model_name: Model name (e.g., 'pythia-70m')
        band: Frequency band
        draw: Draw identifier (e.g., 'draw_1')

    Returns:
        Dict with all structural information
    """
    import torch as t

    info = MODEL_INFO[model_name]
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]

    # Initialize aggregators
    layer_flow = defaultdict(lambda: defaultdict(int))
    component_flow = defaultdict(lambda: defaultdict(int))
    comp_src_totals = defaultdict(int)
    comp_dst_totals = defaultdict(int)

    head_incoming = defaultdict(int)
    head_outgoing = defaultdict(int)
    head_to_head = defaultdict(lambda: defaultdict(int))

    by_layer = defaultdict(lambda: {"total": 0, "attn": 0, "mlp": 0, "resid": 0})
    by_head = defaultdict(int)
    by_component_type = defaultdict(int)

    all_edges = []
    edge_list = []

    n_skip = 0
    n_input = 0
    n_output = 0
    n_local = 0
    n_forward = 0
    total_possible = 0

    for hook_name, scores in prune_scores.items():
        total_possible += scores.numel()

        in_circuit = t.isinf(scores)
        if not in_circuit.any():
            continue

        if scores.dim() == 1:
            indices = t.where(in_circuit)[0].tolist()
            for src_idx in indices:
                edge = EdgeParser.parse_edge(hook_name, None, src_idx, n_heads)
                _accumulate_edge(
                    edge,
                    n_layers,
                    all_edges,
                    edge_list,
                    layer_flow,
                    component_flow,
                    comp_src_totals,
                    comp_dst_totals,
                    head_incoming,
                    head_outgoing,
                    head_to_head,
                    by_layer,
                    by_head,
                    by_component_type,
                )
                n_skip += edge.is_skip_connection
                n_input += edge.is_input_edge
                n_output += (
                    edge.dst_type == "resid_post" and edge.dst_layer == n_layers - 1
                )
                n_local += edge.is_local
                n_forward += edge.layer_distance == 1

        elif scores.dim() == 2:
            indices = t.where(in_circuit)
            for head_idx, src_idx in zip(indices[0].tolist(), indices[1].tolist()):
                edge = EdgeParser.parse_edge(hook_name, head_idx, src_idx, n_heads)
                _accumulate_edge(
                    edge,
                    n_layers,
                    all_edges,
                    edge_list,
                    layer_flow,
                    component_flow,
                    comp_src_totals,
                    comp_dst_totals,
                    head_incoming,
                    head_outgoing,
                    head_to_head,
                    by_layer,
                    by_head,
                    by_component_type,
                )
                n_skip += edge.is_skip_connection
                n_input += edge.is_input_edge
                n_output += (
                    edge.dst_type == "resid_post" and edge.dst_layer == n_layers - 1
                )
                n_local += edge.is_local
                n_forward += edge.layer_distance == 1

    total_edges = len(all_edges)
    edge_fraction = total_edges / total_possible if total_possible > 0 else 0

    # Count active heads
    active_heads = set()
    for edge in all_edges:
        if edge.dst_type == "attn_in" and edge.dst_head is not None:
            active_heads.add(f"A{edge.dst_layer}.{edge.dst_head}")

    total_heads = n_layers * n_heads
    threshold = MODEL_THRESHOLDS[model_name]
    model_short = model_name.replace("-", "_")
    circuit_id = f"{model_short}_{band}_{draw}"

    return {
        "circuit_id": circuit_id,
        "model": model_name,
        "band": band,
        "draw": draw,
        "threshold": threshold,
        "total_edges": total_edges,
        "total_possible_edges": total_possible,
        "edge_fraction": edge_fraction,
        "edges": [e.to_dict() for e in all_edges],
        "edge_list": sorted(edge_list),
        "layer_flow": {
            "flow": {str(k): dict(v) for k, v in layer_flow.items()},
            "total_edges": total_edges,
            "input_edges": n_input,
            "output_edges": n_output,
            "skip_edges": n_skip,
            "local_edges": n_local,
            "forward_edges": n_forward,
        },
        "component_flow": {
            "flow": {k: dict(v) for k, v in component_flow.items()},
            "by_src_type": dict(comp_src_totals),
            "by_dst_type": dict(comp_dst_totals),
        },
        "head_connectivity": {
            "incoming": dict(head_incoming),
            "outgoing": dict(head_outgoing),
            "head_to_head": {k: dict(v) for k, v in head_to_head.items()},
            "active_heads": len(active_heads),
            "total_heads": total_heads,
        },
        "by_layer": {str(k): dict(v) for k, v in sorted(by_layer.items())},
        "by_head": {k: v for k, v in sorted(by_head.items(), key=_head_sort_key)},
        "by_component_type": dict(by_component_type),
        "n_skip_connections": n_skip,
        "n_input_edges": n_input,
        "n_output_edges": n_output,
        "n_local_edges": n_local,
        "n_forward_edges": n_forward,
        "extraction_timestamp": datetime.now().isoformat(),
        "status": "success",
    }


def _accumulate_edge(
    edge,
    n_layers,
    all_edges,
    edge_list,
    layer_flow,
    component_flow,
    comp_src_totals,
    comp_dst_totals,
    head_incoming,
    head_outgoing,
    head_to_head,
    by_layer,
    by_head,
    by_component_type,
):
    """Process a single edge and update all aggregators."""
    all_edges.append(edge)
    edge_list.append(edge.raw)

    # Layer flow
    layer_flow[edge.src_layer][edge.dst_layer] += 1

    # Component flow
    dst_comp = (
        "attn"
        if edge.dst_type == "attn_in"
        else ("mlp" if edge.dst_type == "mlp_in" else "resid")
    )
    component_flow[edge.src_type][dst_comp] += 1
    comp_src_totals[edge.src_type] += 1
    comp_dst_totals[dst_comp] += 1

    # Head connectivity
    if edge.src_type == "attn":
        head_outgoing[edge.src_key] += 1
    if edge.dst_type == "attn_in":
        head_incoming[edge.dst_key] += 1
    if edge.src_type == "attn" and edge.dst_type == "attn_in":
        head_to_head[edge.src_key][edge.dst_key] += 1

    # By layer
    if edge.dst_layer is not None and edge.dst_layer >= 0:
        by_layer[edge.dst_layer]["total"] += 1
        if edge.dst_type == "attn_in":
            by_layer[edge.dst_layer]["attn"] += 1
        elif edge.dst_type == "mlp_in":
            by_layer[edge.dst_layer]["mlp"] += 1
        elif edge.dst_type in ("resid_post", "resid_pre"):
            by_layer[edge.dst_layer]["resid"] += 1

    # By head
    if edge.dst_type == "attn_in" and edge.dst_head is not None:
        by_head[f"A{edge.dst_layer}.{edge.dst_head}"] += 1

    # By component type
    by_component_type[edge.dst_type] += 1


def _head_sort_key(item: Tuple[str, int]) -> Tuple[int, int]:
    """Sort key for head names like 'A3.5'."""
    key = item[0]
    if key.startswith("A"):
        parts = key[1:].split(".")
        return (int(parts[0]), int(parts[1]))
    return (999, 999)


# =============================================================================
# CIRCUIT FILE OPERATIONS
# =============================================================================


def load_prune_scores(circuit_path: str) -> Dict:
    """Load prune_scores from pickle file."""
    with open(circuit_path, "rb") as f:
        return pickle.load(f)


def get_circuit_path(model: str, band: str, draw: str) -> Path:
    """Get path to prune_scores.pkl for a specific circuit."""
    model_dir = MODEL_DIR_NAMES[model]
    return CIRCUITS_BASE / model_dir / band / draw / "prune_scores.pkl"


def process_single_circuit(model: str, band: str, draw: str) -> Dict[str, Any]:
    """
    Load and extract structure for a single circuit.

    Returns dict with all structural information, or error dict on failure.
    """
    import torch as t
    import traceback

    circuit_path = get_circuit_path(model, band, draw)
    model_short = model.replace("-", "_")
    circuit_id = f"{model_short}_{band}_{draw}"

    try:
        prune_scores = load_prune_scores(str(circuit_path))

        # Move to CPU if needed
        prune_scores_cpu = {}
        for k, v in prune_scores.items():
            if isinstance(v, t.Tensor):
                prune_scores_cpu[k] = v.cpu()
            else:
                prune_scores_cpu[k] = v

        structure = extract_circuit_structure(prune_scores_cpu, model, band, draw)
        structure["circuit_file"] = str(circuit_path)
        return structure

    except Exception as e:
        return {
            "circuit_id": circuit_id,
            "model": model,
            "band": band,
            "draw": draw,
            "circuit_file": str(circuit_path),
            "status": "error",
            "error": f"{str(e)}\n{traceback.format_exc()}",
            "extraction_timestamp": datetime.now().isoformat(),
        }


def flatten_structure_to_row(structure: Dict) -> Optional[Dict]:
    """Flatten a circuit structure dict into a single row for CSV."""
    if structure.get("status") == "error":
        return None

    row = {
        "circuit_id": structure["circuit_id"],
        "model": structure["model"],
        "band": structure["band"],
        "draw": structure["draw"],
        "threshold": structure.get("threshold"),
        "total_edges": structure["total_edges"],
        "total_possible": structure["total_possible_edges"],
        "edge_fraction": structure["edge_fraction"],
        "n_skip": structure.get("n_skip_connections", 0),
        "n_input": structure.get("n_input_edges", 0),
        "n_output": structure.get("n_output_edges", 0),
        "n_local": structure.get("n_local_edges", 0),
        "n_forward": structure.get("n_forward_edges", 0),
    }

    # Component type counts
    for comp, count in structure.get("by_component_type", {}).items():
        row[f"{comp}_edges"] = count

    # Head statistics
    by_head = structure.get("by_head", {})
    row["active_heads"] = len(by_head)
    if by_head:
        row["max_edges_per_head"] = max(by_head.values())
        row["mean_edges_per_head"] = sum(by_head.values()) / len(by_head)
    else:
        row["max_edges_per_head"] = 0
        row["mean_edges_per_head"] = 0.0

    return row
