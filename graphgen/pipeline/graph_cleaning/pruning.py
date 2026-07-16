import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)

def prune_graph(graph: nx.DiGraph, processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prune the graph based on edge confidence and node isolation.

    Args:
        graph: The graph to prune in-place.
        processing_config: The 'processing' sub-dict from PipelineSettings
            (i.e. settings.processing.model_dump()).
    """
    if not processing_config.get('enable_pruning', True):
        return {
            "nodes_removed": 0,
            "edges_removed": 0,
            "final_nodes": graph.number_of_nodes(),
            "final_edges": graph.number_of_edges(),
        }

    pruning_threshold = processing_config.get('pruning_threshold', 0.0)

    # 1. Prune low-confidence entity_relation edges
    edges_removed = 0
    if pruning_threshold > 0:
        edges_to_remove = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if data.get('graph_type') == 'entity_relation'
            and data.get('confidence', 1.0) < pruning_threshold
        ]
        graph.remove_edges_from(edges_to_remove)
        edges_removed = len(edges_to_remove)
        logger.info("Pruned %d edges with confidence < %.3f", edges_removed, pruning_threshold)

    # 2. Prune isolated nodes if configured
    nodes_to_remove: list = []
    if processing_config.get('prune_isolated_nodes', True):
        nodes_to_remove = [n for n in graph.nodes() if graph.degree(n) == 0]
        graph.remove_nodes_from(nodes_to_remove)

    return {
        "nodes_removed": len(nodes_to_remove),
        "edges_removed": edges_removed,
        "final_nodes": graph.number_of_nodes(),
        "final_edges": graph.number_of_edges(),
    }
