"""Regenerate all thesis visualisation plots from saved pipeline artefacts.

Usage
-----
Run from the project root::

    python -m graphgen.analytics.regenerate_plots

Or with explicit paths::

    python -m graphgen.analytics.regenerate_plots \\
        --graph output/knowledge_graph.graphml \\
        --output-dir output/thesis_outputs

The script loads the saved knowledge graph, reads the modularity values from
the existing ``thesis_analytics_report.json`` (if present), re-runs the
analytics computation to recover the raw data arrays needed for plotting, and
writes all plots to ``--output-dir``.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict

import networkx as nx
import numpy as np

from graphgen.analytics.reporting import (
    extract_topic_embeddings,
    export_csv_artifacts,
    generate_thesis_analytics_report,
)
from graphgen.analytics.visualizer import (
    generate_interactive_explorer,
    plot_community_sizes,
    plot_node2vec_uplift,
    plot_scale_free,
    plot_similarity_distribution,
    plot_topic_heatmap,
)
from graphgen.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_GRAPH = os.path.join("output", "knowledge_graph.graphml")
_DEFAULT_OUTPUT = os.path.join("output", "thesis_outputs")
_REPORT_FILENAME = "thesis_analytics_report.json"


def _load_existing_report(output_dir: str) -> dict:
    """Read the saved JSON report to recover modularity values."""
    path = os.path.join(output_dir, _REPORT_FILENAME)
    if not os.path.exists(path):
        logger.warning("No existing report found at %s — modularity values will default to 0.", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _attach_embeddings(
    graph: nx.DiGraph,
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """Re-embed community and subcommunity nodes from their stored summaries.

    Embeddings are not preserved in GraphML.  This function uses the same
    sentence-transformers model as the pipeline to recompute them in-place,
    which enables the heatmap and similarity-distribution plots.

    Args:
        graph: The loaded knowledge graph (mutated in place).
        model_name: Sentence-transformers model identifier.

    Returns:
        Number of nodes that received an embedding.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for --recompute-embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    target_types = {"COMMUNITY", "TOPIC", "SUBCOMMUNITY", "SUBTOPIC"}

    nodes_to_embed: Dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type", "")).upper() not in target_types:
            continue
        text = data.get("summary") or data.get("title") or data.get("name") or ""
        if text:
            nodes_to_embed[node_id] = text

    if not nodes_to_embed:
        logger.warning("No community/subcommunity nodes with text found — cannot recompute embeddings.")
        return 0

    logger.info("Re-embedding %d nodes using %s …", len(nodes_to_embed), model_name)
    model = SentenceTransformer(model_name)
    node_ids = list(nodes_to_embed.keys())
    texts = [nodes_to_embed[n] for n in node_ids]
    vectors = model.encode(texts, show_progress_bar=False, batch_size=64)

    for node_id, vector in zip(node_ids, vectors):
        graph.nodes[node_id]["embedding"] = vector

    logger.info("Attached embeddings to %d nodes.", len(node_ids))
    return len(node_ids)


def _derive_community_assignments(graph: nx.DiGraph) -> dict:
    """Build a node→community mapping from graph node attributes.

    The interactive explorer expects ``{entity_node_id: community_id}``.
    This reconstructs that mapping by scanning COMMUNITY nodes and their
    outgoing CONTAINS / HAS_MEMBER edges, falling back to a ``community``
    attribute on entity nodes when no edges are present.
    """
    assignments: dict = {}

    # Prefer explicit community attribute on nodes (set during Leiden step).
    for node, data in graph.nodes(data=True):
        community = data.get("community")
        if community is not None:
            assignments[node] = community

    if assignments:
        return assignments

    # Fallback: walk CONTAINS / HAS_MEMBER edges from COMMUNITY nodes.
    community_node_types = {"COMMUNITY", "TOPIC"}
    for src, dst, edge_data in graph.edges(data=True):
        rel = str(edge_data.get("relation", edge_data.get("type", ""))).upper()
        if rel in ("CONTAINS", "HAS_MEMBER", "BELONGS_TO"):
            src_type = str(graph.nodes[src].get("node_type", "")).upper()
            if src_type in community_node_types:
                assignments[dst] = src

    return assignments


def regenerate(
    graph_path: str,
    output_dir: str,
    recompute_embeddings: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    """Load saved artefacts and regenerate all thesis plots."""
    if not os.path.exists(graph_path):
        logger.error("Graph file not found: %s", graph_path)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading graph from %s …", graph_path)
    graph: nx.DiGraph = nx.read_graphml(graph_path)
    logger.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    if recompute_embeddings:
        _attach_embeddings(graph, model_name=embedding_model)
    else:
        logger.info(
            "Embedding-dependent plots (heatmaps, similarity distributions) are skipped "
            "because embeddings are not stored in GraphML. "
            "Pass --recompute-embeddings to recompute them."
        )

    saved_report = _load_existing_report(output_dir)
    mod_cmp = saved_report.get("modularity_comparison", {})
    baseline_modularity: float = mod_cmp.get("baseline_modularity", 0.0)
    weighted_modularity: float | None = mod_cmp.get("weighted_modularity")

    report_path = os.path.join(output_dir, _REPORT_FILENAME)
    logger.info("Re-running analytics to recover raw data arrays …")
    report = generate_thesis_analytics_report(
        graph=graph,
        baseline_modularity=baseline_modularity,
        weighted_modularity=weighted_modularity,
        weighted_graph=graph,
        output_path=report_path,
    )

    # --- Scale-free plot ---
    sf_result = report.get("scale_free") or {}
    degree_sequence = report.get("_scale_free_degree_sequence")
    if degree_sequence and not sf_result.get("skipped"):
        plot_scale_free(
            degree_sequence=degree_sequence,
            alpha=sf_result.get("alpha", 2.0),
            xmin=sf_result.get("xmin", 1),
            log_log_r_squared=sf_result.get("log_log_r_squared", 0.0),
            output_path=os.path.join(output_dir, "scale_free_degree_distribution.png"),
            ks_p_value=sf_result.get("ks_p_value"),
            is_scale_free=sf_result.get("is_scale_free"),
            n_nodes=sf_result.get("n_nodes"),
        )

    # --- Node2Vec uplift plot ---
    null_distribution = report.get("_null_distribution")
    perm = report.get("permutation_test") or {}
    if null_distribution is not None and len(null_distribution) >= 2 and weighted_modularity is not None:
        plot_node2vec_uplift(
            baseline_modularity=baseline_modularity,
            weighted_modularity=weighted_modularity,
            null_distribution=np.array(null_distribution),
            output_path=os.path.join(output_dir, "node2vec_modularity_uplift.png"),
            p_value=perm.get("p_value"),
            n_permutations=perm.get("n_permutations"),
            null_95th=perm.get("null_95th"),
            null_99th=perm.get("null_99th"),
        )

    # --- Topic / subtopic similarity distribution plots ---
    sim_dist = report.get("similarity_distribution") or {}
    pairwise_values = report.get("_pairwise_values")
    if pairwise_values is not None:
        plot_similarity_distribution(
            pairwise_values=np.array(pairwise_values),
            output_path=os.path.join(output_dir, "topic_similarity_distribution.png"),
            mean=sim_dist.get("mean", 0.0),
            normality_p_value=sim_dist.get("normality_test", {}).get("shapiro_wilk_p_value"),
        )

    sub_dist = report.get("subcommunity_similarity_distribution") or {}
    sub_pairwise_values = report.get("_subcommunity_pairwise_values")
    if sub_pairwise_values is not None:
        plot_similarity_distribution(
            pairwise_values=np.array(sub_pairwise_values),
            output_path=os.path.join(output_dir, "subtopic_similarity_distribution.png"),
            mean=sub_dist.get("mean", 0.0),
            normality_p_value=sub_dist.get("normality_test", {}).get("shapiro_wilk_p_value"),
            title="Subtopic Similarity Distribution",
        )

    # --- Topic / subtopic heatmaps ---
    all_embeddings = extract_topic_embeddings(graph, levels=["COMMUNITY", "SUBCOMMUNITY"])
    topic_embs = all_embeddings.get("COMMUNITY", {})
    subtopic_embs = all_embeddings.get("SUBCOMMUNITY", {})

    if topic_embs:
        plot_topic_heatmap(
            topic_embs,
            {k: k for k in topic_embs},
            os.path.join(output_dir, "topic_similarity_heatmap.png"),
            title="Topic Similarity Heatmap",
        )

    if subtopic_embs:
        plot_topic_heatmap(
            subtopic_embs,
            {k: k for k in subtopic_embs},
            os.path.join(output_dir, "subtopic_similarity_heatmap.png"),
            title="Subtopic Similarity Heatmap",
        )

    # --- Interactive explorer ---
    communities = _derive_community_assignments(graph)
    if communities:
        generate_interactive_explorer(
            graph,
            os.path.join(output_dir, "interactive_graph.html"),
            communities,
        )

    # --- Community size distribution ---
    community_entity_counts: dict = {}
    subcommunity_entity_counts: dict = {}
    for node, data in graph.nodes(data=True):
        node_type = str(data.get("node_type", "")).upper()
        raw_count = data.get("entity_count", 0)
        try:
            count = int(raw_count) if raw_count is not None else 0
        except (ValueError, TypeError):
            count = 0
        if node_type in ("TOPIC", "COMMUNITY"):
            community_entity_counts[node] = count
        elif node_type in ("SUBTOPIC", "SUBCOMMUNITY"):
            subcommunity_entity_counts[node] = count

    if community_entity_counts:
        plot_community_sizes(
            community_sizes=community_entity_counts,
            subcommunity_sizes=subcommunity_entity_counts,
            output_path=os.path.join(output_dir, "community_size_distribution.png"),
        )

    # --- CSV artefacts ---
    export_csv_artifacts(graph, output_dir)

    logger.info("All plots written to %s", output_dir)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Regenerate thesis visualisation plots from saved pipeline artefacts.",
    )
    parser.add_argument(
        "--graph",
        default=_DEFAULT_GRAPH,
        help=f"Path to saved knowledge_graph.graphml (default: {_DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT,
        help=f"Directory to write plots into (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        default=True,
        help=(
            "Re-embed community/subcommunity summaries using sentence-transformers "
            "so that heatmap and similarity distribution plots can be generated. "
            "Requires the 'sentence-transformers' package."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model to use when --recompute-embeddings is set.",
    )
    args = parser.parse_args()

    regenerate(
        graph_path=args.graph,
        output_dir=args.output_dir,
        recompute_embeddings=args.recompute_embeddings,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
