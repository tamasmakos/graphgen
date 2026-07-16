"""
Analytics Reporting Module.

Generates the thesis analytics report by orchestrating:
- Node2Vec modularity comparison (baseline vs weighted)
- Node2Vec edge-weight permutation test
- Topic similarity distribution analysis (Shapiro-Wilk normality)
- CSV exports: topics, communities, top entities by centrality
"""

import csv
import json
import logging
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from graphgen.analytics.metrics import (
    run_node2vec_permutation_test,
    analyze_similarity_distribution,
    test_scale_free,
)

logger = logging.getLogger(__name__)


def extract_topic_embeddings(
    graph: nx.DiGraph,
    levels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract embeddings for TOPIC and SUBTOPIC nodes from the graph.

    Falls back to the mean of member entity embeddings when a topic node
    has no direct embedding stored.

    Args:
        graph: The knowledge graph after summarization.
        levels: Which hierarchy levels to extract.  Defaults to
                ``["COMMUNITY", "SUBCOMMUNITY"]``.

    Returns:
        Nested dict ``{level: {node_id: embedding_array}}``.
    """
    if levels is None:
        levels = ["COMMUNITY", "SUBCOMMUNITY"]

    result: Dict[str, Dict[str, np.ndarray]] = {}

    for level in levels:
        level_embeddings: Dict[str, np.ndarray] = {}

        if level == "COMMUNITY":
            target_node_types = {"COMMUNITY", "TOPIC"}
        elif level == "SUBCOMMUNITY":
            target_node_types = {"SUBCOMMUNITY", "SUBTOPIC"}
        else:
            target_node_types = {level}

        for node_id, node_data in graph.nodes(data=True):
            if str(node_data.get("node_type", "")).upper() not in target_node_types:
                continue

            embedding: Optional[np.ndarray] = None

            raw = node_data.get("embedding")
            if isinstance(raw, list):
                embedding = np.array(raw)
            elif isinstance(raw, np.ndarray):
                embedding = raw

            if embedding is None:
                # Fall back to the mean of connected entity embeddings.
                member_embeddings: List[np.ndarray] = []
                for predecessor in graph.predecessors(node_id):
                    pred_data = graph.nodes.get(predecessor, {})
                    if str(pred_data.get("node_type", "")).upper() in {
                        "ENTITY_CONCEPT",
                        "ENTITY",
                        "NAMEDENTITY",
                        "PLACE",
                    }:
                        raw_pred = pred_data.get("embedding")
                        if isinstance(raw_pred, list):
                            member_embeddings.append(np.array(raw_pred))
                        elif isinstance(raw_pred, np.ndarray):
                            member_embeddings.append(raw_pred)

                if member_embeddings:
                    embedding = np.mean(member_embeddings, axis=0)

            if embedding is not None:
                level_embeddings[node_id] = embedding

        result[level] = level_embeddings

    return result


def _build_similarity_matrix(
    embeddings: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Compute the pairwise cosine similarity matrix for a set of embeddings."""
    if len(embeddings) < 3:
        return None

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        ids = sorted(embeddings.keys())
        matrix = np.vstack([embeddings[i] for i in ids])
        return cosine_similarity(matrix)
    except ImportError:
        # Manual fallback when scikit-learn is absent.
        ids = sorted(embeddings.keys())
        matrix = np.vstack([embeddings[i] for i in ids])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms
        return normed @ normed.T


_ENTITY_NODE_TYPES_SEP = {"ENTITY_CONCEPT", "ENTITY", "NAMEDENTITY", "PLACE"}


def _collect_entity_embeddings_and_labels(
    graph: nx.DiGraph,
) -> tuple:
    """Gather entity embeddings plus their community / subcommunity labels.

    Returns ``(embeddings, community_labels, subcommunity_labels)`` where
    each is a dict keyed by entity node id.  Only entity nodes that carry
    both an ``embedding`` and a ``community_id`` are included.
    """
    embeddings: Dict[str, np.ndarray] = {}
    community_labels: Dict[str, int] = {}
    subcommunity_labels: Dict[str, int] = {}

    for node_id, data in graph.nodes(data=True):
        if str(data.get("node_type", "")).upper() not in _ENTITY_NODE_TYPES_SEP:
            continue
        comm_id = data.get("community_id")
        if comm_id is None:
            continue
        raw = data.get("embedding")
        if isinstance(raw, list):
            emb = np.array(raw)
        elif isinstance(raw, np.ndarray):
            emb = raw
        else:
            continue
        embeddings[node_id] = emb
        community_labels[node_id] = int(comm_id)
        sub_id = data.get("subcommunity_id")
        if sub_id is not None:
            # Namespace the local sub id by its parent community so that
            # sub-cluster 0 of community 1 does not collide with sub-cluster
            # 0 of community 2.
            subcommunity_labels[node_id] = int(comm_id) * 1000 + int(sub_id)

    return embeddings, community_labels, subcommunity_labels


def analyze_entity_community_separation(graph: nx.DiGraph) -> Dict[str, Any]:
    """Quantify structural-coherence vs. semantic-distinctiveness.

    Operationalises the central Chapter-4 dichotomy: entities are grouped
    by the community the *structural* Leiden partition assigned them to,
    and we measure how well-separated those groups are in *semantic*
    (embedding) space.

    * A **silhouette score** near +1 means entities sit closest to their
      own community in embedding space (structure and semantics agree);
      values near 0 mean overlapping clusters; **negative** values mean
      entities are, on average, semantically closer to *other* communities
      than their own — the signature of a structurally-modular but
      semantically-homogeneous discourse.
    * **Global separation** reports the mean pairwise cosine distance /
      similarity across all entity embeddings, independent of the
      partition.

    Both the community-level and subcommunity-level partitions are scored.

    Returns:
        Dict with ``community`` and ``subcommunity`` sub-dicts (each with
        ``silhouette``, ``per_cluster_silhouette``, ``n_entities``,
        ``n_clusters``) plus a top-level ``global_separation`` dict.  A
        ``skipped`` key is present when there are too few embedded,
        community-tagged entities.
    """
    from graphgen.analytics.separation import (
        compute_global_separation,
        run_silhouette_analysis,
    )

    embeddings, community_labels, subcommunity_labels = (
        _collect_entity_embeddings_and_labels(graph)
    )

    if len(embeddings) < 3:
        logger.warning(
            "Community separation skipped: only %d embedded entity node(s) "
            "with a community assignment.",
            len(embeddings),
        )
        return {"skipped": f"Insufficient embedded entities (found {len(embeddings)}, need >= 3)"}

    avg_dist, avg_sim = compute_global_separation(embeddings)

    def _silhouette_block(labels: Dict[str, int]) -> Dict[str, Any]:
        overall, per_cluster = run_silhouette_analysis(embeddings, labels)
        return {
            "silhouette": overall,
            "per_cluster_silhouette": per_cluster,
            "n_entities": len(labels),
            "n_clusters": len(set(labels.values())) if labels else 0,
        }

    result: Dict[str, Any] = {
        "global_separation": {
            "mean_cosine_distance": avg_dist,
            "mean_cosine_similarity": avg_sim,
        },
        "community": _silhouette_block(community_labels),
    }
    if subcommunity_labels:
        result["subcommunity"] = _silhouette_block(subcommunity_labels)
    else:
        result["subcommunity"] = {"skipped": "No subcommunity_id labels present"}

    logger.info(
        "Community separation: community silhouette=%s, subcommunity silhouette=%s, "
        "mean entity similarity=%.3f",
        result["community"].get("silhouette"),
        result.get("subcommunity", {}).get("silhouette"),
        avg_sim,
    )
    return result


def generate_thesis_analytics_report(
    graph: nx.DiGraph,
    baseline_modularity: float,
    weighted_modularity: Optional[float],
    weighted_graph: Optional[nx.Graph],
    output_path: str,
    permutation_n: int = 1000,
    permutation_seed: int = 42,
    seed_stability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate the thesis analytics report.

    The report contains three sections:

    1. **Modularity comparison** — raw lift of Node2Vec-weighted Leiden
       over the unweighted baseline.
    2. **Permutation test** — whether the Node2Vec edge-weight pattern
       produces significantly higher modularity than random weight
       assignments (n=``permutation_n`` shuffles).
    3. **Similarity distribution** — descriptive statistics and a
       Shapiro-Wilk normality test on all pairwise cosine similarities
       between COMMUNITY-level topic-summary embeddings.

    Args:
        graph: Knowledge graph after summarization (must contain TOPIC/
               COMMUNITY nodes with ``embedding`` attributes).
        baseline_modularity: Modularity from unweighted Leiden run.
        weighted_modularity: Modularity from Node2Vec-weighted Leiden run.
                             Pass ``None`` when Node2Vec is disabled.
        weighted_graph: The NetworkX graph with Node2Vec weights applied
                        to edges.  Pass ``None`` when Node2Vec is disabled.
        output_path: File path where the JSON report is written.
        permutation_n: Number of weight-shuffle permutations.
        permutation_seed: Random seed for reproducibility.

    Returns:
        The report as a plain dict (also persisted to ``output_path``).
    """
    logger.info("Generating thesis analytics report...")

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "scale_free": None,
        "modularity_comparison": None,
        "permutation_test": None,
        "seed_stability": None,
        "similarity_distribution": None,
    }

    # Seed-stability significance test (computed upstream during community
    # detection); the raw per-run distributions are stashed for plotting and
    # stripped from the JSON-facing copy.
    if seed_stability is not None:
        report["_seed_stability_baseline"] = seed_stability.get("baseline_distribution")
        report["_seed_stability_weighted"] = seed_stability.get("weighted_distribution")
        report["seed_stability"] = {
            k: v
            for k, v in seed_stability.items()
            if k not in ("baseline_distribution", "weighted_distribution")
        }

    # ------------------------------------------------------------------
    # 0. Scale-free property test
    # ------------------------------------------------------------------
    sf_result = test_scale_free(graph)
    degree_sequence = sf_result.pop("degree_sequence", None)
    report["scale_free"] = sf_result
    # Stash raw degree sequence for plotting (excluded from JSON).
    report["_scale_free_degree_sequence"] = degree_sequence

    # ------------------------------------------------------------------
    # 1. Modularity comparison
    # ------------------------------------------------------------------
    if weighted_modularity is not None:
        delta = weighted_modularity - baseline_modularity
        report["modularity_comparison"] = {
            "baseline_modularity": float(baseline_modularity),
            "weighted_modularity": float(weighted_modularity),
            "delta": float(delta),
            "relative_lift_pct": float(delta / baseline_modularity * 100)
            if baseline_modularity > 0
            else None,
        }
        logger.info(
            "Modularity: baseline=%.4f  weighted=%.4f  delta=%+.4f",
            baseline_modularity,
            weighted_modularity,
            delta,
        )
    else:
        report["modularity_comparison"] = {
            "baseline_modularity": float(baseline_modularity),
            "weighted_modularity": None,
            "delta": None,
            "relative_lift_pct": None,
        }
        logger.info(
            "Node2Vec disabled — only baseline modularity recorded (%.4f).",
            baseline_modularity,
        )

    # ------------------------------------------------------------------
    # 2. Permutation test
    # ------------------------------------------------------------------
    if weighted_graph is not None and weighted_modularity is not None:
        logger.info(
            "Running Node2Vec permutation test (n=%d)...", permutation_n
        )
        perm_result = run_node2vec_permutation_test(
            graph=weighted_graph,
            observed_modularity=weighted_modularity,
            n_permutations=permutation_n,
            seed=permutation_seed,
        )
        # Stash the raw null distribution for plotting (excluded from JSON).
        report["_null_distribution"] = perm_result.get("null_distribution")
        # Strip from the JSON-facing copy to keep the file readable.
        perm_for_file = {
            k: v for k, v in perm_result.items() if k != "null_distribution"
        }
        report["permutation_test"] = perm_for_file
    else:
        report["permutation_test"] = {"skipped": "Node2Vec not enabled"}
        logger.info("Permutation test skipped (Node2Vec disabled).")

    # ------------------------------------------------------------------
    # 3. Similarity distribution
    # ------------------------------------------------------------------
    all_embeddings = extract_topic_embeddings(graph, levels=["COMMUNITY"])
    community_embeddings = all_embeddings.get("COMMUNITY", {})

    if len(community_embeddings) >= 3:
        sim_matrix = _build_similarity_matrix(community_embeddings)
        if sim_matrix is not None:
            labels = list(sorted(community_embeddings.keys()))
            dist_result = analyze_similarity_distribution(
                sim_matrix, labels=labels
            )
            # Keep pairwise_values in the returned dict for callers that
            # need to plot the distribution, but strip it from the JSON
            # file to keep it readable.
            pairwise_for_file = {
                k: v for k, v in dist_result.items() if k != "pairwise_values"
            }
            report["similarity_distribution"] = pairwise_for_file
            # Stash the raw array under a separate key for the caller.
            report["_pairwise_values"] = dist_result.get("pairwise_values")
        else:
            report["similarity_distribution"] = {
                "skipped": "Could not build similarity matrix"
            }
    else:
        n = len(community_embeddings)
        logger.warning(
            "Similarity distribution skipped: need >= 3 topic embeddings "
            "(found %d).",
            n,
        )
        report["similarity_distribution"] = {
            "skipped": f"Insufficient topic embeddings (found {n}, need >= 3)"
        }

    # ------------------------------------------------------------------
    # 4. Subcommunity similarity distribution
    # ------------------------------------------------------------------
    sub_embeddings = extract_topic_embeddings(graph, levels=["SUBCOMMUNITY"])
    subcommunity_embeddings = sub_embeddings.get("SUBCOMMUNITY", {})

    if len(subcommunity_embeddings) >= 3:
        sub_matrix = _build_similarity_matrix(subcommunity_embeddings)
        if sub_matrix is not None:
            sub_labels = list(sorted(subcommunity_embeddings.keys()))
            sub_dist_result = analyze_similarity_distribution(
                sub_matrix, labels=sub_labels
            )
            sub_pairwise_for_file = {
                k: v for k, v in sub_dist_result.items() if k != "pairwise_values"
            }
            report["subcommunity_similarity_distribution"] = sub_pairwise_for_file
            report["_subcommunity_pairwise_values"] = sub_dist_result.get("pairwise_values")
        else:
            report["subcommunity_similarity_distribution"] = {
                "skipped": "Could not build similarity matrix"
            }
    else:
        n_sub = len(subcommunity_embeddings)
        logger.warning(
            "Subcommunity similarity distribution skipped: need >= 3 embeddings "
            "(found %d).",
            n_sub,
        )
        report["subcommunity_similarity_distribution"] = {
            "skipped": f"Insufficient subcommunity embeddings (found {n_sub}, need >= 3)"
        }

    # ------------------------------------------------------------------
    # 5. Structural-coherence vs. semantic-distinctiveness (silhouette)
    # ------------------------------------------------------------------
    report["community_separation"] = analyze_entity_community_separation(graph)

    # ------------------------------------------------------------------
    # Persist (private underscore keys are for callers only, not the file)
    # ------------------------------------------------------------------
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json_report = {k: v for k, v in report.items() if not k.startswith("_")}
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(json_report, fh, indent=2, ensure_ascii=False)
        logger.info("Thesis analytics report saved to: %s", output_path)
    except Exception:
        logger.exception("Failed to save thesis analytics report.")

    return report


def export_csv_artifacts(
    graph: nx.DiGraph,
    output_dir: str,
    top_k_centrality: int = 20,
) -> Dict[str, str]:
    """Export topics, communities, and top entities by centrality to CSV files.

    Writes three files into *output_dir*:

    * ``topics.csv``         — one row per top-level COMMUNITY/TOPIC node.
    * ``communities.csv``    — full hierarchy (COMMUNITY + SUBCOMMUNITY) with a
                               ``level`` column and a ``parent_node_id`` column
                               for sub-community rows.
    * ``top_entities_by_centrality.csv`` — top-``top_k_centrality`` entities
                                           ranked by each centrality measure,
                                           with all measures as columns.

    Args:
        graph: The knowledge graph after summarization and community detection.
        output_dir: Directory where the CSV files are written.
        top_k_centrality: How many top entities to include per centrality measure.

    Returns:
        Dict mapping ``{"topics", "communities", "top_entities_by_centrality"}``
        to the absolute path of each written file.  A key is absent when the
        export was skipped (e.g. no nodes of that type found).
    """
    from graphgen.analytics.centrality import (
        calculate_centrality_measures,
        get_top_entities_global,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. topics.csv — top-level COMMUNITY / TOPIC nodes
    # ------------------------------------------------------------------
    topic_rows: List[Dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        ntype = str(data.get("node_type", "")).upper()
        if ntype not in ("COMMUNITY", "TOPIC"):
            continue
        topic_rows.append({
            "node_id": node_id,
            "name": data.get("name", node_id),
            "summary": data.get("summary", ""),
            "entity_count": data.get("entity_count", ""),
            "community_id": data.get("community_id", ""),
        })

    if topic_rows:
        topic_rows.sort(key=lambda r: str(r["name"]))
        topics_path = out / "topics.csv"
        fieldnames = ["node_id", "name", "summary", "entity_count", "community_id"]
        with open(topics_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(topic_rows)
        logger.info("topics.csv saved to %s (%d rows)", topics_path, len(topic_rows))
        written["topics"] = str(topics_path)
    else:
        logger.warning("topics.csv skipped: no COMMUNITY/TOPIC nodes found.")

    # ------------------------------------------------------------------
    # 2. communities.csv — full hierarchy (COMMUNITY + SUBCOMMUNITY)
    # ------------------------------------------------------------------
    # Build a reverse map: subcommunity → parent community
    parent_map: Dict[str, str] = {}
    for src, dst, edge_data in graph.edges(data=True):
        src_type = str(graph.nodes[src].get("node_type", "")).upper()
        dst_type = str(graph.nodes[dst].get("node_type", "")).upper()
        if src_type in ("COMMUNITY", "TOPIC") and dst_type in ("SUBCOMMUNITY", "SUBTOPIC"):
            parent_map[dst] = src

    community_rows: List[Dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        ntype = str(data.get("node_type", "")).upper()
        if ntype in ("COMMUNITY", "TOPIC"):
            level = "community"
        elif ntype in ("SUBCOMMUNITY", "SUBTOPIC"):
            level = "subcommunity"
        else:
            continue
        community_rows.append({
            "node_id": node_id,
            "level": level,
            "name": data.get("name", node_id),
            "summary": data.get("summary", ""),
            "entity_count": data.get("entity_count", ""),
            "community_id": data.get("community_id", ""),
            "parent_node_id": parent_map.get(node_id, ""),
        })

    if community_rows:
        community_rows.sort(key=lambda r: (r["level"], str(r["name"])))
        communities_path = out / "communities.csv"
        fieldnames = [
            "node_id", "level", "name", "summary",
            "entity_count", "community_id", "parent_node_id",
        ]
        with open(communities_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(community_rows)
        logger.info(
            "communities.csv saved to %s (%d rows)", communities_path, len(community_rows)
        )
        written["communities"] = str(communities_path)
    else:
        logger.warning("communities.csv skipped: no community nodes found.")

    # ------------------------------------------------------------------
    # 3. top_entities_by_centrality.csv
    # ------------------------------------------------------------------
    entity_node_types = {"ENTITY_CONCEPT", "ENTITY", "NAMEDENTITY", "PLACE"}
    available_types = {
        str(d.get("node_type", "")).upper()
        for _, d in graph.nodes(data=True)
    }
    matched_type = next(
        (t for t in ["ENTITY_CONCEPT", "ENTITY", "NAMEDENTITY", "PLACE"] if t in available_types),
        None,
    )

    if matched_type is None:
        logger.warning(
            "top_entities_by_centrality.csv skipped: no entity nodes found "
            "(looked for %s).",
            entity_node_types,
        )
    else:
        centrality_results = calculate_centrality_measures(graph, node_type_filter=matched_type)
        if centrality_results:
            global_top = get_top_entities_global(centrality_results, graph, top_k=top_k_centrality)

            # Collect all unique node_ids that appear in any measure's top-K list.
            seen_nodes: Dict[str, Dict[str, Any]] = {}
            for measure, rankings in global_top.items():
                for entry in rankings:
                    nid = entry["node_id"]
                    if nid not in seen_nodes:
                        node_data = graph.nodes[nid]
                        comm_id = node_data.get("community_id", "")
                        # community_id may be stored as an integer on entity nodes
                        seen_nodes[nid] = {
                            "node_id": nid,
                            "name": entry.get("name", nid),
                            "community_id": comm_id,
                            "description": entry.get("description", ""),
                        }
                    seen_nodes[nid][measure] = round(entry["score"], 6)

            measure_names = list(centrality_results.keys())
            cent_rows = list(seen_nodes.values())
            # Sort by degree centrality descending as a sensible default.
            cent_rows.sort(key=lambda r: r.get("degree", 0.0), reverse=True)

            cent_path = out / "top_entities_by_centrality.csv"
            fieldnames = ["node_id", "name", "community_id", "description"] + measure_names
            with open(cent_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(cent_rows)
            logger.info(
                "top_entities_by_centrality.csv saved to %s (%d rows)",
                cent_path,
                len(cent_rows),
            )
            written["top_entities_by_centrality"] = str(cent_path)
        else:
            logger.warning(
                "top_entities_by_centrality.csv skipped: centrality calculation returned "
                "no results for node type '%s'.",
                matched_type,
            )

    return written
