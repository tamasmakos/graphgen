"""Metrics for graph quality and embedding analysis."""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def calculate_modularity(graph: nx.Graph, communities: Dict[str, int]) -> float:
    """Calculate modularity of the graph given community assignments."""
    try:
        # Create community sets for nodes that exist in both graph and communities
        community_sets = defaultdict(set)
        common_nodes = set()
        
        for node, comm_id in communities.items():
            if node in graph:
                community_sets[comm_id].add(node)
                common_nodes.add(node)
        
        # Filter out empty communities
        valid_communities = [nodes for nodes in community_sets.values() if nodes]
        
        if not valid_communities:
            logger.info(
                "Skipping modularity: no valid communities (nodes=%d, communities=%d).",
                graph.number_of_nodes(),
                len(community_sets),
            )
            return 0.0
            
        # Create subgraph of only the nodes involved in communities
        # Modularity requires partition to cover all nodes in the graph
        subgraph = graph.subgraph(common_nodes)
        if len(subgraph) == 0:
             return 0.0

        return nx.community.modularity(subgraph, valid_communities)
    except Exception:
        logger.exception("Failed to calculate modularity.")
        return 0.0

def calculate_topic_overlap(topic_embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate average cosine similarity between all pairs of topic embeddings."""
    try:
        if len(topic_embeddings) < 2:
            logger.info(
                "Skipping topic overlap: need at least 2 embeddings (found=%d).",
                len(topic_embeddings),
            )
            return 0.0
        
        embeddings = list(topic_embeddings.values())
        if not embeddings:
            return 0.0
            
        # Stack embeddings
        matrix = np.vstack(embeddings)
        
        # Calculate pairwise cosine similarity
        sim_matrix = cosine_similarity(matrix)
        
        # Get upper triangle excluding diagonal
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        if len(upper_tri) == 0:
            logger.info("Skipping topic overlap: no pairwise similarities computed.")
            return 0.0
            
        return float(np.mean(upper_tri))
    except Exception:
        logger.exception("Failed to calculate topic overlap.")
        return 0.0

def analyze_modularity_vs_overlap(
    results_history: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Correlate modularity with topic overlap across iterations.
    Expects list of dicts with keys 'modularity' and 'topic_overlap'.
    """
    if len(results_history) < 3:
        logger.info(
            "Skipping correlation analysis: need >= 3 data points (found=%d).",
            len(results_history),
        )
        return {}
        
    stats = {}
    try:
        mods = [r.get('modularity', 0) for r in results_history]
        overlaps = [r.get('topic_overlap', 0) for r in results_history]
        
        # Pearson correlation
        p_corr, p_val = pearsonr(mods, overlaps)
        stats['pearson_correlation'] = p_corr
        stats['pearson_p_value'] = p_val
        
        # Spearman correlation (robust to outliers)
        s_corr, s_val = spearmanr(mods, overlaps)
        stats['spearman_correlation'] = s_corr
        stats['spearman_p_value'] = s_val
        
    except Exception:
        logger.exception("Correlation analysis failed.")
        
    return stats

def evaluate_kge_model_quality(
    graph: nx.DiGraph, 
    model_name: str = 'TransE', 
    embedding_dim: int = 64,
    epochs: int = 50
) -> Dict[str, Any]:
    """
    Train a KGE model and return quality metrics (MRR, Hits@k).
    Wraps PyKEEN pipeline.
    """
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.warning("PyKEEN not installed. Skipping KGE evaluation.")
        return {}
        
    # Extract triples
    triples = []
    for u, v, data in graph.edges(data=True):
        rel = data.get('relation_type') or data.get('label') or 'RELATED_TO'
        triples.append((str(u), str(rel), str(v)))
    
    if not triples:
        logger.info("Skipping KGE evaluation: no triples found.")
        return {}
        
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    tf = TriplesFactory.from_labeled_triples(df[['head', 'relation', 'tail']].values)
    
    training, testing = tf.split([0.8, 0.2], random_state=42)
    logger.info(
        "KGE evaluation dataset: triples=%d, train=%d, test=%d.",
        len(triples),
        training.num_triples,
        testing.num_triples,
    )
    
    try:
        result = pipeline(
            model=model_name,
            training=training,
            testing=testing,
            model_kwargs=dict(embedding_dim=embedding_dim),
            training_kwargs=dict(num_epochs=epochs),
            random_seed=42
        )
        
        metrics = result.get_metric_results().to_dict()
        # Simplify metrics
        simple_metrics = {
            'mrr': metrics.get('both', {}).get('realistic', {}).get('inverse_harmonic_mean_rank'),
            'hits_at_10': metrics.get('both', {}).get('realistic', {}).get('hits_at_10'),
            'loss': result.losses[-1] if result.losses else None
        }
        return simple_metrics
        
    except Exception:
        logger.exception("PyKEEN evaluation failed for %s.", model_name)
        return {}
