"""
PyKeen Knowledge Graph Embedding Training.

Trains a KGE model (e.g., DistMult) on entity-relation triples and provides:
1. Entity embeddings for use in downstream tasks.
2. Edge weight computation based on embedding similarity (for Leiden community detection).

Ported from thesis/entity_kg.py with clean architecture.
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# PyKEEN imports
try:
    import torch
    import pandas as pd
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logger.warning("PyKEEN not available. KGE training will be disabled.")


def collect_entity_relation_triples(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Collect all entity-relation triples from the graph.
    
    Extracts edges where graph_type == 'entity_relation' and forms
    (head, relation, tail) triples suitable for KGE training.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        
    Returns:
        List of (head, relation, tail) triples
    """
    triples: List[Tuple[str, str, str]] = []
    
    for u, v, data in graph.edges(data=True):
        if data.get('graph_type') == 'entity_relation':
            rel = data.get('relation_type') or data.get('label') or 'RELATED_TO'
            triples.append((str(u), str(rel), str(v)))
    
    logger.info(f"Collected {len(triples)} entity-relation triples for KGE training")
    return triples


def train_global_kge(
    graph: nx.DiGraph,
    config: Any
) -> Dict[str, np.ndarray]:
    """
    Train a global KGE model over the full entity-relation graph.
    
    Uses PyKEEN to train a knowledge graph embedding model (default: DistMult)
    and returns entity embeddings as numpy arrays.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        config: KGESettings object with training configuration
        
    Returns:
        Dictionary mapping entity IDs to their embedding vectors
    """
    if not PYKEEN_AVAILABLE:
        logger.warning("PyKEEN not available; skipping KGE training.")
        return {}
    
    # Collect triples
    triples = collect_entity_relation_triples(graph)
    
    if len(triples) < 10:
        logger.warning(f"Insufficient triples for KGE training ({len(triples)} < 10). Skipping.")
        return {}
    
    try:
        # Create DataFrame for PyKEEN
        df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
        tf = TriplesFactory.from_labeled_triples(df[['head', 'relation', 'tail']].values)
        
        logger.info(f"Training KGE model: {config.model} (dim={config.embedding_dim}, epochs={config.num_epochs})")
        
        # Run PyKEEN pipeline
        result = pipeline(
            model=config.model,
            training=tf,
            validation=tf,
            testing=tf,
            model_kwargs=dict(embedding_dim=config.embedding_dim),
            optimizer_kwargs=dict(lr=config.learning_rate),
            training_kwargs=dict(
                num_epochs=config.num_epochs, 
                use_tqdm_batch=False, 
                checkpoint_frequency=0
            ),
            evaluation_kwargs=dict(use_tqdm=False),
            stopper='early',
            stopper_kwargs=dict(
                frequency=2, 
                patience=config.early_stopping_patience
            ),
            random_seed=42
        )
        
        # Extract entity embeddings
        entity_to_id = tf.entity_to_id
        if not entity_to_id:
            logger.warning("No entities found in trained model")
            return {}
        
        all_ids = torch.arange(len(entity_to_id))
        with torch.no_grad():
            embs = result.model.entity_representations[0](all_ids).cpu().numpy()
        
        # Build embedding dictionary
        id_to_entity = {idx: ent for ent, idx in entity_to_id.items()}
        embeddings = {id_to_entity[i]: embs[i] for i in range(len(id_to_entity))}
        
        logger.info(f"KGE training complete: {len(embeddings)} entity embeddings generated")
        return embeddings
        
    except Exception as e:
        logger.error(f"KGE training failed: {e}")
        return {}


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_edge_weights_from_kge(
    graph: nx.DiGraph,
    embeddings: Dict[str, np.ndarray],
    weight_transform: str = "similarity"
) -> int:
    """
    Compute edge weights from KGE embeddings and add them to the graph.
    
    For each entity-relation edge, computes the cosine similarity between
    the head and tail entity embeddings. This similarity is used as the
    edge weight for community detection.
    
    Args:
        graph: NetworkX DiGraph (modified in-place)
        embeddings: Dictionary mapping entity IDs to embedding vectors
        weight_transform: How to transform similarity to weight
            - "similarity": Use raw cosine similarity (0 to 1)
            - "affinity": Use (1 + similarity) / 2 to ensure positive weights
            
    Returns:
        Number of edges with weights added
    """
    if not embeddings:
        logger.warning("No embeddings provided for edge weight computation")
        return 0
    
    edges_weighted = 0
    
    for u, v, data in graph.edges(data=True):
        if data.get('graph_type') != 'entity_relation':
            continue
        
        u_emb = embeddings.get(str(u))
        v_emb = embeddings.get(str(v))
        
        if u_emb is None or v_emb is None:
            # Default weight for entities without embeddings
            data['weight'] = 1.0
            continue
        
        similarity = _cosine_similarity(u_emb, v_emb)
        
        if weight_transform == "affinity":
            # Transform to [0.5, 1.0] range for positive weights
            weight = (1.0 + similarity) / 2.0
        else:
            # Use raw similarity, clamped to be positive
            weight = max(0.01, similarity)
        
        data['weight'] = weight
        edges_weighted += 1
    
    logger.info(f"Added KGE-based weights to {edges_weighted} edges")
    
    # Log weight statistics
    if edges_weighted > 0:
        weights = [d.get('weight', 1.0) for _, _, d in graph.edges(data=True) 
                   if d.get('graph_type') == 'entity_relation']
        logger.info(f"Edge weight stats: min={min(weights):.4f}, max={max(weights):.4f}, "
                   f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
    
    return edges_weighted


def store_embeddings_in_graph(
    graph: nx.DiGraph,
    embeddings: Dict[str, np.ndarray]
) -> int:
    """
    Store KGE embeddings as node attributes in the graph.
    
    Adds a 'kge_embedding' attribute to each entity node that has an embedding.
    This allows downstream tasks to access the embeddings directly from the graph.
    
    Args:
        graph: NetworkX DiGraph (modified in-place)
        embeddings: Dictionary mapping entity IDs to embedding vectors
        
    Returns:
        Number of nodes with embeddings stored
    """
    nodes_updated = 0
    
    for entity_id, embedding in embeddings.items():
        if graph.has_node(entity_id):
            # Store as list for JSON serialization compatibility
            graph.nodes[entity_id]['kge_embedding'] = embedding.tolist()
            nodes_updated += 1
    
    logger.info(f"Stored KGE embeddings in {nodes_updated} graph nodes")
    return nodes_updated
