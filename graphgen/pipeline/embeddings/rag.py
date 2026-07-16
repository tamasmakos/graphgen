
import networkx as nx
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Singleton model instance
_EMBEDDING_MODEL = None

def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
    """Lazy load sentence-transformer model.

    Device resolution order: explicit ``device`` arg > ``EMBEDDING_DEVICE``
    env var > auto (cuda if available, else cpu).  Set ``EMBEDDING_DEVICE=cpu``
    to keep embeddings off the GPU entirely.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            import os
            from sentence_transformers import SentenceTransformer
            import torch

            requested = device or os.environ.get('EMBEDDING_DEVICE') or 'auto'
            if requested == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = requested
            logger.info(f"Loading SentenceTransformer model {model_name} on {device}...")
            _EMBEDDING_MODEL = SentenceTransformer(model_name, device=device)
        except ImportError:
            logger.error("sentence-transformers not installed. Cannot generate embeddings.")
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    return _EMBEDDING_MODEL

def generate_rag_embeddings(
    graph: nx.DiGraph, 
    node_types: Optional[List[str]] = None,
    batch_size: int = 32
):
    """
    Generate meaningful vector embeddings for graph nodes using SentenceTransformers.
    
    Args:
        graph: NetworkX graph to enrich
        node_types: List of node types to generate embeddings for. 
                    Defaults to ['CHUNK', 'TextChunk', 'Document', 'TOPIC', 'ENTITY_CONCEPT']
    """
    if node_types is None:
        node_types = ['CHUNK', 'TextChunk', 'Document', 'TOPIC', 'ENTITY_CONCEPT']
        
    model = get_embedding_model()
    if not model:
        logger.warning("Skipping embedding generation (model not available).")
        return

    nodes_to_embed = []
    texts = []
    
    # Collect nodes that need embeddings
    for node_id, data in graph.nodes(data=True):
        if data.get('node_type') in node_types:
            # Skip if already has embedding (to save time in iterative loops)
            # BUT: In iterative mode, descriptions might update? 
            # For now, let's re-compute to be safe/simple, or check if we want updates.
            # actually, descriptions for entities don't change much. 
            # optimization: check if 'embedding' is present AND is numpy array
            if 'embedding' in data and isinstance(data['embedding'], np.ndarray):
                continue
                
            # Construct text representation
            text_parts = []
            
            # 1. Name/Title
            name = data.get('name') or data.get('title') or str(node_id)
            text_parts.append(str(name))
            
            # 2. Description/Summary
            desc = data.get('description') or data.get('summary') or data.get('text')
            if desc and isinstance(desc, str) and len(desc) > 5:
                 # Truncate very long descriptions for speed
                text_parts.append(desc[:1000])
                
            # 3. Ontology/Type info
            ontology = data.get('ontology_class') or data.get('llm_type')
            if ontology:
                text_parts.append(f"Type: {ontology}")
            
            full_text = " ".join(text_parts)
            
            nodes_to_embed.append(node_id)
            texts.append(full_text)
            
    if not nodes_to_embed:
        return
        
    logger.info(f"Generating embeddings for {len(nodes_to_embed)} nodes using {model.device}...")
    
    try:
        # Generate in batches
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Assign back to nodes
        for i, node_id in enumerate(nodes_to_embed):
            graph.nodes[node_id]['embedding'] = embeddings[i]
            
        logger.info(f"Successfully added embeddings to {len(nodes_to_embed)} nodes.")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
