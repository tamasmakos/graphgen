import logging
from pathlib import Path
import numpy as np
import networkx as nx

# Optional plotting deps
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

logger = logging.getLogger(__name__)

def plot_kge_communities(graph: nx.Graph, partition: dict, output_path: str):
    """
    Generate a 3D plot of KGE embeddings colored by community.
    
    Args:
        graph: NetworkX graph containing 'kge_embedding' node attributes.
        partition: Dictionary mapping node_id -> community_id.
        output_path: Path to save the plot.
    """
    if not VIZ_AVAILABLE:
        logger.warning("Visualization dependencies (matplotlib, sklearn) not found. Skipping plot.")
        return

    # Extract nodes that have BOTH an embedding and a community assignment
    valid_nodes = []
    embeddings = []
    communities = []
    
    for node, data in graph.nodes(data=True):
        emb = data.get('kge_embedding')
        comm = partition.get(node)
        
        if emb is not None and comm is not None:
            valid_nodes.append(node)
            embeddings.append(np.array(emb))
            communities.append(comm)
            
    if len(valid_nodes) < 3:
        logger.warning(f"Not enough nodes with embeddings to plot ({len(valid_nodes)}). Needing > 3 for PCA.")
        return
        
    try:
        X = np.array(embeddings)
        
        # Reduce to 3D
        # Handle cases with small N
        n_components = min(3, len(valid_nodes), X.shape[1])
        if n_components < 3:
             logger.warning(f"Embedding dimension or node count too low for 3D plot. Skipping.")
             return

        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        
        node_pos = {n: X_3d[i] for i, n in enumerate(valid_nodes)}
        
        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter nodes
        scatter = ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], 
                             c=communities, cmap='tab20', s=100, edgecolors='k', alpha=0.8)
        
        # Draw edges (only if both ends are valid)
        # To avoid clutter, we can limit to top weighted edges or just all edges if graph is small-ish
        # For iteration reports, let's plot edges with weight > threshold or all if small
        
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            if u in node_pos and v in node_pos:
                weight = data.get('weight', 0.1)
                # Filter weak edges for visualization clarity?
                # Let's verify: KGE weights are 0-1.
                if weight < 0.2: 
                    continue
                    
                p1 = node_pos[u]
                p2 = node_pos[v]
                
                alpha = min(0.8, max(0.05, weight))
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        c='gray', alpha=alpha, linewidth=0.5)
                edge_count += 1
                
        ax.set_title(f"3D KGE Landscape ({len(valid_nodes)} nodes, {edge_count} visible edges)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        
        # Legend (if not too many communities)
        unique_comms = len(set(communities))
        if unique_comms <= 20:
            legend1 = ax.legend(*scatter.legend_elements(), title="Communities", loc="upper right")
            ax.add_artist(legend1)
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"Saved KGE visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate KGE plot: {e}", exc_info=True)
