"""Visualization utilities for analytics outputs."""

import logging
import os
from typing import Dict, Optional, List, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

def plot_topic_heatmap(
    topic_embeddings: Dict[str, np.ndarray], 
    topic_labels: Dict[str, str],
    output_path: str
) -> None:
    """
    Generate and save a heatmap of topic similarity.
    """
    try:
        if not topic_embeddings:
            logger.info("Skipping heatmap: no topic embeddings available.")
            return

        sorted_ids = sorted(topic_embeddings.keys())
        matrix = np.vstack([topic_embeddings[tid] for tid in sorted_ids])
        labels = [topic_labels.get(tid, tid) for tid in sorted_ids]
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(matrix)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, cmap="YlOrRd")
        plt.title("Topic Similarity Heatmap")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved heatmap to {output_path}")
        
    except Exception:
        logger.exception("Failed to plot heatmap.")

def generate_interactive_explorer(
    graph: nx.DiGraph, 
    output_path: str,
    communities: Optional[Dict[str, int]] = None
) -> None:
    """
    Generate an interactive HTML visualization using PyVis with search capability.
    """
    try:
        from pyvis.network import Network  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("PyVis not installed. Skipping interactive explorer.")
        return

    # Create network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Add nodes and edges
    # Filter for Entities and Communities only (skip chunks to reduce noise)
    valid_types = {'ENTITY_CONCEPT', 'COMMUNITY', 'SUBCOMMUNITY'}
    
    display_graph = nx.DiGraph()
    
    for n, data in graph.nodes(data=True):
        if data.get('node_type') in valid_types:
            # Color by community
            color = "#97c2fc"
            if communities and n in communities:
                import matplotlib.colors as mcolors
                comm_id = communities[n]
                # Generate color from hash
                color = list(mcolors.TABLEAU_COLORS.values())[comm_id % 10]
                
            label = data.get('name', str(n))
            title = f"{label}\nType: {data.get('node_type')}"
            if 'description' in data:
                title += f"\n{data['description'][:100]}..."
                
            net.add_node(n, label=label, title=title, color=color, group=communities.get(n, 0) if communities else 0)

    for u, v, data in graph.edges(data=True):
        if u in net.get_nodes() and v in net.get_nodes():
            net.add_edge(u, v)

    # Inject custom search javascript
    # This is a bit hacky but standard for adding search to PyVis
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.save_graph(output_path)
    
    # Post-process HTML to add search bar
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()
        
    search_script = """
    <div id="search-container" style="position: absolute; top: 10px; left: 10px; z-index: 1000;">
        <input type="text" id="node-search" placeholder="Search entities..." onkeyup="searchNodes()">
        <div id="search-results" style="max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd;"></div>
    </div>
    <script type="text/javascript">
        function searchNodes() {
            var input = document.getElementById("node-search");
            var filter = input.value.toUpperCase();
            var nodes = network.body.data.nodes.get();
            var resultsDiv = document.getElementById("search-results");
            resultsDiv.innerHTML = "";
            
            if (filter === "") return;
            
            var matched = nodes.filter(n => n.label && n.label.toUpperCase().includes(filter));
            
            matched.forEach(node => {
                var div = document.createElement("div");
                div.style.padding = "5px";
                div.style.cursor = "pointer";
                div.innerText = node.label;
                div.onclick = function() {
                    network.focus(node.id, {
                        scale: 1.5,
                        animation: true
                    });
                };
                resultsDiv.appendChild(div);
            });
        }
    </script>
    """
    
    # Insert before </body>
    if "</body>" in html:
        html = html.replace("</body>", f"{search_script}</body>")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
    logger.info(f"Saved interactive explorer to {output_path}")

def plot_node2vec_impact(
    results_history: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Generate and save a plot comparing Baseline vs Node2Vec modularity.
    """
    try:
        if len(results_history) < 2:
            logger.info("Skipping Node2Vec plot: need at least 2 iterations.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        df = pd.DataFrame(results_history)
        iterations = df['iteration']
        modularity = df['modularity']
        baseline = df['modularity_baseline']
        
        plt.plot(iterations, modularity, marker='o', label='Node2Vec Modularity', linewidth=2, color='#1f77b4')
        plt.plot(iterations, baseline, marker='s', label='Baseline Modularity', linewidth=2, linestyle='--', color='#ff7f0e')
        
        plt.title('Impact of Node2Vec on Community Modularity', fontsize=16)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Modularity', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(iterations)
        
        # Add annotations for delta
        for i, (m, b) in enumerate(zip(modularity, baseline)):
            delta = m - b
            plt.annotate(f'{delta:+.3f}', 
                         xy=(iterations[i], m), 
                         xytext=(0, 10), 
                         textcoords='offset points',
                         ha='center', 
                         color='green' if delta > 0 else 'red',
                         fontsize=9)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Node2Vec impact plot to {output_path}")
        
    except Exception:
        logger.exception("Failed to plot Node2Vec impact.")
