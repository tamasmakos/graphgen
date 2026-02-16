"""Analytics runner for graph quality reports."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import networkx as nx
import numpy as np

from graphgen.analytics.metrics import (
    calculate_modularity, 
    calculate_topic_overlap
)
from graphgen.analytics.visualizer import (
    plot_topic_heatmap, 
    generate_interactive_explorer
)
from graphgen.analytics.reporting import generate_topic_separation_report
from graphgen.analytics.diversity import calculate_topic_diversity


logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """Run configured analytics and persist outputs."""

    def __init__(self, config: Dict[str, Any], output_base_dir: str):
        self.config = config
        self.enabled = config.get('enabled', False)
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            output_base_dir, 
            config.get('output_dir', 'analytics_reports'),
            timestamp
        )
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def run_full_analysis(self, graph: nx.DiGraph, communities: Dict[str, int]) -> Dict[str, Any]:
        """
        Execute all configured analysis steps.
        """
        if not self.enabled:
            logger.info("Analytics disabled.")
            return {}
            
        logger.info(f"Starting full graph analysis. Output: {self.output_dir}")
        results = {}
        
        # 1. Modularity & Basic Stats
        logger.info(
            "Calculating modularity stats (nodes=%d, communities=%d)...",
            graph.number_of_nodes(),
            len(communities),
        )
        modularity = calculate_modularity(graph, communities)
        results['modularity'] = modularity
        logger.info(f"Modularity: {modularity:.4f}")
        
        # 2. Topic Analysis (if embeddings present)
        topic_embeddings = self._extract_topic_embeddings(graph)
        if topic_embeddings:
            logger.info("Calculating topic overlap (embeddings=%d)...", len(topic_embeddings))
            overlap = calculate_topic_overlap(topic_embeddings)
            results['topic_overlap'] = overlap
            logger.info("Topic overlap: %.4f", overlap)

            # Heatmap
            if self.config.get('visualization', {}).get('heatmap', True):
                topic_labels = self._extract_topic_labels(graph)
                plot_topic_heatmap(
                    topic_embeddings,
                    topic_labels,
                    os.path.join(self.output_dir, "topic_similarity_heatmap.png")
                )

            # Raw overlap matrix export
            if self.config.get("save_raw_overlap_matrix", False):
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    labels = self._extract_topic_labels(graph)
                    label_ids = list(topic_embeddings.keys())
                    matrix = cosine_similarity(np.vstack([topic_embeddings[k] for k in label_ids]))
                    np.save(os.path.join(self.output_dir, "topic_similarity_matrix.npy"), matrix)
                    with open(os.path.join(self.output_dir, "topic_similarity_labels.json"), "w") as f:
                        json.dump({k: labels.get(k, k) for k in label_ids}, f, indent=2)
                    results["topic_similarity_matrix"] = "topic_similarity_matrix.npy"
                    results["topic_similarity_labels"] = "topic_similarity_labels.json"
                except Exception:
                    logger.exception("Failed to save raw overlap matrix.")
        else:
            logger.info("Skipping topic overlap: no topic embeddings available.")
                
        # 4. Interactive Visualization
        if self.config.get('visualization', {}).get('interactive', True):
            logger.info("Generating interactive explorer...")
            generate_interactive_explorer(
                graph, 
                os.path.join(self.output_dir, "interactive_graph.html"),
                communities
            )
            
        if self.config.get('calculate_diversity', False):
            # Extract top words for diversity
            # This requires access to top words which might be in node attributes
            top_words_list = []
            for n, data in graph.nodes(data=True):
                if data.get('node_type') == 'TOPIC':
                    # Assuming 'top_words' attribute exists or can be parsed from summary/description
                    # For now, placeholder or if extraction is possible
                    pass
            # diversity = calculate_topic_diversity(top_words_list)
            # results['topic_diversity'] = diversity

        # 3. Separation Report (if not already run by orchestrator, or if we want to re-run)
        if self.config.get('run_separation_report', False):
             sep_report = generate_topic_separation_report(
                 graph, 
                 os.path.join(self.output_dir, "topic_separation_report.json"),
                 self.config
             )
             results['separation'] = sep_report

        # Save summary report
        with open(os.path.join(self.output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Graph analysis complete.")
        return results
        
    def _extract_topic_embeddings(self, graph: nx.DiGraph) -> Dict[str, np.ndarray]:
        """Extract topic embeddings (from summary-based embedding when available)."""
        embeddings = {}
        for n, data in graph.nodes(data=True):
            node_type = data.get('node_type')
            if node_type not in ('COMMUNITY', 'TOPIC') or 'embedding' not in data:
                continue
            emb = data['embedding']
            if isinstance(emb, list):
                embeddings[n] = np.array(emb)
            else:
                embeddings[n] = emb
        return embeddings

    def _extract_topic_labels(self, graph: nx.DiGraph) -> Dict[str, str]:
        """Labels for topic nodes (title or node id)."""
        labels = {}
        for n, data in graph.nodes(data=True):
            if data.get('node_type') in ('COMMUNITY', 'TOPIC'):
                labels[n] = data.get('title') or data.get('name', n)
        return labels
