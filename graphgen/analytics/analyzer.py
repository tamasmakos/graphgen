import os
import logging
import json
from datetime import datetime
import networkx as nx
from typing import Dict, Any

from graphgen.analytics.metrics import (
    calculate_modularity, 
    calculate_topic_overlap, 
    evaluate_kge_model_quality
)
from graphgen.analytics.visualizer import (
    plot_topic_heatmap, 
    generate_interactive_explorer
)

logger = logging.getLogger(__name__)

class GraphAnalyzer:
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
        logger.info("Calculating modularity stats...")
        modularity = calculate_modularity(graph, communities)
        results['modularity'] = modularity
        logger.info(f"Modularity: {modularity:.4f}")
        
        # 2. Topic Analysis (if embeddings present)
        topic_embeddings = self._extract_topic_embeddings(graph)
        if topic_embeddings:
            logger.info("Calculating topic overlap...")
            overlap = calculate_topic_overlap(topic_embeddings)
            results['topic_overlap'] = overlap
            
            # Heatmap
            if self.config.get('visualization', {}).get('heatmap', True):
                topic_labels = self._extract_topic_labels(graph)
                plot_topic_heatmap(
                    topic_embeddings, 
                    topic_labels, 
                    os.path.join(self.output_dir, "topic_similarity_heatmap.png")
                )
                
        # 3. KGE Comparison (if enabled)
        kge_conf = self.config.get('kge_comparison', {})
        if kge_conf.get('enabled', False):
            logger.info("Running KGE model comparison...")
            models = kge_conf.get('models', ['TransE', 'DistMult'])
            kge_results = {}
            for model in models:
                logger.info(f"Evaluating KGE model: {model}")
                metrics = evaluate_kge_model_quality(
                    graph, 
                    model_name=model, 
                    epochs=kge_conf.get('epochs', 50)
                )
                kge_results[model] = metrics
            results['kge_comparison'] = kge_results
            
            # Save KGE results
            with open(os.path.join(self.output_dir, "kge_comparison.json"), 'w') as f:
                json.dump(kge_results, f, indent=2)

        # 4. Interactive Visualization
        if self.config.get('visualization', {}).get('interactive', True):
            logger.info("Generating interactive explorer...")
            generate_interactive_explorer(
                graph, 
                os.path.join(self.output_dir, "interactive_graph.html"),
                communities
            )
            
        # Save summary report
        with open(os.path.join(self.output_dir, "analysis_summary.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Graph analysis complete.")
        return results
        
    def _extract_topic_embeddings(self, graph: nx.DiGraph):
        embeddings = {}
        for n, data in graph.nodes(data=True):
            if data.get('node_type') == 'COMMUNITY' and 'embedding' in data:
                # Handle cases where embedding might be a list or numpy array
                emb = data['embedding']
                import numpy as np
                if isinstance(emb, list):
                    embeddings[n] = np.array(emb)
                else:
                    embeddings[n] = emb
        return embeddings

    def _extract_topic_labels(self, graph: nx.DiGraph):
        labels = {}
        for n, data in graph.nodes(data=True):
            if data.get('node_type') == 'COMMUNITY':
                labels[n] = data.get('title', n)
        return labels
