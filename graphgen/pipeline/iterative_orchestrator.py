import logging
import asyncio
import networkx as nx
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

from graphgen.config.settings import PipelineSettings
from graphgen.types import PipelineContext
from graphgen.pipeline.iterative_loader import IterativeLoader
from graphgen.pipeline.lexical_graph_building.builder import add_segments_to_graph
from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphgen.pipeline.community.detection import CommunityDetector
from graphgen.pipeline.community.subcommunities import add_enhanced_community_attributes_to_graph
from graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
from graphgen.pipeline.summarization.core import generate_community_summaries
from graphgen.pipeline.analysis.topic_separation import generate_topic_separation_report
from graphgen.utils.vector_embedder.rag import generate_rag_embeddings
from graphgen.config.llm import get_langchain_llm
from graphgen.config.schema import GraphSchema

logger = logging.getLogger(__name__)

class IterativeOrchestrator:
    def __init__(self, settings: PipelineSettings, uploader=None, extractor=None):
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        # Use file_pattern from extraction settings
        file_pattern = settings.extraction.file_pattern if settings.extraction else "*.txt"
        self.loader = IterativeLoader(settings.infra.input_dir, settings.iterative, file_pattern)
        self.results = []

    async def run(self):
        if not self.settings.iterative.enabled:
            logger.info("Iterative mode disabled.")
            return

        logger.info(f"Starting Iterative Experiment: {self.settings.iterative.iterations} iterations, batch size {self.settings.iterative.batch_size}")

        # Initialize cumulative graph
        graph = nx.DiGraph() 
        ctx = PipelineContext(graph=graph)
        
        # Parse schema
        schema = None
        if self.settings.schema_config:
             try:
                 schema = GraphSchema(**self.settings.schema_config)
             except Exception:
                 pass
        
        config_dict = self.settings.model_dump()

        for i in range(self.settings.iterative.iterations):
            logger.info(f"=== Iteration {i+1}/{self.settings.iterative.iterations} ===")
            
            # 1. Load Batch
            segments = self.loader.get_batch(i)
            if not segments:
                logger.warning("No segments loaded. Stopping.")
                break
                
            logger.info(f"Loaded batch of {len(segments)} segments.")

            # 2. Add to Graph (Lexical)
            by_file = defaultdict(list)
            for seg in segments:
                filename = seg.metadata.get('filename', f'batch_{i}')
                by_file[filename].append(seg)
            
            for filename, file_segments in by_file.items():
                doc_id = f"DOC_{filename}"
                # Ensure Doc node exists (idempotent)
                if not ctx.graph.has_node(doc_id):
                    ctx.graph.add_node(doc_id, node_type="Document", name=filename)
                
                await add_segments_to_graph(ctx, file_segments, doc_id, config_dict, schema)

            # 3. Extraction (for new chunks only? extract_all checks tasks)
            # extraction_tasks are appended by add_segments_to_graph
            await extract_all_entities_relations(ctx, config_dict, self.extractor)
            
            # Clear tasks after extraction to avoid re-extracting old chunks next iteration
            # But wait, ctx is cumulative. 
            # `extract_all_entities_relations` dedups tasks. 
            # But we should probably clear `ctx.extraction_tasks` to keep memory low and logic clean
            ctx.extraction_tasks = []

            # 4. Enrichment
            # Generate RAG embeddings for Entities (needed for resolution)
            generate_rag_embeddings(ctx.graph, node_types=['ENTITY_CONCEPT'])
            resolve_entities_semantically(ctx.graph, similarity_threshold=self.settings.processing.similarity_threshold)
            
            # 5. KGE & Edge Weighting (New)
            if self.settings.kge.enabled:
                from graphgen.pipeline.embeddings.kge import (
                    train_global_kge, 
                    compute_edge_weights_from_kge, 
                    store_embeddings_in_graph
                )
                
                logger.info("Training KGE model for community detection...")
                kge_embeddings = train_global_kge(ctx.graph, self.settings.kge)
                
                if kge_embeddings:
                    store_embeddings_in_graph(ctx.graph, kge_embeddings)
                    compute_edge_weights_from_kge(ctx.graph, kge_embeddings)
            
            # 6. Community Detection
            # Note: We re-run detection on the FULL graph every iteration
            detector = CommunityDetector(self.settings.community)
            comm_results = detector.detect_communities(ctx.graph)
            
            # --- VISUALIZATION (New) ---
            if self.settings.kge.enabled:
                from graphgen.pipeline.visualization.kge_plot import plot_kge_communities
                plot_path = f"iteration_{i+1}_kge_plot.png"
                plot_kge_communities(ctx.graph, comm_results['assignments'], plot_path)

            modularity = comm_results.get('modularity', 0.0)
            communities = comm_results['assignments']
            
            # 7. Summarization
            # Detect subcommunities
            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)
            
            # Generate Summaries (and Topic Embeddings)
            # Ideally we only summarize NEW or CHANGED communities?
            # But Leiden can change assignments globally.
            # For robustness, we re-summarize (or at least update embeddings).
            llm = get_langchain_llm(config_dict, purpose='summarization')
            await generate_community_summaries(ctx.graph, llm)
            
            # Ensure TOPIC nodes have embeddings (generate_community_summaries might generate text, but not embeddings explicitly if not in its logic? 
            # `update_community_node_with_summary_async` adds title/summary.
            # We need `generate_rag_embeddings` for TOPIC nodes after summary generation.
            generate_rag_embeddings(ctx.graph, node_types=['TOPIC', 'SUBTOPIC'])

            # 8. Graph Analytics
            if self.settings.analytics.enabled:
                 from graphgen.analytics.analyzer import GraphAnalyzer
                 analyzer = GraphAnalyzer(
                     config=self.settings.analytics.model_dump(), 
                     output_base_dir=self.settings.infra.output_dir
                 )
                 # Reconstruct communities mapping
                 communities = comm_results['assignments']
                 
                 results = analyzer.run_full_analysis(ctx.graph, communities)
                 
                 global_separation = results.get('modularity', 0.0) # Map to relevant metric or store all
                 topic_overlap = results.get('topic_overlap', 0.0)
            else:
                 # Minimal fallback if disabled but loop expects vars?
                 global_separation = 0.0
                 topic_overlap = 0.0
            
            result = {
                'iteration': i + 1,
                'cumulative_speeches': (i + 1) * self.settings.iterative.batch_size,
                'modularity': modularity,
                'topic_separation': global_separation,
                'topic_overlap': topic_overlap,
                'nodes': ctx.graph.number_of_nodes(),
                'edges': ctx.graph.number_of_edges(),
                'communities': comm_results.get('community_count', 0)
            }
            self.results.append(result)
            logger.info(f"Iteration {i+1} Stats: Modularity={modularity:.4f}, Separation={global_separation:.4f}, Overlap={topic_overlap:.4f}")
            
            # Optional: Upload intermediate?
            if self.uploader:
                 # maybe not upload every time for speed, unless requested
                 pass

        # Final Summary
        try:
            df = pd.DataFrame(self.results)
            output_csv = f"{self.settings.infra.output_dir}/iterative_experiment_results.csv"
            df.to_csv(output_csv, index=False)
            logger.info(f"Experiment complete. Results saved to {output_csv}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

        # Final Upload
        if self.uploader:
             db_type = self.settings.infra.graph_db_type
             logger.info(f"Uploading final iterative graph to {db_type}...")
             try:
                 if self.uploader.connect():
                     # We use clean_start from settings. 
                     # Since ctx.graph is cumulative, we can wipe and replace or just merge.
                     # Wiping ensures consistency with the in-memory graph.
                     self.uploader.upload(ctx.graph, clean_database=self.settings.infra.clean_start)
                     self.uploader.close()
                     logger.info("Upload complete.")
                 else:
                     logger.warning("Failed to connect to graph database for upload. Continuing without DB upload (output graphs saved locally).")
             except Exception as e:
                 logger.warning(f"Upload failed: {e}. Continuing without DB upload (output graphs saved locally).")
