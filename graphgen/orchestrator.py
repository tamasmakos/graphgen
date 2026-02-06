"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import asyncio
import uuid
import logging
import networkx as nx
from typing import Dict, Any, List

from graphgen.types import PipelineContext
from graphgen.config.settings import PipelineSettings
from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
from graphgen.pipeline.lexical_graph_building.builder import build_lexical_graph
from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphgen.pipeline.entity_relation.extractors import BaseExtractor
from graphgen.pipeline.graph_cleaning.pruning import prune_graph
from graphgen.utils.utils import create_output_directory
from graphgen.utils.schema_utils import save_graph_schema
# from graphgen.utils.health import check_falkordb

logger = logging.getLogger(__name__)

class KnowledgePipeline:
    """
    The main pipeline orchestrator.
    
    It accepts all necessary dependencies (infrastructure, configuration) via the constructor.
    It does NOT instantiate heavy objects itself.
    """

    def __init__(
        self, 
        settings: PipelineSettings,
        uploader: Neo4jGraphUploader,
        extractor: Any = None
    ):
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        self.run_id = str(uuid.uuid4())[:8]
        
    async def run(self):
        """
        Execute the full knowledge graph generation pipeline:
        1. Build Lexical Graph from Input Dir
        2. Extract Entities/Relations
        3. Semantic Enrichment (Embeddings, Similarity, Resolution)
        3.5. KGE Training (Optional - for weighted community detection)
        4. Community Detection & Summarization
        4.5. Topic Analysis (Statistical tests on community embeddings)
        5. Pruning
        6. Upload to Graph Database
        7. Save Artifacts to Disk
        """
        logger.info(f"🚀 Starting KnowledgePipeline run [{self.run_id}]...")
        
        # Preflight Checks
        self._run_preflight_checks()

        # 0. Initialize Pipeline Context (The "Bus")
        graph = nx.DiGraph()
        # PipelineContext (aliased as AgentDependencies) holds the state
        ctx = PipelineContext(graph=graph)
        
        # Convert settings to dict for legacy functions
        # TODO: Refactor downstream functions to accept PipelineSettings object directly
        config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
        
        try:
            # 1. Build Lexical Graph
            from graphgen.config.schema import GraphSchema
            # Parse schema from settings
            schema = None
            if self.settings.schema_config:
                 try:
                     schema = GraphSchema(**self.settings.schema_config)
                 except Exception as ex:
                     logger.warning(f"Failed to parse schema: {ex}")

            await self._step_lexical_graph(ctx, config_dict, schema=schema)

            # 2. Extraction
            await self._step_extraction(ctx, config_dict)

            # 3. Semantic Enrichment
            await self._step_enrichment(ctx)

            # 3.5. KGE Training (Optional)
            await self._step_kge_training(ctx)

            # 4. Community Detection & Summarization
            await self._step_communities(ctx, config_dict)

            # 4.5. Topic Separation Analysis
            await self._step_topic_analysis(ctx)

            # 5. Pruning
            await self._step_pruning(ctx)
            
            # 6. Upload
            await self._step_upload(ctx)

            # 7. Save Artifacts
            self._step_save_artifacts(ctx)
        
        except Exception as e:
            logger.critical(f"🔥 Pipeline [{self.run_id}] failed: {e}", exc_info=True)
            raise
        
        logger.info(f"✅ Pipeline Run [{self.run_id}] Finished Successfully.")

    def _run_preflight_checks(self):
        """Check external dependencies."""
        logger.info("Performing preflight health checks...")
        
        # Basic check via uploader connectivity
        if self.uploader and not self.uploader.connect():
             error_msg = f"Preflight check failed: Neo4j is not reachable."
             logger.critical(f"{error_msg} Aborting pipeline.")
             raise ConnectionError(error_msg)
        if self.uploader:
            self.uploader.close() # Close after check

    async def _step_lexical_graph(self, ctx: PipelineContext, config: Dict[str, Any], schema: Any = None):
        input_dir = self.settings.infra.input_dir
        logger.info(f"Step 1: Building Lexical Graph from {input_dir}")
        
        results = await build_lexical_graph(ctx, input_dir, config, schema=schema)
        
        ctx.stats['lexical'] = results
        logger.info(f"Lexical Graph Built: {results.get('documents_processed')} docs, {results.get('total_segments')} segments")

    async def _step_extraction(self, ctx: PipelineContext, config: Dict[str, Any]):
        if not self.extractor:
             logger.warning("Step 2: Skipped (No extractor provided).")
             return

        logger.info("Step 2: Extracting Entities & Relations...")
        extract_results = await extract_all_entities_relations(ctx, config, extractor=self.extractor)
        
        ctx.stats['extraction'] = extract_results
        logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")

    async def _step_enrichment(self, ctx: PipelineContext):
        try:
            from graphgen.pipeline.embeddings.rag import generate_rag_embeddings
            from graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
            
            logger.info("Step 3: Semantic Enrichment")
            
            logger.info("  3.1: Generating RAG Embeddings...")
            generate_rag_embeddings(ctx.graph)
            
            logger.info("  3.2: Semantic Resolution...")
            resolve_entities_semantically(ctx.graph)
            
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            ctx.add_error("enrichment", str(e))

    async def _step_kge_training(self, ctx: PipelineContext):
        """Train PyKeen KGE and add edge weights for community detection."""
        if not self.settings.kge.enabled:
            logger.info("Step 3.5: KGE Training (Skipped - disabled in config)")
            return
        
        try:
            from graphgen.pipeline.embeddings.kge import (
                train_global_kge, 
                compute_edge_weights_from_kge,
                store_embeddings_in_graph
            )
            
            logger.info("Step 3.5: KGE Training")
            
            # Train KGE model
            embeddings = train_global_kge(ctx.graph, self.settings.kge)
            
            if embeddings:
                # Compute edge weights from embeddings
                edges_weighted = compute_edge_weights_from_kge(ctx.graph, embeddings)
                
                # Store embeddings in graph nodes
                nodes_updated = store_embeddings_in_graph(ctx.graph, embeddings)
                
                ctx.stats['kge'] = {
                    'entities_embedded': len(embeddings),
                    'edges_weighted': edges_weighted,
                    'nodes_updated': nodes_updated
                }
                logger.info(f"KGE Complete: {len(embeddings)} embeddings, {edges_weighted} weighted edges")
            else:
                logger.warning("KGE training produced no embeddings")
                ctx.stats['kge'] = {'entities_embedded': 0}
                
        except Exception as e:
            logger.error(f"KGE training failed: {e}")
            ctx.add_error("kge", str(e))

    async def _step_communities(self, ctx: PipelineContext, config: Dict[str, Any]):
        try:
            from graphgen.pipeline.community.detection import CommunityDetector
            from graphgen.pipeline.community.subcommunities import add_enhanced_community_attributes_to_graph
            from graphgen.pipeline.summarization.core import generate_community_summaries
            from graphgen.config.llm import get_langchain_llm
            
            logger.info("Step 4: Community Detection & Summarization")
            
            logger.info("  4.1: Detecting Communities...")
            detector = CommunityDetector(self.settings.community)
            comm_results = detector.detect_communities(ctx.graph)
            communities = comm_results['assignments']
            
            # Save stats
            ctx.stats['communities'] = comm_results
            
            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)
            
            logger.info("  4.2: Generating Summaries...")
            llm = get_langchain_llm(config, purpose='summarization')
            summary_stats = await generate_community_summaries(ctx.graph, llm)
            ctx.stats['summarization'] = summary_stats
            
        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")
            ctx.add_error("communities", str(e))

    async def _step_topic_analysis(self, ctx: PipelineContext):
        """Run statistical tests on topic embeddings."""
        if not self.settings.analysis.topic_separation_test:
            logger.info("Step 4.5: Topic Analysis (Skipped - disabled in config)")
            return
        
        try:
            from graphgen.pipeline.analysis.topic_separation import generate_topic_separation_report
            
            logger.info("Step 4.5: Topic Separation Analysis")
            
            # Generate statistical report
            output_path = os.path.join(
                self.settings.infra.output_dir, 
                self.settings.analysis.output_file
            )
            
            report = generate_topic_separation_report(
                ctx.graph, 
                output_path, 
                self.settings.analysis
            )
            
            ctx.stats['topic_analysis'] = {
                'output_file': output_path,
                'community_silhouette': report.get('community_level', {}).get('silhouette_score') if report.get('community_level') else None,
                'subcommunity_silhouette': report.get('subcommunity_level', {}).get('silhouette_score') if report.get('subcommunity_level') else None,
                'overall_interpretation': report.get('overall_interpretation', 'N/A')
            }
            
            logger.info(f"Topic Analysis Complete: {report.get('overall_interpretation', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            ctx.add_error("topic_analysis", str(e))

    async def _step_pruning(self, ctx: PipelineContext):
        logger.info("Step 5: Pruning Graph...")
        prune_stats = prune_graph(ctx.graph, {'pruning_threshold': 0.01})
        ctx.stats['pruning'] = prune_stats
        logger.info(f"Pruning Stats: {prune_stats}")

    async def _step_upload(self, ctx: PipelineContext):
        if not self.uploader:
            return
            
        settings = self.settings
        db_type = settings.infra.graph_db_type if hasattr(settings.infra, 'graph_db_type') else "falkordb"
        logger.info(f"Step 6: Uploading to {db_type}...")
        try:
            if self.uploader.connect():
                stats = self.uploader.upload(ctx.graph, clean_database=settings.infra.clean_start)
                ctx.stats['upload'] = stats
                logger.info(f"Upload Stats: {stats}")
                self.uploader.close()
            else:
                logger.warning("Uploader could not connect.")
                ctx.add_error("upload", "Could not connect")
        except Exception as e:
             logger.error(f"Upload failed: {e}")
             ctx.add_error("upload", str(e))

    def _step_save_artifacts(self, ctx: PipelineContext):
        output_dir = self.settings.infra.output_dir
        logger.info(f"Step 7: Saving artifacts to {output_dir}")
        create_output_directory(output_dir)
        
        try:
            save_graph_schema(ctx.graph, output_dir)
            
            # Save GraphML
            graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
            clean_graph = ctx.graph.copy()
            
            # Serialize complex types for GraphML
            import json
            from datetime import date, datetime
            
            for _, d in clean_graph.nodes(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()
                        
            for _, _, d in clean_graph.edges(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()
            
            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            ctx.add_error("artifacts", str(e))
