"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import asyncio
import uuid
import logging
from datetime import datetime
import networkx as nx
from typing import Dict, Any, List

from graphgen.data_types import PipelineContext
from graphgen.config.settings import PipelineSettings
from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
from graphgen.pipeline.lexical_graph_building.builder import build_lexical_graph
from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphgen.pipeline.entity_relation.extractors import BaseExtractor
from graphgen.pipeline.graph_cleaning.pruning import prune_graph
from graphgen.utils.utils import create_output_directory
from graphgen.utils.schema_utils import save_graph_schema
from graphgen.utils.provenance import (
    write_analysis_run_manifest,
    write_pipeline_config_snapshot,
)

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
        
    async def run(self) -> None:
        """
        Execute the full knowledge graph generation pipeline:
        1. Build Lexical Graph from Input Dir
        2. Extract Entities/Relations
        3. Semantic Enrichment (Embeddings, Similarity, Resolution)
        4. Community Detection & Summarization
        5. Topic Analysis (Statistical tests)
        6. Pruning
        7. Upload to Graph Database
        8. Save Artifacts to Disk
        """
        logger.info(f"Starting KnowledgePipeline run [{self.run_id}]...")
        self.run_started_at = datetime.now()
        output_dir = self.settings.infra.output_dir
        create_output_directory(output_dir)

        thesis_output_dir = os.path.join(output_dir, self.settings.analytics.outputs_subdir)
        create_output_directory(thesis_output_dir)
        if self.settings.analytics.save_provenance:
            write_pipeline_config_snapshot(thesis_output_dir, self.settings)
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="started",
                run_id=self.run_id,
                started_at=self.run_started_at.isoformat(),
            )
        
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
            schema = GraphSchema(**self.settings.schema_config)

            await self._step_lexical_graph(ctx, config_dict, schema=schema)

            # 2. Extraction
            await self._step_extraction(ctx, config_dict)

            # 3. Semantic Enrichment
            await self._step_enrichment(ctx)

            # 4. Community Detection & Summarization
            await self._step_communities(ctx, config_dict)

            # 4.5. Thesis Analytics
            await self._step_thesis_analytics(ctx)

            # 5. Pruning
            await self._step_pruning(ctx)
            
            # 6. Upload
            await self._step_upload(ctx)

            # 7. Save Artifacts
            self._step_save_artifacts(ctx)
        
        except Exception as e:
            logger.critical(f"Pipeline [{self.run_id}] failed: {e}", exc_info=True)
            if self.settings.analytics.save_provenance:
                write_analysis_run_manifest(
                    thesis_output_dir,
                    self.settings,
                    stage="failed",
                    run_id=self.run_id,
                    started_at=self.run_started_at.isoformat(),
                    completed_at=datetime.now().isoformat(),
                    extra={"error": str(e)},
                )
            raise
        
        if self.settings.analytics.save_provenance:
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="completed",
                run_id=self.run_id,
                started_at=self.run_started_at.isoformat(),
                completed_at=datetime.now().isoformat(),
                extra={
                    "stats": ctx.stats,
                    "graph": {
                        "nodes": ctx.graph.number_of_nodes(),
                        "edges": ctx.graph.number_of_edges(),
                    },
                },
            )

        logger.info(f"Pipeline run [{self.run_id}] finished successfully.")

    def _run_preflight_checks(self) -> None:
        """Check external dependencies."""
        logger.debug("Performing preflight health checks...")
        
        # Basic check via uploader connectivity
        if self.uploader and not self.uploader.connect():
            error_msg = "Preflight check failed: Neo4j is not reachable."
            logger.critical(f"{error_msg} Aborting pipeline.")
            raise ConnectionError(error_msg)
        if self.uploader:
            self.uploader.close() # Close after check

    async def _step_lexical_graph(
        self,
        ctx: PipelineContext,
        config: Dict[str, Any],
        schema: Any = None,
    ) -> None:
        input_dir = self.settings.infra.input_dir
        logger.info(f"Step 1: Building Lexical Graph from {input_dir}")
        
        results = await build_lexical_graph(ctx, input_dir, config, schema=schema)
        
        ctx.stats['lexical'] = results
        logger.info(f"Lexical Graph Built: {results.get('documents_processed')} docs, {results.get('total_segments')} segments")

    async def _step_extraction(self, ctx: PipelineContext, config: Dict[str, Any]) -> None:
        if not self.extractor:
            logger.warning("Step 2: Skipped (No extractor provided).")
            return

        logger.debug("Step 2: Extracting Entities & Relations...")
        extract_results = await extract_all_entities_relations(ctx, config, extractor=self.extractor)
        
        ctx.stats['extraction'] = extract_results
        logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")

    async def _step_enrichment(self, ctx: PipelineContext) -> None:
        try:
            from graphgen.pipeline.embeddings.rag import generate_rag_embeddings
            from graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
            
            logger.debug("Step 3: Semantic Enrichment")
            
            logger.info("  3.1: Generating RAG Embeddings...")
            generate_rag_embeddings(ctx.graph)
            
            logger.info("  3.2: Semantic Resolution...")
            resolution_stats = resolve_entities_semantically(ctx.graph)
            ctx.stats['entity_resolution'] = resolution_stats
            logger.info(f"Resolution Stats: {resolution_stats}")
            
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            ctx.add_error("enrichment", str(e))

    async def _step_communities(self, ctx: PipelineContext, config: Dict[str, Any]) -> None:
        try:
            from graphgen.pipeline.community.detection import CommunityDetector
            from graphgen.pipeline.community.subcommunities import add_enhanced_community_attributes_to_graph
            from graphgen.pipeline.summarization.core import generate_community_summaries
            from graphgen.config.llm import get_langchain_llm

            logger.debug("Step 4: Community Detection & Summarization")

            detector = CommunityDetector(self.settings.community)

            # --- 4.1a: Baseline (unweighted) Leiden ---
            logger.info("  4.1: Detecting communities (unweighted baseline)...")
            baseline_graph = ctx.graph.copy()
            for _, _, d in baseline_graph.edges(data=True):
                if d.get("graph_type") == "entity_relation":
                    d["weight"] = 1.0
            baseline_res = detector.detect_communities(baseline_graph)
            modularity_baseline = baseline_res.get("modularity", 0.0)
            logger.info("  Baseline modularity: %.4f", modularity_baseline)

            # --- 4.1b: Node2Vec-weighted Leiden (optional) ---
            modularity_weighted = None
            seed_stability = None
            if self.settings.community.node2vec_enabled:
                from graphgen.pipeline.embeddings.node2vec_wrapper import compute_node2vec_weights

                logger.info("  4.1: Computing Node2Vec weights...")
                weights = compute_node2vec_weights(
                    ctx.graph,
                    dimensions=self.settings.community.node2vec_dimensions,
                    walk_length=self.settings.community.node2vec_walk_length,
                    num_walks=self.settings.community.node2vec_num_walks,
                    seed=self.settings.community.seed or 42,
                )
                weighted_count = 0
                for (u, v), w in weights.items():
                    if ctx.graph.has_edge(u, v):
                        ctx.graph[u][v]["weight"] = w
                        weighted_count += 1
                logger.info("  Applied Node2Vec weights to %d edges.", weighted_count)

                comm_results = detector.detect_communities(ctx.graph)
                modularity_weighted = comm_results.get("modularity", 0.0)
                logger.info(
                    "  Node2Vec modularity: %.4f  (delta=%+.4f)",
                    modularity_weighted,
                    modularity_weighted - modularity_baseline,
                )

                # --- 4.1c: Seed-stability significance test ---
                # Run Leiden many times with different seeds on both the
                # unweighted-baseline and the Node2Vec-weighted graph, then
                # test whether the uplift exceeds seed-to-seed variance.
                from graphgen.analytics.metrics import compare_modularity_distributions

                n_runs = 20
                baseline_dist = detector.modularity_distribution(baseline_graph, n_runs=n_runs)
                weighted_dist = detector.modularity_distribution(ctx.graph, n_runs=n_runs)
                seed_stability = compare_modularity_distributions(baseline_dist, weighted_dist)
                seed_stability["baseline_distribution"] = baseline_dist
                seed_stability["weighted_distribution"] = weighted_dist
                logger.info(
                    "  Seed-stability (n=%d/condition): baseline=%.4f±%.4f  weighted=%.4f±%.4f  "
                    "Mann-Whitney p=%s  Cohen's d=%s",
                    n_runs,
                    seed_stability.get("baseline_mean", float("nan")),
                    seed_stability.get("baseline_std", float("nan")),
                    seed_stability.get("weighted_mean", float("nan")),
                    seed_stability.get("weighted_std", float("nan")),
                    seed_stability.get("p_value"),
                    seed_stability.get("cohens_d"),
                )
            else:
                comm_results = baseline_res

            del baseline_graph

            communities = comm_results["assignments"]

            ctx.stats["communities"] = {
                **comm_results,
                "modularity_baseline": modularity_baseline,
                "modularity_weighted": modularity_weighted,
                "seed_stability": seed_stability,
            }

            # --- 4.2: Subcommunities & graph attributes ---
            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)

            # --- 4.3: Summarization ---
            logger.info("  4.2: Generating summaries...")
            llm = get_langchain_llm(config, purpose="summarization")
            summary_stats = await generate_community_summaries(ctx.graph, llm)
            ctx.stats["summarization"] = summary_stats

        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")
            ctx.add_error("communities", str(e))

    async def _step_thesis_analytics(self, ctx: PipelineContext) -> None:
        """Run thesis analytics: modularity comparison, permutation test, similarity distribution."""
        if not self.settings.analytics.enabled:
            logger.info("Step 4.5: Thesis Analytics (Skipped — set ANALYTICS_ENABLED=true to enable)")
            return

        try:
            from graphgen.analytics.reporting import generate_thesis_analytics_report

            logger.info("Step 4.5: Thesis Analytics")

            comm_stats = ctx.stats.get("communities", {})
            modularity_baseline = comm_stats.get("modularity_baseline", 0.0)
            modularity_weighted = comm_stats.get("modularity_weighted")

            # ctx.graph carries Node2Vec weights when node2vec_enabled=True;
            # pass None for weighted_graph when Node2Vec was not used so the
            # permutation test is correctly skipped.
            weighted_graph = ctx.graph if self.settings.community.node2vec_enabled else None

            output_path = os.path.join(
                self.settings.infra.output_dir,
                self.settings.analytics.outputs_subdir,
                "thesis_analytics_report.json",
            )

            report = generate_thesis_analytics_report(
                graph=ctx.graph,
                baseline_modularity=modularity_baseline,
                weighted_modularity=modularity_weighted,
                weighted_graph=weighted_graph,
                output_path=output_path,
                seed_stability=comm_stats.get("seed_stability"),
            )

            sim_dist = report.get("similarity_distribution") or {}
            perm = report.get("permutation_test") or {}
            ctx.stats["thesis_analytics"] = {
                "output_file": output_path,
                "modularity_baseline": modularity_baseline,
                "modularity_weighted": modularity_weighted,
                "similarity_mean": sim_dist.get("mean"),
                "similarity_normality_p": (
                    sim_dist.get("normality_test", {}).get("shapiro_wilk_p_value")
                ),
                "permutation_p_value": perm.get("p_value"),
            }

            logger.info(
                "Thesis Analytics: similarity_mean=%.3f  permutation_p=%.4f",
                sim_dist.get("mean") or 0.0,
                perm.get("p_value") or float("nan"),
            )

            plots_dir = os.path.join(
                self.settings.infra.output_dir,
                self.settings.analytics.outputs_subdir,
            )

            # --- Scale-free plot ---
            sf_result = report.get("scale_free") or {}
            degree_sequence = report.get("_scale_free_degree_sequence")
            if degree_sequence and not sf_result.get("skipped"):
                import numpy as np
                from graphgen.analytics.visualizer import plot_scale_free

                sf_plot_path = os.path.join(plots_dir, "scale_free_degree_distribution.png")
                plot_scale_free(
                    degree_sequence=degree_sequence,
                    alpha=sf_result.get("alpha", 2.0),
                    xmin=sf_result.get("xmin", 1),
                    log_log_r_squared=sf_result.get("log_log_r_squared", 0.0),
                    output_path=sf_plot_path,
                    ks_p_value=sf_result.get("ks_p_value"),
                    is_scale_free=sf_result.get("is_scale_free"),
                    n_nodes=sf_result.get("n_nodes"),
                    ks_distance=sf_result.get("ks_distance"),
                    distribution_comparison=sf_result.get("distribution_comparison"),
                )
                logger.info("Scale-free plot saved to %s", sf_plot_path)

            # --- Node2Vec uplift plot ---
            null_distribution = report.get("_null_distribution")
            if (
                null_distribution is not None
                and len(null_distribution) >= 2
                and modularity_weighted is not None
            ):
                import numpy as np
                from graphgen.analytics.visualizer import plot_node2vec_uplift

                uplift_path = os.path.join(plots_dir, "node2vec_modularity_uplift.png")
                seed_stab = report.get("seed_stability") or {}
                baseline_dist = report.get("_seed_stability_baseline")
                weighted_dist = report.get("_seed_stability_weighted")
                plot_node2vec_uplift(
                    baseline_modularity=modularity_baseline,
                    weighted_modularity=modularity_weighted,
                    null_distribution=np.array(null_distribution),
                    output_path=uplift_path,
                    p_value=perm.get("p_value"),
                    n_permutations=perm.get("n_permutations"),
                    null_95th=perm.get("null_95th"),
                    null_99th=perm.get("null_99th"),
                    baseline_distribution=np.array(baseline_dist) if baseline_dist else None,
                    weighted_distribution=np.array(weighted_dist) if weighted_dist else None,
                    seed_stability_p=seed_stab.get("p_value"),
                )
                logger.info("Node2Vec uplift plot saved to %s", uplift_path)

            # --- Topic similarity distribution plot ---
            import numpy as np
            from graphgen.analytics.visualizer import plot_similarity_distribution

            pairwise_values = report.get("_pairwise_values")
            if pairwise_values is not None:
                sim_plot_path = os.path.join(plots_dir, "topic_similarity_distribution.png")
                normality_p = sim_dist.get("normality_test", {}).get("shapiro_wilk_p_value")
                plot_similarity_distribution(
                    pairwise_values=np.array(pairwise_values),
                    output_path=sim_plot_path,
                    mean=sim_dist.get("mean", 0.0),
                    normality_p_value=normality_p,
                )
                logger.info("Topic similarity distribution plot saved to %s", sim_plot_path)

            # --- Subtopic similarity distribution plot ---
            sub_pairwise_values = report.get("_subcommunity_pairwise_values")
            if sub_pairwise_values is not None:
                sub_dist = report.get("subcommunity_similarity_distribution") or {}
                sub_sim_plot_path = os.path.join(
                    plots_dir, "subtopic_similarity_distribution.png"
                )
                sub_normality_p = sub_dist.get("normality_test", {}).get(
                    "shapiro_wilk_p_value"
                )
                plot_similarity_distribution(
                    pairwise_values=np.array(sub_pairwise_values),
                    output_path=sub_sim_plot_path,
                    mean=sub_dist.get("mean", 0.0),
                    normality_p_value=sub_normality_p,
                    title="Subtopic Similarity Distribution",
                )
                logger.info("Subtopic similarity distribution plot saved to %s", sub_sim_plot_path)

            # --- Heatmap + interactive explorer ---
            communities = ctx.stats.get("communities", {}).get("assignments", {})
            from graphgen.analytics.reporting import extract_topic_embeddings
            from graphgen.analytics.visualizer import (
                plot_topic_heatmap,
                generate_interactive_explorer,
            )

            # Use the same extractor used by the similarity distribution so
            # that the member-entity fallback is applied consistently.
            all_embeddings = extract_topic_embeddings(
                ctx.graph, levels=["COMMUNITY", "SUBCOMMUNITY"]
            )
            topic_embs = all_embeddings.get("COMMUNITY", {})
            subtopic_embs = all_embeddings.get("SUBCOMMUNITY", {})

            if topic_embs:
                plot_topic_heatmap(
                    topic_embs,
                    {k: k for k in topic_embs},
                    os.path.join(plots_dir, "topic_similarity_heatmap.png"),
                    title="Topic Similarity Heatmap",
                )
                logger.info("Topic heatmap saved to %s", plots_dir)

            if subtopic_embs:
                plot_topic_heatmap(
                    subtopic_embs,
                    {k: k for k in subtopic_embs},
                    os.path.join(plots_dir, "subtopic_similarity_heatmap.png"),
                    title="Subtopic Similarity Heatmap",
                )
                logger.info("Subtopic heatmap saved to %s", plots_dir)

            if communities:
                generate_interactive_explorer(
                    ctx.graph,
                    os.path.join(plots_dir, "interactive_graph.html"),
                    communities,
                )
                logger.info("Interactive explorer saved to %s/interactive_graph.html", plots_dir)

            # --- CSV exports: topics, communities, top entities by centrality ---
            from graphgen.analytics.reporting import export_csv_artifacts

            csv_paths = export_csv_artifacts(ctx.graph, plots_dir)
            ctx.stats["thesis_analytics"]["csv_exports"] = csv_paths

            # --- Quantitative ground-truth (EPRS) alignment ---
            try:
                from graphgen.analytics.ground_truth import (
                    align_communities_to_ground_truth,
                    export_ground_truth_alignment,
                    load_ground_truth_themes,
                )
                from graphgen.analytics.visualizer import plot_ground_truth_alignment

                gt_path = getattr(self.settings.analytics, "ground_truth_themes_path", None)
                themes = load_ground_truth_themes(gt_path)
                alignment = align_communities_to_ground_truth(
                    ctx.graph,
                    themes=themes,
                    embed_model_name=self.settings.embedding.model_name,
                )
                if alignment and not alignment.get("skipped"):
                    export_ground_truth_alignment(alignment, plots_dir)
                    plot_ground_truth_alignment(
                        alignment["similarity_matrix"],
                        alignment["community_labels"],
                        alignment["theme_labels"],
                        os.path.join(plots_dir, "ground_truth_alignment_heatmap.png"),
                    )
                    ctx.stats["thesis_analytics"]["ground_truth"] = alignment.get("metrics")
                else:
                    logger.info(
                        "Ground-truth alignment skipped: %s",
                        (alignment or {}).get("skipped"),
                    )
            except Exception:
                logger.exception("Ground-truth alignment failed.")

            # --- Community & subcommunity size distribution ---
            from graphgen.analytics.visualizer import plot_community_sizes

            community_entity_counts: dict = {}
            subcommunity_entity_counts: dict = {}
            for n, data in ctx.graph.nodes(data=True):
                node_type = str(data.get("node_type", "")).upper()
                raw_count = data.get("entity_count", 0)
                try:
                    count = int(raw_count) if raw_count is not None else 0
                except (ValueError, TypeError):
                    count = 0
                if node_type in ("TOPIC", "COMMUNITY"):
                    community_entity_counts[n] = count
                elif node_type in ("SUBTOPIC", "SUBCOMMUNITY"):
                    subcommunity_entity_counts[n] = count

            if community_entity_counts:
                size_plot_path = os.path.join(plots_dir, "community_size_distribution.png")
                plot_community_sizes(
                    community_sizes=community_entity_counts,
                    subcommunity_sizes=subcommunity_entity_counts,
                    output_path=size_plot_path,
                )
                logger.info("Community size distribution plot saved to %s", size_plot_path)

        except Exception as e:
            logger.error(f"Thesis analytics failed: {e}")
            ctx.add_error("thesis_analytics", str(e))

    async def _step_pruning(self, ctx: PipelineContext) -> None:
        logger.info("Step 5: Pruning Graph...")
        prune_stats = prune_graph(ctx.graph, self.settings.processing.model_dump())
        ctx.stats['pruning'] = prune_stats
        logger.info(f"Pruning Stats: {prune_stats}")

    async def _step_upload(self, ctx: PipelineContext) -> None:
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

    def _step_save_artifacts(self, ctx: PipelineContext) -> None:
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
            
            import numpy as _np

            def _serialize_value(v):
                if v is None:
                    return None
                if isinstance(v, _np.ndarray):
                    return json.dumps(v.tolist(), ensure_ascii=False)
                if isinstance(v, (dict, list)):
                    return json.dumps(v, ensure_ascii=False)
                if isinstance(v, (date, datetime)):
                    return v.isoformat()
                return v

            for _, d in clean_graph.nodes(data=True):
                for k, v in list(d.items()):
                    result = _serialize_value(v)
                    if result is None:
                        del d[k]
                    else:
                        d[k] = result

            for _, _, d in clean_graph.edges(data=True):
                for k, v in list(d.items()):
                    result = _serialize_value(v)
                    if result is None:
                        del d[k]
                    else:
                        d[k] = result
            
            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
            
            # Save Entity Resolution Report
            er_report_path = os.path.join(output_dir, "entity_resolution_report.json")
            er_stats = {
                "extraction": ctx.stats.get('extraction', {}),
                "entity_resolution": ctx.stats.get('entity_resolution', {}),
                "pruning": ctx.stats.get('pruning', {})
            }
            with open(er_report_path, 'w') as f:
                json.dump(er_stats, f, indent=2)
            logger.info(f"Entity Resolution Report saved to {er_report_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            ctx.add_error("artifacts", str(e))
