import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import networkx as nx

from graphgen.pipeline.entity_relation.extraction import (
    _extract_entities_for_chunk,
    enrich_graph_per_segment,
    process_extraction_task,
)
from graphgen.data_types import PipelineContext, ChunkExtractionTask


class GLiNER2IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gliner2_backend_uses_candidate_schema_and_preserves_metadata(self):
        fake_model = Mock()
        fake_model.extract.return_value = {
            "entities": {
                "person": [
                    {"text": "Mario Draghi", "confidence": 0.99, "start": 0, "end": 12}
                ],
                "organization": [
                    {"text": "European Parliament", "confidence": 0.95, "start": 20, "end": 39}
                ],
            }
        }
        config = {
            "extraction": {
                "backend": "gliner2",
                "gliner_model": "fastino/gliner2-base-v1",
                "gliner_threshold": 0.25,
                "gliner2_top_k_labels": 2,
                "gliner2_label_descriptions": {
                    "PERSON": "A named person.",
                    "ORGANIZATION": "A named institution.",
                },
            }
        }
        labels = ["PERSON", "ORGANIZATION", "LOCATION"]

        with patch("graphgen.pipeline.entity_relation.extraction.get_gliner2_model", return_value=fake_model):
            entities = await _extract_entities_for_chunk(
                "Mario Draghi addressed the European Parliament.",
                config,
                labels,
                label_profiles=[
                    {
                        "label": "PERSON",
                        "aliases": ["person"],
                        "description": "A named person.",
                        "source": "config",
                    },
                    {
                        "label": "ORGANIZATION",
                        "aliases": ["organization"],
                        "description": "A named institution.",
                        "source": "config",
                    },
                    {
                        "label": "LOCATION",
                        "aliases": ["location"],
                        "description": "A named place.",
                        "source": "config",
                    },
                ],
            )

        self.assertEqual(entities[0]["text"], "MARIO_DRAGHI")
        self.assertEqual(entities[0]["label"], "PERSON")
        self.assertIn("ontology_label", entities[0])
        self.assertIn("confidence", entities[0])
        self.assertIn("candidate_labels", entities[0])
        self.assertLessEqual(len(entities[0]["candidate_labels"]), 2)
        fake_model.extract.assert_called_once()
        schema = fake_model.extract.call_args.args[1]
        self.assertEqual(schema["entities"]["person"], "A named person.")
        self.assertEqual(schema["entities"]["organization"], "A named institution.")

    async def test_process_extraction_task_forwards_label_profiles_to_ner_stage(self):
        ctx = PipelineContext()
        chunk_id = "chunk-profiles"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Harry met Hermione")
        extractor = Mock()
        extractor.extract_relations = Mock(return_value=([], [], {}))
        label_profiles = [
            {"label": "PERSON", "aliases": ["wizard"], "description": "A named wizard.", "source": "config"},
            {"label": "SPELL", "aliases": ["spell"], "description": "A magical spell.", "source": "config"},
        ]

        async def fake_ner(text, config, labels, return_diagnostics=False, label_profiles=None):
            self.assertEqual(labels, ["PERSON", "SPELL"])
            self.assertEqual(label_profiles, [
                {"label": "PERSON", "aliases": ["wizard"], "description": "A named wizard.", "source": "config"},
                {"label": "SPELL", "aliases": ["spell"], "description": "A magical spell.", "source": "config"},
            ])
            entities = [{
                "text": "HARRY",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.99,
                "candidate_labels": ["PERSON"],
            }]
            if return_diagnostics:
                return entities, {"backend": "gliner2"}
            return entities

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", side_effect=fake_ner), \
             patch("graphgen.pipeline.entity_relation.extraction.extract_relations_with_llm_async", return_value=([], [], {})):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                extractor,
                {"extraction": {"backend": "gliner2"}},
                ["PERSON", "SPELL"],
                label_profiles=label_profiles,
            )

        self.assertTrue(result["success"])


class ResolutionSignalPreservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_extraction_task_keeps_rich_gliner_entities_for_downstream_resolution(self):
        ctx = PipelineContext()
        chunk_id = "chunk-1"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Mario Draghi addressed Parliament")
        extractor = Mock()
        async def fake_extract_relations(*args, **kwargs):
            return ([], [])
        extractor.extract_relations = fake_extract_relations
        config = {"extraction": {"backend": "gliner2"}}
        fake_entities = [
            {
                "text": "MARIO_DRAGHI",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.98,
                "start": 0,
                "end": 12,
                "candidate_labels": ["PERSON", "ORGANIZATION"],
            }
        ]

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=fake_entities):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                extractor,
                config,
                ["PERSON", "ORGANIZATION"],
            )

        self.assertTrue(result["success"])
        stored = ctx.graph.nodes[chunk_id]["gliner_entities"][0]
        self.assertEqual(stored["ontology_label"], "PERSON")
        self.assertAlmostEqual(stored["confidence"], 0.98)
        self.assertEqual(stored["candidate_labels"], ["PERSON", "ORGANIZATION"])

    async def test_enrich_graph_per_segment_prefers_longer_person_surface_forms_from_gliner_entities(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        segment_id = "SEG_1"
        chunk_id = "CHUNK_1"
        ctx.graph.add_node(segment_id, node_type="SEGMENT")
        ctx.graph.add_node(
            chunk_id,
            node_type="CHUNK",
            raw_extraction={
                "relations": [("MR_DURSLEY", "DIRECTOR_OF", "GRUNNINGS", {"confidence": 1.0})],
                "nodes": [
                    {"id": "MR_DURSLEY", "type": "PERSON"},
                    {"id": "GRUNNINGS", "type": "ORGANIZATION"},
                ],
            },
            gliner_entities=[
                {
                    "text": "MR_DURSLEY",
                    "label": "PERSON",
                    "ontology_label": "PERSON",
                    "confidence": 0.99,
                    "candidate_labels": ["PERSON"],
                },
                {
                    "text": "MRS_DURSLEY",
                    "label": "PERSON",
                    "ontology_label": "PERSON",
                    "confidence": 0.98,
                    "candidate_labels": ["PERSON"],
                },
                {
                    "text": "DUDLEY",
                    "label": "PERSON",
                    "ontology_label": "PERSON",
                    "confidence": 0.95,
                    "candidate_labels": ["PERSON"],
                },
            ],
        )
        ctx.graph.add_edge(segment_id, chunk_id, label="HAS_CHUNK")
        ctx.stats["pipeline_config"] = {"extraction": {"backend": "gliner2"}}

        result = await enrich_graph_per_segment(ctx)

        self.assertEqual(result["segments_processed"], 1)
        self.assertIn("MR_DURSLEY", ctx.graph.nodes)
        self.assertIn("MRS_DURSLEY", ctx.graph.nodes)
        self.assertIn("DUDLEY", ctx.graph.nodes)
        self.assertNotIn("MR", ctx.graph.nodes)
        self.assertNotIn("MRS", ctx.graph.nodes)
        self.assertEqual(ctx.graph.nodes["MR_DURSLEY"].get("ontology_class"), "PERSON")
        self.assertEqual(ctx.graph.nodes["MRS_DURSLEY"].get("ontology_class"), "PERSON")
        self.assertEqual(ctx.graph.nodes["DUDLEY"].get("ontology_class"), "PERSON")

    async def test_enrich_graph_per_segment_suppresses_short_and_ambiguous_fragment_entities(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        segment_id = "SEG_2"
        chunk_id = "CHUNK_2"
        ctx.graph.add_node(segment_id, node_type="SEGMENT")
        ctx.graph.add_node(
            chunk_id,
            node_type="CHUNK",
            raw_extraction={"relations": [], "nodes": []},
            gliner_entities=[
                {"text": "MR", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.91, "candidate_labels": ["PERSON"]},
                {"text": "MRS", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.92, "candidate_labels": ["PERSON"]},
                {"text": "DURSLEY", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.88, "candidate_labels": ["PERSON"]},
                {"text": "MR_DURSLEY", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.99, "candidate_labels": ["PERSON"]},
                {"text": "MRS_DURSLEY", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.98, "candidate_labels": ["PERSON"]},
                {"text": "DUDLEY", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.95, "candidate_labels": ["PERSON"]},
            ],
        )
        ctx.graph.add_edge(segment_id, chunk_id, label="HAS_CHUNK")
        ctx.stats["pipeline_config"] = {"extraction": {"backend": "gliner2"}}

        result = await enrich_graph_per_segment(ctx)

        self.assertEqual(result["segments_processed"], 1)
        self.assertIn("MR_DURSLEY", ctx.graph.nodes)
        self.assertIn("MRS_DURSLEY", ctx.graph.nodes)
        self.assertIn("DUDLEY", ctx.graph.nodes)
        self.assertNotIn("MR", ctx.graph.nodes)
        self.assertNotIn("MRS", ctx.graph.nodes)
        self.assertNotIn("DURSLEY", ctx.graph.nodes)
