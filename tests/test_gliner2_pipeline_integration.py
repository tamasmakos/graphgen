import asyncio
import unittest
from unittest.mock import Mock, patch

from graphgen.pipeline.entity_relation.extraction import (
    _extract_entities_for_chunk,
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
            )

        self.assertEqual(entities[0]["text"], "MARIO_DRAGHI")
        self.assertEqual(entities[0]["label"], "PERSON")
        self.assertIn("ontology_label", entities[0])
        self.assertIn("confidence", entities[0])
        self.assertIn("candidate_labels", entities[0])
        self.assertLessEqual(len(entities[0]["candidate_labels"]), 2)
        fake_model.extract.assert_called_once()


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
