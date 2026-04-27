import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from graphgen.config.llm import get_langchain_llm
from graphgen.config.settings import PipelineSettings
from graphgen.config.settings import ExtractionSettings
from graphgen.data_types import ChunkExtractionTask, PipelineContext
from graphgen.orchestrator import KnowledgePipeline
from graphgen.pipeline.entity_relation.extraction import (
    _extract_entities_for_chunk,
    _build_relation_eligible_entities,
    extract_relations_with_llm_async,
    process_extraction_task,
)
from graphgen.utils.labels import resolve_entity_labels
from graphgen.pipeline.entity_relation.extractors import (
    DSPyExtractor,
    LangChainExtractor,
    _candidate_grounded_in_evidence,
    _endpoint_matches_hint,
    _is_grounded_relation_endpoint,
    _is_ungrounded_relation_triplet,
    _relation_endpoints_in_hints,
    get_extractor,
)
from graphgen.utils.diagnostics import diagnostics_enabled, write_diagnostic_json
from graphgen.pipeline.summarization.summarizer import DSPySummarizer
from graphgen.main import resolve_env_file


class MainEnvResolutionRegressionTests(unittest.TestCase):
    def test_resolve_env_file_prefers_repo_root_dotenv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            graphgen_dir = repo / "graphgen"
            graphgen_dir.mkdir()
            main_file = graphgen_dir / "main.py"
            main_file.write_text("# stub", encoding="utf-8")
            repo_env = repo / ".env"
            package_env = graphgen_dir / ".env"
            repo_env.write_text("GROQ_API_KEY=root\n", encoding="utf-8")
            package_env.write_text("GROQ_API_KEY=package\n", encoding="utf-8")

            with patch("graphgen.main.__file__", str(main_file)), patch.dict(os.environ, {}, clear=True):
                resolved = resolve_env_file()

            self.assertEqual(resolved, str(repo_env))

    def test_resolve_env_file_honors_override(self):
        with patch.dict(os.environ, {"GRAPHGEN_ENV_FILE": "/tmp/custom.env"}, clear=True):
            self.assertEqual(resolve_env_file(), "/tmp/custom.env")


class ExtractionRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gliner_predictions_are_flattened_before_normalization(self):
        class FakeModel:
            def inference(self, sentences, labels, threshold=0.5):
                return [
                    [{"text": "Emmanuel Macron", "label": "Person"}],
                    [{"text": "Brussels", "label": "Location"}],
                ]

        config = {"extraction": {"backend": "gliner", "gliner_threshold": 0.5}}
        with patch("graphgen.pipeline.entity_relation.extraction.get_gliner_model", return_value=FakeModel()):
            entities = await _extract_entities_for_chunk(
                "Emmanuel Macron visited Brussels.",
                config,
                ["PERSON", "LOCATION"],
            )

        extracted = [{"text": e["text"], "label": e["label"]} for e in entities]
        self.assertEqual(
            extracted,
            [
                {"text": "EMMANUEL_MACRON", "label": "PERSON"},
                {"text": "BRUSSELS", "label": "LOCATION"},
            ],
        )
        self.assertIn("ontology_label", entities[0])
        self.assertIn("confidence", entities[0])
        self.assertEqual(entities[0]["backend"], "gliner")


class LLMConfigRegressionTests(unittest.TestCase):
    def test_resolve_entity_labels_keeps_default_seed_when_ontology_enabled_without_manual_labels(self):
        labels = resolve_entity_labels({
            "ontology": {"enabled": True, "merge_with_manual": True},
        })
        self.assertIn("PERSON", labels)
        self.assertIn("EVENT", labels)

    @patch("graphgen.config.llm.ChatGroq")
    def test_get_langchain_llm_uses_env_groq_key_when_config_missing_it(self, mock_chatgroq):
        mock_chatgroq.return_value = object()
        config = {
            "llm": {
                "base_model": "llama-3.1-8b-instant",
                "summarization_model": "llama-3.1-8b-instant",
                "temperature": 0.0,
            },
            "infra": {},
        }

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            llm = get_langchain_llm(config, purpose="summarization")

        self.assertIs(llm, mock_chatgroq.return_value)
        mock_chatgroq.assert_called_once()

    @patch("graphgen.config.llm.ChatGroq")
    def test_get_langchain_llm_accepts_masked_secret_from_model_dump(self, mock_chatgroq):
        mock_chatgroq.return_value = object()
        settings = PipelineSettings(
            infra={"input_dir": "input", "output_dir": "output", "groq_api_key": "real-secret"},
            llm={"base_model": "llama-3.1-8b-instant", "extraction_model": "llama-3.1-8b-instant"},
        )
        dumped = settings.model_dump()

        llm = get_langchain_llm(dumped, purpose="extraction")

        self.assertIs(llm, mock_chatgroq.return_value)
        mock_chatgroq.assert_called_once()
        self.assertEqual(mock_chatgroq.call_args.kwargs["api_key"], "real-secret")

    def test_diagnostics_helpers_write_json_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "infra": {"output_dir": tmpdir},
                "extraction": {"diagnostic_mode": True, "diagnostic_output_subdir": "diagnostics"},
            }
            self.assertTrue(diagnostics_enabled(config))
            path = write_diagnostic_json(config, "sample", {"ok": True})
            self.assertTrue(Path(path).exists())


class DSPyConfigRegressionTests(unittest.TestCase):
    def test_candidate_grounded_in_evidence_matches_exact_phrase_tokens(self):
        self.assertTrue(_candidate_grounded_in_evidence("policy changes", "responding to policy changes in health"))

    def test_endpoint_matches_hint_allows_hint_substrings_for_compound_endpoints(self):
        self.assertTrue(_endpoint_matches_hint("PRIME_MINISTER_OF_ITALY", ["ITALY", "MARIO_DRAGHI"]))
        self.assertTrue(_endpoint_matches_hint("PRESIDENT_OF_THE_EUROPEAN_CENTRAL_BANK", ["EUROPEAN_CENTRAL_BANK"]))
        self.assertFalse(_endpoint_matches_hint("COUNTRY", ["ITALY", "MARIO_DRAGHI"]))

    def test_grounded_relation_endpoint_accepts_entity_hint(self):
        self.assertTrue(_is_grounded_relation_endpoint("ENERGY_DEPENDENCE", ["ENERGY_DEPENDENCE", "KREMLIN"], ["POLICY_INSTRUMENT"], "energy dependence on the Kremlin"))

    def test_grounded_relation_endpoint_rejects_pure_ontology_class_without_hint(self):
        self.assertFalse(_is_grounded_relation_endpoint("POLICY_INSTRUMENT", ["HEALTH", "CLIMATE", "SECURITY"], ["POLICY_INSTRUMENT"], "policy changes in health, climate, security"))

    def test_ungrounded_relation_triplet_flags_placeholder_class_triplet(self):
        self.assertTrue(_is_ungrounded_relation_triplet("POLICY_INSTRUMENT", "HEALTH", ["HEALTH", "CLIMATE", "SECURITY"], ["POLICY_INSTRUMENT"], "policy changes in health, climate, security"))

    def test_ungrounded_relation_triplet_keeps_grounded_entity_pair(self):
        self.assertFalse(_is_ungrounded_relation_triplet("ENERGY_DEPENDENCE", "KREMLIN", ["ENERGY_DEPENDENCE", "KREMLIN"], ["POLICY_INSTRUMENT", "ORGANIZATION"], "energy dependence on the Kremlin"))

    def test_relation_endpoints_in_hints_requires_both_endpoints(self):
        self.assertTrue(_relation_endpoints_in_hints("ENERGY_DEPENDENCE", "KREMLIN", ["ENERGY_DEPENDENCE", "KREMLIN"]))
        self.assertFalse(_relation_endpoints_in_hints("PRIME_MINISTER", "COUNTRY", ["PRIME_MINISTER"]))

    @patch("graphgen.pipeline.entity_relation.extractors.dspy.configure")
    @patch("graphgen.pipeline.entity_relation.extractors.dspy.LM")
    def test_dspy_extractor_prefixes_groq_models_when_groq_key_present(self, mock_lm, mock_configure):
        mock_lm.return_value = object()
        config = {
            "llm": {
                "extraction_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "base_model": "llama-3.1-8b-instant",
            },
            "infra": {"groq_api_key": "secret"},
        }

        DSPyExtractor(config)

        kwargs = mock_lm.call_args.kwargs
        self.assertEqual(kwargs["model"], "groq/meta-llama/llama-4-scout-17b-16e-instruct")
        self.assertEqual(kwargs["api_base"], "https://api.groq.com/openai/v1")
        mock_configure.assert_called_once()

    @patch("graphgen.pipeline.entity_relation.extractors.dspy.configure")
    @patch("graphgen.pipeline.entity_relation.extractors.dspy.LM")
    def test_dspy_extractor_prefixes_models_when_config_has_real_secret_objects(self, mock_lm, mock_configure):
        mock_lm.return_value = object()
        settings = PipelineSettings(
            infra={"input_dir": "input", "output_dir": "output", "groq_api_key": "real-secret"},
            llm={"extraction_model": "meta-llama/llama-4-scout-17b-16e-instruct"},
        )

        DSPyExtractor(settings.model_dump())

        kwargs = mock_lm.call_args.kwargs
        self.assertEqual(kwargs["model"], "groq/meta-llama/llama-4-scout-17b-16e-instruct")
        self.assertEqual(kwargs["api_base"], "https://api.groq.com/openai/v1")
        self.assertEqual(kwargs["api_key"], "real-secret")
        mock_configure.assert_called_once()

    def test_dspy_extractor_runtime_model_reports_groq_provider(self):
        config = {
            "llm": {"extraction_model": "meta-llama/llama-4-scout-17b-16e-instruct"},
            "infra": {"groq_api_key": "fake-key"},
        }

        DSPyExtractor(config)

        import dspy
        lm = dspy.settings.lm
        self.assertEqual(lm.model, "groq/meta-llama/llama-4-scout-17b-16e-instruct")
        self.assertEqual(lm._provider_name, "groq")
        self.assertEqual(lm.kwargs["api_base"], "https://api.groq.com/openai/v1")

    def test_dspy_summarizer_runtime_model_reports_groq_provider(self):
        config = {
            "llm": {"summarization_model": "llama-3.1-8b-instant"},
            "infra": {"groq_api_key": "fake-key"},
        }

        DSPySummarizer(config)

        import dspy
        lm = dspy.settings.lm
        self.assertEqual(lm.model, "groq/llama-3.1-8b-instant")
        self.assertEqual(lm._provider_name, "groq")
        self.assertTrue(lm.supports_function_calling)

    def test_get_extractor_uses_dspy_for_gliner2_ner_backend(self):
        extractor = get_extractor({"extraction": {"backend": "gliner2"}})
        self.assertIsInstance(extractor, DSPyExtractor)

    def test_get_extractor_uses_langchain_for_llm_backend(self):
        extractor = get_extractor({"extraction": {"backend": "llm"}})
        self.assertIsInstance(extractor, LangChainExtractor)

    def test_extraction_settings_alias_maps_backend_to_ner_backend(self):
        settings = ExtractionSettings(backend="gliner2")
        self.assertEqual(settings.ner_backend, "gliner2")
        self.assertEqual(settings.backend, "gliner2")

    def test_extraction_settings_defaults_relation_backend_for_gliner2(self):
        settings = ExtractionSettings(ner_backend="gliner2")
        self.assertEqual(settings.relation_backend, "dspy")

    def test_extraction_settings_defaults_relation_backend_for_llm(self):
        settings = ExtractionSettings(ner_backend="llm")
        self.assertEqual(settings.relation_backend, "langchain")

    def test_get_extractor_honors_explicit_relation_backend(self):
        extractor = get_extractor({"extraction": {"ner_backend": "gliner2", "relation_backend": "langchain"}})
        self.assertIsInstance(extractor, LangChainExtractor)
    def test_extract_relations_with_llm_async_accepts_legacy_two_tuple_result(self):
        class LegacyExtractor:
            async def extract_relations(self, **kwargs):
                return (
                    [("MARIO_DRAGHI", "LED", "ITALY", {"confidence": 0.9})],
                    [{"id": "MARIO_DRAGHI", "type": "PERSON", "properties": {}}],
                )

        relations, nodes, diagnostics = asyncio.run(
            extract_relations_with_llm_async(
                text="Mario Draghi led Italy.",
                extractor=LegacyExtractor(),
                entities=["MARIO_DRAGHI", "ITALY"],
                abstract_concepts=["PERSON", "LOCATION"],
            )
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(diagnostics, {})


class PipelineRobustnessRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_preflight_does_not_abort_when_uploader_is_unreachable(self):
        class FakeUploader:
            def __init__(self):
                self.close_called = False

            def connect(self):
                return False

            def close(self):
                self.close_called = True

        settings = PipelineSettings(
            infra={"input_dir": "input", "output_dir": "output"},
            analytics={"save_provenance": False},
            analysis={"topic_separation_test": False},
        )
        pipeline = KnowledgePipeline(settings=settings, uploader=FakeUploader(), extractor=None)

        pipeline._run_preflight_checks()

        self.assertIsNone(pipeline.uploader)

    def test_pipeline_saves_diagnostic_index_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = PipelineSettings(
                infra={"input_dir": "input", "output_dir": tmpdir},
                analytics={"save_provenance": False},
                analysis={"topic_separation_test": False},
            )
            pipeline = KnowledgePipeline(settings=settings, uploader=None, extractor=None)
            ctx = PipelineContext(graph=nx.DiGraph())
            ctx.diagnostics = {"chunk_diagnostics": ["/tmp/a.json"]}
            pipeline._step_save_artifacts(ctx)
            diag_index = Path(tmpdir) / "diagnostics" / "diagnostic_index.json"
            self.assertTrue(diag_index.exists())

    async def test_chunk_diagnostics_include_full_entity_and_relation_context(self):
        ctx = PipelineContext()
        chunk_id = "chunk-diag"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Mario Draghi led Italy.")

        class FakeExtractor:
            async def extract_relations(self, **kwargs):
                return (
                    [("MARIO_DRAGHI", "LED", "ITALY", {"confidence": 0.9, "evidence": "Mario Draghi led Italy"})],
                    [{"id": "MARIO_DRAGHI", "type": "PERSON", "properties": {}}, {"id": "ITALY", "type": "LOCATION", "properties": {}}],
                    {
                        "raw_triplets": [
                            {
                                "source": "MARIO_DRAGHI",
                                "relation": "LED",
                                "target": "ITALY",
                                "confidence": 0.9,
                                "evidence": "Mario Draghi led Italy",
                            },
                            {
                                "source": "COUNTRY",
                                "relation": "GUIDED",
                                "target": "ITALY",
                                "confidence": 0.2,
                                "evidence": "Mario Draghi led Italy",
                            },
                        ],
                        "triplet_decisions": [
                            {
                                "source": "MARIO_DRAGHI",
                                "relation": "LED",
                                "target": "ITALY",
                                "kept": True,
                                "drop_reason": None,
                            },
                            {
                                "source": "COUNTRY",
                                "relation": "GUIDED",
                                "target": "ITALY",
                                "kept": False,
                                "drop_reason": "source_not_grounded_in_hints",
                            },
                        ],
                    },
                )

        fake_entities = [
            {
                "text": "MARIO_DRAGHI",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.99,
                "candidate_labels": ["PERSON", "LOCATION"],
                "backend": "gliner2",
            },
            {
                "text": "ITALY",
                "label": "LOCATION",
                "ontology_label": "LOCATION",
                "confidence": 0.98,
                "candidate_labels": ["PERSON", "LOCATION"],
                "backend": "gliner2",
            },
        ]
        ner_diag = {
            "backend": "gliner2",
            "candidate_labels": ["PERSON", "LOCATION"],
            "raw_entities": fake_entities,
            "entities": fake_entities,
        }
        config = {
            "infra": {"output_dir": tempfile.gettempdir()},
            "extraction": {"backend": "gliner2", "diagnostic_mode": True, "diagnostic_output_subdir": "diag-test"},
        }

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=(fake_entities, ner_diag)):
            await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                FakeExtractor(),
                config,
                ["PERSON", "LOCATION"],
            )

        diag_path = Path(ctx.diagnostics["chunk_diagnostics"][0])
        payload = json.loads(diag_path.read_text())
        self.assertIn("ner", payload)
        self.assertIn("raw_entities", payload["ner"])
        self.assertIn("entities", payload["ner"])
        self.assertIn("relation_eligible_entities", payload)
        self.assertIn("raw_relations", payload)
        self.assertIn("raw_nodes", payload)
        self.assertIn("accepted_relations", payload)
        self.assertIn("accepted_nodes", payload)
        self.assertIn("relation_decisions", payload)
        self.assertIn("raw_triplets", payload)
        self.assertIn("entity_surface_metadata", payload)
        self.assertTrue(payload["relation_diagnostics_available"])
        self.assertEqual(payload["accepted_relations"][0][0], "MARIO_DRAGHI")
        self.assertEqual(len(payload["relation_decisions"]), 2)
        self.assertTrue(payload["relation_decisions"][0]["kept"])
        self.assertEqual(payload["relation_decisions"][1]["drop_reason"], "source_not_grounded_in_hints")
        self.assertEqual(payload["entity_surface_metadata"][0]["surface_form_class"], "named_entity")

    async def test_segment_diagnostics_include_final_entities_and_relations(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        segment_id = "SEG_1"
        chunk_id = "CHUNK_1"
        ctx.graph.add_node(segment_id, node_type="SEGMENT")
        ctx.graph.add_node(chunk_id, node_type="CHUNK", gliner_entities=[
            {"text": "KREMLIN", "label": "LOCATION", "ontology_label": "ORGANIZATION", "confidence": 0.9, "candidate_labels": ["ORGANIZATION"]}
        ], raw_extraction={
            "relations": [("KREMLIN", "COORDINATES", "SANCTIONS", {"confidence": 0.9})],
            "nodes": [
                {"id": "KREMLIN", "type": "ORGANIZATION"},
                {"id": "SANCTIONS", "type": "POLICY_INSTRUMENT"},
            ],
        })
        ctx.graph.add_edge(segment_id, chunk_id, label="HAS_CHUNK")
        ctx.stats['pipeline_config'] = {
            "infra": {"output_dir": tempfile.gettempdir()},
            "extraction": {"diagnostic_mode": True, "diagnostic_output_subdir": "diag-test"},
        }

        from graphgen.pipeline.entity_relation.extraction import enrich_graph_per_segment
        result = await enrich_graph_per_segment(ctx)
        self.assertEqual(result["segments_processed"], 1)
        diag_path = Path(ctx.diagnostics["segment_diagnostics"][0])
        payload = json.loads(diag_path.read_text())
        self.assertIn("final_entities", payload)
        self.assertIn("final_relations", payload)
        self.assertGreaterEqual(len(payload["final_entities"]), 1)
        self.assertGreaterEqual(len(payload["final_relations"]), 1)


class ExtractionTaskAccountingRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_build_relation_eligible_entities_prefers_grounded_named_entities(self):
        entities = [
            {"text": "MARIO_DRAGHI", "ontology_label": "PERSON", "confidence": 0.99},
            {"text": "PRIME_MINISTER", "ontology_label": "PERSON", "confidence": 0.999},
            {"text": "UKRAINE", "ontology_label": "LOCATION", "confidence": 0.98},
            {"text": "AID", "ontology_label": "POLICY_INSTRUMENT", "confidence": 0.84},
            {"text": "WE", "ontology_label": "ORGANIZATION", "confidence": 0.99},
        ]
        eligible = _build_relation_eligible_entities(entities)
        self.assertIn("MARIO_DRAGHI", eligible)
        self.assertIn("UKRAINE", eligible)
        self.assertIn("AID", eligible)
        self.assertNotIn("WE", eligible)

    def test_build_relation_eligible_entities_falls_back_to_all_entities_when_filter_is_empty(self):
        entities = [
            {"text": "WE", "ontology_label": "ORGANIZATION", "confidence": 0.99},
            {"text": "US", "ontology_label": "ORGANIZATION", "confidence": 0.99},
        ]
        eligible = _build_relation_eligible_entities(entities)
        self.assertEqual(sorted(eligible), ["US", "WE"])

    async def test_process_extraction_task_uses_ontology_labels_for_node_filtering(self):
        ctx = PipelineContext()
        chunk_id = "chunk-ontology-filter"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="health climate security")

        class FakeExtractor:
            async def extract_relations(self, **kwargs):
                self.kwargs = kwargs
                return ([], [])

        extractor = FakeExtractor()
        fake_entities = [
            {
                "text": "HEALTH",
                "label": "POLICY",
                "ontology_label": "POLICY_INSTRUMENT",
                "confidence": 0.99,
                "candidate_labels": ["POLICY"],
                "backend": "gliner2",
            }
        ]

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=fake_entities):
            await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                extractor,
                {"extraction": {"backend": "gliner2"}},
                ["POLICY_INSTRUMENT"],
            )

        self.assertEqual(extractor.kwargs["abstract_concepts"], ["POLICY_INSTRUMENT"])
        self.assertEqual(extractor.kwargs["entities"], ["HEALTH"])

    async def test_process_extraction_task_counts_gliner_entities_when_llm_returns_no_relations(self):
        ctx = PipelineContext()
        chunk_id = "chunk-1"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Mario Draghi addressed Parliament")

        class FakeExtractor:
            async def extract_relations(self, **kwargs):
                return ([], [])

        fake_entities = [
            {
                "text": "MARIO_DRAGHI",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.98,
                "start": 0,
                "end": 12,
                "candidate_labels": ["PERSON", "ORGANIZATION"],
                "backend": "gliner2",
            },
            {
                "text": "EUROPEAN_PARLIAMENT",
                "label": "ORGANIZATION",
                "ontology_label": "ORGANIZATION",
                "confidence": 0.95,
                "start": 23,
                "end": 43,
                "candidate_labels": ["PERSON", "ORGANIZATION"],
                "backend": "gliner2",
            },
        ]

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=fake_entities):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                FakeExtractor(),
                {"extraction": {"backend": "gliner2"}},
                ["PERSON", "ORGANIZATION"],
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["entity_count"], 2)
        self.assertEqual(result["relation_count"], 0)
        self.assertTrue(ctx.graph.nodes[chunk_id]["extraction_successful"])
