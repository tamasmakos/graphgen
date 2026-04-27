import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graphgen.config.llm import get_langchain_llm
from graphgen.config.settings import PipelineSettings
from graphgen.config.settings import ExtractionSettings
from graphgen.data_types import ChunkExtractionTask, PipelineContext
from graphgen.orchestrator import KnowledgePipeline
from graphgen.pipeline.entity_relation.extraction import (
    _extract_entities_for_chunk,
    process_extraction_task,
)
from graphgen.pipeline.entity_relation.extractors import (
    DSPyExtractor,
    LangChainExtractor,
    get_extractor,
)
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


class DSPyConfigRegressionTests(unittest.TestCase):
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
        self.assertTrue(lm.supports_function_calling)

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


class PipelineRobustnessRegressionTests(unittest.TestCase):
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


class ExtractionTaskAccountingRegressionTests(unittest.IsolatedAsyncioTestCase):
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
