import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from pydantic import ValidationError
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

from graphgen.config.llm import get_langchain_llm
from graphgen.config.settings import PipelineSettings
from graphgen.config.settings import ExtractionSettings
from graphgen.config.schema import GraphSchema, NodeSchema
from graphgen.data_types import ChunkExtractionTask, PipelineContext
from graphgen.orchestrator import KnowledgePipeline
from graphgen.pipeline.lexical_graph_building.builder import build_lexical_graph
from graphgen.pipeline.entity_relation.extraction import (
    _extract_entities_for_chunk,
    _build_relation_eligible_entities,
    extract_relations_with_llm_async,
    process_extraction_task,
)
from graphgen.pipeline.embeddings.rag import generate_rag_embeddings as mock_generate_rag_embeddings
from graphgen.utils.vector_embedder.rag import generate_rag_embeddings as real_generate_rag_embeddings
from graphgen.utils.labels import resolve_entity_labels, resolve_entity_label_profiles
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
from graphgen.pipeline.entity_relation.label_space import build_label_profiles, build_gliner2_schema, select_candidate_label_profiles
from graphgen.utils.schema_utils import save_graph_schema
from graphgen.utils.diagnostics import diagnostics_enabled, write_diagnostic_json
from graphgen.pipeline.summarization.summarizer import DSPySummarizer
from graphgen.pipeline.summarization.core import _parse_summary_response, process_batch_tasks
from graphgen.pipeline.summarization.models import SummarizationTask
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

    @patch("graphgen.utils.graphdb.neo4j_adapter.Neo4jGraphUploader")
    @patch("graphgen.pipeline.entity_relation.extractors.get_extractor")
    @patch("graphgen.main.KnowledgePipeline")
    @patch("graphgen.main.configure_logging")
    def test_run_pipeline_uses_knowledge_pipeline_even_when_iterative_flag_is_true(
        self,
        mock_configure_logging,
        mock_knowledge_pipeline,
        mock_get_extractor,
        mock_uploader,
    ):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock
        from graphgen.main import run_pipeline

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run = AsyncMock()
        mock_knowledge_pipeline.return_value = mock_pipeline_instance
        mock_get_extractor.return_value = object()

        settings = SimpleNamespace(
            debug=False,
            infra=SimpleNamespace(
                input_dir="input/txt/translated",
                output_dir="output",
                neo4j_host="localhost",
                neo4j_port=7687,
                neo4j_user="neo4j",
                neo4j_password="password",
            ),
            iterative=SimpleNamespace(enabled=True),
            model_dump=lambda: {"extraction": {"backend": "dspy"}},
        )

        with patch("graphgen.main.PipelineSettings.load", return_value=settings):
            asyncio.run(run_pipeline())

        mock_knowledge_pipeline.assert_called_once()
        mock_pipeline_instance.run.assert_awaited_once()
        mock_configure_logging.assert_called()
        mock_uploader.assert_called_once()
        mock_get_extractor.assert_called_once_with(settings.model_dump())


class ConfigRegressionTests(unittest.TestCase):
    def test_requirements_include_node2vec_dependency(self):
        requirements_path = Path("/root/graphgen/requirements.txt")
        requirements = requirements_path.read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?im)^node2vec(?:[<>=!~].*)?$")

    def test_pipeline_settings_reject_invalid_extraction_mode(self):
        with self.assertRaises(ValidationError):
            PipelineSettings(extraction={"mode": "improvise"})

    def test_pipeline_settings_loads_external_schema_and_label_profiles_relative_to_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            schema_path = tmp_path / "schema.json"
            labels_path = tmp_path / "labels.json"
            config_path = tmp_path / "config.yaml"

            schema_path.write_text(json.dumps({
                "nodes": {
                    "Doc": {"label": "DOC", "source_type": "document", "attributes": ["filename"]},
                    "Passage": {"label": "PASSAGE", "source_type": "segment", "attributes": ["content"]},
                    "Snippet": {"label": "SNIPPET", "source_type": "chunk", "attributes": ["text"]},
                },
                "edges": [
                    {"source_label": "DOC", "target_label": "PASSAGE", "relation_type": "HAS_PASSAGE", "is_hierarchical": True},
                    {"source_label": "PASSAGE", "target_label": "SNIPPET", "relation_type": "HAS_SNIPPET", "is_hierarchical": True},
                ],
            }), encoding="utf-8")
            labels_path.write_text(json.dumps([
                {"label": "Wizard", "description": "A magical person.", "aliases": ["wizard"]}
            ]), encoding="utf-8")
            config_path.write_text(
                "schema_path: schema.json\n"
                "extraction:\n"
                "  label_profiles_path: labels.json\n",
                encoding="utf-8",
            )

            settings = PipelineSettings.load(config_path=str(config_path), env_file="/root/graphgen/.env")
            profiles = resolve_entity_label_profiles({
                "label_profiles_path": settings.extraction.label_profiles_path,
                "ontology": {"enabled": False},
            })

            self.assertEqual(settings.extraction.label_profiles_path, str(labels_path.resolve()))
            self.assertEqual(settings.schema_config["nodes"]["Snippet"]["label"], "SNIPPET")
            self.assertEqual(profiles[0]["label"], "WIZARD")
            self.assertEqual(profiles[0]["description"], "A magical person.")

    def test_save_graph_schema_uses_explicit_schema_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            schema = {
                "nodes": {
                    "Doc": {"label": "DOC", "source_type": "document", "attributes": ["filename"]},
                    "Passage": {"label": "PASSAGE", "source_type": "segment", "attributes": ["content"]},
                },
                "edges": [
                    {"source_label": "DOC", "target_label": "PASSAGE", "relation_type": "HAS_PASSAGE", "is_hierarchical": True}
                ],
            }

            save_graph_schema(nx.DiGraph(), str(output_dir), schema_config=schema)

            saved = json.loads((output_dir / "graph_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["nodes"]["Passage"]["label"], "PASSAGE")

    def test_run_with_config_normalizes_relative_paths_from_config_directory(self):
        import importlib.util
        import sys
        import tempfile
        from pathlib import Path as StdPath
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = StdPath(tmpdir)
            config_path = tmp_path / "config.custom.yaml"
            config_path.write_text("infra:\n  input_dir: input/txt\n", encoding="utf-8")

            spec = importlib.util.spec_from_file_location("run_with_config_test", "/root/graphgen/run_with_config.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            original_cwd = os.getcwd()
            old_argv = sys.argv[:]
            try:
                sys.argv = ["run_with_config.py", str(config_path)]
                with patch.object(module, "resolve_env_file", return_value="/root/graphgen/.env"), \
                     patch.object(module, "configure_logging"), \
                     patch.object(module, "get_extractor", return_value=object()), \
                     patch.object(module, "KnowledgePipeline") as pipeline_cls, \
                     patch.object(module.PipelineSettings, "load") as mock_load, \
                     patch.object(module.asyncio, "run"):
                    mock_settings = Mock(debug=False)
                    mock_settings.model_dump.return_value = {}
                    mock_load.return_value = mock_settings
                    pipeline_cls.return_value.run.return_value = object()

                    module.main()

                    self.assertEqual(os.getcwd(), str(tmp_path))
                    self.assertEqual(mock_load.call_args.kwargs["config_path"], str(config_path.resolve()))
            finally:
                os.chdir(original_cwd)
                sys.argv = old_argv

    def test_repo_config_disables_iterative_runtime_by_default(self):
        settings = PipelineSettings.load(config_path="/root/graphgen/config.yaml", env_file="/root/graphgen/.env")
        self.assertFalse(settings.iterative.enabled)

    def test_repo_config_keeps_node2vec_enabled_for_thesis_workflow(self):
        settings = PipelineSettings.load(config_path="/root/graphgen/config.yaml", env_file="/root/graphgen/.env")
        self.assertTrue(settings.community.node2vec_enabled)

    def test_analysis_settings_expose_default_topic_separation_output_file(self):
        settings = PipelineSettings()
        self.assertEqual(settings.analysis.output_file, "topic_separation_report.json")
        self.assertEqual(settings.analytics.output_file, "topic_separation_report.json")

    def test_pipeline_settings_sync_analysis_from_analytics_when_only_analytics_is_configured(self):
        settings = PipelineSettings(analytics={"output_file": "custom_topic_report.json", "topic_separation_test": False})
        self.assertEqual(settings.analytics.output_file, "custom_topic_report.json")
        self.assertEqual(settings.analysis.output_file, "custom_topic_report.json")
        self.assertFalse(settings.analysis.topic_separation_test)

    def test_extraction_settings_support_optional_chunk_budget(self):
        settings = PipelineSettings(extraction={"max_chunks": 12})
        self.assertEqual(settings.extraction.max_chunks, 12)

    def test_test_mode_settings_support_optional_chunk_budget(self):
        settings = PipelineSettings(test_mode={"enabled": True, "max_chunks": 7})
        self.assertEqual(settings.test_mode.max_chunks, 7)

    def test_extraction_settings_allow_disabling_eager_gliner_preload(self):
        settings = PipelineSettings(extraction={"gliner_preload": False})
        self.assertFalse(settings.extraction.gliner_preload)


class ExtractionRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_entities_for_chunk_skips_label_based_ner_in_schemaless_mode(self):
        with patch("graphgen.pipeline.entity_relation.extraction.get_gliner_model") as mock_get_gliner_model:
            entities = await _extract_entities_for_chunk(
                "Harry met Hermione.",
                {"extraction": {"mode": "schemaless", "backend": "gliner"}},
                ["PERSON"],
            )

        self.assertEqual(entities, [])
        mock_get_gliner_model.assert_not_called()

    async def test_extract_all_entities_relations_uses_schemaless_mode_without_label_profiles(self):
        deps = PipelineContext(graph=nx.DiGraph())
        deps.graph.add_node("chunk-1", node_type="CHUNK")
        deps.extraction_tasks = [
            ChunkExtractionTask(
                chunk_id="chunk-1",
                chunk_text="The Potters had a son, Harry.",
                segment_id="seg-1",
                parent_doc_id="doc-1",
                keywords=[],
                entities=[],
            )
        ]

        class FakeExtractor:
            async def extract_relations(self, text, keywords=None, entities=None, abstract_concepts=None):
                return [], [], {}

        async def fake_process_task(deps_arg, task, semaphore, extractor, config, ontology_labels, label_profiles=None):
            self.assertEqual(config["extraction"]["mode"], "schemaless")
            self.assertEqual(ontology_labels, [])
            self.assertEqual(label_profiles, [])
            deps_arg.graph.nodes[task.chunk_id]["knowledge_triplets"] = []
            deps_arg.graph.nodes[task.chunk_id]["raw_extraction"] = {"relations": [], "nodes": []}
            deps_arg.graph.nodes[task.chunk_id]["gliner_entities"] = []
            deps_arg.graph.nodes[task.chunk_id]["extraction_successful"] = True
            return {"success": True, "chunk_id": task.chunk_id, "entity_count": 1, "relation_count": 1}

        with patch("graphgen.pipeline.entity_relation.extraction.process_extraction_task", side_effect=fake_process_task), \
             patch("graphgen.pipeline.entity_relation.extraction.enrich_graph_per_segment", return_value={"errors": []}):
            from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations

            result = await extract_all_entities_relations(
                deps,
                {"extraction": {"mode": "schemaless", "ner_backend": "dspy", "gliner_preload": False}},
                extractor=FakeExtractor(),
            )

        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["total_entities_extracted"], 1)
        self.assertEqual(result["total_relations_extracted"], 1)

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
    def test_resolve_entity_labels_returns_empty_when_ontology_enabled_without_manual_labels(self):
        with patch("graphgen.utils.labels.extract_ontology_profiles", return_value=[]):
            labels = resolve_entity_labels({
                "ontology": {"enabled": True, "merge_with_manual": True},
            })
        self.assertEqual(labels, [])

    def test_resolve_entity_labels_returns_empty_without_manual_or_ontology_labels(self):
        labels = resolve_entity_labels({
            "ontology": {"enabled": False},
        })
        self.assertEqual(labels, [])

    def test_build_label_profiles_generates_descriptions_without_manual_hints(self):
        profiles = build_label_profiles([
            {"label": "Political Party", "aliases": [], "description": None, "source": "config"}
        ])
        self.assertEqual(profiles[0]["label"], "POLITICAL_PARTY")
        self.assertIn("political party", profiles[0]["description"].lower())

    def test_select_candidate_label_profiles_prefers_semantic_match(self):
        profiles = build_label_profiles([
            {"label": "PERSON", "aliases": [], "description": "A named person.", "source": "config"},
            {"label": "ORGANIZATION", "aliases": [], "description": "A named organization.", "source": "config"},
        ])

        class FakeEmbeddingModel:
            is_available = True

            def encode(self, texts, batch_size=None):
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    if "harry met hermione" in lowered:
                        vectors.append([1.0, 0.0])
                    elif "named person" in lowered:
                        vectors.append([0.95, 0.05])
                    else:
                        vectors.append([0.0, 1.0])
                return np.array(vectors, dtype=float)

        with patch("graphgen.pipeline.entity_relation.label_space.get_model", return_value=FakeEmbeddingModel()):
            selection = select_candidate_label_profiles("Harry met Hermione in the hall.", profiles, top_k=1)

        self.assertEqual(selection["candidate_labels"], ["PERSON"])
        self.assertGreater(selection["profile_scores"][0]["semantic_score"], 0.9)

    def test_build_gliner2_schema_uses_profile_descriptions(self):
        schema = build_gliner2_schema([
            {"label": "PERSON", "description": "Named entity or concept labeled person."}
        ])
        self.assertEqual(schema["entities"]["person"], "Named entity or concept labeled person.")

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
    def test_summary_parser_repairs_missing_closing_finding_brace(self):
        content = '''```json
{
  "title": "Renew Europe's Commitment to European Values and the Rule of Law",
  "summary": "This community is centered around Renew Europe's unwavering commitment to upholding European values, particularly the rule of law.",
  "findings": [
    {
      "summary": "Renew Europe's Support for the European Public Prosecutor's Office",
      "explanation": "Renew Europe strongly supports the European Public Prosecutor's Office."
    },
    {
      "summary": "Collaborative Efforts in Protecting the Rule of Law",
      "explanation": "The relationships between Renew Europe and other entities highlight their collaborative efforts in protecting and strengthening the rule of law."
  ]
}
```'''
        data = _parse_summary_response(content)
        self.assertEqual(data["title"], "Renew Europe's Commitment to European Values and the Rule of Law")
        self.assertEqual(len(data["findings"]), 2)
        self.assertEqual(data["findings"][-1]["summary"], "Collaborative Efforts in Protecting the Rule of Law")

    def test_summary_parser_repairs_unescaped_inner_quotes_in_summary_text(self):
        content = '''```json
{
  "title": "European Union's Efforts for Peace and Circular Economy",
  "summary": "This community revolves around the European Union's efforts to achieve peace, implement the circular economy, and promote solidarity and unity.",
  "findings": [
    {
      "summary": "EU's Efforts for Energy Independence",
      "explanation": "The European Union's efforts for energy independence are a significant aspect of this community. The EU's engagement in activities related to energy independence, such as working towards a "just transition," highlights its commitment to reducing its reliance on Russian gas and oil."
    }
  ]
}
```'''
        data = _parse_summary_response(content)
        self.assertEqual(data["title"], "European Union's Efforts for Peace and Circular Economy")
        self.assertEqual(len(data["findings"]), 1)
        self.assertIn('"just transition,"', data["findings"][0]["explanation"])

    def test_summary_parser_raises_when_no_json_object_can_be_found(self):
        with self.assertRaises(ValueError):
            _parse_summary_response("not json at all")

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


class KnowledgePipelineNode2VecRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_communities_applies_node2vec_weights_when_enabled(self):
        settings = PipelineSettings(
            infra={"input_dir": "input", "output_dir": "output"},
            analytics={"save_provenance": False},
            analysis={"topic_separation_test": False},
            community={"node2vec_enabled": True},
        )
        pipeline = KnowledgePipeline(settings=settings, uploader=None, extractor=None)
        ctx = PipelineContext(graph=nx.DiGraph())
        ctx.graph.add_node("A", node_type="ENTITY_CONCEPT")
        ctx.graph.add_node("B", node_type="ENTITY_CONCEPT")
        ctx.graph.add_edge("A", "B", graph_type="entity_relation", weight=1.0)

        detector_instance = Mock()
        detector_instance.detect_communities.side_effect = [
            {"assignments": {"A": 0, "B": 0}, "modularity": 0.41},
            {"assignments": {"A": 0, "B": 0}, "modularity": 0.57},
        ]
        detector_instance.detect_subcommunities_leiden.return_value = {}

        with patch("graphgen.pipeline.community.detection.CommunityDetector", return_value=detector_instance), \
             patch("graphgen.pipeline.embeddings.node2vec_wrapper.compute_node2vec_weights", return_value={("A", "B"): 0.83}) as mock_weights, \
             patch("graphgen.pipeline.community.subcommunities.add_enhanced_community_attributes_to_graph"), \
             patch("graphgen.pipeline.summarization.core.generate_community_summaries", return_value={"topics": 1}) as mock_summaries, \
             patch("graphgen.config.llm.get_langchain_llm", return_value=object()):
            await pipeline._step_communities(ctx, settings.model_dump())

        self.assertEqual(ctx.graph["A"]["B"]["weight"], 0.83)
        self.assertEqual(ctx.stats["communities"]["modularity_baseline"], 0.41)
        self.assertEqual(ctx.stats["communities"]["modularity"], 0.57)
        self.assertTrue(ctx.stats["communities"]["node2vec_enabled"])
        mock_weights.assert_called_once()
        mock_summaries.assert_awaited_once()

class LexicalGraphBudgetRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_extraction_settings_support_segmentation_options(self):
        settings = PipelineSettings(extraction={"segmentation_mode": "paragraph", "min_segment_words": 12})
        self.assertEqual(settings.extraction.segmentation_mode, "paragraph")
        self.assertEqual(settings.extraction.min_segment_words, 12)

    def test_extraction_settings_reject_unknown_segmentation_mode(self):
        with self.assertRaises(ValidationError):
            PipelineSettings(extraction={"segmentation_mode": "scene"})

    def test_extraction_settings_reject_unknown_label_selection_strategy(self):
        with self.assertRaises(ValidationError):
            PipelineSettings(extraction={"label_selection_strategy": "manual-hints"})

    def test_build_segments_from_text_defaults_to_nonempty_lines(self):
        from graphgen.pipeline.lexical_graph_building.builder import _build_segments_from_text

        segments = _build_segments_from_text("Alpha\n\nBeta\n", "DOC_demo", {"extraction": {"segmentation_mode": "line"}})

        self.assertEqual([segment.content for segment in segments], ["Alpha", "Beta"])
        self.assertEqual([segment.line_number for segment in segments], [1, 3])

    def test_build_segments_from_text_supports_paragraph_mode(self):
        from graphgen.pipeline.lexical_graph_building.builder import _build_segments_from_text

        text = "Title\n\nFirst paragraph line one.\nStill first paragraph.\n\nSecond paragraph."
        segments = _build_segments_from_text(text, "DOC_demo", {"extraction": {"segmentation_mode": "paragraph"}})

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].content, "Title")
        self.assertEqual(segments[0].line_number, 1)
        self.assertEqual(segments[1].content, "First paragraph line one. Still first paragraph.")
        self.assertEqual(segments[1].line_number, 3)
        self.assertEqual(segments[2].content, "Second paragraph.")
        self.assertEqual(segments[2].line_number, 6)

    def test_build_segments_from_text_preserves_paragraph_start_line_numbers_with_leading_blank_lines(self):
        from graphgen.pipeline.lexical_graph_building.builder import _build_segments_from_text

        text = "\n\nIntro line.\nStill intro.\n\n\nSecond paragraph."
        segments = _build_segments_from_text(text, "DOC_demo", {"extraction": {"segmentation_mode": "paragraph"}})

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].line_number, 3)
        self.assertEqual(segments[0].content, "Intro line. Still intro.")
        self.assertEqual(segments[1].line_number, 7)
        self.assertEqual(segments[1].content, "Second paragraph.")

    def test_build_segments_from_text_filters_short_segments(self):
        from graphgen.pipeline.lexical_graph_building.builder import _build_segments_from_text

        text = "CHAPTER ONE\n\nThis is a real paragraph with enough words to keep.\n\nHi there."
        segments = _build_segments_from_text(
            text,
            "DOC_demo",
            {"extraction": {"segmentation_mode": "paragraph", "min_segment_words": 5}},
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].content, "This is a real paragraph with enough words to keep.")

    async def test_build_lexical_graph_supports_custom_segment_and_chunk_labels(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        schema = GraphSchema(nodes={
            "Doc": NodeSchema(label="DOC", source_type="document"),
            "Segment": NodeSchema(label="PASSAGE", source_type="segment"),
            "Chunk": NodeSchema(label="SNIPPET", source_type="chunk"),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doc1.txt").write_text("Alpha paragraph with enough words to chunk.\n\nBeta paragraph with enough words too.", encoding="utf-8")
            result = await build_lexical_graph(
                ctx,
                tmpdir,
                {"extraction": {"segmentation_mode": "paragraph", "chunk_size": 64, "chunk_overlap": 8}},
                schema=schema,
            )

        self.assertEqual(result["documents_processed"], 1)
        self.assertGreater(result["total_chunks"], 0)
        node_types = {data.get("node_type") for _, data in ctx.graph.nodes(data=True)}
        self.assertIn("PASSAGE", node_types)
        self.assertIn("SNIPPET", node_types)

    async def test_build_lexical_graph_respects_chunk_budget(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        async def fake_process_single_document_lexical(deps, filename, input_dir, config=None, schema=None):
            doc_id = f"DOC_{filename}"
            deps.graph.add_node(doc_id, node_type="DOC", segment_count=2)
            deps.graph.add_node(f"{doc_id}_S0", node_type="SEGMENT")
            deps.graph.add_node(f"{doc_id}_S1", node_type="SEGMENT")
            deps.graph.add_edge(doc_id, f"{doc_id}_S0", label="HAS_SEGMENT")
            deps.graph.add_edge(doc_id, f"{doc_id}_S1", label="HAS_SEGMENT")
            for idx in range(3):
                chunk_id = f"{doc_id}_S0_C{idx}" if idx < 2 else f"{doc_id}_S1_C0"
                parent = f"{doc_id}_S0" if idx < 2 else f"{doc_id}_S1"
                deps.graph.add_node(chunk_id, node_type="CHUNK")
                deps.graph.add_edge(parent, chunk_id, label="HAS_CHUNK")
                deps.extraction_tasks.append(ChunkExtractionTask(chunk_id=chunk_id, chunk_text="x", entities=[], abstract_concepts=[], keywords=[]))
            return {"segments_added": 2, "chunks_added": 3, "errors": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doc1.txt").write_text("a", encoding="utf-8")
            config = {
                "extraction": {"file_pattern": "*.txt", "max_chunks": 2},
                "test_mode": {"enabled": True, "max_documents": 1},
            }
            with patch(
                "graphgen.pipeline.lexical_graph_building.builder.process_single_document_lexical",
                new=fake_process_single_document_lexical,
            ):
                result = await build_lexical_graph(ctx, tmpdir, config)

        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(len(ctx.extraction_tasks), 2)
        chunk_nodes = [n for n, d in ctx.graph.nodes(data=True) if d.get("node_type") == "CHUNK"]
        self.assertEqual(len(chunk_nodes), 2)

    async def test_build_lexical_graph_respects_chunk_budget_for_custom_chunk_label(self):
        ctx = PipelineContext(graph=nx.DiGraph())
        schema = GraphSchema(nodes={
            "Doc": NodeSchema(label="DOC", source_type="document"),
            "Segment": NodeSchema(label="SEGMENT", source_type="segment"),
            "Chunk": NodeSchema(label="ChunkNode", source_type="chunk"),
        })

        async def fake_process_single_document_lexical(deps, filename, input_dir, config=None, schema=None):
            doc_id = f"DOC_{filename}"
            deps.graph.add_node(doc_id, node_type="DOC", segment_count=2)
            deps.graph.add_node(f"{doc_id}_S0", node_type="SEGMENT")
            deps.graph.add_node(f"{doc_id}_S1", node_type="SEGMENT")
            deps.graph.add_edge(doc_id, f"{doc_id}_S0", label="HAS_SEGMENT")
            deps.graph.add_edge(doc_id, f"{doc_id}_S1", label="HAS_SEGMENT")
            for idx in range(3):
                chunk_id = f"{doc_id}_S0_C{idx}" if idx < 2 else f"{doc_id}_S1_C0"
                parent = f"{doc_id}_S0" if idx < 2 else f"{doc_id}_S1"
                deps.graph.add_node(chunk_id, node_type="ChunkNode")
                deps.graph.add_edge(parent, chunk_id, label="HAS_CHUNK")
                deps.extraction_tasks.append(ChunkExtractionTask(chunk_id=chunk_id, chunk_text="x", entities=[], abstract_concepts=[], keywords=[]))
            return {"segments_added": 2, "chunks_added": 3, "errors": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doc1.txt").write_text("a", encoding="utf-8")
            config = {
                "extraction": {"file_pattern": "*.txt", "max_chunks": 2},
                "test_mode": {"enabled": True, "max_documents": 1},
            }
            with patch(
                "graphgen.pipeline.lexical_graph_building.builder.process_single_document_lexical",
                new=fake_process_single_document_lexical,
            ):
                result = await build_lexical_graph(ctx, tmpdir, config, schema=schema)

        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(len(ctx.extraction_tasks), 2)
        chunk_nodes = [n for n, d in ctx.graph.nodes(data=True) if d.get("node_type") == "ChunkNode"]
        self.assertEqual(len(chunk_nodes), 2)

    async def test_step_topic_analysis_writes_report_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = PipelineSettings(
                infra={"input_dir": "input", "output_dir": tmpdir},
                analytics={"save_provenance": False},
                analysis={"topic_separation_test": True, "output_file": "topic_separation_report.json"},
            )
            pipeline = KnowledgePipeline(settings=settings, uploader=None, extractor=None)
            ctx = PipelineContext(graph=nx.DiGraph())
            ctx.graph.add_node("TOPIC_0", node_type="TOPIC", embedding=[1.0, 0.0], title="T0", summary="S0")
            ctx.graph.add_node("TOPIC_1", node_type="TOPIC", embedding=[0.0, 1.0], title="T1", summary="S1")
            ctx.graph.add_node("ENTITY_A", node_type="ENTITY_CONCEPT")
            ctx.graph.add_node("ENTITY_B", node_type="ENTITY_CONCEPT")
            ctx.graph.add_edge("ENTITY_A", "ENTITY_B", graph_type="entity_relation", weight=1.0)
            with patch("graphgen.analytics.reporting.generate_topic_separation_report", return_value={
                "community_level": {"silhouette_score": 0.12},
                "subcommunity_level": {"silhouette_score": 0.07},
                "overall_interpretation": "weak but present structure",
            }) as mock_report:
                await pipeline._step_topic_analysis(ctx)

            self.assertIn("topic_analysis", ctx.stats)
            self.assertTrue(ctx.stats["topic_analysis"]["output_file"].endswith("topic_separation_report.json"))
            mock_report.assert_called_once()

    async def test_step_communities_generates_topic_and_subtopic_embeddings_after_summaries(self):
        settings = PipelineSettings(
            infra={"input_dir": "input", "output_dir": "output"},
            analytics={"save_provenance": False},
            analysis={"topic_separation_test": False},
            community={"node2vec_enabled": False},
        )
        pipeline = KnowledgePipeline(settings=settings, uploader=None, extractor=None)
        ctx = PipelineContext(graph=nx.DiGraph())
        ctx.graph.add_node("ENTITY_A", node_type="ENTITY_CONCEPT", name="UKRAINE")
        ctx.graph.add_node("ENTITY_B", node_type="ENTITY_CONCEPT", name="ENERGY_SECURITY")
        ctx.graph.add_edge("ENTITY_A", "ENTITY_B", graph_type="entity_relation", weight=1.0)

        detector_instance = Mock()
        detector_instance.detect_communities.return_value = {"assignments": {"ENTITY_A": 0, "ENTITY_B": 0}, "modularity": 0.57}
        detector_instance.detect_subcommunities_leiden.return_value = {"ENTITY_A": (0, 0), "ENTITY_B": (0, 0)}

        async def fake_summaries(graph, llm):
            graph.add_node("TOPIC_0", node_type="TOPIC", title="Energy Security", summary="Energy security in Europe")
            graph.add_node("SUBTOPIC_0_0", node_type="SUBTOPIC", title="Gas Supply", summary="Gas supply resilience")
            graph.add_edge("ENTITY_A", "TOPIC_0", label="IN_TOPIC")
            graph.add_edge("ENTITY_B", "SUBTOPIC_0_0", label="IN_TOPIC")
            return {"topics": 1, "subtopics": 1}

        with patch("graphgen.pipeline.community.detection.CommunityDetector", return_value=detector_instance), \
             patch("graphgen.pipeline.community.subcommunities.add_enhanced_community_attributes_to_graph"), \
             patch("graphgen.pipeline.summarization.core.generate_community_summaries", side_effect=fake_summaries) as mock_summaries, \
             patch("graphgen.config.llm.get_langchain_llm", return_value=object()), \
             patch("graphgen.utils.vector_embedder.rag.generate_rag_embeddings", return_value={"TOPIC_0": [0.1, 0.2], "SUBTOPIC_0_0": [0.2, 0.3]}) as mock_embed:
            await pipeline._step_communities(ctx, settings.model_dump())

        mock_summaries.assert_awaited_once()
        mock_embed.assert_called_once()

    def test_real_rag_embeddings_cover_topic_and_subtopic_nodes(self):
        graph = nx.DiGraph()
        graph.add_node("ENTITY_A", node_type="ENTITY_CONCEPT", name="UKRAINE", entity_type="PLACE")
        graph.add_node("ENTITY_B", node_type="ENTITY_CONCEPT", name="ENERGY_SECURITY", entity_type="CONCEPT")
        graph.add_node("TOPIC_0", node_type="TOPIC", title="Energy and Security", summary="Discussion of energy security and Ukraine")
        graph.add_node("SUBTOPIC_0_0", node_type="SUBTOPIC", title="Energy Dependence", summary="Gas supply and resilience")
        graph.add_edge("ENTITY_A", "TOPIC_0", label="IN_TOPIC")
        graph.add_edge("ENTITY_B", "TOPIC_0", label="IN_TOPIC")
        graph.add_edge("ENTITY_B", "SUBTOPIC_0_0", label="IN_TOPIC")

        embeddings = real_generate_rag_embeddings(graph, batch_size=4, node_types=["ENTITY_CONCEPT", "TOPIC", "SUBTOPIC"])

        self.assertIn("TOPIC_0", embeddings)
        self.assertIn("SUBTOPIC_0_0", embeddings)
        self.assertIn("embedding", graph.nodes["TOPIC_0"])
        self.assertIn("embedding", graph.nodes["SUBTOPIC_0_0"])
        self.assertGreater(len(graph.nodes["TOPIC_0"]["embedding"]), 0)
        self.assertGreater(len(graph.nodes["SUBTOPIC_0_0"]["embedding"]), 0)

    def test_topic_separation_report_analyzes_topic_and_subtopic_embeddings(self):
        from graphgen.analytics.reporting import generate_topic_separation_report

        graph = nx.DiGraph()
        graph.add_node("TOPIC_0", node_type="TOPIC", embedding=[1.0, 0.0], title="T0", summary="S0")
        graph.add_node("TOPIC_1", node_type="TOPIC", embedding=[0.0, 1.0], title="T1", summary="S1")
        graph.add_node("TOPIC_2", node_type="TOPIC", embedding=[0.9, 0.1], title="T2", summary="S2")
        graph.add_node("TOPIC_3", node_type="TOPIC", embedding=[0.1, 0.9], title="T3", summary="S3")
        graph.add_node("SUBTOPIC_0_0", node_type="SUBTOPIC", embedding=[1.0, 0.0], title="ST00", summary="SS00")
        graph.add_node("SUBTOPIC_0_1", node_type="SUBTOPIC", embedding=[0.95, 0.05], title="ST01", summary="SS01")
        graph.add_node("SUBTOPIC_1_0", node_type="SUBTOPIC", embedding=[0.0, 1.0], title="ST10", summary="SS10")
        graph.add_node("SUBTOPIC_1_1", node_type="SUBTOPIC", embedding=[0.05, 0.95], title="ST11", summary="SS11")

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_topic_separation_report(
                graph,
                str(Path(tmpdir) / "topic_separation_report.json"),
                PipelineSettings().analysis,
            )

        self.assertIsNotNone(report["community_level"])
        self.assertIsNotNone(report["subcommunity_level"])
        self.assertGreater(report["global_separation"], 0.0)
        self.assertGreater(report["community_level"]["n_samples"], 3)
        self.assertGreater(report["subcommunity_level"]["n_samples"], 3)

    def test_topic_separation_report_saved_json_serializes_pairwise_significance(self):
        from graphgen.analytics.reporting import generate_topic_separation_report

        graph = nx.DiGraph()
        graph.add_node("TOPIC_0", node_type="TOPIC", title="T0", summary="S0")
        graph.add_node("TOPIC_1", node_type="TOPIC", title="T1", summary="S1")
        graph.add_node("ENTITY_A0", node_type="ENTITY_CONCEPT", embedding=[1.0, 0.0])
        graph.add_node("ENTITY_A1", node_type="ENTITY_CONCEPT", embedding=[0.9, 0.1])
        graph.add_node("ENTITY_B0", node_type="ENTITY_CONCEPT", embedding=[0.0, 1.0])
        graph.add_node("ENTITY_B1", node_type="ENTITY_CONCEPT", embedding=[0.1, 0.9])
        graph.add_edge("ENTITY_A0", "TOPIC_0")
        graph.add_edge("ENTITY_A1", "TOPIC_0")
        graph.add_edge("ENTITY_B0", "TOPIC_1")
        graph.add_edge("ENTITY_B1", "TOPIC_1")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topic_separation_report.json"
            generate_topic_separation_report(
                graph,
                str(output_path),
                PipelineSettings().analysis,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(payload["pairwise_comparisons"]), 1)
        self.assertIsInstance(payload["pairwise_comparisons"][0]["significant"], bool)

    async def test_process_batch_tasks_retries_rate_limit_errors(self):
        task = SummarizationTask(
            task_id="TOPIC_0",
            community_id=0,
            subcommunity_id=None,
            is_topic=True,
            chunk_texts=["Example chunk text"],
            entities=[{"name": "EUROPEAN_UNION", "id": "EUROPEAN_UNION", "type": "ORG", "degree": 10}],
            relationships=[("EUROPEAN_UNION", "SUPPORTS", "UKRAINE")],
            chunk_ids=["CHUNK_0"],
            entity_ids=["EUROPEAN_UNION"],
            sub_summaries=[{"id": "SUBTOPIC_0_0", "summary": "Prior subtopic summary"}],
        )

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class FakeLLM:
            def __init__(self):
                self.calls = 0
            async def ainvoke(self, prompt):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("Error code: 429 - rate limited")
                return FakeResponse('{"title":"Recovered summary","summary":"Recovered after retry","findings":[{"summary":"Recovered","explanation":"Retry succeeded"}]}')

        llm = FakeLLM()
        with patch("graphgen.pipeline.summarization.core.RATE_LIMIT_RETRY_BASE_DELAY", 0.01):
            results = await process_batch_tasks(llm, [task])

        self.assertEqual(llm.calls, 3)
        self.assertEqual(results[0].title, "Recovered summary")
        self.assertEqual(results[0].summary, "Recovered after retry")
        self.assertEqual(results[0].findings[0]["summary"], "Recovered")


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
            ctx.diagnostics = {"chunk_diagnostics": ["/tmp/a.json"], "entity_resolution_diagnostics": "/tmp/b.json"}
            pipeline._step_save_artifacts(ctx)
            diag_index = Path(tmpdir) / "diagnostics" / "diagnostic_index.json"
            self.assertTrue(diag_index.exists())
            payload = json.loads(diag_index.read_text())
            self.assertIn("entity_resolution_diagnostics", payload)

    async def test_enrichment_writes_entity_resolution_diagnostics_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = PipelineSettings(
                infra={"input_dir": "input", "output_dir": tmpdir},
                analytics={"save_provenance": False},
                analysis={"topic_separation_test": False},
                extraction={"diagnostic_mode": True, "diagnostic_output_subdir": "diagnostics"},
            )
            pipeline = KnowledgePipeline(settings=settings, uploader=None, extractor=None)
            ctx = PipelineContext(graph=nx.DiGraph())
            ctx.stats["pipeline_config"] = settings.model_dump()
            ctx.graph.add_node("ECB_1", node_type="ENTITY_CONCEPT", name="EUROPEAN_CENTRAL_BANK", embedding=[1.0, 0.0])
            ctx.graph.add_node("ECB_2", node_type="ENTITY_CONCEPT", name="European Central Bank", embedding=[1.0, 0.0])
            ctx.graph.add_node("MARIO_DRAGHI", node_type="ENTITY_CONCEPT", name="MARIO_DRAGHI", embedding=[0.0, 1.0])
            ctx.graph.add_edge("MARIO_DRAGHI", "ECB_1", label="LEADS")
            ctx.graph.add_edge("MARIO_DRAGHI", "ECB_2", label="LEADS")

            with patch("graphgen.utils.vector_embedder.rag.generate_rag_embeddings", return_value={}) as mock_embed:
                await pipeline._step_enrichment(ctx)

            mock_embed.assert_called_once()

            self.assertIn("entity_resolution_diagnostics", ctx.diagnostics)
            diag_path = Path(ctx.diagnostics["entity_resolution_diagnostics"])
            self.assertTrue(diag_path.exists())
            payload = json.loads(diag_path.read_text())
            self.assertIn("entity_resolution", payload)
            self.assertIn("evaluation", payload)
            self.assertIn("merged_pairs", payload["entity_resolution"])

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

    async def test_process_extraction_task_allows_schemaless_relation_extraction_without_entity_labels(self):
        ctx = PipelineContext()
        chunk_id = "chunk-no-labels"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Mr. and Mrs. Dursley lived at number four, Privet Drive.")

        class FakeExtractor:
            async def extract_relations(self, **kwargs):
                self.kwargs = kwargs
                return ([
                    ("MR_AND_MRS_DURSLEY", "LIVES_AT", "NUMBER_FOUR_PRIVET_DRIVE", {"confidence": 1.0})
                ], [
                    {"id": "MR_AND_MRS_DURSLEY", "type": "ENTITY"},
                    {"id": "NUMBER_FOUR_PRIVET_DRIVE", "type": "ENTITY"},
                ])

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=[]):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                FakeExtractor(),
                {"extraction": {"backend": "gliner2"}},
                [],
                label_profiles=[],
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["relation_count"], 1)
        self.assertEqual(ctx.graph.nodes[chunk_id]["knowledge_triplets"][0][0], "MR_AND_MRS_DURSLEY")

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

    async def test_process_extraction_task_uses_schemaless_triplets_when_filter_drops_all_relations(self):
        ctx = PipelineContext()
        chunk_id = "chunk-schemaless"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="Rubeus Hagrid, Keeper of Keys and Grounds at Hogwarts.")

        fake_entities = [
            {
                "text": "RUBEUS_HAGRID",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.99,
                "candidate_labels": ["PERSON", "SCHOOL"],
                "backend": "gliner2",
            }
        ]

        async def fake_extract_relations_with_llm_async(**kwargs):
            return (
                [],
                [],
                {
                    "raw_triplets": [
                        {
                            "source": "Rubeus Hagrid",
                            "relation": "is",
                            "target": "Keeper of Keys and Grounds at Hogwarts",
                            "source_type": "Person",
                            "target_type": "Role",
                            "confidence": 1.0,
                            "evidence": "Rubeus Hagrid, Keeper of Keys and Grounds at Hogwarts.",
                        }
                    ],
                    "triplet_decisions": [
                        {
                            "source": "Rubeus Hagrid",
                            "relation": "is",
                            "target": "Keeper of Keys and Grounds at Hogwarts",
                            "source_type": "Person",
                            "target_type": "Role",
                            "confidence": 1.0,
                            "evidence": "Rubeus Hagrid, Keeper of Keys and Grounds at Hogwarts.",
                            "kept": False,
                            "drop_reason": "endpoint_not_grounded_in_hints",
                            "source_grounded": True,
                            "target_grounded": True,
                        }
                    ],
                },
            )

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=(fake_entities, {"backend": "gliner2"})), \
             patch("graphgen.pipeline.entity_relation.extraction.extract_relations_with_llm_async", side_effect=fake_extract_relations_with_llm_async):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                SimpleNamespace(),
                {"extraction": {"backend": "gliner2", "diagnostic_mode": True, "diagnostic_output_subdir": "diag-test"}, "infra": {"output_dir": tempfile.gettempdir()}},
                ["PERSON", "SCHOOL"],
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["relation_count"], 2)
        relations = ctx.graph.nodes[chunk_id]["knowledge_triplets"]
        normalized = {(src, rel, tgt) for src, rel, tgt, _ in relations}
        self.assertIn(("RUBEUS_HAGRID", "IS", "KEEPER_OF_KEYS_AND_GROUNDS_AT_HOGWARTS"), normalized)
        self.assertIn(("RUBEUS_HAGRID", "AT", "HOGWARTS"), normalized)
        self.assertEqual(ctx.graph.nodes[chunk_id]["raw_extraction"]["nodes"][0]["id"], "RUBEUS_HAGRID")

    async def test_process_extraction_task_does_not_readd_ungrounded_triplets_in_schemaless_fallback(self):
        ctx = PipelineContext()
        chunk_id = "chunk-schemaless-drop"
        ctx.graph.add_node(chunk_id, node_type="CHUNK")
        task = ChunkExtractionTask(chunk_id=chunk_id, chunk_text="They just didn't hold with such nonsense.")

        fake_entities = [
            {
                "text": "MR_AND_MRS_DURSLEY",
                "label": "PERSON",
                "ontology_label": "PERSON",
                "confidence": 0.99,
                "candidate_labels": ["PERSON"],
                "backend": "gliner2",
            }
        ]

        async def fake_extract_relations_with_llm_async(**kwargs):
            return (
                [],
                [],
                {
                    "raw_triplets": [
                        {
                            "source": "Mr. and Mrs. Dursley",
                            "relation": "disbelieves_in",
                            "target": "strange or mysterious things",
                            "source_type": "Person",
                            "target_type": "Concept",
                            "confidence": 1.0,
                            "evidence": "They just didn't hold with such nonsense.",
                        }
                    ],
                    "triplet_decisions": [
                        {
                            "source": "Mr. and Mrs. Dursley",
                            "relation": "disbelieves_in",
                            "target": "strange or mysterious things",
                            "source_type": "Person",
                            "target_type": "Concept",
                            "confidence": 1.0,
                            "evidence": "They just didn't hold with such nonsense.",
                            "kept": False,
                            "drop_reason": "endpoint_not_grounded_in_hints",
                            "source_grounded": False,
                            "target_grounded": False,
                        }
                    ],
                },
            )

        with patch("graphgen.pipeline.entity_relation.extraction._extract_entities_for_chunk", return_value=fake_entities), \
             patch("graphgen.pipeline.entity_relation.extraction.extract_relations_with_llm_async", side_effect=fake_extract_relations_with_llm_async):
            result = await process_extraction_task(
                ctx,
                task,
                asyncio.Semaphore(1),
                SimpleNamespace(),
                {"extraction": {"backend": "gliner2"}},
                ["PERSON"],
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["relation_count"], 0)
        self.assertEqual(ctx.graph.nodes[chunk_id]["knowledge_triplets"], [])
