import asyncio
import tempfile
import unittest
from pathlib import Path

from graphgen.smoke import LocalSmokeExtractor, build_smoke_settings, run_local_smoke


class SmokeSettingsTests(unittest.TestCase):
    def test_build_smoke_settings_disables_heavy_features(self):
        settings = build_smoke_settings(input_dir="input/smoke", output_dir="output/smoke", max_documents=1)

        self.assertEqual(settings.infra.input_dir, "input/smoke")
        self.assertEqual(settings.infra.output_dir, "output/smoke")
        self.assertEqual(settings.extraction.backend, "smoke")
        self.assertEqual(settings.extraction.device, "cpu")
        self.assertEqual(settings.extraction.max_concurrent_chunks, 1)
        self.assertFalse(settings.iterative.enabled)
        self.assertFalse(settings.analytics.enabled)
        self.assertFalse(settings.analysis.topic_separation_test)
        self.assertFalse(settings.community.node2vec_enabled)


class LocalSmokeExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_creates_entities_and_relations_from_local_text(self):
        extractor = LocalSmokeExtractor()

        relations, nodes = await extractor.extract_relations(
            text="Emmanuel Macron met Ursula von der Leyen in Brussels to discuss Europe."
        )

        node_ids = {node["id"] for node in nodes}
        self.assertIn("EMMANUEL_MACRON", node_ids)
        self.assertIn("URSULA_VON_DER_LEYEN", node_ids)
        self.assertGreaterEqual(len(relations), 1)
        self.assertEqual(relations[0][1], "CO_OCCURS_WITH")


class LocalSmokeRunTests(unittest.TestCase):
    def test_run_local_smoke_writes_graph_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            output_dir = base / "output"
            input_dir.mkdir()
            (input_dir / "sample.txt").write_text(
                "Emmanuel Macron met Ursula von der Leyen in Brussels.\n"
                "The European Union discussed migration policy.\n",
                encoding="utf-8",
            )

            result = asyncio.run(
                run_local_smoke(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    max_documents=1,
                )
            )

            self.assertEqual(result["documents_processed"], 1)
            self.assertGreater(result["total_segments"], 0)
            self.assertTrue((output_dir / "knowledge_graph.graphml").exists())
            self.assertTrue((output_dir / "entity_resolution_report.json").exists())
            self.assertTrue((output_dir / "graph_schema.json").exists())
