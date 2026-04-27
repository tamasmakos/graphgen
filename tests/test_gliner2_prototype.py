import json
import tempfile
import unittest
from pathlib import Path

from graphgen.prototype_gliner2_ontology import (
    build_top_level_label_space,
    select_candidate_labels,
    refine_entities_with_ontology,
    export_graphml_and_memgraph_artifacts,
)


class PrototypeLabelSpaceTests(unittest.TestCase):
    def test_build_top_level_label_space_groups_ontology_labels(self):
        labels = [
            "PERSON",
            "ORGANIZATION",
            "POLITICAL_PARTY",
            "COUNTRY",
            "CITY",
            "POLICY",
            "REGULATION",
        ]

        label_space = build_top_level_label_space(labels)

        self.assertIn("PERSON", label_space)
        self.assertIn("ORGANIZATION", label_space)
        self.assertIn("LOCATION", label_space)
        self.assertIn("POLICY", label_space)
        self.assertIn("COUNTRY", label_space["LOCATION"]["children"])
        self.assertIn("CITY", label_space["LOCATION"]["children"])


class PrototypeCandidateSelectionTests(unittest.TestCase):
    def test_select_candidate_labels_prefers_text_relevant_labels(self):
        label_space = {
            "PERSON": {"aliases": ["person", "president"], "children": ["PERSON"]},
            "ORGANIZATION": {"aliases": ["organization", "commission", "parliament"], "children": ["ORGANIZATION"]},
            "LOCATION": {"aliases": ["location", "country", "city", "brussels"], "children": ["CITY", "COUNTRY"]},
        }
        text = "The European Commission met in Brussels with national leaders."

        selected = select_candidate_labels(text, label_space, top_k=2)

        self.assertIn("ORGANIZATION", selected)
        self.assertIn("LOCATION", selected)
        self.assertLess(selected.index("ORGANIZATION"), 2)


class PrototypeRefinementTests(unittest.TestCase):
    def test_refine_entities_with_ontology_assigns_child_type(self):
        label_space = {
            "LOCATION": {
                "aliases": ["location", "country", "city", "brussels"],
                "children": ["CITY", "COUNTRY"],
                "child_aliases": {
                    "CITY": ["city", "brussels"],
                    "COUNTRY": ["country", "france"],
                },
            }
        }
        entities = [
            {"text": "Brussels", "label": "LOCATION", "score": 0.91},
            {"text": "France", "label": "LOCATION", "score": 0.88},
        ]

        refined = refine_entities_with_ontology(entities, label_space)

        self.assertEqual(refined[0]["ontology_label"], "CITY")
        self.assertEqual(refined[1]["ontology_label"], "COUNTRY")


class PrototypeExportTests(unittest.TestCase):
    def test_export_graphml_and_memgraph_artifacts_writes_files(self):
        entities = [
            {"text": "Brussels", "label": "LOCATION", "ontology_label": "CITY", "score": 0.9},
            {"text": "European Commission", "label": "ORGANIZATION", "ontology_label": "ORGANIZATION", "score": 0.93},
        ]
        relations = [
            {"source": "European Commission", "target": "Brussels", "relation": "MENTIONS_LOCATION"}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = export_graphml_and_memgraph_artifacts(entities, relations, output_dir)

            self.assertTrue(Path(paths["graphml"]).exists())
            self.assertTrue(Path(paths["memgraph_cypher"]).exists())
            summary = json.loads(Path(paths["summary_json"]).read_text())
            self.assertEqual(summary["entity_count"], 2)
            self.assertEqual(summary["relation_count"], 1)
