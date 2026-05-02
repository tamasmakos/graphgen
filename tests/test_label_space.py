import unittest
from unittest.mock import patch

import numpy as np


class LabelSpaceProfileTests(unittest.TestCase):
    def test_build_label_profiles_generates_aliases_and_descriptions_without_manual_hints(self):
        from graphgen.pipeline.entity_relation.label_space import build_label_profiles

        profiles = build_label_profiles([
            {
                "label": "Political Party",
                "aliases": [],
                "description": None,
                "source": "config",
            }
        ])

        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["label"], "POLITICAL_PARTY")
        self.assertIn("political party", profiles[0]["aliases"])
        self.assertIn("political", profiles[0]["aliases"])
        self.assertIn("party", profiles[0]["aliases"])
        self.assertIn("political party", profiles[0]["description"].lower())

    def test_build_gliner2_schema_uses_generated_profile_descriptions(self):
        from graphgen.pipeline.entity_relation.label_space import build_gliner2_schema

        schema = build_gliner2_schema([
            {
                "label": "PERSON",
                "description": "Named entity or concept labeled person.",
            }
        ])

        self.assertEqual(schema["entities"]["person"], "Named entity or concept labeled person.")


class LabelSpaceSelectionTests(unittest.TestCase):
    def test_select_candidate_label_profiles_uses_semantic_ranking_without_manual_hints(self):
        from graphgen.pipeline.entity_relation.label_space import build_label_profiles, select_candidate_label_profiles

        profiles = build_label_profiles([
            {"label": "PERSON", "aliases": [], "description": "A named person.", "source": "config"},
            {"label": "ORGANIZATION", "aliases": [], "description": "A named organization.", "source": "config"},
            {"label": "LOCATION", "aliases": [], "description": "A named location.", "source": "config"},
        ])

        class FakeEmbeddingModel:
            is_available = True

            def encode(self, texts, batch_size=None):
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    if "harry met hermione" in lowered:
                        vectors.append([1.0, 0.0, 0.0])
                    elif "named person" in lowered:
                        vectors.append([0.95, 0.05, 0.0])
                    elif "named organization" in lowered:
                        vectors.append([0.0, 1.0, 0.0])
                    else:
                        vectors.append([0.0, 0.0, 1.0])
                return np.array(vectors, dtype=float)

        with patch("graphgen.pipeline.entity_relation.label_space.get_model", return_value=FakeEmbeddingModel()):
            result = select_candidate_label_profiles(
                "Harry met Hermione in the corridor.",
                profiles,
                top_k=1,
                strategy="hybrid",
            )

        self.assertEqual(result["candidate_labels"], ["PERSON"])
        self.assertEqual(result["candidate_profiles"][0]["label"], "PERSON")
        self.assertGreater(result["profile_scores"][0]["semantic_score"], 0.9)

    def test_select_candidate_label_profiles_skips_embedding_model_for_lexical_strategy(self):
        from graphgen.pipeline.entity_relation.label_space import build_label_profiles, select_candidate_label_profiles

        profiles = build_label_profiles([
            {"label": "PERSON", "aliases": ["wizard"], "description": "A named person.", "source": "config"},
            {"label": "SPELL", "aliases": ["expelliarmus"], "description": "A named spell.", "source": "config"},
        ])

        with patch("graphgen.pipeline.entity_relation.label_space.get_model") as mock_get_model:
            result = select_candidate_label_profiles(
                "Harry shouted expelliarmus in class.",
                profiles,
                top_k=1,
                strategy="lexical",
            )

        mock_get_model.assert_not_called()
        self.assertEqual(result["candidate_labels"], ["SPELL"])

    def test_select_candidate_label_profiles_falls_back_to_lexical_when_embeddings_fail(self):
        from graphgen.pipeline.entity_relation.label_space import build_label_profiles, select_candidate_label_profiles

        profiles = build_label_profiles([
            {"label": "SPELL", "aliases": ["expelliarmus"], "description": "A magical spell.", "source": "config"},
            {"label": "PERSON", "aliases": ["wizard"], "description": "A named person.", "source": "config"},
        ])

        class BrokenEmbeddingModel:
            is_available = True

            def encode(self, texts, batch_size=None):
                raise RuntimeError("embedding unavailable")

        with patch("graphgen.pipeline.entity_relation.label_space.get_model", return_value=BrokenEmbeddingModel()):
            result = select_candidate_label_profiles(
                "Harry shouted expelliarmus in class.",
                profiles,
                top_k=1,
                strategy="hybrid",
            )

        self.assertEqual(result["candidate_labels"], ["SPELL"])
        self.assertGreaterEqual(result["profile_scores"][0]["lexical_score"], 1.0)
        self.assertEqual(result["profile_scores"][0]["semantic_score"], 0.0)


class LabelResolutionTests(unittest.TestCase):
    def test_resolve_entity_label_profiles_returns_structured_entries_without_ontology(self):
        from graphgen.utils.labels import resolve_entity_label_profiles

        profiles = resolve_entity_label_profiles(
            {
                "entity_labels": ["Person", "School of Magic"],
                "ontology": {"enabled": False},
            }
        )

        self.assertEqual([item["label"] for item in profiles], ["PERSON", "SCHOOL_OF_MAGIC"])
        self.assertEqual(profiles[0]["source"], "config")
        self.assertIn("school of magic", profiles[1]["aliases"])

    def test_resolve_entity_label_profiles_returns_empty_without_manual_or_ontology_labels(self):
        from graphgen.utils.labels import resolve_entity_label_profiles

        profiles = resolve_entity_label_profiles(
            {
                "ontology": {"enabled": False},
            }
        )

        self.assertEqual(profiles, [])

    def test_resolve_entity_label_profiles_applies_custom_gliner2_descriptions(self):
        from graphgen.utils.labels import resolve_entity_label_profiles

        profiles = resolve_entity_label_profiles(
            {
                "entity_labels": ["Person", "Spell"],
                "gliner2_label_descriptions": {
                    "PERSON": "A named wizard or witch.",
                    "SPELL": "A magical incantation or spell name.",
                },
                "ontology": {"enabled": False},
            }
        )

        by_label = {profile["label"]: profile for profile in profiles}
        self.assertEqual(by_label["PERSON"]["description"], "A named wizard or witch.")
        self.assertEqual(by_label["SPELL"]["description"], "A magical incantation or spell name.")
