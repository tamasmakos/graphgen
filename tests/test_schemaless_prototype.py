import unittest
from unittest.mock import patch


class SchemalessPrototypeTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_schemaless_relations_falls_back_to_raw_triplets_when_filtered_output_is_empty(self):
        from graphgen.pipeline.entity_relation.schemaless_prototype import extract_schemaless_relations

        class FakeDSPyExtractor:
            def __init__(self, config):
                self.config = config

            async def extract_relations(self, text, entities=None, abstract_concepts=None):
                return (
                    [],
                    [],
                    {
                        "raw_triplets": [
                            {
                                "source": "The Potters",
                                "relation": "has_son",
                                "target": "Harry",
                                "source_type": "",
                                "target_type": "",
                                "confidence": 1.0,
                                "evidence": "their son, Harry",
                            }
                        ]
                    },
                )

        with patch("graphgen.pipeline.entity_relation.schemaless_prototype.DSPyExtractor", FakeDSPyExtractor):
            result = await extract_schemaless_relations("their son, Harry", config={})

        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["relations"][0][0], "THE_POTTERS")
        self.assertEqual(result["relations"][0][1], "PARENT_OF")
        self.assertEqual(result["relations"][0][2], "HARRY")
        node_ids = {node["id"] for node in result["nodes"]}
        self.assertIn("THE_POTTERS", node_ids)
        self.assertIn("HARRY", node_ids)

    def test_postprocess_triplets_adds_location_relation_for_appositive_at_phrase(self):
        from graphgen.pipeline.entity_relation.schemaless_prototype import postprocess_triplets

        processed = postprocess_triplets(
            [
                {
                    "source": "Rubeus Hagrid",
                    "relation": "is",
                    "target": "Keeper of Keys and Grounds at Hogwarts",
                    "source_type": "PERSON",
                    "target_type": "ROLE",
                    "confidence": 1.0,
                    "evidence": "Rubeus Hagrid, Keeper of Keys and Grounds at Hogwarts.",
                }
            ]
        )

        normalized = {(src, rel, tgt) for src, rel, tgt, _ in processed}
        self.assertIn(("RUBEUS_HAGRID", "IS", "KEEPER_OF_KEYS_AND_GROUNDS_AT_HOGWARTS"), normalized)
        self.assertIn(("RUBEUS_HAGRID", "AT", "HOGWARTS"), normalized)
        derived = next(props for src, rel, tgt, props in processed if rel == "AT")
        self.assertEqual(derived["target_type"], "LOCATION")

    def test_postprocess_triplets_normalizes_generic_relations_without_schema(self):
        from graphgen.pipeline.entity_relation.schemaless_prototype import postprocess_triplets

        processed = postprocess_triplets(
            [
                {
                    "source": "Quidditch",
                    "relation": "played with",
                    "target": "broomsticks",
                    "source_type": "",
                    "target_type": "",
                    "confidence": 1.0,
                    "evidence": "played up in the air on broomsticks",
                }
            ]
        )

        self.assertEqual(processed[0][0], "QUIDDITCH")
        self.assertEqual(processed[0][1], "IS_PLAYED_ON")
        self.assertEqual(processed[0][2], "BROOMSTICKS")
