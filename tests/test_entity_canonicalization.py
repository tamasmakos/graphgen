import unittest

from graphgen.pipeline.graph_cleaning.canonicalization import (
    are_potential_aliases,
    classify_surface_form,
    normalize_surface_form,
    surface_forms_conflict,
)
from graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences


class EntityCanonicalizationTests(unittest.TestCase):
    def test_normalize_surface_form_unifies_spacing_case_and_underscores(self):
        self.assertEqual(normalize_surface_form("European Central Bank"), "EUROPEAN_CENTRAL_BANK")
        self.assertEqual(normalize_surface_form(" european_central_bank "), "EUROPEAN_CENTRAL_BANK")

    def test_normalize_surface_form_keeps_eu_distinct_from_europe(self):
        self.assertEqual(normalize_surface_form("EU"), "EU")
        self.assertEqual(normalize_surface_form("Europe"), "EUROPE")
        self.assertNotEqual(normalize_surface_form("EU"), normalize_surface_form("Europe"))

    def test_classify_surface_form_detects_role_artifacts(self):
        self.assertEqual(classify_surface_form("PRIME_MINISTER"), "role_artifact")
        self.assertEqual(classify_surface_form("President"), "role_artifact")

    def test_classify_surface_form_detects_concept_like_terms(self):
        self.assertEqual(classify_surface_form("MIGRATION"), "concept_like")
        self.assertEqual(classify_surface_form("WHATEVER_IT_TAKES"), "concept_like")
        self.assertEqual(classify_surface_form("ENERGY_INDEPENDENCE"), "concept_like")

    def test_classify_surface_form_detects_named_entities(self):
        self.assertEqual(classify_surface_form("MARIO_DRAGHI"), "named_entity")
        self.assertEqual(classify_surface_form("European Central Bank"), "named_entity")

    def test_are_potential_aliases_matches_surface_variants(self):
        self.assertTrue(are_potential_aliases("European Central Bank", "EUROPEAN_CENTRAL_BANK"))

    def test_are_potential_aliases_rejects_role_to_person_match(self):
        self.assertFalse(are_potential_aliases("PRIME_MINISTER", "MARIO_DRAGHI"))

    def test_are_potential_aliases_rejects_eu_and_europe_equivalence(self):
        self.assertFalse(are_potential_aliases("EU", "EUROPE"))

    def test_are_potential_aliases_rejects_opposite_meaning_policy_terms(self):
        self.assertFalse(are_potential_aliases("ENERGY_INDEPENDENCE", "ENERGY_DEPENDENCE"))

    def test_are_potential_aliases_rejects_country_and_demonym_forms(self):
        self.assertFalse(are_potential_aliases("RUSSIA", "RUSSIAN"))

    def test_are_potential_aliases_rejects_region_and_adjectival_forms(self):
        self.assertFalse(are_potential_aliases("NORTHERN_EUROPE", "NORTHERN_EUROPEAN"))

    def test_surface_forms_conflict_rejects_mr_and_mrs_name_variants(self):
        self.assertTrue(surface_forms_conflict("MR_DURSLEY", "MRS_DURSLEY"))

    def test_resolve_extraction_coreferences_keeps_distinct_honorific_name_variants_separate(self):
        result = resolve_extraction_coreferences(
            [("MR_DURSLEY", "DIRECTOR_OF", "GRUNNINGS", {"confidence": 1.0})],
            ["MR_DURSLEY", "MRS_DURSLEY", "DUDLEY", "GRUNNINGS"],
        )

        self.assertEqual(result["entity_mappings"]["MR_DURSLEY"], "MR_DURSLEY")
        self.assertEqual(result["entity_mappings"]["MRS_DURSLEY"], "MRS_DURSLEY")
        self.assertEqual(result["cleaned_relations"][0][0], "MR_DURSLEY")
