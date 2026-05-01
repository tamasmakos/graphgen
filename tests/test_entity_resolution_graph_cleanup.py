import unittest

import networkx as nx
import numpy as np

from graphgen.pipeline.graph_cleaning.resolution import BlockingResolver, EntityRecord, resolve_entities_semantically


class GraphCleanupResolutionTests(unittest.TestCase):
    def test_blocking_resolver_merges_obvious_aliases(self):
        resolver = BlockingResolver(similarity_threshold=0.9)
        resolver.add_records([
            EntityRecord(id="ECB_1", text="EUROPEAN_CENTRAL_BANK", type="ORGANIZATION"),
            EntityRecord(id="ECB_2", text="European Central Bank", type="ORGANIZATION"),
        ])

        resolved = resolver.resolve()

        self.assertEqual(resolved["ECB_1"], resolved["ECB_2"])

    def test_blocking_resolver_rejects_role_to_person_merge(self):
        resolver = BlockingResolver(similarity_threshold=0.85)
        resolver.add_records([
            EntityRecord(id="ROLE", text="PRIME_MINISTER", type="ROLE"),
            EntityRecord(id="PERSON", text="MARIO_DRAGHI", type="PERSON"),
        ])

        resolved = resolver.resolve()

        self.assertNotEqual(resolved["ROLE"], resolved["PERSON"])

    def test_blocking_resolver_rejects_concept_to_named_entity_merge(self):
        resolver = BlockingResolver(similarity_threshold=0.85)
        resolver.add_records([
            EntityRecord(id="CONCEPT", text="MIGRATION", type="POLICY_INSTRUMENT"),
            EntityRecord(id="COUNTRY", text="ITALY", type="LOCATION"),
        ])

        resolved = resolver.resolve()

        self.assertNotEqual(resolved["CONCEPT"], resolved["COUNTRY"])

    def test_blocking_resolver_rejects_opposite_meaning_policy_terms(self):
        resolver = BlockingResolver(similarity_threshold=0.9)
        resolver.add_records([
            EntityRecord(id="INDEPENDENCE", text="ENERGY_INDEPENDENCE", type="POLICY_INSTRUMENT"),
            EntityRecord(id="DEPENDENCE", text="ENERGY_DEPENDENCE", type="POLICY_INSTRUMENT"),
        ])

        resolved = resolver.resolve()

        self.assertNotEqual(resolved["INDEPENDENCE"], resolved["DEPENDENCE"])

    def test_blocking_resolver_rejects_region_and_adjectival_forms(self):
        resolver = BlockingResolver(similarity_threshold=0.9)
        resolver.add_records([
            EntityRecord(id="REGION", text="NORTHERN_EUROPE", type="LOCATION"),
            EntityRecord(id="ADJECTIVAL", text="NORTHERN_EUROPEAN", type="LOCATION"),
        ])

        resolved = resolver.resolve()

        self.assertNotEqual(resolved["REGION"], resolved["ADJECTIVAL"])

    def test_resolve_entities_semantically_merges_alias_nodes(self):
        graph = nx.DiGraph()
        emb = np.array([1.0, 0.0, 0.0])
        graph.add_node("ECB_1", node_type="ENTITY_CONCEPT", name="EUROPEAN_CENTRAL_BANK", embedding=emb)
        graph.add_node("ECB_2", node_type="ENTITY_CONCEPT", name="European Central Bank", embedding=emb)
        graph.add_node("UKRAINE", node_type="ENTITY_CONCEPT", name="UKRAINE", embedding=np.array([0.0, 1.0, 0.0]))
        graph.add_edge("ECB_1", "UKRAINE", label="SUPPORTS")
        graph.add_edge("ECB_2", "UKRAINE", label="SUPPORTS")

        stats = resolve_entities_semantically(graph, similarity_threshold=0.9)

        self.assertEqual(stats["clusters_found"], 1)
        self.assertEqual(stats["merged_nodes"], 1)
        self.assertIn("ECB_1", graph.nodes)
        self.assertNotIn("ECB_2", graph.nodes)

    def test_resolve_entities_semantically_rejects_opposite_meaning_policy_terms(self):
        graph = nx.DiGraph()
        emb = np.array([1.0, 0.0, 0.0])
        graph.add_node(
            "ENERGY_INDEPENDENCE",
            node_type="ENTITY_CONCEPT",
            name="ENERGY_INDEPENDENCE",
            ontology_class="POLICY_INSTRUMENT",
            embedding=emb,
        )
        graph.add_node(
            "ENERGY_DEPENDENCE",
            node_type="ENTITY_CONCEPT",
            name="ENERGY_DEPENDENCE",
            ontology_class="POLICY_INSTRUMENT",
            embedding=emb,
        )

        stats = resolve_entities_semantically(graph, similarity_threshold=0.9)

        self.assertEqual(stats["clusters_found"], 0)
        self.assertEqual(stats["merged_nodes"], 0)
        self.assertIn("ENERGY_INDEPENDENCE", graph.nodes)
        self.assertIn("ENERGY_DEPENDENCE", graph.nodes)

    def test_resolve_entities_semantically_rejects_role_to_person_merge_using_node_types(self):
        graph = nx.DiGraph()
        emb = np.array([1.0, 0.0, 0.0])
        graph.add_node(
            "ROLE_NODE",
            node_type="ENTITY_CONCEPT",
            name="PRIME_MINISTER_OF_ITALY",
            ontology_class="ROLE",
            embedding=emb,
        )
        graph.add_node(
            "PERSON_NODE",
            node_type="ENTITY_CONCEPT",
            name="MARIO_DRAGHI",
            ontology_class="PERSON",
            embedding=emb,
        )

        stats = resolve_entities_semantically(graph, similarity_threshold=0.85)

        self.assertEqual(stats["clusters_found"], 0)
        self.assertEqual(stats["merged_nodes"], 0)
        self.assertIn("ROLE_NODE", graph.nodes)
        self.assertIn("PERSON_NODE", graph.nodes)

    def test_resolve_entities_semantically_rejects_region_and_adjectival_forms(self):
        graph = nx.DiGraph()
        emb = np.array([1.0, 0.0, 0.0])
        graph.add_node(
            "NORTHERN_EUROPE",
            node_type="ENTITY_CONCEPT",
            name="NORTHERN_EUROPE",
            ontology_class="LOCATION",
            embedding=emb,
        )
        graph.add_node(
            "NORTHERN_EUROPEAN",
            node_type="ENTITY_CONCEPT",
            name="NORTHERN_EUROPEAN",
            ontology_class="LOCATION",
            embedding=emb,
        )

        stats = resolve_entities_semantically(graph, similarity_threshold=0.9)

        self.assertEqual(stats["clusters_found"], 0)
        self.assertEqual(stats["merged_nodes"], 0)
        self.assertIn("NORTHERN_EUROPE", graph.nodes)
        self.assertIn("NORTHERN_EUROPEAN", graph.nodes)
