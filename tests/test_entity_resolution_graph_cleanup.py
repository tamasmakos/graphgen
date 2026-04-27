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
