import unittest

import networkx as nx

from graphgen.evaluation.entity_resolution_eval import summarize_entity_resolution_effects


class EntityResolutionEvaluationTests(unittest.TestCase):
    def test_summarize_entity_resolution_effects_reports_basic_graph_stats(self):
        before = nx.DiGraph()
        before.add_node("EUROPEAN_CENTRAL_BANK", node_type="ENTITY_CONCEPT", name="EUROPEAN_CENTRAL_BANK")
        before.add_node("European Central Bank", node_type="ENTITY_CONCEPT", name="European Central Bank")
        before.add_node("MARIO_DRAGHI", node_type="ENTITY_CONCEPT", name="MARIO_DRAGHI")
        before.add_edge("MARIO_DRAGHI", "EUROPEAN_CENTRAL_BANK")
        before.add_edge("MARIO_DRAGHI", "European Central Bank")

        after = nx.DiGraph()
        after.add_node("EUROPEAN_CENTRAL_BANK", node_type="ENTITY_CONCEPT", name="EUROPEAN_CENTRAL_BANK")
        after.add_node("MARIO_DRAGHI", node_type="ENTITY_CONCEPT", name="MARIO_DRAGHI")
        after.add_edge("MARIO_DRAGHI", "EUROPEAN_CENTRAL_BANK")

        summary = summarize_entity_resolution_effects(before, after)

        self.assertEqual(summary["entity_nodes_before"], 3)
        self.assertEqual(summary["entity_nodes_after"], 2)
        self.assertEqual(summary["merged_nodes"], 1)
        self.assertIn("top_degree_nodes_before", summary)
        self.assertIn("top_degree_nodes_after", summary)
        self.assertIn("surface_form_class_counts", summary)
