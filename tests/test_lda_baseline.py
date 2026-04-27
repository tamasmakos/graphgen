import tempfile
import unittest
from pathlib import Path


class LDABaselinePreprocessingTests(unittest.TestCase):
    def test_clean_segment_text_removes_transcript_prefix_and_normalizes_whitespace(self):
        from graphgen.evaluation.lda_baseline import clean_segment_text

        raw = "[EN] Roberta | Metsola said from the NULL that   Thank   you,   dear colleagues.  "
        cleaned = clean_segment_text(raw)

        self.assertEqual(cleaned, "Thank you, dear colleagues.")

    def test_load_segment_corpus_collects_nonempty_clean_segments(self):
        from graphgen.evaluation.lda_baseline import load_segment_corpus

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.txt").write_text(
                "[EN] A | B said from the NULL that First useful segment.\n\nSecond useful segment.\n",
                encoding="utf-8",
            )
            (root / "b.txt").write_text(
                "[EN] C | D said from the PPE that Third segment here.\n",
                encoding="utf-8",
            )

            segments = load_segment_corpus(root, min_words=2)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0], "First useful segment.")
        self.assertEqual(segments[1], "Second useful segment.")
        self.assertEqual(segments[2], "Third segment here.")


class LDABaselineSelectionTests(unittest.TestCase):
    def test_select_best_model_prefers_npmi_then_diversity_then_lower_perplexity(self):
        from graphgen.evaluation.lda_baseline import select_best_model

        candidates = [
            {"k": 8, "mean_npmi": 0.11, "mean_umass": -1.80, "topic_diversity": 0.80, "test_perplexity": 1200.0},
            {"k": 10, "mean_npmi": 0.14, "mean_umass": -1.60, "topic_diversity": 0.60, "test_perplexity": 1400.0},
            {"k": 12, "mean_npmi": 0.14, "mean_umass": -1.50, "topic_diversity": 0.75, "test_perplexity": 1600.0},
            {"k": 14, "mean_npmi": 0.14, "mean_umass": -1.50, "topic_diversity": 0.75, "test_perplexity": 1800.0},
        ]

        best = select_best_model(candidates)

        self.assertEqual(best["k"], 12)

    def test_select_best_model_avoids_pathological_high_topic_count_choice(self):
        from graphgen.evaluation.lda_baseline import select_best_model

        candidates = [
            {"k": 6, "mean_npmi": 0.137, "mean_umass": -1.784, "topic_diversity": 0.717, "test_perplexity": 6975.6},
            {"k": 8, "mean_npmi": 0.145, "mean_umass": -1.754, "topic_diversity": 0.613, "test_perplexity": 10615.7},
            {"k": 12, "mean_npmi": 0.148, "mean_umass": -1.551, "topic_diversity": 0.517, "test_perplexity": 19685.3},
            {"k": 24, "mean_npmi": 0.160, "mean_umass": -1.685, "topic_diversity": 0.492, "test_perplexity": 64094.0},
            {"k": 36, "mean_npmi": 0.167, "mean_umass": -1.677, "topic_diversity": 0.469, "test_perplexity": 156665.9},
        ]

        best = select_best_model(candidates)

        self.assertEqual(best["k"], 12)


class LDABaselineEndToEndTests(unittest.TestCase):
    def test_evaluate_lda_baseline_returns_serializable_summary(self):
        from graphgen.evaluation.lda_baseline import evaluate_lda_baseline

        documents = [
            "Energy prices and gas security dominate the European debate.",
            "Ukraine and sanctions shape European security policy.",
            "Migration and asylum remain contested in the Mediterranean.",
            "The energy crisis affects households and industry across Europe.",
            "Support for Ukraine and defence coordination are central themes.",
            "Border control and asylum reform remain politically divisive.",
        ]

        summary = evaluate_lda_baseline(
            documents,
            topic_counts=[2, 3],
            vectorizer_kwargs={"min_df": 1, "max_df": 1.0, "max_features": 500},
            lda_kwargs={"max_iter": 10},
            test_size=0.33,
            random_state=42,
            top_words=8,
        )

        self.assertEqual(summary["corpus"]["documents"], 6)
        self.assertEqual(summary["selected_model"]["k"], 2)
        self.assertIn("mean_npmi", summary["selected_model"])
        self.assertIn("topics", summary["selected_model"])
        self.assertGreater(len(summary["selected_model"]["topics"]), 0)
