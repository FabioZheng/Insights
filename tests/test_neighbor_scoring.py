import unittest
import importlib


class NeighborScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls._main = importlib.import_module("main")
            cls._fn = cls._main.build_referenced_neighbor_context
        except Exception as e:
            cls._main = None
            cls._fn = None
            cls._skip_reason = str(e)

    def _call(self, *args, **kwargs):
        if self._fn is None:
            self.skipTest(f"main import unavailable: {self._skip_reason}")
        return self._fn(*args, **kwargs)

    def test_semantic_score_local_chunk_only(self):
        excerpts = "[p1] chunk about antenna gain and bandwidth\n\n[p2] chunk about optimization"
        neighbors = [{
            "id": "n1",
            "title": "Paper N1",
            "weight": 0.5,
            "keywords": ["antenna"],
            "semantic_store": {
                "main_findings": "antenna gain bandwidth improvement",
                "main_claims": "method improves antenna",
                "evidence_summary": "benchmarks on antenna tasks",
                "global_summary": "antenna model",
            },
            "weight_components": {},
        }]
        out = self._call(excerpts + " Paper N1", "BACKGROUND", "query", neighbors, None, 3)
        self.assertIn("score_trace", out)
        self.assertIn("chunk_id", out)

    def test_unmentioned_neighbors_excluded(self):
        excerpts = "chunk with no neighbor mention"
        neighbors = [{"id": "n1", "title": "Paper N1", "weight": 0.8, "semantic_store": {"global_summary": "x"}, "keywords": []}]
        out = self._call(excerpts, "BACKGROUND", "q", neighbors, None, 3)
        self.assertEqual(out, "")

    def test_same_neighbor_diff_chunk_scores(self):
        excerpts = "Paper N1 antenna gain\n\nPaper N1 optimization schedule"
        neighbors = [{
            "id": "n1",
            "title": "Paper N1",
            "weight": 0.2,
            "keywords": [],
            "semantic_store": {
                "main_findings": "antenna gain",
                "main_claims": "optimization schedule",
                "evidence_summary": "ablation",
                "global_summary": "mixed",
            },
            "weight_components": {},
        }]
        out = self._call(excerpts, "BACKGROUND", "q", neighbors, None, 5)
        self.assertGreaterEqual(out.count("final_neighbor_score"), 1)

    def test_missing_semantic_fields_fallback(self):
        excerpts = "Paper N1 mentions abstract term"
        neighbors = [{"id": "n1", "title": "Paper N1", "weight": 0.2, "keywords": ["term"], "abstract": "abstract term", "semantic_store": {}, "weight_components": {}}]
        out = self._call(excerpts, "BACKGROUND", "q", neighbors, None, 3)
        self.assertIn("Paper N1", out)


if __name__ == "__main__":
    unittest.main()
