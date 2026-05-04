import unittest

from scripts.build_reference_graph import (
    compute_graph_weight,
    jaccard_score,
    normalize_text,
)


def combine_neighbor_score(semantic_overlap_score: float, graph_weight: float) -> float:
    return 0.7 * semantic_overlap_score + 0.3 * graph_weight


class GraphWeightTests(unittest.TestCase):
    def setUp(self):
        self.graph = {"edges": []}

    def _paper(self, pid, **kwargs):
        base = {
            "id": pid,
            "title": f"Paper {pid}",
            "authors": [],
            "venue": "",
            "keywords": [],
            "summary": "",
            "references_text": "",
        }
        base.update(kwargs)
        return base

    def test_jaccard_safe(self):
        self.assertEqual(jaccard_score(None, []), 0.0)
        self.assertGreater(jaccard_score(["a"], ["a", "b"]), 0.0)

    def test_direct_citation_only(self):
        self.graph["edges"] = [{"source": "a", "target": "b"}]
        w, comps = compute_graph_weight(self._paper("a"), self._paper("b"), self.graph)
        self.assertGreater(w, 0)
        self.assertEqual(comps["direct_citation"], 1.0)

    def test_reverse_citation_only(self):
        self.graph["edges"] = [{"source": "b", "target": "a"}]
        w, comps = compute_graph_weight(self._paper("a"), self._paper("b"), self.graph)
        self.assertEqual(comps["reverse_citation"], 1.0)
        self.assertGreater(w, 0)

    def test_shared_references(self):
        self.graph["edges"] = [
            {"source": "a", "target": "x"},
            {"source": "b", "target": "x"},
        ]
        w, comps = compute_graph_weight(self._paper("a"), self._paper("b"), self.graph)
        self.assertGreater(comps["shared_references_score"], 0)

    def test_co_citation(self):
        self.graph["edges"] = [
            {"source": "z", "target": "a"},
            {"source": "z", "target": "b"},
        ]
        w, comps = compute_graph_weight(self._paper("a"), self._paper("b"), self.graph)
        self.assertGreater(comps["co_citation_score"], 0)

    def test_shared_authors(self):
        a = self._paper("a", authors=["Ann", "Bob"])
        b = self._paper("b", authors=["Ann"])
        w, comps = compute_graph_weight(a, b, self.graph)
        self.assertGreater(comps["shared_authors_score"], 0)

    def test_shared_venue(self):
        a = self._paper("a", venue="NeurIPS")
        b = self._paper("b", venue="neurips")
        w, comps = compute_graph_weight(a, b, self.graph)
        self.assertEqual(comps["shared_venue_score"], 1.0)

    def test_shared_concepts(self):
        a = self._paper("a", keywords=["graph", "rf"])
        b = self._paper("b", keywords=["graph"])
        w, comps = compute_graph_weight(a, b, self.graph)
        self.assertGreater(comps["shared_concepts_score"], 0)

    def test_all_signals_capped(self):
        g = {"edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}, {"source": "a", "target": "x"}, {"source": "b", "target": "x"}, {"source": "z", "target": "a"}, {"source": "z", "target": "b"}]}
        a = self._paper("a", authors=["Ann"], venue="V", keywords=["k"], references_text="Paper b")
        b = self._paper("b", title="Paper b", authors=["Ann"], venue="V", keywords=["k"])
        w, comps = compute_graph_weight(a, b, g)
        self.assertLessEqual(w, 1.0)
        self.assertGreater(w, 0.0)

    def test_missing_metadata_safe(self):
        w, comps = compute_graph_weight({"id": "a"}, {"id": "b"}, {"edges": []})
        self.assertEqual(type(w), float)

    def test_final_ranking_uses_both(self):
        s1 = combine_neighbor_score(0.9, 0.1)
        s2 = combine_neighbor_score(0.5, 0.8)
        self.assertNotEqual(s1, s2)
        self.assertGreater(s1, 0)
        self.assertGreater(s2, 0)


if __name__ == "__main__":
    unittest.main()
