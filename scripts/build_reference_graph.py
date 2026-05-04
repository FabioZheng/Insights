#!/usr/bin/env python3
"""Build and store a weighted paper reference graph.

Input format (JSON/JSONL): each paper object should include:
- id (required)
- title (optional)
- abstract (optional)
- keywords (optional list)
- authors (optional list)
- references (optional list of paper ids this paper cites)

Output graph JSON format:
{
  "nodes": [{"id":..., "title":..., "abstract":..., "keywords":[...], "authors":[...]}],
  "edges": [{"source": citing_id, "target": cited_id, "weight": float}]
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def jaccard_score(a, b) -> float:
    sa = {str(x).strip().lower() for x in (a or []) if str(x).strip()}
    sb = {str(x).strip().lower() for x in (b or []) if str(x).strip()}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def get_references(graph: Dict[str, Any], paper_id: str) -> set[str]:
    out = set()
    for e in graph.get("edges", []) or []:
        if str(e.get("source", "")).strip() == paper_id:
            t = str(e.get("target", "")).strip()
            if t:
                out.add(t)
    return out


def get_citers(graph: Dict[str, Any], paper_id: str) -> set[str]:
    out = set()
    for e in graph.get("edges", []) or []:
        if str(e.get("target", "")).strip() == paper_id:
            s = str(e.get("source", "")).strip()
            if s:
                out.add(s)
    return out


def get_concepts(paper) -> set[str]:
    return {normalize_text(x) for x in (paper.get("keywords", []) or []) if normalize_text(x)}


def compute_graph_weight(paper_a, paper_b, graph) -> tuple[float, dict]:
    id_a = str(paper_a.get("id", "")).strip()
    id_b = str(paper_b.get("id", "")).strip()

    refs_a = get_references(graph, id_a)
    refs_b = get_references(graph, id_b)
    citers_a = get_citers(graph, id_a)
    citers_b = get_citers(graph, id_b)

    title_b = normalize_text(paper_b.get("title", ""))
    blob_a = normalize_text(
        " ".join([
            str(paper_a.get("title", "")),
            str(paper_a.get("abstract", "")),
            str(paper_a.get("summary", "")),
            str(paper_a.get("references_text", "")),
        ])
    )

    direct_citation = 1.0 if id_b in refs_a else 0.0
    reverse_citation = 1.0 if id_a in refs_b else 0.0
    mention_count = blob_a.count(title_b) if title_b else 0
    repeated_mentions_score = min(float(mention_count) / 3.0, 1.0)
    shared_references_score = jaccard_score(refs_a, refs_b)
    co_citation_score = jaccard_score(citers_a, citers_b)
    shared_authors_score = jaccard_score(paper_a.get("authors", []), paper_b.get("authors", []))
    shared_venue_score = 1.0 if normalize_text(paper_a.get("venue", "")) and normalize_text(paper_a.get("venue", "")) == normalize_text(paper_b.get("venue", "")) else 0.0
    shared_concepts_score = jaccard_score(get_concepts(paper_a), get_concepts(paper_b))

    components = {
        "direct_citation": direct_citation,
        "reverse_citation": reverse_citation,
        "repeated_mentions_score": repeated_mentions_score,
        "shared_references_score": shared_references_score,
        "co_citation_score": co_citation_score,
        "shared_authors_score": shared_authors_score,
        "shared_venue_score": shared_venue_score,
        "shared_concepts_score": shared_concepts_score,
    }
    raw_graph_weight = (
        1.0 * direct_citation
        + 0.8 * reverse_citation
        + 0.7 * repeated_mentions_score
        + 0.4 * shared_references_score
        + 0.4 * co_citation_score
        + 0.2 * shared_authors_score
        + 0.2 * shared_venue_score
        + 0.2 * shared_concepts_score
    )
    graph_weight = min(raw_graph_weight / 3.9, 1.0)
    components["raw_graph_weight"] = raw_graph_weight
    return graph_weight, components


def load_papers(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("papers"), list):
        return parsed["papers"]
    raise ValueError("Unsupported input format. Use JSON list, {papers:[...]}, or JSONL.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build weighted reference graph JSON.")
    ap.add_argument("--input", required=True, help="Paper metadata file (.json or .jsonl)")
    ap.add_argument("--out", default="reference_graph.json", help="Output graph path")
    args = ap.parse_args()

    papers = load_papers(Path(args.input))
    paper_by_id = {str(p.get("id", "")).strip(): p for p in papers if str(p.get("id", "")).strip()}

    nodes = []
    for pid, p in paper_by_id.items():
        nodes.append({
            "id": pid,
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "keywords": p.get("keywords", []) or [],
            "authors": p.get("authors", []) or [],
            "year": p.get("year"),
            "venue": p.get("venue", ""),
        })

    graph = {"nodes": nodes, "edges": [], "meta": {}}
    edges = graph["edges"]
    for pid, p in paper_by_id.items():
        refs = p.get("references", []) or []
        for rid in refs:
            rid = str(rid).strip()
            if not rid or rid not in paper_by_id:
                continue
            pa = dict(paper_by_id[pid]); pa["id"] = pid
            pb = dict(paper_by_id[rid]); pb["id"] = rid
            w, comps = compute_graph_weight(pa, pb, graph)
            edge = {"source": pid, "target": rid, "relation": "cites", "weight": round(w, 6), "weight_components": comps}
            edges.append(edge)

    graph["meta"] = {"num_nodes": len(nodes), "num_edges": len(edges)}
    Path(args.out).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_w = (sum(e["weight"] for e in edges) / len(edges)) if edges else 0.0
    print(f"Wrote graph: {args.out}")
    print(f"Nodes={len(nodes)} Edges={len(edges)} AvgWeight={avg_w:.3f}")


if __name__ == "__main__":
    main()
