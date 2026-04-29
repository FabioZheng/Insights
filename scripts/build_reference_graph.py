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
import math
import re
from pathlib import Path
from typing import Dict, List, Any


def _norm_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def compute_edge_weight(src: Dict[str, Any], dst: Dict[str, Any]) -> float:
    src_kw = set(map(str.lower, src.get("keywords", []) or []))
    dst_kw = set(map(str.lower, dst.get("keywords", []) or []))
    kw_overlap = _jaccard(src_kw, dst_kw)

    src_auth = set(map(str.lower, src.get("authors", []) or []))
    dst_auth = set(map(str.lower, dst.get("authors", []) or []))
    author_overlap = _jaccard(src_auth, dst_auth)

    src_txt = _norm_tokens((src.get("title") or "") + " " + (src.get("abstract") or ""))
    dst_txt = _norm_tokens((dst.get("title") or "") + " " + (dst.get("abstract") or ""))
    text_sim = _jaccard(src_txt, dst_txt)

    base = 1.0
    weight = base + 1.5 * kw_overlap + 1.0 * author_overlap + 2.0 * text_sim
    return round(weight, 6)


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
        })

    edges = []
    for pid, p in paper_by_id.items():
        refs = p.get("references", []) or []
        for rid in refs:
            rid = str(rid).strip()
            if not rid or rid not in paper_by_id:
                continue
            w = compute_edge_weight(p, paper_by_id[rid])
            edges.append({"source": pid, "target": rid, "weight": w})

    graph = {"nodes": nodes, "edges": edges, "meta": {"num_nodes": len(nodes), "num_edges": len(edges)}}
    Path(args.out).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_w = (sum(e["weight"] for e in edges) / len(edges)) if edges else 0.0
    print(f"Wrote graph: {args.out}")
    print(f"Nodes={len(nodes)} Edges={len(edges)} AvgWeight={avg_w:.3f}")


if __name__ == "__main__":
    main()
