#!/usr/bin/env python3
"""Visualize stored reference graph JSON as a PNG image."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize a reference graph JSON.")
    ap.add_argument("--graph", required=True, help="Path to graph JSON")
    ap.add_argument("--out", default="reference_graph.png", help="Output image path")
    ap.add_argument("--min_weight", type=float, default=0.0, help="Filter edges below this weight")
    args = ap.parse_args()

    data = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    g = nx.DiGraph()
    for n in nodes:
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        g.add_node(nid, title=n.get("title", ""))

    for e in edges:
        w = float(e.get("weight", 1.0))
        if w < args.min_weight:
            continue
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if s and t:
            g.add_edge(s, t, weight=w)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, seed=42, k=0.9)

    edge_w = [max(0.6, g[u][v].get("weight", 1.0) / 2.5) for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, width=edge_w, alpha=0.35, arrows=True, arrowsize=10)
    nx.draw_networkx_nodes(g, pos, node_size=240, alpha=0.9)

    labels = {n: n for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    plt.title(f"Reference Graph: {Path(args.graph).name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved graph visualization to {args.out}")


if __name__ == "__main__":
    main()
