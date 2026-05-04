#!/usr/bin/env python3
"""Inspect reference graph and semantic store content from graph JSON."""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Inspect graph edges and semantic-store summaries.')
    ap.add_argument('--graph', required=True, help='Path to reference graph JSON')
    ap.add_argument('--paper_id', default='', help='Optional paper id to inspect in detail')
    ap.add_argument('--max_nodes', type=int, default=10)
    args = ap.parse_args()

    data = json.loads(Path(args.graph).read_text(encoding='utf-8'))
    nodes = data.get('nodes', []) or []
    edges = data.get('edges', []) or []

    print(f"Graph: {args.graph}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")

    with_store = [n for n in nodes if isinstance(n.get('semantic_store', {}), dict) and n.get('semantic_store', {}).get('global_summary')]
    print(f"Nodes with semantic_store.global_summary: {len(with_store)}")

    print("\nSample nodes:")
    for n in nodes[: args.max_nodes]:
        nid = n.get('id')
        title = str(n.get('title', '')).strip()
        ss = n.get('semantic_store', {}) if isinstance(n.get('semantic_store', {}), dict) else {}
        print(f"- id={nid} title={title[:80]}")
        if ss:
            print(f"  global_summary: {(ss.get('global_summary') or '')[:160]}")
            print(f"  main_findings: {(ss.get('main_findings') or '')[:160]}")
            print(f"  main_claims: {(ss.get('main_claims') or '')[:160]}")
            print(f"  evidence_summary: {(ss.get('evidence_summary') or '')[:160]}")

    if args.paper_id:
        print(f"\nDetailed for paper_id={args.paper_id}")
        node = next((n for n in nodes if str(n.get('id', '')).strip() == args.paper_id), None)
        if not node:
            print('Paper not found.')
            return
        print(json.dumps(node, ensure_ascii=False, indent=2)[:8000])

        print("\nConnected edges:")
        for e in edges:
            s = str(e.get('source', '')).strip()
            t = str(e.get('target', '')).strip()
            if s == args.paper_id or t == args.paper_id:
                print(json.dumps(e, ensure_ascii=False))


if __name__ == '__main__':
    main()
