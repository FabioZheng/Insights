#!/usr/bin/env python3
"""Create one global semantic summary for each paper node in a graph JSON using Ollama.

Adds to each node:
semantic_store:
  global_summary: str
  main_findings: str
  main_claims: str
  evidence_summary: str
"""

import argparse
import json
from pathlib import Path
from typing import List

import requests


def ollama_generate(prompt: str, model: str, base_url: str, num_predict: int = 220) -> str:
    url = base_url.rstrip('/') + '/api/generate'
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.2, 'top_p': 0.9, 'num_predict': num_predict},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get('response', '') or '').strip()


def main() -> None:
    ap = argparse.ArgumentParser(description='Build semantic store in graph nodes using an LLM.')
    ap.add_argument('--graph', required=True, help='Input reference graph JSON')
    ap.add_argument('--out', default='', help='Output graph JSON path (default: overwrite input)')
    ap.add_argument('--model', default='llama3.1:8b')
    ap.add_argument('--ollama_url', default='http://127.0.0.1:11434')
    args = ap.parse_args()

    path = Path(args.graph)
    graph = json.loads(path.read_text(encoding='utf-8'))
    nodes = graph.get('nodes', [])

    for i, node in enumerate(nodes):
        title = str(node.get('title', '')).strip()
        abstract = str(node.get('abstract', '')).strip()
        keywords = node.get('keywords', []) or []
        kw = ', '.join(str(k) for k in keywords[:12])
        source = f"TITLE: {title}\nKEYWORDS: {kw}\nABSTRACT: {abstract}".strip()

        global_prompt = (
            "Summarize the following paper in 3-5 sentences with concrete technical details.\n"
            "Focus on objective contribution and reported outcomes.\n\n"
            f"{source}\n"
        )
        global_summary = ollama_generate(global_prompt, args.model, args.ollama_url, num_predict=220)
        findings_prompt = (
            "Extract the main findings of this paper as concise bullet-like sentences.\n"
            "Focus on concrete outcomes/results.\n\n"
            f"{source}\n"
        )
        claims_prompt = (
            "Extract the central technical claims of this paper as concise bullet-like sentences.\n"
            "State claims factually.\n\n"
            f"{source}\n"
        )
        evidence_prompt = (
            "Summarize how the paper supports/proves its claims and findings.\n"
            "Mention the evidence type (experiments, benchmarks, analysis, ablation, etc.).\n\n"
            f"{source}\n"
        )
        main_findings = ollama_generate(findings_prompt, args.model, args.ollama_url, num_predict=200)
        main_claims = ollama_generate(claims_prompt, args.model, args.ollama_url, num_predict=200)
        evidence_summary = ollama_generate(evidence_prompt, args.model, args.ollama_url, num_predict=220)

        node['semantic_store'] = {
            'global_summary': global_summary,
            'main_findings': main_findings,
            'main_claims': main_claims,
            'evidence_summary': evidence_summary,
        }
        print(f"[{i+1}/{len(nodes)}] semantic_store added for node id={node.get('id')}")

    out_path = Path(args.out) if args.out else path
    out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote semantic graph/store to {out_path}")


if __name__ == '__main__':
    main()
