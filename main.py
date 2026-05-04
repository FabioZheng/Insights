#!/usr/bin/env python3
"""
paper_to_insight_text.py — Offline scientific paper -> "Technological Insight" (TEXT ONLY)

v3 fixes:
1. STRICTER TEMPLATE DETECTION  - requires 6/8 fields AND TITLE present (was: 3/8)
2. TIGHTER HEADING MATCHING     - prefix match requires ≥60% of field name (was: ≥8 chars)
3. FIELD COUNT VALIDATION       - forces reformat pass if <6 fields found after Pass 1
4. SAFER strip_markdown()       - DOTALL flag removed; heading regex made non-greedy per line
5. TIGHTENING PRESERVES STRUCTURE - re-validates substructure after tightening; reverts if broken
6. BETTER REFORMAT PROMPT       - numbered field list at the top for small-model compliance
7. COMPLETENESS RETRY           - if last field is incomplete, re-generates with continuation prompt

v4 patches (minimal):
8. FIELD-SPECIFIC RETRIEVAL     - each field gets a dedicated excerpt set via FIELD_QUERIES
9. FIRST-PAGE PATH              - TITLE/SUBTITLE/RESEARCH GROUP use first-page text directly
10. NUMERIC EVIDENCE POOL       - lines with units/metrics appended to metric-heavy sections
11. SOFT ANCHOR BONUS           - anchors boost chunk ranking instead of hard-filtering chunks
12. FIXED TECH INTRO RULE TEXT  - malformed string literal in build_single_section_prompt fixed
13. JSON SIDE-OUTPUT            - .json version of parsed sections saved alongside .txt

Usage:
  python paper_to_insight_text.py --pdf paper.pdf --model llama3.1:8b --out insight.txt

Optional flags:
  --ollama_url          http://127.0.0.1:11434
  --no_semantic         disable semantic retrieval
  --embed_model         sentence-transformers/all-MiniLM-L6-v2
  --top_k               15
  --include_first_pages 3
  --max_excerpts_chars  12000
  --max_chunk_chars     900
  --min_fields          6   (minimum filled fields to accept Pass 1 without reformat)
"""

import argparse
import difflib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import yaml

# PDF extraction
try:
    import pdfplumber
except ImportError:
    sys.exit("Missing dependency: pdfplumber.  Run: pip install pdfplumber")

# Optional semantic retrieval
_HAS_SEMANTIC = False
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _HAS_SEMANTIC = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: int
    page_start: int
    page_end: int
    text: str


DEFAULT_INSIGHT_SCHEMA = {
    "sections": [
        {
            "name": "TITLE",
            "query": "paper title name of the work",
            "description": "Write a concise, specific paper title on one line.",
        },
        {
            "name": "SUBTITLE",
            "query": "value proposition key contribution one sentence",
            "description": "Write one sentence describing practical value/positioning.",
        },
        {
            "name": "BACKGROUND",
            "query": "motivation problem statement prior art limitations background",
            "description": (
                "Under 200 words. Cover (1) why the field matters and (2) current limitations. "
                "Prefer concrete details from excerpts."
            ),
        },
        {
            "name": "TECHNOLOGICAL INTRODUCTION",
            "query": "method technique design innovation approach implementation",
            "description": (
                "Under 350 words total. Must include: Methodology & achievement (1-3 sentences), "
                "3 Main Innovations with exactly 3 bullets, and a mini markdown table "
                "(Method | Key features) with 3-6 rows."
            ),
        },
        {
            "name": "RESEARCH GROUP",
            "query": "author affiliation university institution research group",
            "description": "State lead researcher/group, affiliation, and research area if present.",
        },
        {
            "name": "TECHNICAL ADVANTAGES",
            "query": "performance improvement benchmark comparison result advantage",
            "description": (
                "Under 70 words. Use bullets. Every bullet must contain concrete "
                "numbers/ranges/comparisons from excerpts."
            ),
        },
        {
            "name": "CURRENT STATUS & FUTURE CHALLENGES",
            "query": "current state future work limitation challenge outlook",
            "description": "Under 40 words.",
        },
        {
            "name": "BUSINESS SUGGESTION",
            "query": "application industry market potential use case deployment",
            "description": "Write 1-2 complete, domain-specific sentences ending with a period.",
        },
    ]
}

FIELDS: List[str] = []
FIELD_QUERIES: Dict[str, str] = {}
FIELD_DESCRIPTIONS: Dict[str, str] = {}

# v4: regex for lines containing measurement units — used by numeric evidence helper
_NUMERIC_UNITS_RE = re.compile(
    r"\b\d[\d.,]*\s*(?:GHz|MHz|kHz|dBi|dB|%|mm|cm|μm|nm|ns|ps|W|mW|dBm)\b",
    re.IGNORECASE,
)

apply_insight_schema(DEFAULT_INSIGHT_SCHEMA)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_ws(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def safe_cut_chars(txt: str, n: int) -> str:
    if len(txt) <= n:
        return txt
    cut = txt[:n]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " ..."


def strip_markdown(text: str) -> str:
    """
    Remove markdown decoration LLMs commonly add around headings.
    FIX v3: operate line-by-line for bold/italic to avoid cross-line greedy matches.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # **FIELD**: or __FIELD__: -> FIELD:  (bold wrapper around ALL-CAPS heading)
        line = re.sub(r"[*_]{1,3}([A-Z][A-Z &]{2,})[*_]{1,3}(\s*:)", r"\1\2", line)
        # Inline bold/italic (safe per-line, non-greedy)
        line = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)
        line = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", line)
        # Leading ### headings
        line = re.sub(r"^#{1,6}\s*", "", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def apply_insight_schema(schema: Dict[str, object]) -> None:
    """Apply external schema to global section settings."""
    sections = schema.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Invalid insight config: 'sections' must be a non-empty list.")

    fields: List[str] = []
    queries: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}

    for i, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            raise ValueError(f"Invalid insight config: section #{i} must be a mapping.")
        name = str(section.get("name", "")).strip().upper()
        if not name:
            raise ValueError(f"Invalid insight config: section #{i} missing 'name'.")
        if name in descriptions:
            raise ValueError(f"Invalid insight config: duplicate section name '{name}'.")
        fields.append(name)
        queries[name] = str(section.get("query", "")).strip()
        descriptions[name] = str(section.get("description", "")).strip()

    global FIELDS, FIELD_QUERIES, FIELD_DESCRIPTIONS
    FIELDS = fields
    FIELD_QUERIES = queries
    FIELD_DESCRIPTIONS = descriptions


def load_insight_schema(config_path: str) -> None:
    """Load section names, retrieval queries, and section descriptions from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}
    if not isinstance(schema, dict):
        raise ValueError("Invalid insight config: root must be a mapping/object.")
    apply_insight_schema(schema)


# ---------------------------------------------------------------------------
# PDF -> pages
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = normalize_ws(text)
            if text:
                pages.append((i + 1, text))
    return pages


# ---------------------------------------------------------------------------
# Pages -> chunks
# ---------------------------------------------------------------------------
def make_chunks(
    pages: List[Tuple[int, str]],
    max_chars: int = 3500,
    overlap_chars: int = 400,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_len = 0
    chunk_id = 0
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def flush(keep_overlap: bool) -> None:
        nonlocal chunk_id, buf, buf_len, cur_start, cur_end
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page_start=cur_start or 1,
                page_end=cur_end or (cur_start or 1),
                text=text,
            ))
            chunk_id += 1
        if keep_overlap and text:
            tail = text[-overlap_chars:]
            buf, buf_len, cur_start = [tail], len(tail), cur_end
        else:
            buf, buf_len, cur_start = [], 0, None
        cur_end = None

    for page_no, page_text in pages:
        for para in re.split(r"\n\s*\n", page_text):
            para = para.strip()
            if not para:
                continue
            if cur_start is None:
                cur_start = page_no
            cur_end = page_no
            if buf_len + len(para) + 2 > max_chars and buf:
                flush(keep_overlap=True)
            buf.append(para)
            buf_len += len(para) + 1
            if buf_len >= max_chars:
                flush(keep_overlap=True)
    flush(keep_overlap=False)
    return chunks


# ---------------------------------------------------------------------------
# Dynamic keyword extraction
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
    "with", "that", "this", "we", "our", "by", "on", "at", "as", "from",
    "which", "be", "been", "have", "has", "can", "also", "it", "its",
    "their", "they", "these", "those", "such", "not", "but", "both",
    "more", "than", "into", "via", "using", "used", "based", "new",
    "show", "shows", "result", "results", "while", "each", "all",
    "however", "thus", "since", "due", "through", "between", "under",
    "about", "further", "where", "when", "how", "what", "use",
    "doi", "org", "http", "https", "www", "vol", "volume", "article",
    "nature", "journal", "page", "pages", "figure", "fig", "table",
    "supplementary", "note", "ref", "refs", "received", "accepted",
    "published", "online", "check", "updates", "springer", "ieee",
    "elsevier", "acm", "arxiv", "preprint", "copyright",
}


def extract_paper_keywords(
    pages: List[Tuple[int, str]], n_keywords: int = 30
) -> List[str]:
    first_text = " ".join(t for _, t in pages[:3])
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", first_text.lower())
    tokens = [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]

    bigrams  = [f"{tokens[i]} {tokens[i+1]}"                for i in range(len(tokens) - 1)]
    trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]

    scored: List[Tuple[float, str]] = []
    for term, cnt in Counter(trigrams).items():
        if cnt >= 2:
            scored.append((cnt * 3.0, term))
    for term, cnt in Counter(bigrams).items():
        if cnt >= 2:
            scored.append((cnt * 2.0, term))
    for term, cnt in Counter(tokens).items():
        if cnt >= 3 and len(term) >= 5:
            scored.append((cnt * 1.0, term))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen: List[str] = []
    for _, term in scored:
        if any(term in kept for kept in seen):
            continue
        seen.append(term)
        if len(seen) >= n_keywords:
            break
    return seen


def build_dynamic_query(keywords: List[str]) -> str:
    return " ".join(keywords[:25])


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def keyword_score(query: str, text: str) -> float:
    q_terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not q_terms:
        return 0.0
    t = text.lower()
    return sum(t.count(term) for term in q_terms) / max(1.0, math.log(10 + len(text)))


class Retriever:
    def __init__(self, chunks: List[Chunk], semantic: bool = True,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.semantic_enabled = semantic and _HAS_SEMANTIC
        self._model = None
        self._index = None
        if self.semantic_enabled:
            self._build(embed_model)

    def _build(self, embed_model: str) -> None:
        try:
            self._model = SentenceTransformer(embed_model)
        except Exception as e:
            print(f"[WARN] Semantic retrieval disabled: {e}")
            self.semantic_enabled = False
            return
        texts = [c.text for c in self.chunks]
        emb = self._model.encode(texts, batch_size=16, show_progress_bar=True,
                                 normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        self._index = idx

    def search(self, query: str, top_k: int = 10) -> List[Chunk]:
        if self.semantic_enabled and self._index is not None:
            q = self._model.encode([query], normalize_embeddings=True)
            _, idxs = self._index.search(np.asarray(q, dtype=np.float32), top_k)
            return [self.chunks[int(i)] for i in idxs[0] if i >= 0]
        scored = sorted(
            [(keyword_score(query, c.text), c) for c in self.chunks],
            key=lambda x: x[0], reverse=True,
        )
        return [c for _, c in scored[:top_k]]


def chunk_matches_anchors(text: str, anchors: List[str]) -> bool:
    t = text.lower()
    return any(a.lower() in t for a in anchors)


# v4: count how many anchors appear in a chunk — used as a soft ranking bonus
def _anchor_bonus(text: str, anchors: List[str]) -> int:
    t = text.lower()
    return sum(1 for a in anchors if a.lower() in t)


def select_chunks(
    chunks: List[Chunk], retriever: Retriever, query: str,
    top_k: int, include_first_pages: int, anchors: List[str],
) -> List[Chunk]:
    first_page_chunks = [c for c in chunks if c.page_start <= include_first_pages]
    retrieved = retriever.search(query, top_k=top_k)

    by_id: Dict[int, Chunk] = {}
    for c in first_page_chunks + retrieved:
        by_id[c.chunk_id] = c
    merged = list(by_id.values())

    # v4: soft anchor bonus — rank by anchor hits descending; keep all chunks.
    # Previously anchors were used as a hard filter (chunks without anchors discarded).
    # Now every chunk is kept, but anchor-rich chunks float to the top.
    merged.sort(key=lambda c: _anchor_bonus(c.text, anchors), reverse=True)
    return merged


def build_excerpts_block(
    chunks: List[Chunk],
    max_total_chars: int = 12000,
    max_chunk_chars: int = 900,
) -> str:
    parts: List[str] = []
    total = 0
    for c in chunks:
        txt = safe_cut_chars((c.text or "").strip(), max_chunk_chars)
        if not txt:
            continue
        block = f"[p{c.page_start}-p{c.page_end}] {txt}"
        if total + len(block) + 2 > max_total_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def infer_paper_id_from_pages(
    pages: List[Tuple[int, str]],
    node_titles: Dict[str, str],
    explicit_paper_id: str = "",
) -> str:
    if explicit_paper_id:
        return explicit_paper_id

    if not pages:
        return ""

    first_page = pages[0][1]
    line_candidates = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    if not line_candidates:
        return ""

    candidate = line_candidates[0]
    if len(line_candidates) > 1 and len(candidate.split()) < 3:
        candidate = line_candidates[1]

    if candidate in node_titles:
        return candidate

    norm_to_id = {_norm_key(v): k for k, v in node_titles.items() if v}
    cand_norm = _norm_key(candidate)
    if cand_norm in norm_to_id:
        return norm_to_id[cand_norm]

    fuzzy = difflib.get_close_matches(cand_norm, list(norm_to_id.keys()), n=1, cutoff=0.65)
    if fuzzy:
        return norm_to_id[fuzzy[0]]
    return ""


def build_reference_neighbor_context(
    graph_path: str,
    pages: List[Tuple[int, str]],
    explicit_paper_id: str = "",
    max_neighbors: int = 8,
) -> List[Dict[str, object]]:
    """
    Load a stored reference graph and build neighbor context for the current paper.
    This context can be appended to excerpt blocks to improve generation quality.
    """
    if not graph_path or not os.path.exists(graph_path):
        return []

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return []

    node_map: Dict[str, Dict[str, object]] = {}
    node_titles: Dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        node_map[nid] = n
        node_titles[nid] = str(n.get("title", "")).strip()

    paper_id = infer_paper_id_from_pages(pages, node_titles, explicit_paper_id)
    if not paper_id or paper_id not in node_map:
        return []

    neighbor_scores: Dict[str, float] = {}
    neighbor_components: Dict[str, Dict[str, float]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = str(e.get("source", "")).strip()
        dst = str(e.get("target", "")).strip()
        weight = float(e.get("weight", 1.0))
        comps = e.get("weight_components", {}) if isinstance(e.get("weight_components", {}), dict) else {}
        if src == paper_id and dst in node_map:
            neighbor_scores[dst] = neighbor_scores.get(dst, 0.0) + weight
            base = neighbor_components.get(dst, {})
            for k, v in comps.items():
                if isinstance(v, (int, float)):
                    base[k] = base.get(k, 0.0) + float(v)
            neighbor_components[dst] = base
        elif dst == paper_id and src in node_map:
            neighbor_scores[src] = neighbor_scores.get(src, 0.0) + weight
            base = neighbor_components.get(src, {})
            for k, v in comps.items():
                if isinstance(v, (int, float)):
                    base[k] = base.get(k, 0.0) + float(v)
            neighbor_components[src] = base

    if not neighbor_scores:
        return []

    ranked = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
    neighbors: List[Dict[str, object]] = []
    for nid, score in ranked:
        node = node_map[nid]
        title = str(node.get("title", "")).strip()
        abstract = safe_cut_chars(str(node.get("abstract", "")).strip(), 240)
        keywords = node.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)] if str(keywords).strip() else []
        neighbors.append({
            "id": nid,
            "weight": float(score),
            "title": title,
            "abstract": abstract,
            "keywords": [str(k) for k in keywords[:8]],
            "semantic_store": node.get("semantic_store", {}),
            "weight_components": neighbor_components.get(nid, {}),
        })
    return neighbors


def _reference_is_mentioned(excerpts: str, neighbor: Dict[str, object]) -> bool:
    t = excerpts.lower()
    nid = str(neighbor.get("id", "")).strip().lower()
    title = str(neighbor.get("title", "")).strip().lower()
    if nid and nid in t:
        return True
    if title and title in t:
        return True
    return False


def build_bibliography_citation_map(
    pages: List[Tuple[int, str]],
    neighbors: List[Dict[str, object]],
) -> Dict[str, str]:
    """
    Build mapping from citation markers like [12] -> neighbor paper id using
    bibliography lines and fuzzy title matching.
    """
    if not pages or not neighbors:
        return {}

    all_text = "\n".join(t for _, t in pages)
    lines = [ln.strip() for ln in all_text.splitlines() if ln.strip()]
    ref_lines = [ln for ln in lines if re.match(r"^\[\d+\]\s+", ln)]
    if not ref_lines:
        return {}

    title_to_id: Dict[str, str] = {}
    for nb in neighbors:
        t = str(nb.get("title", "")).strip()
        nid = str(nb.get("id", "")).strip()
        if t and nid:
            title_to_id[_norm_key(t)] = nid

    if not title_to_id:
        return {}

    mapping: Dict[str, str] = {}
    for ln in ref_lines:
        m = re.match(r"^\[(\d+)\]\s+(.*)$", ln)
        if not m:
            continue
        idx, rest = m.group(1), m.group(2)
        norm_rest = _norm_key(rest)
        candidates = difflib.get_close_matches(norm_rest, list(title_to_id.keys()), n=1, cutoff=0.55)
        if candidates:
            mapping[idx] = title_to_id[candidates[0]]
    return mapping


def build_referenced_neighbor_context(
    excerpts: str,
    field: str,
    field_query: str,
    neighbors: List[Dict[str, object]],
    citation_map: Optional[Dict[str, str]] = None,
    max_neighbors: int = 3,
) -> str:
    """
    Add neighbor context only when the current excerpt explicitly mentions the reference.
    Also keep only neighbor info relevant to the current field query.
    """
    if not excerpts.strip() or not neighbors:
        return ""

    query_terms = {
        t for t in re.findall(r"[a-zA-Z0-9]+", field_query.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    }
    if not query_terms:
        query_terms = {"method", "result", "performance"}

    scored: List[Tuple[float, Dict[str, object], str, str]] = []

    def _extract_relevant_neighbor_snippet(text: str, terms: set, max_sentences: int = 2) -> str:
        if not text.strip():
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return safe_cut_chars(text.strip(), 200)
        sent_scored: List[Tuple[int, str]] = []
        for sent in sentences:
            s_low = sent.lower()
            overlap = sum(1 for q in terms if q in s_low)
            if overlap > 0:
                sent_scored.append((overlap, sent))
        if not sent_scored:
            return safe_cut_chars(text.strip(), 200)
        sent_scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for _, s in sent_scored[:max_sentences]]
        return safe_cut_chars(" ".join(chosen), 200)

    cited_ids = set()
    if citation_map:
        for cidx in re.findall(r"\[(\d+)\]", excerpts):
            nid = citation_map.get(cidx)
            if nid:
                cited_ids.add(nid)

    for nb in neighbors:
        nid = str(nb.get("id", "")).strip()
        explicitly_mentioned = _reference_is_mentioned(excerpts, nb)
        bracket_cited = nid in cited_ids if nid else False
        if not explicitly_mentioned and not bracket_cited:
            continue
        title = str(nb.get("title", "")).strip()
        abstract = str(nb.get("abstract", "")).strip()
        semantic_store = nb.get("semantic_store", {}) if isinstance(nb.get("semantic_store", {}), dict) else {}
        global_summary = str(semantic_store.get("global_summary", "")).strip()
        main_findings = str(semantic_store.get("main_findings", "")).strip()
        main_claims = str(semantic_store.get("main_claims", "")).strip()
        evidence_summary = str(semantic_store.get("evidence_summary", "")).strip()
        content_for_context = "\n".join(
            x for x in [main_findings, main_claims, evidence_summary, global_summary] if x
        ).strip() or abstract
        keywords = nb.get("keywords", [])
        kw_txt = ", ".join(str(k) for k in keywords) if isinstance(keywords, list) else str(keywords)
        findings_blob = main_findings.lower()
        claims_blob = main_claims.lower()
        evidence_blob = evidence_summary.lower()
        summary_blob = global_summary.lower()
        title_kw_blob = f"{title} {kw_txt}".lower()
        overlap_findings = sum(1 for q in query_terms if q in findings_blob)
        overlap_claims = sum(1 for q in query_terms if q in claims_blob)
        overlap_evidence = sum(1 for q in query_terms if q in evidence_blob)
        overlap_summary = sum(1 for q in query_terms if q in summary_blob)
        overlap_title_kw = sum(1 for q in query_terms if q in title_kw_blob)
        overlap = overlap_findings + overlap_claims + overlap_evidence + overlap_summary + overlap_title_kw
        if overlap <= 0:
            continue
        snippet = _extract_relevant_neighbor_snippet(content_for_context, query_terms, max_sentences=2)
        semantic_overlap_score = min(
            (
                (1.2 * overlap_findings)
                + (1.1 * overlap_claims)
                + (1.0 * overlap_evidence)
                + (0.9 * overlap_summary)
                + (0.8 * overlap_title_kw)
            ) / 6.0,
            1.0,
        )
        graph_weight = float(nb.get("weight", 0.0))
        final_neighbor_score = (0.7 * semantic_overlap_score) + (0.3 * graph_weight)
        scored.append((final_neighbor_score, nb, kw_txt, snippet, semantic_overlap_score, graph_weight))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    lines = ["REFERENCED NEIGHBOR CONTEXT (only cited/mentioned neighbors):"]
    for relevance, nb, kw_txt, snippet, semantic_overlap_score, graph_weight in scored[:max_neighbors]:
        lines.append(
            f"- id={nb.get('id')} weight={float(nb.get('weight', 1.0)):.3f} score={relevance:.3f} "
            f"title={str(nb.get('title', '')).strip()}"
        )
        if kw_txt:
            lines.append(f"  relevant_keywords: {kw_txt}")
        if snippet:
            lines.append(f"  relevant_summary_snippet: {snippet}")
        if main_findings:
            lines.append(f"  findings: {safe_cut_chars(main_findings, 180)}")
        if main_claims:
            lines.append(f"  claims: {safe_cut_chars(main_claims, 180)}")
        if evidence_summary:
            lines.append(f"  evidence: {safe_cut_chars(evidence_summary, 180)}")
        comps = nb.get("weight_components", {}) if isinstance(nb.get("weight_components", {}), dict) else {}
        lines.append("  score_trace:")
        lines.append(f"    semantic_overlap_score: {semantic_overlap_score:.4f}")
        lines.append(f"    graph_weight: {graph_weight:.4f}")
        lines.append(f"    final_neighbor_score: {relevance:.4f}")
        lines.append("    graph_components:")
        lines.append(f"      direct_citation: {float(comps.get('direct_citation', 0.0)):.4f}")
        lines.append(f"      reverse_citation: {float(comps.get('reverse_citation', 0.0)):.4f}")
        lines.append(f"      explicit_mention: {float(comps.get('explicit_mention', 0.0)):.4f}")
        lines.append(f"      shared_references_score: {float(comps.get('shared_references_score', 0.0)):.4f}")
        lines.append(f"      co_citation_score: {float(comps.get('co_citation_score', 0.0)):.4f}")
        lines.append(f"      shared_authors_score: {float(comps.get('shared_authors_score', 0.0)):.4f}")
        lines.append(f"      shared_venue_score: {float(comps.get('shared_venue_score', 0.0)):.4f}")
        lines.append(f"      shared_concepts_score: {float(comps.get('shared_concepts_score', 0.0)):.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v4: field-specific retrieval helpers
# ---------------------------------------------------------------------------

def get_first_page_text(pages: List[Tuple[int, str]], n: int = 2) -> str:
    """Return concatenated text of the first n pages — used for metadata fields."""
    return "\n\n".join(t for _, t in pages[:n])


def extract_numeric_evidence(chunks: List[Chunk], max_lines: int = 20) -> str:
    """
    v4: scan all chunks for lines that contain measurement units/metrics.
    Returns a compact block appended to prompts for metric-heavy sections.
    Lightweight — no table parsing, just line-level grep.
    """
    seen: set = set()
    lines_out: List[str] = []
    for c in chunks:
        for line in c.text.splitlines():
            if _NUMERIC_UNITS_RE.search(line):
                stripped = line.strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    lines_out.append(stripped)
                    if len(lines_out) >= max_lines:
                        break
        if len(lines_out) >= max_lines:
            break
    if not lines_out:
        return ""
    return "NUMERIC EVIDENCE CANDIDATES:\n" + "\n".join(f"  {l}" for l in lines_out)


def select_chunks_for_field(
    field: str,
    chunks: List[Chunk],
    retriever: Retriever,
    anchors: List[str],
    top_k: int,
    include_first_pages: int,
    max_excerpts_chars: int,
    max_chunk_chars: int,
) -> str:
    """
    v4: retrieve a dedicated excerpt block for a single field using FIELD_QUERIES.
    Falls back to keyword search if semantic is disabled.
    Reuses existing select_chunks + build_excerpts_block helpers.
    """
    query = FIELD_QUERIES.get(field, "")
    chosen = select_chunks(chunks, retriever, query,
                           top_k=top_k,
                           include_first_pages=include_first_pages,
                           anchors=anchors)
    return build_excerpts_block(chosen,
                                max_total_chars=max_excerpts_chars,
                                max_chunk_chars=max_chunk_chars)
# ---------------------------------------------------------------------------
def ollama_generate(
    prompt: str, model: str,
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.2, top_p: float = 0.9,
    num_predict: int = 2000, timeout_s: int = 900,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "top_p": top_p,
                    "num_predict": num_predict},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()


def ollama_with_retry(
    prompt_fn, model: str, base_url: str, excerpts: str,
    num_predict: int = 2000, timeout_s: int = 900,
) -> str:
    budgets = [len(excerpts), 12000, 9000, 6500, 4500]
    last = ""
    for b in budgets:
        ex = excerpts[:b]
        try:
            out = ollama_generate(prompt_fn(ex), model=model, base_url=base_url,
                                  num_predict=num_predict, timeout_s=timeout_s)
        except Exception as e:
            last = str(e)
            continue
        if out.strip():
            return out.strip()
        last = f"empty response at budget={b}"
    raise RuntimeError(last or "Ollama returned empty after all retries.")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_HEADING_EXAMPLE = (
    "TITLE: <concise paper title>\n"
    "SUBTITLE: <one-sentence value proposition>\n"
    "BACKGROUND: <content>\n"
    "TECHNOLOGICAL INTRODUCTION: <content>\n"
    "RESEARCH GROUP: <Lead researcher name>, <Affiliation>, <Research area>\n"
    "TECHNICAL ADVANTAGES: <content>\n"
    "CURRENT STATUS & FUTURE CHALLENGES: <content>\n"
    "BUSINESS SUGGESTION: <content>"
)

def build_field_list_block() -> str:
    fields_numbered = "\n".join(f"  {i+1}. {f}:" for i, f in enumerate(FIELDS))
    return (
        f"YOU MUST OUTPUT EXACTLY THESE {len(FIELDS)} SECTIONS IN THIS ORDER "
        "(all-caps, no markdown, no bold, no #):\n"
        + fields_numbered
        + "\n\n"
    )


def build_template_rules() -> str:
    return (
        build_field_list_block()
        + "OUTPUT FORMAT EXAMPLE:\n"
        + _HEADING_EXAMPLE
        + "\n\n"
        "STRICT WORD LIMITS:\n"
        "- BACKGROUND: under 200 words. Cover (1) why the field matters; (2) current limitations.\n"
        "- TECHNOLOGICAL INTRODUCTION: under 350 words TOTAL. Must contain:\n"
        "    Methodology & achievement: 1-3 sentences.\n"
        "    3 Main Innovations:\n"
        "    - <innovation 1>\n"
        "    - <innovation 2>\n"
        "    - <innovation 3>\n"
        "    Mini Table (Method | Key features):\n"
        "    | Method | Key features |\n"
        "    | ... | ... |\n"
        "    (3-6 rows)\n"
        "- TECHNICAL ADVANTAGES: under 70 words. Use bullets. MUST include specific numbers from the paper.\n"
        "- CURRENT STATUS & FUTURE CHALLENGES: under 40 words.\n"
        "- BUSINESS SUGGESTION: 1-2 complete sentences specific to this paper's technology domain.\n\n"
        "CRITICAL QUANTITATIVE REQUIREMENTS:\n"
        "★ ALWAYS extract and report SPECIFIC NUMBERS from the excerpts:\n"
        "  - Percentage improvements (e.g., '17.2% wider bandwidth')\n"
        "  - Absolute values (e.g., '300 EM simulations vs 1000 baseline')\n"
        "  - Ranges (e.g., '6-14 variables reduced to 2-5 dimensions')\n"
        "  - Comparisons (e.g., '40% sample reduction vs full kriging')\n"
        "  - Time/cost savings (e.g., '70% faster than PSO')\n"
        "★ In TECHNICAL ADVANTAGES, every bullet MUST anchor claims with concrete metrics.\n"
        "★ Prefer RANGES over single numbers when ranges are given.\n"
        "★ Avoid vague phrases like 'significant gains' — always use concrete numbers.\n\n"
        "CRITICAL FORMATTING RULES:\n"
        "- Do NOT wrap headings in **, __, or # characters.\n"
        "- Each heading must start at the beginning of a new line, followed by a colon.\n"
        "- Do NOT add any preamble before TITLE:.\n"
        f"- COMPLETE all {len(FIELDS)} sections fully. Do NOT end abruptly mid-sentence.\n"
        "- If information is absent from EXCERPTS write: Not found in provided text.\n"
    )


def build_insight_prompt(excerpts: str) -> str:
    return (
        "You are a technical analyst writing a 'Technological Insight' report based ONLY on the EXCERPTS below.\n"
        "Your report will be read by researchers and business stakeholders who need SPECIFIC, QUANTITATIVE details.\n\n"
        "GOLDEN RULE: Extract and report ALL concrete numbers, percentages, ranges, and comparisons from the excerpts.\n"
        "Never use vague language when specific metrics are available in the text.\n\n"
        + build_template_rules()
        + "\nEXCERPTS:\n"
        + excerpts
        + "\n\nBegin your output with TITLE: on the very first line. "
        f"Write ALL {len(FIELDS)} sections completely — do not stop early.\n"
    )


def build_reformat_prompt(excerpts: str, draft: str) -> str:
    # FIX v3: numbered list at the very top for maximum small-model compliance
    return (
        "REFORMAT TASK: Convert the DRAFT below into the exact 8-section 'Technological Insight' template.\n\n"
        + build_template_rules()
        + "\nEXCERPTS (source of truth for facts and numbers):\n"
        + excerpts
        + "\n\nDRAFT (reformat this — add missing numbers from EXCERPTS if absent):\n"
        + draft
        + "\n\nBegin your output with TITLE: on the very first line. "
        "Output ALL 8 sections completely — do not end abruptly.\n"
    )


def build_continuation_prompt(excerpts: str, partial: str, missing_fields: List[str]) -> str:
    """Used when final output is missing some trailing fields."""
    fields_str = ", ".join(missing_fields)
    return (
        f"The following sections are MISSING from a Technological Insight report: {fields_str}\n\n"
        "Write ONLY those missing sections, using the EXCERPTS as the source of truth.\n\n"
        + build_field_list_block()
        + "CRITICAL FORMATTING RULES:\n"
        "- Use ALL-CAPS heading followed by colon (e.g. RESEARCH GROUP:)\n"
        "- No markdown, no bold, no # symbols.\n"
        "- BUSINESS SUGGESTION must be 1-2 complete sentences ending with a period.\n\n"
        "EXCERPTS:\n"
        + excerpts
        + "\n\nPARTIAL REPORT SO FAR (do NOT repeat these sections):\n"
        + partial
        + f"\n\nNow write ONLY the missing sections ({fields_str}), starting with the first missing heading:\n"
    )


def build_single_section_prompt(field: str, excerpts: str, context_so_far: str = "",
                                numeric_evidence: str = "") -> str:
    """Generate exactly one requested field at a time."""
    prior_context = ""
    if context_so_far.strip():
        prior_context = (
            "CONTEXT FROM PREVIOUSLY GENERATED SECTIONS (for consistency only, do not repeat verbatim):\n"
            + context_so_far.strip()
            + "\n\n"
        )

    return (
        "You are generating a Technological Insight report ONE SECTION AT A TIME based ONLY on EXCERPTS.\n"
        "Return exactly one heading and its content, with this exact heading format:\n"
        f"{field}: <content>\n\n"
        "Do not output any other headings, preamble, markdown fences, or explanations.\n"
        "If data is absent, write exactly: Not found in provided text.\n\n"
        f"SECTION-SPECIFIC RULES for {field}:\n"
        f"- {FIELD_DESCRIPTIONS.get(field, 'Use only facts from excerpts.')}\n\n"
        "Global numeric rule: include concrete numbers/percentages/ranges/comparisons whenever available.\n\n"
        + prior_context
        + "EXCERPTS:\n"
        + excerpts
        # v4: append numeric evidence block for metric-heavy sections
        + ("\n\n" + numeric_evidence if numeric_evidence else "")
        + "\n"
    )


def generate_sections_separately(
    excerpts: str,
    model: str,
    base_url: str,
    # v4: extra args for field-specific retrieval
    chunks: Optional[List[Chunk]] = None,
    retriever: Optional["Retriever"] = None,
    anchors: Optional[List[str]] = None,
    pages: Optional[List[Tuple[int, str]]] = None,
    top_k: int = 15,
    include_first_pages: int = 3,
    max_excerpts_chars: int = 12000,
    max_chunk_chars: int = 900,
    reference_neighbors: Optional[List[Dict[str, object]]] = None,
    citation_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Primary generation path: produce each template field in separate model calls.

    v4: each field now gets its own excerpt block via FIELD_QUERIES.
    TITLE/SUBTITLE/RESEARCH GROUP use first-page text to capture author metadata.
    TECHNOLOGICAL INTRODUCTION and TECHNICAL ADVANTAGES get a numeric evidence block.
    """
    parsed: Dict[str, str] = {f: "" for f in FIELDS}
    context_lines: List[str] = []

    # v4: pre-build first-page text for metadata fields
    first_page_text = get_first_page_text(pages) if pages else ""

    # v4: pre-build numeric evidence once (cheap scan, reused across two fields)
    numeric_evidence = extract_numeric_evidence(chunks) if chunks else ""

    # v4: fields that should use first-page text instead of normal retrieval
    _FIRST_PAGE_FIELDS = {"TITLE", "SUBTITLE", "RESEARCH GROUP"}
    # v4: fields that receive the numeric evidence block appended to their prompt
    _NUMERIC_FIELDS = {"TECHNOLOGICAL INTRODUCTION", "TECHNICAL ADVANTAGES"}

    for field in FIELDS:
        context_so_far = "\n\n".join(context_lines)

        # v4: select field-specific excerpts when retriever is available
        if retriever is not None and chunks is not None and field not in _FIRST_PAGE_FIELDS:
            field_excerpts = select_chunks_for_field(
                field, chunks, retriever, anchors or [],
                top_k=top_k,
                include_first_pages=include_first_pages,
                max_excerpts_chars=max_excerpts_chars,
                max_chunk_chars=max_chunk_chars,
            )
            # Fall back to global excerpts if field-specific retrieval returned nothing
            if not field_excerpts.strip():
                field_excerpts = excerpts
        elif field in _FIRST_PAGE_FIELDS and first_page_text:
            # v4: metadata fields get first-page text directly
            field_excerpts = safe_cut_chars(first_page_text, max_excerpts_chars)
        else:
            field_excerpts = excerpts
        field_query = FIELD_QUERIES.get(field, "")
        contextual_neighbor_block = build_referenced_neighbor_context(
            field_excerpts,
            field,
            field_query,
            reference_neighbors or [],
            citation_map=citation_map,
            max_neighbors=3,
        )
        if contextual_neighbor_block:
            # augment only when reference is present in chosen excerpts
            field_excerpts = safe_cut_chars(
                field_excerpts + "\n\n" + contextual_neighbor_block,
                max_excerpts_chars,
            )

        ev = numeric_evidence if field in _NUMERIC_FIELDS else ""

        raw = ollama_with_retry(
            lambda ex, _f=field, _ctx=context_so_far, _ev=ev: build_single_section_prompt(
                _f, ex, _ctx, numeric_evidence=_ev
            ),
            model=model,
            base_url=base_url,
            excerpts=field_excerpts,
            num_predict=900,
        )

        parsed_field = parse_all_fields(raw).get(field, "").strip()

        # Fallback when model returns only body text without heading.
        if not parsed_field:
            cleaned = strip_markdown(raw).strip()
            cleaned = re.sub(rf"^{re.escape(field)}\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            parsed_field = cleaned

        parsed[field] = parsed_field or "Not found in provided text."
        context_lines.append(f"{field}:\n{parsed[field]}")
        print(f"      Generated section: {field}")

    return parsed


# ---------------------------------------------------------------------------
# Robust field parser  (v3: tighter prefix match)
# ---------------------------------------------------------------------------
def _normalise_heading(line: str) -> Optional[str]:
    """
    Match a line to a FIELD regardless of markdown decoration or capitalisation.

    Prefix match now requires the candidate to cover ≥60% of field length
    and at least 4 characters to reduce accidental matches while allowing
    customizable section names.
    """
    # Strip leading decoration: whitespace, #, *, _, -, bullets, numbered list markers
    clean = re.sub(r"^[\s#*_•\-]*(?:\d+[.):]?\s*|\(\d+\)\s*)?", "", line)
    clean = re.sub(r"[\s*_]+$", "", clean)

    if ":" not in clean:
        return None

    heading_part = clean.split(":", 1)[0].strip().upper()
    # Normalise common LLM substitutions
    heading_part = heading_part.replace(" AND ", " & ")
    heading_part = re.sub(r"\s+", " ", heading_part)

    for field in FIELDS:
        if heading_part == field:
            return field
        # Stricter prefix — must cover ≥60% of field length and ≥4 chars
        if (
            field.startswith(heading_part)
            and len(heading_part) >= 4
            and len(heading_part) >= int(len(field) * 0.6)
        ):
            return field

    return None


def parse_all_fields(raw: str) -> Dict[str, str]:
    """
    Robust multi-field extractor:
    - Strips markdown before processing.
    - Handles inline content on the same line as the heading.
    - Accumulates continuation lines until the next recognised heading.
    """
    text = strip_markdown(raw)
    lines = text.splitlines()

    result: Dict[str, str] = {f: "" for f in FIELDS}
    current_field: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if current_field is not None:
            result[current_field] = "\n".join(buf).strip()

    for line in lines:
        field = _normalise_heading(line)
        if field is not None:
            flush()
            current_field = field
            buf = []
            after_colon = line.split(":", 1)[1].strip() if ":" in line else ""
            after_colon = re.sub(r"^[\s*_#]+", "", after_colon).strip()
            if after_colon:
                buf.append(after_colon)
        else:
            if current_field is not None:
                buf.append(line)

    flush()
    return result


# ---------------------------------------------------------------------------
# Validation helpers  (v3: stricter thresholds)
# ---------------------------------------------------------------------------
def count_filled_fields(parsed: Dict[str, str]) -> int:
    return sum(1 for v in parsed.values() if v.strip())


def looks_like_template(raw: str, min_fields: int = 6) -> bool:
    """
    FIX v3: requires min_fields (default 6/8) AND TITLE must be present.
    Old code only required 3/8.
    """
    parsed = parse_all_fields(raw)
    has_title = bool(parsed.get("TITLE", "").strip())
    filled = count_filled_fields(parsed)
    return filled >= min_fields and has_title


def is_field_complete(field_name: str, content: str) -> bool:
    content = content.strip()
    if not content or content == "Not found in provided text.":
        return True

    if len(content) < 20:
        return False
    if content[-1] in ",:;-":
        return False

    if field_name == "BUSINESS SUGGESTION":
        if content[-1] not in ".!?":
            return False
        if word_count(content) < 10:
            return False

    return True


def get_missing_fields(parsed: Dict[str, str]) -> List[str]:
    """Return fields that are empty or not found."""
    return [
        f for f in FIELDS
        if not parsed.get(f, "").strip()
        or parsed[f] == "Not found in provided text."
    ]


def intro_has_substructure(text: str) -> bool:
    """Check TECHNOLOGICAL INTRODUCTION still has innovations + table after tightening."""
    has_innovations = bool(re.search(r"(main innovation|innovation\s*[1-3]|^\s*[-•])", text,
                                     re.IGNORECASE | re.MULTILINE))
    has_table = "|" in text
    return has_innovations and has_table


# ---------------------------------------------------------------------------
# Tightening
# ---------------------------------------------------------------------------
def tighten_prompt(section_name: str, max_words: int,
                   text: str, excerpts: str) -> str:
    must_keep = ""
    if section_name.upper() == "TECHNOLOGICAL INTRODUCTION":
        must_keep = (
            "Preserve this internal structure exactly:\n"
            "  Methodology & achievement: 1-3 sentences.\n"
            "  3 Main Innovations:\n"
            "  - ...\n  - ...\n  - ...\n"
            "  Mini Table (Method | Key features): markdown table with 3-6 rows\n\n"
        )

    quant_reminder = ""
    if section_name.upper() in ["TECHNICAL ADVANTAGES", "BACKGROUND"]:
        quant_reminder = (
            "CRITICAL: Keep all specific numbers, percentages, ranges, and comparisons.\n"
            "Remove vague adjectives first, never quantitative data.\n\n"
        )

    return (
        f"Shorten ONLY the '{section_name}' section to under {max_words} words.\n"
        "Do NOT change topic or add new claims. Keep bullet structure.\n\n"
        + must_keep
        + quant_reminder
        + f"ORIGINAL TEXT:\n{text}\n\n"
        f"EXCERPTS (for reference):\n{excerpts}\n\n"
        "Return ONLY the shortened text — no heading, no preamble.\n"
    )


def assemble_text(parsed: Dict[str, str]) -> str:
    lines = []
    for f in FIELDS:
        lines.append(f"{f}:")
        lines.append(parsed.get(f, "Not found in provided text.").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a scientific PDF to a Technological Insight text report."
    )
    ap.add_argument("--pdf",               required=True,  help="Path to input PDF")
    ap.add_argument("--out",               default="insight.txt", help="Output text file")
    ap.add_argument("--model",             default="llama3.1:8b", help="Ollama model name")
    ap.add_argument("--ollama_url",        default="http://127.0.0.1:11434")
    ap.add_argument("--no_semantic",       action="store_true")
    ap.add_argument("--embed_model",       default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_chars",         type=int, default=3500)
    ap.add_argument("--overlap_chars",     type=int, default=400)
    ap.add_argument("--top_k",             type=int, default=15)
    ap.add_argument("--include_first_pages", type=int, default=3)
    ap.add_argument("--max_excerpts_chars",  type=int, default=12000)
    ap.add_argument("--max_chunk_chars",     type=int, default=900)
    ap.add_argument("--min_fields",          type=int, default=6,
                    help="Min filled fields in Pass 1 to skip reformat (default: 6)")
    ap.add_argument(
        "--insight_config",
        default="insight_config.yaml",
        help="YAML file defining sections with retrieval queries and descriptions.",
    )
    ap.add_argument("--reference_graph", default="",
                    help="Optional path to a stored reference graph JSON.")
    ap.add_argument("--paper_id", default="",
                    help="Paper id in the reference graph (auto-inferred from first page title if omitted).")
    ap.add_argument("--max_neighbor_papers", type=int, default=8,
                    help="Maximum number of graph neighbors to inject as context.")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"PDF not found: {args.pdf}")
    if os.path.exists(args.insight_config):
        try:
            load_insight_schema(args.insight_config)
            print(f"      Loaded insight config: {args.insight_config}")
        except Exception as e:
            sys.exit(f"Failed to load --insight_config '{args.insight_config}': {e}")
    else:
        print(
            f"      Insight config not found at {args.insight_config}; "
            "using built-in defaults."
        )

    # 1. Extract
    print("[1/5] Extracting PDF text ...")
    pages = extract_pages(args.pdf)
    if not pages:
        sys.exit("No text extracted. PDF may be image-only (needs OCR).")
    print(f"      {len(pages)} pages extracted.")

    # 2. Chunk
    print("[2/5] Chunking ...")
    chunks = make_chunks(pages, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    print(f"      {len(chunks)} chunks from {len(pages)} pages.")

    # 3. Dynamic keywords
    print("[3/5] Extracting domain keywords from paper ...")
    keywords = extract_paper_keywords(pages, n_keywords=30)
    query    = build_dynamic_query(keywords)
    anchors  = keywords[:15]
    print(f"      Keywords : {', '.join(keywords[:10])} ...")
    print(f"      Anchors  : {', '.join(anchors[:8])} ...")

    # 4. Retrieve
    print("[4/5] Building retriever ...")
    retriever = Retriever(chunks, semantic=(not args.no_semantic),
                          embed_model=args.embed_model)
    print(f"      Semantic : {'ENABLED' if retriever.semantic_enabled else 'DISABLED (keyword fallback)'}")

    chosen   = select_chunks(chunks, retriever, query,
                              top_k=args.top_k,
                              include_first_pages=args.include_first_pages,
                              anchors=anchors)
    excerpts = build_excerpts_block(chosen,
                                    max_total_chars=args.max_excerpts_chars,
                                    max_chunk_chars=args.max_chunk_chars)
    reference_neighbors = build_reference_neighbor_context(
        args.reference_graph,
        pages,
        explicit_paper_id=args.paper_id,
        max_neighbors=args.max_neighbor_papers,
    )
    citation_map = build_bibliography_citation_map(pages, reference_neighbors)
    if reference_neighbors:
        print("      Loaded reference-graph neighbors for conditional context injection.")
    if citation_map:
        print(f"      Built bibliography citation map for {len(citation_map)} [n] references.")

    with open("debug_excerpts.txt", "w", encoding="utf-8") as f:
        f.write(excerpts)
    print(f"      {len(chosen)} chunks selected ({len(excerpts)} chars). "
          "Saved -> debug_excerpts.txt")

    if not excerpts.strip():
        sys.exit(
            "Excerpts are empty. "
            "Try: --no_semantic --top_k 20 --include_first_pages 3 --max_excerpts_chars 15000"
        )

    # 5. Generate each section separately
    print("[5/5] Generating insight sections one-by-one ...")
    # v4: pass retriever, chunks, pages so each field gets field-specific excerpts
    parsed = generate_sections_separately(
        excerpts, args.model, args.ollama_url,
        chunks=chunks,
        retriever=retriever,
        anchors=anchors,
        pages=pages,
        top_k=args.top_k,
        include_first_pages=args.include_first_pages,
        max_excerpts_chars=args.max_excerpts_chars,
        max_chunk_chars=args.max_chunk_chars,
        reference_neighbors=reference_neighbors,
        citation_map=citation_map,
    )

    with open("debug_raw_final.txt", "w", encoding="utf-8") as f:
        f.write(assemble_text(parsed))

    # FIX v3: continuation pass for trailing missing fields
    missing = [f for f in FIELDS if parsed[f] == "Not found in provided text."]
    if missing:
        print(f"      Missing fields after parse: {missing} — attempting continuation pass ...")
        partial_text = assemble_text(parsed)
        continuation = ollama_with_retry(
            lambda ex: build_continuation_prompt(ex, partial_text, missing),
            args.model, args.ollama_url, excerpts, num_predict=800,
        )
        with open("debug_raw_continuation.txt", "w", encoding="utf-8") as f:
            f.write(continuation)
        extra = parse_all_fields(continuation)
        for f in missing:
            if extra.get(f, "").strip():
                parsed[f] = extra[f]
                print(f"        Recovered: {f}")
        # Re-check still-missing
        for f in FIELDS:
            if not parsed.get(f, "").strip():
                parsed[f] = "Not found in provided text."

    # Completeness check
    incomplete = []
    for f in FIELDS:
        if parsed[f] != "Not found in provided text." and not is_field_complete(f, parsed[f]):
            incomplete.append(f)
    if incomplete:
        print(f"      WARNING: Fields appear incomplete/cut-off: {incomplete}")

    # Report
    filled = [f for f in FIELDS if parsed[f] != "Not found in provided text."]
    empty  = [f for f in FIELDS if parsed[f] == "Not found in provided text."]
    print(f"      Fields filled  : {filled}")
    if empty:
        print(f"      Fields missing : {empty}")

    if not filled:
        print("\n=== RAW MODEL OUTPUT (first 2000 chars) ===")
        print(assemble_text(parsed)[:2000])
        print("===========================================\n")
        sys.exit(
            "ERROR: All fields are empty after parsing.\n"
            "Check debug_raw_final.txt to see what the model returned.\n"
            "Common causes:\n"
            "  - Model ignored the format -> try a larger model.\n"
            "  - Context too large -> try --max_excerpts_chars 8000.\n"
            "  - Ollama not running -> check your server.\n"
        )

    # Tighten over-length sections
    limits = {
        "BACKGROUND": 200,
        "TECHNOLOGICAL INTRODUCTION": 350,
        "TECHNICAL ADVANTAGES": 70,
        "CURRENT STATUS & FUTURE CHALLENGES": 40,
    }
    for sec, maxw in limits.items():
        txt = parsed.get(sec, "")
        if txt == "Not found in provided text.":
            continue
        wc = word_count(txt)
        if wc >= maxw:
            print(f"      Tightening '{sec}' ({wc} -> <{maxw} words) ...")
            tightened = ollama_with_retry(
                lambda _ex: tighten_prompt(sec, maxw - 1, txt, excerpts),
                args.model, args.ollama_url, excerpts, num_predict=600,
            )
            if tightened.strip():
                # FIX v3: for TECHNOLOGICAL INTRODUCTION, revert if substructure lost
                if sec == "TECHNOLOGICAL INTRODUCTION" and not intro_has_substructure(tightened):
                    print(f"        Tightening broke substructure — keeping original.")
                else:
                    parsed[sec] = tightened.strip()

    # Write output
    out_text = assemble_text(parsed)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"\nDone. Output -> {args.out}")

    # v4: save a .json version of the parsed sections alongside the .txt
    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"      JSON  -> {json_out}")

    if incomplete:
        print(f"\n⚠ WARNING: Some fields may be incomplete. Review output manually.")
        print(f"   Consider a larger model or increasing --max_excerpts_chars")


if __name__ == "__main__":
    main()
