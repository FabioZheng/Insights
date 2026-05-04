# paper-to-insight

Convert scientific PDF papers into structured **Technological Insight** reports using a local LLM via [Ollama](https://ollama.com). No cloud APIs required — runs fully offline.

---

## Overview

`main.py` extracts key content from a research PDF and generates a structured 8-section report covering the paper's background, technology, innovations, research group, business potential, and more.

`insight_to_md.py` takes the plain-text `.txt` output and converts it into a clean, readable Markdown document with formatting, tables, and section styling.

---

## Features

- **Fully offline** — uses Ollama for local LLM inference
- **Semantic retrieval** — optional FAISS + sentence-transformers for smarter chunk selection
- **Strict template enforcement** — 8 structured sections with word limits
- **Quantitative extraction** — actively prompts for specific numbers, percentages, and ranges
- **Retry logic** — multiple fallback passes if sections are missing or incomplete
- **Clean Markdown output** — `insight_to_md.py` renders the report beautifully

---

## Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install pdfplumber requests numpy
# for external insight config loading
pip install pyyaml
# for graph visualization scripts
pip install networkx matplotlib
# for semantic-store builder
pip install requests
```

For semantic retrieval (recommended):

```bash
pip install sentence-transformers faiss-cpu
```

[Install Ollama](https://ollama.com/download) and pull one of the supported models (see [Supported Models](#supported-models) below).

---

## Usage

### Step 1 — Generate the insight report (plain text)

```bash
python main.py --pdf paper.pdf --model llama3.1:8b --out insight.txt
```

With citation-neighbor context from a prebuilt graph:

```bash
python main.py --pdf paper.pdf --model llama3.1:8b --out insight.txt \
  --reference_graph reference_graph.json --paper_id <paper_id>
```

### Step 2 — Convert to readable Markdown

```bash
python insight_to_md.py --input insight.txt --out insight.md
```

---

## All CLI Options

### `main.py`

| Flag | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to the input PDF |
| `--out` | `insight.txt` | Output plain text file |
| `--model` | `llama3.1:8b` | Ollama model name |
| `--ollama_url` | `http://127.0.0.1:11434` | Ollama server URL |
| `--no_semantic` | `False` | Disable semantic retrieval (use keyword fallback) |
| `--embed_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for semantic search |
| `--top_k` | `15` | Number of chunks to retrieve |
| `--include_first_pages` | `3` | Always include chunks from first N pages |
| `--max_excerpts_chars` | `12000` | Max characters sent to model as excerpts |
| `--max_chunk_chars` | `900` | Max characters per individual chunk |
| `--min_fields` | `6` | Minimum filled fields to accept Pass 1 without reformatting |
| `--insight_config` | `insight_config.yaml` | YAML file with customizable sections, retrieval queries, and section descriptions |
| `--reference_graph` | *(empty)* | Optional weighted reference-graph JSON to inject neighbor context |
| `--paper_id` | *(auto)* | Graph paper id for the current PDF; inferred from first-page title if omitted |
| `--max_neighbor_papers` | `8` | Number of neighboring papers to include in generation context |

### Insight config YAML (`insight_config.yaml`)

You can customize section names, retrieval queries, and generation instructions without editing `main.py`.

```yaml
sections:
  - name: TITLE
    query: "paper title name of the work"
    description: "Write a concise, specific paper title on one line."
```

Each `sections` entry supports:
- `name`: heading used in final output
- `query`: retrieval query used to pull field-specific excerpts
- `description`: section-specific writing instruction used in the generation prompt

### `insight_to_md.py`

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the `.txt` insight file |
| `--out` | `insight.md` | Output Markdown file |
| `--title` | *(auto-detected)* | Override the document title |

---

## Output Format

The report covers 8 structured sections:

1. **TITLE** — Concise paper title
2. **SUBTITLE** — One-sentence value proposition
3. **BACKGROUND** — Field context and current limitations (≤200 words)
4. **TECHNOLOGICAL INTRODUCTION** — Methodology, 3 main innovations, and a method/feature table (≤350 words)
5. **RESEARCH GROUP** — Lead researcher, affiliation, and research area
6. **TECHNICAL ADVANTAGES** — Bullet points with specific metrics and comparisons (≤70 words)
7. **CURRENT STATUS & FUTURE CHALLENGES** — Brief status summary (≤40 words)
8. **BUSINESS SUGGESTION** — Domain-specific commercial opportunity (1–2 sentences)

---

## Supported Models

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | 4.9 GB | Fast, suitable for most use cases |
| `deepseek-r1:32b` | 19 GB | Higher quality output, requires more RAM |

Pull a model before running:

```bash
ollama pull llama3.1:8b
# or
ollama pull deepseek-r1:32b
```

### Which model should be used for semantic summarization?

If you are using:
- `--model "deepseek-r1:32b"`
- `--embed_model "...all-MiniLM-L6-v2..."`

Then for `scripts/build_semantic_store.py`, use **`deepseek-r1:32b`** as the summarization model.

Reason:
- `build_semantic_store.py` calls Ollama generation (`/api/generate`) and needs a **text-generation model**.
- `all-MiniLM-L6-v2` is an **embedding model** used for retrieval/vector similarity, not text generation.

Example:

```bash
python scripts/build_semantic_store.py \
  --graph reference_graph.json \
  --model deepseek-r1:32b \
  --ollama_url http://127.0.0.1:11434
```

---

## Debug Files

During a run, two debug files are written to the current directory:

- `debug_excerpts.txt` — The exact text chunks sent to the model
- `debug_raw_final.txt` — The assembled raw output before any tightening


These are useful for diagnosing template compliance issues.

---

## Troubleshooting

**All fields are empty after parsing**
- Check `debug_raw_final.txt` to see what the model returned
- Try a larger model: `--model deepseek-r1:32b`
- Reduce context: `--max_excerpts_chars 8000`

**Semantic retrieval fails to load**
- Install dependencies: `pip install sentence-transformers faiss-cpu`
- Or bypass with `--no_semantic`

**PDF has no extracted text**
- The PDF is likely image-only and requires OCR preprocessing
- Try tools like `ocrmypdf` before passing to this script

**Ollama connection refused**
- Ensure Ollama is running: `ollama serve`
- Check the URL with `--ollama_url http://127.0.0.1:11434`

---

## Project Structure

```
.
├── main.py                    # Main extraction and generation script
├── insight_to_md.py           # Converts .txt output to clean Markdown
├── insight_config.yaml         # External section/query/description config
├── scripts/build_reference_graph.py    # Build + store weighted citation graph JSON
├── scripts/visualize_reference_graph.py # Render graph image from stored JSON
├── scripts/build_semantic_store.py     # Add semantic summaries per paper node/section
├── scripts/inspect_graph_store.py      # Inspect graph edges + semantic summary fields
├── README.md
├── debug_excerpts.txt         # (generated) chunks sent to model
└── debug_raw_final.txt        # (generated) raw model output
```

---

## Reference graph workflow

1. Build and store a weighted reference graph JSON from metadata:

```bash
python scripts/build_reference_graph.py --input papers.json --out reference_graph.json
```

2. Visualize the graph:

```bash
python scripts/visualize_reference_graph.py --graph reference_graph.json --out reference_graph.png
```

3. Build semantic summaries for each paper node (global only):

```bash
python scripts/build_semantic_store.py --graph reference_graph.json --model llama3.1:8b
```

This updates each graph node with:
- `semantic_store.global_summary`
- `semantic_store.main_findings`
- `semantic_store.main_claims`
- `semantic_store.evidence_summary` (how findings/claims are supported)

4. Use the graph neighbor information during insight generation:

```bash
python main.py --pdf paper.pdf --out insight.txt --reference_graph reference_graph.json
```

5. Inspect graph and summary content to verify correctness:

```bash
python scripts/inspect_graph_store.py --graph reference_graph.json --max_nodes 10
python scripts/inspect_graph_store.py --graph reference_graph.json --paper_id <paper_id>
```

This check shows:
- number of nodes/edges,
- how many nodes have semantic summaries,
- sample summary fields (`global_summary`, `main_findings`, `main_claims`, `evidence_summary`),
- connected edges for a target paper, including `weight` and `weight_components`.

### How the reference graph improves answer generation

The graph helps by injecting related-paper context directly into the model input before section generation.

1. **Weighted neighbors are selected**  
   For the current paper, `main.py` finds connected neighbor papers from `reference_graph.json` and ranks them by edge weight. Higher-weight neighbors are treated as more relevant context.  
2. **Neighbor metadata is summarized into context**  
   The generator builds a compact context block containing neighbor title, keywords, abstract snippet, and weight.  
3. **Context is appended conditionally**  
   Graph neighbor context is appended only when the currently selected excerpt already mentions a referenced neighbor (by id/title).  
4. **Only relevant neighbor info is used to augment retrieval/generation**  
   For each section query, only neighbor fields relevant to that query (keyword/title/abstract overlap) are used, reducing noisy augmentation.

Implementation notes:
- Neighbor extraction and formatting are implemented in `build_reference_neighbor_context(...)`.  
- The context is injected before generation in `main()`, and passed through per-section generation.  

---

## FAQ

### When is `DEFAULT_INSIGHT_SCHEMA` used instead of YAML?

`main.py` always initializes with `DEFAULT_INSIGHT_SCHEMA` first. Then:

- If `--insight_config` exists on disk (default: `insight_config.yaml`), it is loaded and overrides the defaults at runtime.
- If that file does not exist, the script keeps using `DEFAULT_INSIGHT_SCHEMA`.
- If the file exists but is invalid, the script exits with an error instead of silently falling back.

### Are generator queries dependent on the YAML schema?

Yes. In the schema, each section has:
- `name` (section heading),
- `query` (field-specific retrieval query),
- `description` (field-specific generation instruction).

These values are loaded into runtime mappings:
- `FIELDS` from `name`,
- `FIELD_QUERIES` from `query`,
- `FIELD_DESCRIPTIONS` from `description`.

During generation:
- field-specific retrieval uses `FIELD_QUERIES[field]` (via `select_chunks_for_field`) to choose excerpts for that section,
- section prompt instructions use `FIELD_DESCRIPTIONS[field]` (via `build_single_section_prompt`).

So changing YAML directly changes both retrieval behavior and prompt behavior per section.

### Is the semantic store used when retrieving neighbors from the graph?

Yes, but in two stages:

1. **Neighbor retrieval (graph stage):** neighbors are selected using graph connectivity/edge weight.
2. **Neighbor ranking + context stage:** once candidate neighbors are selected, semantic store fields
   (`global_summary`, `main_findings`, `main_claims`, `evidence_summary`) are used to compute
   semantic overlap with the field query and to build the appended context block.

So graph structure chooses *who* is a neighbor; semantic store influences *which neighbor context*
is most relevant and what text is appended to generation input.

### How were queries shaped before vs after YAML creation?

**Before YAML (`insight_config.yaml`)**
- Section retrieval queries were hard-coded in `main.py` inside a static `FIELD_QUERIES` dictionary.
- The generator always used that built-in mapping, so changing query wording required editing Python code.
- Section writing instructions were also hard-coded in `build_single_section_prompt()`.

**After YAML**
- Query shaping is externalized per section in YAML under `sections[].query`.
- At startup, YAML values are loaded into `FIELD_QUERIES` and used by `select_chunks_for_field()` for field-specific retrieval.
- Section instructions are also externalized via `sections[].description` and loaded into `FIELD_DESCRIPTIONS`.
- Net effect: query scope and phrasing can be tuned per section without code changes, and the same retrieval/generation logic consumes the new query text dynamically.

### What component played the role of field descriptors before YAML?

Before YAML, field descriptors were hard-coded directly in `main.py` inside
`build_single_section_prompt()` as an internal `field_rules` mapping
(e.g., per-section constraints for TITLE, BACKGROUND, TECHNICAL ADVANTAGES, etc.).

After YAML, that role moved to `sections[].description` in `insight_config.yaml`,
which is loaded into `FIELD_DESCRIPTIONS` and used in
`build_single_section_prompt()` at runtime.

### How does the graph improve insight generation end-to-end?

The graph contribution is a **second-stage contextual signal** that augments
section generation after the normal excerpt retrieval.

#### 1) Neighbor candidates are loaded from the citation graph

- `build_reference_neighbor_context(...)` loads the graph JSON (`nodes`, `edges`),
  infers the current paper id, and gathers neighbor papers connected to it.
- For each neighbor, edge weights are aggregated so stronger connected papers rank higher.
- The function returns structured neighbor objects (`id`, `weight`, `title`, `keywords`, `abstract`)
  instead of raw text.

#### 2) Neighbor info is only used when the chosen excerpt mentions the reference

- During field generation, `build_referenced_neighbor_context(...)` checks whether
  the currently selected excerpt text mentions a neighbor by id/title.
- If a neighbor is not mentioned in the excerpt, it is ignored for that field.
- For papers that cite by bracket index (e.g., `[1]`, `[12]`), bibliography lines are
  parsed to build a `[n] -> paper_id` map, and `[n]` mentions are treated as direct references.
- If `semantic_store` exists on neighbor nodes, generation uses `global_summary`
  plus findings/claims/evidence fields when available (fallback: abstract).

#### Where summaries are stored and how documents are divided for many summaries

- Storage location: in the same graph JSON, under each node:
  - `node.semantic_store.global_summary`
  - `node.semantic_store.main_findings`
  - `node.semantic_store.main_claims`
  - `node.semantic_store.evidence_summary`
- The semantic store is produced by `scripts/build_semantic_store.py` via LLM prompts over title/keywords/abstract.
- During neighbor augmentation, claim/finding/evidence content is appended to generator input for referenced neighbors.

#### 3) Relevance weighting combines global graph relevance + local chunk relevance

- `graph_weight` is global paper-paper relevance from structural signals (citation/co-citation/shared metadata).
- `semantic_score` is local chunk-neighbor relevance (current chunk text vs neighbor semantic fields).
- `final_neighbor_score = 0.7 * semantic_score + 0.3 * graph_weight`.
- A neighbor relevance score is computed from:
  - graph strength (`neighbor.weight`, weighted 2x), and
  - weighted textual overlap between query terms and semantic fields:
    - findings overlap (1.2x),
    - claims overlap (1.1x),
    - evidence overlap (1.0x),
    - global summary overlap (0.9x),
    - title/keywords overlap (0.8x).
- Only neighbors with positive overlap are retained.
- Top-ranked neighbors are converted to a compact context block.

#### 3.1) How the "right chunk" is extracted from each neighbor

- Neighbor abstracts are split into sentences.
- Sentences are scored by overlap with the current field query terms.
- Only top-scoring sentences (up to 2) are kept as the `relevant_snippet`.
- If no sentence overlaps, a short fallback abstract snippet is used.
- This gives per-neighbor, per-field focused chunks instead of copying whole abstracts.

#### 4) The selected neighbor context is appended to that field's excerpt input

- The compact block is appended to `field_excerpts` only when non-empty.
- Then `build_single_section_prompt(...)` receives that augmented excerpt context for
  the LLM call (`ollama_with_retry(...)`).
- This improves grounding because the model sees nearby cited work that is both
  graph-connected and query-relevant to the specific section being generated.
