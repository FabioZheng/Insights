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

3. Use the graph neighbor information during insight generation:

```bash
python main.py --pdf paper.pdf --out insight.txt --reference_graph reference_graph.json
```
