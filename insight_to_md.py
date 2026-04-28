#!/usr/bin/env python3
"""
insight_to_md.py — Convert a paper_to_insight_text.py .txt output into a clean, readable Markdown file.

Usage:
    python insight_to_md.py --input insight.txt --out insight.md

Optional flags:
    --title     Override the document title (default: auto-detected from TITLE field)
"""

import argparse
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Field definitions (must match paper_to_insight_text.py order)
# ---------------------------------------------------------------------------
FIELDS = [
    "TITLE",
    "SUBTITLE",
    "BACKGROUND",
    "TECHNOLOGICAL INTRODUCTION",
    "RESEARCH GROUP",
    "TECHNICAL ADVANTAGES",
    "CURRENT STATUS & FUTURE CHALLENGES",
    "BUSINESS SUGGESTION",
]

# Section display names and emoji icons for Markdown output
SECTION_META = {
    "TITLE":                              ("", ""),
    "SUBTITLE":                           ("", ""),
    "BACKGROUND":                         ("", "Background"),
    "TECHNOLOGICAL INTRODUCTION":         ("", "Technological Introduction"),
    "RESEARCH GROUP":                     ("", "Research Group"),
    "TECHNICAL ADVANTAGES":              ("", "Technical Advantages"),
    "CURRENT STATUS & FUTURE CHALLENGES": ("", "Current Status & Future Challenges"),
    "BUSINESS SUGGESTION":               ("", "Business Suggestion"),
}

PLACEHOLDER = "Not found in provided text."


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_txt(path: str) -> Dict[str, str]:
    """
    Parse the structured .txt file produced by paper_to_insight_text.py.
    Each field starts with its ALL-CAPS name followed by a colon on its own line.
    Content continues until the next recognised field heading.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    result: Dict[str, str] = {field: "" for field in FIELDS}

    # Build a regex that splits on any known field heading
    field_pattern = "|".join(re.escape(f) for f in FIELDS)
    # Match heading line: optional whitespace + FIELD_NAME + colon
    splitter = re.compile(
        rf"^[ \t]*({field_pattern})[ \t]*:[ \t]*\n?",
        re.MULTILINE,
    )

    parts = splitter.split(raw)
    # parts = [pre_text, field1, content1, field2, content2, ...]
    i = 1
    while i < len(parts) - 1:
        field_name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if field_name in result:
            result[field_name] = content
        i += 2

    return result


# ---------------------------------------------------------------------------
# Content renderers
# ---------------------------------------------------------------------------
def render_table(text: str) -> str:
    """
    Ensure markdown tables have proper GFM separator rows.
    Adds a separator row after the header if one is missing.
    """
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_table_row = line.strip().startswith("|") and line.strip().endswith("|")
        if is_table_row:
            out.append(line)
            # Check if next line is already a separator
            if (i + 1 < len(lines) and
                    re.match(r"^\s*\|[\s\-|:]+\|\s*$", lines[i + 1])):
                i += 1
                out.append(lines[i])
            else:
                # Insert separator based on column count
                col_count = line.count("|") - 1
                sep = "| " + " | ".join(["---"] * col_count) + " |"
                out.append(sep)
        else:
            out.append(line)
        i += 1
    return "\n".join(out)


def render_bullets(text: str) -> str:
    """
    Normalise bullet characters (•, ★, -, *) to consistent Markdown dashes.
    Also ensure each bullet starts on its own line.
    """
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        # Normalise common bullet characters
        stripped = re.sub(r"^[•▸▶➤✓✗]\s*", "- ", stripped)
        # Normalise asterisk bullets (but not bold markers)
        stripped = re.sub(r"^\*\s+", "- ", stripped)
        out.append(stripped)
    return "\n".join(out)


def render_research_group(text: str) -> str:
    """
    Format RESEARCH GROUP as a small info block.
    Handles both comma-separated format and free-form text.
    """
    if not text or text == PLACEHOLDER:
        return f"*{PLACEHOLDER}*"

    # Try comma-separated: Name, Affiliation, Research Area
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2:
        lines = []
        labels = ["**Lead Researcher:**", "**Affiliation:**", "**Research Area:**"]
        for label, part in zip(labels, parts):
            if part:
                lines.append(f"{label} {part}")
        if lines:
            return "\n\n".join(lines)

    return text


def render_tech_intro(text: str) -> str:
    """
    Ensure the TECHNOLOGICAL INTRODUCTION sub-sections render cleanly.
    Normalises 'Main Innovations' heading and the mini table.
    """
    if not text or text == PLACEHOLDER:
        return f"*{PLACEHOLDER}*"

    # Normalise "3 Main Innovations:" sub-heading -> bold
    text = re.sub(
        r"(?i)(3\s+main\s+innovations?)\s*:",
        r"\n**\1:**",
        text,
    )

    # Normalise "Methodology & achievement:" sub-heading -> bold
    text = re.sub(
        r"(?i)(methodology\s+[&and]+\s+achievement)\s*:",
        r"\n**\1:**",
        text,
    )

    # Ensure table is well-formed
    text = render_table(text)

    # Normalise bullets
    text = render_bullets(text)

    return text.strip()


def render_section_body(field: str, text: str) -> str:
    """Route each field to its appropriate renderer."""
    if not text or text == PLACEHOLDER:
        return f"*{PLACEHOLDER}*\n"

    if field == "RESEARCH GROUP":
        return render_research_group(text) + "\n"

    if field == "TECHNOLOGICAL INTRODUCTION":
        return render_tech_intro(text) + "\n"

    if field == "TECHNICAL ADVANTAGES":
        return render_bullets(text) + "\n"

    # Default: ensure tables are well-formed
    return render_table(text) + "\n"


# ---------------------------------------------------------------------------
# Markdown assembler
# ---------------------------------------------------------------------------
def build_markdown(
    parsed: Dict[str, str],
    title_override: Optional[str] = None,
    source_file: Optional[str] = None,
) -> str:
    title = title_override or parsed.get("TITLE", "").strip() or "Technological Insight Report"
    subtitle = parsed.get("SUBTITLE", "").strip()

    lines: List[str] = []

    # ── Document header ──────────────────────────────────────────────────
    lines.append(f"# {title}")
    lines.append("")

    if subtitle and subtitle != PLACEHOLDER:
        lines.append(f"> {subtitle}")
        lines.append("")

    # Metadata bar
    date_str = datetime.now().strftime("%B %d, %Y")
    meta_parts = [f"*Generated: {date_str}*"]
    if source_file:
        meta_parts.append(f"*Source: `{os.path.basename(source_file)}`*")
    lines.append(" &nbsp;·&nbsp; ".join(meta_parts))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Table of Contents ────────────────────────────────────────────────
    lines.append("## Table of Contents")
    lines.append("")
    toc_fields = [f for f in FIELDS if f not in ("TITLE", "SUBTITLE")]
    for field in toc_fields:
        emoji, display = SECTION_META[field]
        anchor = display.lower().replace(" ", "-").replace("&", "").replace("--", "-")
        anchor = re.sub(r"-+", "-", anchor).strip("-")
        lines.append(f"- [{display}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Sections ─────────────────────────────────────────────────────────
    for field in FIELDS:
        if field in ("TITLE", "SUBTITLE"):
            continue

        emoji, display = SECTION_META[field]
        content = parsed.get(field, "").strip()

        lines.append(f"## {display}")
        lines.append("")
        lines.append(render_section_body(field, content))
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append(
        "<sub>Generated by "
        "[paper-to-insight](https://github.com/your-org/paper-to-insight) · "
        f"Report date: {date_str}</sub>"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation / warnings
# ---------------------------------------------------------------------------
def warn_missing(parsed: Dict[str, str]) -> None:
    missing = [f for f in FIELDS if not parsed.get(f, "").strip() or
               parsed[f].strip() == PLACEHOLDER]
    if missing:
        print(f"  WARNING: Sections with no content: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a paper_to_insight_text.py .txt output to clean Markdown."
    )
    ap.add_argument("--input",  required=True, help="Path to the .txt insight file")
    ap.add_argument("--out",    default="insight.md", help="Output Markdown file")
    ap.add_argument("--title",  default=None, help="Override document title")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}")

    print(f"[1/3] Parsing {args.input} ...")
    parsed = parse_txt(args.input)

    filled = sum(
        1 for f in FIELDS
        if parsed.get(f, "").strip() and parsed[f].strip() != PLACEHOLDER
    )
    print(f"      {filled}/{len(FIELDS)} sections found.")
    warn_missing(parsed)

    print("[2/3] Rendering Markdown ...")
    md = build_markdown(parsed, title_override=args.title, source_file=args.input)

    print(f"[3/3] Writing {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nDone -> {args.out}  ({len(md):,} chars)")


if __name__ == "__main__":
    main()
