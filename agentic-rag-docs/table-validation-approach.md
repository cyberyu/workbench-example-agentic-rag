# Table Validation Approach

## Overview

Every table extracted from a source document (DOCX) is validated against two independent signals: a **content cross-check** from the rendered PDF, and a **style snapshot** captured by a vision LLM reading the original page image.

Together these form a complete ground-truth snapshot that can be compared against any re-transformed version of the table (re-exported PDF, re-rendered HTML, iXBRL output, etc.).

---

## The Snapshot Model

The canonical representation of a table is a **CALS XML** document stored in `data/table_catalog.json` under the `"xml"` key. Every `<entry>` element carries the following attributes:

| Attribute | Values | Source | Meaning |
|---|---|---|---|
| text content | string | python-docx | The cell's text or number, span-aware |
| `verify` | `"ok"` / `"unconfirmed"` | `verify_table()` | Whether PDF independently confirmed this value |
| `align` | `"left"` / `"right"` / `"center"` | DOCX XML | Horizontal text alignment |
| `bold` | `"true"` / absent | VLM (Qwen2.5-VL) | Visually bold in the original page image |
| `indent` | `"0"` – `"3"` | VLM (Qwen2.5-VL) | Visual indent level from the left margin |
| `namest` / `nameend` | column names | DOCX XML | Horizontal cell span |
| `colwidth` (on `<colspec>`) | proportional widths | DOCX XML | Relative column widths |

This snapshot is populated in two passes:

### Pass 1 — Content extraction (on upload)

`_load_docx_direct()` parses the DOCX with python-docx, converts each table to CALS XML preserving spans, column widths, and cell text.  `verify_table()` then cross-checks every cell value against pdfplumber + Camelot extractions of the rendered PDF and writes `verify="ok"` or `verify="unconfirmed"` onto each `<entry>`.

### Pass 2 — Style annotation (VLM, on demand)

`_annotate_entry_styles_with_vlm()` sends the page image (resized to ≤800 px JPEG) and a batch of cell texts to Qwen2.5-VL via LM Studio. The model returns `bold` (bool) and `indent` (0–3) for each cell. These are written back as XML attributes and the catalog is re-saved to disk.

---

## Content Validation — `verify_table()`

Validates that the text/numeric content of the CALS extraction matches the independently rendered PDF.

```
CALS XML  ──[normalise]──▶  cals_values  (set of normalised strings)
                                  │
                                  ▼  set difference
PDF page  ──[pdfplumber            │
           + Camelot               │
           + multi-line            │
             reconstruction]──▶  pdf_values
                                  │
                         ┌────────┴─────────┐
                    in_cals_not_pdf    in_pdf_not_cals
                    (unconfirmed)       (extra in PDF)
```

**Normalisation** strips `$`, `,`, `()`, whitespace and casts numbers to float strings so `"1,234"` and `1234.0` compare equal.

**Verdict thresholds:**

| Verdict | Condition |
|---|---|
| `PASS` | coverage ≥ 85 % AND no unconfirmed values |
| `WARN` | coverage ≥ 60 % OR some unconfirmed values |
| `FAIL` | coverage < 60 % |

**Output:** `verify="ok"` / `"unconfirmed"` on each `<entry>`, displayed as green / red cell backgrounds in the HTML table viewer and PDF output.

---

## Style Validation — VLM Snapshot

The VLM pass captures the **visual hierarchy** of the original document:

- **Bold rows** identify section headers and total/subtotal lines (e.g., "Total current assets", "Net income").
- **Indent levels** identify the indentation hierarchy of line items (e.g., indent=2 for "Cash and cash equivalents" under indent=1 "Current assets:").

These are ground-truth observations of the source document's intended structure, not inferred from text patterns.

### Current use

The `bold` and `indent` attributes are currently consumed by the rendering layer:

- `_cals_to_interactive_html()` — applies `font-weight:bold` and `padding-left:N×1.5em` inline CSS per cell.  
- `_cals_to_fop_pdf()` — (XSL-FO rendering, bold/indent not yet wired; see gap below).

### Gap — Style comparison not yet implemented

The style snapshot exists and is persisted, but there is no automated pass that **compares** the VLM snapshot of the source against a re-transformed output. This is the next logical validation layer.

---

## Full Snapshot Comparison — Design (NGTR-inspired)

### Background: NGTR

[NGTR](https://github.com/lqzxt/NGTR) (Neighbor-Guided Toolchain Reasoner, IJCAI 2025) is a VLLM-based table recognition framework built around three ideas that directly inform the gaps here:

1. **TEDS** (Tree Edit Distance Similarity) — a single `0–1` metric that compares two table trees on both structure (colspan/rowspan grid) and content (cell text via Levenshtein), with a `structure_only` mode that ignores content entirely.
2. **Neighbor-guided few-shot prompting** — find the most visually similar table in a reference set (ORB/SIFT image matching), use its ground-truth annotation as a context example when prompting the VLM.
3. **Reflection-driven toolchain selection** — before running the main recognition, a VLM selects image preprocessing steps (upscale, line-enhance, binarize, crop), evaluates each toolchain on neighbors using TEDS, then applies a per-step reflection: "did this step preserve or degrade information?"

### How these ideas close the gaps

#### 1. TEDS over CALS XML — unified structure + content score

NGTR's `TEDS` class operates on HTML trees (`<table>/<tr>/<td>` with `colspan`/`rowspan`). Our CALS XML is structurally equivalent. A `cals_to_table_tree()` converter would map:

| CALS | HTML/TEDS equivalent |
|---|---|
| `<entry>` text | `<td>` content |
| span from `namest`/`nameend` + `colspec` | `colspan` / `rowspan` |
| row sequence | `<tr>` order |

This gives `TEDS(snapshot_A, snapshot_B)` as a single correctness score. Running it in `structure_only=True` mode checks layout preservation independently of content drift.

The `CustomConfig.rename()` cost function in NGTR compares `tag + colspan + rowspan + content`. We extend it to also penalise style divergence:

```python
def rename(self, node1, node2):
    cost = 0.0
    if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
        return 1.0                       # structural mismatch — maximum cost
    if node1.content or node2.content:
        cost += 0.7 * levenshtein(node1.content, node2.content)  # content weight
    if node1.bold != node2.bold:
        cost += 0.15                     # bold changed
    if node1.indent != node2.indent:
        cost += 0.15 * abs(node1.indent - node2.indent) / 3  # indent drift
    return min(cost, 1.0)
```

This produces two scores per table comparison:

| Score | What it measures |
|---|---|
| `TEDS_full` | Structure + content + bold/indent |
| `TEDS_struct` | Structure only (colspan/rowspan grid, row order) |

#### 2. Neighbor-guided validation

For tables of the same type across different documents (e.g., Balance Sheets from Q1 and Q2 filings), ORB/SIFT image similarity identifies the closest neighbor in `table_catalog.json`. The neighbor's annotated CALS XML becomes a reference for VLM prompting:

- When re-annotating a re-transformed table, include the neighbor's HTML rendering as a few-shot example: "this is how a table of this type should look" — improving bold/indent detection accuracy.
- When comparing two snapshots, also report `TEDS(snapshot_A, neighbor)` vs `TEDS(snapshot_B, neighbor)` — a large divergence between the two TEDS-vs-neighbor scores signals structural regression even without a direct A/B comparison.

#### 3. VLM reflection on re-transformations

NGTR's `image_reflection` prompt asks the VLM to compare a before/after image pair and decide which retains more information. The same pattern applies here as a post-transformation check:

```
[Original page image]  vs  [Re-rendered PDF/HTML screenshot]
      → VLM: "Does the re-rendered table preserve the visual hierarchy
               (bold headers, indent levels, column alignment) of the original?"
      → Returns: { "preserved": bool, "issues": ["lost bold on Total row", ...] }
```

This is a qualitative check complementary to the quantitative TEDS score.

#### 4. Image preprocessing toolchain for VLM annotation

NGTR selects preprocessing tools (upscale → line-enhance → crop) based on a VLM-scored toolchain, evaluated on neighbors. Our VLM annotation step currently sends a plain resized JPEG. Applying the same approach — try several preprocessing variants on a neighbor table where the ground truth is known, pick the variant with the best annotation quality — would improve `bold`/`indent` detection on low-quality scans or borderless tables.

### Proposed `compare_snapshots` architecture

```
snapshot_A (CALS XML, original)
snapshot_B (CALS XML, re-transformed)
          │
          ▼
  cals_to_table_tree(A)    cals_to_table_tree(B)
          │                        │
          └─────────┬──────────────┘
                    │
              TEDS(A, B, CustomConfig)
                    │
         ┌──────────┴────────────┐
    TEDS_full              TEDS_struct
  (content+style         (span grid +
   +structure)            row order only)
                    │
              per-cell diff
    { "lost_bold": [...],
      "indent_changed": [(text, old, new), ...],
      "missing_values": [...],
      "span_changed": [...] }
```

```python
def compare_snapshots(xml_a: str, xml_b: str) -> dict:
    """
    Returns:
      {
        "teds_full":       float,   # 0–1, all dimensions
        "teds_struct":     float,   # 0–1, structure only
        "verdict":         str,     # "PASS" / "WARN" / "FAIL"
        "lost_bold":       list,    # cell texts that lost bold="true"
        "gained_bold":     list,    # bold added that wasn't in original
        "indent_changed":  list,    # (text, old_indent, new_indent)
        "missing_values":  list,    # in A not in B (normalised)
        "extra_values":    list,    # in B not in A
        "span_changed":    list,    # (text, old_span, new_span)
      }
    """
```

**Verdict thresholds (proposed):**

| Verdict | Condition |
|---|---|
| `PASS` | `teds_full ≥ 0.95` AND `teds_struct = 1.0` |
| `WARN` | `teds_full ≥ 0.80` OR `teds_struct ≥ 0.95` |
| `FAIL` | either below WARN threshold |

---

## File Locations

| File | Purpose |
|---|---|
| `data/table_catalog.json` | Persisted snapshots — array of table entries each with `"xml"` (annotated CALS), `"image_path"`, `"pdf_path"`, `"page_idx"`, `"title"` |
| `data/table_images/page_NNN.png` | Original page images used for VLM annotation and diff overlay |
| `code/chatui/utils/database.py` | All extraction, validation, and rendering logic |
| `code/chatui/pages/converse.py` | UI wiring — upload, re-annotate button, table browser |
