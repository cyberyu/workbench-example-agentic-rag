# Table Extraction & Annotation Pipeline — Word → XPP

This document summarises the end-to-end approach for extracting, annotating, validating, and transforming financial tables from a Word (`.docx`) source document into XPP-ready CALS XML.

---

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         Source Document                         │
│                          (DOCX / Word)                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │  python-docx
                                ▼
                   ┌────────────────────────┐
                   │   CALS XML (per table) │  ← cell text, spans,
                   │   stored in catalog    │    column widths
                   └────────────┬───────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
   ┌───────────────────────┐       ┌──────────────────────────┐
   │  PASS 1 — Content     │       │  PASS 2 — Style          │
   │  Verification         │       │  Annotation              │
   │  (pdfplumber+Camelot) │       │  (pdfplumber + VLM)      │
   └───────────┬───────────┘       └──────────────┬───────────┘
               │                                  │
               │  verify="ok"|"unconfirmed"        │  bold="true"
               │  on every <entry>                 │  indent="0"–"3"
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
                   ┌────────────────────────┐
                   │  Annotated CALS XML    │   ← canonical snapshot
                   │  (table_catalog.json)  │
                   └────────────┬───────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
   ┌───────────────────────┐       ┌──────────────────────────┐
   │  Interactive HTML     │       │  XSL-FO PDF (FOP)        │
   │  (cell colouring,     │       │  ready for XPP ingestion │
   │   verify badges)      │       │                          │
   └───────────────────────┘       └──────────────────────────┘
```
python-docx — structural extraction
python-docx reads the raw .docx XML directly. It is the primary extraction tool and is used to build the canonical CALS XML:

Cell text — reads <w:tc> paragraph runs, preserving whitespace and special characters exactly as authored
Merged cells — decodes <w:hMerge>, <w:vMerge> into namest/nameend/morerows CALS span attributes
Column widths — reads <w:tblGrid><w:gridCol w:w="..."/> proportional widths → colwidth on <colspec>
Alignment — reads <w:jc> paragraph justification → align attribute per cell
None of this information survives in a rendered PDF. A PDF is a flat page description — it has no concept of "this number is in column 3, row 2, spanning 2 columns". python-docx is the only tool that can recover the logical table structure.
---

## Pass 1 — Content Verification

The DOCX is rendered to PDF by LibreOffice.  Two independent PDF readers extract every cell value from that rendered page; their union is compared against the CALS extraction.

```
DOCX ──[LibreOffice]──▶ PDF
                          │
             ┌────────────┴────────────┐
             │                        │
             ▼                        ▼
      pdfplumber                  Camelot
  (word-level bounding        (lattice / stream
   box extraction)             line-detection)
             │                        │
             └────────────┬───────────┘
                          │ union of extracted values
                          ▼
                ┌──────────────────┐
                │  pdf_values set  │
                └────────┬─────────┘
                         │  normalised diff
                         ▼
                ┌──────────────────┐
                │  cals_values set │ ← from python-docx CALS
                └────────┬─────────┘
                         │
              ┌──────────┴───────────┐
              │                      │
         confirmed               unconfirmed
       verify="ok"           verify="unconfirmed"
    (green in viewer)          (red in viewer)
```

**Normalisation** strips `$`, `,`, parentheses, and whitespace so `"(1,234)"` and `-1234.0` compare as equal.

| Verdict | Condition |
|---|---|
| `PASS` | coverage ≥ 85 % and no unconfirmed values |
| `WARN` | coverage ≥ 60 % or some unconfirmed values |
| `FAIL` | coverage < 60 % |

---

## Pass 2 — Style Annotation

Style attributes (`bold`, `indent`) capture the **visual hierarchy** of the original document — section headers, totals rows, and indented line items.  Two independent signals are collected and then reconciled by an agent.

### Signal A — pdfplumber (font metadata from PDF)

```
LibreOffice PDF
      │
      ▼  pdfplumber.extract_words(extra_attrs=["fontname","size"])
┌───────────────────────────────────────────────────────┐
│  word record: { text, x0, top, fontname }             │
│                                                       │
│  "LiberationSans-Bold"  →  bold = true                │
│  "LiberationSans"       →  bold = false               │
│                                                       │
│  x0 relative to page left margin:                     │
│    Δx < 14 pt  →  indent 0                            │
│    14–28 pt    →  indent 1                            │
│    28–42 pt    →  indent 2                            │
│    > 42 pt     →  indent 3                            │
└───────────────────────────────────────────────────────┘
      │
      ▼
 bold / indent written onto label-column <entry> elements
 (value columns: bold from font; indent always stripped)
```

**Performance:** `_build_pdf_page_cache()` opens the PDF **once**, pre-computes `(line_lookup, word_lookup, x0_min, prefix_index)` for every page, then passes the cache to the annotator — so all 60+ tables in a single document share one file-open.  Cell-text matching uses an O(1) prefix index instead of a linear scan.

### Signal B — Vision LLM (page image)

```
page_NNN.png (LibreOffice render, ≤ 512 px longest side)
      │
      ▼  base64-encoded JPEG → vision LLM (OpenAI-compatible API)
┌───────────────────────────────────────────────────────┐
│  Prompt: list of cell texts from non-header rows      │
│                                                       │
│  Expected response (JSON per cell):                   │
│    { "text": "...", "bold": true/false,               │
│                     "indent": 0|1|2|3 }               │
└───────────────────────────────────────────────────────┘
      │
      ▼
 bold / indent written onto matching <entry> elements
```

Long tables are batched (25 cells per request) so the context window is never exceeded.

---

## Annotation Agent — Reconciliation Loop

The agent runs both signals on every table and reconciles disagreements via a local text-only LLM, then stores all versions for review.

```
┌──────────────────────────────────────────────────────────────────┐
│                      Annotation Agent                           │
│  (runs on demand for all tables in the catalog)                 │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼  for each table
   ┌─────────────────────┐     ┌─────────────────────┐
   │   pdfplumber XML    │     │     VLM XML          │
   │  (font-accurate)    │     │  (layout-aware but   │
   │                     │     │   image-limited)     │
   └──────────┬──────────┘     └──────────┬──────────┘
              │                           │
              └────────────┬──────────────┘
                           │  _compare_annotation_sets()
                           ▼
                    conflict list
          [{text, pdf:{bold,indent}, vlm:{bold,indent}}, …]
                           │
                           │  if conflicts > 0
                           ▼
                  _reconcile_with_llm()
               (text-only LLM, rule-based:
                trust pdfplumber for bold/indent)
                           │
                           ▼
                  _apply_reconciliation()
                           │
                           ▼
               ┌───────────────────────┐
               │  Final CALS XML       │
               │  xml_pdfplumber ─┐    │   all three versions
               │  xml_vlm ────────┤    │   stored in catalog
               │  xml (final) ────┘    │
               └───────────────────────┘
```

Results are persisted every 5 tables (not after every table) to reduce I/O on large documents.

### Annotation Review Tab

The Table Browser UI includes a **🔎 Annotation Review** panel that shows a side-by-side per-entry comparison:

| Cell text | PDF bold | PDF indent | VLM bold | VLM indent | Match | Final bold | Final indent |
|---|---|---|---|---|---|---|---|
| Revenue: | false | 0 | **true** | 0 | ⚠ | false | 0 |
| Total revenue | false | 2 | false | 0 | ⚠ | false | 2 |
| Operating income | false | 1 | false | 1 | ✓ | false | 1 |

Conflicting rows are highlighted in amber.  A summary badge shows the annotation method, entry count, and number of conflicts.

---

## CALS XML Schema (annotated)

Each table is stored as a CALS XML document.  Every `<entry>` may carry:

```xml
<tgroup cols="3">
  <colspec colname="c1" colwidth="3*"/>   <!-- proportional widths from DOCX -->
  <colspec colname="c2" colwidth="1*"/>
  <colspec colname="c3" colwidth="1*"/>
  <tbody>
    <row>
      <entry bold="true" indent="0" verify="ok">Total revenue</entry>
      <entry bold="true"            verify="ok">77,673</entry>
      <entry bold="true"            verify="ok">61,751</entry>
    </row>
    <row>
      <entry indent="1" verify="ok">Product</entry>
      <entry            verify="ok">26,764</entry>
      <entry            verify="ok">20,984</entry>
    </row>
  </tbody>
</tgroup>
```

| Attribute | Values | Source |
|---|---|---|
| `bold` | `"true"` / absent | pdfplumber font name or VLM |
| `indent` | `"0"` – `"3"` | pdfplumber x0 offset or VLM |
| `verify` | `"ok"` / `"unconfirmed"` | pdfplumber + Camelot value diff |
| `namest` / `nameend` | colspec names | python-docx (horizontal span) |
| `morerows` | integer | python-docx (vertical span) |
| `align` | `"left"` / `"right"` / `"center"` | DOCX paragraph alignment |

---

## Output — XSL-FO / XPP

The annotated CALS XML is transformed to XSL-FO by `_cals_to_fop_pdf()` and rendered by Apache FOP.  The `bold` and `indent` attributes drive typographic decisions:

```
CALS XML
   │
   ▼  _cals_to_fop_pdf()  (inline XSL-FO generation)
┌─────────────────────────────────────────────────────────────┐
│  <fo:table>                                                │
│    <fo:table-column column-width="proportional-column-     │
│      width(3)"/>  …                                        │
│    <fo:table-body>                                         │
│      <fo:table-row>                                        │
│        <fo:table-cell>                                     │
│          <fo:block font-weight="bold"          ← bold="true"│
│                    start-indent="1.5em">       ← indent="1" │
│            Total revenue                                   │
│          </fo:block>                                       │
│        </fo:table-cell>                                    │
│      </fo:table-row>                                       │
│    </fo:table-body>                                        │
│  </fo:table>                                               │
└─────────────────────────────────────────────────────────────┘
   │
   ▼  Apache FOP
 output.pdf  (XPP-ingestible)
```

---

## Key Files

| File | Role |
|---|---|
| `code/chatui/utils/database.py` | All extraction, verification, annotation, and rendering logic |
| `code/chatui/pages/converse.py` | Gradio UI — upload, table browser, annotation agent button |
| `data/table_catalog.json` | Persisted catalog — annotated CALS XML + metadata per table |
| `data/table_images/page_NNN.png` | Page images used for VLM annotation and diff overlays |
| `data/table_images/*.pdf` | LibreOffice-rendered PDF used by pdfplumber and Camelot |

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `STYLE_ANNOTATE_METHOD` | `none` | `pdfplumber` / `vlm` / `none` — which annotator runs on upload |
| `VLM_LMSTUDIO_URL` | `http://localhost:1234/v1` | OpenAI-compatible API endpoint for the vision LLM |
| `VLM_STYLE_ANNOTATE` | `no` | Legacy alias: `yes` → sets `STYLE_ANNOTATE_METHOD=vlm` |
