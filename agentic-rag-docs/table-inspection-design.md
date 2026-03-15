# Table Extraction Inspection — Design Document

## Problem Statement

When extracting tables from `.docx` files via python-docx and rendering them as
`tabulate` text grids, the extracted text must be verified against the original
document visuals to ensure:

1. All numbers are complete and correct
2. All numbers are in the correct relative positions within the tabular structure
3. All text values are in the correct relative positions within the tabular structure

The ground-truth reference is the page PNG rendered by LibreOffice (already
produced during document loading as `data/table_images/page_NNN.png`) and the
intermediate PDF (`data/table_images/<docname>.pdf`).

---

## Available Inputs

| Artifact | How produced | Contents |
|---|---|---|
| `cell_rows[row][col]` | python-docx extraction | Extracted table text |
| `data/table_images/<name>.pdf` | LibreOffice headless | Pixel-faithful PDF render |
| `data/table_images/page_NNN.png` | pdf2image at 150 DPI | Page image for display |

---

## Approach A — PDF Structured Text with pdfplumber (Recommended Phase 1)

**Best for:** number completeness, cell position correctness, shape verification.

`pdfplumber` extracts tables from PDFs with cell bounding boxes, returning a 2D
list that matches the visual grid exactly — including empty cells and column spans.

```
PDF (LibreOffice render)
  └── pdfplumber.extract_table()
        └── reference_rows[row][col]   ← structural ground truth
                    ↕ compare
              cell_rows[row][col]       ← python-docx extraction
```

**What it verifies:**
- ✅ Number completeness: `set(numbers_in_reference) - set(numbers_in_extracted)`
- ✅ Position correctness: `reference_rows[r][c] == cell_rows[r][c]`
- ✅ Row/column count match (shape check)
- ✅ Text position correctness

**Weakness:** pdfplumber's lattice-mode table detection can fail on tables with no
visible grid lines; `stream` mode (whitespace-based) is needed as fallback.

**Installation:** `pip install pdfplumber`

---

## Approach B — pypdf Layout Mode (Already installed, weaker)

`pypdf 6.x` `extraction_mode="layout"` preserves whitespace column alignment.
Parse the resulting text into lines, infer columns from whitespace gaps.

```python
from pypdf import PdfReader
text = PdfReader(pdf_path).pages[page_idx].extract_text(extraction_mode="layout")
```

**What it verifies:**
- ✅ Number completeness (regex-extract all numbers, compare sets)
- ⚠️ Position: rough — column alignment inferred from whitespace, unreliable for
  complex merged-cell tables
- ❌ Merged cell structure lost

No new dependencies. Suitable as a lightweight pre-check before pdfplumber.

---

## Approach C — OCR on Page PNG (Image-based, most robust)

Crop the table region from the page PNG, run Tesseract with `--psm 6`
(uniform block of text), cluster resulting words by Y/X coordinates into rows
and columns.

```
page_NNN.png
  └── pytesseract (--psm 6, bounding boxes)
        └── words with (x, y, w, h)
              └── cluster by Y → rows
                    └── cluster by X → columns
                          └── ocr_rows[row][col]
                                    ↕ compare
                              cell_rows[row][col]
```

**What it verifies:**
- ✅ All of the above
- ✅ Catches rendering artifacts invisible to text extraction
- ✅ Works even if PDF text extraction fails (scanned/image-embedded tables)
- ⚠️ OCR errors on small fonts or very low DPI renders (use ≥200 DPI for OCR)

**Installation:** `pip install pytesseract` + system `tesseract-ocr` package

---

## Approach D — Vision LLM as Judge (Highest accuracy, highest cost)

Send the page PNG and extracted table text to a multimodal model (e.g. Gemma3,
GPT-4V). Prompt:

> *"Here is an extracted table and the original page image. List any values that
> are missing, incorrect, or in the wrong position."*

**What it verifies:** everything — including color-coded cells, footnote markers
(`*`, `†`), merged-cell semantics, and formatting nuances.

**Tradeoffs:** slow, requires a vision-capable endpoint, non-deterministic output.
Best used as a final audit on tables where Phase 1/2 report failures.

---

## Recommended Pipeline: Two-Phase Inspection

```
Phase 1 — Structural number check  (fast, text-based, pdfplumber)
  ├── Open existing PDF  →  pdfplumber.extract_table() on the matched page
  ├── Normalize cells: strip "$", ",", "(", ")" → float where numeric
  ├── Diff against cell_rows[r][c]
  └── Report:
        - shape_match: bool
        - missing_values: list of values in reference but not in extracted
        - misplaced_values: list of {value, reference_pos, extracted_pos}
        - extra_values: list of values in extracted but not in reference

Phase 2 — Visual spot-check  (image-based, on demand or when Phase 1 fails)
  ├── Tesseract OCR on page PNG  →  ocr_rows[r][c]
  ├── Compare with both reference_rows and cell_rows
  └── Report: which source (docx vs pdf) is closer to the image ground truth
```

### Output format — diff report per table

```json
{
  "table": "BALANCE SHEETS",
  "page_image": "data/table_images/page_003.png",
  "shape": {
    "extracted": [42, 3],
    "reference": [42, 3],
    "match": true
  },
  "missing_values": ["(1,234)", "49,401"],
  "misplaced": [
    {"value": "1,234", "extracted_pos": [5, 2], "reference_pos": [5, 1]}
  ],
  "extra_values": [],
  "verdict": "FAIL"
}
```

---

## Implementation Plan

### Step 1 — Install pdfplumber

```bash
pip install pdfplumber
```

Add to `requirements.txt`.

### Step 2 — `verify_table()` in `database.py`

```python
def verify_table(cell_rows, pdf_path, page_idx, table_title=""):
    """
    Phase 1: Compare extracted cell_rows against pdfplumber's PDF table extraction.

    Returns a dict with keys: table, shape, missing_values, misplaced, extra_values, verdict.
    """
    import pdfplumber, re

    def _normalize(v):
        v = re.sub(r"[$,\(\)\s]", "", str(v))
        try:
            return float(v.replace("—", "").strip()) if v else None
        except ValueError:
            return v.lower().strip() if v else None

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        ref = page.extract_table()          # lattice mode
        if ref is None:
            ref = page.extract_table(table_settings={"vertical_strategy": "text",
                                                      "horizontal_strategy": "text"})

    if ref is None:
        return {"table": table_title, "verdict": "SKIP", "reason": "pdfplumber found no table"}

    # Flatten to sets of normalized values for completeness check
    ref_values  = {_normalize(c) for row in ref       for c in row if c}
    ext_values  = {_normalize(c) for row in cell_rows for c in row if c}

    missing = sorted(str(v) for v in ref_values  - ext_values  if v is not None)
    extra   = sorted(str(v) for v in ext_values  - ref_values  if v is not None)

    # Cell-by-cell position check (up to the shorter of the two)
    misplaced = []
    for r in range(min(len(ref), len(cell_rows))):
        for c in range(min(len(ref[r]), len(cell_rows[r]))):
            rv = _normalize(ref[r][c])
            ev = _normalize(cell_rows[r][c])
            if rv != ev and rv is not None and ev is not None:
                misplaced.append({"value_ref": str(rv), "value_ext": str(ev),
                                   "pos": [r, c]})

    shape_match = (len(ref) == len(cell_rows) and
                   len(ref[0]) == len(cell_rows[0]) if ref and cell_rows else False)

    verdict = "PASS" if not missing and not misplaced and shape_match else "FAIL"

    return {
        "table": table_title,
        "shape": {"extracted": [len(cell_rows), len(cell_rows[0]) if cell_rows else 0],
                  "reference": [len(ref), len(ref[0]) if ref else 0],
                  "match": shape_match},
        "missing_values": missing[:20],   # cap for readability
        "misplaced": misplaced[:20],
        "extra_values": extra[:20],
        "verdict": verdict,
    }
```

### Step 3 — Wire into the UI (optional)

Add a "Verify Extraction" button in the Documents tab that:
1. Iterates all loaded `.docx` files
2. Calls `verify_table()` for each table
3. Displays the JSON diff report in the Monitor tab (Response Trace)
4. Shows a pass/fail badge per table

---

## Known Limitations

- Word documents with **no-border tables** may confuse pdfplumber's lattice mode;
  stream mode fallback mitigates this but may mis-align columns.
- **Footnote markers** (`*`, `1`, `(a)`) attached to cell values will cause false
  positives in the number comparison; strip them in `_normalize()` if needed.
- **Thousands separators** in non-US locales (`.` instead of `,`) need locale-aware
  normalization.
- **Nested tables** (sub-tables inside a cell) are flattened by python-docx into the
  parent cell's text; pdfplumber may represent them differently.
