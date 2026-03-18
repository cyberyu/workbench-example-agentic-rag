"""
Pipeline:
  1. python-docx: recolor all runs in-memory:
       - Non-table paragraph runs → red font
       - Table cell runs          → blue font
  2. Save the recolored copy as a temp .docx
  3. LibreOffice (headless): colored.docx → colored.pdf
  4. PyMuPDF (fitz): render each PDF page → PNG at 150 DPI
  5. PNGs saved to ./pages/page_NNNN.png
"""

import os
import subprocess
import sys
import tempfile

import fitz  # PyMuPDF

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.shared import RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
RED  = RGBColor(0xCC, 0x00, 0x00)   # dark red  – non-table text
BLUE = RGBColor(0x00, 0x00, 0xCC)   # dark blue – table text
DPI  = 150


# ── iter_block_items ──────────────────────────────────────────────────────────

def iter_block_items(parent):
    """Yield Paragraph / Table objects in document order.
    Works for Document, _Cell, and header/footer objects."""
    if isinstance(parent, DocumentType):
        parent_elm = parent.element.body
    elif hasattr(parent, "_tc"):
        parent_elm = parent._tc        # _Cell
    else:
        parent_elm = parent._element   # Header / Footer
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


# ── coloring helpers ──────────────────────────────────────────────────────────

def color_paragraph(para, color):
    """Apply color to every run in a paragraph."""
    for run in para.runs:
        run.font.color.rgb = color


def color_table(table, color):
    """Recursively color every paragraph in every cell of a table."""
    for row in table.rows:
        for cell in row.cells:
            for block in iter_block_items(cell):
                if isinstance(block, Paragraph):
                    color_paragraph(block, color)
                elif isinstance(block, Table):
                    color_table(block, color)


# ── recolor ───────────────────────────────────────────────────────────────────

def recolor_header_footer(hf_part, color):
    """Recolor all paragraphs and tables inside a header or footer."""
    if hf_part is None:
        return
    for block in iter_block_items(hf_part):
        if isinstance(block, Paragraph):
            color_paragraph(block, color)
        elif isinstance(block, Table):
            color_table(block, color)


def recolor_document(src_path, dst_path):
    doc = Document(src_path)

    # Recolor body blocks
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            color_paragraph(block, RED)
        elif isinstance(block, Table):
            color_table(block, BLUE)

    # Recolor running headers and footers (stored outside the body)
    for section in doc.sections:
        for hf in (
            section.header,
            section.footer,
            section.even_page_header,
            section.even_page_footer,
            section.first_page_header,
            section.first_page_footer,
        ):
            recolor_header_footer(hf, RED)

    doc.save(dst_path)
    print(f"  Recolored docx → {dst_path}")


# ── LibreOffice: docx → PDF ───────────────────────────────────────────────────

def docx_to_pdf(docx_path, outdir):
    cmd = [
        SOFFICE,
        "--headless",
        "--norestore",
        "--convert-to", "pdf",
        "--outdir", outdir,
        docx_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice failed:\n{result.stderr}")
    stem = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = os.path.join(outdir, stem + ".pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Expected PDF not found: {pdf_path}")
    print(f"  PDF → {pdf_path}")
    return pdf_path


# ── PyMuPDF: PDF → per-page PNGs ─────────────────────────────────────────────

def pdf_to_images(pdf_path, out_dir, dpi=DPI):
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"  Rendering {total} pages at {dpi} DPI …")
    for page_num in range(total):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        dst = os.path.join(out_dir, f"page_{page_num + 1:04d}.png")
        pix.save(dst)
        print(f"    page {page_num + 1:>4}/{total} → {dst}")
    doc.close()
    return total


# ── entry point ───────────────────────────────────────────────────────────────

def main(docx_path):
    docx_path = os.path.abspath(docx_path)
    stem = os.path.splitext(os.path.basename(docx_path))[0]
    out_dir = os.path.join(os.path.dirname(docx_path), f"{stem}_pages")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1 – recolor with python-docx
        print("[1/3] Recoloring document …")
        colored_docx = os.path.join(tmp, "colored.docx")
        recolor_document(docx_path, colored_docx)

        # Step 2 – docx → PDF via LibreOffice
        print("[2/3] Converting to PDF via LibreOffice …")
        pdf_path = docx_to_pdf(colored_docx, tmp)

        # Step 3 – PDF → PNG pages via PyMuPDF
        print("[3/3] Rendering pages via PyMuPDF …")
        total = pdf_to_images(pdf_path, out_dir)

    print(f"\nDone. {total} page image(s) written to: {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_pages.py <document.docx>")
        sys.exit(1)
    main(sys.argv[1])
