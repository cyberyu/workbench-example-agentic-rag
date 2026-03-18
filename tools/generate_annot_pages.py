#!/usr/bin/env python3
"""
generate_annot_pages.py
=======================
Standalone script that regenerates annotated page images (blue = table content,
grey = standalone paragraph text) by recolouring the source .docx with
python-docx, converting it to PDF via LibreOffice, and rendering each page
to PNG via PyMuPDF.

This mirrors the logic in database._create_annotated_pages_from_docx() so
you can iterate quickly without restarting the app or re-uploading.

Usage
-----
  python3 tools/generate_annot_pages.py [--pages 2 5 10] [--out data/annot_debug]

Arguments
---------
  --pages N [N ...]   0-based page indices to render (default: all pages)
  --out DIR           Output directory (default: data/annot_debug)
  --docx PATH         .docx source file (auto-detected from data/ if omitted)
  --dpi N             Render resolution in DPI (default: 150)
  --verbose           Print progress messages
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pages", nargs="*", type=int, default=None,
                   help="0-based page indices to render (default: all)")
    p.add_argument("--out", default="data/annot_debug",
                   help="Output directory for annotated PNGs")
    p.add_argument("--docx", default=None,
                   help=".docx source file (auto-detected from data/ if omitted)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Render resolution in DPI (default: 150)")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress messages")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_docx(data_dir="data"):
    """Auto-detect the first .docx file under data/."""
    import glob
    hits = glob.glob(os.path.join(data_dir, "*.docx")) + \
           glob.glob(os.path.join(data_dir, "**", "*.docx"), recursive=True)
    return hits[0] if hits else None


# ── Recolor the .docx (mirrors database._recolor_docx_for_annotation) ────────

def _recolor_docx(src_docx, dst_docx):
    """Recolor every run in src_docx and save to dst_docx.

    paragraph runs  → GREY  (130, 130, 130)
    table cell runs → BLUE  ( 41, 128, 185)
    header/footer   → GREY
    """
    from docx import Document
    from docx.document import Document as DocumentType
    from docx.oxml.ns import qn
    from docx.shared import RGBColor
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    BLUE = RGBColor(0x29, 0x80, 0xB9)
    RED  = RGBColor(0xFF, 0x00, 0x00)

    def _iter_blocks(parent):
        if isinstance(parent, DocumentType):
            elm = parent.element.body
        elif hasattr(parent, "_tc"):
            elm = parent._tc
        else:
            elm = parent._element
        for child in elm.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, parent)
            elif child.tag == qn("w:tbl"):
                yield Table(child, parent)

    def _color_para(para, color):
        for run in para.runs:
            run.font.color.rgb = color

    def _color_table(table, color):
        for row in table.rows:
            for cell in row.cells:
                for block in _iter_blocks(cell):
                    if isinstance(block, Paragraph):
                        _color_para(block, color)
                    elif isinstance(block, Table):
                        _color_table(block, color)

    doc = Document(src_docx)
    for block in _iter_blocks(doc):
        if isinstance(block, Paragraph):
            _color_para(block, RED)
        elif isinstance(block, Table):
            _color_table(block, BLUE)
    for section in doc.sections:
        for hf in (section.header, section.footer,
                   section.even_page_header, section.even_page_footer,
                   section.first_page_header, section.first_page_footer):
            if hf is None:
                continue
            for block in _iter_blocks(hf):
                if isinstance(block, Paragraph):
                    _color_para(block, RED)
                elif isinstance(block, Table):
                    _color_table(block, BLUE)
    doc.save(dst_docx)
    print(f"[recolor] saved {dst_docx}")


# ── Render recolored pages to PNG ─────────────────────────────────────────────

def render_annotated_pages(docx_path, out_dir, page_indices=None, dpi=150, verbose=False):
    """Full pipeline: recolor docx → LibreOffice PDF → PyMuPDF PNGs.

    Returns dict: page_idx (0-based) → absolute path of annotated PNG.
    """
    try:
        import fitz
    except ImportError:
        fitz = None
        print("[render] PyMuPDF (fitz) not found — falling back to pdf2image")

    os.makedirs(out_dir, exist_ok=True)
    result_map = {}

    with tempfile.TemporaryDirectory() as tmp:
        colored_docx = os.path.join(tmp, "colored_annot.docx")
        try:
            _recolor_docx(docx_path, colored_docx)
        except Exception as exc:
            print(f"[render] recolor failed: {exc}")
            import traceback; traceback.print_exc()
            return result_map

        pdf_out = os.path.join(tmp, "colored_annot.pdf")
        try:
            r = subprocess.run(
                ["libreoffice", "--headless", "--norestore",
                 "--convert-to", "pdf", "--outdir", tmp, colored_docx],
                capture_output=True, text=True, timeout=180,
            )
            if verbose:
                if r.stdout: print("[libreoffice]", r.stdout.strip())
                if r.stderr: print("[libreoffice stderr]", r.stderr.strip()[:400])
            if r.returncode != 0 or not os.path.exists(pdf_out):
                print(f"[render] LibreOffice failed (rc={r.returncode})")
                return result_map
        except Exception as exc:
            print(f"[render] LibreOffice error: {exc}")
            return result_map

        if verbose:
            print(f"[render] PDF ready: {pdf_out}")

        if fitz:
            zoom = dpi / 72.0
            mat  = fitz.Matrix(zoom, zoom)
            doc  = fitz.open(pdf_out)
            total = len(doc)
            for pnum in range(total):
                if page_indices is not None and pnum not in page_indices:
                    continue
                pix = doc[pnum].get_pixmap(matrix=mat, alpha=False)
                out = os.path.join(out_dir, f"page_{pnum:03d}_annot.png")
                pix.save(out)
                result_map[pnum] = out
                if verbose:
                    print(f"  page {pnum:3d} → {out}")
            doc.close()
        else:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_out, dpi=dpi)
            for pnum, img in enumerate(images):
                if page_indices is not None and pnum not in page_indices:
                    continue
                out = os.path.join(out_dir, f"page_{pnum:03d}_annot.png")
                img.save(out, "PNG")
                result_map[pnum] = out
                if verbose:
                    print(f"  page {pnum:3d} → {out}")

    print(f"[render] {len(result_map)} page(s) written to {out_dir}/")
    return result_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve paths relative to workspace root
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(workspace)

    docx_path = args.docx or find_docx()
    if not docx_path or not os.path.exists(docx_path):
        sys.exit("[ERROR] No .docx found. Use --docx to specify one.")
    print(f"docx: {docx_path}")

    page_indices = set(args.pages) if args.pages is not None else None
    if page_indices is not None:
        print(f"Rendering pages: {sorted(page_indices)}")
    else:
        print("Rendering all pages")

    result = render_annotated_pages(
        docx_path,
        args.out,
        page_indices=page_indices,
        dpi=args.dpi,
        verbose=args.verbose,
    )

    if result:
        print(f"\nDone — {len(result)} page(s) in {args.out}/")
    else:
        print("\n[ERROR] No pages rendered.")
        sys.exit(1)


if __name__ == "__main__":
    main()
