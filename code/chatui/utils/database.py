# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader 
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse
import os
import shutil
import mimetypes



# Handling which API to use based on public vs NVIDIA internal
# Check if the INTERNAL_API environment variable is set to yes. Don't set if not NVIDIA employee, as you can't access them.
INTERNAL_API = os.getenv('INTERNAL_API', 'no')

# Local embedding model (runs via sentence-transformers, no API key needed)
EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'

# Get project root dynamically - works both inside and outside AI Workbench
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Set the chunk size and overlap for the text splitter. Uses defaults but allows them to be set as environment variables.
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 0

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))


print(f"[config] Using local embedding model: {EMBEDDINGS_MODEL}")


# Adding nltk data
import nltk

def download_nltk_if_missing():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_if_missing()

    
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

# Functions for dealing with URLs
def is_valid_url(url: str) -> bool:
    """ This is a helper function for checking if the URL is valid. It isn't fail proof, but it will catch most common errors. """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def safe_load(url):
    """ This is a helper function for loading the URL. It protects against false negatives from is_value_url and 
        filters for actual web pages. Returns None if it fails.
    """
    try:
        return WebBaseLoader(url).load()
    except Exception as e:
        print(f"[upload] Skipping {url}: {e}")
        return None


def upload(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

    urls = [url for url in urls if is_valid_url(url)]

    docs = []
    for url in urls:
        result = safe_load(url)
        if result is not None:
            docs.append(result)

    docs_list = [item for sublist in docs for item in sublist]

    if not docs_list:
        # If no documents were loaded, return None
        print("[upload] No URLs provided.")
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
            persist_directory=DATA_DIR,
        )
        return vectorstore

    except Exception as e:
        print(f"[upload] Vectorstore creation failed: {e}")
        return None


# Functions for dealing with file uploads/embeddings

## Helper functions

def _indent_xml(elem, level: int = 0) -> None:
    """Add pretty-print indentation to an ElementTree element in-place."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        last = None
        for child in elem:
            _indent_xml(child, level + 1)
            last = child
        if last is not None and (not last.tail or not last.tail.strip()):
            last.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def _build_cals_xml(table, title: str = "", table_id: str = "") -> str:
    """Build an OASIS CALS table XML string from a python-docx Table object.

    Maps Word XML properties to CALS attributes:
      - w:gridSpan   → namest / nameend  (horizontal column span)
      - w:vMerge     → morerows="N"     (vertical row span count)
      - paragraph jc → align            (right / center / justify; left is default)
      - w:tblGrid    → proportional colwidth values
      - first row    → <thead>  (remaining rows go into <tbody>)

    Returns a well-formed XML fragment (no XML declaration) ready for embedding
    in an SGML/XML XPP source document.
    """
    import xml.etree.ElementTree as ET
    from docx.oxml.ns import qn

    tbl = table._tbl  # lxml element representing <w:tbl>

    # Convenience: extract all text from a <w:tc> lxml element
    def _tc_text(tc_el) -> str:
        parts = []
        for p in tc_el.findall(qn("w:p")):
            run_text = "".join(t.text or "" for t in p.findall(f".//{qn('w:t')}")).strip()
            if run_text:
                parts.append(run_text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 1. Column widths from <w:tblGrid>
    # ------------------------------------------------------------------
    tbl_grid = tbl.find(qn("w:tblGrid"))
    col_widths: list = []
    if tbl_grid is not None:
        for gc in tbl_grid.findall(qn("w:gridCol")):
            w_val = gc.get(qn("w:w"))
            try:
                col_widths.append(int(w_val) if w_val else 1440)
            except ValueError:
                col_widths.append(1440)

    if not col_widths:
        # Fallback: count unique cells in first row
        try:
            seen: set = set()
            for c in table.rows[0].cells:
                seen.add(id(c._tc))
            col_widths = [1440] * len(seen)
        except Exception:
            col_widths = [1440]

    num_cols = len(col_widths)
    total_w = sum(col_widths) or 1

    # ------------------------------------------------------------------
    # 2. Build per-row cell info list using raw XML (not row.cells)
    #
    # row.cells uses python-docx's virtual grid and can return cells from
    # OTHER rows to fill empty grid positions — which breaks span detection.
    # Iterating _tr.findall(qn('w:tc')) gives only actual XML cells.
    #
    # vMerge values:
    #   None       – not part of any vertical merge
    #   "restart"  – first (content) row of a vertical span
    #   "continue" – subsequent (covered) rows of a vertical span
    # ------------------------------------------------------------------
    rows = table.rows
    nrows = len(rows)
    row_data: list = []

    for row in rows:
        row_entry: list = []
        col_cursor = 0  # tracks grid-column index as we walk left→right

        for tc in row._tr.findall(qn("w:tc")):
            # Horizontal span
            gs_el = tc.find(f"{qn('w:tcPr')}/{qn('w:gridSpan')}")
            gridspan = 1
            if gs_el is not None:
                try:
                    gridspan = int(gs_el.get(qn("w:val")) or 1)
                except ValueError:
                    pass

            # Vertical span marker from Word XML
            vm_el = tc.find(f"{qn('w:tcPr')}/{qn('w:vMerge')}")
            vmerge = None
            if vm_el is not None:
                vm_val = vm_el.get(qn("w:val"))
                # "restart" = first row of span (has content)
                # absent val or any other val = continuation (no content)
                vmerge = "restart" if vm_val == "restart" else "continue"

            # Paragraph alignment from the first <w:p>
            align = "left"
            p_el = tc.find(qn("w:p"))
            if p_el is not None:
                jc_el = p_el.find(f"{qn('w:pPr')}/{qn('w:jc')}")
                if jc_el is not None:
                    jc_val = jc_el.get(qn("w:val")) or ""
                    _amap = {
                        "right": "right", "center": "center",
                        "both": "justify", "distribute": "justify",
                    }
                    align = _amap.get(jc_val, "left")

            # Cell text — collect text runs from all paragraphs
            text = _tc_text(tc)

            row_entry.append({
                "col_start": col_cursor,
                "col_end": col_cursor + gridspan,
                "text": text,
                "gridSpan": gridspan,
                "vMerge": vmerge,
                "align": align,
            })
            col_cursor += gridspan

        row_data.append(row_entry)

    # ------------------------------------------------------------------
    # 3. Map each cell to its starting grid-column (already in cell_dict)
    #    and pre-compute morerows for "restart" vMerge cells.
    # ------------------------------------------------------------------
    # Build a lookup: (row_idx, col_start) → vMerge type, for morerows counting
    vm_lookup: dict = {}
    for r_idx, row_entry in enumerate(row_data):
        for cd in row_entry:
            vm_lookup[(r_idx, cd["col_start"])] = cd["vMerge"]

    def _morerows(start_r: int, col_start: int) -> int:
        """Count consecutive vMerge-continuation cells below start_r at col_start."""
        count = 0
        for r in range(start_r + 1, nrows):
            if vm_lookup.get((r, col_start)) == "continue":
                count += 1
            else:
                break
        return count

    # cell_grid kept as flat alias for row_data (col_start/col_end already in dicts)
    cell_grid = row_data

    # ------------------------------------------------------------------
    # 4. Assemble CALS XML using ElementTree
    # ------------------------------------------------------------------
    root = ET.Element("table")
    if table_id:
        root.set("id", table_id)
    if title:
        t_el = ET.SubElement(root, "title")
        t_el.text = title

    tgroup = ET.SubElement(root, "tgroup")
    tgroup.set("cols", str(num_cols))

    for i, w in enumerate(col_widths):
        cs = ET.SubElement(tgroup, "colspec")
        cs.set("colnum", str(i + 1))
        cs.set("colname", f"c{i + 1}")
        # Proportional width: widths sum to num_cols * (so columns are comparable)
        cs.set("colwidth", f"{w / total_w * num_cols:.3f}*")

    # First row → <thead> when there are multiple rows
    head_rows = 1 if nrows > 1 else 0
    thead_el = ET.SubElement(tgroup, "thead") if head_rows else None
    tbody_el = ET.SubElement(tgroup, "tbody")

    for r_idx, row_positions in enumerate(cell_grid):
        parent = thead_el if (r_idx < head_rows and thead_el is not None) else tbody_el
        row_el = ET.SubElement(parent, "row")

        for cell_dict in row_positions:
            if cell_dict["vMerge"] == "continue":
                continue  # covered by a row-spanning entry above

            col_start = cell_dict["col_start"]
            col_end = cell_dict["col_end"]

            entry_el = ET.SubElement(row_el, "entry")

            if cell_dict["gridSpan"] > 1:
                entry_el.set("namest", f"c{col_start + 1}")
                entry_el.set("nameend", f"c{col_end}")
            else:
                entry_el.set("colname", f"c{col_start + 1}")

            mr = _morerows(r_idx, col_start) if cell_dict["vMerge"] == "restart" else 0
            if mr:
                entry_el.set("morerows", str(mr))

            if cell_dict["align"] != "left":
                entry_el.set("align", cell_dict["align"])

            entry_el.text = cell_dict["text"] if cell_dict["text"] else None

    _indent_xml(root)
    return ET.tostring(root, encoding="unicode")


def _cals_to_html(cals_xml: str) -> str:
    """Convert a CALS XML string (produced by _build_cals_xml) to a styled HTML table.

    Colour coding matches the verification annotation:
      - green background  : verify="ok"
      - red background    : verify="unconfirmed"
      - default           : no verify attribute (e.g. un-inspected tables)
    """
    import xml.etree.ElementTree as ET

    if not cals_xml:
        return "<p style='color:#888; font-family:sans-serif;'>No CALS XML available</p>"

    try:
        root = ET.fromstring(cals_xml)
    except ET.ParseError as exc:
        return f"<p style='color:red; font-family:sans-serif;'>XML parse error: {exc}</p>"

    title_el = root.find("title")
    title_html = ""
    if title_el is not None and title_el.text:
        title_html = (
            f"<caption style='font-weight:bold;text-align:left;"
            f"padding:4px 0;font-family:sans-serif;'>{title_el.text}</caption>"
        )

    TABLE_STYLE = (
        "border-collapse:collapse;width:100%;font-family:monospace;"
        "font-size:11px;margin-top:4px;color:#111;background:#fff;"
    )
    TH_STYLE = "border:1px solid #888;padding:4px 8px;background:#d8e8f8;color:#111;font-weight:bold;"
    TD_STYLE = "border:1px solid #ccc;padding:4px 8px;color:#111;background:#fff;"
    OK_STYLE  = "border:1px solid #5a5;padding:4px 8px;background:#d4f8d4;color:#111;"
    BAD_STYLE = "border:1px solid #a55;padding:4px 8px;background:#f8d4d4;color:#111;"

    def _cell_markup(entry_el, is_head=False):
        tag = "th" if is_head else "td"

        # colspan from namest/nameend (column names are "c1", "c2", ...)
        namest  = entry_el.get("namest")
        nameend = entry_el.get("nameend")
        colspan = 1
        if namest and nameend:
            try:
                colspan = int(nameend.lstrip("c")) - int(namest.lstrip("c")) + 1
            except (ValueError, AttributeError):
                pass

        # rowspan from morerows attribute
        morerows = entry_el.get("morerows")
        rowspan = 1
        if morerows:
            try:
                rowspan = int(morerows) + 1
            except ValueError:
                pass

        verify = entry_el.get("verify")
        if is_head:
            style = TH_STYLE
        elif verify == "ok":
            style = OK_STYLE
        elif verify == "unconfirmed":
            style = BAD_STYLE
        else:
            style = TD_STYLE

        text = (entry_el.text or "").replace("<", "&lt;").replace(">", "&gt;")

        attrs = f'style="{style}"'
        if colspan > 1:
            attrs += f' colspan="{colspan}"'
        if rowspan > 1:
            attrs += f' rowspan="{rowspan}"'
        return f"<{tag} {attrs}>{text}</{tag}>"

    parts = [f'<div style="overflow-x:auto;color:#111;background:#fff;padding:4px;"><table style="{TABLE_STYLE}">{title_html}']

    for tgroup in root.findall("tgroup"):
        thead = tgroup.find("thead")
        if thead is not None:
            parts.append("<thead>")
            for row in thead.findall("row"):
                parts.append("<tr>")
                for entry in row.findall("entry"):
                    parts.append(_cell_markup(entry, is_head=True))
                parts.append("</tr>")
            parts.append("</thead>")

        tbody = tgroup.find("tbody")
        if tbody is not None:
            parts.append("<tbody>")
            for row in tbody.findall("row"):
                parts.append("<tr>")
                for entry in row.findall("entry"):
                    parts.append(_cell_markup(entry, is_head=False))
                parts.append("</tr>")
            parts.append("</tbody>")

    parts.append("</table></div>")
    return "\n".join(parts)


def _cals_to_interactive_html(cals_xml: str):
    """Return (table_html, xml_panel_html) for the interactive Table Browser.

    table_html     — rendered HTML table; each cell has an onclick handler that
                     scrolls to and highlights the matching <entry> span inside
                     xml_panel_html.
    xml_panel_html — dark-themed <pre> block with the pretty-printed CALS XML;
                     each <entry> is wrapped in <span id="entry-rN-cK"> so the
                     onclick handler can locate it via document.getElementById.

    Because Gradio injects gr.HTML content via innerHTML (which does not execute
    <script> tags), all JavaScript is written as inline onclick= attributes.
    """
    import xml.etree.ElementTree as ET
    import html as _html
    import re

    if not cals_xml:
        empty = "<p style='color:#888;font-family:sans-serif;'>No CALS XML available</p>"
        return empty, empty

    try:
        root = ET.fromstring(cals_xml)
    except ET.ParseError as exc:
        err = f"<p style='color:red;font-family:sans-serif;'>XML parse error: {exc}</p>"
        return err, err

    # CSS injected into both components so the highlight class is always defined.
    HI_CSS = (
        "<style>.cals-hi{background:#ffe080!important;color:#111!important;"
        "outline:2px solid #d4a000;border-radius:2px;}</style>"
    )

    # Inline onclick JS — no named function needed (innerHTML doesn't run <script>).
    # Single quotes inside JS are safe inside a double-quoted HTML attribute.
    def _onclick(eid):
        return (
            "var _a=document.querySelectorAll('.cals-hi');"
            "for(var _i=0;_i<_a.length;_i++)_a[_i].classList.remove('cals-hi');"
            f"var _el=document.getElementById('{eid}');"
            "if(_el){_el.classList.add('cals-hi');"
            "_el.scrollIntoView({behavior:'smooth',block:'center'});}"
        )

    # Assign stable DOM IDs:  entry-r{row_idx}-{col_key}
    def _assign_ids(xml_root):
        eids = {}
        for tg in xml_root.findall("tgroup"):
            row_idx = 0
            for section in (tg.find("thead"), tg.find("tbody")):
                if section is None:
                    continue
                for row in section.findall("row"):
                    for entry in row.findall("entry"):
                        col_key = (
                            entry.get("colname")
                            or entry.get("namest")
                            or f"x{row_idx}"
                        )
                        eids[id(entry)] = f"entry-r{row_idx}-{col_key}"
                    row_idx += 1
        return eids

    entry_ids = _assign_ids(root)

    # ── Rendered table HTML (onclick on every cell) ────────────────────────
    TABLE_STYLE = (
        "border-collapse:collapse;width:100%;font-family:monospace;"
        "font-size:11px;margin-top:4px;color:#111;background:#fff;"
    )
    TH  = "border:1px solid #888;padding:4px 8px;background:#d8e8f8;color:#111;font-weight:bold;cursor:pointer;"
    TD  = "border:1px solid #ccc;padding:4px 8px;color:#111;background:#fff;cursor:pointer;"
    OK  = "border:1px solid #5a5;padding:4px 8px;background:#d4f8d4;color:#111;cursor:pointer;"
    BAD = "border:1px solid #a55;padding:4px 8px;background:#f8d4d4;color:#111;cursor:pointer;"

    title_el = root.find("title")
    caption  = ""
    if title_el is not None and title_el.text:
        caption = (
            "<caption style='font-weight:bold;text-align:left;padding:4px 0;"
            f"font-family:sans-serif;color:#111;'>{_html.escape(title_el.text)}</caption>"
        )

    def _norm_display(v):
        """Return (raw, normalized) strings for mismatch explanation."""
        raw = str(v or "").strip()
        n = re.sub(r"[\$,\(\)\s\u2014\u2013]", "", raw).replace("-", "").strip()
        if not n:
            return raw, "(empty after normalisation)"
        try:
            return raw, str(float(n))
        except ValueError:
            return raw, n.lower()

    def _cell(entry_el, is_head=False):
        tag  = "th" if is_head else "td"
        eid  = entry_ids.get(id(entry_el), "")
        namest, nameend = entry_el.get("namest"), entry_el.get("nameend")
        colspan = 1
        if namest and nameend:
            try:
                colspan = int(nameend.lstrip("c")) - int(namest.lstrip("c")) + 1
            except (ValueError, AttributeError):
                pass
        morerows = entry_el.get("morerows")
        rowspan  = (int(morerows) + 1) if morerows else 1
        verify   = entry_el.get("verify")
        style    = TH if is_head else (OK if verify == "ok" else (BAD if verify == "unconfirmed" else TD))
        text     = _html.escape(entry_el.text or "")

        # Build onclick: always highlight XML panel entry; for mismatched cells
        # also populate the info banner via data-mismatch attribute.
        base_oc = _onclick(eid) if eid else ""
        if not is_head and verify == "unconfirmed":
            raw_txt, norm_txt = _norm_display(entry_el.text or "")
            # &#10; encodes newline inside an HTML attribute; textContent renders it
            # as a real newline with white-space:pre-wrap on the container.
            detail = (
                f"CALS cell text : {_html.escape(raw_txt)}&#10;"
                f"Normalised to  : {_html.escape(norm_txt)}&#10;"
                f"&#10;"
                f"Not found in pdfplumber cross-check.&#10;"
                f"Common causes  : leading/trailing spaces, line-breaks inside&#10;"
                f"                 a cell, merged-cell text split across rows,&#10;"
                f"                 or special characters dropped by pdfplumber."
            )
            show_oc = (
                "var _b=document.getElementById('cals-mismatch-info');"
                "_b.style.display='block';"
                "document.getElementById('cals-mismatch-body').textContent="
                "this.getAttribute('data-mismatch');"
            )
            extra = (
                f' onclick="{base_oc}{show_oc}"'
                f' data-mismatch="{detail}"'
                f' title="Mismatch — click for details"'
            )
        elif not is_head and verify == "ok":
            hide_oc = (
                "var _b=document.getElementById('cals-mismatch-info');"
                "_b.style.display='none';"
            )
            extra = f' onclick="{base_oc}{hide_oc}" title="Confirmed by pdfplumber"'
        else:
            extra = f' onclick="{base_oc}" title="Click to locate in XML"' if base_oc else ""

        if colspan > 1:
            extra += f' colspan="{colspan}"'
        if rowspan > 1:
            extra += f' rowspan="{rowspan}"'
        return f'<{tag} style="{style}"{extra}>{text}</{tag}>'

    # Info banner — shown/hidden by the onclick handlers above.
    INFO_BOX = (
        '<div id="cals-mismatch-info" style="display:none;background:#fff3cd;'
        'border:1px solid #c9a700;padding:8px 14px;margin-bottom:10px;'
        'border-radius:4px;font-family:monospace;font-size:11px;color:#111;'
        'white-space:pre-wrap;">'
        '<b style="color:#7a5200;">&#9888; Mismatch detail</b>'
        '<span style="float:right;cursor:pointer;font-size:14px;color:#7a5200;" '
        'onclick="this.parentElement.style.display=\'none\'">&#x2715;</span>'
        '<div id="cals-mismatch-body" style="margin-top:6px;"></div>'
        '</div>'
    )

    tbl = [
        HI_CSS,
        '<div style="overflow-x:auto;color:#111;background:#fff;padding:4px;">',
        INFO_BOX,
        f'<table style="{TABLE_STYLE}">{caption}',
    ]
    for tg in root.findall("tgroup"):
        thead = tg.find("thead")
        if thead is not None:
            tbl.append("<thead>")
            for row in thead.findall("row"):
                tbl.append("<tr>" + "".join(_cell(e, True)  for e in row.findall("entry")) + "</tr>")
            tbl.append("</thead>")
        tbody = tg.find("tbody")
        if tbody is not None:
            tbl.append("<tbody>")
            for row in tbody.findall("row"):
                tbl.append("<tr>" + "".join(_cell(e, False) for e in row.findall("entry")) + "</tr>")
            tbl.append("</tbody>")
    tbl.append("</table></div>")
    table_html = "\n".join(tbl)

    # ── XML panel HTML (dark code view with span IDs per entry) ───────────
    # Clone the tree, inject _eid attributes, pretty-print, then build HTML
    # with each <entry> line wrapped in a <span id="..."> for the onclick hook.
    root2 = ET.fromstring(cals_xml)
    eids2 = _assign_ids(root2)
    for tg in root2.findall("tgroup"):
        for section in (tg.find("thead"), tg.find("tbody")):
            if section is None:
                continue
            for row in section.findall("row"):
                for entry in row.findall("entry"):
                    eid = eids2.get(id(entry), "")
                    if eid:
                        entry.set("_eid", eid)
    _indent_xml(root2)
    xml_raw = ET.tostring(root2, encoding="unicode")

    xml_lines = []
    in_multi  = False
    for line in xml_raw.splitlines():
        m = re.search(r'_eid="([^"]+)"', line)
        if m:
            eid   = m.group(1)
            clean = re.sub(r'\s*_eid="[^"]*"', "", line)
            esc   = _html.escape(clean)
            if "</entry>" in line:
                xml_lines.append(
                    f'<span id="{eid}" class="cals-entry" style="display:block;">{esc}</span>'
                )
            else:
                xml_lines.append(
                    f'<span id="{eid}" class="cals-entry" style="display:block;">{esc}'
                )
                in_multi = True
        elif in_multi and "</entry>" in line:
            xml_lines.append(f'{_html.escape(line)}</span>')
            in_multi = False
        else:
            xml_lines.append(_html.escape(line))

    PRE = (
        "font-family:'Courier New',monospace;font-size:11px;line-height:1.6;"
        "background:#1e1e1e;color:#d4d4d4;padding:12px;margin:0;"
        "white-space:pre;overflow-x:auto;"
    )
    xml_panel_html = (
        HI_CSS
        + '<div style="min-height:1440px;overflow-y:auto;background:#1e1e1e;border-radius:4px;">'
        + f'<pre style="{PRE}">'
        + "\n".join(xml_lines)
        + "</pre></div>"
    )

    return table_html, xml_panel_html


def _render_docx_pages_to_images(docx_path: str, img_dir: str, dpi: int = 150) -> list:
    """Level 2: Convert a .docx to per-page PNG images via LibreOffice + pdf2image.

    Returns a list of absolute image paths, one per page (0-indexed).
    """
    import subprocess
    from pdf2image import convert_from_path

    os.makedirs(img_dir, exist_ok=True)

    # Convert docx → PDF using LibreOffice headless
    pdf_path = os.path.join(img_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", img_dir, docx_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"[render_pages] LibreOffice failed: {result.stderr}")
            return [], pdf_path
    except Exception as e:
        print(f"[render_pages] LibreOffice error: {e}")
        return [], pdf_path

    if not os.path.exists(pdf_path):
        print(f"[render_pages] PDF not found after conversion: {pdf_path}")
        return [], pdf_path

    # Convert PDF pages → PNG images
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"[render_pages] pdf2image error: {e}")
        return [], pdf_path

    page_paths = []
    for i, page in enumerate(pages):
        page_path = os.path.join(img_dir, f"page_{i:03d}.png")
        page.save(page_path, "PNG")
        page_paths.append(page_path)

    print(f"[render_pages] Rendered {len(page_paths)} pages to {img_dir}")
    return page_paths, pdf_path


def _load_docx_direct(fpath: str):
    """Load a .docx file using python-docx directly, bypassing unstructured.

    Level 1: Uses tabulate to render each table as an aligned text grid.
    Level 2: Renders the full document to per-page PNGs via LibreOffice + pdf2image.
             Each table element gets metadata['image_path'] pointing to the page
             image it appears on (tracked by table order vs page count).

    Iterates the document body in order so that the paragraph immediately
    preceding each table (its title/caption) is attached to that table element.
    """
    import docx
    from langchain.schema import Document
    from tabulate import tabulate

    doc = docx.Document(fpath)
    elements = []
    table_index = 0

    para_map = {p._element: p for p in doc.paragraphs}
    table_map = {t._element: t for t in doc.tables}
    pending_para = ""

    img_dir = os.path.join(os.path.dirname(fpath), "table_images")

    # Level 2: convert to PDF and render all pages
    page_images, pdf_path = _render_docx_pages_to_images(fpath, img_dir)
    total_pages = len(page_images)

    # Extract per-page text from the PDF to detect which pages contain tables
    page_texts = []
    if pdf_path and os.path.exists(pdf_path):
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            page_texts = [p.extract_text() or "" for p in reader.pages]
            print(f"[_load_docx_direct] Extracted text from {len(page_texts)} PDF pages")
        except Exception as exc:
            print(f"[_load_docx_direct] pypdf error: {exc}")

    # Normalise page texts once for case-insensitive matching
    page_texts_norm = [pt.lower() for pt in page_texts]

    # Pre-scan: collect (cell_rows, title) per table so the title paragraph can
    # join the fingerprint — it is usually the most distinctive text on that page.
    table_cells_by_idx: dict = {}  # tidx -> (rows, title_str)
    _scan_idx = 0
    _pre_title = ""
    for _child in doc.element.body.iterchildren():
        _tag = _child.tag.split("}")[-1] if "}" in _child.tag else _child.tag
        if _tag == "p":
            _para = para_map.get(_child)
            _txt = _para.text.strip() if _para else ""
            if _txt:
                _pre_title = _txt
        elif _tag == "tbl":
            _tbl = table_map.get(_child)
            if _tbl is not None:
                _rows = []
                for _row in _tbl.rows:
                    _seen_tc: set = set()
                    _rcells = []
                    for _c in _row.cells:
                        if id(_c._tc) not in _seen_tc:
                            _seen_tc.add(id(_c._tc))
                            _rcells.append(_c.text.strip())
                    _rows.append(_rcells)
                table_cells_by_idx[_scan_idx] = ([r for r in _rows if any(r)], _pre_title)
                _scan_idx += 1
            _pre_title = ""

    def _build_tokens(cell_rows, title):
        """Collect lowercase fingerprint tokens from title + all cell values."""
        tokens = []
        # Title words are the most distinctive — add each word ≥4 chars individually
        for word in title.split():
            w = word.strip("(),.:;-").lower()
            if len(w) >= 4:
                tokens.append(w)
        # Whole-cell values from every row (not just the first few)
        for row in cell_rows:
            for cell in row:
                cell_norm = cell.strip().lower()
                if len(cell_norm) >= 4:
                    tokens.append(cell_norm)
                # Also split multi-word cells so e.g. "Total Assets" → "total", "assets"
                for word in cell_norm.split():
                    w = word.strip("(),.:;-")
                    if len(w) >= 4 and w != cell_norm:
                        tokens.append(w)
        return tokens

    def _best_page_for_table(cell_rows, title=""):
        """Return 0-based page index scored by length-weighted token hits, or None."""
        if not page_texts_norm:
            return None
        tokens = _build_tokens(cell_rows, title)
        if not tokens:
            return None
        # Weight each hit by token length — longer tokens are more distinctive
        scores = [
            sum(len(t) for t in tokens if t in pt)
            for pt in page_texts_norm
        ]
        best = max(range(len(scores)), key=lambda i: scores[i])
        return best if scores[best] > 0 else None

    # Build table_index → image_path mapping (None for tables with no page match)
    table_to_page_img: dict = {}
    for tidx, (rows, title) in table_cells_by_idx.items():
        pg = _best_page_for_table(rows, title)
        if pg is not None and pg < total_pages:
            table_to_page_img[tidx] = page_images[pg]
            print(f"[_load_docx_direct] table {tidx} '{title[:40]}' → page {pg}")
        elif page_images and total_pages > 0:
            # Fallback to proportional distribution if pypdf gave no match
            fallback = min(int(tidx * total_pages / max(len(table_cells_by_idx), 1)), total_pages - 1)
            table_to_page_img[tidx] = page_images[fallback]
            print(f"[_load_docx_direct] table {tidx} '{title[:40]}' → fallback page {fallback}")

    table_pages_matched = sum(1 for v in table_to_page_img.values() if v is not None)
    text_only_pages = total_pages - len({v for v in table_to_page_img.values() if v})
    print(f"[_load_docx_direct] {table_pages_matched}/{len(table_cells_by_idx)} tables matched to pages; "
          f"{text_only_pages} text-only pages (no image stored)")

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "p":
            para = para_map.get(child)
            text = para.text.strip() if para else ""
            if text:
                pending_para = text
                elements.append(Document(
                    page_content=text,
                    metadata={"source": fpath, "type": "paragraph"}
                ))

        elif tag == "tbl":
            table = table_map.get(child)
            if table is None:
                continue

            # Extract cell data, de-duplicating only truly merged cells.
            # Use id(cell._tc) (XML element identity) so two distinct cells with
            # the same text (e.g. two "0" columns) are both kept, while a merged
            # cell that python-docx returns multiple times for each column it spans
            # is included only once.
            def _cell_text(cell):
                """Return full text of a cell, including any nested table cells."""
                parts = [p.text for p in cell.paragraphs]
                for nested in cell.tables:
                    for nrow in nested.rows:
                        seen_ntc: set = set()
                        for ncell in nrow.cells:
                            if id(ncell._tc) not in seen_ntc:
                                seen_ntc.add(id(ncell._tc))
                                t = ncell.text.strip()
                                if t:
                                    parts.append(t)
                return " ".join(p.strip() for p in parts if p.strip())

            cell_rows = []
            for row in table.rows:
                seen_tc: set = set()
                row_cells = []
                for cell in row.cells:
                    tc_id = id(cell._tc)
                    if tc_id not in seen_tc:
                        seen_tc.add(tc_id)
                        row_cells.append(_cell_text(cell))
                cell_rows.append(row_cells)

            cell_rows = [r for r in cell_rows if any(c for c in r)]
            print(f"[table {table_index}] '{pending_para[:40]}' — {len(cell_rows)} rows, "
                  f"sample: {cell_rows[0] if cell_rows else []}")
            if not cell_rows:
                pending_para = ""
                continue

            # Level 1: tabulate text rendering — use "simple" (not "grid") to keep
            # char count low enough that large tables (e.g. full balance sheets) fit
            # within the LLM context budget without truncation.
            table_text = tabulate(cell_rows, tablefmt="simple")
            title_line = f"[Title: {pending_para}]\n" if pending_para else ""

            # Level 2: use fingerprint-matched page image
            img_path = table_to_page_img.get(table_index)

            # Build CALS XML first — it is the reference for the inspection step below
            try:
                cals_xml = _build_cals_xml(table, title=pending_para, table_id=f"t{table_index}")
            except Exception as _xe:
                print(f"[_build_cals_xml] table {table_index}: {_xe}")
                cals_xml = None

            # Phase 1 inspection: compare CALS XML against pdfplumber PDF cross-check
            page_idx_for_table = next(
                (pi for pi, pg_img in enumerate(page_images) if pg_img == img_path), None
            )
            inspection = None
            if pdf_path and os.path.exists(pdf_path) and page_idx_for_table is not None and cals_xml:
                inspection = verify_table(
                    cals_xml, pdf_path, page_idx_for_table, pending_para,
                    img_path=img_path or ""
                )

            import json as _json
            elements.append(Document(
                page_content=f"{title_line}{table_text}",
                metadata={
                    "source": fpath,
                    "type": "table",
                    "table_index": table_index,
                    "title": pending_para,
                    "image_path": img_path or "",
                    "pdf_path": pdf_path or "",
                    "page_idx": page_idx_for_table if page_idx_for_table is not None else -1,
                    "cell_rows": _json.dumps(cell_rows),
                    "inspection": _json.dumps(inspection) if inspection is not None else "",
                    "xml": cals_xml or "",
                }
            ))
            table_index += 1
            pending_para = ""

    print(f"[_load_docx_direct] Loaded {table_index} tables, {len(page_images)} page images")
    return elements


def _annotate_cals_xml(cals_xml: str, unconfirmed_values: set) -> str:
    """Return a copy of the CALS XML with verify="unconfirmed" on entries
    whose text is not confirmed by the pdfplumber cross-check.
    Entries that are confirmed get verify="ok".
    """
    import re
    import xml.etree.ElementTree as ET

    def _norm(v):
        v = re.sub(r"[\$,\(\)\s\u2014\u2013]", "", str(v or ""))
        v = v.replace("-", "").strip()
        if not v:
            return None
        try:
            return str(float(v))
        except ValueError:
            return v.lower()

    if not cals_xml:
        return cals_xml
    try:
        root = ET.fromstring(cals_xml)
    except ET.ParseError:
        return cals_xml

    for entry in root.iter("entry"):
        text = (entry.text or "").strip()
        if not text:
            continue
        norm = _norm(text)
        if norm in unconfirmed_values:
            entry.set("verify", "unconfirmed")
        else:
            entry.set("verify", "ok")

    _indent_xml(root)
    return ET.tostring(root, encoding="unicode")


def _annotate_page_image(img_path: str, pdf_path: str, page_idx: int,
                         unconfirmed_values: set, confirmed_values: set = None,
                         dpi: int = 150) -> str:
    """Draw colour-coded boxes on every table cell in the page image:
      - GREEN  (semi-transparent) : cell value confirmed by pdfplumber cross-check
      - RED    (semi-transparent) : cell value in CALS XML but NOT confirmed by pdfplumber
      - YELLOW (semi-transparent) : cell has no text / empty

    Uses pdfplumber's per-cell bboxes for precise placement.
    Saves the annotated image as <original>_diff.png and returns its path.
    Falls back to returning img_path unchanged on any error.
    """
    import re
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pdfplumber
    except ImportError as exc:
        print(f"[annotate_page_image] missing dependency: {exc}")
        return img_path

    def _norm(v):
        v = re.sub(r"[\$,\(\)\s\u2014\u2013]", "", str(v or ""))
        v = v.replace("-", "").strip()
        if not v:
            return None
        try:
            return str(float(v))
        except ValueError:
            return v.lower()

    if not img_path or not os.path.exists(img_path):
        return img_path
    if not os.path.exists(pdf_path):
        return img_path

    confirmed_values = confirmed_values or set()

    # Colour palette  (R, G, B, A)
    COLOR_OK       = (40,  180,  60, 70)    # green fill
    COLOR_OK_BORD  = (20,  140,  30, 210)   # green border
    COLOR_BAD      = (220,  50,  50, 90)    # red fill
    COLOR_BAD_BORD = (200,   0,   0, 220)   # red border
    COLOR_EMPTY    = (200, 180,   0, 50)    # yellow fill (no text)
    COLOR_EMPTY_B  = (160, 140,   0, 160)   # yellow border

    try:
        img = Image.open(img_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        scale = dpi / 72.0

        # Try to load a small font for labels; fall back to default if unavailable
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        cells_drawn = 0
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return img_path
            page = pdf.pages[page_idx]
            # Lattice (border-based) first; fall back to stream (text-alignment) for
            # borderless tables produced by LibreOffice from Word .docx files.
            tables = page.find_tables()
            if not tables:
                tables = page.find_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                })
            for tbl in tables:
                extracted = tbl.extract()   # [[text, ...], ...]
                for row_cells, row_texts in zip(tbl.rows, extracted or []):
                    for cell_bbox, cell_text in zip(row_cells.cells, row_texts or []):
                        if cell_bbox is None:
                            continue
                        x0, top, x1, bottom = cell_bbox
                        px0 = int(x0     * scale)
                        py0 = int(top    * scale)
                        px1 = int(x1     * scale)
                        py1 = int(bottom * scale)

                        norm = _norm(cell_text or "")
                        if not norm:
                            fill, border, label = COLOR_EMPTY, COLOR_EMPTY_B, ""
                        elif norm in unconfirmed_values:
                            fill, border, label = COLOR_BAD, COLOR_BAD_BORD, "\u2717"
                        else:
                            # confirmed (either in confirmed_values set or simply not flagged)
                            fill, border, label = COLOR_OK, COLOR_OK_BORD, "\u2713"

                        draw.rectangle([px0, py0, px1, py1],
                                       fill=fill, outline=border, width=2)
                        # Small label in top-left corner of cell when there's room
                        if label and (px1 - px0) > 14 and (py1 - py0) > 10:
                            draw.text((px0 + 3, py0 + 2), label, fill=border, font=font)
                        cells_drawn += 1

        annotated = Image.alpha_composite(img, overlay).convert("RGB")
        out_path = img_path.replace(".png", "_diff.png")
        annotated.save(out_path, "PNG")
        print(f"[annotate_page_image] {cells_drawn} cells annotated \u2192 {out_path}")
        return out_path
    except Exception as exc:
        print(f"[annotate_page_image] {exc}")
        return img_path


def verify_table(cals_xml: str, pdf_path: str, page_idx: int,
                 table_title: str = "", img_path: str = "") -> dict:
    """Inspect extraction quality by comparing CALS XML entries (canonical Word source)
    against pdfplumber's independent PDF extraction.

    Using CALS XML as the reference (rather than raw cell_rows) gives a span-aware
    comparison: merged cells appear exactly once, column widths are preserved, and
    the extracted value set is not polluted by python-docx's virtual grid fill.

    Returns a dict with: table, cals_shape, pdf_shape, in_cals_not_pdf,
    in_pdf_not_cals, coverage_pct, verdict.
    """
    import re
    import xml.etree.ElementTree as ET

    try:
        import pdfplumber
    except ImportError:
        return {"table": table_title, "verdict": "SKIP", "reason": "pdfplumber not installed"}

    def _normalize(v):
        v = re.sub(r"[\$,\(\)\s\u2014\u2013]", "", str(v or ""))
        v = v.replace("-", "").strip()
        if not v:
            return None
        try:
            return str(float(v))   # normalise numeric strings: '1,234' → float → str
        except ValueError:
            return v.lower()

    # ------------------------------------------------------------------
    # 1. Extract value set from CALS XML (span-aware canonical source)
    # ------------------------------------------------------------------
    if not cals_xml:
        return {"table": table_title, "verdict": "SKIP", "reason": "no CALS XML available"}
    try:
        root = ET.fromstring(cals_xml)
    except ET.ParseError as exc:
        return {"table": table_title, "verdict": "SKIP", "reason": f"CALS parse error: {exc}"}

    cals_entries = []
    for entry in root.iter("entry"):
        text = (entry.text or "").strip()
        if text:
            cals_entries.append(text)

    tgroup = root.find("tgroup")
    cals_cols = int(tgroup.get("cols", 0)) if tgroup is not None else 0
    cals_rows = sum(1 for _ in root.iter("row"))
    cals_values = {_normalize(v) for v in cals_entries if _normalize(v) is not None}

    # ------------------------------------------------------------------
    # 2. Extract value set from pdfplumber (independent PDF render check)
    # ------------------------------------------------------------------
    if not os.path.exists(pdf_path):
        return {"table": table_title, "verdict": "SKIP", "reason": f"PDF not found: {pdf_path}"}

    pdf_ref = None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return {"table": table_title, "verdict": "SKIP",
                        "reason": f"page_idx {page_idx} out of range ({len(pdf.pages)} pages)"}
            page = pdf.pages[page_idx]
            pdf_ref = page.extract_table()
            if pdf_ref is None:
                pdf_ref = page.extract_table(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                })
    except Exception as exc:
        return {"table": table_title, "verdict": "SKIP", "reason": str(exc)}

    if pdf_ref is None:
        return {"table": table_title, "verdict": "SKIP",
                "reason": "pdfplumber found no table on this page"}

    pdf_rows_count = len(pdf_ref)
    pdf_cols_count = max((len(r or []) for r in pdf_ref), default=0)
    pdf_values = {_normalize(c) for row in pdf_ref
                  for c in (row or []) if _normalize(c) is not None}

    # ------------------------------------------------------------------
    # 3. Compare: CALS is the reference, PDF is the cross-check
    # ------------------------------------------------------------------
    in_cals_not_pdf = sorted(str(v) for v in cals_values - pdf_values)
    in_pdf_not_cals = sorted(str(v) for v in pdf_values - cals_values)

    # Coverage: what % of CALS values are confirmed by pdfplumber
    confirmed = len(cals_values & pdf_values)
    coverage = confirmed / max(len(cals_values), 1)

    verdict = "PASS" if coverage >= 0.85 and not in_cals_not_pdf else "WARN" if coverage >= 0.6 else "FAIL"

    unconfirmed_set = set(in_cals_not_pdf)  # already normalized strings
    confirmed_set   = set(str(v) for v in cals_values & pdf_values)

    # ------------------------------------------------------------------
    # 4. Build annotations — CALS XML with verify attributes, page image with colour boxes
    # ------------------------------------------------------------------
    annotated_xml = None
    annotated_image_path = None
    if in_cals_not_pdf:
        try:
            annotated_xml = _annotate_cals_xml(cals_xml, unconfirmed_set)
        except Exception as _ae:
            print(f"[verify_table] annotate_cals_xml failed: {_ae}")
    try:
        annotated_image_path = _annotate_page_image(
            img_path, pdf_path, page_idx, unconfirmed_set, confirmed_set
        )
    except Exception as _ae:
        print(f"[verify_table] annotate_page_image failed: {_ae}")

    report = {
        "table": table_title,
        "cals_shape": {"rows": cals_rows, "cols": cals_cols, "unique_values": len(cals_values)},
        "pdf_shape":  {"rows": pdf_rows_count, "cols": pdf_cols_count, "unique_values": len(pdf_values)},
        "confirmed_by_pdf": confirmed,
        "coverage_pct": f"{coverage:.0%}",
        "in_cals_not_pdf": in_cals_not_pdf[:20],
        "in_pdf_not_cals": in_pdf_not_cals[:20],
        "verdict": verdict,
        "annotated_xml": annotated_xml,
        "annotated_image_path": annotated_image_path,
    }
    print(f"[verify_table] '{table_title[:40]}' → {verdict} | "
          f"coverage={coverage:.0%} in_cals_not_pdf={len(in_cals_not_pdf)} "
          f"in_pdf_not_cals={len(in_pdf_not_cals)}")
    return report

def _extract_via_pdfplumber_lattice(pdf_path: str, page_idx: int):
    """Extract table rows using pdfplumber lattice (border-based) mode."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return None
            tbl = pdf.pages[page_idx].extract_table()
            return [[str(c or "").strip() for c in row] for row in tbl] if tbl else None
    except Exception as exc:
        print(f"[pdfplumber-lattice] {exc}")
        return None


def _extract_via_pdfplumber_stream(pdf_path: str, page_idx: int):
    """Extract table rows using pdfplumber stream (whitespace) mode."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return None
            tbl = pdf.pages[page_idx].extract_table(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "intersection_x_tolerance": 15,
                "intersection_y_tolerance": 15,
            })
            return [[str(c or "").strip() for c in row] for row in tbl] if tbl else None
    except Exception as exc:
        print(f"[pdfplumber-stream] {exc}")
        return None


def _extract_via_pypdf_layout(pdf_path: str, page_idx: int):
    """Extract table rows using pypdf layout-preserving text extraction."""
    try:
        import re
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        if page_idx >= len(reader.pages):
            return None
        text = reader.pages[page_idx].extract_text(extraction_mode="layout")
        if not text:
            return None
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            cells = [c.strip() for c in re.split(r'  +', line) if c.strip()]
            if cells:
                rows.append(cells)
        return rows if rows else None
    except Exception as exc:
        print(f"[pypdf-layout] {exc}")
        return None


def optimize_extraction_agent(title: str, original_cell_rows: list,
                              pdf_path: str, page_idx: int) -> dict:
    """Iteratively tries 4 extraction strategies for a single table.

    Scoring: builds the union of all values found across all strategies, then
    scores each strategy by what percentage of that union it covers.  Avoids
    unreliable PASS/FAIL verdicts from pdfplumber (which truncates cell text).

    Returns a dict with 'title', 'recommended_strategy', and 'rounds'.
    """
    import re

    def _norm(v):
        return re.sub(r'[\s$,.()\-\u2014]', '', str(v or '')).lower()

    def _vset(rows):
        return {_norm(c) for row in rows for c in row if len(_norm(c)) >= 3}

    strategies = [("python-docx (current)", original_cell_rows)]
    if os.path.exists(pdf_path):
        for name, rows in [
            ("pdfplumber-lattice", _extract_via_pdfplumber_lattice(pdf_path, page_idx)),
            ("pdfplumber-stream",  _extract_via_pdfplumber_stream(pdf_path, page_idx)),
            ("pypdf-layout",       _extract_via_pypdf_layout(pdf_path, page_idx)),
        ]:
            if rows:
                strategies.append((name, rows))

    # Build union of all distinct values found across every strategy
    strategy_vsets = {name: _vset(rows) for name, rows in strategies}
    union = set().union(*strategy_vsets.values())

    rounds = []
    for name, rows in strategies:
        vs = strategy_vsets[name]
        coverage = len(vs & union) / max(len(union), 1)
        missing_from_this = sorted(union - vs)[:10]
        rounds.append({
            "strategy": name,
            "shape": [len(rows), max((len(r) for r in rows), default=0)],
            "unique_values_found": len(vs),
            "union_coverage": f"{coverage:.0%}",
            "sample_values_missing_from_this_strategy": missing_from_this,
        })
        print(f"[optimize] '{name}': {len(vs)} values, {coverage:.0%} coverage")

    best_idx = max(range(len(rounds)), key=lambda i: rounds[i]["unique_values_found"])
    rounds[best_idx]["recommended"] = True

    return {
        "title": title,
        "total_unique_values_across_strategies": len(union),
        "recommended_strategy": rounds[best_idx]["strategy"],
        "rounds": rounds,
    }

def load_documents_from_files(file_paths: List[str]) -> List[Any]:
    """Load and return documents from supported file types."""
    docs = []

    for fpath in file_paths:
        ext = os.path.splitext(fpath)[-1].lower()

        try:
            if ext == ".pdf":
                loader = UnstructuredPDFLoader(fpath, mode="elements")
                loaded = loader.load()
            elif ext in [".docx", ".doc"]:
                # Use python-docx directly — unstructured is incompatible with this docx version
                loaded = _load_docx_direct(fpath)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(fpath)
                loaded = loader.load()
            elif ext == ".csv":
                loader = CSVLoader(fpath)
                loaded = loader.load()
            else:
                print(f"[load_documents] Skipping unsupported file type: {fpath}")
                continue

            docs.append(loaded)
            print(f"[load_documents] Successfully loaded {fpath} ({len(loaded)} elements)")
        except Exception as e:
            print(f"[load_documents] Failed to load {fpath}: {e}")

    return [item for sublist in docs for item in sublist]


def split_documents(docs: List[Any]):
    """Split documents into smaller chunks using recursive splitter."""
    print(f"[split_documents] Splitting {len(docs)} docs with chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def embed_documents(doc_splits: List[Any]):
    """Embed and store the split documents into Chroma vectorstore."""
    try:
        print(f"[embed_documents] Embedding {len(doc_splits)} chunks using model: {EMBEDDINGS_MODEL}")

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
            persist_directory=DATA_DIR,
        )
        return vectorstore

    except Exception as e:
        print(f"[embed_documents] Vectorstore creation failed: {e}")
        return None


## Main function that use helper functions 

# Sidecar file that records the paths of the most recently uploaded files
# so the direct_generate graph node can read them without going through the vectorstore.
_UPLOADED_FILES_RECORD = os.path.join(DATA_DIR, "_uploaded_files.json")

def record_uploaded_files(file_paths: List[str]) -> None:
    """Persist the list of uploaded file paths to disk."""
    import json
    with open(_UPLOADED_FILES_RECORD, "w") as f:
        json.dump(file_paths, f)

def get_uploaded_files() -> List[str]:
    """Return the list of previously uploaded file paths.
    Falls back to scanning DATA_DIR for document files if the record is missing.
    """
    import json
    if os.path.exists(_UPLOADED_FILES_RECORD):
        try:
            with open(_UPLOADED_FILES_RECORD) as f:
                paths = json.load(f)
            # Only return paths that still exist on disk
            return [p for p in paths if os.path.exists(p)]
        except Exception:
            pass
    # Fallback: scan DATA_DIR for any supported document files
    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}
    found = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in supported
    ]
    if found:
        print(f"[get_uploaded_files] No record file found, discovered from data dir: {found}")
        record_uploaded_files(found)  # write the record for next time
    return found

def clear_uploaded_files() -> None:
    """Clear the uploaded files record."""
    if os.path.exists(_UPLOADED_FILES_RECORD):
        os.remove(_UPLOADED_FILES_RECORD)


def upload_files(file_paths: List[str]):
    """Upload files into the vector store pipeline.

    Returns:
        (vectorstore, table_catalog) — vectorstore may be None on failure;
        table_catalog is a list of dicts (one per table) with all display fields.
    """
    import json as _json

    if not file_paths:
        print("[upload_files] No documents provided.")
        return None, []

    try:
        # Copy files to DATA_DIR so they persist beyond Gradio's /tmp lifetime
        persistent_paths = []
        for fpath in file_paths:
            dest = os.path.join(DATA_DIR, os.path.basename(fpath))
            if os.path.abspath(fpath) != os.path.abspath(dest):
                shutil.copy2(fpath, dest)
            persistent_paths.append(dest)

        record_uploaded_files(persistent_paths)  # save persistent paths before any processing

        docs_list = load_documents_from_files(persistent_paths)

        if not docs_list:
            print("[upload_files] No documents successfully loaded.")
            return None, []

        # Build table catalog from the pre-split documents so metadata is intact
        table_catalog = []
        for doc in docs_list:
            if doc.metadata.get("type") != "table":
                continue
            m = doc.metadata
            try:
                cell_rows = _json.loads(m["cell_rows"]) if m.get("cell_rows") else []
            except Exception:
                cell_rows = []
            try:
                inspection = _json.loads(m["inspection"]) if m.get("inspection") else {}
            except Exception:
                inspection = {}
            table_catalog.append({
                "title":       m.get("title", ""),
                "page_idx":    m.get("page_idx", -1),
                "image_path":  m.get("image_path", ""),
                "pdf_path":    m.get("pdf_path", ""),
                "table_index": m.get("table_index", 0),
                "xml":         m.get("xml", ""),
                "inspection":  inspection,
                "cell_rows":   cell_rows,
                "source":      m.get("source", ""),
            })

        # Persist catalog alongside uploaded data for potential reload
        catalog_path = os.path.join(DATA_DIR, "table_catalog.json")
        try:
            with open(catalog_path, "w", encoding="utf-8") as _f:
                _json.dump(table_catalog, _f, ensure_ascii=False, indent=2)
            print(f"[upload_files] Saved table catalog ({len(table_catalog)} tables) → {catalog_path}")
        except Exception as _ce:
            print(f"[upload_files] Could not write table catalog: {_ce}")

        doc_splits = split_documents(docs_list)
        vs = embed_documents(doc_splits)
        return vs, table_catalog

    except Exception as e:
        print(f"[upload_files] Pipeline failed: {e}")
        return None, []




def _clear(
    persist_directory: str = None,
    collection_name: str = "rag-chroma",
    delete_all: bool = True
):
    """Clear the Chroma collection and optionally delete all shard folders (excluding hidden files)."""
    if persist_directory is None:
        persist_directory = DATA_DIR
    try:
        # Clear the collection via Chroma client
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
            persist_directory=persist_directory,
        )
        vectorstore._client.delete_collection(name=collection_name)
        vectorstore._client.create_collection(name=collection_name)
        print(f"[clear] Collection '{collection_name}' cleared.")

        if delete_all:
            for item in os.listdir(persist_directory):
                if item.startswith(".") or item.startswith("chroma.") or item.startswith("readme-images"):
                    continue  # Skip hidden files like .gitkeep and the sqlite3 file

                path = os.path.join(persist_directory, item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"[clear] Removed file: {item}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"[clear] Removed directory: {item}")
                except Exception as file_err:
                    print(f"[clear] Could not delete {item}: {file_err}")

    except Exception as e:
        print(f"[clear] Failed to clear vector store: {e}")


      
# def clear():
#     """ This is a helper function for emptying the collection the vector store. """
#     vectorstore = Chroma(
#         collection_name="rag-chroma",
#         embedding_function=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
#         persist_directory="/project/data",
#     )
    
#     vectorstore._client.delete_collection(name="rag-chroma")
#     vectorstore._client.create_collection(name="rag-chroma")

def get_retriever(): 
    """ This is a helper function for returning the retriever object of the vector store. """
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
        persist_directory=DATA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever
