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

### This module contains the chatui gui for having a conversation. ###

import functools
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import shutil
import os
import subprocess
import time
import sys
import json

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError 

from requests.exceptions import HTTPError
import traceback


from chatui.utils.error_messages import QUERY_ERROR_MESSAGES
from chatui.utils.graph import TavilyAPIError

# UI names and labels
SELF_HOSTED_TAB_NAME = "Self-Hosted Endpoint"
HOST_NAME = "Local NIM or Remote IP/Hostname"
HOST_PORT = "Host Port"
HOST_MODEL = "Model Name"


# Set recursion limit 
DEFAULT_RECURSION_LIMIT = 10
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT))


# Model identifiers with prefix
LLAMA = 'meta/llama-3.3-70b-instruct'  
MISTRAL = "mistralai/mixtral-8x22b-instruct-v0.1"
QWEN = "qwen/qwen3-235b-a22b"

# check if the internal API is set
INTERNAL_API = os.getenv('INTERNAL_API', 'no')

# Modify model identifiers (to use the internal endpoints if that variable is set).
if INTERNAL_API == 'yes':
    LLAMA = 'nvdev/meta/llama-3.3-70b-instruct'
    MISTRAL = 'nvdev/mistralai/mixtral-8x22b-instruct-v0.1'
    QWEN = 'nvdev/qwen/qwen-235b'

# URLs for default example docs for the RAG.
doc_links = (
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/desktop-app.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/command-line-interface.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/installation-overview.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/desktop-app-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/full-local-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/remote-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/uninstall.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/update.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/onboarding-project.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/project-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/location-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/single-container-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/compose-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/application-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/versioning-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/understand-project-specification.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/create-clone-publish.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/file-browser.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/deep-linking.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/versioning.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/add-existing-location.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/add-brev.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/vs-code.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/cursor.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/windsurf.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/package-management.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/prebuild-script.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/postbuild-script.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/runtime-configuration.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/hardware.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/multi-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/github-gitlab.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/self-hosted-gitlab.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/nvidia-integrations.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/app-sharing.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/use-custom-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/convert-repo.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/user-interface/desktop-app.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/user-interface/cli.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/glossary.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/runtime-configuration-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/hardware-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/custom-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/compose-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/compose-patterns-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/spec.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/base-environments.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/remote-locations.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/components.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/settings.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/customize-the-ui.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/runtimes.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/customca.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/proxy.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/support-matrix.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/versioning/git-configuration-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/windows-full-local-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/troubleshooting/troubleshooting.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/troubleshooting/logging.html",
)
EXAMPLE_LINKS_LEN = 10

EXAMPLE_LINKS = "\n".join(doc_links)

from chatui import assets, chat_client
from chatui.prompts import prompts_llama3, prompts_mistral
from chatui.utils import compile, database, logger, gpu_compatibility

from langgraph.graph import END, StateGraph

PATH = "/"
TITLE = "Word Document Table Transformation"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j .svelte-s1r2yt .svelte-cmf5ev {
    color: #76b900 !important;
}
.mode-banner {
    font-size: 1.05rem;
    font-weight: 500;
    background-color: #f0f4f8;
    padding: 0.5em 0.75em;
    border-left: 2px solid #76b900;
    margin-bottom: 0.5em;
    border-radius: 2px;
}
#tb-cals-xml-view {
    min-height: 600px !important;
    overflow-y: auto !important;
}
#tb-cals-xml-view > div {
    min-height: 600px !important;
    overflow-y: auto !important;
}
.cals-hi {
    background: #ffe080 !important;
    color: #111 !important;
    outline: 2px solid #d4a000 !important;
    border-radius: 2px;
}
#overwrite-storage-cb input[type="checkbox"] {
    border-color: #aaaaaa !important;
    border: 1.5px solid #aaaaaa !important;
}
"""

_LOCAL_JS = """
() => {
    // ── Deferred Edit-Clean-XML jump ────────────────────────────────────
    // When the user voluntarily clicks the "Edit Clean XML" tab after clicking
    // a cell, jump CM6 to the line stored by the last cell onclick.
    window._calsJumpLine = -1;
    document.addEventListener('click', function(e) {
        var btn = e.target.closest('#tb-right-tabs button[role=tab]');
        if (!btn || btn.textContent.indexOf('Edit Clean XML') < 0) return;
        var line = window._calsJumpLine;
        if (line < 0) return;
        window._calsJumpLine = -1;
        var lh = 17.6, target = line * lh, tries = 0;
        (function _jump() {
            var scr = document.querySelector('#tb-cals-xml .cm-scroller');
            if (!scr || scr.scrollHeight === 0) {
                if (++tries < 20) setTimeout(_jump, 100);
                return;
            }
            scr.scrollTop = Math.max(0, target - 5 * lh);
            setTimeout(function() {
                var prev = document.querySelectorAll('#tb-cals-xml .cals-edit-hi');
                for (var j = 0; j < prev.length; j++) {
                    prev[j].classList.remove('cals-edit-hi');
                    prev[j].style.backgroundColor = '';
                }
                var cms = document.querySelectorAll('#tb-cals-xml .cm-line');
                var best = null, bestd = Infinity;
                for (var k = 0; k < cms.length; k++) {
                    var d = Math.abs(cms[k].offsetTop - target);
                    if (d < bestd) { bestd = d; best = cms[k]; }
                }
                if (best) {
                    best.classList.add('cals-edit-hi');
                    best.style.backgroundColor = '#264f78';
                    best.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }, 200);
        })();
    });

    // ── Inline cell editor popup ─────────────────────────────────────────
    // Called from onclick attrs on CALS-entry spans in the annotation panel.
    // Reads data-celltext / data-verify / data-reason from the span, shows a
    // modal textarea; Save dispatches "eid||new_text" to #tb-entry-edit.
    window._calsEditPopup = function(eid) {
        var el = document.getElementById(eid);
        if (!el) return;
        var text   = el.getAttribute('data-celltext') || '';
        var verify = el.getAttribute('data-verify')   || '';
        var reason = el.getAttribute('data-reason')   || '';

        var old = document.getElementById('cals-edit-popup');
        if (old) old.remove();

        var badge = verify === 'ok'
            ? '<span style="color:#4ec94e;font-size:12px;font-weight:bold;">&#9679; Confirmed</span>'
            : (verify === 'unconfirmed'
                ? '<span style="color:#f14c4c;font-size:12px;font-weight:bold;">&#9679; Unconfirmed</span>'
                : '');

        function _esc(s) {
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
        var reasonHtml = reason
            ? '<div style="margin:8px 0 10px;font-size:11px;color:#9cdcfe;white-space:pre-wrap;'
              + 'border-left:2px solid #555;padding-left:8px;line-height:1.5;">'
              + _esc(reason) + '</div>'
            : '';

        var d = document.createElement('div');
        d.id = 'cals-edit-popup';
        d.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;'
            + 'background:rgba(0,0,0,0.55);z-index:99999;'
            + 'display:flex;align-items:center;justify-content:center;';
        d.innerHTML =
            '<div style="background:#1e1e1e;color:#d4d4d4;border-radius:8px;padding:24px;'
            + 'min-width:440px;max-width:660px;width:90%;font-family:monospace;'
            + 'box-shadow:0 8px 40px rgba(0,0,0,0.7);border:1px solid #444;">'
            + '<div style="margin-bottom:10px;font-size:11px;color:#808080;">Entry: '
            + '<span style="color:#ce9178;">' + _esc(eid) + '</span></div>'
            + badge + reasonHtml
            + '<textarea id="_cals_edit_ta" style="width:100%;box-sizing:border-box;'
            + 'margin-top:10px;height:80px;background:#252526;color:#d4d4d4;'
            + 'border:1px solid #555;border-radius:4px;padding:8px;'
            + 'font-family:monospace;font-size:13px;resize:vertical;">'
            + _esc(text) + '</textarea>'
            + '<div style="margin-top:12px;display:flex;gap:8px;justify-content:flex-end;">'
            + '<button id="_cals_cancel_btn" style="padding:6px 18px;background:#3a3a3a;'
            + 'color:#d4d4d4;border:1px solid #555;border-radius:4px;cursor:pointer;">Cancel</button>'
            + '<button id="_cals_save_btn" style="padding:6px 18px;background:#0e639c;'
            + 'color:#fff;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">'
            + '&#128190;&nbsp;Save</button>'
            + '</div></div>';
        document.body.appendChild(d);

        var ta = document.getElementById('_cals_edit_ta');
        if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }

        document.getElementById('_cals_cancel_btn').onclick = function() {
            document.getElementById('cals-edit-popup').remove();
        };
        document.getElementById('_cals_save_btn').onclick = function() {
            var val = document.getElementById('_cals_edit_ta').value;
            var sig = eid + '||' + val;
            var inp = document.querySelector('#tb-entry-edit input, #tb-entry-edit textarea');
            if (inp) { inp.value = sig; inp.dispatchEvent(new Event('input', {bubbles:true})); }
            document.getElementById('cals-edit-popup').remove();
        };
        // Click backdrop to dismiss
        d.onclick = function(ev) { if (ev.target === d) d.remove(); };
    };

    window._tblListSelect = function(row, idx) {
        var el = document.querySelector('#tb-list-click input');
        if (el) { el.value = idx; el.dispatchEvent(new Event('input', {bubbles:true})); }
        document.querySelectorAll('.tbl-list-row').forEach(function(r) {
            r.classList.remove('tbl-list-active');
        });
        row.classList.add('tbl-list-active');
    };
    window._paraListSelect = function(row, seq) {
        var el = document.querySelector('#tb-para-click input');
        if (el) { el.value = seq; el.dispatchEvent(new Event('input', {bubbles:true})); }
        document.querySelectorAll('.tbl-list-row').forEach(function(r) {
            r.classList.remove('tbl-list-active');
        });
        row.classList.add('tbl-list-active');
    };
}
"""

import os
log_path = os.environ.get("LOG_FILE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output.log"))
sys.stdout = logger.Logger(log_path)

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")
    
    """ Compile the agentic graph. """
    
    workflow = compile.compile_graph()
    app = workflow.compile()

    """ List of currently supported models. """
    
    model_list = [LLAMA, MISTRAL, QWEN]

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS, js=_LOCAL_JS) as page:
        gr.Markdown(f"# {TITLE}")

        """ Keep state of which queries need to use NIMs vs API Endpoints. """
        
        router_use_nim = gr.State(True)
        retrieval_use_nim = gr.State(True)
        generator_use_nim = gr.State(True)
        hallucination_use_nim = gr.State(True)
        answer_use_nim = gr.State(True)
        table_refs_state = gr.State([])   # stores selected table refs for the optimize agent
        all_tables_state = gr.State([])   # stores full table catalog after document upload
        current_table_state = gr.State({})  # metadata of the table currently shown in Table Browser
        cals_theme_state    = gr.State("verify")  # active CSS theme for the Rendered CALS XML panel

        """ Build the Chat Application. """
        
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=16, min_width=350):

                # Main chatbot panel. 
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=350):
                        chatbot = gr.Chatbot(show_label=False, height=120, type="messages")

                # Table page image viewer — shown only when direct document access finds table images
                with gr.Row(equal_height=True):
                    table_gallery = gr.Gallery(
                        label="Source Table Page(s)",
                        show_label=True,
                        visible=False,
                        columns=1,
                        rows=1,
                        height=500,
                        object_fit="contain",
                        elem_id="table-gallery",
                    )

                # Inspection report — shown only for direct table queries
                with gr.Row(equal_height=True):
                    inspection_box = gr.JSON(
                        label="Extraction Inspection Report (pdfplumber vs python-docx)",
                        visible=False,
                        elem_id="inspection-box",
                    )

                # Optimize button — appears after a direct table query
                with gr.Row(equal_height=True):
                    optimize_btn = gr.Button(
                        "⚙️ Optimize The Extraction",
                        visible=False,
                        variant="secondary",
                    )

                # Optimization agent round-by-round report
                with gr.Row(equal_height=True):
                    optimization_box = gr.JSON(
                        label="Optimization Agent — Strategy Comparison",
                        visible=False,
                        elem_id="optimization-box",
                    )

                # CALS XML view — shown when a direct table query returns CALS XML
                with gr.Row(equal_height=True):
                    xml_box = gr.Code(
                        label="CALS XML (XPP-ready)",
                        visible=False,
                        lines=30,
                        elem_id="cals-xml-box",
                    )

                # Message box for user input
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=450):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            container=False,
                            interactive=True,
                        )

                    with gr.Column(scale=1, min_width=150):
                        _ = gr.ClearButton([msg, chatbot], value="Clear Chat History")


            
            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")
            
            # Right column to display all relevant settings
            with gr.Column(scale=12, min_width=350) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:

                    with gr.TabItem("Quickstart", id=0) as instructions_tab:
                        
                        # Diagram of the agentic websearch RAG workflow
                        with gr.Row():
                            # Find image path dynamically
                            image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "agentic-flow.png")
                            agentic_flow = gr.Image(image_path, 
                                                    show_label=False,
                                                    show_download_button=False,
                                                    interactive=False)

                        with gr.Column():
                            step_1 = gr.Markdown(
                                """
                                * Extract Tables among Word Document
                                * Create Insertion Points for All Tables
                                * Create CASL XML representation for all Tables
                                * Enable RAG search for all Tables
                                """,
                                visible=False
                            )

                            step_2_btn = gr.Button("Step 1: Upload the Word file in Documents tab", elem_id="rag-inputs")
                            step_2 = gr.Markdown(
                                """
                                Go to the **Documents** tab → **Files** and upload a `.docx` file. The pipeline will:

                                1. **Extract tables** from the Word document using `python-docx` and match each table to its rendered PDF page
                                2. **Build CALS XML** — each table is converted to a structured CALS XML representation with insertion-point markers
                                3. **Annotate styles** — `pdfplumber` reads font data from the PDF to set `bold` and `indent` attributes on each cell *(optionally enhanced by a vision LLM)*
                                4. **Save the table catalog** — all tables are persisted to `table_catalog.json` and browsable in the **Table Browser** tab
                                5. **Embed into vector store** — table text is chunked and embedded using `all-MiniLM-L6-v2` into a local **Chroma** database, enabling RAG search over all tables
                                """,
                                visible=True
                            )

                            step_3_btn = gr.Button("Step 3: Resubmit the sample query", elem_id="rag-inputs")
                            step_3 = gr.Markdown(
                                """
                                ### Purpose: Generate and evaluate a generic response&nbsp;<ins>with</ins>&nbsp;added RAG context

                                * Select the same sample query from Step 1.
                                * Wait for the response to generate and evaluate the relevance of the response.
                                """,
                                visible=False
                            )

                            step_4_btn = gr.Button("Step 4: Monitor the results", elem_id="rag-inputs")
                            step_4 = gr.Markdown(
                                """
                                ### Purpose: Understand the actions the agent takes in generating responses

                                * Select the **Monitor** tab on the right-hand side of the browser window.
                                * Take a look at the actions taken by the agent under **Actions Console**.
                                * Take a look at the latest response generation details under **Response Trace**.
                                """,
                                visible=False
                            )

                            step_5_btn = gr.Button("Step 5: Next steps", elem_id="rag-inputs")
                            step_5 = gr.Markdown(
                                """
                                ### Purpose: Customize the project to your own documents and datasets

                                * To customize, clear out the database and upload your own data under **Documents**.
                                * Configure the **Router Prompt** to your RAG topic(s) under the **Models** tab.
                                * Submit a custom query to the RAG agent and evaluate the response.
                                """,
                                visible=False
                            )


                    # Settings for each component model of the agentic workflow
                    with gr.TabItem("Models", id=1, visible=False) as agent_settings:
                            gr.Markdown(
                                        """
                                        ##### Use the Models tab to configure individual model components
                                        - Click a component below (e.g. Router) and select API or NIM 
                                        - For APIs, select the model from the dropdown
                                        - For self-hosted endpoints, see instructions [here](https://github.com/nv-twhitehouse/workbench-example-agentic-rag/blob/twhitehouse/april-16/agentic-rag-docs/self-host.md)
                                        - (optional) Customize component behavior by changing the prompts
                                        """
                            )
                            gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                                    
                            ########################
                            ##### ROUTER MODEL #####
                            ########################
                            router_btn = gr.Button("Router", variant="sm")
                            with gr.Group(visible=False) as group_router:
                                with gr.Tabs(selected=1) as router_tabs:
                                    with gr.TabItem("API Endpoints", id=0) as router_api:
                                        router_mode_banner = gr.Markdown(value="🛠️ **Using Self-Hosted Endpoint**", elem_classes=["mode-banner"])
                                        model_router = gr.Dropdown(model_list, 
                                                                value=model_list[0],
                                                                label="Select a Model",
                                                                elem_id="rag-inputs", 
                                                                interactive=True)
                                        
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as router_nim:
                                        # with gr.Row():
                                        #     nim_router_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_router_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_router_ip = gr.Textbox(
                                                value = "localhost",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_router_port = gr.Textbox(
                                                value="1234",
                                                label=HOST_PORT,
                                                info="LM Studio default: 1234",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_router_id = gr.Textbox(
                                            value = "openai/gpt-oss-120b",
                                            label=HOST_MODEL,
                                            info="LM Studio model identifier",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_router_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_router_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as router_hide:
                                        gr.Markdown("")

                                with gr.Accordion("Configure the Router Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_router:
                                    prompt_router = gr.Textbox(value=prompts_llama3.router_prompt,
                                                            lines=12,
                                                            show_label=False,
                                                            interactive=True)
        
                            ##################################
                            ##### RETRIEVAL GRADER MODEL #####
                            ##################################
                            retrieval_btn = gr.Button("Retrieval Grader", variant="sm")
                            with gr.Group(visible=False) as group_retrieval:
                                with gr.Tabs(selected=1) as retrieval_tabs:
                                    retrieval_mode_banner = gr.Markdown(value="🛠️ **Using Self-Hosted Endpoint**", elem_classes=["mode-banner"])

                                    with gr.TabItem("API Endpoints", id=0) as retrieval_api:
                                        model_retrieval = gr.Dropdown(model_list, 
                                                                            value=model_list[0],
                                                                            label="Select a Model",
                                                                            elem_id="rag-inputs", 
                                                                            interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as retrieval_nim:
                                        # with gr.Row():
                                        #     nim_retrieval_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_retrieval_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_retrieval_ip = gr.Textbox(
                                                value = "localhost",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_retrieval_port = gr.Textbox(
                                                value="1234",
                                                label=HOST_PORT,
                                                info="LM Studio default: 1234",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_retrieval_id = gr.Textbox(
                                            value = "openai/gpt-oss-120b",
                                            label=HOST_MODEL,
                                            info="LM Studio model identifier",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )                                        
                                        # nim_retrieval_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_retrieval_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as retrieval_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Retrieval Grader Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_retrieval:
                                    prompt_retrieval = gr.Textbox(value=prompts_llama3.retrieval_prompt,
                                                                        lines=21,
                                                                        show_label=False,
                                                                        interactive=True)
        
                            ###########################
                            ##### GENERATOR MODEL #####
                            ###########################
                            generator_btn = gr.Button("Generator", variant="sm")
                            with gr.Group(visible=False) as group_generator:
                                with gr.Tabs(selected=1) as generator_tabs:
                                    generator_mode_banner = gr.Markdown(value="🛠️ **Using Self-Hosted Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as generator_api:
                                        model_generator = gr.Dropdown(model_list, 
                                                                    value=model_list[0],
                                                                    label="Select a Model",
                                                                    elem_id="rag-inputs", 
                                                                    interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as generator_nim:
                                        # with gr.Row():
                                        #     nim_generator_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_generator_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_generator_ip = gr.Textbox(
                                                value = "localhost",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_generator_port = gr.Textbox(
                                                value="1234",
                                                label=HOST_PORT,
                                                info="LM Studio default: 1234",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_generator_id = gr.Textbox(
                                            value = "openai/gpt-oss-120b",
                                            label=HOST_MODEL,
                                            info="LM Studio model identifier",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_generator_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_generator_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as generator_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Generator Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_generator:
                                    prompt_generator = gr.Textbox(value=prompts_llama3.generator_prompt,
                                                            lines=15,
                                                            show_label=False,
                                                            interactive=True)
        
                            ######################################
                            ##### HALLUCINATION GRADER MODEL #####
                            ######################################
                            hallucination_btn = gr.Button("Hallucination Grader", variant="sm")
                            with gr.Group(visible=False) as group_hallucination:
                                with gr.Tabs(selected=1) as hallucination_tabs:
                                    hallucination_mode_banner = gr.Markdown(value="🛠️ **Using Self-Hosted Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as hallucination_api:
                                        model_hallucination = gr.Dropdown(model_list, 
                                                                                value=model_list[0],
                                                                                label="Select a Model",
                                                                                elem_id="rag-inputs", 
                                                                                interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as hallucination_nim:
                                        # with gr.Row():
                                        #     nim_hallucination_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_hallucination_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_hallucination_ip = gr.Textbox(
                                                value = "localhost",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_hallucination_port = gr.Textbox(
                                                value="1234",
                                                label=HOST_PORT,
                                                info="LM Studio default: 1234",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_hallucination_id = gr.Textbox(
                                            value = "openai/gpt-oss-120b",
                                            label=HOST_MODEL,
                                            info="LM Studio model identifier",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_hallucination_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_hallucination_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as hallucination_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Hallucination Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_hallucination:
                                    prompt_hallucination = gr.Textbox(value=prompts_llama3.hallucination_prompt,
                                                                            lines=17,
                                                                            show_label=False,
                                                                            interactive=True)
        
                            ###############################
                            ##### ANSWER GRADER MODEL #####
                            ###############################
                            answer_btn = gr.Button("Answer Grader", variant="sm")
                            with gr.Group(visible=False) as group_answer:
                                with gr.Tabs(selected=1) as answer_tabs:
                                    answer_mode_banner = gr.Markdown(value="🛠️ **Using Self-Hosted Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as answer_api:
                                        model_answer = gr.Dropdown(model_list, 
                                                                        value=model_list[0],
                                                                        elem_id="rag-inputs",
                                                                        label="Select a Model",
                                                                        interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as answer_nim:
                                        # with gr.Row():
                                        #     nim_answer_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_answer_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_answer_ip = gr.Textbox(
                                                value = "localhost",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_answer_port = gr.Textbox(
                                                value="1234",
                                                label=HOST_PORT,
                                                info="LM Studio default: 1234",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_answer_id = gr.Textbox(
                                            value = "openai/gpt-oss-120b",
                                            label=HOST_MODEL,
                                            info="LM Studio model identifier",
                                            elem_id="rag-inputs",
                                            interactive=True
                                            )   

                                        # nim_answer_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_answer_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as answer_hide:
                                        gr.Markdown("")
                                        
                                with gr.Accordion("Configure the Answer Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_answer:
                                    prompt_answer = gr.Textbox(value=prompts_llama3.answer_prompt,
                                                                    lines=17,
                                                                    show_label=False,
                                                                    interactive=True)
                        
                    # Third tab item is for uploading to and clearing the vector database
                    with gr.TabItem("Documents", id=2) as document_settings:
                        gr.Markdown(
                            """                            
                            ##### Upload a Word document (.docx) to extract and transform its tables
                            - **Upload**: Drop a `.docx` file to extract all tables, generate insertion-point markers, and build CALS XML representations
                            - **VLM Annotation** *(optional)*: Use a vision model to annotate bold/indent styles on each table cell
                            - **Overwrite**: Check to clear all existing tables, images, and the vector store before processing
                            - **Clear Context**: Removes all extracted tables and resets the RAG index
                            """
                            )
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=1) as document_tabs:
                            with gr.TabItem("Webpages", id=0, visible=False) as url_tab:
                                url_docs = gr.Textbox(value=EXAMPLE_LINKS,
                                                      lines=EXAMPLE_LINKS_LEN, 
                                                      info="Enter a list of URLs, one per line", 
                                                      show_label=False, 
                                                      interactive=True)
                            
                                with gr.Row():
                                    url_docs_upload = gr.Button(value="Add to Context")
                                    url_docs_clear = gr.Button(value="Clear Context")

                            with gr.TabItem("Files", id=1) as pdf_tab:
                                docs_upload = gr.File(interactive=True, 
                                                          show_label=False, 
                                                          file_types=[".pdf", ".txt", ".csv", ".md", ".docx"], 
                                                          file_count="multiple")
                                overwrite_storage = gr.Checkbox(
                                    label="⚠️ Overwrite all existing storage (clears vectorstore, images, catalog)",
                                    value=False,
                                    interactive=True,
                                    elem_id="overwrite-storage-cb",
                                )
                                docs_clear = gr.Button(value="Clear Context")
                                upload_progress = gr.HTML(value="", visible=False, elem_id="upload-progress")
                                with gr.Accordion("🎨 VLM Style Annotation", open=False):
                                    gr.Markdown(
                                        "Enable to send each table page image to **Qwen2.5-VL** "
                                        "(or any vision model loaded in LM Studio) after upload "
                                        "to annotate `bold` and `indent` on each cell."
                                    )
                                    vlm_style_toggle = gr.Checkbox(
                                        label="Annotate bold/indent with vision LLM on upload",
                                        value=database.STYLE_ANNOTATE_METHOD == "vlm",
                                        interactive=True,
                                    )
                                    vlm_url_box = gr.Textbox(
                                        label="LM Studio URL",
                                        value=database.VLM_LMSTUDIO_URL,
                                        interactive=True,
                                        placeholder="http://localhost:1234/v1",
                                    )
    
                    # Fourth tab item is for the actions output console. 
                    with gr.TabItem("Monitor", id=3) as console_settings:
                        gr.Markdown(
                            """
                            ##### Use the Monitor tab to see the agent in action
                            - Actions Console: View the actions taken by the agent
                            - Response Trace: Full analysis behind the latest response
                            """
                            )
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=0) as console_tabs:
                            with gr.TabItem("Actions Console", id=0) as actions_tab:
                                logs = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False)
                            with gr.TabItem("Response Trace", id=1) as trace_tab:
                                actions = gr.JSON(
                                    scale=1,
                                    show_label=False,
                                    visible=True,
                                    elem_id="contextbox",
                                )
                    
                    # Fifth tab item is for collapsing the entire settings pane for readability. 
                    with gr.TabItem("Hide All Settings", id=4) as hide_all_settings:
                        gr.Markdown("")

        # ── TABLE BROWSER ─────────────────────────────────────────────────────
        # Full-width section, hidden until a document with tables is uploaded.
        with gr.Row(visible=False) as table_browser_section:
            with gr.Column():
                gr.Markdown("## Table Browser")
                gr.Markdown(
                    "Select a table — click any cell in the **Rendered CALS XML** panel "
                    "to jump to the corresponding `<entry>` in the **CALS XML** panel."
                )

                # ── Table menu: dropdown (default) + expandable full list ──────
                with gr.Row(equal_height=True):
                    table_selector = gr.Dropdown(
                        choices=[],
                        label="Tables in uploaded document(s)",
                        interactive=True,
                        type="index",
                        scale=9,
                        elem_id="tb-table-dropdown",
                    )
                    tb_list_toggle = gr.Button(
                        "≡ List",
                        variant="secondary",
                        size="sm",
                        scale=1,
                        elem_id="tb-list-toggle",
                    )
                    tb_vlm_annotate_btn = gr.Button(
                        "🎨 Re-annotate with VLM",
                        variant="secondary",
                        size="sm",
                        scale=2,
                        elem_id="tb-vlm-annotate-btn",
                    )
                    tb_validate_btn = gr.Button(
                        "🔍 Validate vs. Original",
                        variant="secondary",
                        size="sm",
                        scale=2,
                        elem_id="tb-validate-btn",
                    )
                    tb_fop_verify_btn = gr.Button(
                        "📊 FOP Verification",
                        variant="primary",
                        size="sm",
                        scale=2,
                        elem_id="tb-fop-verify-btn",
                    )
                    tb_agent_btn = gr.Button(
                        "🤖 Annotation Agent",
                        variant="primary",
                        size="sm",
                        scale=2,
                        elem_id="tb-agent-btn",
                    )
                    tb_export_pdf_btn = gr.Button(
                        "📄 Export Full Document PDF",
                        variant="secondary",
                        size="sm",
                        scale=2,
                        elem_id="tb-export-pdf-btn",
                    )
                    tb_export_docx_btn = gr.Button(
                        "📝 Export Edited .docx",
                        variant="secondary",
                        size="sm",
                        scale=2,
                        elem_id="tb-export-docx-btn",
                    )
                # Review panel — appears when the user clicks either export button
                tb_export_review_panel = gr.HTML(
                    value="", visible=False, elem_id="tb-export-review-panel"
                )
                with gr.Row():
                    tb_confirm_pdf_btn = gr.Button(
                        "\u2705 Confirm Export PDF",
                        variant="primary", size="sm",
                        visible=False, scale=2,
                        elem_id="tb-confirm-pdf-btn",
                    )
                    tb_confirm_docx_btn = gr.Button(
                        "\u2705 Confirm Export .docx",
                        variant="primary", size="sm",
                        visible=False, scale=2,
                        elem_id="tb-confirm-docx-btn",
                    )
                    tb_cancel_export_btn = gr.Button(
                        "\u2716 Cancel",
                        variant="stop", size="sm",
                        visible=False, scale=1,
                        elem_id="tb-cancel-export-btn",
                    )
                tb_vlm_status = gr.HTML(value="", elem_id="tb-vlm-status")
                tb_validate_status = gr.HTML(value="", elem_id="tb-validate-status")
                tb_fop_verify_status = gr.HTML(value="", elem_id="tb-fop-verify-status")
                tb_agent_status = gr.HTML(value="", elem_id="tb-agent-status")
                tb_export_pdf_status = gr.HTML(value="", elem_id="tb-export-pdf-status")
                tb_export_pdf_file = gr.File(
                    label="Download Full Document PDF",
                    visible=False,
                    elem_id="tb-export-pdf-file",
                )
                tb_export_docx_status = gr.HTML(value="", elem_id="tb-export-docx-status")
                tb_export_docx_file = gr.File(
                    label="Download Edited .docx",
                    visible=False,
                    elem_id="tb-export-docx-file",
                )

                # "Text + Tables" toggle — revealed only when the list panel is open
                tb_show_text_cb = gr.Checkbox(
                    label="Show Text + Tables (full document sequence)",
                    value=False,
                    visible=False,
                    elem_id="tb-show-text-cb",
                )

                # Hidden expanded list panel (toggled by the button above)
                tb_table_list = gr.HTML(
                    value="",
                    visible=False,
                    elem_id="tb-table-list",
                )
                # Hidden number input — list-row clicks write an index here to
                # trigger _show_table without a second gr.Dropdown change event.
                tb_list_click = gr.Number(value=-1, visible=False, elem_id="tb-list-click")
                # Hidden number input — paragraph-row clicks write doc_sequence seq here.
                tb_para_click = gr.Number(value=-1, visible=False, elem_id="tb-para-click")

                gr.HTML('<hr style="border:1px solid #ccc; margin:10px 0;">')

                # Side-by-side: left 3-tab panel | right 3-tab panel
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        with gr.Tabs(elem_id="tb-left-tabs"):
                            with gr.TabItem("🗂 Rendered CALS XML"):
                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=3):
                                        gr.Markdown("Click any cell to jump to its `<entry>` in the **View Annotation XML** tab")
                                    tb_cals_theme = gr.Dropdown(
                                        choices=[
                                            ("✅ Verify (green/red)",   "verify"),
                                            ("📊 iXBRL (XII)",          "ixbrl"),
                                            ("💼 Finance",              "finance"),
                                            ("🌙 Dark",                 "dark"),
                                            ("📄 Minimal",              "minimal"),
                                            ("🔲 Striped",              "striped"),
                                        ],
                                        value="verify",
                                        label="Style",
                                        interactive=True,
                                        scale=1,
                                        elem_id="tb-cals-theme",
                                    )
                                tb_cals_html = gr.HTML(
                                    elem_id="tb-cals-html",
                                )
                            with gr.TabItem("📄 PDF Page"):
                                tb_orig_img = gr.Image(
                                    label="Original PDF Page",
                                    show_label=False,
                                    elem_id="tb-orig-img",
                                )
                            with gr.TabItem("🔍 FOP vs Word Diff"):
                                tb_diff_img = gr.Image(
                                    label="Annotated Diff",
                                    show_label=False,
                                    elem_id="tb-diff-img",
                                )
                                tb_diff_ts = gr.HTML(
                                    value="",
                                    elem_id="tb-diff-ts",
                                )
                    with gr.Column(scale=1):
                        with gr.Tabs(elem_id="tb-right-tabs"):
                            with gr.TabItem("📄 View Annotation XML"):
                                tb_cals_xml_view = gr.HTML(
                                    elem_id="tb-cals-xml-view",
                                )
                            with gr.TabItem("✏️ Edit Clean XML"):
                                tb_cals_xml = gr.Code(
                                    label="",
                                    interactive=True,
                                    lines=30,
                                    elem_id="tb-cals-xml",
                                )
                                with gr.Row():
                                    tb_save_btn = gr.Button("💾 Save Clean XML", variant="primary", size="sm")
                                tb_save_status = gr.HTML(value="", elem_id="tb-save-status")
                                gr.HTML("<hr style='border:none;border-top:1px solid #333;margin:8px 0 4px;'>")
                                with gr.Row():
                                    tb_transform_dd = gr.Dropdown(
                                        choices=[
                                            ("\u2014 None \u2014",     "none"),
                                            ("🎨 Stripe rows", "stripe"),
                                        ],
                                        value="none",
                                        label="Style Transform",
                                        interactive=True,
                                        scale=2,
                                        elem_id="tb-transform-dd",
                                    )
                                    tb_transform_color_dd = gr.Dropdown(
                                        choices=[
                                            ("🔵 Light blue",   "EBF3FB"),
                                            ("🟢 Light green",  "E8F5E9"),
                                            ("🟡 Light yellow", "FFFDE7"),
                                            ("🟣 Lavender",     "EDE7F6"),
                                            ("🦷 Light pink",   "FCE4EC"),
                                        ],
                                        value="EBF3FB",
                                        label="Color",
                                        interactive=True,
                                        scale=1,
                                        elem_id="tb-transform-color-dd",
                                    )
                                with gr.Row():
                                    tb_apply_transform_btn = gr.Button(
                                        "✨ Apply Transform",
                                        variant="secondary",
                                        size="sm",
                                        scale=2,
                                        elem_id="tb-apply-transform-btn",
                                    )
                                    tb_clear_transform_btn = gr.Button(
                                        "✖ Clear",
                                        variant="stop",
                                        size="sm",
                                        scale=1,
                                        elem_id="tb-clear-transform-btn",
                                    )
                                tb_transform_status = gr.HTML(value="", elem_id="tb-transform-status")
                            with gr.TabItem("🖨 FOP Preview"):
                                tb_fop_btn = gr.Button("Render with Apache FOP", variant="primary", size="sm")
                                tb_fop_html = gr.HTML(
                                    value="<p style='color:#888;font-family:sans-serif;font-size:12px;'>Click the button to render the CALS XML through Apache FOP and view a typeset PDF preview.</p>",
                                    elem_id="tb-fop-html",
                                )
                            with gr.TabItem("📄 Original PDF"):
                                gr.Markdown("Source Word → PDF page &nbsp;*(compare against any left-panel tab)*")
                                tb_orig_img_r = gr.Image(
                                    label="Original PDF Page",
                                    show_label=False,
                                    elem_id="tb-orig-img-r",
                                )
                            with gr.TabItem("🔎 Annotation Review"):
                                gr.Markdown(
                                    "Per-entry comparison of **pdfplumber** vs **VLM** annotations. "
                                    "Run `🤖 Annotation Agent` to populate. "
                                    "**B** = bold · number = indent level · **⚠** = conflict."
                                )
                                tb_annot_review = gr.HTML(
                                    value="<p style='color:#888;font-family:sans-serif;font-size:12px;'>"
                                          "No annotation data yet — run 🤖 Annotation Agent first.</p>",
                                    elem_id="tb-annot-review",
                                )

                gr.HTML('<hr style="border:1px solid #ccc; margin:10px 0;">')

                # Secondary row (collapsed by default via Accordion)
                with gr.Accordion("Plain Text / Inspection Report", open=False):
                    with gr.Row(equal_height=False):
                        tb_plain_text = gr.Code(
                            label="Extracted Plain Text",
                            lines=20,
                            scale=1,
                            elem_id="tb-plain-text",
                        )
                        tb_inspection = gr.JSON(
                            label="Extraction Inspection Report",
                            scale=1,
                            elem_id="tb-inspection",
                        )
        # Hidden textbox: receives "eid||new_text" from the inline popup editor.
                        tb_entry_edit = gr.Textbox(visible=False, elem_id="tb-entry-edit")
        # ── END TABLE BROWSER ─────────────────────────────────────────────────

        gr.Timer(value=1).tick(logger.read_logs, inputs=None, outputs=logs)

        def _on_page_load():
            """Restore UI state from the previously persisted upload on startup."""
            _vs, table_catalog = database.load_persisted_state()
            if not table_catalog:
                return (
                    gr.update(), gr.update(), gr.update(),
                    [], gr.update(), gr.update(), gr.update(),
                )
            choices = []
            for i, t in enumerate(table_catalog):
                title = (t.get("title") or "").strip()
                pg    = t.get("page_idx", -1)
                if title:
                    label = f"[{i + 1}] {title}"
                elif pg >= 0:
                    label = f"[{i + 1}] Page {pg + 1}"
                else:
                    label = f"[{i + 1}] Table {i + 1}"
                choices.append(label)
            return (
                gr.update(value="Clear Docs", variant="secondary", interactive=True),
                gr.update(value="Clear Docs", variant="secondary", interactive=True),
                gr.update(visible=True),
                table_catalog,
                gr.update(choices=choices, value=None),
                gr.update(value=_build_table_list_html(table_catalog), visible=False),
                gr.update(visible=bool(table_catalog)),
            )

        page.load(
            _on_page_load, inputs=None,
            outputs=[
                url_docs_clear, docs_clear, agentic_flow,
                all_tables_state, table_selector, tb_table_list, table_browser_section,
            ],
        )

        """ These helper functions hide all other quickstart steps when one step is expanded. """

        def _toggle_quickstart_steps(step):
            steps = ["Step 1: Upload the sample dataset",
                     "Step 3: Resubmit the sample query",
                     "Step 4: Monitor the results",
                     "Step 5: Next steps"]
            visible = [False, False, False, False]
            visible[steps.index(step)] = True
            return {
                step_2: gr.update(visible=visible[0]),
                step_3: gr.update(visible=visible[1]),
                step_4: gr.update(visible=visible[2]),
                step_5: gr.update(visible=visible[3]),
            }

        step_2_btn.click(_toggle_quickstart_steps, [step_2_btn], [step_2, step_3, step_4, step_5])
        step_3_btn.click(_toggle_quickstart_steps, [step_3_btn], [step_2, step_3, step_4, step_5])
        step_4_btn.click(_toggle_quickstart_steps, [step_4_btn], [step_2, step_3, step_4, step_5])
        step_5_btn.click(_toggle_quickstart_steps, [step_5_btn], [step_2, step_3, step_4, step_5])

        """ These helper functions hide all settings when collapsed, and displays all settings when expanded. """

        def _toggle_hide_all_settings():
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        def _toggle_show_all_settings():
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=0),
                hidden_settings_column: gr.update(visible=False),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])
        show_settings.click(_toggle_show_all_settings, None, [settings_column, settings_tabs, hidden_settings_column])

        """ These helper functions hide the expanded component model settings when the Hide tab is clicked. """
        
        def _toggle_hide_router():
            return {
                group_router: gr.update(visible=False),
                router_tabs: gr.update(selected=0),
                router_btn: gr.update(visible=True),
            }

        def _toggle_hide_retrieval():
            return {
                group_retrieval: gr.update(visible=False),
                retrieval_tabs: gr.update(selected=0),
                retrieval_btn: gr.update(visible=True),
            }

        def _toggle_hide_generator():
            return {
                group_generator: gr.update(visible=False),
                generator_tabs: gr.update(selected=0),
                generator_btn: gr.update(visible=True),
            }

        def _toggle_hide_hallucination():
            return {
                group_hallucination: gr.update(visible=False),
                hallucination_tabs: gr.update(selected=0),
                hallucination_btn: gr.update(visible=True),
            }

        def _toggle_hide_answer():
            return {
                group_answer: gr.update(visible=False),
                answer_tabs: gr.update(selected=0),
                answer_btn: gr.update(visible=True),
            }

        router_hide.select(_toggle_hide_router, None, [group_router, router_tabs, router_btn])
        retrieval_hide.select(_toggle_hide_retrieval, None, [group_retrieval, retrieval_tabs, retrieval_btn])
        generator_hide.select(_toggle_hide_generator, None, [group_generator, generator_tabs, generator_btn])
        hallucination_hide.select(_toggle_hide_hallucination, None, [group_hallucination, hallucination_tabs, hallucination_btn])
        answer_hide.select(_toggle_hide_answer, None, [group_answer, answer_tabs, answer_btn])

        """ These helper functions set state and prompts when either the NIM or API Endpoint tabs are selected. """
        
        def _update_gpu_counts(component: str, gpu_type: str):
            """Update the available GPU counts for selected GPU type."""
            counts = gpu_compatibility.get_supported_gpu_counts(gpu_type)
            components = {
                "router": [nim_router_gpu_count, nim_router_id, nim_router_warning],
                "retrieval": [nim_retrieval_gpu_count, nim_retrieval_id, nim_retrieval_warning],
                "generator": [nim_generator_gpu_count, nim_generator_id, nim_generator_warning],
                "hallucination": [nim_hallucination_gpu_count, nim_hallucination_id, nim_hallucination_warning],
                "answer": [nim_answer_gpu_count, nim_answer_id, nim_answer_warning]
            }
            return {
                components[component][0]: gr.update(choices=counts, value=None, interactive=True),
                components[component][1]: gr.update(choices=[], value=None, interactive=False),
                components[component][2]: gr.update(visible=False, value="")
            }
        
        def _update_compatible_models(component: str, gpu_type: str, num_gpus: str):
            """Update the compatible models list based on GPU configuration."""
            if not gpu_type or not num_gpus:
                components = {
                    "router": [nim_router_id, nim_router_warning],
                    "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                    "generator": [nim_generator_id, nim_generator_warning],
                    "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                    "answer": [nim_answer_id, nim_answer_warning]
                }
                return {
                    components[component][0]: gr.update(choices=[], value=None, interactive=False),
                    components[component][1]: gr.update(visible=False, value="")
                }
            
            compatibility = gpu_compatibility.get_compatible_models(gpu_type, num_gpus)
            
            if compatibility["warning_message"]:
                components = {
                    "router": [nim_router_id, nim_router_warning],
                    "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                    "generator": [nim_generator_id, nim_generator_warning],
                    "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                    "answer": [nim_answer_id, nim_answer_warning]
                }
                return {
                    components[component][0]: gr.update(choices=[], value=None, interactive=False),
                    components[component][1]: gr.update(visible=True, value=f"⚠️ {compatibility['warning_message']}")
                }
            
            components = {
                "router": [nim_router_id, nim_router_warning],
                "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                "generator": [nim_generator_id, nim_generator_warning],
                "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                "answer": [nim_answer_id, nim_answer_warning]
            }
            return {
                components[component][0]: gr.update(
                    choices=compatibility["compatible_models"],
                    value=compatibility["compatible_models"][0] if compatibility["compatible_models"] else None,
                    interactive=True
                ),
                components[component][1]: gr.update(visible=False, value="")
            }

        # Add the event handlers for all components
        # nim_router_gpu_type.change(lambda x: _update_gpu_counts("router", x), nim_router_gpu_type, 
        #                          [nim_router_gpu_count, nim_router_id, nim_router_warning])
        # nim_router_gpu_count.change(lambda x, y: _update_compatible_models("router", x, y), 
        #                           [nim_router_gpu_type, nim_router_gpu_count], 
        #                           [nim_router_id, nim_router_warning])

        # nim_retrieval_gpu_type.change(lambda x: _update_gpu_counts("retrieval", x), nim_retrieval_gpu_type, 
        #                             [nim_retrieval_gpu_count, nim_retrieval_id, nim_retrieval_warning])
        # nim_retrieval_gpu_count.change(lambda x, y: _update_compatible_models("retrieval", x, y), 
        #                              [nim_retrieval_gpu_type, nim_retrieval_gpu_count], 
        #                              [nim_retrieval_id, nim_retrieval_warning])

        # nim_generator_gpu_type.change(lambda x: _update_gpu_counts("generator", x), nim_generator_gpu_type, 
        #                             [nim_generator_gpu_count, nim_generator_id, nim_generator_warning])
        # nim_generator_gpu_count.change(lambda x, y: _update_compatible_models("generator", x, y), 
        #                              [nim_generator_gpu_type, nim_generator_gpu_count], 
        #                              [nim_generator_id, nim_generator_warning])

        # nim_hallucination_gpu_type.change(lambda x: _update_gpu_counts("hallucination", x), nim_hallucination_gpu_type, 
        #                                 [nim_hallucination_gpu_count, nim_hallucination_id, nim_hallucination_warning])
        # nim_hallucination_gpu_count.change(lambda x, y: _update_compatible_models("hallucination", x, y), 
        #                                  [nim_hallucination_gpu_type, nim_hallucination_gpu_count], 
        #                                  [nim_hallucination_id, nim_hallucination_warning])

        # nim_answer_gpu_type.change(lambda x: _update_gpu_counts("answer", x), nim_answer_gpu_type, 
        #                          [nim_answer_gpu_count, nim_answer_id, nim_answer_warning])
        # nim_answer_gpu_count.change(lambda x, y: _update_compatible_models("answer", x, y), 
        #                           [nim_answer_gpu_type, nim_answer_gpu_count], 
        #                           [nim_answer_id, nim_answer_warning])

        """ These helper functions track the API Endpoint selected and regenerates the prompt accordingly. """
        
        def _toggle_model_router(selected_model: str):
            match selected_model:
                case str() if selected_model == LLAMA:
                    return gr.update(value=prompts_llama3.router_prompt)
                case str() if selected_model == MISTRAL:
                    return gr.update(value=prompts_mistral.router_prompt)
                case _:
                    return gr.update(value=prompts_llama3.router_prompt)
        
        def _toggle_model_retrieval(selected_model: str):
            match selected_model:
                case str() if selected_model == LLAMA:
                    return gr.update(value=prompts_llama3.retrieval_prompt)
                case str() if selected_model == MISTRAL:
                    return gr.update(value=prompts_mistral.retrieval_prompt)
                case _:
                    return gr.update(value=prompts_llama3.retrieval_prompt)

        def _toggle_model_generator(selected_model: str):
            match selected_model:
                case str() if selected_model == LLAMA:
                    return gr.update(value=prompts_llama3.generator_prompt)
                case str() if selected_model == MISTRAL:
                    return gr.update(value=prompts_mistral.generator_prompt)
                case _:
                    return gr.update(value=prompts_llama3.generator_prompt)
            
        def _toggle_model_hallucination(selected_model: str):
            match selected_model:
                case str() if selected_model == LLAMA:
                    return gr.update(value=prompts_llama3.hallucination_prompt)
                case str() if selected_model == MISTRAL:
                    return gr.update(value=prompts_mistral.hallucination_prompt)
                case _:
                    return gr.update(value=prompts_llama3.hallucination_prompt)
            
        def _toggle_model_answer(selected_model: str):
            match selected_model:
                case str() if selected_model == LLAMA:
                    return gr.update(value=prompts_llama3.answer_prompt)
                case str() if selected_model == MISTRAL:
                    return gr.update(value=prompts_mistral.answer_prompt)
                case _:
                    return gr.update(value=prompts_llama3.answer_prompt)

        # Update default prompts when an API endpoint model is selected from the dropdown
        # (This applies only to the "API Endpoints" tab — not to self-hosted NIM configurations)

        model_router.change(_toggle_model_router, [model_router], [prompt_router])
        model_retrieval.change(_toggle_model_retrieval, [model_retrieval], [prompt_retrieval])
        model_generator.change(_toggle_model_generator, [model_generator], [prompt_generator])
        model_hallucination.change(_toggle_model_hallucination, [model_hallucination], [prompt_hallucination])
        model_answer.change(_toggle_model_answer, [model_answer], [prompt_answer])

        # Toggle between NIM and API mode by setting `*_use_nim` state based on selected tab
        # - Selecting "API Endpoints" sets use_nim = False (use hosted model)
        # - Selecting "Self-Hosted Endpoint" sets use_nim = True (use local NIM container)
        
        # router eventhandlers
        router_api.select(lambda: (False,), [], [router_use_nim])
        router_nim.select(lambda: (True,), [], [router_use_nim])

        router_api.select(lambda: "💻 **Using API Endpoint**", [], [router_mode_banner])
        router_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [router_mode_banner])

        # retrieval eventhandlers   
        retrieval_api.select(lambda: (False,), [], [retrieval_use_nim])
        retrieval_nim.select(lambda: (True,), [], [retrieval_use_nim])

        retrieval_api.select(lambda: "💻 **Using API Endpoint**", [], [retrieval_mode_banner])
        retrieval_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [retrieval_mode_banner])

        # generator eventhandlers
        generator_api.select(lambda: (False,), [], [generator_use_nim])
        generator_nim.select(lambda: (True,), [], [generator_use_nim])

        generator_api.select(lambda: "💻 **Using API Endpoint**", [], [generator_mode_banner])
        generator_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [generator_mode_banner])

        # hallucination eventhandlers
        hallucination_api.select(lambda: (False,), [], [hallucination_use_nim])
        hallucination_nim.select(lambda: (True,), [], [hallucination_use_nim])

        hallucination_api.select(lambda: "💻 **Using API Endpoint**", [], [hallucination_mode_banner])
        hallucination_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [hallucination_mode_banner])

        # answer eventhandlers
        answer_api.select(lambda: (False,), [], [answer_use_nim])
        answer_nim.select(lambda: (True,), [], [answer_use_nim])

        answer_api.select(lambda: "💻 **Using API Endpoint**", [], [answer_mode_banner])
        answer_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [answer_mode_banner])
        
        """ These helper functions upload and clear the documents and webpages to/from the ChromaDB. """

        def _build_table_list_html(catalog, show_text=False):
            """Build the expanded list as an HTML string.

            Default (show_text=False): tables only — one clickable row per table.
            When show_text=True: full document sequence loaded from doc_segments.json;
            paragraph rows are tagged "X" (non-clickable), table rows are tagged "T"
            (clickable, same behaviour as the tables-only view).
            """
            if not catalog:
                return ""
            import html as _html
            import json as _json

            if show_text:
                # Load the ordered document sequence from disk
                segments_path = os.path.join(database.DATA_DIR, "doc_segments.json")
                segments: list = []
                if os.path.exists(segments_path):
                    try:
                        with open(segments_path, encoding="utf-8") as _f:
                            segments = _json.load(_f)
                    except Exception:
                        pass

                # table_index in segments → position in catalog list
                tidx_to_catidx = {t.get("table_index", i): i for i, t in enumerate(catalog)}

                # ── Build flat row list ────────────────────────────────
                flat_rows = []
                x_count = 0
                t_count = 0
                for seg in sorted(segments, key=lambda s: s.get("seq", 0)):
                    if seg["type"] == "paragraph":
                        text = (seg.get("text") or "").strip()
                        if not text:
                            continue
                        x_count += 1
                        flat_rows.append({
                            "type":             "paragraph",
                            "page_idx":         seg.get("page_idx", -1),
                            "label":            f"X{x_count}",
                            "text":             text,
                            "seq":              seg.get("seq", -1),
                            "annotated_image_path": seg.get("annotated_image_path", ""),
                        })
                    elif seg["type"] == "table":
                        tidx  = seg.get("table_index", -1)
                        cat_i = tidx_to_catidx.get(tidx, tidx)
                        t_count += 1
                        t_ent = catalog[cat_i] if 0 <= cat_i < len(catalog) else {}
                        flat_rows.append({
                            "type":     "table",
                            "page_idx": t_ent.get("page_idx", seg.get("page_idx", -1)),
                            "label":    f"T{t_count}",
                            "title":    (t_ent.get("title") or seg.get("title") or "").strip()
                                        or f"Table {tidx + 1}",
                            "pg_label": (f"{t_ent.get('page_idx', -1) + 1}"
                                         if t_ent.get("page_idx", -1) >= 0 else "—"),
                            "n_rows":   len(t_ent.get("cell_rows") or []),
                            "cat_i":    cat_i,
                        })

                # ── Pre-pass: mark joint page groups ──────────────────
                # Consecutive rows that share a page AND include both types
                # get a spanning connector cell in column 0.
                i = 0
                while i < len(flat_rows):
                    pg = flat_rows[i]["page_idx"]
                    if pg < 0:
                        flat_rows[i]["connector"] = "none"
                        i += 1
                        continue
                    j = i + 1
                    while j < len(flat_rows) and flat_rows[j]["page_idx"] == pg:
                        j += 1
                    grp = flat_rows[i:j]
                    has_both = (any(r["type"] == "paragraph" for r in grp)
                                and any(r["type"] == "table" for r in grp))
                    if has_both and len(grp) >= 2:
                        flat_rows[i]["connector"]       = "start"
                        flat_rows[i]["connector_span"]  = len(grp)
                        flat_rows[i]["connector_label"] = f"p.{pg + 1}"
                        for k in range(i + 1, j):
                            flat_rows[k]["connector"] = "cont"
                    else:
                        for k in range(i, j):
                            flat_rows[k]["connector"] = "none"
                    i = j

                # ── Render rows ────────────────────────────────────────
                rows_html = []
                for rd in flat_rows:
                    conn = rd.get("connector", "none")
                    if conn == "start":
                        conn_td = (f'<td class="tbl-conn-joint" rowspan="{rd["connector_span"]}">'
                                   f'<div class="tbl-conn-bar">{rd["connector_label"]}</div></td>')
                    elif conn == "cont":
                        conn_td = ""
                    else:
                        conn_td = '<td class="tbl-conn-empty"></td>'

                    if rd["type"] == "paragraph":
                        short = (rd["text"][:80] + "\u2026") if len(rd["text"]) > 80 else rd["text"]
                        rows_html.append(
                            f'<tr class="tbl-list-row tbl-list-text" onclick="_paraListSelect(this,{rd["seq"]})">'
                            f'{conn_td}'
                            f'<td class="tbl-list-tag tbl-list-tag-x"><span>X</span></td>'
                            f'<td class="tbl-list-num tbl-list-xnum">{rd["label"]}</td>'
                            f'<td class="tbl-list-name tbl-list-textcontent" colspan="2">{_html.escape(short)}</td>'
                            f'</tr>'
                        )
                    else:
                        rows_html.append(
                            f'<tr class="tbl-list-row tbl-list-table" onclick="_tblListSelect(this,{rd["cat_i"]})">'
                            f'{conn_td}'
                            f'<td class="tbl-list-tag tbl-list-tag-t"><span>T</span></td>'
                            f'<td class="tbl-list-num tbl-list-tnum">{rd["label"]}</td>'
                            f'<td class="tbl-list-name">{_html.escape(rd["title"])}</td>'
                            f'<td class="tbl-list-pg">{rd["pg_label"]}</td>'
                            f'</tr>'
                        )
                th_first = '<th></th><th>Tag</th><th>#</th>'
                th_rest  = '<th>Name</th><th>Page</th>'
            else:
                rows_html = []
                for i, t in enumerate(catalog):
                    title    = (t.get("title") or "").strip() or f"Table {i + 1}"
                    pg       = t.get("page_idx", -1)
                    pg_label = f"{pg + 1}" if pg >= 0 else "—"
                    n_rows   = len(t.get("cell_rows") or [])
                    rows_html.append(
                        f'<tr class="tbl-list-row" onclick="_tblListSelect(this,{i})">'
                        f'<td class="tbl-list-num">{i + 1}</td>'
                        f'<td class="tbl-list-name">{_html.escape(title)}</td>'
                        f'<td class="tbl-list-pg">{pg_label}</td>'
                        f'<td class="tbl-list-rows">{n_rows}</td>'
                        f'</tr>'
                    )
                th_first = '<th>#</th>'
                th_rest  = '<th>Name</th><th>Page</th><th>Rows</th>'

            _css = (
                '<style>'
                '#tb-table-list{margin:6px 0 10px 0;border-radius:6px;overflow:hidden;border:1px solid #444;}'
                '#tb-table-list table{width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;}'
                '#tb-table-list th{background:#2a2a2a;color:#aaa;padding:5px 10px;text-align:left;'
                'border-bottom:1px solid #555;font-weight:600;letter-spacing:.04em;}'
                '.tbl-list-row{cursor:pointer;background:#1e1e1e;color:#ddd;border-bottom:1px solid #333;}'
                '.tbl-list-row:hover{background:#2d3a4a;color:#fff;}'
                '.tbl-list-active{background:#1a3a5c!important;color:#7ec8e3!important;}'
                '.tbl-list-num{color:#888;width:36px;padding:5px 8px;text-align:right;}'
                '.tbl-list-xnum{color:#f5c542;width:40px;font-size:10px;text-align:right;}'
                '.tbl-list-tnum{color:#7ec8e3;width:40px;font-size:10px;text-align:right;}'
                '.tbl-conn-empty{width:18px;padding:0 4px;border-right:1px solid #2a2a2a;}'
                '.tbl-conn-joint{width:18px;padding:0 3px;vertical-align:middle;text-align:center;'
                'background:rgba(52,152,219,0.10);border-left:2px solid #3498db;border-right:2px solid #3498db;}'
                '.tbl-conn-bar{writing-mode:vertical-lr;transform:rotate(180deg);font-size:9px;'
                'color:#3498db;font-weight:700;letter-spacing:.08em;}'
                '.tbl-list-tag{width:34px;padding:4px 6px;text-align:center;}'
                '.tbl-list-tag span{display:inline-block;font-weight:700;font-size:10px;letter-spacing:.05em;'
                'border-radius:3px;padding:2px 5px;}'
                '.tbl-list-tag-x span{background:#4a3a00;color:#f5c542;border:1px solid #f5c542;}'
                '.tbl-list-tag-t span{background:#0d2e44;color:#7ec8e3;border:1px solid #7ec8e3;}'
                '.tbl-list-text{cursor:pointer;}'
                '.tbl-list-textcontent{color:#aaa!important;font-style:italic;}'
                '.tbl-list-name{padding:5px 12px;max-width:520px;overflow:hidden;'
                'text-overflow:ellipsis;white-space:nowrap;}'
                '.tbl-list-pg{color:#aaa;width:52px;padding:5px 8px;text-align:center;}'
                '.tbl-list-rows{color:#aaa;width:52px;padding:5px 8px;text-align:center;}'
                '</style>'
            )
            _js = ""
            return (
                _css + _js
                + '<table><thead><tr>'
                + th_first
                + th_rest
                + '</tr></thead><tbody>'
                + ''.join(rows_html)
                + '</tbody></table>'
            )

        _tb_list_visible = [False]  # mutable closure flag

        def _toggle_table_list(current_visible, catalog, show_text):
            """Toggle the expanded table list panel and its Text+Tables checkbox."""
            new_vis = not current_visible
            html = _build_table_list_html(catalog, show_text=show_text) if new_vis else ""
            label = "▲ List" if new_vis else "≡ List"
            return (
                new_vis,
                gr.update(value=html, visible=new_vis),
                gr.update(value=label),
                gr.update(visible=new_vis),
            )

        _tb_list_vis_state = gr.State(False)
        tb_list_toggle.click(
            _toggle_table_list,
            inputs=[_tb_list_vis_state, all_tables_state, tb_show_text_cb],
            outputs=[_tb_list_vis_state, tb_table_list, tb_list_toggle, tb_show_text_cb],
        )

        def _on_show_text_toggle(show_text, current_vis, catalog):
            """Rebuild the list HTML immediately when the Text+Tables checkbox changes."""
            if not current_vis:
                return gr.update()
            html = _build_table_list_html(catalog, show_text=show_text)
            return gr.update(value=html, visible=True)

        tb_show_text_cb.change(
            _on_show_text_toggle,
            inputs=[tb_show_text_cb, _tb_list_vis_state, all_tables_state],
            outputs=[tb_table_list],
        )

        def _set_vlm_annotate(enabled):
            database.STYLE_ANNOTATE_METHOD = "vlm" if bool(enabled) else "none"

        def _set_vlm_url(url):
            u = (url or "").strip()
            if u:
                database.VLM_LMSTUDIO_URL = u

        vlm_style_toggle.change(_set_vlm_annotate, inputs=[vlm_style_toggle], outputs=[])
        vlm_url_box.change(_set_vlm_url, inputs=[vlm_url_box], outputs=[])

        def _upload_progress_html(done, total, stage):
            """Build a styled progress bar HTML for the upload status panel."""
            import html as _h
            label = _h.escape(str(stage))
            if total > 0:
                pct = min(100, int(done * 100 / total))
                bar_w = f"{pct}%"
                pct_txt = f"{pct}%"
                pulse = ""
            else:
                pct = None
                bar_w = "100%"
                pct_txt = "\u2026"
                pulse = "animation:_upulse 1.2s ease-in-out infinite;"
            bar_col = "#4a9eff" if pct is not None else "#5a6eff"
            return (
                "<style>@keyframes _upulse{0%,100%{opacity:.3}50%{opacity:.9}}</style>"
                "<div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:7px;"
                "padding:10px 14px;margin:6px 0;font-family:monospace;font-size:13px;color:#ddd;'>"
                f"<div style='margin-bottom:7px;color:#7ec8e3;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis;'>\u23f3 {label}</div>"
                "<div style='background:#2a2a3e;border-radius:3px;height:7px;overflow:hidden;'>"
                f"<div style='background:{bar_col};height:100%;width:{bar_w};"
                f"transition:width 0.35s ease;{pulse}'></div>"
                "</div>"
                f"<div style='text-align:right;font-size:11px;color:#777;margin-top:4px;'>{pct_txt}</div>"
                "</div>"
            )

        def _upload_documents_files(files, overwrite, progress=gr.Progress()):
            import queue as _queue
            import threading as _threading

            q = _queue.Queue()
            result_box = [None, None]  # [vs, table_catalog]

            def _on_progress(done, total, stage):
                q.put((done, total, stage))

            def _worker():
                try:
                    if overwrite:
                        _on_progress(0, 0, "Clearing existing storage…")
                        database._clear(delete_all=True)
                    vs, catalog = database.upload_files(files, on_progress=_on_progress)
                    result_box[0], result_box[1] = vs, catalog
                except Exception as _we:
                    print(f"[upload worker] {_we}")
                    result_box[0], result_box[1] = None, []
                finally:
                    q.put(None)  # sentinel: worker finished

            _threading.Thread(target=_worker, daemon=True).start()

            # Stream progress updates until the worker signals done
            _noop = gr.update()
            while True:
                item = q.get()
                if item is None:  # sentinel
                    break
                done, total, stage = item
                progress(done / max(total, 1), desc=stage)
                yield (
                    _noop, _noop, _noop, _noop, _noop,
                    _noop, _noop, _noop, _noop,
                    gr.update(value=_upload_progress_html(done, total, stage), visible=True),
                )

            # Worker is done — build the final update
            table_catalog = result_box[1] or []
            choices = []
            for i, t in enumerate(table_catalog):
                title = (t.get("title") or "").strip()
                pg    = t.get("page_idx", -1)
                if title:
                    label = f"[{i + 1}] {title}"
                elif pg >= 0:
                    label = f"[{i + 1}] Page {pg + 1}"
                else:
                    label = f"[{i + 1}] Table {i + 1}"
                choices.append(label)

            yield (
                gr.update(value="Clear Docs", variant="secondary", interactive=True),
                gr.update(value="Clear Docs", variant="secondary", interactive=True),
                gr.update(visible=True),
                table_catalog,
                gr.update(choices=choices, value=None),
                gr.update(value=_build_table_list_html(table_catalog), visible=False),
                False,
                gr.update(value="\u2261 List"),
                gr.update(visible=bool(table_catalog)),
                gr.update(value="", visible=False),
            )

        def _upload_documents(docs: str, progress=gr.Progress()):
            progress(0.2, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.4, desc="Processing URL List")
            docs_list = docs.splitlines()
            progress(0.6, desc="Creating Context")
            vectorstore = database.upload(docs_list)
            progress(0.8, desc="Cleaning Up")
            time.sleep(0.75)
            if vectorstore is None:
                return {
                    url_docs_upload: gr.update(value="No valid URLS - Try again", variant="secondary", interactive=True),
                    url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=False),
                    docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=False),
                    agentic_flow: gr.update(visible=False),  # or leave as-is if flow is independent
                }
            return {
                url_docs_upload: gr.update(value="Context Created", variant="primary", interactive=False),
                url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        def _clear_documents(progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Clearing Context")
            database._clear()
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_upload:       gr.update(value="Add to Context", variant="secondary", interactive=True),
                url_docs_clear:        gr.update(value="Context Cleared", variant="primary", interactive=False),
                docs_upload:           gr.update(value=None),
                docs_clear:            gr.update(value="Context Cleared", variant="primary", interactive=False),
                agentic_flow:          gr.update(visible=True),
                all_tables_state:      [],
                table_selector:        gr.update(choices=[], value=None),
                tb_table_list:         gr.update(value="", visible=False),
                _tb_list_vis_state:    False,
                tb_list_toggle:        gr.update(value="≡ List"),
                table_browser_section: gr.update(visible=False),
            }

        url_docs_upload.click(_upload_documents, [url_docs], [url_docs_upload, url_docs_clear, docs_clear, agentic_flow])
        url_docs_clear.click(_clear_documents, [], [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow, all_tables_state, table_selector, tb_table_list, _tb_list_vis_state, tb_list_toggle, table_browser_section])
        docs_upload.upload(_upload_documents_files, [docs_upload, overwrite_storage], [url_docs_clear, docs_clear, agentic_flow, all_tables_state, table_selector, tb_table_list, _tb_list_vis_state, tb_list_toggle, table_browser_section, upload_progress])
        docs_clear.click(_clear_documents, [], [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow, all_tables_state, table_selector, tb_table_list, _tb_list_vis_state, tb_list_toggle, table_browser_section])

        def _diff_ts_html(path):
            """Return a small HTML snippet showing how long ago a diff image was created."""
            import datetime as _dt
            if not path or not os.path.exists(path):
                return ""
            try:
                mtime = os.path.getmtime(path)
                dt    = _dt.datetime.fromtimestamp(mtime)
                delta = int((_dt.datetime.now() - dt).total_seconds())
                if delta < 5:
                    rel = "just now"
                elif delta < 60:
                    rel = f"{delta}s ago"
                elif delta < 3600:
                    rel = f"{delta // 60}m ago"
                elif delta < 86400:
                    rel = f"{delta // 3600}h {(delta % 3600) // 60}m ago"
                else:
                    rel = f"{delta // 86400}d ago"
                ts_str = dt.strftime("%Y-%m-%d  %H:%M:%S")
                return (
                    f'<p style="font-family:sans-serif;font-size:11px;color:#888;'
                    f'margin:4px 2px 0;line-height:1.4;">'
                    f'&#128337;&nbsp;Generated: <code style="font-size:11px;">{ts_str}</code>'
                    f'&ensp;<span style="color:#aaa;">({rel})</span></p>'
                )
            except Exception:
                return ""

        def _show_table(selected_idx, all_tables, theme="verify"):
            """Populate the 6-panel Table Browser when a table is selected."""
            from tabulate import tabulate as _tabulate

            _empty = gr.update(value=None)
            if selected_idx is None or not all_tables or selected_idx >= len(all_tables):
                return _empty, _empty, _empty, _empty, gr.update(value=""), _empty, gr.update(value=""), _empty, _empty, {}, gr.update(value="")

            t = all_tables[selected_idx]

            # Top-left: plain-text tabulate rendering
            cell_rows = t.get("cell_rows", [])
            plain = _tabulate(cell_rows, tablefmt="simple") if cell_rows else ""
            title = (t.get("title") or "").strip()
            if title:
                plain = f"[Title: {title}]\n\n{plain}"

            # Top-middle: CALS XML
            # clean_xml  — original CALS with only structural attributes (bold, indent, colspan…)
            #              shown in the editable code panel so the user can edit clean markup
            # display_xml — annotated version with verify="ok"/"unconfirmed" baked in,
            #              used only for the rendered colour-coded view
            inspection = t.get("inspection") or {}
            clean_xml   = t.get("xml", "")
            display_xml = (inspection.get("annotated_xml") if isinstance(inspection, dict) else None) or clean_xml

            # Top-right: full inspection report
            inspection_data = inspection if isinstance(inspection, dict) else {}

            # Bottom-left: original page PNG (prefer joint annotated version if available)
            annot_path = t.get("annotated_image_path", "") or None
            orig_path  = t.get("image_path", "") or None
            orig_img   = next(
                (p for p in [annot_path, orig_path] if p and os.path.exists(p)), None
            )
            orig_img = database._draw_bbox_on_image(orig_img, t.get("bbox"), color="blue")

            # Left panel + dark view panel
            table_html, xml_panel_html = database._cals_to_interactive_html(display_xml, theme=theme, clean_xml=clean_xml)

            # Bottom-right: annotated diff PNG (fall back to original page)
            diff_path = (inspection.get("annotated_image_path") if isinstance(inspection, dict) else None)
            if not diff_path or not os.path.exists(diff_path):
                diff_path = orig_path
            diff_img = diff_path if (diff_path and os.path.exists(diff_path)) else None

            # Metadata for the focused table (used by _rerender_cals for scoped re-verification)
            table_meta = {
                "pdf_path":   t.get("pdf_path", ""),
                "page_idx":   t.get("page_idx", -1),
                "image_path": t.get("image_path", ""),
                "title":      title,
            }

            return (
                gr.update(value=plain),
                gr.update(value=clean_xml),        # tb_cals_xml  — raw editable text (clean, no verify attrs)
                gr.update(value=inspection_data),
                gr.update(value=orig_img),
                gr.update(value=table_html),       # tb_cals_html — left rendered panel
                gr.update(value=diff_img),
                gr.update(value=_diff_ts_html(diff_img)),  # tb_diff_ts
                gr.update(value=xml_panel_html),   # tb_cals_xml_view — dark panel with span IDs
                gr.update(value=orig_img),         # tb_orig_img_r — right panel mirror
                table_meta,                        # current_table_state
                gr.update(value=database._build_annotation_review_html(t)),  # tb_annot_review
            )

        def _reannotate_all_with_vlm(all_tables, selected_idx, theme):
            """Run VLM style annotation on every table in the catalog using stored image paths.
            Updates table XMLs in-memory and persists the catalog, then re-renders the current table.
            """
            import json as _json

            if not all_tables:
                yield (
                    all_tables, gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    "<span style='color:#c0392b;font-family:monospace;font-size:12px;'>No tables loaded.</span>",
                )
                return

            url = database.VLM_LMSTUDIO_URL
            total = len(all_tables)
            updated = 0

            for i, t in enumerate(all_tables):
                img_path = t.get("image_path", "")
                cals_xml = t.get("xml", "")
                label = (t.get("title") or f"Table {i+1}")[:40]
                pct = int(i * 100 / total)
                yield (
                    all_tables, gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    f"<span style='color:#7ec8e3;font-family:monospace;font-size:12px;'>"
                    f"&#9203; [{i+1}/{total}] {label}… ({pct}%)</span>",
                )
                if not cals_xml or not img_path:
                    continue
                try:
                    annotated = database._annotate_entry_styles_with_vlm(cals_xml, img_path, lmstudio_url=url)
                    if annotated and annotated != cals_xml:
                        all_tables[i]["xml"] = annotated
                        updated += 1
                except Exception as _e:
                    print(f"[reannotate] table {i}: {_e}")

                # Persist after each table so progress survives a mid-run stop
                catalog_path = os.path.join(database.DATA_DIR, "table_catalog.json")
                try:
                    with open(catalog_path, "w", encoding="utf-8") as _f:
                        _json.dump(all_tables, _f, ensure_ascii=False, indent=2)
                except Exception as _e:
                    print(f"[reannotate] catalog save error: {_e}")

            # Re-render currently selected table with updated XML
            show_outs = _show_table(selected_idx, all_tables, theme)
            status_html = (
                f"<span style='color:#2ecc71;font-family:monospace;font-size:12px;'>"
                f"&#10003; Done — {updated}/{total} tables annotated.</span>"
            )
            yield (all_tables, *show_outs, status_html)

        # ── Annotation Agent ─────────────────────────────────────────────────
        def _agent_status_html(log_lines, done, total):
            pct = int(done * 100 / max(total, 1))
            bar_col = "#2ecc71" if done >= total else "#3498db"
            lines_html = "".join(
                "<div style='color:{};font-size:11px;line-height:1.5;'>{}</div>".format(
                    "#2ecc71" if "✓" in ln else "#f39c12" if "⚠" in ln else "#7ec8e3",
                    ln,
                )
                for ln in log_lines[-18:]
            )
            return (
                "<div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:6px;"
                "padding:10px 14px;font-family:monospace;margin-top:4px;'>"
                f"<div style='color:#aaa;font-size:11px;margin-bottom:6px;'>"
                f"Annotation Agent — {done}/{total} ({pct}%)</div>"
                "<div style='background:#2a2a3e;border-radius:3px;height:5px;margin-bottom:8px;overflow:hidden;'>"
                f"<div style='background:{bar_col};height:100%;width:{pct}%;'></div></div>"
                f"{lines_html}</div>"
            )

        def _run_annotation_agent(all_tables, selected_idx, theme):
            """Agent loop: pdfplumber + optional VLM + LLM reconciliation on every table.

            Streams live per-table status via Gradio generator yields.
            The local LLM (Qwen by default, loaded in LM Studio) reconciles
            any bold/indent conflicts between the two annotation sources.
            Results are written back to all_tables and persisted to disk after
            each table so progress survives a mid-run stop.
            """
            import json as _json

            if not all_tables:
                yield (
                    all_tables,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    "<span style='color:#c0392b;font-family:monospace;font-size:12px;'>"
                    "No tables loaded.</span>",
                )
                return

            total = len(all_tables)
            log_lines = []
            catalog_path = os.path.join(database.DATA_DIR, "table_catalog.json")

            # Build PDF page cache once per unique PDF path (avoids re-opening the
            # same file for every table — typically all 60+ tables share one PDF).
            unique_pdfs = {
                e.get("pdf_path", "")
                for e in all_tables
                if e.get("pdf_path", "") and e.get("page_idx", -1) >= 0
            }
            pdf_cache: dict = {}
            for _pp in unique_pdfs:
                if _pp and os.path.exists(_pp):
                    try:
                        pdf_cache[_pp] = database._build_pdf_page_cache(_pp)
                        log_lines.append(f"[cache] built page cache: {os.path.basename(_pp)}")
                    except Exception as _ep:
                        print(f"[agent] cache build failed for {_pp}: {_ep}")

            # Resolve the LM Studio model name ONCE here, before the per-table loop,
            # so neither _annotate_entry_styles_with_vlm nor _reconcile_with_llm
            # ever calls models.list() during the loop.  The result is stored in
            # database._vlm_model_id_cache so it is picked up automatically.
            _resolved_model = "local-model"
            try:
                from openai import OpenAI as _OpenAI
                _lm_client = _OpenAI(base_url=database.VLM_LMSTUDIO_URL, api_key="lm-studio")
                _lm_models = _lm_client.models.list()
                _resolved_model = _lm_models.data[0].id if _lm_models.data else "local-model"
                database._vlm_model_id_cache[database.VLM_LMSTUDIO_URL] = _resolved_model
                log_lines.append(f"[cache] LM Studio model: {_resolved_model}")
            except Exception as _em:
                log_lines.append(f"[cache] LM Studio not reachable — VLM will be skipped ({_em})")

            # Derive a short slug from the model name to use as the catalog key.
            # Each distinct model version gets its own key so no run ever overwrites
            # a previous model's results.  The legacy bare "xml_vlm" key (written by
            # earlier Qwen2.5-VL runs) is never touched again — it shows up in the
            # Review tab labelled "Qwen" automatically via _model_label().
            import re as _re
            _mid_lower = _resolved_model.lower()
            if "gemma" in _mid_lower:
                # e.g. gemma-3-12b  →  "gemma3_12b", or just "gemma" if no version found
                _gver = _re.search(r"gemma[-_]?(\d+(?:[._]\d+)?)", _mid_lower)
                _model_slug = "gemma" + _gver.group(1).replace(".", "").replace("-", "").replace("_", "") if _gver else "gemma"
            elif "qwen" in _mid_lower:
                # e.g. qwen2.5-vl-7b → "qwen25vl", qwen3.5-35b-a3b → "qwen35"
                _qver = _re.search(r"qwen(\d+(?:[._]\d+)*)", _mid_lower)
                _qver_str = _qver.group(1).replace(".", "").replace("_", "") if _qver else ""
                _qsuffix = "vl" if "vl" in _mid_lower else ""
                _model_slug = f"qwen{_qver_str}{_qsuffix}" if _qver_str else "qwen"
            elif "phi" in _mid_lower:
                _pver = _re.search(r"phi[-_]?(\d+(?:[._]\d+)?)", _mid_lower)
                _model_slug = "phi" + _pver.group(1).replace(".", "").replace("-", "").replace("_", "") if _pver else "phi"
            elif "llama" in _mid_lower:
                _lver = _re.search(r"llama[-_]?(\d+(?:[._]\d+)?)", _mid_lower)
                _model_slug = "llama" + _lver.group(1).replace(".", "").replace("-", "").replace("_", "") if _lver else "llama"
            elif "mistral" in _mid_lower:
                _model_slug = "mistral"
            else:
                _model_slug = _re.sub(r"[^a-z0-9]", "_", _mid_lower)[:12].strip("_") or "vlm"
            # Storage keys for this run (always non-empty slug now)
            _vlm_key       = f"xml_vlm_{_model_slug}"
            _conflicts_key = f"annotation_conflicts_{_model_slug}"
            _method_suffix = _model_slug
            log_lines.append(f"[cache] VLM catalog key: {_vlm_key}")

            # Pre-flight sanity: count how many tables already have this model's key.
            _already_done = sum(1 for e in all_tables if e.get(_vlm_key))
            if _already_done == total:
                log_lines.append(
                    f"⚠ ALL {total} tables already annotated with key '{_vlm_key}'. "
                    f"If you want to re-run, clear that key first. "
                    f"Is the correct model loaded in LM Studio?"
                )
            elif _already_done > 0:
                log_lines.append(
                    f"[cache] {_already_done}/{total} tables already have '{_vlm_key}' — "
                    f"those will be skipped, {total - _already_done} remaining."
                )
            else:
                log_lines.append(
                    f"[cache] 0/{total} tables have '{_vlm_key}' — full run."
                )

            # Stream the pre-flight status before starting per-table work
            yield (
                all_tables,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(),
                _agent_status_html(log_lines, 0, total),
            )

            for i, entry in enumerate(all_tables):
                title = (entry.get("title") or f"Table {i + 1}")[:40]
                log_lines.append(f"[{i + 1}/{total}] {title}…")
                # Stream status before starting work on this table
                yield (
                    all_tables,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    _agent_status_html(log_lines, i, total),
                )

                cals_xml = entry.get("xml", "")
                pdf_path = entry.get("pdf_path", "")
                page_idx = entry.get("page_idx", -1)
                img_path = entry.get("image_path", "")

                if not cals_xml:
                    log_lines[-1] += " ⚠ no XML"
                    continue

                # Step 1 — pdfplumber annotation (font name + x0 from LibreOffice PDF)
                # Pass pre-built cache so the PDF is not re-opened per table.
                pdf_xml = cals_xml
                if pdf_path and page_idx >= 0:
                    try:
                        pdf_xml = database._annotate_entry_styles_with_pdfplumber(
                            cals_xml, pdf_path, page_idx,
                            _page_cache=pdf_cache.get(pdf_path),
                        )
                    except Exception as _e:
                        log_lines[-1] += " ⚠ pdf-err"
                        print(f"[agent] table {i} pdfplumber: {_e}")

                # Step 2 — VLM annotation.
                # If this model already annotated this table (key present and non-empty)
                # reuse the cached result — preserves Qwen work when running Gemma and
                # vice versa.  Skip entirely if image is missing or LM Studio is down.
                vlm_xml = entry.get(_vlm_key) or None
                if vlm_xml:
                    method_tag = f"pdf+{_method_suffix}"
                    log_lines[-1] = (
                        f"[{i + 1}/{total}] {title} [{_vlm_key} cached] ✓"
                    )
                elif img_path and os.path.exists(img_path):
                    try:
                        vlm_xml = database._annotate_entry_styles_with_vlm(
                            cals_xml, img_path, lmstudio_url=database.VLM_LMSTUDIO_URL
                        )
                        method_tag = f"pdf+{_method_suffix}"
                    except Exception as _e:
                        method_tag = "pdf"
                        print(f"[agent] table {i} VLM skipped: {_e}")
                else:
                    method_tag = "pdf"

                # Step 3 — Compare and find conflicts (for review panel only).
                # We do NOT call _reconcile_with_llm here: the reconciliation
                # rules always say "trust pdfplumber for bold/indent", so a
                # third LLM inference per table would just confirm what
                # pdfplumber already produced. Skipping it saves ~63 inferences
                # (the dominant time cost). The Review tab still shows all
                # conflicts so they can be inspected manually.
                conflicts = database._compare_annotation_sets(pdf_xml, vlm_xml) if vlm_xml else []
                final_xml = pdf_xml  # pdfplumber is authoritative

                # Step 4 — Write back.
                # - xml / xml_pdfplumber are always updated (pdfplumber is authoritative).
                # - VLM result is written to the model-specific key only; other models'
                #   keys (e.g. xml_vlm for Qwen when running Gemma) are left untouched.
                all_tables[i]["xml"] = final_xml
                all_tables[i]["xml_pdfplumber"] = pdf_xml
                all_tables[i][_vlm_key] = vlm_xml or ""
                all_tables[i][_conflicts_key] = len(conflicts)
                all_tables[i]["annotation_version"] = method_tag

                # Persist catalog every 5 tables (not every table) to reduce I/O;
                # always write on the last table so nothing is lost.
                if (i + 1) % 5 == 0 or i == total - 1:
                    try:
                        with open(catalog_path, "w", encoding="utf-8") as _f:
                            _json.dump(all_tables, _f, ensure_ascii=False, indent=2)
                    except Exception as _e:
                        print(f"[agent] catalog save: {_e}")

                conflict_info = (
                    f" {len(conflicts)} conflicts" if conflicts else ""
                )
                log_lines[-1] = (
                    f"[{i + 1}/{total}] {title} [{method_tag}]{conflict_info} ✓"
                )
                # Yield after each table so the ✓ is visible immediately in the UI
                # (without this, the completion marker only shows on the next
                # iteration's pre-work yield — meaning the last table never shows ✓
                # unless the post-loop yield fires successfully).
                yield (
                    all_tables,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    _agent_status_html(log_lines, i + 1, total),
                )

            # Re-render the currently selected table with updated annotations.
            # Wrap in try/except so a rendering error never prevents the
            # completion message from reaching the UI.
            try:
                show_outs = _show_table(selected_idx, all_tables, theme)
            except Exception as _e:
                print(f"[agent] final _show_table: {_e}")
                show_outs = tuple(gr.update() for _ in range(11))
            log_lines.append(f"✓ Agent complete — {total} tables processed.")
            yield (all_tables, *show_outs, _agent_status_html(log_lines, total, total))

        tb_vlm_annotate_btn.click(
            _reannotate_all_with_vlm,
            inputs=[all_tables_state, table_selector, tb_cals_theme],
            outputs=[
                all_tables_state,
                tb_plain_text, tb_cals_xml, tb_inspection, tb_orig_img,
                tb_cals_html, tb_diff_img, tb_diff_ts, tb_cals_xml_view,
                tb_orig_img_r, current_table_state,
                tb_annot_review,
                tb_vlm_status,
            ],
        )

        tb_agent_btn.click(
            _run_annotation_agent,
            inputs=[all_tables_state, table_selector, tb_cals_theme],
            outputs=[
                all_tables_state,
                tb_plain_text, tb_cals_xml, tb_inspection, tb_orig_img,
                tb_cals_html, tb_diff_img, tb_diff_ts, tb_cals_xml_view,
                tb_orig_img_r, current_table_state,
                tb_annot_review,
                tb_agent_status,
            ],
        )

        table_selector.change(
            _show_table,
            inputs=[table_selector, all_tables_state, tb_cals_theme],
            outputs=[tb_plain_text, tb_cals_xml, tb_inspection, tb_orig_img, tb_cals_html, tb_diff_img, tb_diff_ts, tb_cals_xml_view, tb_orig_img_r, current_table_state, tb_annot_review],
        )
        tb_list_click.change(
            _show_table,
            inputs=[tb_list_click, all_tables_state, tb_cals_theme],
            outputs=[tb_plain_text, tb_cals_xml, tb_inspection, tb_orig_img, tb_cals_html, tb_diff_img, tb_diff_ts, tb_cals_xml_view, tb_orig_img_r, current_table_state, tb_annot_review],
        )

        def _show_para_page(seq_val):
            """Show the page image for a paragraph row clicked in the list view."""
            import json as _json
            _noop = gr.update()
            if seq_val is None or seq_val < 0:
                return _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, {}, _noop
            segments_path = os.path.join(database.DATA_DIR, "doc_segments.json")
            if not os.path.exists(segments_path):
                return _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, {}, _noop
            try:
                with open(segments_path, encoding="utf-8") as _f:
                    segments = _json.load(_f)
            except Exception:
                return _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, {}, _noop
            seg = next((s for s in segments if s.get("seq") == int(seq_val) and s.get("type") == "paragraph"), None)
            if seg is None:
                return _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, _noop, {}, _noop
            img_path   = seg.get("image_path", "") or None
            annot_path = seg.get("annotated_image_path", "") or None
            img = next((p for p in [annot_path, img_path] if p and os.path.exists(p)), None)
            img = database._draw_bbox_on_image(img, seg.get("bbox"), color="red")
            text = seg.get("text", "")
            empty_html = "<p style='color:#888;font-family:sans-serif;'>No XML — text paragraph</p>"
            return (
                gr.update(value=text),    # tb_plain_text
                gr.update(value=""),      # tb_cals_xml
                gr.update(value={}),      # tb_inspection
                gr.update(value=img),     # tb_orig_img
                gr.update(value=empty_html),  # tb_cals_html
                gr.update(value=img),     # tb_diff_img
                gr.update(value=""),      # tb_diff_ts
                gr.update(value=empty_html),  # tb_cals_xml_view
                gr.update(value=img),     # tb_orig_img_r
                {},                        # current_table_state
                gr.update(value=""),      # tb_annot_review
            )

        tb_para_click.change(
            _show_para_page,
            inputs=[tb_para_click],
            outputs=[tb_plain_text, tb_cals_xml, tb_inspection, tb_orig_img, tb_cals_html, tb_diff_img, tb_diff_ts, tb_cals_xml_view, tb_orig_img_r, current_table_state, tb_annot_review],
        )

        def _rerender_cals(xml_str, table_meta, theme="verify"):
            """Live re-render rendered table + XML view + diff for the focused table only."""
            empty_html = "<p style='color:#888;font-family:sans-serif;'>No XML</p>"
            if not xml_str or not xml_str.strip():
                return gr.update(value=empty_html), gr.update(value=empty_html), gr.update(value=None), gr.update(value="")
            try:
                # Re-verify against the original PDF (scoped to this table only)
                annotated_xml, new_diff = database._recheck_xml(xml_str, table_meta or {})
                table_html, xml_panel_html = database._cals_to_interactive_html(annotated_xml, theme=theme, clean_xml=xml_str)
                diff_update = gr.update(value=new_diff) if new_diff else gr.update()
                ts_update = gr.update(value=_diff_ts_html(new_diff)) if new_diff else gr.update(value="")
                return gr.update(value=table_html), gr.update(value=xml_panel_html), diff_update, ts_update
            except Exception as e:
                err = f"<p style='color:red;font-family:monospace;font-size:11px;'>XML error: {e}</p>"
                return gr.update(value=err), gr.update(value=err), gr.update(), gr.update(value="")

        tb_cals_xml.change(
            _rerender_cals,
            inputs=[tb_cals_xml, current_table_state, tb_cals_theme],
            outputs=[tb_cals_html, tb_cals_xml_view, tb_diff_img, tb_diff_ts],
        )
        tb_cals_theme.change(
            _rerender_cals,
            inputs=[tb_cals_xml, current_table_state, tb_cals_theme],
            outputs=[tb_cals_html, tb_cals_xml_view, tb_diff_img, tb_diff_ts],
        )

        def _save_xml(xml_str, all_tables, selected_idx):
            updated, msg = database.save_table_xml(all_tables, selected_idx, xml_str)
            color = "#1a7a1a" if msg.startswith("✅") else "#7a1a1a"
            status_html = (
                f"<p style='font-family:sans-serif;font-size:12px;color:{color};"
                f"margin:4px 0;'>{msg}</p>"
            )
            return updated, status_html

        tb_save_btn.click(
            _save_xml,
            inputs=[tb_cals_xml, all_tables_state, table_selector],
            outputs=[all_tables_state, tb_save_status],
        )

        def _show_transform_status(selected_idx, all_tables):
            """Reflect the stored transform for the current table in the transform controls."""
            if selected_idx is None or not all_tables or selected_idx >= len(all_tables):
                return gr.update(value=""), gr.update(value="none"), gr.update(value="EBF3FB")
            tr     = (all_tables[int(selected_idx)].get("transform") or {})
            ttype  = tr.get("type",  "none") or "none"
            color  = tr.get("color", "EBF3FB") or "EBF3FB"
            if ttype == "none":
                return gr.update(value=""), gr.update(value="none"), gr.update(value="EBF3FB")
            _cnames = {
                "EBF3FB": "light blue",  "E8F5E9": "light green",
                "FFFDE7": "light yellow", "EDE7F6": "lavender", "FCE4EC": "light pink",
            }
            badge = (
                f"<span style='font-family:sans-serif;font-size:11px;color:#2ecc71;'>"
                f"\u2705 Transform: <b>{ttype}</b> \u2013 {_cnames.get(color, color)}</span>"
            )
            return gr.update(value=badge), gr.update(value=ttype), gr.update(value=color)

        def _apply_table_transform(transform_type, color, all_tables, selected_idx):
            """Store (or clear) a style transform in the selected table's catalog entry."""
            import json as _json, copy as _copy
            if selected_idx is None or not all_tables or selected_idx >= len(all_tables):
                return (
                    all_tables,
                    "<span style='color:#c00;font-family:sans-serif;font-size:11px;'>"
                    "No table selected.</span>",
                )
            idx     = int(selected_idx)
            updated = _copy.deepcopy(all_tables)
            _cnames = {
                "EBF3FB": "light blue",  "E8F5E9": "light green",
                "FFFDE7": "light yellow", "EDE7F6": "lavender", "FCE4EC": "light pink",
            }
            if transform_type == "none":
                updated[idx].pop("transform", None)
                badge = "<span style='font-family:sans-serif;font-size:11px;color:#888;'>Transform cleared.</span>"
            else:
                updated[idx]["transform"] = {
                    "type":  transform_type,
                    "color": (color or "EBF3FB").lstrip("#").upper(),
                }
                badge = (
                    f"<span style='font-family:sans-serif;font-size:11px;color:#2ecc71;'>"
                    f"\u2705 <b>{transform_type}</b> ({_cnames.get(color, color)}) set on "
                    f"table {idx + 1}. Will apply on .docx export.</span>"
                )
            # Persist so transforms survive a page reload
            cat_path = os.path.join(database.DATA_DIR, "table_catalog.json")
            try:
                with open(cat_path, "w", encoding="utf-8") as _f:
                    _json.dump(updated, _f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return updated, badge

        tb_apply_transform_btn.click(
            _apply_table_transform,
            inputs=[tb_transform_dd, tb_transform_color_dd, all_tables_state, table_selector],
            outputs=[all_tables_state, tb_transform_status],
        )
        tb_clear_transform_btn.click(
            lambda at, si: _apply_table_transform("none", "EBF3FB", at, si),
            inputs=[all_tables_state, table_selector],
            outputs=[all_tables_state, tb_transform_status],
        )
        # Reflect stored transform when navigating between tables
        table_selector.change(
            _show_transform_status,
            inputs=[table_selector, all_tables_state],
            outputs=[tb_transform_status, tb_transform_dd, tb_transform_color_dd],
        )
        tb_list_click.change(
            _show_transform_status,
            inputs=[tb_list_click, all_tables_state],
            outputs=[tb_transform_status, tb_transform_dd, tb_transform_color_dd],
        )

        def _handle_entry_edit(signal, all_tables, selected_idx, table_meta, theme):
            """Process inline popup editor save: update one entry, re-verify, re-render."""
            import copy as _copy
            empty_html = "<p style='color:#888;font-family:sans-serif;'>No XML</p>"
            if not signal or "||" not in signal:
                return gr.update(), gr.update(), gr.update(), all_tables
            eid, new_text = signal.split("||", 1)
            eid = eid.strip()
            updated, new_clean_xml, msg = database.update_entry_text(
                all_tables, selected_idx, eid, new_text
            )
            if not new_clean_xml:
                return gr.update(), gr.update(), gr.update(), all_tables
            try:
                annotated_xml, _ = database._recheck_xml(new_clean_xml, table_meta or {})
            except Exception:
                annotated_xml = new_clean_xml
            try:
                table_html, xml_panel_html = database._cals_to_interactive_html(
                    annotated_xml, theme=theme or "verify", clean_xml=new_clean_xml
                )
            except Exception:
                table_html = xml_panel_html = empty_html
            return (
                gr.update(value=table_html),
                gr.update(value=xml_panel_html),
                gr.update(value=new_clean_xml),
                updated,
            )

        tb_entry_edit.change(
            _handle_entry_edit,
            inputs=[tb_entry_edit, all_tables_state, table_selector, current_table_state, tb_cals_theme],
            outputs=[tb_cals_html, tb_cals_xml_view, tb_cals_xml, all_tables_state],
        )

        def _render_fop(xml_str, table_meta, theme="verify"):
            """Re-verify XML against PDF, render through Apache FOP, and refresh all panels."""
            import xml.etree.ElementTree as _ET
            empty_html = "<p style='color:#888;font-family:sans-serif;'>No XML to render.</p>"
            if not xml_str or not xml_str.strip():
                return gr.update(value=empty_html), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="")
            try:
                annotated_xml, new_diff = database._recheck_xml(xml_str, table_meta or {})
            except Exception:
                annotated_xml, new_diff = xml_str, None
            # Re-render interactive HTML with updated verify= colours
            try:
                table_html, xml_panel_html = database._cals_to_interactive_html(annotated_xml, theme=theme, clean_xml=xml_str)
            except Exception:
                table_html = empty_html
                xml_panel_html = empty_html
            # FOP PDF render — clean version (no verify= colours) for the iframe preview
            try:
                root_clean = _ET.fromstring(annotated_xml)
                for e in root_clean.iter("entry"):
                    e.attrib.pop("verify", None)
                    e.attrib.pop("verify-reason", None)
                database._indent_xml(root_clean)
                clean_xml_for_fop = _ET.tostring(root_clean, encoding="unicode")
            except Exception:
                clean_xml_for_fop = xml_str
            # Match FOP page size to the source Word PDF page, if available
            _fop_pw, _fop_ph = None, None
            try:
                _tm = table_meta or {}
                _wp = _tm.get("pdf_path", "")
                _wi = _tm.get("page_idx", -1)
                if _wp and os.path.exists(_wp) and _wi >= 0:
                    import pdfplumber as _pplumber
                    with _pplumber.open(_wp) as _pdoc:
                        if _wi < len(_pdoc.pages):
                            _pg = _pdoc.pages[_wi]
                            _PT_TO_MM = 25.4 / 72.0
                            _fop_pw = round(_pg.width  * _PT_TO_MM, 1)
                            _fop_ph = round(_pg.height * _PT_TO_MM, 1)
            except Exception as _pe:
                print(f"[render_fop] page size read failed: {_pe}")
            fop_b64, fop_err = database._cals_to_fop_pdf(clean_xml_for_fop, _fop_pw, _fop_ph, theme=theme)
            if fop_err:
                fop_html = (
                    "<div style='font-family:monospace;font-size:11px;color:#c00;"
                    "background:#fff8f8;padding:10px;border:1px solid #f00;"
                    f"border-radius:4px;white-space:pre-wrap;'><b>FOP error:</b>\n{fop_err}</div>"
                )
            else:
                fop_html = (
                    f'<iframe src="data:application/pdf;base64,{fop_b64}" '
                    'style="width:100%;height:700px;border:1px solid #ccc;border-radius:4px;" '
                    'title="FOP-rendered PDF preview"></iframe>'
                )
            # FOP-vs-Word diff image:
            #   LEFT  — FOP rendered WITH verify= colours (green=confirmed, red=unconfirmed)
            #   RIGHT — original Word→PDF page (raw, no annotation overlay)
            fop_b64_annotated, _ = database._cals_to_fop_pdf(annotated_xml, _fop_pw, _fop_ph, theme=theme)
            word_img_path = (table_meta or {}).get("image_path", "")
            fop_diff_path = None
            if fop_b64_annotated:
                try:
                    fop_diff_path = database._fop_vs_word_diff_image(fop_b64_annotated, word_img_path)
                except Exception as _de:
                    print(f"[render_fop] diff image failed: {_de}")
            diff_img = fop_diff_path or new_diff
            return (
                gr.update(value=fop_html),       # tb_fop_html
                gr.update(value=table_html),     # tb_cals_html  — red/green boxes updated
                gr.update(value=xml_panel_html), # tb_cals_xml_view — XML panel updated
                gr.update(value=annotated_xml),  # tb_cals_xml  — editor updated with verify= attrs
                gr.update(value=diff_img) if diff_img else gr.update(),  # tb_diff_img
                gr.update(value=_diff_ts_html(diff_img)),                # tb_diff_ts
            )

        tb_fop_btn.click(
            _render_fop,
            inputs=[tb_cals_xml, current_table_state, tb_cals_theme],
            outputs=[tb_fop_html, tb_cals_html, tb_cals_xml_view, tb_cals_xml, tb_diff_img, tb_diff_ts],
        )

        def _validate_current_table(xml_current, all_tables, selected_idx):
            """Compare the current editor XML against the catalog (original) snapshot using TEDS."""
            if not xml_current or not xml_current.strip():
                return "<span style='color:#c0392b;font-family:monospace;font-size:12px;'>No XML in editor.</span>"

            # Resolve catalog snapshot for the selected table
            try:
                idx = int(selected_idx) if selected_idx is not None else -1
            except (TypeError, ValueError):
                idx = -1
            xml_original = None
            if all_tables and 0 <= idx < len(all_tables):
                xml_original = all_tables[idx].get("xml", "")

            if not xml_original:
                # No catalog snapshot — just parse the editor XML and report its shape
                try:
                    result = database.compare_snapshots(xml_current, xml_current)
                    return (
                        "<span style='color:#f39c12;font-family:monospace;font-size:12px;'>"
                        "&#9888; No catalog snapshot found — comparing XML to itself."
                        "</span>"
                    )
                except Exception as exc:
                    return (
                        f"<span style='color:#c0392b;font-family:monospace;font-size:12px;'>"
                        f"Error: {exc}</span>"
                    )

            try:
                result = database.compare_snapshots(xml_original, xml_current)
            except Exception as exc:
                return (
                    f"<span style='color:#c0392b;font-family:monospace;font-size:12px;'>"
                    f"compare_snapshots error: {exc}</span>"
                )

            if "error" in result:
                return (
                    f"<span style='color:#c0392b;font-family:monospace;font-size:12px;'>"
                    f"{result['error']}</span>"
                )

            verdict = result["verdict"]
            verdict_colour = {"PASS": "#2ecc71", "WARN": "#f39c12", "FAIL": "#c0392b"}.get(verdict, "#888")
            verdict_icon  = {"PASS": "&#10003;", "WARN": "&#9888;", "FAIL": "&#10007;"}.get(verdict, "?")

            details = []
            if result["lost_bold"]:
                details.append(f"<b>Lost bold</b>: {', '.join(result['lost_bold'][:5])}")
            if result["gained_bold"]:
                details.append(f"<b>Gained bold</b>: {', '.join(result['gained_bold'][:5])}")
            if result["indent_changed"]:
                ic = result["indent_changed"][:3]
                details.append("<b>Indent changed</b>: " + "; ".join(f"{t} ({a}→{b})" for t, a, b in ic))
            if result["missing_values"]:
                details.append(f"<b>Missing values</b>: {', '.join(result['missing_values'][:5])}")
            if result["span_changed"]:
                sc = result["span_changed"][:3]
                details.append("<b>Span changed</b>: " + "; ".join(f"{t}" for t, _, _ in sc))

            detail_html = ("<br>".join(details)) if details else "No per-cell differences detected."

            return (
                f"<div style='font-family:monospace;font-size:12px;line-height:1.6;padding:6px 0;'>"
                f"<span style='color:{verdict_colour};font-size:14px;font-weight:bold;'>"
                f"{verdict_icon} {verdict}</span>&nbsp;&nbsp;"
                f"TEDS full=<b>{result['teds_full']:.3f}</b> &nbsp;"
                f"struct=<b>{result['teds_struct']:.3f}</b> &nbsp;"
                f"(nodes: orig={result['tree_sizes']['a']} editor={result['tree_sizes']['b']})"
                f"<br>{detail_html}"
                f"</div>"
            )

        tb_validate_btn.click(
            _validate_current_table,
            inputs=[tb_cals_xml, all_tables_state, table_selector],
            outputs=[tb_validate_status],
        )

        def _run_fop_verification(xml_current, table_meta, theme):
            """Render the current CALS XML through Apache FOP, re-extract with
            pdfplumber, run TEDS comparison, and return a rich HTML report."""

            _SPIN = (
                "<span style='color:#7ec8e3;font-family:monospace;font-size:12px;'>"
                "&#9203; Running FOP render + pdfplumber extraction + TEDS…</span>"
            )

            if not xml_current or not xml_current.strip():
                return (
                    "<span style='color:#c0392b;font-family:monospace;font-size:12px;'>"
                    "No XML in editor — select a table first.</span>"
                )

            # Resolve source-PDF page size from table metadata
            pw, ph = None, None
            try:
                tm = table_meta or {}
                wp = tm.get("pdf_path", "")
                wi = tm.get("page_idx", -1)
                if wp and os.path.exists(wp) and wi >= 0:
                    import pdfplumber as _ppl
                    with _ppl.open(wp) as _pd:
                        if wi < len(_pd.pages):
                            _pg = _pd.pages[wi]
                            PT = 25.4 / 72.0
                            pw = round(_pg.width  * PT, 1)
                            ph = round(_pg.height * PT, 1)
            except Exception as _pe:
                print(f"[fop_verify] page-size read failed: {_pe}")

            result = database.fop_teds_verify(xml_current, pw, ph, theme=theme)

            if "error" in result:
                return (
                    f"<div style='font-family:monospace;font-size:12px;color:#c0392b;padding:6px 0;'>"
                    f"<b>&#10007; FOP Verification failed</b><br>{result['error']}</div>"
                )

            verdict        = result["verdict"]
            teds_full      = result["teds_full"]
            teds_struct    = result["teds_struct"]
            lost_bold      = result["lost_bold"]
            gained_bold    = result["gained_bold"]
            indent_changed = result["indent_changed"]
            missing_values = result["missing_values"]
            extra_values   = result["extra_values"]
            span_changed   = result["span_changed"]
            tree_a         = result["tree_sizes"]["a"]
            tree_b         = result["tree_sizes"]["b"]
            n_pages        = result.get("fop_pdf_pages", "?")
            fop_rows       = result.get("fop_rows", "?")
            fop_cols       = result.get("fop_cols", "?")

            vc = {"PASS": "#2ecc71", "WARN": "#f39c12", "FAIL": "#c0392b"}.get(verdict, "#888")
            vi = {"PASS": "&#10003;", "WARN": "&#9888;", "FAIL": "&#10007;"}.get(verdict, "?")

            # ---- score bar ----
            def _bar(score):
                pct = int(score * 100)
                col = "#2ecc71" if score >= 0.95 else "#f39c12" if score >= 0.80 else "#c0392b"
                return (
                    f"<div style='display:inline-block;width:120px;background:#2a2a3a;"
                    f"border-radius:3px;height:10px;vertical-align:middle;margin:0 4px 0 2px;'>"
                    f"<div style='width:{pct}%;background:{col};height:10px;border-radius:3px;'></div>"
                    f"</div><b>{score:.3f}</b>"
                )

            # ---- section builder ----
            def _section(title, items, fmt_fn, colour="#f0f0f0"):
                if not items:
                    return ""
                n = len(items)
                rows_html = "".join(
                    f"<tr><td style='padding:2px 8px;color:{colour};font-family:monospace;font-size:11px;'>{fmt_fn(x)}</td></tr>"
                    for x in items[:10]
                )
                more = f"<tr><td style='color:#888;font-size:10px;padding:2px 8px;'>… and {n-10} more</td></tr>" if n > 10 else ""
                return (
                    f"<div style='margin-top:8px;'>"
                    f"<span style='font-weight:bold;color:#bbb;font-size:11px;'>{title} ({n})</span>"
                    f"<table style='border-collapse:collapse;margin-top:2px;width:100%;'>{rows_html}{more}</table>"
                    f"</div>"
                )

            lost_bold_sec   = _section("&#128308; Lost bold",    lost_bold,      lambda x: x,                        "#e74c3c")
            gained_bold_sec = _section("&#128994; Gained bold",  gained_bold,    lambda x: x,                        "#2ecc71")
            indent_sec      = _section("&#8649; Indent changed", indent_changed, lambda x: f"{x[0]} &nbsp;<span style='color:#888'>({x[1]}→{x[2]})</span>", "#f39c12")
            missing_sec     = _section("&#10007; Missing values",missing_values, lambda x: x,                        "#e74c3c")
            extra_sec       = _section("&#43; Extra values",     extra_values,   lambda x: x,                        "#3498db")
            span_sec        = _section("&#9633; Span changed",   span_changed,
                                       lambda x: f"{x[0]} &nbsp;<span style='color:#888'>({x[1][0]}x{x[1][1]}→{x[2][0]}x{x[2][1]})</span>",
                                       "#9b59b6")

            html = (
                f"<div style='font-family:sans-serif;font-size:12px;line-height:1.7;"
                f"background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:12px 16px;margin-top:6px;'>"

                # --- Header row ---
                f"<div style='margin-bottom:8px;'>"
                f"<span style='font-size:16px;font-weight:bold;color:{vc};'>{vi} {verdict}</span>"
                f"&nbsp;&nbsp;<span style='color:#888;font-size:11px;'>FOP round-trip TEDS</span>"
                f"</div>"

                # --- Score table ---
                f"<table style='border-collapse:collapse;width:100%;margin-bottom:4px;'>"
                f"<tr>"
                f"<td style='color:#aaa;padding:2px 12px 2px 0;font-size:11px;'>TEDS full</td>"
                f"<td>{_bar(teds_full)}</td>"
                f"<td style='color:#555;font-size:10px;padding-left:10px;'>structure + content + bold/indent</td>"
                f"</tr>"
                f"<tr>"
                f"<td style='color:#aaa;padding:2px 12px 2px 0;font-size:11px;'>TEDS struct</td>"
                f"<td>{_bar(teds_struct)}</td>"
                f"<td style='color:#555;font-size:10px;padding-left:10px;'>span grid + row order only</td>"
                f"</tr>"
                f"</table>"

                # --- FOP metadata ---
                f"<div style='color:#666;font-size:10px;margin:4px 0 8px;'>"
                f"FOP PDF: {n_pages} page(s) &nbsp;|"
                f" pdfplumber: {fop_rows} rows &times; {fop_cols} cols &nbsp;|"
                f" nodes: orig={tree_a} &rarr; fop={tree_b}"
                f"</div>"

                # --- Verdict thresholds legend ---
                f"<div style='color:#555;font-size:10px;border-top:1px solid #2a2a3a;padding-top:4px;margin-bottom:4px;'>"
                f"PASS: full&ge;0.95 &amp; struct=1.0 &nbsp;|"
                f" WARN: full&ge;0.80 or struct&ge;0.95 &nbsp;|"
                f" FAIL: below WARN"
                f"</div>"

                # --- Per-dimension detail sections ---
                + lost_bold_sec
                + gained_bold_sec
                + indent_sec
                + missing_sec
                + extra_sec
                + span_sec

                + ("<div style='color:#555;font-size:11px;margin-top:8px;'>No per-cell differences detected.</div>"
                   if not any([lost_bold, gained_bold, indent_changed, missing_values, extra_values, span_changed])
                   else "")

                + "</div>"
            )
            return html

        tb_fop_verify_btn.click(
            _run_fop_verification,
            inputs=[tb_cals_xml, current_table_state, tb_cals_theme],
            outputs=[tb_fop_verify_status],
        )

        def _export_document_pdf(all_tables, theme):
            """Reconstruct all tables + paragraphs into a single A4 PDF via FOP."""
            import tempfile, base64 as _b64, os as _os

            # Use neutral "finance" or "minimal" theme for document output;
            # fall back to theme from UI but strip verify colours by passing
            # the catalog directly so reconstruct_to_pdf can strip verify= attrs.
            export_theme = theme if theme in ("finance", "minimal", "ixbrl", "striped") else "finance"

            status_running = (
                "<span style='font-family:sans-serif;font-size:12px;color:#555;'>"
                "⏳ Generating full-document PDF…</span>"
            )
            yield (
                gr.update(value=status_running),
                gr.update(visible=False, value=None),
            )

            b64_pdf, err = database.reconstruct_to_pdf(
                table_catalog=all_tables if all_tables else None,
                theme=export_theme,
            )

            if err or not b64_pdf:
                err_html = (
                    "<span style='font-family:sans-serif;font-size:12px;color:#c00;'>"
                    f"❌ Export failed: {err or 'unknown error'}</span>"
                )
                yield gr.update(value=err_html), gr.update(visible=False, value=None)
                return

            # Write PDF to a named temp file so Gradio can serve it for download
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", prefix="document_export_"
            )
            tmp.write(_b64.b64decode(b64_pdf))
            tmp.close()

            ok_html = (
                "<span style='font-family:sans-serif;font-size:12px;color:#080;'>"
                "✅ PDF ready — click the file below to download.</span>"
            )
            yield gr.update(value=ok_html), gr.update(visible=True, value=tmp.name)

        def _export_document_docx(all_tables):
            """Write edited table XML back into the original .docx and offer for download."""
            yield (
                gr.update(value="<span style='font-family:sans-serif;font-size:12px;color:#555;'>⏳ Writing tables back to .docx\u2026</span>"),
                gr.update(visible=False, value=None),
            )
            out_path, status_msg = database.write_back_to_docx(
                table_catalog=all_tables if all_tables else None,
            )
            if out_path is None:
                yield (
                    gr.update(value=f"<span style='font-family:sans-serif;font-size:12px;color:#c00;'>\u274c {status_msg}</span>"),
                    gr.update(visible=False, value=None),
                )
                return
            ok_html = (
                f"<span style='font-family:sans-serif;font-size:12px;color:#080;'>"
                f"{status_msg}</span>"
            )
            yield gr.update(value=ok_html), gr.update(visible=True, value=out_path)

        # ── Export preview helpers ────────────────────────────────────────────
        def _build_export_preview_html(data: dict, export_type: str = "pdf") -> str:
            """Build styled HTML for the export review panel."""
            import html as _h
            if "error" in data:
                return (
                    "<div style='background:#1a1a2e;border:1px solid #5c2a2a;border-radius:8px;"
                    "padding:14px;font-family:sans-serif;font-size:12px;color:#e74c3c;'>"
                    + _h.escape(data["error"]) + "</div>"
                )

            ts       = data.get("text_stats", {})
            trows    = data.get("table_rows", [])
            n_total  = data.get("total_tables", 0)
            n_edited = data.get("edited_tables", 0)
            n_trans  = data.get("transformed_tables", 0)
            type_icon  = "\U0001f4c4" if export_type == "pdf" else "\U0001f4dd"
            type_label = "Full Document PDF" if export_type == "pdf" else "Edited .docx"

            # TEDS score bar
            def _bar(score, self_baseline=False):
                if score is None:
                    return "<span style='color:#555;font-size:10px;'>N/A</span>"
                pct = int(score * 100)
                col = "#2ecc71" if score >= 0.95 else "#f39c12" if score >= 0.80 else "#e74c3c"
                dagger = (
                    "<span title='No pdfplumber baseline \u2014 compared against original extracted XML'"
                    " style='color:#888;font-size:9px;margin-left:2px;'>\u2020</span>"
                    if self_baseline else ""
                )
                return (
                    "<div style='display:inline-flex;align-items:center;gap:4px;'>"
                    "<div style='width:50px;background:#2a2a3e;border-radius:2px;height:7px;'>"
                    f"<div style='width:{pct}%;background:{col};height:7px;border-radius:2px;'></div></div>"
                    f"<span style='font-size:10px;color:{col};font-weight:bold;'>{score:.3f}</span>"
                    + dagger +
                    "</div>"
                )

            # stat row for text block
            def _srow(label, val):
                return (
                    "<tr>"
                    f"<td style='color:#aaa;padding:3px 16px 3px 0;font-size:11px;'>{label}</td>"
                    f"<td style='color:#eee;font-family:monospace;font-weight:bold;font-size:11px;'>{val:,}</td>"
                    "</tr>"
                )

            text_table = (
                "<table style='border-collapse:collapse;'>"
                + _srow("Total characters",               ts.get("char_count",  0))
                + _srow("Total words",                    ts.get("word_count",  0))
                + _srow("Whitespace characters",          ts.get("ws_count",    0))
                + _srow("Other chars (punct / symbols)",  ts.get("other_count", 0))
                + "</table>"
            )

            SEG_LABEL = (
                f"{ts.get('seg_count', 0)} paragraph"
                + ("s" if ts.get('seg_count', 1) != 1 else "")
                + " \u2014 read-only from source"
            )

            color_names = {
                "EBF3FB": "light blue",  "E8F5E9": "light green",
                "FFFDE7": "light yellow", "EDE7F6": "lavender", "FCE4EC": "light pink",
            }

            tbody_html = ""
            for r in trows:
                edited_cell = (
                    "<span style='background:#1a3a60;color:#7ec8e3;border:1px solid #3a6a9a;"
                    "border-radius:3px;padding:1px 4px;font-size:9px;font-weight:700;'>\u270f edited</span>"
                    if r["edited"] else
                    "<span style='color:#444;font-size:10px;'>\u2014</span>"
                )
                if r["has_transform"]:
                    cname = color_names.get(r.get("transform_color", ""), r.get("transform_color", ""))
                    label = r["transform_type"] + (f" ({cname})" if cname else "")
                    tr_cell = (
                        "<span style='background:#1a3a1a;color:#2ecc71;border:1px solid #2a6a2a;"
                        "border-radius:3px;padding:1px 4px;font-size:9px;font-weight:700;'>"
                        "\U0001f3a8 " + _h.escape(label) + "</span>"
                    )
                else:
                    tr_cell = "<span style='color:#444;font-size:10px;'>\u2014</span>"
                title_esc = _h.escape(r["title"])
                tbody_html += (
                    "<tr style='border-bottom:1px solid #252535;'>"
                    f"<td style='color:#666;padding:5px 8px;font-size:10px;text-align:right;'>{r['index'] + 1}</td>"
                    f"<td style='color:#ddd;padding:5px 10px;font-size:11px;max-width:210px;"
                    f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title='{title_esc}'>{title_esc}</td>"
                    f"<td style='padding:5px 8px;text-align:center;'>{edited_cell}</td>"
                    f"<td style='padding:5px 8px;'>{tr_cell}</td>"
                    f"<td style='padding:5px 8px;'>{_bar(r['teds_full'],  r.get('baseline_is_self', False))}</td>"
                    f"<td style='padding:5px 8px;'>{_bar(r['teds_struct'], r.get('baseline_is_self', False))}</td>"
                    "</tr>"
                )

            summary_parts = [f"<b style='color:#eee;'>{n_total}</b>&nbsp;tables"]
            if n_edited:
                summary_parts.append(f"<b style='color:#7ec8e3;'>{n_edited}</b>&nbsp;edited")
            if n_trans:
                summary_parts.append(f"<b style='color:#2ecc71;'>{n_trans}</b>&nbsp;transformed")

            HDR = "border-bottom:1px solid #2a2a4a;padding-bottom:4px;margin-bottom:8px;"
            SEC_HDR = (
                "font-size:11px;font-weight:700;text-transform:uppercase;"
                "letter-spacing:.07em;color:#7ec8e3;"
            )

            return (
                "<div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:8px;"
                "padding:16px 20px;margin:6px 0;font-family:sans-serif;'>"

                # Header
                "<div style='display:flex;justify-content:space-between;align-items:center;"
                "margin-bottom:14px;'>"
                f"<span style='color:#7ec8e3;font-size:14px;font-weight:bold;'>"
                f"\U0001f4cb Export Preview &nbsp;\u2014&nbsp; {type_icon} {_h.escape(type_label)}</span>"
                "<span style='color:#555;font-size:10px;'>Review changes before confirming export</span>"
                "</div>"

                # Text segments
                f"<div style='margin-bottom:16px;'>"
                f"<div style='{SEC_HDR}{HDR}'>"
                f"Text Segments &nbsp;<span style='color:#555;font-weight:400;text-transform:none;font-size:10px;'>{_h.escape(SEG_LABEL)}</span></div>"
                + text_table +
                "</div>"

                # Table segments
                "<div>"
                f"<div style='{SEC_HDR}{HDR}'>"
                "Table Segments &nbsp;"
                f"<span style='color:#555;font-weight:400;text-transform:none;font-size:10px;'>"
                + "&nbsp;\u00b7&nbsp;".join(summary_parts) +
                "</span></div>"
                "<div style='overflow-x:auto;'>"
                "<table style='border-collapse:collapse;width:100%;'>"
                "<thead><tr style='background:#1e1e2e;'>"
                "<th style='color:#555;padding:4px 8px;font-size:10px;text-align:right;font-weight:600;'>#</th>"
                "<th style='color:#555;padding:4px 10px;font-size:10px;text-align:left;font-weight:600;'>Title</th>"
                "<th style='color:#555;padding:4px 8px;font-size:10px;font-weight:600;'>Edited</th>"
                "<th style='color:#555;padding:4px 8px;font-size:10px;text-align:left;font-weight:600;'>Transform</th>"
                "<th style='color:#555;padding:4px 8px;font-size:10px;font-weight:600;' "
                "title='TEDS full vs pdfplumber baseline'>TEDS full</th>"
                "<th style='color:#555;padding:4px 8px;font-size:10px;font-weight:600;' "
                "title='TEDS structural (span grid only)'>TEDS struct</th>"
                "</tr></thead>"
                f"<tbody>{tbody_html}</tbody>"
                "</table></div>"
                "<div style='color:#444;font-size:10px;margin-top:6px;'>"
                "TEDS vs pdfplumber baseline &nbsp;\u2014&nbsp; "
                "\u2265\u202f0.95&thinsp;<span style='color:#2ecc71;'>\u25cf</span>&ensp;"
                "\u2265\u202f0.80&thinsp;<span style='color:#f39c12;'>\u25cf</span>&ensp;"
                "&lt;\u202f0.80&thinsp;<span style='color:#e74c3c;'>\u25cf</span>&ensp;"
                "\u2020\u202f=\u202fno pdfplumber baseline, compared against original extracted XML"
                "</div></div></div>"
            )

        _EXPORT_SPINNER = (
            "<div style='background:#1a1a2e;border:1px solid #3a3a5c;border-radius:8px;"
            "padding:14px 18px;margin:6px 0;font-family:sans-serif;font-size:12px;color:#7ec8e3;'>"
            "\u23f3 Computing export preview\u2026</div>"
        )

        def _preview_for_pdf(all_tables, theme):
            """Show the export preview panel when the user clicks Export PDF."""
            yield (
                gr.update(value=_EXPORT_SPINNER, visible=True),
                gr.update(visible=True),   # confirm_pdf_btn
                gr.update(visible=False),  # confirm_docx_btn
                gr.update(visible=True),   # cancel_btn
            )
            data = database.compute_export_preview(table_catalog=all_tables or None)
            html = _build_export_preview_html(data, "pdf")
            yield (
                gr.update(value=html, visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        def _preview_for_docx(all_tables):
            """Show the export preview panel when the user clicks Export .docx."""
            yield (
                gr.update(value=_EXPORT_SPINNER, visible=True),
                gr.update(visible=False),  # confirm_pdf_btn
                gr.update(visible=True),   # confirm_docx_btn
                gr.update(visible=True),   # cancel_btn
            )
            data = database.compute_export_preview(table_catalog=all_tables or None)
            html = _build_export_preview_html(data, "docx")
            yield (
                gr.update(value=html, visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        def _confirm_export_pdf_fn(all_tables, theme):
            """Confirm + run full-document PDF export, hiding the review panel."""
            _hide = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            for out in _export_document_pdf(all_tables, theme):
                yield (*out, *_hide)

        def _confirm_export_docx_fn(all_tables):
            """Confirm + run .docx export, hiding the review panel."""
            _hide = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            for out in _export_document_docx(all_tables):
                yield (*out, *_hide)

        # ── Export button wiring ──────────────────────────────────────────────
        # First click → preview; Confirm → actual export
        tb_export_pdf_btn.click(
            _preview_for_pdf,
            inputs=[all_tables_state, tb_cals_theme],
            outputs=[tb_export_review_panel, tb_confirm_pdf_btn, tb_confirm_docx_btn, tb_cancel_export_btn],
        )
        tb_export_docx_btn.click(
            _preview_for_docx,
            inputs=[all_tables_state],
            outputs=[tb_export_review_panel, tb_confirm_pdf_btn, tb_confirm_docx_btn, tb_cancel_export_btn],
        )
        tb_confirm_pdf_btn.click(
            _confirm_export_pdf_fn,
            inputs=[all_tables_state, tb_cals_theme],
            outputs=[tb_export_pdf_status, tb_export_pdf_file,
                     tb_export_review_panel, tb_confirm_pdf_btn, tb_cancel_export_btn],
        )
        tb_confirm_docx_btn.click(
            _confirm_export_docx_fn,
            inputs=[all_tables_state],
            outputs=[tb_export_docx_status, tb_export_docx_file,
                     tb_export_review_panel, tb_confirm_docx_btn, tb_cancel_export_btn],
        )
        tb_cancel_export_btn.click(
            lambda: (
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
            ),
            inputs=[],
            outputs=[tb_export_review_panel, tb_confirm_pdf_btn,
                     tb_confirm_docx_btn, tb_cancel_export_btn],
        )

        def _toggle_model_tab():
            return {
                group_router: gr.update(visible=False),
                group_retrieval: gr.update(visible=False),
                group_generator: gr.update(visible=False),
                group_hallucination: gr.update(visible=False),
                group_answer: gr.update(visible=False),
                router_btn: gr.update(visible=True),
                retrieval_btn: gr.update(visible=True),
                generator_btn: gr.update(visible=True),
                hallucination_btn: gr.update(visible=True),
                answer_btn: gr.update(visible=True),
            }
        
        agent_settings.select(_toggle_model_tab, [], [group_router,
                                                      group_retrieval,
                                                      group_generator,
                                                      group_hallucination,
                                                      group_answer,
                                                      router_btn,
                                                      retrieval_btn,
                                                      generator_btn,
                                                      hallucination_btn,
                                                      answer_btn])

        """ This helper function ensures only one component model settings are expanded at a time when selected. """

        def _toggle_model(btn: str):
            if btn == "Router":
                group_visible = [True, False, False, False, False]
                button_visible = [False, True, True, True, True]
            elif btn == "Retrieval Grader":
                group_visible = [False, True, False, False, False]
                button_visible = [True, False, True, True, True]
            elif btn == "Generator":
                group_visible = [False, False, True, False, False]
                button_visible = [True, True, False, True, True]
            elif btn == "Hallucination Grader":
                group_visible = [False, False, False, True, False]
                button_visible = [True, True, True, False, True]
            elif btn == "Answer Grader":
                group_visible = [False, False, False, False, True]
                button_visible = [True, True, True, True, False]
            return {
                group_router: gr.update(visible=group_visible[0]),
                group_retrieval: gr.update(visible=group_visible[1]),
                group_generator: gr.update(visible=group_visible[2]),
                group_hallucination: gr.update(visible=group_visible[3]),
                group_answer: gr.update(visible=group_visible[4]),
                router_btn: gr.update(visible=button_visible[0]),
                retrieval_btn: gr.update(visible=button_visible[1]),
                generator_btn: gr.update(visible=button_visible[2]),
                hallucination_btn: gr.update(visible=button_visible[3]),
                answer_btn: gr.update(visible=button_visible[4]),
            }

        router_btn.click(_toggle_model, [router_btn], [group_router,
                                                       group_retrieval,
                                                       group_generator,
                                                       group_hallucination,
                                                       group_answer,
                                                       router_btn,
                                                       retrieval_btn,
                                                       generator_btn,
                                                       hallucination_btn,
                                                       answer_btn])
        
        retrieval_btn.click(_toggle_model, [retrieval_btn], [group_router,
                                                                           group_retrieval,
                                                                           group_generator,
                                                                           group_hallucination,
                                                                           group_answer,
                                                                           router_btn,
                                                                           retrieval_btn,
                                                                           generator_btn,
                                                                           hallucination_btn,
                                                                           answer_btn])
        
        generator_btn.click(_toggle_model, [generator_btn], [group_router,
                                                             group_retrieval,
                                                             group_generator,
                                                             group_hallucination,
                                                             group_answer,
                                                             router_btn,
                                                             retrieval_btn,
                                                             generator_btn,
                                                             hallucination_btn,
                                                             answer_btn])
        
        hallucination_btn.click(_toggle_model, [hallucination_btn], [group_router,
                                                                                   group_retrieval,
                                                                                   group_generator,
                                                                                   group_hallucination,
                                                                                   group_answer,
                                                                                   router_btn,
                                                                                   retrieval_btn,
                                                                                   generator_btn,
                                                                                   hallucination_btn,
                                                                                   answer_btn])
        
        answer_btn.click(_toggle_model, [answer_btn], [group_router,
                                                                     group_retrieval,
                                                                     group_generator,
                                                                     group_hallucination,
                                                                     group_answer,
                                                                     router_btn,
                                                                     retrieval_btn,
                                                                     generator_btn,
                                                                     hallucination_btn,
                                                                     answer_btn])

        """ This helper function builds out the submission function call when a user submits a query. """
        
        _my_build_stream = functools.partial(_stream_predict, client, app)

        msg.submit(
            _my_build_stream, [msg, 
                               model_generator,
                               model_router,
                               model_retrieval,
                               model_hallucination,
                               model_answer,
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               router_use_nim,
                               retrieval_use_nim,
                               generator_use_nim,
                               hallucination_use_nim,
                               answer_use_nim,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions, table_gallery, inspection_box, table_refs_state, optimize_btn, xml_box]
        )

        """ Optimize The Extraction button — runs multi-strategy comparison agent. """

        def _optimize_extraction(table_refs, progress=gr.Progress()):
            if not table_refs:
                return gr.update(
                    value={"error": "No table data available. Run a direct table query first."},
                    visible=True,
                )
            all_results = []
            for i, ref in enumerate(table_refs):
                progress((i + 1) / len(table_refs),
                         desc=f"Optimizing '{ref.get('title', '')[:30]}'...")
                result = database.optimize_extraction_agent(
                    title=ref.get("title", ""),
                    original_cell_rows=ref.get("cell_rows", []),
                    pdf_path=ref.get("pdf_path", ""),
                    page_idx=ref.get("page_idx") or 0,
                )
                all_results.append(result)
            return gr.update(value=all_results, visible=True)

        optimize_btn.click(
            _optimize_extraction,
            inputs=[table_refs_state],
            outputs=[optimization_box],
        )

    page.queue()
    return page

""" This helper function verifies that a user query is nonempty. """

def valid_input(query: str):
    return False if query.isspace() or query is None or query == "" or query == '' else True


""" This helper function provides error outputs for the query. """
def _get_query_error_message(e: Exception) -> str:
    if isinstance(e, GraphRecursionError):
        err = QUERY_ERROR_MESSAGES["GraphRecursionError"]
    elif isinstance(e, HTTPError):
        if e.response is not None and e.response.status_code == 401:
            err = QUERY_ERROR_MESSAGES["AuthenticationError"]
        else:
            err = QUERY_ERROR_MESSAGES["HTTPError"]
    elif isinstance(e, TavilyAPIError):
        err = QUERY_ERROR_MESSAGES["TavilyAPIError"]
    else:
        err = QUERY_ERROR_MESSAGES["Unknown"]

    return f"{err['title']}\n\n{err['body']}"




""" This helper function executes and generates a response to the user query. """
def _stream_predict(
    client: chat_client.ChatClient,
    app, 
    question: str,
    model_generator: str,
    model_router: str,
    model_retrieval: str,
    model_hallucination: str,
    model_answer: str,
    prompt_generator: str,
    prompt_router: str,
    prompt_retrieval: str,
    prompt_hallucination: str,
    prompt_answer: str,
    router_use_nim: bool,
    retrieval_use_nim: bool,
    generator_use_nim: bool,
    hallucination_use_nim: bool,
    answer_use_nim: bool,
    nim_generator_ip: str,
    nim_router_ip: str,
    nim_retrieval_ip: str,
    nim_hallucination_ip: str,
    nim_answer_ip: str,
    nim_generator_port: str,
    nim_router_port: str,
    nim_retrieval_port: str,
    nim_hallucination_port: str,
    nim_answer_port: str,
    nim_generator_id: str,
    nim_router_id: str,
    nim_retrieval_id: str,
    nim_hallucination_id: str,
    nim_answer_id: str,
    chat_history: List[dict],
) -> Any:

    inputs = {"question": question, 
              "generator_model_id": model_generator, 
              "router_model_id": model_router, 
              "retrieval_model_id": model_retrieval, 
              "hallucination_model_id": model_hallucination, 
              "answer_model_id": model_answer, 
              "prompt_generator": prompt_generator, 
              "prompt_router": prompt_router, 
              "prompt_retrieval": prompt_retrieval, 
              "prompt_hallucination": prompt_hallucination, 
              "prompt_answer": prompt_answer, 
              "router_use_nim": router_use_nim, 
              "retrieval_use_nim": retrieval_use_nim, 
              "generator_use_nim": generator_use_nim, 
              "hallucination_use_nim": hallucination_use_nim, 
              "nim_generator_ip": nim_generator_ip,
              "nim_router_ip": nim_router_ip,
              "nim_retrieval_ip": nim_retrieval_ip,
              "nim_hallucination_ip": nim_hallucination_ip,
              "nim_answer_ip": nim_answer_ip,
              "nim_generator_port": nim_generator_port,
              "nim_router_port": nim_router_port,
              "nim_retrieval_port": nim_retrieval_port,
              "nim_hallucination_port": nim_hallucination_port,
              "nim_answer_port": nim_answer_port,
              "nim_generator_id": nim_generator_id,
              "nim_router_id": nim_router_id,
              "nim_retrieval_id": nim_retrieval_id,
              "nim_hallucination_id": nim_hallucination_id,
              "nim_answer_id": nim_answer_id,
              "answer_use_nim": answer_use_nim}
    
    if not valid_input(question):
        yield "", chat_history + [{"role": "user", "content": str(question)}, {"role": "assistant", "content": "*** ERR: Unable to process query. Query cannot be empty. ***"}], gr.update(show_label=False), gr.update(visible=False), gr.update(visible=False), [], gr.update(visible=False), gr.update(visible=False)
    else: 
        try:
            actions = {}
            config = RunnableConfig(recursion_limit=RECURSION_LIMIT)
            for output in app.stream(inputs, config=config):
                actions.update(output)
                yield "", chat_history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Working on getting you the best answer..."}], gr.update(value=actions), gr.update(visible=False), gr.update(visible=False), [], gr.update(visible=False), gr.update(visible=False)
                for key, value in output.items():
                    final_value = value
            images = final_value.get("table_images") or []
            reports = final_value.get("inspection_reports") or []
            refs = final_value.get("selected_table_refs") or []
            # Prefer annotated CALS XML (with verify= attributes) when inspection found differences;
            # fall back to plain CALS XML from the best-ranked table ref.
            cals_xml = None
            if reports:
                cals_xml = reports[0].get("annotated_xml")
            if not cals_xml and refs:
                cals_xml = refs[0].get("xml")
            yield (
                "",
                chat_history + [{"role": "user", "content": question}, {"role": "assistant", "content": final_value["generation"]}],
                gr.update(show_label=False),
                gr.update(value=images if images else None, visible=bool(images)),
                gr.update(value=reports if reports else None, visible=bool(reports)),
                refs,
                gr.update(visible=bool(images)),
                gr.update(value=cals_xml, visible=bool(cals_xml)),
            )

        except Exception as e:
            traceback.print_exc()
            message = _get_query_error_message(e)
            yield "", chat_history + [{"role": "user", "content": question}, {"role": "assistant", "content": message}], gr.update(show_label=False), gr.update(visible=False), gr.update(visible=False), [], gr.update(visible=False), gr.update(visible=False)


_support_matrix_cache = None

def load_gpu_support_matrix() -> Dict:
    global _support_matrix_cache
    if _support_matrix_cache is None:
        matrix_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nim_gpu_support_matrix.json')
        with open(matrix_path, 'r') as f:
            _support_matrix_cache = json.load(f)
    return _support_matrix_cache
