"""Streamlit UI - talks to the HTTP API so the same backend can serve other clients.

uv run streamlit run ui/app.py      (API_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
client = httpx.Client(base_url=API, timeout=600, trust_env=False)

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")

EXAMPLES = [
    "When does contract AP-SA-2024-071 end?",
    "Which law governs the agreement with Harbour Metal Works Ltd?",
    "List all contracts that end in 2027.",
    "What is the total value of all EUR-denominated contracts?",
    "What minimum insurance coverage must Kestrel Logistics B.V. maintain?",
]


def api(method: str, path: str, **kw):
    try:
        r = client.request(method, path, **kw)
        r.raise_for_status()
        return r
    except httpx.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()


with st.sidebar:
    st.header("Documents")
    health = api("GET", "/health").json()
    st.caption(f"Model `{health['model']}` · {health['documents']} documents")
    uploads = st.file_uploader("Add PDF / DOCX / TXT", accept_multiple_files=True, type=["pdf", "docx", "txt", "md"])
    if uploads and st.button("Ingest", type="primary"):
        with st.spinner("Parsing, extracting and indexing…"):
            res = api("POST", "/documents", files=[("files", (u.name, u.getvalue())) for u in uploads]).json()
        for r in res:
            (st.success if r["status"] != "error" else st.error)(f"{r['filename']}: {r['status']} {r.get('error', '')}")
    docs = api("GET", "/documents").json()
    if docs:
        df = pd.DataFrame(docs)
        df["pages"] = df.apply(lambda r: f"{r['pages']} (OCR)" if r["ocr_pages"] else str(r["pages"]), axis=1)
        st.dataframe(df[["filename", "pages"]], hide_index=True, use_container_width=True, height=420)

tab_chat, tab_table = st.tabs(["💬 Ask", "📊 Extracted data"])

with tab_chat:
    st.caption("Answers cite the document and page. Questions across many documents are answered from the extracted table.")
    if "history" not in st.session_state:
        st.session_state.history = []
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for c, ex in zip(cols, EXAMPLES):
        if c.button(ex, use_container_width=True):
            clicked = ex
    question = st.chat_input("Ask about your documents…") or clicked

    if question:
        with st.spinner("Thinking…"):
            st.session_state.history.append(api("POST", "/ask", json={"question": question}).json())

    for a in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(a["question"])
        with st.chat_message("assistant"):
            st.write(a["answer"])
            meta = {"sql": "from extracted table", "search": "from document search", "not_found": "no supporting text"}[a["route"]]
            st.caption(f"{meta} · {a['seconds']} s")
            if a.get("rows"):
                df = pd.DataFrame(a["rows"], columns=a["columns"])
                st.dataframe(df, hide_index=True, use_container_width=True)
                with st.expander("SQL"):
                    st.code(a["sql"], language="sql")
            for s in a.get("sources", []):
                where = f"{s['filename']}, page {s['page']}" if s["page"] else s["filename"]
                with st.expander(f"[{s['n']}] {where}"):
                    st.write(s["text"])

with tab_table:
    records = api("GET", "/records").json()
    if not records:
        st.info("No extracted data yet - ingest some documents.")
    else:
        st.caption("One row per document, extracted once at ingest. Edit the schema in `schemas/contract.yaml`.")
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
        st.download_button(
            "Download Excel",
            api("GET", "/records.xlsx").content,
            file_name="contracts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
