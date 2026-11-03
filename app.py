"""ContractIQ Streamlit application. Run with: streamlit run app.py"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from core.document_manager import DocumentManager, DocumentRecord
from core.embeddings import EmbeddingError, HuggingFaceEmbedder
from core.llm_client import LLMError, create_llm_client
from core.rag_pipeline import RagAnswer, RagPipeline, build_document_context
from features.clause_extraction import ExtractedField, extract_clauses
from features.contract_summary import generate_summary
from features.risk_analysis import LEVEL_ORDER, RiskFinding, analyze_risks
from utils.parsing import escape_markdown, sanitize_markdown

DASHBOARD = "Dashboard"
LIBRARY = "Document Library"
INTELLIGENCE = "Contract Intelligence"
ASSISTANT = "AI Assistant"
PAGES = [DASHBOARD, LIBRARY, INTELLIGENCE, ASSISTANT]

DISCLAIMER = (
    "ContractIQ is an informational document-review aid, not legal advice. "
    "Check important points against the contract and with a qualified professional."
)

STATUS_LABELS = {
    "found": "Found",
    "not_found": "Not found",
    "needs_review": "Needs review",
}

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1120px; }
  h1, h2, h3 { color: #0F2342; letter-spacing: -0.01em; }
  [data-testid="stSidebar"] { border-right: 1px solid #E4E9F2; }
  .ciq-tagline { color: #52627A; margin-top: -0.6rem; margin-bottom: 1.4rem; }
  .ciq-card {
    background: #FFFFFF; border: 1px solid #E4E9F2; border-radius: 12px;
    padding: 1.05rem 1.2rem;
  }
  .ciq-metric-label { color: #52627A; font-size: 0.85rem; }
  .ciq-metric-value { color: #0F2342; font-size: 2rem; font-weight: 650; }
  .ciq-badge {
    display: inline-block; border-radius: 999px; padding: 0.08rem 0.65rem;
    font-size: 0.78rem; font-weight: 600;
  }
  .ciq-high { background: #FEE4E2; color: #912018; }
  .ciq-medium { background: #FEF0C7; color: #93370D; }
  .ciq-low { background: #DBEAFE; color: #1E40AF; }
</style>
"""

BADGE_CLASSES = dict(zip(LEVEL_ORDER, ("ciq-high", "ciq-medium", "ciq-low")))


@st.cache_resource(show_spinner=False)
def get_manager() -> DocumentManager:
    return DocumentManager(HuggingFaceEmbedder())


def init_state() -> None:
    defaults = {
        "page": DASHBOARD,
        "messages": [],
        "questions_asked": 0,
        "selected_doc_ids": None,  # None means every document is selected.
        "upload_results": [],
        "uploader_key": 0,
        "analysis": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    st.session_state.page = page


def selected_ids(manager: DocumentManager) -> list[str]:
    """Return selected documents that still exist."""
    available = [d.doc_id for d in manager.documents()]
    chosen = st.session_state.selected_doc_ids
    if chosen is None:
        return available
    return [doc_id for doc_id in available if doc_id in chosen]


def toggle_selection(manager: DocumentManager, doc_id: str, widget_key: str) -> None:
    chosen = set(selected_ids(manager))
    if st.session_state[widget_key]:
        chosen.add(doc_id)
    else:
        chosen.discard(doc_id)
    st.session_state.selected_doc_ids = chosen


def format_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def metric_card(column, label: str, value: int) -> None:
    column.markdown(
        f'<div class="ciq-card"><div class="ciq-metric-label">{html.escape(label)}'
        f'</div><div class="ciq-metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_literal(text: str) -> None:
    """Display untrusted document or model text with no formatting applied."""
    st.markdown(escape_markdown(text))


def show_quote(text: str) -> None:
    st.markdown(f"> {escape_markdown(text)}")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(
        f'<p class="ciq-tagline">{html.escape(subtitle)}</p>', unsafe_allow_html=True
    )


def render_dashboard(manager: DocumentManager) -> None:
    page_header(
        "ContractIQ",
        "Understand Contracts. Extract Insights. Make Informed Decisions.",
    )

    documents = manager.documents()
    columns = st.columns(3)
    metric_card(columns[0], "Uploaded contracts", len(documents))
    metric_card(columns[1], "Indexed chunks", manager.total_chunks)
    metric_card(columns[2], "Questions asked", st.session_state.questions_asked)

    st.subheader("Recent documents")
    if documents:
        table = pd.DataFrame(
            [
                {
                    "Document": d.filename,
                    "Pages": d.page_count,
                    "Status": d.status,
                    "Uploaded": format_date(d.uploaded_at),
                }
                for d in documents[:5]
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("No contracts yet. Upload one in the Document Library to begin.")

    st.subheader("Quick navigation")
    columns = st.columns(3)
    for column, target in zip(columns, (LIBRARY, INTELLIGENCE, ASSISTANT)):
        column.button(
            target, on_click=go_to, args=(target,), key=f"nav_{target}", width="stretch"
        )


def process_uploads(manager: DocumentManager, files: list) -> None:
    outcomes: list[tuple[str, str]] = []
    with st.status(f"Processing {len(files)} file(s)...", expanded=True) as status:
        for file in files:
            st.write(f"Processing {file.name}...")
            result = manager.ingest(file.name, file.getvalue())
            outcomes.append((result.status, result.message))
        status.update(label="Processing finished.", state="complete", expanded=False)

    st.session_state.upload_results = outcomes
    # A new key clears the uploader so that files are not processed twice.
    st.session_state.uploader_key += 1


def remove_document(manager: DocumentManager, doc_id: str) -> None:
    manager.remove(doc_id)
    st.session_state.analysis.pop(doc_id, None)


def render_library(manager: DocumentManager) -> None:
    page_header("Document Library", "Upload and manage your PDF contracts.")

    files = st.file_uploader(
        "Upload PDF contracts",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="Text-based PDFs only. Scanned PDFs are not supported.",
    )
    st.caption(
        "The first upload downloads the embedding model (about 90 MB), so it "
        "takes longer."
    )
    if files and st.button("Process documents", type="primary"):
        process_uploads(manager, files)
        st.rerun()

    show_message = {"indexed": st.success, "duplicate": st.info, "failed": st.error}
    for status, message in st.session_state.upload_results:
        show_message[status](message)

    st.subheader("Your documents")
    documents = manager.documents()
    if not documents:
        st.info("No documents have been processed yet.")
        return

    st.caption("Ticked documents are searched by the AI Assistant.")
    selected = set(selected_ids(manager))
    for document in documents:
        with st.container(border=True):
            info, tick, remove = st.columns([6, 2, 1.4], vertical_alignment="center")
            info.markdown(f"**{html.escape(document.filename)}**")
            info.caption(
                f"{document.status} · {document.page_count} pages · "
                f"{document.chunk_count} chunks"
            )
            widget_key = f"select_{document.doc_id}"
            st.session_state[widget_key] = document.doc_id in selected
            tick.checkbox(
                "Selected",
                key=widget_key,
                on_change=toggle_selection,
                args=(manager, document.doc_id, widget_key),
            )
            remove.button(
                "Remove",
                key=f"remove_{document.doc_id}",
                on_click=remove_document,
                args=(manager, document.doc_id),
            )
            if document.empty_pages:
                pages = ", ".join(str(page) for page in document.empty_pages)
                st.warning(f"No text was found on page(s) {pages}.")


def run_analysis(manager: DocumentManager, record: DocumentRecord) -> dict:
    """Run the three analyses, keeping any that succeed if another fails."""
    llm = create_llm_client(manager.settings)
    context = build_document_context(
        record, manager.embedder, manager.store, manager.settings
    )
    tasks = {
        "summary": lambda: generate_summary(llm, record, context),
        "clauses": lambda: extract_clauses(llm, record, context),
        "risks": lambda: analyze_risks(llm, record, context, manager.settings),
    }
    results: dict = {"errors": {}, "is_partial": context.is_partial}
    for name, task in tasks.items():
        try:
            results[name] = task()
        except LLMError as exc:
            results["errors"][name] = str(exc)
    return results


def render_clause(field: ExtractedField) -> None:
    status = STATUS_LABELS[field.status]
    with st.expander(f"{field.label}: {status}", expanded=field.status != "not_found"):
        if field.status == "not_found":
            st.write(field.value)
            return
        show_literal(field.value)
        if field.page is not None:
            st.caption(f"Source: page {field.page}")
        if field.quote:
            show_quote(field.quote)
        if field.note:
            st.warning(field.note)


def render_finding(finding: RiskFinding) -> None:
    badge = BADGE_CLASSES[finding.level]
    with st.container(border=True):
        st.markdown(
            f'<span class="ciq-badge {badge}">{html.escape(finding.level)}</span> '
            f"<strong>{html.escape(finding.issue)}</strong>",
            unsafe_allow_html=True,
        )
        st.write("**Why it deserves attention**")
        show_literal(finding.reason)
        if finding.is_missing_item:
            location = "Not stated anywhere in the document"
        elif finding.verified:
            location = f"Page {finding.page}"
        else:
            location = "Not verified: the supporting text was not found in the document"
        st.caption(f"Source: {finding.filename} · {location}")
        if finding.quote:
            show_quote(finding.quote)


def render_intelligence(manager: DocumentManager) -> None:
    page_header(
        "Contract Intelligence",
        "Summaries, key clauses and points that may need human review.",
    )
    documents = manager.documents()
    if not documents:
        st.info("Upload a contract in the Document Library first.")
        return

    names = {d.doc_id: d.filename for d in documents}
    doc_id = st.selectbox(
        "Selected contract", list(names), format_func=lambda key: names[key]
    )
    record = manager.get(doc_id)

    if st.button("Analyse contract", type="primary"):
        with st.spinner("Analysing the contract. This can take a minute..."):
            try:
                st.session_state.analysis[doc_id] = run_analysis(manager, record)
            except (LLMError, EmbeddingError) as exc:
                st.error(str(exc))

    results = st.session_state.analysis.get(doc_id)
    if not results:
        st.caption("Results appear here after you run the analysis.")
        return

    if results["is_partial"]:
        st.warning(
            "This contract is very long, so only its first page and the most "
            "relevant passages were analysed. \"Not found\" may not mean absent."
        )

    errors = results["errors"]

    st.subheader("Executive summary")
    if "summary" in results:
        st.markdown(sanitize_markdown(results["summary"].text))
    else:
        st.error(errors.get("summary", "The summary is unavailable."))

    st.subheader("Extracted clauses")
    if "clauses" in results:
        for field in results["clauses"]:
            render_clause(field)
    else:
        st.error(errors.get("clauses", "Extraction is unavailable."))

    st.subheader("Potential risk findings")
    st.caption(DISCLAIMER)
    if "risks" not in results:
        st.error(errors.get("risks", "Risk analysis is unavailable."))
    elif not results["risks"]:
        st.info("No findings were returned. This does not mean the contract is safe.")
    else:
        for finding in results["risks"]:
            render_finding(finding)


def render_citations(answer_status: str, citations: list[dict]) -> None:
    if not citations:
        return
    title = "Sources" if answer_status == "answered" else "Retrieved passages"
    with st.expander(f"{title} ({len(citations)})"):
        for citation in citations:
            st.markdown(
                f"**[{citation['label']}] {html.escape(citation['filename'])}, "
                f"page {citation['page_number']}**"
            )
            show_quote(citation["text"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(sanitize_markdown(message["content"]))
        if message.get("status") == "uncited":
            st.warning(
                "This answer did not cite the contract, so treat it as unverified. "
                "The passages searched are listed below."
            )
        if message["role"] == "assistant":
            render_citations(message.get("status", ""), message.get("citations", []))


def to_message(answer: RagAnswer) -> dict:
    return {
        "role": "assistant",
        "content": answer.answer,
        "status": answer.status,
        "citations": [
            {
                "label": c.label,
                "filename": c.filename,
                "page_number": c.page_number,
                "text": c.text,
            }
            for c in answer.citations
        ],
    }


def clear_chat() -> None:
    st.session_state.messages = []


def render_assistant(manager: DocumentManager) -> None:
    page_header("AI Assistant", "Ask questions and get answers with sources.")
    documents = manager.documents()
    if not documents:
        st.info("Upload a contract in the Document Library first.")
        return

    doc_ids = selected_ids(manager)
    names = [d.filename for d in documents if d.doc_id in doc_ids]
    left, right = st.columns([5, 1.4], vertical_alignment="center")
    if names:
        left.caption("Searching: " + ", ".join(html.escape(n) for n in names))
    else:
        left.warning("No documents are selected in the Document Library.")
    right.button("Clear chat", on_click=clear_chat)

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask about your contracts", disabled=not doc_ids)
    if not question or not question.strip():
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    st.session_state.questions_asked += 1
    render_message(user_message)

    with st.spinner("Searching your contracts..."):
        try:
            pipeline = RagPipeline(
                manager.embedder,
                manager.store,
                create_llm_client(manager.settings),
                manager.settings,
            )
            reply = to_message(pipeline.ask(question, doc_ids))
        except (LLMError, EmbeddingError) as exc:
            reply = {"role": "assistant", "content": str(exc), "error": True}

    st.session_state.messages.append(reply)
    render_message(reply)


def render_sidebar(manager: DocumentManager) -> None:
    with st.sidebar:
        st.markdown("## ContractIQ")
        st.caption("Contract intelligence assistant")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        if manager.settings.has_api_key:
            st.caption(f"LLM: {manager.settings.gemini_model}")
        else:
            st.warning("No Gemini API key. Add GEMINI_API_KEY to your .env file.")
        st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="ContractIQ", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    manager = get_manager()
    render_sidebar(manager)

    renderers = {
        DASHBOARD: render_dashboard,
        LIBRARY: render_library,
        INTELLIGENCE: render_intelligence,
        ASSISTANT: render_assistant,
    }
    renderers[st.session_state.page](manager)


if __name__ == "__main__":
    main()
