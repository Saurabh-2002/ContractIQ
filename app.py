"""ContractIQ Streamlit application. Run with: streamlit run app.py"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

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
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1180px;
  }

  /* Headings & Brand */
  h1, h2, h3 {
    color: #0F172A;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .ciq-tagline {
    color: #475569;
    font-size: 1.05rem;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
  }

  /* Metric & Feature Cards */
  .ciq-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
    transition: all 0.2s ease-in-out;
  }
  .ciq-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    transform: translateY(-2px);
  }

  .ciq-card-icon {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
    display: inline-block;
  }

  .ciq-metric-label {
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .ciq-metric-value {
    color: #0F172A;
    font-size: 2.2rem;
    font-weight: 750;
    line-height: 1.2;
    margin-top: 0.2rem;
  }

  /* Status Badges */
  .ciq-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 999px;
    padding: 0.18rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 650;
  }
  .ciq-high { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
  .ciq-medium { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
  .ciq-low { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
  .ciq-success { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
  .ciq-neutral { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }

  /* Risk Summary Banner */
  .risk-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .risk-pill-box {
    padding: 0.9rem 1.1rem;
    border-radius: 12px;
    font-weight: 600;
    display: flex;
    flex-direction: column;
  }
  .risk-pill-box.high { background: #FEF2F2; border-left: 4px solid #EF4444; color: #991B1B; }
  .risk-pill-box.medium { background: #FFFBEB; border-left: 4px solid #F59E0B; color: #92400E; }
  .risk-pill-box.low { background: #EFF6FF; border-left: 4px solid #3B82F6; color: #1E40AF; }

  /* Prompt Chips */
  .prompt-chip {
    display: inline-block;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    margin: 0.2rem 0.2rem;
    font-size: 0.85rem;
    color: #334155;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .prompt-chip:hover {
    background: #EFF6FF;
    border-color: #BFDBFE;
    color: #1D4ED8;
  }

  /* Quote Styling */
  blockquote {
    border-left: 3px solid #3B82F6 !important;
    background-color: #F8FAFC;
    padding: 0.6rem 1rem !important;
    border-radius: 0 8px 8px 0;
    color: #334155;
    font-style: normal;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
  }
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
        "active_prompt": None,
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


def metric_card(column, label: str, value: int, icon: str = "📊") -> None:
    column.markdown(
        f'<div class="ciq-card">'
        f'<div class="ciq-card-icon">{icon}</div>'
        f'<div class="ciq-metric-label">{html.escape(label)}</div>'
        f'<div class="ciq-metric-value">{value}</div>'
        f'</div>',
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


def load_sample_contracts(manager: DocumentManager) -> list[tuple[str, str]]:
    samples_dir = Path(__file__).resolve().parent / "data" / "sample_contracts"
    if not samples_dir.exists():
        return [("failed", "Sample contracts directory not found.")]

    outcomes = []
    for pdf_file in sorted(samples_dir.glob("*.pdf")):
        res = manager.ingest(pdf_file.name, pdf_file.read_bytes())
        outcomes.append((res.status, res.message))
    return outcomes


def render_dashboard(manager: DocumentManager) -> None:
    page_header(
        "ContractIQ",
        "AI-Powered Document Intelligence & Risk Analysis Platform",
    )

    documents = manager.documents()
    columns = st.columns(3)
    metric_card(columns[0], "Uploaded contracts", len(documents), icon="📁")
    metric_card(columns[1], "Indexed vector chunks", manager.total_chunks, icon="🧩")
    metric_card(columns[2], "Inquiries answered", st.session_state.questions_asked, icon="💬")

    st.markdown("---")

    # Quick sample loader if empty
    if not documents:
        st.info("💡 **Getting started:** Your workspace is clean. Click below to load 3 ready-made enterprise contracts.")
        if st.button("🚀 Load 3 Sample Contracts", type="primary", use_container_width=False):
            with st.spinner("Ingesting and indexing sample contracts..."):
                load_sample_contracts(manager)
            st.success("Sample contracts indexed successfully!")
            st.rerun()

    st.subheader("Recent Contracts")
    if documents:
        table = pd.DataFrame(
            [
                {
                    "Document Name": d.filename,
                    "Pages": d.page_count,
                    "Chunks Indexed": d.chunk_count,
                    "Status": d.status.title(),
                    "Added": format_date(d.uploaded_at),
                }
                for d in documents[:6]
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.caption("No contracts uploaded yet. Navigate to the Document Library to upload PDFs.")

    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### 📁 Document Library")
            st.caption("Upload new contracts, review chunks, and toggle search selection.")
            st.button("Open Library", on_click=go_to, args=(LIBRARY,), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown("### 🧠 Intelligence")
            st.caption("Extract executive summaries, 9 standard clauses, and risk findings.")
            st.button("Review Intelligence", on_click=go_to, args=(INTELLIGENCE,), use_container_width=True)
    with col3:
        with st.container(border=True):
            st.markdown("### 💬 AI Assistant")
            st.caption("Chat with verified citations referencing source pages directly.")
            st.button("Launch Assistant", on_click=go_to, args=(ASSISTANT,), use_container_width=True)


def process_uploads(manager: DocumentManager, files: list) -> None:
    outcomes: list[tuple[str, str]] = []
    with st.status(f"Processing {len(files)} document(s)...", expanded=True) as status:
        for file in files:
            st.write(f"Extracting and indexing **{file.name}**...")
            result = manager.ingest(file.name, file.getvalue())
            outcomes.append((result.status, result.message))
        status.update(label="Document ingestion complete!", state="complete", expanded=False)

    st.session_state.upload_results = outcomes
    st.session_state.uploader_key += 1


def remove_document(manager: DocumentManager, doc_id: str) -> None:
    manager.remove(doc_id)
    st.session_state.analysis.pop(doc_id, None)


def render_library(manager: DocumentManager) -> None:
    page_header("Document Library", "Upload, inspect, and manage contract vector indices.")

    top_col1, top_col2 = st.columns([4, 2], vertical_alignment="bottom")
    with top_col2:
        if st.button("📥 Load Sample Contracts", use_container_width=True):
            with st.spinner("Ingesting sample agreements..."):
                results = load_sample_contracts(manager)
                st.session_state.upload_results = results
            st.rerun()

    files = st.file_uploader(
        "Upload PDF Contracts",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="Text-based PDFs only. Scanned image-only PDFs require OCR.",
    )

    if files:
        if st.button("⚡ Index Uploaded Documents", type="primary", use_container_width=True):
            process_uploads(manager, files)
            st.rerun()

    show_message = {"indexed": st.success, "duplicate": st.info, "failed": st.error}
    for status, message in st.session_state.upload_results:
        show_message[status](message)

    st.markdown("---")
    st.subheader("Indexed Documents")
    documents = manager.documents()
    if not documents:
        st.info("No documents are currently indexed.")
        return

    st.caption("Check or uncheck contracts to scope the AI Assistant's semantic search.")
    selected = set(selected_ids(manager))

    for document in documents:
        with st.container(border=True):
            info, tick, remove = st.columns([6, 2, 1.4], vertical_alignment="center")
            info.markdown(f"📄 **{html.escape(document.filename)}**")
            info.caption(
                f"Status: `{document.status}` · {document.page_count} pages · "
                f"{document.chunk_count} vector chunks"
            )
            widget_key = f"select_{document.doc_id}"
            st.session_state[widget_key] = document.doc_id in selected
            tick.checkbox(
                "Include in Search",
                key=widget_key,
                on_change=toggle_selection,
                args=(manager, document.doc_id, widget_key),
            )
            remove.button(
                "Delete",
                key=f"remove_{document.doc_id}",
                on_click=remove_document,
                args=(manager, document.doc_id),
                type="secondary",
            )
            if document.empty_pages:
                pages = ", ".join(str(page) for page in document.empty_pages)
                st.warning(f"Note: No readable text was detected on page(s) {pages}.")


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
    status_badge = {
        "found": '<span class="ciq-badge ciq-success">✓ Found</span>',
        "not_found": '<span class="ciq-badge ciq-neutral">✗ Not found</span>',
        "needs_review": '<span class="ciq-badge ciq-medium">⚠ Needs review</span>',
    }.get(field.status, "")

    with st.expander(f"{field.label} — {STATUS_LABELS.get(field.status, field.status)}", expanded=(field.status == "needs_review")):
        st.markdown(status_badge, unsafe_allow_html=True)
        if field.status == "not_found":
            st.write(field.value)
            return
        st.markdown(f"**Extracted Terms:**")
        show_literal(field.value)
        if field.page is not None:
            st.caption(f"📍 **Citation**: Page {field.page}")
        if field.quote:
            st.markdown("**Verbatim Clause Context:**")
            show_quote(field.quote)
        if field.note:
            st.warning(field.note)


def render_finding(finding: RiskFinding) -> None:
    badge = BADGE_CLASSES[finding.level]
    with st.container(border=True):
        st.markdown(
            f'<span class="ciq-badge {badge}">{html.escape(finding.level)} ATTENTION</span> '
            f"<strong style='font-size: 1.05rem; margin-left: 0.4rem;'>{html.escape(finding.issue)}</strong>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
        st.write("**Why this requires review:**")
        show_literal(finding.reason)

        if finding.is_missing_item:
            location = "Not stated anywhere in document"
        elif finding.verified:
            location = f"Page {finding.page}"
        else:
            location = "Unverified: Quote text could not be mapped to document"

        st.caption(f"Source: `{finding.filename}` · {location}")
        if finding.quote:
            show_quote(finding.quote)


def render_intelligence(manager: DocumentManager) -> None:
    page_header(
        "Contract Intelligence",
        "Executive summaries, structured clause extraction, and automated risk detection.",
    )
    documents = manager.documents()
    if not documents:
        st.info("Please upload or load a contract in the Document Library first.")
        return

    names = {d.doc_id: d.filename for d in documents}
    col_sel, col_btn = st.columns([4, 2], vertical_alignment="bottom")
    with col_sel:
        doc_id = st.selectbox(
            "Select contract to inspect:", list(names), format_func=lambda key: names[key]
        )
    record = manager.get(doc_id)

    with col_btn:
        if st.button("🔍 Run Full Intelligence Audit", type="primary", use_container_width=True):
            with st.spinner("Extracting insights and performing legal risk audit..."):
                try:
                    st.session_state.analysis[doc_id] = run_analysis(manager, record)
                except (LLMError, EmbeddingError) as exc:
                    st.error(str(exc))

    results = st.session_state.analysis.get(doc_id)
    if not results:
        st.info("👆 Click **'Run Full Intelligence Audit'** to analyze this contract.")
        return

    if results["is_partial"]:
        st.warning(
            "⚠️ This contract exceeds standard analysis size. The audit focused on core terms and "
            "most relevant passages. 'Not found' does not guarantee complete absence."
        )

    errors = results["errors"]

    # Tabs for modern interactive browsing
    tab_summary, tab_clauses, tab_risks, tab_export = st.tabs([
        "📋 Executive Summary",
        "📑 Key Clauses (9 Fields)",
        "🛡️ Risk Findings",
        "📤 Export Report",
    ])

    with tab_summary:
        st.subheader("Executive Briefing")
        if "summary" in results:
            st.markdown(sanitize_markdown(results["summary"].text))
        else:
            st.error(errors.get("summary", "Summary unavailable."))

    with tab_clauses:
        st.subheader("Key Contract Clauses")
        if "clauses" in results:
            clause_list = results["clauses"]
            found_count = sum(1 for c in clause_list if c.status == "found")
            review_count = sum(1 for c in clause_list if c.status == "needs_review")
            missing_count = sum(1 for c in clause_list if c.status == "not_found")

            cols = st.columns(3)
            cols[0].metric("Verified Found", found_count)
            cols[1].metric("Needs Human Review", review_count)
            cols[2].metric("Not Found", missing_count)

            # Filter pills
            filter_choice = st.radio(
                "Filter clauses:",
                ["All", "Found Only", "Needs Review", "Not Found"],
                horizontal=True,
            )

            for field in clause_list:
                if filter_choice == "Found Only" and field.status != "found":
                    continue
                if filter_choice == "Needs Review" and field.status != "needs_review":
                    continue
                if filter_choice == "Not Found" and field.status != "not_found":
                    continue
                render_clause(field)
        else:
            st.error(errors.get("clauses", "Clause extraction unavailable."))

    with tab_risks:
        st.subheader("Potential Risk Findings")
        st.caption(DISCLAIMER)

        if "risks" not in results:
            st.error(errors.get("risks", "Risk analysis unavailable."))
        elif not results["risks"]:
            st.success("🎉 No high-severity concerns identified in this review.")
        else:
            risk_list = results["risks"]
            high_count = sum(1 for r in risk_list if r.level == "High")
            med_count = sum(1 for r in risk_list if r.level == "Medium")
            low_count = sum(1 for r in risk_list if r.level == "Low")

            st.markdown(
                f'<div class="risk-summary-grid">'
                f'<div class="risk-pill-box high"><span>High Attention</span><strong style="font-size: 1.5rem;">{high_count}</strong></div>'
                f'<div class="risk-pill-box medium"><span>Medium Attention</span><strong style="font-size: 1.5rem;">{med_count}</strong></div>'
                f'<div class="risk-pill-box low"><span>Low Attention</span><strong style="font-size: 1.5rem;">{low_count}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            risk_filter = st.radio(
                "Filter risk severity:",
                ["All", "High Only", "Medium Only", "Low Only"],
                horizontal=True,
            )

            for finding in risk_list:
                if risk_filter == "High Only" and finding.level != "High":
                    continue
                if risk_filter == "Medium Only" and finding.level != "Medium":
                    continue
                if risk_filter == "Low Only" and finding.level != "Low":
                    continue
                render_finding(finding)

    with tab_export:
        st.subheader("Export Audit Report")
        st.caption("Download a formatted Markdown summary of the audit findings.")

        # Build clean export
        summary_text = results.get("summary").text if "summary" in results else "N/A"
        clauses_data = []
        if "clauses" in results:
            for c in results["clauses"]:
                clauses_data.append(f"### {c.label} ({c.status})\n- **Value**: {c.value}\n- **Page**: {c.page}\n")
        risks_data = []
        if "risks" in results:
            for r in results["risks"]:
                risks_data.append(f"### [{r.level}] {r.issue}\n- **Reason**: {r.reason}\n- **Page**: {r.page}\n")

        full_report = (
            f"# ContractIQ Audit Report: {record.filename}\n\n"
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## Executive Summary\n{summary_text}\n\n"
            f"## Key Clauses\n" + "\n".join(clauses_data) + "\n\n"
            f"## Risk Findings\n" + "\n".join(risks_data) + "\n\n"
            f"---\n*{DISCLAIMER}*"
        )

        st.download_button(
            label="📥 Download Audit Report (.md)",
            data=full_report,
            file_name=f"ContractIQ_{record.filename}_Audit.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_citations(answer_status: str, citations: list[dict]) -> None:
    if not citations:
        return
    title = "Verified Citations" if answer_status == "answered" else "Retrieved Source Passages"
    with st.expander(f"📍 {title} ({len(citations)} source passages)", expanded=True):
        for citation in citations:
            st.markdown(
                f"**[{citation['label']}] `{html.escape(citation['filename'])}` · Page {citation['page_number']}**"
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
                "Notice: This response could not be strictly verified against the retrieved text. "
                "Review the cited passages below."
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
    page_header("AI Assistant", "Contract Q&A with grounded citations and page references.")
    documents = manager.documents()
    if not documents:
        st.info("Upload or load a contract in the Document Library to begin chatting.")
        return

    doc_ids = selected_ids(manager)
    names = [d.filename for d in documents if d.doc_id in doc_ids]

    top_l, top_r = st.columns([5, 1.4], vertical_alignment="center")
    with top_l:
        if names:
            st.caption("Active Knowledge Base: " + ", ".join(f"`{html.escape(n)}`" for n in names))
        else:
            st.warning("⚠️ No documents selected. Check at least one contract in the Document Library.")
    with top_r:
        st.button("Clear Chat", on_click=clear_chat, use_container_width=True)

    # Interactive suggested questions
    st.markdown("**Suggested Questions:**")
    suggested = [
        "What is the annual licence fee for the software?",
        "What notice period is required to terminate?",
        "What is the cap on the licensor's liability?",
        "Are there non-compete restrictions?",
    ]
    chip_cols = st.columns(len(suggested))
    selected_prompt = None
    for idx, prompt_text in enumerate(suggested):
        if chip_cols[idx].button(prompt_text, key=f"chip_{idx}", use_container_width=True):
            selected_prompt = prompt_text

    # Render previous messages
    for message in st.session_state.messages:
        render_message(message)

    # Input handling
    input_text = st.chat_input("Ask a question about the contracts...", disabled=not doc_ids)
    question = selected_prompt or input_text

    if question and question.strip():
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        st.session_state.questions_asked += 1
        render_message(user_message)

        with st.spinner("Searching contract chunks & generating cited response..."):
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
        st.markdown("## ⚖️ ContractIQ")
        st.caption("Enterprise Contract Intelligence")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()

        # System & API status
        st.markdown("**System Status**")
        if manager.settings.has_api_key:
            st.markdown(f'<span class="ciq-badge ciq-success">● Connected</span>', unsafe_allow_html=True)
            st.caption(f"Model: `{manager.settings.gemini_model}`")
        else:
            st.markdown(f'<span class="ciq-badge ciq-high">● No API Key</span>', unsafe_allow_html=True)
            st.caption("Add GEMINI_API_KEY to `.env` for AI features.")

        st.caption(f"Vector Index: `{manager.total_chunks} chunks`")
        st.divider()
        st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="ContractIQ", page_icon="⚖️", layout="wide")
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
