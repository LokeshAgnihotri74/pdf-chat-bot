"""
Streamlit Web Application: Smart AI Agent Assistant
Dual-Component System:
  1. Advanced 'Chat with your PDF' RAG Pipeline (ChromaDB + Strict Guardrails + Groq)
  2. Tool-Calling AI Agent (Open-Meteo Weather + Stock Calculator + Groq Function Calling)
"""

import os
import streamlit as st
from config import (
    GROQ_API_KEY,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    DISTANCE_METRICS,
    DEFAULT_DISTANCE_METRIC,
    STRICT_GUARDRAIL_FALLBACK
)
from rag_pipeline import (
    extract_text_from_pdf,
    chunk_document_pages,
    ChromaRAGManager,
    generate_rag_answer_stream,
    PDFProcessingError,
    PDFEmptyError
)
from agent_tools import run_agent_turn

# Page Configuration
st.set_page_config(
    page_title="Smart AI Agent Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 0.5rem;
    }
    .tool-box {
        border-left: 4px solid #8b5cf6;
        padding: 0.75rem 1rem;
        background-color: rgba(139, 92, 246, 0.05);
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .guardrail-alert {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = []

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []

if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []

if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

if "selected_metric" not in st.session_state:
    st.session_state.selected_metric = DEFAULT_DISTANCE_METRIC


# Sidebar: Configuration & Status
with st.sidebar:
    st.title("⚡ Smart Assistant")
    st.caption("100% Free & Open-Source Stack")

    st.divider()

    # API Key Configuration
    st.subheader("🔑 Inference Key")
    env_has_key = bool(GROQ_API_KEY)
    user_api_key = st.text_input(
        "Groq API Key",
        value=GROQ_API_KEY,
        type="password",
        help="Free key from https://console.groq.com/keys"
    )
    active_key = user_api_key.strip() or GROQ_API_KEY

    if active_key:
        st.success("✅ Groq API Key Configured", icon="🟢")
    else:
        st.warning("⚠️ No Groq API Key found. Add to .env or enter above.", icon="⚠️")

    # Model Selection
    selected_model = st.selectbox(
        "🧠 LLM Model",
        options=SUPPORTED_MODELS,
        index=0,
        help="Runs on Groq Free Tier at 500+ tokens/sec"
    )

    st.divider()

    # RAG Distance Metric Setting
    st.subheader("📐 Vector Search Metric")
    metric_choice = st.radio(
        "ChromaDB Distance Metric",
        options=["cosine", "l2"],
        format_func=lambda x: "Cosine Similarity" if x == "cosine" else "Euclidean Distance (L2)",
        index=0 if st.session_state.selected_metric == "cosine" else 1,
        help="Configure the vector search space metric in ChromaDB."
    )

    # Handle metric switch
    if metric_choice != st.session_state.selected_metric:
        st.session_state.selected_metric = metric_choice
        if st.session_state.rag_manager and st.session_state.rag_chunks:
            with st.spinner(f"Rebuilding vector index with {metric_choice}..."):
                st.session_state.rag_manager.switch_metric(metric_choice, st.session_state.rag_chunks)
            st.toast(f"Switched vector index to {metric_choice.upper()}!", icon="🔄")

    # Metric Math Explainer
    with st.expander("ℹ️ Metric Math & Theory"):
        metric_info = DISTANCE_METRICS[metric_choice]
        st.markdown(f"**{metric_info['name']}**")
        st.caption(metric_info["description"])
        st.code(metric_info["formula"], language="markdown")

    st.divider()

    # System Status Monitor
    st.subheader("📡 Tool Status")
    st.markdown("• **Open-Meteo Weather**: 🟢 Online *(Public, No Key)*")
    st.markdown("• **Stock Calculator**: 🟢 Online *(Yahoo Finance)*")
    st.markdown("• **ChromaDB**: 🟢 Online *(Local all-MiniLM-L6-v2)*")

    st.divider()

    # Reset Buttons
    if st.button("🗑️ Clear All Sessions", use_container_width=True):
        st.session_state.rag_chat_history = []
        st.session_state.agent_chat_history = []
        st.session_state.rag_manager = None
        st.session_state.rag_chunks = []
        st.session_state.current_pdf_name = None
        st.rerun()


# Main Header
st.markdown('<div class="main-header">⚡ Smart AI Agent Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade AI Assistant featuring Advanced PDF RAG with Strict Guardrails '
    'and Tool-Calling Intelligence powered by Groq & LangChain.</div>',
    unsafe_allow_html=True
)

# Tabs
tab_rag, tab_agent = st.tabs([
    "📄 Component 1: Advanced PDF RAG",
    "🤖 Component 2: Tool-Calling AI Agent"
])


# ==============================================================================
# TAB 1: ADVANCED "CHAT WITH YOUR PDF" RAG PIPELINE
# ==============================================================================
with tab_rag:
    col_upload, col_info = st.columns([1.2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a PDF document to query",
            type=["pdf"],
            help="Upload research papers, manuals, financial reports, or invoices."
        )

    # Process PDF when uploaded
    if uploaded_file and uploaded_file.name != st.session_state.current_pdf_name:
        with st.spinner("Processing document, verifying text, and building vector index..."):
            try:
                pages_data = extract_text_from_pdf(uploaded_file, filename=uploaded_file.name)
                chunks = chunk_document_pages(pages_data)

                rag_mgr = ChromaRAGManager(
                    collection_name="pdf_rag_collection",
                    distance_metric=st.session_state.selected_metric
                )
                rag_mgr.index_chunks(chunks)

                st.session_state.rag_manager = rag_mgr
                st.session_state.rag_chunks = chunks
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.rag_chat_history = []
                st.success(f"Successfully indexed '{uploaded_file.name}' with {len(chunks)} chunks!")

            except PDFEmptyError as e:
                st.error(f"❌ Empty or Scanned PDF: {str(e)}")
                st.session_state.rag_manager = None
                st.session_state.rag_chunks = []
                st.session_state.current_pdf_name = None

            except PDFProcessingError as e:
                st.error(f"❌ Corrupted or Invalid PDF: {str(e)}")
                st.session_state.rag_manager = None
                st.session_state.rag_chunks = []
                st.session_state.current_pdf_name = None

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                st.session_state.rag_manager = None
                st.session_state.rag_chunks = []
                st.session_state.current_pdf_name = None

    with col_info:
        if st.session_state.current_pdf_name and st.session_state.rag_chunks:
            st.markdown(f"**Active Document:** `{st.session_state.current_pdf_name}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Indexed Chunks", len(st.session_state.rag_chunks))
            m2.metric("Distance Metric", st.session_state.selected_metric.upper())
            m3.metric("Guardrails", "STRICT")
        else:
            st.info("💡 Upload a PDF file to activate the document retrieval pipeline.")

    st.divider()

    # RAG Chat Messages Area
    st.subheader("💬 Document Conversation")

    # Display chat history
    for msg in st.session_state.rag_chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for s_idx, s in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Chunk {s_idx}** (Page {s['page']} | Distance: {s['distance']:.4f})")
                        st.caption(s["text"])

    # Chat Input for RAG
    if rag_query := st.chat_input("Ask a question about the uploaded document...", key="rag_input"):
        if not st.session_state.rag_manager or not st.session_state.rag_chunks:
            st.warning("Please upload a valid PDF document before asking questions.")
        elif not active_key:
            st.error("Groq API Key is missing. Please set it in `.env` or in the sidebar.")
        else:
            # 1. Append user message
            st.session_state.rag_chat_history.append({"role": "user", "content": rag_query})
            with st.chat_message("user"):
                st.markdown(rag_query)

            # 2. Retrieve relevant chunks
            with st.spinner("Searching vector index..."):
                retrieved = st.session_state.rag_manager.retrieve(rag_query, top_k=4)

            # Format source metadata
            sources_meta = []
            for chunk, dist in retrieved:
                sources_meta.append({
                    "text": chunk.text,
                    "page": chunk.metadata.get("page", 1),
                    "chunk_id": chunk.metadata.get("chunk_id", ""),
                    "distance": dist
                })

            # 3. Stream answer with strict hallucination guardrail
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                stream_generator = generate_rag_answer_stream(
                    query=rag_query,
                    retrieved_chunks=retrieved,
                    api_key=active_key,
                    model_name=selected_model
                )
                full_response = response_placeholder.write_stream(stream_generator)

                # Show sources if context was found
                if full_response.strip() != STRICT_GUARDRAIL_FALLBACK and sources_meta:
                    with st.expander("🔍 View Retrieved Sources & Distance Scores"):
                        for s_idx, s in enumerate(sources_meta, start=1):
                            st.markdown(f"**Source Chunk {s_idx}** (Page {s['page']} | Distance: `{s['distance']:.4f}`)")
                            st.text(s["text"])

            # 4. Save to chat history
            st.session_state.rag_chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_meta
            })


# ==============================================================================
# TAB 2: TOOL-CALLING AI AGENT
# ==============================================================================
with tab_agent:
    st.subheader("🤖 Autonomous Tool-Calling Agent")
    st.caption("Understands user intent, validates arguments, calls Python tools, and synthesizes answers.")

    # Suggested Prompts
    st.markdown("**Suggested Prompts:**")
    prompt_cols = st.columns(3)
    suggested_prompts = [
        "What is the real-time weather in Bhopal right now?",
        "What is the stock price of Apple (AAPL) and calculate the cost for 25 shares?",
        "Explain what a Python generator is and how it saves memory."
    ]

    selected_prompt = None
    for i, p in enumerate(suggested_prompts):
        with prompt_cols[i]:
            if st.button(f"👉 {p[:38]}...", key=f"sugg_{i}", use_container_width=True):
                selected_prompt = p

    st.divider()

    # Display Agent Chat History
    for turn in st.session_state.agent_chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if "tool_executions" in turn and turn["tool_executions"]:
                with st.expander(f"🛠️ Tool Activity ({len(turn['tool_executions'])} tool call(s))"):
                    for t_idx, t in enumerate(turn["tool_executions"], start=1):
                        st.markdown(f"**Step {t_idx}: Tool `{t['tool_name']}`** ({t['duration_ms']} ms)")
                        st.markdown("**Arguments:**")
                        st.json(t["arguments"])
                        st.markdown("**Tool Output:**")
                        try:
                            st.json(t["result"])
                        except Exception:
                            st.code(t["result"])

    # Agent Query Input
    agent_query = st.chat_input("Ask a question, request weather, or calculate stock costs...", key="agent_input")

    # Use prompt from button click if selected
    query_to_run = selected_prompt or agent_query

    if query_to_run:
        if not active_key:
            st.error("Groq API Key is missing. Please set it in `.env` or in the sidebar.")
        else:
            # 1. Display user query
            st.session_state.agent_chat_history.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            # 2. Run agent execution turn
            with st.chat_message("assistant"):
                with st.spinner("Agent is reasoning and selecting appropriate tools..."):
                    result = run_agent_turn(
                        user_message=query_to_run,
                        conversation_history=st.session_state.agent_chat_history[:-1],
                        api_key=active_key,
                        model_name=selected_model
                    )

                # Show tool activities if any
                if result["tool_executions"]:
                    with st.expander(f"🛠️ Executed {len(result['tool_executions'])} Tool(s) Under the Hood", expanded=True):
                        for t_idx, t in enumerate(result["tool_executions"], start=1):
                            st.markdown(f"**Step {t_idx}: Tool `{t['tool_name']}`** ({t['duration_ms']} ms)")
                            st.markdown("**Input Arguments:**")
                            st.json(t["arguments"])
                            st.markdown("**Raw Tool Result:**")
                            try:
                                st.json(t["result"])
                            except Exception:
                                st.code(t["result"])

                # Render final answer
                st.markdown(result["final_answer"])

            # 3. Store in history
            st.session_state.agent_chat_history.append({
                "role": "assistant",
                "content": result["final_answer"],
                "tool_executions": result["tool_executions"]
            })
