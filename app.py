import streamlit as st

import config
from preprocessing.pdf_loader import load_and_split_pdf
from preprocessing.doc_loader import load_and_split_docx
from preprocessing.txt_loader import load_and_split_txt
from database.vector_store import build_vector_store
from llm.rag_chain import build_rag_chain

st.set_page_config(page_title="RAG Mini Project", page_icon="📄")
st.title("📄 Document Q&A (RAG)")

if not config.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. Add it to your .env file (local) "
        "or Streamlit secrets (cloud)."
    )
    st.stop()

# --- NEW: PLACED SIDE-BY-SIDE WITH UPLOAD BAR ---
# Create 2 columns (85% width for file input, 15% width for the reset action)
col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")

with col1:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        # Wipe all cached values in memory safely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Force website to refresh cleanly
        st.rerun()
# ------------------------------------------------

if uploaded_file is not None:
    # Check if this specific file has already been indexed in this session
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        name = uploaded_file.name.lower()
        with st.spinner("Reading and indexing your document..."):
            if name.endswith(".pdf"):
                chunks = load_and_split_pdf(uploaded_file)
            elif name.endswith(".docx"):
                chunks = load_and_split_docx(uploaded_file)
            elif name.endswith(".txt"):
                chunks = load_and_split_txt(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()

            if not chunks:
                st.error("Couldn't extract any text from this file.")
                st.stop()

            # Save the pipeline directly into safe memory states
            st.session_state["vectorstore"] = build_vector_store(chunks)
            st.session_state["rag_chain"] = build_rag_chain(st.session_state["vectorstore"])
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["chunk_count"] = len(chunks)

    # UI status indicator using state tracking values
    st.success(f"Indexed {st.session_state['chunk_count']} chunks from {st.session_state['current_file']}. Ask a question below.")

    question = st.text_input("Your question")
    if question:
        # Prevent calling a chain that hasn't finished compiling safely
        if "rag_chain" in st.session_state:
            with st.spinner("Thinking..."):
                result = st.session_state["rag_chain"].invoke({"input": question})
            st.write(result["answer"])
        else:
            st.error("Pipeline missing. Please upload your document again.")
