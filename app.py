import streamlit as st

import config
from preprocessing.pdf_loader import load_and_split_pdf
from preprocessing.doc_loader import load_and_split_docx
from preprocessing.txt_loader import load_and_split_txt
from database.vector_store import build_vector_store
from llm.rag_chain import build_rag_chain

st.set_page_config(page_title="RAG Mini Project", page_icon="📄")
st.title("📄 Document Q&A (RAG)")

# --- ADDED: CLEAR SESSION BUTTON IN SIDEBAR ---
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Clear Session", use_container_width=True):
        # Wipe all cached values in memory
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Force website to refresh with a completely fresh state
        st.rerun()
# ----------------------------------------------

if not config.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. Add it to your .env file (local) "
        "or Streamlit secrets (cloud)."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
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

        vectorstore = build_vector_store(chunks)
        rag_chain = build_rag_chain(vectorstore)

    st.success(f"Indexed {len(chunks)} chunks. Ask a question below.")

    question = st.text_input("Your question")
    if question:
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({"input": question})
        st.write(result["answer"])
