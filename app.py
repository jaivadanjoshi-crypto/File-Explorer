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

# --- STEP 1: INITIALIZE ROTATING WIDGET KEY COUNTER ---
if "uploader_key_version" not in st.session_state:
    st.session_state["uploader_key_version"] = 0
# ------------------------------------------------------

col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")

with col1:
    # STEP 2: LINK ROTATING VERSION VALUE TO WIDGET IDENTITY
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"],
        key=f"doc_uploader_v_{st.session_state['uploader_key_version']}"
    )

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        # Clear out computational variables from system state memory
        for key in list(st.session_state.keys()):
            if key != "uploader_key_version":  # Keep the counter variable intact
                del st.session_state[key]
        
        # STEP 3: INCREMENT RE-RENDER COUNTER TO WIPE PHYSICAL FILE WIDGET
        st.session_state["uploader_key_version"] += 1
        st.rerun()

if uploaded_file is not None:
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

            st.session_state["vectorstore"] = build_vector_store(chunks)
            st.session_state["rag_chain"] = build_rag_chain(st.session_state["vectorstore"])
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["chunk_count"] = len(chunks)

    st.success(f"Indexed {st.session_state['chunk_count']} chunks from {st.session_state['current_file']}. Ask a question below.")

    question = st.text_input("Your question")
    if question:
        if "rag_chain" in st.session_state:
            with st.spinner("Thinking..."):
                result = st.session_state["rag_chain"].invoke({"input": question})
            st.write(result["answer"])
        else:
            st.error("Pipeline missing. Please upload your document again.")
