RAG Mini Project - Technical Documentation
Project Overview
This project is a Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, Hugging Face embeddings, and the Groq API. It enables users to upload documents (PDF, DOCX, PPTX, XLSX, TXT), converts them into vector embeddings, retrieves relevant context, and generates accurate answers using a Large Language Model.

app.py
Main entry point. Handles Streamlit UI, file uploads, document loading, vector creation, user queries, and answer display.
config.py
Loads environment variables, API keys, and configuration settings.
database/vector_store.py
Creates Hugging Face embeddings and stores them in a FAISS vector database.
llm/rag_chain.py
Builds the Retrieval-Augmented Generation pipeline by connecting the retriever, prompt template, and Groq LLM.
preprocessing/pdf_loader.py
Extracts text from PDF files, chunks the text, and returns LangChain documents.
preprocessing/doc_loader.py
Reads Microsoft Word documents, extracts paragraph text, and prepares chunks.
preprocessing/ppt_loader.py
Processes PowerPoint slides, extracts slide text, and creates document chunks.
preprocessing/excel_loader.py
Reads Excel worksheets, converts rows into readable text, and prepares documents for embedding.
preprocessing/txt_loader.py
Processes plain text files and splits them into chunks.
requirements.txt
Lists project dependencies.
.env_example
Template for environment variables.
Architecture
Upload Document → Loader → Text Chunking → Hugging Face Embeddings → FAISS Vector Store → Retriever → Groq LLM → Answer
Key Features
•	Supports multiple document formats
•	Semantic search using FAISS
•	LangChain orchestration
•	Groq LLM integration
•	Modular codebase
•	Streamlit interface
Suggested Future Enhancements
•	Persistent FAISS index
•	Multiple document upload
•	Conversation memory
•	Source citations
•	Improved logging and testing
