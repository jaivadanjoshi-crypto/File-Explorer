from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config


def build_vector_store(documents):
    """Embed the document chunks and build a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.from_documents(documents, embeddings)