from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_and_split_txt(uploaded_file) -> List[Document]:
    """Reads an uploaded TXT file and returns split document chunks."""
    uploaded_file.seek(0)
    text = uploaded_file.read().decode("utf-8",errors="ignore")

    documents = [
        Document(
            page_content=text,
            metadata={"source": uploaded_file.name,"type": "txt"},
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    return text_splitter.split_documents(documents)