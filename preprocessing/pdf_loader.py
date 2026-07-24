from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_and_split_pdf(uploaded_file) -> List[Document]:
    """Read a Streamlit-uploaded PDF and split it into chunks.

    `uploaded_file` is a Streamlit UploadedFile (a file-like object),
    so we hand its stream straight to pypdf without touching disk.
    """
    reader = PdfReader(uploaded_file)

    page_docs: List[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        page_docs.append(
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": page_number},
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(page_docs)