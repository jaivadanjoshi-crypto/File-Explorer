from typing import List
import pandas as pd
from langchain_core.documents import Document

def load_and_split_excel(uploaded_file) -> List[Document]:
    """Read a Streamlit Excel file (.xlsx, .xls) and convert rows to Documents.
    
    Each row is turned into a single document string, keeping columns and 
    values together to preserve tabular context for your vector database.
    """
    documents = []
    
    # Read all worksheets inside the file
    excel_file = pd.ExcelFile(uploaded_file)
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Strip completely blank rows from processing
        df = df.dropna(how="all")
        
        for index, row in df.iterrows():
            # Build string like: "ColumnA: Value1 | ColumnB: Value2"
            row_items = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            row_text = " | ".join(row_items)
            
            if not row_text.strip():
                continue
                
            metadata = {
                "source": uploaded_file.name,
                "sheet": sheet_name,
                "row": index + 1
            }
            
            documents.append(Document(page_content=row_text, metadata=metadata))
            
    return documents
