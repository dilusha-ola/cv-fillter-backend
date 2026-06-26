import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_document(file_path: str) -> List[Document]:
    """
    Loads a PDF or Word document, extracts its text, and chunks it.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found at: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    # 1. Load document text based on extension
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only PDF and DOCX are supported.")
        
    docs = loader.load()
    
    # 2. Split the extracted text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(docs)
    return chunks
