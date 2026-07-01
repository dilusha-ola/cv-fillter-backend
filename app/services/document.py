import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _extract_pdf_annotation_urls(file_path: str) -> List[str]:
    """
    Reads PDF link annotations (URI actions) to find URLs attached to icons,
    images, or any non-text element — these never appear in extracted plain text.
    """
    from pypdf import PdfReader
    urls: List[str] = []
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if "/Annots" not in page:
                continue
            for annot_ref in page["/Annots"]:
                try:
                    annot = annot_ref.get_object()
                    if annot.get("/Subtype") != "/Link":
                        continue
                    action = annot.get("/A")
                    if action is None:
                        continue
                    if hasattr(action, "get_object"):
                        action = action.get_object()
                    if action.get("/S") == "/URI":
                        uri = action.get("/URI")
                        if uri:
                            urls.append(str(uri))
                except Exception:
                    continue  # skip malformed annotations
    except Exception:
        pass  # skip unreadable PDFs gracefully
    return urls


def _extract_docx_hyperlink_urls(file_path: str) -> List[str]:
    """
    Reads DOCX relationship table to find hyperlinks attached to images/icons
    that won't appear in plain text extraction.
    """
    urls: List[str] = []
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        for rel in doc.part.rels.values():
            if "hyperlink" in rel.reltype:
                url = rel.target_ref
                if url and url.startswith(("http://", "https://")):
                    urls.append(url)
    except ImportError:
        pass  # python-docx not installed — skip silently
    except Exception:
        pass
    return urls


def extract_embedded_urls(file_path: str) -> List[str]:
    """
    Extracts URLs that are embedded as clickable hyperlinks/annotations in a
    PDF or DOCX — e.g. a GitHub icon that links to the candidate's profile.
    These URLs are invisible to plain-text regex extraction.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf_annotation_urls(file_path)
    elif ext == ".docx":
        return _extract_docx_hyperlink_urls(file_path)
    return []

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


def load_document_text(file_path: str) -> str:
    """
    Loads a PDF or DOCX and returns the full raw text (all pages joined).
    Used for URL extraction before chunking.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    docs = loader.load()
    return "\n".join(d.page_content for d in docs)


def text_to_chunks(text: str, source: str = "web") -> List[Document]:
    """
    Splits arbitrary text into Document chunks with a 'source' metadata tag.
    Used to index scraped web profile data alongside CV chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(
        [Document(page_content=text, metadata={"source": source})]
    )
