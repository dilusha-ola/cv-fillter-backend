import os
from typing import List
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

def get_embeddings(api_key: str = None):
    """
    Creates and returns the embedding model instance based on the active provider.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "huggingface":
        # Initialize local HuggingFace embeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == "groq":
        # Groq compatible embeddings
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=api_key or settings.GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1"
        )
    else:
        # Standard OpenAI embeddings
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=api_key or settings.OPENAI_API_KEY
        )

def create_and_save_vector_store(chunks: List[Document], cv_id: str, api_key: str = None) -> None:
    """
    Embeds CV document chunks and saves them into a specific Chroma collection.
    """
    embeddings = get_embeddings(api_key)
    
    # Initialize Chroma and persist documents to the collection
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=cv_id,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

def retrieve_combined_chunks(
    cv_id: str,
    position: str,
    jd: str,
    conditions: List[str],
    api_key: str = None,
    k_per_query: int = 3
) -> List[Document]:
    """
    Performs multi-query hybrid retrieval to fetch CV sections matching general JD + specific conditions.
    """
    # 1. Verify that collection exists using native Chroma persistent client
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    existing_collections = [c.name for c in client.list_collections()]
    if cv_id not in existing_collections:
        raise FileNotFoundError(f"CV vector store index collection not found for ID: {cv_id}")
        
    embeddings = get_embeddings(api_key)
    
    # 2. Load Chroma collection
    db = Chroma(
        collection_name=cv_id,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
    
    retrieved_docs = []
    seen_contents = set()
    
    # Query A: Job title and snippet of job description
    general_query = f"{position} {jd[:200]}"
    general_docs = db.similarity_search(query=general_query, k=k_per_query)
    for doc in general_docs:
        if doc.page_content not in seen_contents:
            retrieved_docs.append(doc)
            seen_contents.add(doc.page_content)
            
    # Query B: Individual conditions
    for condition in conditions:
        cond_docs = db.similarity_search(query=condition, k=k_per_query)
        for doc in cond_docs:
            if doc.page_content not in seen_contents:
                retrieved_docs.append(doc)
                seen_contents.add(doc.page_content)
                
    return retrieved_docs
