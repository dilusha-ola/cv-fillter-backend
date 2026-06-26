import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

def get_embeddings(openai_api_key: str) -> OpenAIEmbeddings:
    """
    Creates and returns the embedding model instance.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

def get_vector_store_path(cv_id: str) -> str:
    """
    Returns the file path for serialized vector store data.
    """
    return os.path.join(settings.VECTOR_STORE_DIR, f"{cv_id}.pkl")

def create_and_save_vector_store(chunks: List[Document], cv_id: str, openai_api_key: str) -> None:
    """
    Embeds CV document chunks and serializes the local vector store dictionary.
    """
    embeddings = get_embeddings(openai_api_key)
    
    # 1. Initialize transient in-memory vector store
    vector_store = InMemoryVectorStore(embeddings)
    
    # 2. Load and embed chunks
    vector_store.add_documents(documents=chunks)
    
    # 3. Access internal dict and dump to pickle file
    store_data = getattr(vector_store, "store", getattr(vector_store, "_store", {}))
    
    file_path = get_vector_store_path(cv_id)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(store_data, f)

def retrieve_combined_chunks(
    cv_id: str,
    position: str,
    jd: str,
    conditions: List[str],
    openai_api_key: str,
    k_per_query: int = 3
) -> List[Document]:
    """
    Performs multi-query hybrid retrieval to fetch CV sections matching general JD + specific conditions.
    """
    embeddings = get_embeddings(openai_api_key)
    file_path = get_vector_store_path(cv_id)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CV vector store index not found for ID: {cv_id}")
        
    # 1. Load serialized store dictionary
    with open(file_path, "rb") as f:
        store_data = pickle.load(f)
        
    # 2. Reconstruct InMemoryVectorStore
    vector_store = InMemoryVectorStore(embeddings)
    if hasattr(vector_store, "store"):
        vector_store.store = store_data
    else:
        vector_store._store = store_data
        
    # 3. Retrieve chunks using general and specific condition queries
    retrieved_docs = []
    seen_contents = set()
    
    # Query A: Job title and snippet of job description
    general_query = f"{position} {jd[:200]}"
    general_docs = vector_store.similarity_search(query=general_query, k=k_per_query)
    for doc in general_docs:
        if doc.page_content not in seen_contents:
            retrieved_docs.append(doc)
            seen_contents.add(doc.page_content)
            
    # Query B: Individual conditions
    for condition in conditions:
        cond_docs = vector_store.similarity_search(query=condition, k=k_per_query)
        for doc in cond_docs:
            if doc.page_content not in seen_contents:
                retrieved_docs.append(doc)
                seen_contents.add(doc.page_content)
                
    return retrieved_docs
