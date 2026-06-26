from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    LLM_PROVIDER: str = "groq"
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Storage settings
    UPLOAD_DIR: str = "data/uploads"
    VECTOR_STORE_DIR: str = "data/vector_store"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
