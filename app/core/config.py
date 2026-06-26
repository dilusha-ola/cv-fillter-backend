from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    # Storage settings
    UPLOAD_DIR: str = "data/uploads"
    VECTOR_STORE_DIR: str = "data/vector_store"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
