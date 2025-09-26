from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.3
    
    # Embeddings Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    
    # Hypothesis Generation
    max_hypotheses: int = 5
    min_confidence: float = 0.1
    max_confidence: float = 0.95
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Validate OpenAI API key
if not settings.openai_api_key:
    # Try to get from environment
    settings.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
