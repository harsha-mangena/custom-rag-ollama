import os
from pathlib import Path

class Settings:
    # App settings
    APP_NAME = "Enhanced RAG System"
    VERSION = "2.0.0"
    
    # Database
    DB_PATH = "enhanced_rag.db"
    
    # File processing
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx", "md"]
    
    # Embeddings
    DEFAULT_EMBEDDING_MODEL = "bge-small"
    EMBEDDING_MODELS = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    LOG_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    
    # API Settings
    OLLAMA_API_URL = "http://localhost:11434/api"
    HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)