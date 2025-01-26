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

    SEARXNG_URL = "http://localhost:8888"
    SEARCH_RESULTS_LIMIT = 10
    SEARCH_CATEGORIES = ["general", "science", "tech"]
    
    # Mind Map Settings
    SIMILARITY_THRESHOLD = 0.5
    MINDMAP_OUTPUT_DIR = BASE_DIR / "mindmaps"
    
    # Model Training Settings
    MODEL_OUTPUT_DIR = BASE_DIR / "trained_models"
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    TRAINING_EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5

    SEARXNG_INSTANCES = [
        "https://searx.be",
        "https://searx.fmac.xyz",
        "https://searx.tiekoetter.com",
        "https://searx.lyxx.ca",
        "https://searx.nicfab.eu"
    ]
    SEARXNG_TIMEOUT = 10