from typing import Dict, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import requests
from config.settings import Settings
import logging

class EmbeddingManager:
    def __init__(self, hf_token: Optional[str] = None):
        self.models: Dict[str, SentenceTransformer] = {}
        self.hf_token = hf_token
        self._setup_logging()
        self._load_models()
    
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """Load embedding models."""
        for model_name, model_path in Settings.EMBEDDING_MODELS.items():
            try:
                model = SentenceTransformer(model_path)
                if torch.cuda.is_available():
                    model.to('cuda')
                self.models[model_name] = model
                self.logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")

    async def get_embedding(
        self, 
        text: str, 
        model_name: str = Settings.DEFAULT_EMBEDDING_MODEL
    ) -> Tuple[np.ndarray, float]:
        """Get embedding for text using specified model."""
        if model_name in self.models:
            return self._get_local_embedding(text, model_name)
        elif model_name == "huggingface" and self.hf_token:
            return await self._get_hf_embedding(text)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _get_local_embedding(
        self, 
        text: str, 
        model_name: str
    ) -> Tuple[np.ndarray, float]:
        """Get embedding using local model."""
        model = self.models[model_name]
        embedding = model.encode(text)
        return embedding, 0.0  # TODO: Add timing

    async def _get_hf_embedding(self, text: str) -> Tuple[np.ndarray, float]:
        """Get embedding using Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(
            Settings.HF_API_URL,
            headers=headers,
            json={"inputs": text}, 
        timeout=60)
        
        if response.status_code == 200:
            return np.array(response.json()), 0.0  # TODO: Add timing
        else:
            raise Exception(f"HF API error: {response.text}")
