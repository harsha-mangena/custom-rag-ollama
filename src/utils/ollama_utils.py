import aiohttp
import logging
from typing import List, Optional
import json

logger = logging.getLogger(__name__)

async def get_available_models() -> List[str]:
    """
    Fetch available models from Ollama API.
    
    Returns:
        List of available model names
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return sorted(models) if models else []
                else:
                    logger.error(f"Failed to fetch models: {await response.text()}")
                    return []
    except Exception as e:
        logger.error(f"Error connecting to Ollama service: {e}")
        return []

async def check_ollama_status() -> bool:
    """
    Check if Ollama service is running and accessible.
    
    Returns:
        True if service is available, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                return response.status == 200
    except Exception:
        return False

async def generate_response(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Generate response using Ollama model.
    
    Args:
        model: Name of the model to use
        prompt: Main prompt text
        system_prompt: Optional system prompt
        
    Returns:
        Generated response or None if failed
    """
    try:
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response")
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {error_text}")
                    return None
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return None