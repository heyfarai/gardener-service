import httpx
import logging
import hashlib
import random
from typing import List
from config import USE_OPENAI, OPENAI_API_KEY

logger = logging.getLogger(__name__)

def embed(text: str) -> List[float]:
    """Generate embeddings using OpenAI API or deterministic fallback."""
    if USE_OPENAI and OPENAI_API_KEY:
        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
    
    # Fallback to deterministic embeddings
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10**6
    random.seed(seed)
    return [random.random() for _ in range(384)]
