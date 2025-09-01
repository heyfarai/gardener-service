from fastapi import APIRouter
from config import USE_OPENAI, OPENAI_API_KEY
from database import get_db_pool

router = APIRouter()

@router.get("/")
def root():
    return {"service": "gardener", "status": "running"}

@router.get("/health")
def health():
    db_pool = get_db_pool()
    return {
        "status": "healthy",
        "database": "connected" if db_pool else "memory",
        "embedding": "openai" if (USE_OPENAI and OPENAI_API_KEY) else "fallback"
    }
