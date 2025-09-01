import httpx
import logging
import hashlib
import random
import numpy as np
from typing import List, Dict, Any, Optional
from config import USE_OPENAI, OPENAI_API_KEY, DATABASE_URL
from database import get_db_pool

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

async def store_embedding(text: str, embedding: List[float], topic_id: str = None, chat_id: str = None, ts: str = None) -> int:
    """Store text and its embedding in the database."""
    db = get_db_pool()
    if not db:
        return -1

    try:
        if DATABASE_URL:
            async with db.acquire() as conn:
                # Try to insert as VECTOR first, fall back to BYTEA if that fails
                try:
                    return await _store_embedding_postgres(conn, text, embedding, topic_id, chat_id, ts, use_vector=True)
                except Exception as e:
                    logger.warning(f"Failed to store as VECTOR, falling back to BYTEA: {e}")
                    return await _store_embedding_postgres(conn, text, embedding, topic_id, chat_id, ts, use_vector=False)
        else:
            return await _store_embedding_sqlite(db, text, embedding, topic_id, chat_id, ts)
    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        return -1


async def _store_embedding_postgres(conn, text: str, embedding: List[float], topic_id: str = None, 
                                  chat_id: str = None, ts: str = None, use_vector: bool = True) -> int:
    """Store embedding in PostgreSQL database."""
    try:
        if use_vector:
            query = """
                INSERT INTO snippets (topic_id, text, embedding, chat_id, ts)
                VALUES ($1, $2, $3::vector, $4, $5)
                RETURNING id
            """
        else:
            # Convert embedding to bytes for BYTEA storage
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            query = """
                INSERT INTO snippets (topic_id, text, embedding, chat_id, ts)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            embedding = embedding_bytes
            
        result = await conn.fetchval(
            query,
            topic_id,
            text,
            embedding,
            chat_id,
            ts
        )
        return result
    except Exception as e:
        if 'operator does not exist' in str(e) and 'vector' in str(e):
            # If vector operations fail, re-raise to try with BYTEA
            raise ValueError("Vector operation failed, try with BYTEA")
        raise


async def _store_embedding_sqlite(db, text: str, embedding: List[float], topic_id: str = None, 
                                 chat_id: str = None, ts: str = None) -> int:
    """Store embedding in SQLite database."""
    # Convert embedding to bytes for storage
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    query = """
        INSERT INTO snippets (topic_id, text, embedding, chat_id, ts)
        VALUES (?, ?, ?, ?, ?)
    """
    cursor = await db.execute(
        query,
        (topic_id, text, embedding_bytes, chat_id, ts)
    )
    await db.commit()
    return cursor.lastrowid


async def get_similar_embeddings(embedding: List[float], limit: int = 5, topic_id: str = None) -> List[Dict[str, Any]]:
    """Find similar embeddings in the database."""
    db = get_db_pool()
    if not db:
        return []

    try:
        if DATABASE_URL:
            async with db.acquire() as conn:
                # First try with vector similarity
                try:
                    query = """
                        SELECT id, text, 1 - (embedding <=> $1::vector) as similarity
                        FROM snippets
                        WHERE ($2::text IS NULL OR topic_id = $2)
                        ORDER BY embedding <=> $1::vector
                        LIMIT $3
                    """
                    results = await conn.fetch(query, embedding, topic_id, limit)
                    return [dict(r) for r in results]
                except Exception as e:
                    if 'operator does not exist' in str(e) and 'vector' in str(e):
                        # Fall back to BYTEA comparison if vector operations fail
                        logger.warning("Vector operations not available, using BYTEA comparison")
                        return await _get_similar_embeddings_bytea(conn, embedding, limit, topic_id)
                    raise
        else:
            return await _get_similar_embeddings_sqlite(db, embedding, limit, topic_id)
    except Exception as e:
        logger.error(f"Error finding similar embeddings: {e}")
        return []


async def _get_similar_embeddings_bytea(conn, embedding: List[float], limit: int = 5, topic_id: str = None) -> List[Dict[str, Any]]:
    """Find similar embeddings using BYTEA comparison (fallback when vector extension is not available)."""
    from scipy.spatial.distance import cosine
    
    # Convert input embedding to numpy array
    query_embedding = np.array(embedding, dtype=np.float32)
    
    # Get all relevant embeddings from the database
    query = """
        SELECT id, text, embedding
        FROM snippets
        WHERE ($1::text IS NULL OR topic_id = $1)
    """
    rows = await conn.fetch(query, topic_id)
    
    # Calculate similarities
    results = []
    for row in rows:
        try:
            # Convert BYTEA to numpy array
            db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, db_embedding)
            results.append({
                'id': row['id'],
                'text': row['text'],
                'similarity': float(similarity)
            })
        except Exception as e:
            logger.warning(f"Error processing embedding {row['id']}: {e}")
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: -x['similarity'])
    return results[:limit]


async def _get_similar_embeddings_sqlite(db, embedding: List[float], limit: int = 5, topic_id: str = None) -> List[Dict[str, Any]]:
    """Find similar embeddings in SQLite database."""
    from scipy.spatial.distance import cosine
    
    # Convert input embedding to numpy array
    query_embedding = np.array(embedding, dtype=np.float32)
    
    # Get all relevant embeddings from the database
    query = """
        SELECT id, text, embedding
        FROM snippets
        WHERE (? IS NULL OR topic_id = ?)
    """
    cursor = await db.execute(query, (topic_id, topic_id))
    rows = await cursor.fetchall()
    
    # Calculate similarities
    results = []
    for row in rows:
        try:
            # Convert BLOB to numpy array
            db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, db_embedding)
            results.append({
                'id': row['id'],
                'text': row['text'],
                'similarity': float(similarity)
            })
        except Exception as e:
            logger.warning(f"Error processing embedding {row['id']}: {e}")
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: -x['similarity'])
    return results[:limit]
