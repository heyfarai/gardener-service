import uuid
import time
import logging
import asyncio
import json
from typing import List
from datetime import datetime

import numpy as np
from pgvector.asyncpg import register_vector

from models import TopicStatus
from services.utils import cosine_similarity
from autotitler import auto_titler
from config import DATABASE_URL, AUTO_TITLE_ENABLED, AUTO_TITLE_THRESHOLD
from database import get_db_pool, create_pool

logger = logging.getLogger(__name__)

# Fallback in-memory stores for development
SNIPS = [] # {id, chat_id, ts, text, vec, topic_id}
TOPICS = {} # id -> {label, centroid, updated}


async def get_db_connection():
    """Get database pool."""
    db_pool = get_db_pool()
    if not db_pool:
        # Try to create the pool if it doesn't exist
        await create_pool()
        db_pool = get_db_pool()
        if not db_pool:
            logger.warning("No database connection available, using in-memory storage")
            return None
    
    # Return the pool itself, not a connection
    return db_pool


async def process_text_db(text: str, v: List[float], chat_id: str, ts: str):
    """Process text and store it in the database."""
    db = await get_db_connection()

    try:
        async with db.acquire() as conn:
            await _process_text_postgres(conn, text, v, chat_id, ts)
    except Exception as e:
        logger.error(f"Error processing text in database: {str(e)}", exc_info=True)
        raise

async def _process_text_postgres(conn, text, v, chat_id, ts):
    """Process text using PostgreSQL with pgvector."""
    
    # Register vector type with connection
    await register_vector(conn)
    
    # Convert list to numpy array for pgvector if it's not already
    v_array = np.array(v) if not isinstance(v, np.ndarray) else v
    
    # Find similar topics using vector similarity
    similar_topics = await conn.fetch("""
        SELECT topic_id, label, 1 - (centroid <=> $1) as similarity
        FROM topics 
        WHERE centroid IS NOT NULL
        ORDER BY centroid <=> $1
        LIMIT 5
    """, v_array)

    topic_id = None
    if similar_topics:
        topic_record = similar_topics[0]
        similarity = topic_record['similarity']
        if similarity > 0.8:  # Similarity threshold
            topic_id = topic_record['topic_id']
            # Update topic centroid (EMA)
            alpha = 0.1
            current_centroid = await conn.fetchval("SELECT centroid FROM topics WHERE topic_id = $1", topic_id)
            current_centroid = np.array(current_centroid) if not isinstance(current_centroid, np.ndarray) else current_centroid
            new_centroid = (1 - alpha) * current_centroid + alpha * v_array
            await conn.execute(
                "UPDATE topics SET centroid = $1, updated_at = NOW() WHERE topic_id = $2",
                new_centroid, topic_id
            )

    if not topic_id:
        # Create a new topic
        topic_id = str(uuid.uuid4())
        title = f"Topic {topic_id[:8]}"
        
        await conn.execute(
            "INSERT INTO topics (topic_id, label, centroid, created_at, updated_at) VALUES ($1, $2, $3, NOW(), NOW())",
            topic_id, title, v_array
        )

    # Insert the snippet with numpy array for embedding
    await conn.execute(
        "INSERT INTO snippets (topic_id, text, embedding, chat_id, ts, created_at) VALUES ($1, $2, $3, $4, $5, NOW())",
        topic_id, text, v_array, chat_id, ts
    )



async def process_text_memory(text: str, v: List[float], chat_id: str, ts: str):
    """Process text using in-memory storage (fallback)"""
    best_id, best = None, 0.0
    for tid, meta in TOPICS.items():
        s = cosine_similarity(v, meta["centroid"])
        if s > best: best, best_id = s, tid
    
    if best_id is None or best < 0.78:
        label = " ".join(text.split()[:4]).strip(" ,.") or f"Topic {len(TOPICS)+1}"
        best_id = f"topic_{uuid.uuid4().hex[:6]}"
        TOPICS[best_id] = {"label": label, "centroid": v, "status": TopicStatus.SEEDLING.value, "label_confidence": 0.0, "keywords": [], "blurb": None, "snippet_count": 0, "created_at": time.time(), "updated_at": time.time()}
    
    sid = f"s_{uuid.uuid4().hex[:6]}"
    SNIPS.append({"id": sid, "chat_id": chat_id, "ts": ts, "text": text, "vec": v, "topic_id": best_id})
    
    # Update centroid (EMA)
    c = TOPICS[best_id]["centroid"]; alpha = 0.15
    TOPICS[best_id]["centroid"] = [(1-alpha)*c[i] + alpha*v[i] for i in range(len(c))]
    TOPICS[best_id]["updated_at"] = time.time()


async def trigger_auto_titling(topic_id: str, conn) -> None:
    """Generate topic metadata using PostgreSQL."""
    try:
        topic = await conn.fetchrow(
            "SELECT topic_id, label FROM topics WHERE topic_id = $1", topic_id
        )
        snippets = await conn.fetch(
            "SELECT s.text, (s.embedding <=> t.centroid) as similarity, s.created_at FROM snippets s JOIN topics t ON s.topic_id = t.topic_id WHERE s.topic_id = $1 ORDER BY similarity ASC, s.created_at DESC LIMIT 10",
            topic_id
        )

        if not topic:
            logger.warning(f"Topic {topic_id} not found for auto-titling")
            return
        if not snippets:
            logger.warning(f"No snippets found for topic {topic_id}")
            return

        snippets_data = [{'text': s['text']} for s in snippets]
        title, keywords, confidence, blurb = await auto_titler.generate_topic_metadata(snippets_data)
        
        if not title or not keywords:
            logger.warning(f"Failed to generate metadata for topic {topic_id}")
            return

        await conn.execute(
            "UPDATE topics SET label = COALESCE($1, label), blurb = $2, keywords = $3, updated_at = NOW() WHERE topic_id = $4",
            title, blurb, json.dumps(keywords), topic_id
        )

        logger.info(f"Auto-titled topic {topic_id}: {title} (confidence: {confidence:.2f})")
    except Exception as e:
        logger.error(f"Error in auto-titling for topic {topic_id}: {str(e)}", exc_info=True)
