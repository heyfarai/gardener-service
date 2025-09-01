import uuid
import time
import logging
import asyncio
import json
from typing import List
from datetime import datetime

import numpy as np

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
    """Process text using PostgreSQL with VECTOR/BYTEA fallback."""
    import numpy as np
    from services.utils import cosine_similarity
    
    # Try VECTOR operations first, fallback to BYTEA
    try:
        # Find the closest topic using vector operations
        topic_record = await conn.fetchrow(
            "SELECT topic_id, label, centroid FROM topics ORDER BY centroid <=> $1::vector LIMIT 1", v
        )
        use_vector = True
    except Exception:
        # Fallback to BYTEA operations
        topic_records = await conn.fetch("SELECT topic_id, label, centroid FROM topics")
        topic_record = None
        best_similarity = -1
        
        for record in topic_records:
            if record['centroid']:
                centroid = np.frombuffer(record['centroid'], dtype=np.float32).tolist()
                similarity = cosine_similarity(v, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    topic_record = record
        use_vector = False

    topic_id = None
    if topic_record:
        if use_vector:
            # Cosine similarity is 1 - cosine distance
            similarity = 1 - (await conn.fetchval("SELECT $1::vector <=> $2::vector", v, topic_record['centroid']))
        else:
            centroid = np.frombuffer(topic_record['centroid'], dtype=np.float32).tolist()
            similarity = cosine_similarity(v, centroid)
            
        if similarity > 0.8: # Similarity threshold
            topic_id = topic_record['topic_id']
            # Update topic centroid (EMA)
            alpha = 0.1
            if use_vector:
                new_centroid = [(1 - alpha) * c + alpha * v[i] for i, c in enumerate(topic_record['centroid'])]
                await conn.execute(
                    "UPDATE topics SET centroid = $1, updated_at = NOW() WHERE topic_id = $2",
                    new_centroid, topic_id
                )
            else:
                current_centroid = np.frombuffer(topic_record['centroid'], dtype=np.float32).tolist()
                new_centroid = [(1 - alpha) * c + alpha * v[i] for i, c in enumerate(current_centroid)]
                new_centroid_blob = np.array(new_centroid, dtype=np.float32).tobytes()
                await conn.execute(
                    "UPDATE topics SET centroid = $1, updated_at = NOW() WHERE topic_id = $2",
                    new_centroid_blob, topic_id
                )

    if not topic_id:
        # Create a new topic
        topic_id = str(uuid.uuid4())
        title = f"Topic {topic_id[:8]}"
        
        if use_vector:
            await conn.execute(
                "INSERT INTO topics (topic_id, label, centroid, created_at, updated_at) VALUES ($1, $2, $3, NOW(), NOW())",
                topic_id, title, v
            )
        else:
            embedding_blob = np.array(v, dtype=np.float32).tobytes()
            await conn.execute(
                "INSERT INTO topics (topic_id, label, centroid, created_at, updated_at) VALUES ($1, $2, $3, NOW(), NOW())",
                topic_id, title, embedding_blob
            )

    # Insert the snippet
    if use_vector:
        await conn.execute(
            "INSERT INTO snippets (topic_id, text, embedding, created_at) VALUES ($1, $2, $3, NOW())",
            topic_id, text, v
        )
    else:
        embedding_blob = np.array(v, dtype=np.float32).tobytes()
        await conn.execute(
            "INSERT INTO snippets (topic_id, text, embedding, created_at) VALUES ($1, $2, $3, NOW())",
            topic_id, text, embedding_blob
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
