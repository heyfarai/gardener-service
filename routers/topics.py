import logging
import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from services.embedding_service import embed
from services.utils import cosine_similarity
from database import get_db_pool
from models import TopicMetadata
from services.topic_service import TOPICS, SNIPS
from config import DATABASE_URL

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/topics", response_model=Dict[str, List[TopicMetadata]])
async def list_topics():
    """List all topics with metadata"""
    db = get_db_pool()
    if not db:
        # Fallback to in-memory data if no DB connection
        topic_list = [t.dict() for t in TOPICS.values()]
        return {"topics": topic_list}

    try:
        if DATABASE_URL:
            async with db.acquire() as conn:
                topics_raw = await conn.fetch(
                    "SELECT id, label, status, label_confidence, keywords, blurb, num_snippets, created_at, updated_at FROM topics ORDER BY updated_at DESC"
                )
                topics = [dict(t) for t in topics_raw]
        else:
            cursor = await db.execute(
                "SELECT id, label, status, label_confidence, keywords, blurb, num_snippets, created_at, updated_at FROM topics ORDER BY updated_at DESC"
            )
            topics_raw = await cursor.fetchall()
            topics = [dict(t) for t in topics_raw]
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error fetching topics from DB: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")


@router.get("/topics/search")
async def search(q: str, k: int = 8):
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")

    logger.info(f"Searching topics for query: {q[:50]}...")
    v = embed(q)
    db = get_db_pool()

    if not db:
        # Fallback to in-memory search
        scored = [(tid, meta["label"], cosine_similarity(v, meta["centroid"])) for tid, meta in TOPICS.items()]
    else:
        try:
            if DATABASE_URL:
                async with db.acquire() as conn:
                    topics = await conn.fetch("SELECT id, label, centroid <-> $1 AS distance FROM topics ORDER BY distance LIMIT $2", v, k)
                    scored = [(t['id'], t['label'], 1 - t['distance']) for t in topics]
            else:
                cursor = await db.execute("SELECT id, label, centroid FROM topics")
                topics = await cursor.fetchall()
                scored = []
                for t in topics:
                    centroid = np.frombuffer(t['centroid'], dtype=np.float32).tolist()
                    scored.append((t['id'], t['label'], cosine_similarity(v, centroid)))
        except Exception as e:
            logger.error(f"Database connection failed during search: {e}")
            raise HTTPException(503, "Database temporarily unavailable")

    scored.sort(key=lambda x: x[2], reverse=True)
    results = [{"id": tid, "label": lbl, "score": float(sc)} for tid, lbl, sc in scored[:k]]

    logger.info(f"Found {len(results)} topic matches")
    return {"query": q, "results": results}


@router.get("/topics/{topic_id}/brief")
async def brief(topic_id: str, max_tokens: int = 600):
    logger.info(f"Generating brief for topic: {topic_id}")
    db = get_db_pool()

    if not db:
        # Fallback to in-memory data
        topic = TOPICS.get(topic_id)
        if not topic:
            raise HTTPException(404, f"Topic {topic_id} not found")
        sims = [(s, cosine_similarity(s["vec"], topic["centroid"])) for s in SNIPS if s.get("topic_id") == topic_id]
        sims.sort(key=lambda x: x[1], reverse=True)
        picks = [s["text"] for s, _ in sims[:5]]
        txt = " • ".join(picks)[:max_tokens * 4]
        return {"id": topic_id, "label": topic["label"], "summary_markdown": txt, "updated": topic["updated_at"]}

    try:
        if DATABASE_URL:
            async with db.acquire() as conn:
                return await _get_brief_postgres(conn, topic_id, max_tokens)
        else:
            return await _get_brief_sqlite(db, topic_id, max_tokens)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brief generation error for topic {topic_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Brief generation failed: {str(e)}")

async def _get_brief_postgres(conn, topic_id: str, max_tokens: int):
    topic = await conn.fetchrow("SELECT * FROM topics WHERE id = $1", topic_id)
    if not topic:
        raise HTTPException(404, f"Topic {topic_id} not found")

    snippets = await conn.fetch(
        "SELECT text, embedding <-> $1 AS distance FROM snippets WHERE topic_id = $2 ORDER BY distance LIMIT 5",
        topic['centroid'], topic_id
    )
    picks = [s['text'] for s in snippets]

    return {
        "id": topic_id,
        "label": topic["label"],
        "summary_markdown": " • ".join(picks)[:max_tokens * 4],
        "updated": topic["updated_at"].isoformat()
    }

async def _get_brief_sqlite(db, topic_id: str, max_tokens: int):
    logger.info(f"Fetching topic with topic_id: {topic_id}")
    cursor = await db.execute("SELECT * FROM topics WHERE topic_id = ?", (topic_id,))
    topic = await cursor.fetchone()
    if not topic:
        # Log all available topics for debugging
        cursor = await db.execute("SELECT topic_id FROM topics")
        all_topics = await cursor.fetchall()
        logger.warning(f"Topic {topic_id} not found. Available topics: {[t['topic_id'] for t in all_topics]}")
        raise HTTPException(404, f"Topic {topic_id} not found")

    topic_centroid = np.frombuffer(topic['centroid'], dtype=np.float32).tolist() if topic['centroid'] else None
    if not topic_centroid:
        return {
            "id": topic_id,
            "label": topic["label"] if "label" in topic else "",
            "summary_markdown": "No snippets available for this topic.",
            "updated": topic["updated_at"] if "updated_at" in topic else ""
        }

    cursor = await db.execute("SELECT text, embedding FROM snippets WHERE topic_id = ?", (topic_id,))
    snippets = await cursor.fetchall()

    sims = []
    for s in snippets:
        if not s['embedding']:
            continue
        try:
            embedding = np.frombuffer(s['embedding'], dtype=np.float32).tolist()
            sims.append((s['text'], cosine_similarity(embedding, topic_centroid)))
        except Exception as e:
            logger.warning(f"Error processing snippet embedding: {e}")
            continue

    sims.sort(key=lambda x: x[1], reverse=True)
    picks = [text for text, _ in sims[:5]]

    return {
        "id": topic_id,
        "label": topic["label"] if "label" in topic else "",
        "summary_markdown": " • ".join(picks)[:max_tokens * 4] if picks else "No snippets available for this topic.",
        "updated": topic["updated_at"] if "updated_at" in topic else ""
    }
