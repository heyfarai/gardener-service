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
            async with db.acquire() as conn:
                # Try VECTOR operations first, fallback to BYTEA
                try:
                    topics = await conn.fetch("SELECT topic_id, label, centroid <-> $1 AS distance FROM topics ORDER BY distance LIMIT $2", v, k)
                    scored = [(t['topic_id'], t['label'], 1 - t['distance']) for t in topics]
                except Exception:
                    # Fallback for BYTEA storage
                    import numpy as np
                    topics = await conn.fetch("SELECT topic_id, label, centroid FROM topics")
                    scored = []
                    for t in topics:
                        if t['centroid']:
                            centroid = np.frombuffer(t['centroid'], dtype=np.float32).tolist()
                            similarity = cosine_similarity(v, centroid)
                            scored.append((t['topic_id'], t['label'], similarity))
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
        async with db.acquire() as conn:
            return await _get_brief_postgres(conn, topic_id, max_tokens)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brief generation error for topic {topic_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Brief generation failed: {str(e)}")

async def _get_brief_postgres(conn, topic_id: str, max_tokens: int):
    topic = await conn.fetchrow("SELECT * FROM topics WHERE topic_id = $1", topic_id)
    if not topic:
        raise HTTPException(404, f"Topic {topic_id} not found")

    # Handle both VECTOR and BYTEA storage for snippets
    try:
        snippets = await conn.fetch(
            "SELECT text, embedding <-> $1 AS distance FROM snippets WHERE topic_id = $2 ORDER BY distance LIMIT 5",
            topic['centroid'], topic_id
        )
    except Exception:
        # Fallback for BYTEA storage - get all snippets and sort manually
        import numpy as np
        from services.utils import cosine_similarity
        
        all_snippets = await conn.fetch(
            "SELECT text, embedding FROM snippets WHERE topic_id = $1", topic_id
        )
        
        if topic['centroid']:
            centroid = np.frombuffer(topic['centroid'], dtype=np.float32).tolist()
            snippet_distances = []
            for snippet in all_snippets:
                if snippet['embedding']:
                    embedding = np.frombuffer(snippet['embedding'], dtype=np.float32).tolist()
                    distance = 1 - cosine_similarity(centroid, embedding)
                    snippet_distances.append((snippet['text'], distance))
            
            snippet_distances.sort(key=lambda x: x[1])
            snippets = [{'text': text, 'distance': dist} for text, dist in snippet_distances[:5]]
        else:
            snippets = [{'text': s['text'], 'distance': 0} for s in all_snippets[:5]]
    picks = [s['text'] for s in snippets]

    return {
        "id": topic_id,
        "label": topic["label"],
        "summary_markdown": " • ".join(picks)[:max_tokens * 4],
        "updated": topic["updated_at"].isoformat()
    }

