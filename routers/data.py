import logging
import numpy as np
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from config import API_KEY, DATABASE_URL
from models import Turn, RetrieveReq
from services.embedding_service import embed
from services.topic_service import process_text_db
from database import get_db_pool
from services.utils import cosine_similarity

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/turn")
async def turn(t: Turn, authorization: Optional[str] = Header(None)):
    if authorization != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_text = t.user_text.strip() if t.user_text else ""
    if not user_text:
        raise HTTPException(status_code=400, detail="User text is required")

    embedding = embed(user_text)
    await process_text_db(user_text, embedding, t.chat_id, t.ts)

    return {"message": "Turn processed successfully", "turn_id": t.turn_id}


@router.post("/retrieve")
async def retrieve(r: RetrieveReq):
    try:
        if not r.query and not r.topic_id:
            raise HTTPException(400, "Either query or topic_id must be provided")

        logger.info(f"Retrieving snippets - query: {r.query[:50] if r.query else 'None'}, topic_id: {r.topic_id}, k: {r.k}")

        query_vec = embed(r.query) if r.query else None
        db = get_db_pool()

        scored = []
        if DATABASE_URL:
            async with db.acquire() as conn:
                scored = await _retrieve_postgres(conn, r, query_vec)
        else:
            scored = await _retrieve_sqlite(db, r, query_vec)

        scored.sort(key=lambda x: x[1], reverse=True)
        out = [{"rank": i + 1, "text": s["text"], "chat_id": s.get("chat_id"), "snippet_id": s["id"]} for i, (s, _) in enumerate(scored[:r.k])]

        logger.info(f"Retrieved {len(out)} snippets")
        return {"items": out}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(500, f"Retrieval failed: {str(e)}")

async def _retrieve_postgres(conn, r: RetrieveReq, query_vec):
    topic_vec = None
    if r.topic_id:
        topic_vec = await conn.fetchval("SELECT centroid FROM topics WHERE topic_id = $1", r.topic_id)
        if topic_vec is None:
            raise HTTPException(404, f"Topic {r.topic_id} not found")

    snippets = await conn.fetch("SELECT id, text, chat_id, embedding FROM snippets")
    scored = []
    for s in snippets:
        score = 0.0
        if topic_vec: score += 0.6 * cosine_similarity(s['embedding'], topic_vec)
        if query_vec: score += 0.4 * cosine_similarity(s['embedding'], query_vec)
        scored.append((s, score))
    return scored

async def _retrieve_sqlite(db, r: RetrieveReq, query_vec):
    topic_vec = None
    if r.topic_id:
        cursor = await db.execute("SELECT centroid FROM topics WHERE topic_id = ?", (r.topic_id,))
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(404, f"Topic {r.topic_id} not found")
        topic_vec = np.frombuffer(row['centroid'], dtype=np.float32).tolist()

    cursor = await db.execute("SELECT id, text, embedding FROM snippets")
    snippets = await cursor.fetchall()
    scored = []
    for s in snippets:
        embedding = np.frombuffer(s['embedding'], dtype=np.float32).tolist()
        score = 0.0
        if topic_vec: score += 0.6 * cosine_similarity(embedding, topic_vec)
        if query_vec: score += 0.4 * cosine_similarity(embedding, query_vec)
        scored.append((dict(s), score))
    return scored
