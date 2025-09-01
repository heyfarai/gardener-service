import os
import time
import uuid
import math
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

import asyncpg
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Environment configuration
API_KEY = os.getenv("GARDENER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for database
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    logger.info("Starting gardener service...")
    
    # Initialize database connection
    if DATABASE_URL and DATABASE_URL.strip():
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL)
            logger.info("Database connection established")
            
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id VARCHAR PRIMARY KEY,
                        label VARCHAR NOT NULL,
                        centroid FLOAT[] NOT NULL,
                        updated TIMESTAMP DEFAULT NOW()
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS snippets (
                        id VARCHAR PRIMARY KEY,
                        chat_id VARCHAR NOT NULL,
                        ts VARCHAR,
                        text TEXT NOT NULL,
                        embedding FLOAT[] NOT NULL,
                        topic_id VARCHAR REFERENCES topics(id)
                    )
                """)
                logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Falling back to in-memory storage")
            db_pool = None
    else:
        logger.info("Using in-memory storage (no DATABASE_URL provided)")
    
    logger.info(f"Environment check - USE_OPENAI: {USE_OPENAI}, OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
    if USE_OPENAI and OPENAI_API_KEY:
        logger.info("OpenAI embeddings configured and ready")
    else:
        logger.warning("Using fallback embeddings - set USE_OPENAI=true and provide OPENAI_API_KEY for better results")
    
    logger.info("Gardener service startup complete")
    yield
    
    # Cleanup
    logger.info("Shutting down gardener service...")
    if db_pool:
        await db_pool.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True
)

# Fallback in-memory stores for development
SNIPS = [] # {id, chat_id, ts, text, vec, topic_id}
TOPICS = {} # id -> {label, centroid, updated}

def embed(text: str) -> List[float]:
    """Generate embeddings using OpenAI API or deterministic fallback."""
    if USE_OPENAI and OPENAI_API_KEY:
        try:
            response = requests.post(
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
    import hashlib
    import random
    
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10**6
    random.seed(seed)
    return [random.random() for _ in range(384)]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if not norm_a or not norm_b:
        return 0.0
    
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)

class Turn(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    chat_id: str
    turn_id: str
    user_text: Optional[str] = ""
    model_text: Optional[str] = ""
    model: Optional[str] = None
    ts: Optional[str] = None


class RetrieveReq(BaseModel):
    query: Optional[str] = None
    topic_id: Optional[str] = None
    k: int = 5

@app.get("/")
def root():
    return {"service": "gardener", "status": "running"}

@app.get("/health")
def health(): 
    return {
        "status": "healthy",
        "database": "connected" if db_pool else "memory",
        "embedding": "openai" if (USE_OPENAI and OPENAI_API_KEY) else "fallback"
    }

@app.get("/topics")
async def list_topics():
    """List all topics with metadata"""
    if db_pool:
        async with db_pool.acquire() as conn:
            topics = await conn.fetch(
                "SELECT id, label, updated, (SELECT COUNT(*) FROM snippets WHERE topic_id = topics.id) as snippet_count FROM topics ORDER BY updated DESC"
            )
            return {"topics": [dict(t) for t in topics]}
    else:
        topic_list = []
        for tid, meta in TOPICS.items():
            snippet_count = len([s for s in SNIPS if s["topic_id"] == tid])
            topic_list.append({
                "id": tid,
                "label": meta["label"],
                "updated": meta["updated"],
                "snippet_count": snippet_count
            })
        return {"topics": topic_list}

@app.get("/topics")
async def topics():
    try:
        logger.info("Fetching all topics")
        
        if db_pool:
            async with db_pool.acquire() as conn:
                ts = await conn.fetch("SELECT id, label, updated FROM topics ORDER BY updated DESC")
                result = [{"id": t["id"], "label": t["label"], "updated": t["updated"].isoformat()} for t in ts]
        else:
            result = [{"id": tid, "label": meta["label"], "updated": meta["updated"]} for tid,meta in TOPICS.items()]
        
        logger.info(f"Retrieved {len(result)} topics")
        return result
    
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(500, f"Failed to fetch topics: {str(e)}")

@app.post("/turn")
async def turn(t: Turn, authorization: Optional[str] = Header(None)):
    try:
        if API_KEY and authorization != f"Bearer {API_KEY}":
            raise HTTPException(401, "unauthorized")
        
        text = f"{t.user_text} {t.model_text}".strip()
        if not text:
            logger.warning(f"Empty text received for chat_id: {t.chat_id}")
            return {"status": "empty"}
        
        logger.info(f"Processing turn for chat_id: {t.chat_id}, text length: {len(text)}")
        
        v = embed(text)
        ts = t.ts or str(int(time.time()))
        
        if db_pool:
            await process_text_db(text, v, t.chat_id, ts)
        else:
            await process_text_memory(text, v, t.chat_id, ts)
        
        logger.info(f"Successfully processed turn for chat_id: {t.chat_id}")
        return {"status": "processed", "chat_id": t.chat_id, "turn_id": t.turn_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing turn for chat_id {t.chat_id}: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

async def process_text_db(text: str, v: List[float], chat_id: str, ts: str):
    """Process text using database storage"""
    try:
        async with db_pool.acquire() as conn:
            # Find nearest topic
            topics = await conn.fetch("SELECT id, label, centroid FROM topics")
            best_id, best_score = None, 0.0
            
            for topic in topics:
                centroid = topic['centroid']
                score = cosine_similarity(v, centroid)
                if score > best_score:
                    best_score = score
                    best_id = topic['id']
            
            # Create new topic if similarity too low
            if best_id is None or best_score < 0.78:
                label = " ".join(text.split()[:4]).strip(" ,.") or f"Topic {len(topics)+1}"
                best_id = f"topic_{uuid.uuid4().hex[:6]}"
                await conn.execute(
                    "INSERT INTO topics (id, label, centroid) VALUES ($1, $2, $3)",
                    best_id, label, v
                )
                logger.info(f"Created new topic: {best_id} - {label}")
            else:
                # Update centroid with EMA (alpha=0.1)
                old_centroid = await conn.fetchval(
                    "SELECT centroid FROM topics WHERE id = $1", best_id
                )
                alpha = 0.1
                new_centroid = [(1-alpha)*old_centroid[i] + alpha*v[i] for i in range(len(v))]
                await conn.execute(
                    "UPDATE topics SET centroid = $1, updated = NOW() WHERE id = $2",
                    new_centroid, best_id
                )
                logger.debug(f"Updated topic centroid: {best_id}")
            
            # Store snippet
            sid = f"s_{uuid.uuid4().hex[:6]}"
            await conn.execute(
                "INSERT INTO snippets (id, chat_id, ts, text, embedding, topic_id) VALUES ($1, $2, $3, $4, $5, $6)",
                sid, chat_id, ts, text, v, best_id
            )
            logger.debug(f"Stored snippet: {sid} in topic: {best_id}")
    
    except Exception as e:
        logger.error(f"Database processing error: {e}")
        raise

async def process_text_memory(text: str, v: List[float], chat_id: str, ts: str):
    """Process text using in-memory storage (fallback)"""
    best_id, best = None, 0.0
    for tid, meta in TOPICS.items():
        s = cosine_similarity(v, meta["centroid"])
        if s > best: best, best_id = s, tid
    
    if best_id is None or best < 0.78:
        label = " ".join(text.split()[:4]).strip(" ,.") or f"Topic {len(TOPICS)+1}"
        best_id = f"topic_{uuid.uuid4().hex[:6]}"
        TOPICS[best_id] = {"label": label, "centroid": v, "updated": time.time()}
    
    sid = f"s_{uuid.uuid4().hex[:6]}"
    SNIPS.append({"id": sid, "chat_id": chat_id, "ts": ts, "text": text, "vec": v, "topic_id": best_id})
    
    # Update centroid (EMA)
    c = TOPICS[best_id]["centroid"]; alpha = 0.15
    TOPICS[best_id]["centroid"] = [(1-alpha)*c[i] + alpha*v[i] for i in range(len(c))]
    TOPICS[best_id]["updated"] = time.time()

@app.get("/topics/search")
async def search(q: str, k: int = 8):
    try:
        if not q.strip():
            raise HTTPException(400, "Query cannot be empty")
        
        logger.info(f"Searching topics for query: {q[:50]}...")
        v = embed(q)
        
        if db_pool:
            async with db_pool.acquire() as conn:
                topics = await conn.fetch("SELECT id, label, centroid FROM topics")
                scored = [(t['id'], t['label'], cosine_similarity(v, t['centroid'])) for t in topics]
        else:
            scored = [(tid, meta["label"], cosine_similarity(v, meta["centroid"])) for tid,meta in TOPICS.items()]
        
        scored.sort(key=lambda x: x[2], reverse=True)
        results = [{"id": tid, "label": lbl, "score": float(sc)} for tid,lbl,sc in scored[:k]]
        
        logger.info(f"Found {len(results)} topic matches")
        return {"query": q, "results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Topic search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.get("/topics/{topic_id}/brief")
async def brief(topic_id: str, max_tokens: int = 600):
    try:
        logger.info(f"Generating brief for topic: {topic_id}")
        
        if db_pool:
            async with db_pool.acquire() as conn:
                topic = await conn.fetchrow("SELECT * FROM topics WHERE id = $1", topic_id)
                if not topic:
                    raise HTTPException(404, f"Topic {topic_id} not found")
                
                snippets = await conn.fetch(
                    "SELECT text, embedding FROM snippets WHERE topic_id = $1", topic_id
                )
                sims = [(s['text'], cosine_similarity(s['embedding'], topic['centroid'])) for s in snippets]
                sims.sort(key=lambda x: x[1], reverse=True)
                picks = [text for text, _ in sims[:5]]
                
                return {
                    "id": topic_id, 
                    "label": topic["label"], 
                    "summary_markdown": " • ".join(picks)[:max_tokens*4], 
                    "updated": topic["updated"].isoformat()
                }
        else:
            topic = TOPICS.get(topic_id)
            if not topic:
                raise HTTPException(404, f"Topic {topic_id} not found")
            
            sims = [(s, cosine_similarity(s["vec"], topic["centroid"])) for s in SNIPS if s["topic_id"]==topic_id]
            sims.sort(key=lambda x: x[1], reverse=True)
            picks = [s["text"] for s,_ in sims[:5]]
            txt = " • ".join(picks)[:max_tokens*4]
            return {"id": topic_id, "label": topic["label"], "summary_markdown": txt, "updated": topic["updated"]}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brief generation error for topic {topic_id}: {e}")
        raise HTTPException(500, f"Brief generation failed: {str(e)}")


@app.post("/retrieve")
async def retrieve(r: RetrieveReq):
    try:
        if not r.query and not r.topic_id:
            raise HTTPException(400, "Either query or topic_id must be provided")
        
        logger.info(f"Retrieving snippets - query: {r.query[:50] if r.query else 'None'}, topic_id: {r.topic_id}, k: {r.k}")
        
        topic_vec = None
        query_vec = embed(r.query) if r.query else None
        
        if db_pool:
            async with db_pool.acquire() as conn:
                if r.topic_id:
                    topic_vec = await conn.fetchval(
                        "SELECT centroid FROM topics WHERE id = $1", r.topic_id
                    )
                    if topic_vec is None:
                        raise HTTPException(404, f"Topic {r.topic_id} not found")
                
                snippets = await conn.fetch(
                    "SELECT id, text, chat_id, embedding FROM snippets"
                )
                scored = []
                for s in snippets:
                    score = 0.0
                    if topic_vec: score += 0.6 * cosine_similarity(s['embedding'], topic_vec)
                    if query_vec: score += 0.4 * cosine_similarity(s['embedding'], query_vec)
                    scored.append((s, score))
        else:
            if r.topic_id and r.topic_id not in TOPICS:
                raise HTTPException(404, f"Topic {r.topic_id} not found")
            
            topic_vec = TOPICS.get(r.topic_id, {}).get("centroid") if r.topic_id else None
            scored = []
            for s in SNIPS:
                score = 0.0
                if topic_vec: score += 0.6 * cosine_similarity(s["vec"], topic_vec)
                if query_vec: score += 0.4 * cosine_similarity(s["vec"], query_vec)
                scored.append((s, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if db_pool:
            out = [{"rank": i+1, "text": s["text"], "chat_id": s["chat_id"], "snippet_id": s["id"]} for i,(s,_) in enumerate(scored[:r.k])]
        else:
            out = [{"rank": i+1, "text": s["text"], "chat_id": s["chat_id"], "snippet_id": s["id"]} for i,(s,_) in enumerate(scored[:r.k])]
        
        logger.info(f"Retrieved {len(out)} snippets")
        return {"items": out}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(500, f"Retrieval failed: {str(e)}")