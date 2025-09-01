import os
import time
import uuid
import math
import logging
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

import asyncpg
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import auto-titler
from autotitler import AutoTitler, auto_titler

# Load environment variables from .env file
load_dotenv()

# Environment configuration
API_KEY = os.getenv("GARDENER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Auto-titler configuration
AUTO_TITLE_THRESHOLD = int(os.getenv("AUTO_TITLE_THRESHOLD", "5"))  # Min snippets to trigger auto-titling
AUTO_TITLE_ENABLED = os.getenv("AUTO_TITLE_ENABLED", "true").lower() == "true"

# Initialize auto-titler
auto_titler = AutoTitler(ANTHROPIC_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for database
db_pool = None

class TopicStatus(str, Enum):
    SEEDLING = "seedling"
    SAPLING = "sapling"
    MATURE = "mature"
    ARCHIVED = "archived"

class TopicMetadata(BaseModel):
    id: str
    label: str
    status: TopicStatus = Field(default=TopicStatus.SEEDLING)
    label_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    alias: List[str] = Field(default_factory=list)
    blurb: Optional[str] = None
    snippet_count: int = 0
    created_at: datetime
    updated_at: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    logger.info("Starting gardener service...")
    
    # Initialize database connection
    if DATABASE_URL and DATABASE_URL.strip():
        try:
            db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                max_queries=50000,
                max_inactive_connection_lifetime=300,  # 5 minutes
                command_timeout=60,
                server_settings={
                    'jit': 'off'
                }
            )
            logger.info("Database connection pool established")
            
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id VARCHAR PRIMARY KEY,
                        label VARCHAR NOT NULL,
                        centroid FLOAT[] NOT NULL,
                        num_snippets INTEGER NOT NULL DEFAULT 0,
                        status VARCHAR NOT NULL DEFAULT 'seedling',
                        label_confidence FLOAT NOT NULL DEFAULT 0.0,
                        keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                        blurb TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS snippets (
                        id VARCHAR PRIMARY KEY,
                        topic_id VARCHAR NOT NULL REFERENCES topics(id),
                        text TEXT NOT NULL,
                        embedding FLOAT[] NOT NULL,
                        chat_id VARCHAR NOT NULL,
                        timestamp VARCHAR NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
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
    ts: Optional[str] = Field(default=None, description="Message timestamp")
    seedling: Optional[str] = None

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
                "SELECT id, label, status, label_confidence, keywords, blurb, num_snippets, created_at, updated_at FROM topics ORDER BY updated_at DESC"
            )
            return {"topics": [dict(t) for t in topics]}
    else:
        topic_list = []
        for tid, meta in TOPICS.items():
            snippet_count = len([s for s in SNIPS if s["topic_id"] == tid])
            topic_list.append({
                "id": tid,
                "label": meta["label"],
                "status": meta["status"],
                "label_confidence": meta["label_confidence"],
                "keywords": meta["keywords"],
                "blurb": meta["blurb"],
                "snippet_count": snippet_count,
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"]
            })
        return {"topics": topic_list}

@app.get("/topics")
async def topics():
    try:
        logger.info("Fetching all topics")
        
        if db_pool:
            try:
                async with await get_db_connection() as conn:
                    ts = await conn.fetch("SELECT id, label, status, label_confidence, keywords, blurb, num_snippets, created_at, updated_at FROM topics ORDER BY updated_at DESC")
                    result = [{"id": t["id"], "label": t["label"], "status": t["status"], "label_confidence": t["label_confidence"], "keywords": t["keywords"], "blurb": t["blurb"], "snippet_count": t["num_snippets"], "created_at": t["created_at"].isoformat(), "updated_at": t["updated_at"].isoformat()} for t in ts]
            except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
                logger.error(f"Database connection failed during topics fetch: {e}")
                raise HTTPException(503, "Database temporarily unavailable")
        else:
            result = [{"id": tid, "label": meta["label"], "status": meta["status"], "label_confidence": meta["label_confidence"], "keywords": meta["keywords"], "blurb": meta["blurb"], "snippet_count": len([s for s in SNIPS if s["topic_id"] == tid]), "created_at": meta["created_at"].isoformat(), "updated_at": meta["updated_at"].isoformat()} for tid,meta in TOPICS.items()]
        
        logger.info(f"Retrieved {len(result)} topics")
        return result
    
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(500, f"Failed to fetch topics: {str(e)}")

@app.post("/turn")
async def turn(t: Turn, authorization: Optional[str] = Header(None)):
    if authorization != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Process incoming turn data
    chat_id = t.chat_id
    turn_id = t.turn_id
    user_text = t.user_text.strip() if t.user_text else ""
    model_text = t.model_text or ""
    model = t.model
    seedling = t.seedling  # New field from frontend

    # Generate embedding for user text
    if user_text:
        embedding = await embed(user_text)
    else:
        raise HTTPException(status_code=400, detail="User text is required")

    # Process text in the database
    await process_text_db(user_text, embedding, chat_id, t.ts)

    return {"message": "Turn processed successfully", "turn_id": turn_id}

async def get_db_connection():
    """Get database connection with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return db_pool.acquire()
        except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5 * (attempt + 1))

async def process_text_db(text: str, v: List[float], chat_id: str, ts: str):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with await get_db_connection() as conn:
                topics = await conn.fetch(
                    "SELECT id, label, centroid, num_snippets, status FROM topics ORDER BY centroid <=> $1::vector LIMIT 1",
                    v
                )
                if topics and len(topics) > 0:
                    topic = topics[0]
                    topic_id = topic['id']
                    similarity = cosine_similarity(v, topic['centroid'])
                    alpha = 0.1
                    new_centroid = [(1 - alpha) * c + alpha * v[i] for i, c in enumerate(topic['centroid'])]
                    await conn.execute(
                        "UPDATE topics SET centroid = $1, num_snippets = num_snippets + 1, updated_at = NOW() WHERE id = $2",
                        new_centroid, topic_id
                    )
                else:
                    topic_id = str(uuid.uuid4())
                    label = f"Topic {topic_id[:8]}"
                    await conn.execute(
                        "INSERT INTO topics (id, label, centroid, num_snippets, status, label_confidence, keywords, created_at, updated_at) VALUES ($1, $2, $3, 1, $4, 0.0, '{}', NOW(), NOW())",
                        topic_id, label, v, TopicStatus.SEEDLING.value
                    )
                snippet_id = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO snippets (id, topic_id, text, embedding, chat_id, timestamp, created_at) VALUES ($1, $2, $3, $4, $5, $6, NOW())",
                    snippet_id, topic_id, text, v, chat_id, ts or datetime.utcnow().isoformat()
                )
                if AUTO_TITLE_ENABLED:
                    snippet_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM snippets WHERE topic_id = $1",
                        topic_id
                    )
                    if snippet_count >= AUTO_TITLE_THRESHOLD:
                        topic_status = await conn.fetchval(
                            "SELECT status FROM topics WHERE id = $1",
                            topic_id
                        )
                        if topic_status == TopicStatus.SEEDLING.value:
                            await trigger_auto_titling(topic_id, conn)
                return
        except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError, asyncpg.InternalClientError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Database connection error after {max_retries} attempts: {str(e)}")
                raise
            await asyncio.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Error processing text in database: {str(e)}")
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
        TOPICS[best_id] = {"label": label, "centroid": v, "status": TopicStatus.SEEDLING.value, "label_confidence": 0.0, "keywords": [], "blurb": None, "snippet_count": 0, "created_at": time.time(), "updated_at": time.time()}
    
    sid = f"s_{uuid.uuid4().hex[:6]}"
    SNIPS.append({"id": sid, "chat_id": chat_id, "ts": ts, "text": text, "vec": v, "topic_id": best_id})
    
    # Update centroid (EMA)
    c = TOPICS[best_id]["centroid"]; alpha = 0.15
    TOPICS[best_id]["centroid"] = [(1-alpha)*c[i] + alpha*v[i] for i in range(len(c))]
    TOPICS[best_id]["updated_at"] = time.time()

async def trigger_auto_titling(topic_id: str, conn: asyncpg.Connection) -> None:
    try:
        topic = await conn.fetchrow(
            "SELECT id, label, status, num_snippets FROM topics WHERE id = $1",
            topic_id
        )
        if not topic:
            logger.warning(f"Topic {topic_id} not found for auto-titling")
            return
        snippets = await conn.fetch(
            "SELECT s.text, (s.embedding <=> t.centroid) as similarity, s.created_at FROM snippets s JOIN topics t ON s.topic_id = t.id WHERE s.topic_id = $1 ORDER BY similarity ASC, s.created_at DESC LIMIT 10",
            topic_id
        )
        if not snippets:
            logger.warning(f"No snippets found for topic {topic_id}")
            return
        snippets_data = [
            {'text': s['text'], 'similarity': float(1 - s['similarity']), 'timestamp': s['created_at'].isoformat()}
            for s in snippets
        ]
        title, keywords, confidence, blurb = await auto_titler.generate_topic_metadata(snippets_data)
        if not title or not keywords:
            logger.warning(f"Failed to generate metadata for topic {topic_id}")
            return
        await conn.execute(
            "UPDATE topics SET label = COALESCE($1, label), status = 'sapling', label_confidence = $2, keywords = $3, blurb = $4, updated_at = NOW() WHERE id = $5",
            title, confidence, keywords, blurb, topic_id
        )
        logger.info(f"Auto-titled topic {topic_id}: {title} (confidence: {confidence:.2f})")
    except Exception as e:
        logger.error(f"Error in auto-titling for topic {topic_id}: {str(e)}", exc_info=True)

@app.get("/topics/search")
async def search(q: str, k: int = 8):
    try:
        if not q.strip():
            raise HTTPException(400, "Query cannot be empty")
        
        logger.info(f"Searching topics for query: {q[:50]}...")
        v = embed(q)
        
        if db_pool:
            try:
                async with await get_db_connection() as conn:
                    topics = await conn.fetch("SELECT id, label, centroid FROM topics")
                    scored = [(t['id'], t['label'], cosine_similarity(v, t['centroid'])) for t in topics]
            except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
                logger.error(f"Database connection failed during search: {e}")
                raise HTTPException(503, "Database temporarily unavailable")
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
            try:
                async with await get_db_connection() as conn:
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
                        "updated": topic["updated_at"].isoformat()
                    }
            except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
                logger.error(f"Database connection failed during brief generation: {e}")
                raise HTTPException(503, "Database temporarily unavailable")
        else:
            topic = TOPICS.get(topic_id)
            if not topic:
                raise HTTPException(404, f"Topic {topic_id} not found")
            
            sims = [(s, cosine_similarity(s["vec"], topic["centroid"])) for s in SNIPS if s["topic_id"]==topic_id]
            sims.sort(key=lambda x: x[1], reverse=True)
            picks = [s["text"] for s,_ in sims[:5]]
            txt = " • ".join(picks)[:max_tokens*4]
            return {"id": topic_id, "label": topic["label"], "summary_markdown": txt, "updated": topic["updated_at"]}
    
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
            try:
                async with await get_db_connection() as conn:
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
            except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
                logger.error(f"Database connection failed during retrieval: {e}")
                raise HTTPException(503, "Database temporarily unavailable")
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