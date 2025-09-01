from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import os, time, uuid, math, logging
import asyncpg
from openai import AsyncOpenAI
from contextlib import asynccontextmanager

# Environment configuration
API_KEY = os.getenv("GARDENER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database and embedding model
db_pool = None
embedding_model = None
openai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, embedding_model, openai_client
    logger.info("Starting gardener service...")
    
    # Initialize database connection
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL)
            logger.info("Database connection established")
            
            # Create tables if they don't exist
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id VARCHAR PRIMARY KEY,
                        label VARCHAR NOT NULL,
                        centroid FLOAT[] NOT NULL,
                        updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    # Initialize embedding model
    try:
        if USE_OPENAI and OPENAI_API_KEY:
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI embeddings initialized")
        else:
            logger.info("Using fallback embeddings (set USE_OPENAI=true for better results)")
    except Exception as e:
        logger.error(f"Embedding model initialization failed: {e}")
    
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

async def embed(text: str) -> List[float]:
    """Generate embeddings using OpenAI or fallback"""
    try:
        if USE_OPENAI and openai_client:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        else:
            # Fallback to deterministic fake embeddings
            import hashlib, random
            random.seed(int(hashlib.md5(text.encode()).hexdigest(),16) % 10**6)
            return [random.random() for _ in range(384)]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Fallback to fake embeddings
        import hashlib, random
        random.seed(int(hashlib.md5(text.encode()).hexdigest(),16) % 10**6)
        return [random.random() for _ in range(384)]

def cosine(a,b):
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if not na or not nb: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

class Turn(BaseModel):
    chat_id: str
    turn_id: str
    user_text: Optional[str] = ""
    model_text: Optional[str] = ""
    model: Optional[str] = None
    ts: Optional[str] = None

@app.get("/health")
def health(): 
    try:
        return {
            "ok": True, 
            "database": "connected" if db_pool else "memory",
            "embedding": "openai" if (USE_OPENAI and openai_client) else "fallback",
            "cors_origins": CORS_ORIGINS
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"ok": False, "error": str(e)}

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
        topic_list.sort(key=lambda x: x["updated"], reverse=True)
        return {"topics": topic_list}

@app.post("/turn")
async def turn(t: Turn, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad token")
    
    try:
        for text in [t.user_text, t.model_text]:
            if not text: continue
            v = await embed(text)
            
            # Use database if available, otherwise fallback to memory
            if db_pool:
                await process_text_db(text, v, t.chat_id, t.ts)
            else:
                await process_text_memory(text, v, t.chat_id, t.ts)
        
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        raise HTTPException(500, "Internal server error")

async def process_text_db(text: str, v: List[float], chat_id: str, ts: str):
    """Process text using database storage"""
    async with db_pool.acquire() as conn:
        # Find nearest topic
        topics = await conn.fetch("SELECT id, label, centroid FROM topics")
        best_id, best_score = None, 0.0
        
        for topic in topics:
            centroid = topic['centroid']
            score = cosine(v, centroid)
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
        else:
            # Update centroid with EMA
            current_centroid = await conn.fetchval(
                "SELECT centroid FROM topics WHERE id = $1", best_id
            )
            alpha = 0.15
            new_centroid = [(1-alpha)*current_centroid[i] + alpha*v[i] for i in range(len(v))]
            await conn.execute(
                "UPDATE topics SET centroid = $1, updated = CURRENT_TIMESTAMP WHERE id = $2",
                new_centroid, best_id
            )
        
        # Insert snippet
        sid = f"s_{uuid.uuid4().hex[:6]}"
        await conn.execute(
            "INSERT INTO snippets (id, chat_id, ts, text, embedding, topic_id) VALUES ($1, $2, $3, $4, $5, $6)",
            sid, chat_id, ts, text, v, best_id
        )

async def process_text_memory(text: str, v: List[float], chat_id: str, ts: str):
    """Process text using in-memory storage (fallback)"""
    best_id, best = None, 0.0
    for tid, meta in TOPICS.items():
        s = cosine(v, meta["centroid"])
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
    v = await embed(q)
    
    if db_pool:
        async with db_pool.acquire() as conn:
            topics = await conn.fetch("SELECT id, label, centroid FROM topics")
            scored = [(t['id'], t['label'], cosine(v, t['centroid'])) for t in topics]
    else:
        scored = [(tid, meta["label"], cosine(v, meta["centroid"])) for tid,meta in TOPICS.items()]
    
    scored.sort(key=lambda x: x[2], reverse=True)
    return {"query": q, "results": [{"id": tid, "label": lbl, "score": float(sc)} for tid,lbl,sc in scored[:k]]}

@app.get("/topics/{topic_id}/brief")
async def brief(topic_id: str, max_tokens: int = 600):
    if db_pool:
        async with db_pool.acquire() as conn:
            topic = await conn.fetchrow("SELECT * FROM topics WHERE id = $1", topic_id)
            if not topic: raise HTTPException(404, "no topic")
            
            snippets = await conn.fetch(
                "SELECT text, embedding FROM snippets WHERE topic_id = $1", topic_id
            )
            sims = [(s['text'], cosine(s['embedding'], topic['centroid'])) for s in snippets]
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
        if not topic: raise HTTPException(404, "no topic")
        sims = [(s, cosine(s["vec"], topic["centroid"])) for s in SNIPS if s["topic_id"]==topic_id]
        sims.sort(key=lambda x: x[1], reverse=True)
        picks = [s["text"] for s,_ in sims[:5]]
        txt = " • ".join(picks)[:max_tokens*4]
        return {"id": topic_id, "label": topic["label"], "summary_markdown": txt, "updated": topic["updated"]}

class RetrieveReq(BaseModel):
    query: Optional[str] = None
    topic_id: Optional[str] = None
    k: int = 5

@app.post("/retrieve")
async def retrieve(r: RetrieveReq):
    topic_vec = None
    query_vec = await embed(r.query) if r.query else None
    
    if db_pool:
        async with db_pool.acquire() as conn:
            if r.topic_id:
                topic_vec = await conn.fetchval(
                    "SELECT centroid FROM topics WHERE id = $1", r.topic_id
                )
            
            snippets = await conn.fetch(
                "SELECT id, text, chat_id, embedding FROM snippets"
            )
            scored = []
            for s in snippets:
                score = 0.0
                if topic_vec: score += 0.6 * cosine(s['embedding'], topic_vec)
                if query_vec: score += 0.4 * cosine(s['embedding'], query_vec)
                scored.append((s, score))
    else:
        topic_vec = TOPICS.get(r.topic_id, {}).get("centroid") if r.topic_id else None
        scored = []
        for s in SNIPS:
            score = 0.0
            if topic_vec: score += 0.6 * cosine(s["vec"], topic_vec)
            if query_vec: score += 0.4 * cosine(s["vec"], query_vec)
            scored.append((s, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if db_pool:
        out = [{"rank": i+1, "text": s["text"], "chat_id": s["chat_id"], "snippet_id": s["id"]} for i,(s,_) in enumerate(scored[:r.k])]
    else:
        out = [{"rank": i+1, "text": s["text"], "chat_id": s["chat_id"], "snippet_id": s["id"]} for i,(s,_) in enumerate(scored[:r.k])]
    
    return {"items": out}