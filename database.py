import asyncpg
import aiosqlite
import logging
from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Global connection pool
db_pool = None

async def create_pool():
    """Create a database connection pool for PostgreSQL or SQLite."""
    global db_pool
    if db_pool:
        return

    try:
        if DATABASE_URL:
            logger.info("Connecting to PostgreSQL database...")
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
            logger.info("PostgreSQL connection pool established.")
        else:
            logger.info("Using SQLite database (gardener.db)...")
            db = await aiosqlite.connect("gardener.db")
            db.row_factory = aiosqlite.Row  # Access columns by name
            db_pool = db
            logger.info("SQLite database connection established.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_pool = None

async def close_pool():
    """Close the database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed.")

def get_db_pool():
    """Get the database connection pool."""
    return db_pool

async def init_db():
    """Initialize the database by creating tables if they don't exist."""
    if not db_pool:
        logger.error("Database pool not available, cannot initialize DB.")
        return

    try:
        if DATABASE_URL:
            # PostgreSQL: Use pgvector for embeddings
            async with db_pool.acquire() as connection:
                # Drop existing tables if they exist
                await connection.execute('DROP TABLE IF EXISTS snippets CASCADE')
                await connection.execute('DROP TABLE IF EXISTS topics CASCADE')
                
                await connection.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                await connection.execute("""
                    CREATE TABLE topics (
                        id SERIAL PRIMARY KEY,
                        topic_id VARCHAR(255) UNIQUE NOT NULL,
                        label VARCHAR(255),
                        status VARCHAR(50) DEFAULT 'active',
                        label_confidence FLOAT,
                        keywords TEXT,
                        blurb TEXT,
                        num_snippets INTEGER DEFAULT 0,
                        centroid VECTOR(384),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                await connection.execute("""
                    CREATE TABLE snippets (
                        id SERIAL PRIMARY KEY,
                        topic_id VARCHAR(255) REFERENCES topics(topic_id) ON DELETE CASCADE,
                        text TEXT NOT NULL,
                        embedding VECTOR(384),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        else:
            # SQLite: Store embeddings as BLOBs
            await db_pool.execute('DROP TABLE IF EXISTS snippets')
            await db_pool.execute('DROP TABLE IF EXISTS topics')
            
            await db_pool.execute("""
                CREATE TABLE topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id TEXT UNIQUE NOT NULL,
                    label TEXT,
                    status TEXT DEFAULT 'active',
                    label_confidence REAL,
                    keywords TEXT,
                    blurb TEXT,
                    num_snippets INTEGER DEFAULT 0,
                    centroid BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await db_pool.execute("""
                CREATE TABLE snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id TEXT,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    chat_id TEXT,
                    ts TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (topic_id) REFERENCES topics(topic_id) ON DELETE CASCADE
                );
            """)
            await db_pool.commit()
        logger.info("Database initialized with updated schema.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
