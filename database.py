import asyncpg
import logging
from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Global connection pool
db_pool = None

async def create_pool():
    """Create a database connection pool for PostgreSQL."""
    global db_pool
    if db_pool:
        return

    try:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is required - SQLite support has been removed")
        
        logger.info("Connecting to PostgreSQL database...")
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        logger.info("PostgreSQL connection pool established.")
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
        # PostgreSQL: Use pgvector for embeddings
        async with db_pool.acquire() as connection:
            # Drop existing tables if they exist
            await connection.execute('DROP TABLE IF EXISTS snippets CASCADE')
            await connection.execute('DROP TABLE IF EXISTS topics CASCADE')
            
            # Try to create the vector extension, but continue if it fails
            try:
                await connection.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                logger.info("Vector extension created or already exists")
            except Exception as e:
                logger.warning(f"Could not create vector extension: {e}")
                logger.warning("Continuing without vector extension - some features may be limited")
            
            # First, create tables without vector columns
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
                    centroid BYTEA,  -- Store embeddings as bytes if vector extension is not available
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await connection.execute("""
                CREATE TABLE snippets (
                    id SERIAL PRIMARY KEY,
                    topic_id VARCHAR(255) REFERENCES topics(topic_id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    embedding BYTEA,  -- Store embeddings as bytes if vector extension is not available
                    chat_id TEXT,
                    ts TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Try to alter the tables to use vector type if the extension is available
            try:
                await connection.execute('ALTER TABLE topics ALTER COLUMN centroid TYPE VECTOR(384) USING centroid::VECTOR;')
                await connection.execute('ALTER TABLE snippets ALTER COLUMN embedding TYPE VECTOR(384) USING embedding::VECTOR;')
                logger.info("Successfully converted columns to VECTOR type")
            except Exception as e:
                logger.warning(f"Could not convert columns to VECTOR type: {e}")
                logger.warning("Using BYTEA for vector storage - some features may be limited")
        
        logger.info("Database initialized with updated schema.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
