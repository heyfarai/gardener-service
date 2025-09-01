import asyncpg
import logging
import os
from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Global connection pool
db_pool = None

async def create_pool():
    """Create a database connection pool for PostgreSQL."""
    global db_pool
    if db_pool:
        return
    
    # Disable statement cache in test environment
    statement_cache_size = 0 if os.environ.get("PYTEST_CURRENT_TEST") else 100
    
    try:
        logger.info("Connecting to PostgreSQL database...")
        db_pool = await asyncpg.create_pool(
            DATABASE_URL, 
            min_size=1, 
            max_size=10,
            statement_cache_size=statement_cache_size
        )
        
        logger.info("PostgreSQL connection pool established.")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection pool: {e}")
        raise

async def close_pool():
    """Close the database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed.")

async def reset_pool():
    """Reset the database connection pool."""
    global db_pool
    if db_pool:
        await close_pool()
    await create_pool()
    logger.info("Database connection pool reset.")

def get_db_pool():
    """Get the database connection pool."""
    return db_pool

async def init_db():
    """Initialize the database by creating tables if they don't exist."""
    global db_pool
    if not db_pool:
        logger.error("Database pool not available, cannot initialize DB.")
        return

    try:
        async with db_pool.acquire() as connection:
            # Drop existing tables if they exist
            await connection.execute('DROP TABLE IF EXISTS snippets CASCADE')
            await connection.execute('DROP TABLE IF EXISTS topics CASCADE')
            
            # Create vector extension
            await connection.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            logger.info("Vector extension created")
            
            # Create tables with VECTOR columns directly
            await connection.execute("""
                CREATE TABLE topics (
                    id SERIAL PRIMARY KEY,
                    topic_id VARCHAR(255) UNIQUE NOT NULL,
                    label VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'active',
                    label_confidence FLOAT,
                    keywords TEXT,
                    blurb TEXT,
                    centroid VECTOR(1536),
                    num_snippets INT DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await connection.execute("""
                CREATE TABLE snippets (
                    id SERIAL PRIMARY KEY,
                    topic_id VARCHAR(255) REFERENCES topics(topic_id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    embedding VECTOR(1536),
                    chat_id TEXT,
                    ts TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        
        logger.info("Database initialized with VECTOR schema.")
        await reset_pool()
                    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
