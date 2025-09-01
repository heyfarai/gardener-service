import pytest
import pytest_asyncio
import sys
import os
import asyncio
from httpx import AsyncClient, ASGITransport

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from database import get_db_pool, close_pool, create_pool, init_db

# Use in-memory SQLite for tests
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="function")
async def test_db():
    """Set up test database with tables."""
    # Create a new in-memory database for each test
    os.environ["DATABASE_URL"] = ""  # Empty to force SQLite
    
    # Create and initialize the database
    await create_pool()
    pool = get_db_pool()
    if not pool:
        raise Exception("Failed to create database pool")
        
    # Initialize the database schema
    await init_db()
    
    yield pool
    
    # Clean up
    await close_pool()
    if os.path.exists("test.db"):
        os.remove("test.db")

@pytest_asyncio.fixture(scope="function")
async def test_client(test_db):
    """Create a test client with a clean database for each test."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
