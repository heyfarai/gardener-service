import pytest
from httpx import AsyncClient
from config import API_KEY
from database import get_db_pool

@pytest.mark.asyncio
async def test_list_topics(test_client: AsyncClient):
    response = await test_client.get("/topics")
    assert response.status_code == 200
    json_response = response.json()
    assert "topics" in json_response
    assert isinstance(json_response["topics"], list)

@pytest.mark.asyncio
async def test_search_topics(test_client: AsyncClient):
    response = await test_client.get("/topics/search?q=test")
    assert response.status_code == 200
    json_response = response.json()
    assert "query" in json_response
    assert "results" in json_response
    assert isinstance(json_response["results"], list)

@pytest.mark.asyncio
async def test_search_topics_empty_query(test_client: AsyncClient):
    response = await test_client.get("/topics/search?q=")
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_get_topic_brief_not_found(test_client: AsyncClient):
    response = await test_client.get("/topics/non_existent_topic/brief")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_topic_brief_success(test_client: AsyncClient):
    # First, create a topic
    headers = {"Authorization": API_KEY}
    turn_data = {
        "chat_id": "brief_test_chat",
        "turn_id": "brief_test_turn_1",
        "user_text": "This is the first turn for the brief test."
    }
    ingest_response = await test_client.post("/turn", headers=headers, json=turn_data)
    assert ingest_response.status_code == 200, f"Failed to ingest test data: {ingest_response.text}"
    
    # Get the latest topic ID from the database
    db = get_db_pool()
    cursor = await db.execute("SELECT topic_id FROM topics ORDER BY created_at DESC LIMIT 1")
    topic = await cursor.fetchone()
    assert topic is not None, "No topics found in the database"
    topic_id = topic["topic_id"]
    
    # Retrieve the brief using the topic ID from the database
    response = await test_client.get(f"/topics/{topic_id}/brief")
    assert response.status_code == 200, f"Failed to get topic brief: {response.text}"
    json_response = response.json()
    
    # Check the response structure
    assert "id" in json_response, f"Response missing 'id' field: {json_response}"
    assert "label" in json_response, f"Response missing 'label' field: {json_response}"
    assert "summary_markdown" in json_response, f"Response missing 'summary_markdown' field: {json_response}"
    assert "updated" in json_response, f"Response missing 'updated' field: {json_response}"
