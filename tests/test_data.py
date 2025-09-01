import pytest
from httpx import AsyncClient
from config import API_KEY


@pytest.mark.asyncio
async def test_turn_unauthorized(test_client: AsyncClient):
    response = await test_client.post("/turn", json={
        "chat_id": "test_chat",
        "turn_id": "test_turn",
        "user_text": "Hello, world!"
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_turn_bad_request(test_client: AsyncClient):
    headers = {"Authorization": API_KEY}
    response = await test_client.post("/turn", headers=headers, json={
        "chat_id": "test_chat",
        "turn_id": "test_turn",
        "user_text": ""
    })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_retrieve_bad_request(test_client: AsyncClient):
    response = await test_client.post("/retrieve", json={})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_turn_success(test_client: AsyncClient):
    headers = {"Authorization": API_KEY}
    response = await test_client.post("/turn", headers=headers, json={
        "chat_id": "test_chat_success",
        "turn_id": "test_turn_success",
        "user_text": "This is a successful test."
    })
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_retrieve_success(test_client: AsyncClient):
    # Ingest data first
    headers = {"Authorization": API_KEY}
    turn_data = {
        "chat_id": "retrieval_test_chat",
        "turn_id": "retrieval_test_turn",
        "user_text": "The sky is blue and the grass is green."
    }
    ingest_response = await test_client.post("/turn", headers=headers, json=turn_data)
    assert ingest_response.status_code == 200

    # Now retrieve
    retrieve_data = {"query": "color of the sky", "k": 1}
    retrieve_response = await test_client.post("/retrieve", json=retrieve_data)
    assert retrieve_response.status_code == 200
    json_response = retrieve_response.json()
    assert "items" in json_response
    assert len(json_response["items"]) > 0
