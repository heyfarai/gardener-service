import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_root(test_client: AsyncClient):
    response = await test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "gardener", "status": "running"}

@pytest.mark.asyncio
async def test_health(test_client: AsyncClient):
    response = await test_client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert "database" in json_response
    assert "embedding" in json_response
