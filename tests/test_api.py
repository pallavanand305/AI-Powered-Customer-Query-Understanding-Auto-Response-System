import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_query():
    test_query = {
        "query": "My bill is too high",
        "customer_id": "test_customer_123"
    }
    response = client.post("/analyze-query", json=test_query)
    assert response.status_code == 200
    
    data = response.json()
    assert "intent" in data
    assert "sentiment" in data
    assert "confidence" in data
    assert "response" in data
    assert data["intent"] == "billing"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "query_counts" in response.json()