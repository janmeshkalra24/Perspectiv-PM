import pytest
from fastapi.testclient import TestClient
from ..api.app import app

client = TestClient(app)

def test_text_summary_endpoint():
    # Test data
    test_text = "This is a test text that needs to be summarized. It contains multiple sentences. The summary should capture the main points."
    test_context = ["This is related context", "More context information"]
    
    # Make request to the endpoint
    response = client.post(
        "/api/text_summary",
        json={
            "text": test_text,
            "context": test_context
        }
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "key_points" in data
    assert "sentiment" in data
    assert isinstance(data["summary"], str)
    assert isinstance(data["key_points"], list)
    assert isinstance(data["sentiment"], str) 