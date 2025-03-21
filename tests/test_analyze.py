"""
Tests for the analyze module.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.analyze import extract_technical_topics

@patch('openai.ChatCompletion.create')
def test_extract_technical_topics(mock_create):
    """Test that extract_technical_topics returns a list of topics."""
    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Topic 1\nTopic 2\nTopic 3"
    mock_create.return_value = mock_response
    
    topics = extract_technical_topics("Sample transcript text")
    
    assert isinstance(topics, list)
    assert len(topics) == 3
    assert topics == ["Topic 1", "Topic 2", "Topic 3"]