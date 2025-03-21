"""
Tests for the generate module.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.generate import generate_probing_questions, save_results

@patch('openai.ChatCompletion.create')
def test_generate_probing_questions(mock_create):
    """Test that generate_probing_questions returns a list of questions."""
    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "1. Question one?\n2. Question two?\n3. Question three?"
    mock_create.return_value = mock_response
    
    topics = ["Topic 1", "Topic 2"]
    transcript = "Sample transcript text"
    
    questions = generate_probing_questions(topics, transcript)
    
    assert isinstance(questions, list)
    assert len(questions) == 3
    assert all(q.endswith('?') for q in questions)

def test_save_results(tmpdir):
    """Test that save_results creates a JSON file with the expected content."""
    # Create a temporary directory for test outputs
    output_dir = tmpdir.mkdir("outputs")
    
    # Patch the OUTPUT_DIR constant
    with patch('src.generate.OUTPUT_DIR', str(output_dir)):
        transcript_path = "test_transcript.txt"
        topics = ["Topic 1", "Topic 2"]
        questions = ["Question 1?", "Question 2?"]
        
        output_path = save_results(transcript_path, topics, questions)
        
        # Check that the file exists
        assert os.path.exists(output_path)
        
        # Check the content
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        assert data["transcript_file"] == transcript_path
        assert data["identified_topics"] == topics
        assert data["probing_questions"] == questions