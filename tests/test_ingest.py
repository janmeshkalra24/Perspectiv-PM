"""
Tests for the ingest module.
"""
import os
import pytest
from src.ingest import read_transcript, preprocess_transcript

def test_read_transcript_file_not_found():
    """Test that FileNotFoundError is raised when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        read_transcript("nonexistent_file.txt")

def test_preprocess_transcript():
    """Test that preprocessing removes extra whitespace."""
    raw_text = "This  is  a\n\ntest   transcript."
    processed_text = preprocess_transcript(raw_text)
    assert processed_text == "This is a test transcript."