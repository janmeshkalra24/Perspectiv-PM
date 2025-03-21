"""
Module for ingesting and preprocessing transcript files.
"""

def read_transcript(file_path):
    """
    Read a transcript file and return its contents as a string.
    
    Args:
        file_path (str): Path to the transcript file
        
    Returns:
        str: Contents of the transcript file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading transcript file: {e}")

def preprocess_transcript(transcript_text):
    """
    Preprocess the transcript text to prepare it for analysis.
    
    Args:
        transcript_text (str): Raw transcript text
        
    Returns:
        str: Preprocessed transcript text
    """
    # Remove unnecessary whitespace
    cleaned_text = ' '.join(transcript_text.split())
    
    # Additional preprocessing steps could be added here
    # - Remove timestamps
    # - Normalize speaker identifiers
    # - Handle special characters
    
    return cleaned_text