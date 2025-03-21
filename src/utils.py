"""
Utility functions for the transcript analyzer.
"""
import os
import json

def load_json_file(file_path):
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str): Path to save the JSON file
        
    Returns:
        str: Path to the saved file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return file_path

def get_file_list(directory, extension=None):
    """
    Get a list of files in a directory, optionally filtered by extension.
    
    Args:
        directory (str): Directory path
        extension (str, optional): File extension to filter by
        
    Returns:
        list: List of file paths
    """
    files = []
    for filename in os.listdir(directory):
        if extension is None or filename.endswith(extension):
            files.append(os.path.join(directory, filename))
    return files