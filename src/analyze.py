"""
Module for analyzing transcript content and extracting key topics.
"""
import os
import requests
import json
import time
from config import (
    API_PROVIDER, 
    OPENAI_API_KEY, OPENAI_MODEL_NAME, 
    HF_API_KEY, HF_MODEL_NAME, HF_MODELS,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    TEMPERATURE, USE_CHAIN_OF_THOUGHT, COT_STEPS
)

def extract_technical_topics(transcript_text):
    """
    Extract key technical topics from the transcript using an LLM.
    
    Args:
        transcript_text (str): Preprocessed transcript text
        
    Returns:
        list: List of identified technical topics
    """
    if USE_CHAIN_OF_THOUGHT:
        prompt = _build_cot_prompt_for_topics(transcript_text)
    else:
        prompt = f"""
        Please analyze the following engineering meeting transcript and identify 
        the key technical topics being discussed. Return only a list of topics.
        
        Transcript:
        {transcript_text}
        
        Key technical topics:
        """
    
    if API_PROVIDER == "openai":
        return _extract_topics_openai(prompt)
    elif API_PROVIDER == "huggingface":
        return _extract_topics_huggingface(prompt)
    elif API_PROVIDER == "ollama":
        return _extract_topics_ollama(prompt)
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}")

def _build_cot_prompt_for_topics(transcript_text):
    """Build a chain-of-thought prompt for topic extraction."""
    prompt = f"""
    Please analyze the following engineering meeting transcript and identify 
    the key technical topics being discussed.
    
    Transcript:
    {transcript_text}
    
    Let's think through this step by step:
    
    """
    
    for i, step in enumerate(COT_STEPS, 1):
        prompt += f"{i}. {step}\n"
    
    prompt += "\nBased on this analysis, the key technical topics are:"
    
    return prompt

def _extract_topics_openai(prompt):
    """Use OpenAI API to extract topics."""
    import openai
    openai.api_key = OPENAI_API_KEY
    
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a technical analyst specializing in identifying key engineering topics from meeting transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    
    topics_text = response.choices[0].message.content
    
    # Parse the response to extract topics
    topics = _parse_topics_from_response(topics_text)
    
    return topics

def _extract_topics_huggingface(prompt):
    """Use Hugging Face API to extract topics, trying multiple models if needed."""
    # Try the specified model first
    models_to_try = [HF_MODEL_NAME]
    
    # If the specified model isn't in our list, add the default models as fallbacks
    if HF_MODEL_NAME not in HF_MODELS:
        models_to_try.extend(HF_MODELS)
    
    last_error = None
    
    for model in models_to_try:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {}
            
            if HF_API_KEY:
                headers["Authorization"] = f"Bearer {HF_API_KEY}"
            
            # For text generation models
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": TEMPERATURE,
                    "max_length": 1000,
                    "return_full_text": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 429:  # Rate limit
                print(f"Rate limited on model {model}. Waiting 5 seconds...")
                time.sleep(5)
                continue
                
            if response.status_code != 200:
                print(f"Error with model {model}: {response.status_code} - {response.text}")
                continue
                
            result = response.json()
            
            # Handle different response formats based on model type
            if isinstance(result, list) and "generated_text" in result[0]:
                topics_text = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                topics_text = result["generated_text"]
            else:
                topics_text = str(result)
            
            # Parse the response to extract topics
            topics = _parse_topics_from_response(topics_text)
            
            if topics:  # If we got valid topics, return them
                print(f"Successfully used model: {model}")
                return topics
                
        except Exception as e:
            last_error = e
            print(f"Error with model {model}: {str(e)}")
            continue
    
    # If we get here, all models failed
    if last_error:
        raise last_error
    return ["Failed to extract topics"]

def _extract_topics_ollama(prompt):
    """Use Ollama API to extract topics."""
    API_URL = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        topics_text = result.get("response", "")
        
        # Parse the response to extract topics
        topics = _parse_topics_from_response(topics_text)
        
        return topics
    except Exception as e:
        print(f"Error with Ollama: {str(e)}")
        return ["Failed to extract topics with Ollama"]

def _parse_topics_from_response(response_text):
    """Parse topics from the model's response text."""
    # First, look for a list format (numbered or bulleted)
    lines = response_text.split('\n')
    topics = []
    
    # Try to find a section that looks like a list of topics
    in_list_section = False
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if this line starts a list of topics
        if "topics" in line.lower() and ":" in line:
            in_list_section = True
            continue
            
        # If we're in the list section, extract topics
        if in_list_section:
            # Check for numbered or bulleted list items
            if (line[0].isdigit() and line[1:3] in ['. ', ') ']) or line[0] in ['-', '*', '•']:
                # Remove the bullet/number
                if line[0].isdigit():
                    topic = line[line.find(' ')+1:].strip()
                else:
                    topic = line[1:].strip()
                topics.append(topic)
            elif line.lower().startswith(('key topics', 'technical topics', 'main topics')):
                # This might be a header, skip it
                continue
            elif ":" in line and not line.endswith(':'):  # Might be a topic with description
                topic = line.split(':', 1)[0].strip()
                topics.append(topic)
            elif len(topics) > 0:
                # If we've already found topics and hit a line that doesn't match our patterns,
                # we might be out of the topics section
                break
    
    # If we didn't find a structured list, try to extract any phrases that look like topics
    if not topics:
        # Look for lines that end with a period and aren't too long
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                # Remove any numbering
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:]
                elif line[0] in ['-', '*', '•']:
                    line = line[1:]
                
                line = line.strip()
                if line:
                    topics.append(line)
    
    # Clean up topics
    cleaned_topics = []
    for topic in topics:
        # Remove trailing punctuation
        while topic and topic[-1] in ['.', ',', ';', ':', '!', '?']:
            topic = topic[:-1]
        
        # Add if not empty and not a duplicate
        if topic and topic not in cleaned_topics:
            cleaned_topics.append(topic)
    
    return cleaned_topics