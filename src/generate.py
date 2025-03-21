"""
Module for generating probing questions based on transcript analysis.
"""
import os
import json
import requests
import argparse
import time
from datetime import datetime
from pathlib import Path

from config import (
    API_PROVIDER, 
    OPENAI_API_KEY, OPENAI_MODEL_NAME, 
    HF_API_KEY, HF_MODEL_NAME, HF_MODELS,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    TEMPERATURE, OUTPUT_DIR, USE_CHAIN_OF_THOUGHT, COT_STEPS
)
from src.ingest import read_transcript, preprocess_transcript
from src.analyze import extract_technical_topics

def generate_probing_questions(topics, transcript_context):
    """
    Generate probing questions for the identified technical topics.
    
    Args:
        topics (list): List of technical topics
        transcript_context (str): Preprocessed transcript text for context
        
    Returns:
        list: List of generated probing questions
    """
    if USE_CHAIN_OF_THOUGHT:
        prompt = _build_cot_prompt_for_questions(topics, transcript_context)
    else:
        prompt = f"""
        You are an AI assistant helping a non-technical Product Manager understand a technical engineering discussion.
        
        Based on the following technical topics discussed in an engineering meeting, generate a set of 5-10 probing 
        questions that will help the PM understand the technical content when answered by another LLM.
        
        The questions should:
        - Ask for explanations of technical concepts in business-friendly terms
        - Explore how these technical decisions impact project planning and execution
        - Identify potential risks, dependencies, or trade-offs in the technical approach
        - Provide context that helps the PM communicate effectively with stakeholders
        
        Topics: {', '.join(topics)}
        
        Meeting context:
        {transcript_context[:2000]}...
        
        Generate PM-focused probing questions:
        """
    
    # Generate questions based on the configured API provider
    if API_PROVIDER == "openai":
        questions = _generate_questions_openai(prompt)
    elif API_PROVIDER == "huggingface":
        questions = _generate_questions_huggingface(prompt)
    elif API_PROVIDER == "ollama":
        questions = _generate_questions_ollama(prompt)
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}")
    
    # Ensure we have at least one question
    if not questions:
        questions = ["Failed to generate questions. Please try again."]
    
    return questions

def _build_cot_prompt_for_questions(topics, transcript_context):
    """Build a chain-of-thought prompt for question generation."""
    # Truncate transcript to fit in context window
    max_transcript_len = 1500
    truncated_transcript = transcript_context[:max_transcript_len]
    if len(transcript_context) > max_transcript_len:
        truncated_transcript += "... [transcript truncated]"
    
    prompt = f"""
    You are an AI assistant helping a non-technical Product Manager understand a technical engineering discussion.
    
    Your task is to generate probing questions about these technical topics from an engineering meeting:
    
    Topics: {', '.join(topics)}
    
    Meeting context:
    {truncated_transcript}
    
    These questions will be fed to another LLM that will answer them to help the PM understand the technical content.
    The questions should help the PM understand:
    - What the technical concepts mean in business-friendly terms
    - How these technical decisions impact project timelines, resources, and deliverables
    - What risks or dependencies might exist in the proposed technical approach
    - What context they need to effectively communicate with stakeholders about this work
    
    Think step by step:
    1. Identify technical terms or concepts that would be unfamiliar to a non-technical PM
    2. Consider what business implications these technical decisions might have
    3. Think about how this technical work relates to project planning and execution
    4. Identify potential risks or dependencies that should be highlighted
    5. Create specific questions that would prompt an LLM to provide PM-relevant explanations
    
    Generate 5-10 probing questions that will help the PM understand the engineering discussion.
    """
    
    return prompt

def _generate_questions_openai(prompt):
    """Use OpenAI API to generate questions."""
    import openai
    openai.api_key = OPENAI_API_KEY
    
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a technical translator who helps Product Managers understand engineering discussions by generating insightful questions that bridge technical and business contexts."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    
    questions_text = response.choices[0].message.content
    
    # Parse the response to extract questions
    questions = _parse_questions_from_response(questions_text)
    
    return questions

def _generate_questions_huggingface(prompt):
    """Use Hugging Face API to generate questions, trying multiple models if needed."""
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
                    "max_length": 2000,
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
                questions_text = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                questions_text = result["generated_text"]
            else:
                questions_text = str(result)
            
            # Parse the response to extract questions
            questions = _parse_questions_from_response(questions_text)
            
            if questions:  # If we got valid questions, return them
                print(f"Successfully used model: {model}")
                return questions
                
        except Exception as e:
            last_error = e
            print(f"Error with model {model}: {str(e)}")
            continue
    
    # If we get here, all models failed
    if last_error:
        raise last_error
    return ["Failed to generate questions"]

def _generate_questions_ollama(prompt):
    """Use Ollama API to generate questions."""
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
        
        questions_text = result.get("response", "")
        
        # Parse the response to extract questions
        questions = _parse_questions_from_response(questions_text)
        
        return questions
    except Exception as e:
        print(f"Error with Ollama: {str(e)}")
        return ["Failed to generate questions with Ollama"]

def _parse_questions_from_response(response_text):
    """Parse questions from the model's response text."""
    questions = []
    
    # Split by newlines and process each line
    for line in response_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if the line is a question (ends with a question mark)
        if line.endswith('?'):
            # Remove any numbering at the beginning
            if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                line = line[3:]
            elif line[0] in ['-', '*', '•']:
                line = line[1:]
                
            line = line.strip()
            if line and line not in questions:
                questions.append(line)
    
    # If we didn't find any questions with question marks, try to find sentences that look like questions
    if not questions:
        for line in response_text.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Look for lines that start with question words
            question_starters = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
            
            # Remove any numbering
            if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                line = line[3:]
            elif line[0] in ['-', '*', '•']:
                line = line[1:]
                
            line = line.strip()
            
            # Check if it starts with a question word
            if any(line.lower().startswith(starter) for starter in question_starters):
                if line and line not in questions:
                    # Add a question mark if it doesn't have one
                    if not line.endswith('?'):
                        line += '?'
                    questions.append(line)
    
    return questions

def save_results(transcript_path, topics, questions):
    """
    Save the analysis results to a JSON file.
    
    Args:
        transcript_path (str): Path to the original transcript file
        topics (list): List of identified technical topics
        questions (list): List of generated probing questions
        
    Returns:
        str: Path to the saved output file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(transcript_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_analysis_{timestamp}.json")
    
    # Categorize questions for PM use
    categorized_questions = _categorize_questions(questions)
    
    results = {
        "transcript_file": transcript_path,
        "analysis_timestamp": timestamp,
        "api_provider": API_PROVIDER,
        "model": _get_current_model(),
        "chain_of_thought": USE_CHAIN_OF_THOUGHT,
        "identified_topics": topics,
        "probing_questions": questions,
        "categorized_questions": categorized_questions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return output_path

def _get_current_model():
    """Get the name of the current model being used."""
    if API_PROVIDER == "openai":
        return OPENAI_MODEL_NAME
    elif API_PROVIDER == "huggingface":
        return HF_MODEL_NAME
    elif API_PROVIDER == "ollama":
        return OLLAMA_MODEL
    else:
        return "unknown"
    
def _categorize_questions(questions):
    """
    Categorize questions to make them more useful for the PM.
    
    Args:
        questions (list): List of generated questions
        
    Returns:
        dict: Questions categorized by their focus area
    """
    categories = {
        "Technical Concepts": [],
        "Business Implications": [],
        "Project Planning": [],
        "Risks & Dependencies": [],
        "Stakeholder Communication": [],
        "Other": []
    }
    
    # Simple keyword-based categorization
    for question in questions:
        q_lower = question.lower()
        
        if any(term in q_lower for term in ["what is", "how does", "explain", "define", "mean"]):
            categories["Technical Concepts"].append(question)
        elif any(term in q_lower for term in ["business", "value", "benefit", "cost", "roi", "impact"]):
            categories["Business Implications"].append(question)
        elif any(term in q_lower for term in ["timeline", "schedule", "resource", "team", "effort", "time", "plan"]):
            categories["Project Planning"].append(question)
        elif any(term in q_lower for term in ["risk", "challenge", "issue", "problem", "concern", "depend", "trade"]):
            categories["Risks & Dependencies"].append(question)
        elif any(term in q_lower for term in ["stakeholder", "communicate", "explain to", "present", "executive"]):
            categories["Stakeholder Communication"].append(question)
        else:
            categories["Other"].append(question)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def main():
    parser = argparse.ArgumentParser(description="Generate probing questions from engineering meeting transcripts or recordings")
    parser.add_argument("input_path", help="Path to the transcript file or audio/video recording (m4a/mp4)")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain of thought reasoning")
    parser.add_argument("--provider", choices=["openai", "huggingface", "ollama"], help="Override the API provider")
    parser.add_argument("--model", help="Override the model to use")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use for audio transcription (default: base)")
    args = parser.parse_args()
    
    # Override settings if specified
    if args.no_cot:
        global USE_CHAIN_OF_THOUGHT
        USE_CHAIN_OF_THOUGHT = False
        
    if args.provider:
        global API_PROVIDER
        API_PROVIDER = args.provider
        
    if args.model:
        if API_PROVIDER == "openai":
            global OPENAI_MODEL_NAME
            OPENAI_MODEL_NAME = args.model
        elif API_PROVIDER == "huggingface":
            global HF_MODEL_NAME
            HF_MODEL_NAME = args.model
        elif API_PROVIDER == "ollama":
            global OLLAMA_MODEL
            OLLAMA_MODEL = args.model
    
    # Determine input type and process accordingly
    input_ext = Path(args.input_path).suffix.lower()
    if input_ext in ['.m4a', '.mp4']:
        print("Detected audio/video input. Transcribing...")
        from src.transcribe import transcribe_audio
        transcript_text = transcribe_audio(args.input_path, args.whisper_model)
    else:
        print("Detected text input. Reading transcript...")
        transcript_text = read_transcript(args.input_path)
    
    # Preprocess the transcript
    preprocessed_text = preprocess_transcript(transcript_text)
    
    print(f"\nUsing API provider: {API_PROVIDER}")
    print(f"Chain of thought reasoning: {'Enabled' if USE_CHAIN_OF_THOUGHT else 'Disabled'}")
    
    # Analyze and generate questions
    print("\nExtracting technical topics...")
    topics = extract_technical_topics(preprocessed_text)
    
    print("\nGenerating probing questions...")
    questions = generate_probing_questions(topics, preprocessed_text)
    
    # Save and display results
    output_path = save_results(args.input_path, topics, questions)
    
    print(f"\nAnalysis complete! Results saved to: {output_path}\n")
    print("Identified Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    # Get categorized questions
    categorized_questions = _categorize_questions(questions)
    
    print("\nProbing Questions by Category:")
    for category, category_questions in categorized_questions.items():
        print(f"\n{category}:")
        for i, question in enumerate(category_questions, 1):
            print(f"  {i}. {question}")

if __name__ == "__main__":
    main()