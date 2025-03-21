import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
# Choose which API to use: "openai", "huggingface", "ollama", or "local"
API_PROVIDER = os.getenv("API_PROVIDER", "huggingface")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

# Hugging Face settings
HF_API_KEY = os.getenv("HF_API_KEY")  # Optional, can use without API key with rate limits
# List of models to try (in order of preference)
HF_MODELS = [
    "google/flan-t5-large",           # Good general-purpose model
    "tiiuae/falcon-7b-instruct",   # Powerful instruction-following model
    "mistralai/Mistral-7B-Instruct-v0.2",  # Strong open model
    "facebook/opt-1.3b"            # Smaller but faster model
]
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", HF_MODELS[0])

# Ollama settings (for local inference)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Application Settings
MAX_TOKENS = 8192
TEMPERATURE = 0.7
OUTPUT_DIR = "data/outputs"

# Chain of Thought Settings
# Chain of Thought Settings
USE_CHAIN_OF_THOUGHT = os.getenv("USE_CHAIN_OF_THOUGHT", "True").lower() in ("true", "1", "t")
COT_STEPS = [
    "Identify the key technical concepts that a non-technical PM might not understand",
    "Determine the business implications of the technical decisions being discussed",
    "Analyze how the technical solutions relate to project timelines, resources, and deliverables",
    "Consider what technical context would help the PM communicate effectively with stakeholders",
    "Formulate questions that would help an LLM provide clear, PM-relevant explanations of the technical content"
]

# PM Context Settings
PM_CONTEXT = """
The probing questions will be used to prompt an LLM that will help a non-technical Product Manager understand 
the engineering discussion. The LLM will use these questions along with the meeting transcript to:
1. Explain technical concepts in business-friendly terms
2. Highlight implications for project planning, resources, and timelines
3. Identify potential risks or dependencies that the PM should be aware of
4. Provide context that helps the PM communicate effectively with stakeholders
"""

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)