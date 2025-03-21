# Setting Up Ollama for Local Inference

This guide explains how to set up Ollama to run LLMs locally for the transcript analyzer.

## What is Ollama?

Ollama is a tool that lets you run open-source large language models (like Llama 2, Mistral, etc.) locally on your machine. This gives you:

- Privacy: Your data never leaves your computer
- No API costs: Run models for free
- No internet dependency: Works offline

## Installation

### macOS

1. Download Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Install the application

### Linux
```
bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows

Windows support is available through WSL2 (Windows Subsystem for Linux).

## Running Ollama

After installation, Ollama will start automatically and run in the background.

## Downloading Models

To download a model, run:

```
bash
ollama pull llama2
```

Available models include:
- llama2 (default)
- mistral
- mixtral
- phi
- gemma
- llava (multimodal)

## Using with Transcript Analyzer

1. Make sure Ollama is running
2. Set up your .env file:

```
API_PROVIDER=ollama
OLLAMA_MODEL=llama2
```

3. Run the transcript analyzer as usual:

```
bash
python -m src.generate data/sample_transcripts/engineering_meeting_sample.txt
```

