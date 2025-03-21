# Standup Copilot

A lightweight framework for analyzing engineering meeting transcripts and generating probing questions to gain deeper context on technical discussions.

## Overview

This tool ingests meeting transcripts in text format, analyzes the content using LLMs, and generates a set of probing questions that could be used to gain more context about the technical topics discussed.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your LLM API keys in `config.py`

## Usage

```
python -m src.generate data/sample_transcripts/your_transcript.txt
```

## Sample Output

For a transcript discussing columnar datasources and ETL pipelines, the tool might generate questions like:

1. What is an ETL pipeline?
2. Why is deduplication needed for downstream tables?
3. What are the performance implications of this architecture?