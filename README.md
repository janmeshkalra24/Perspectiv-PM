# Perspectiv Knowledge Base

A lightweight, hierarchical knowledge base for RAG (Retrieval-Augmented Generation) applications, designed to support low-latency streaming retrieval during video conferencing sessions.

## Overview

This knowledge base system is designed as a modular component that can be integrated with LLM applications to provide context-aware information retrieval during engineering standup meetings. It features:

- **Hierarchical Streaming RAG**: Navigate from top-level directories to specific sub-tables based on query relevance
- **Local-First Architecture**: Run entirely on disk without requiring external APIs
- **Session Persistence**: Cache and retain historical information across different meeting sessions
- **Low-Latency Design**: Optimized for real-time retrieval during live meetings
- **Automated Content Population**: Scripts to automatically gather content from various sources

## Knowledge Base Structure

The knowledge base is organized into the following categories:

1. **Tech Decoding**: Technical documentation, diagrams, and guides to help simplify jargon
2. **Product Manager FAQs**: Corpus of questions that experienced PMs might ask
3. **ETAs/Blockers/Dependencies**: Task decomposition and resource requirements
4. **Cache History**: Historical meeting transcripts and context

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/perspectiv-knowledge-base.git
cd perspectiv-knowledge-base

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Populating Your Knowledge Base

The knowledge base comes with powerful scraping scripts to automatically gather content from various sources. Here's how to use them:

1. First, copy and configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your API tokens and credentials
```

2. Configure your content sources:
```bash
cp config.example.json config.json
# Edit config.json to specify your repositories, blogs, and other sources
```

3. Run the population script:
```bash
python scripts/populate_kb.py --kb-dir data --config config.json
```

This will:
- Scrape technical documentation from GitHub, Stack Overflow, and tech blogs
- Gather PM FAQs from product management resources and Reddit
- Track tasks and dependencies from JIRA, GitHub issues, and Slack

The script supports the following sources:

#### Technical Documentation
- GitHub repositories (READMEs, wikis)
- Stack Overflow top answers
- Engineering blogs
- Technical documentation sites

#### Product Management FAQs
- Product management blogs
- Reddit r/ProductManagement
- PM resource websites
- FAQ collections

#### Tasks and Dependencies
- JIRA tickets
- GitHub issues
- Slack threads
- Meeting transcripts

### Managing Your Knowledge Base

The knowledge base comes with a powerful CLI tool for managing and visualizing your data. Here are the main commands:

#### View Knowledge Base Status

```bash
perspectiv status
```

This shows overall statistics and the status of each domain.

#### Adding Documents to Domains

1. First, ensure your documents are organized by domain:
```
data/
├── tech_decoding/           # Technical docs, diagrams, etc.
│   ├── architecture.pdf
│   └── api_specs.md
├── product_manager_faqs/    # PM-related documents
│   └── common_questions.md
├── tasks_blockers_deps/     # Task management docs
│   └── project_timeline.md
└── cache_history/          # Meeting transcripts
    └── standup_2023_06_01.txt
```

2. Add documents using the CLI:
```bash
# Add a technical document
perspectiv add tech_decoding path/to/document.pdf

# Add PM FAQs
perspectiv add product_manager_faqs path/to/faqs.md

# Add project timeline
perspectiv add tasks_blockers_deps path/to/timeline.md

# Add meeting transcript
perspectiv add cache_history path/to/transcript.txt
```

#### Searching the Knowledge Base

```bash
# Search across all domains
perspectiv search "What are the current blockers for authentication?"

# Search in a specific domain
perspectiv search --domain tech_decoding "How does the OAuth flow work?"

# Control number of results
perspectiv search --top-k 5 "database migration plan"
```

#### Managing Sessions

```bash
# List all sessions
perspectiv sessions

# View details of a specific session
perspectiv session session-20230601-123456
```

### Programmatic Usage

You can also use the knowledge base programmatically:

```python
from perspectiv import KnowledgeBase

# Initialize the knowledge base
kb = KnowledgeBase("path/to/data")
kb.initialize()

# Add documents
kb.update_from_transcript("meeting_transcript.txt", session_id="meeting-2023-06-01")

# Query the knowledge base
results = kb.query("What are the common blockers for implementing authentication?")
```

## Directory Structure

```
perspectiv-knowledge-base/
├── data/                           # Knowledge base data storage
│   ├── tech_decoding/             # Technical documentation
│   ├── product_manager_faqs/      # PM-related queries
│   ├── tasks_blockers_deps/       # Task management information
│   └── cache_history/             # Historical session data
├── perspectiv/                    # Main package
│   ├── knowledge_base.py          # Core knowledge base implementation
│   ├── hierarchical_retriever.py  # Hierarchical document retrieval
│   ├── session_manager.py         # Session state management
│   ├── document_processors/       # Document processing utilities
│   ├── vectorstores/             # Vector storage implementations
│   └── schema/                   # Data schemas
├── scripts/                      # Utility scripts
│   ├── tech_scraper.py          # Technical documentation scraper
│   ├── pm_faq_scraper.py        # Product management FAQ scraper
│   ├── task_tracker.py          # Task and dependency tracker
│   └── populate_kb.py           # Knowledge base population script
├── examples/                     # Example usage scenarios
├── tests/                       # Unit and integration tests
├── requirements.txt             # Dependencies
├── config.example.json         # Example configuration
├── .env.example               # Example environment variables
└── README.md                   # Documentation
```

## Integration

This system is designed to be integrated with:

- Video conferencing platforms for transcript streaming
- LLM inference engines for RAG-enhanced responses
- PM assistance tools for real-time guidance

## Best Practices

1. **Document Organization**:
   - Keep documents in their appropriate domains
   - Use descriptive filenames
   - Include metadata when possible

2. **Session Management**:
   - Create a new session for each meeting
   - Use meaningful session IDs (e.g., "standup-20230601")
   - Review session history for context

3. **Query Optimization**:
   - Be specific in your queries
   - Use domain-specific search when appropriate
   - Consider the context of your query

4. **Content Population**:
   - Regularly update your knowledge base using the scraping scripts
   - Customize the configuration file to match your organization's needs
   - Monitor the scraping logs for any issues
   - Keep API tokens and credentials secure
   - Consider setting up scheduled runs for automatic updates