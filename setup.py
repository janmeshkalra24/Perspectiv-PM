#!/usr/bin/env python3
"""
Setup script for the perspectiv-knowledge-base package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base requirements
requirements = [
    "pydantic>=2.0.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-core>=0.1.10",
    "sentence-transformers>=2.2.2",
    "faiss-cpu>=1.7.4",
    "chromadb>=0.4.18",
    "pypdf>=3.17.1",
    "python-pptx>=0.6.21",
    "python-docx>=0.8.11",
    "numpy>=1.24.0",
    "tqdm>=4.66.1",
    "lancedb>=0.3.3",
    "diskcache>=5.6.3",
    "click>=8.0.0",
    "rich>=13.0.0"
]

setup(
    name="perspectiv-knowledge-base",
    version="0.1.0",
    author="Perspectiv",
    author_email="author@example.com",
    description="A lightweight hierarchical knowledge base for RAG applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/perspectiv-knowledge-base",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'perspectiv=perspectiv.cli:main',
        ],
    },
) 