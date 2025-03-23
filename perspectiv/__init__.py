"""
Perspectiv Knowledge Base

A lightweight, hierarchical knowledge base for RAG (Retrieval-Augmented Generation) applications,
designed to support low-latency streaming retrieval during video conferencing sessions.
"""

__version__ = "0.1.0"

from .knowledge_base import KnowledgeBase
from .hierarchical_retriever import HierarchicalRetriever
from .session_manager import SessionManager
from .embeddings import PerspectivEmbeddings

__all__ = [
    'KnowledgeBase',
    'HierarchicalRetriever',
    'SessionManager',
    'PerspectivEmbeddings',
] 