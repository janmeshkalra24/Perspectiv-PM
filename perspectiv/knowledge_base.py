"""
Core Knowledge Base implementation for Perspectiv.

This module provides the main KnowledgeBase class that serves as the primary interface
for interacting with the hierarchical RAG system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import diskcache

from .hierarchical_retriever import HierarchicalRetriever
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    A hierarchical knowledge base for storing and retrieving information.
    
    This knowledge base supports:
    - Hierarchical document retrieval
    - Session persistence
    - Low-latency streaming RAG
    - Document updating from transcripts
    """
    
    def __init__(
        self, 
        data_dir: str,
        cache_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_size_limit: int = 2 * 1024 * 1024 * 1024  # 2GB default
    ):
        """
        Initialize the knowledge base.
        
        Args:
            data_dir: Directory where knowledge base data is stored
            cache_dir: Directory for system cache, defaults to data_dir/.system_cache if None
            embedding_model: Model name for document embeddings
            cache_size_limit: Maximum size of the cache in bytes
        """
        self.data_dir = os.path.abspath(data_dir)
        self.cache_dir = cache_dir or os.path.join(self.data_dir, ".system_cache")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Expected sub-directories for different knowledge domains
        self.expected_subdirs = [
            "tech_decoding",
            "product_manager_faqs",
            "tasks_blockers_deps",
            "cache_history"
        ]
        
        # Initialize subdirectories
        for subdir in self.expected_subdirs:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)
        
        # Initialize cache for session persistence
        self.cache = diskcache.Cache(self.cache_dir, size_limit=cache_size_limit)
        
        # Initialize the hierarchical retriever
        self.retriever = HierarchicalRetriever(
            data_dir=self.data_dir,
            embedding_model=embedding_model,
            index_cache_dir=os.path.join(self.cache_dir, "indexes")
        )
        
        # Initialize session manager
        self.session_manager = SessionManager(
            cache=self.cache,
            history_dir=os.path.join(self.data_dir, "cache_history")
        )
        
        # Metadata store
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        self.metadata = self._load_metadata()
        
        logger.info(f"Knowledge base initialized at {self.data_dir}")
        logger.info(f"System cache directory: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load knowledge base metadata from disk."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return self._create_default_metadata()
        else:
            return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata structure."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "0.1.0",
            "stats": {
                "document_count": 0,
                "total_tokens": 0,
                "sessions": 0
            },
            "domains": {}
        }
        
        # Initialize domain metadata
        for domain in self.expected_subdirs:
            metadata["domains"][domain] = {
                "document_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        # Save the default metadata
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save metadata to disk."""
        if metadata is None:
            metadata = self.metadata
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the knowledge base by scanning all documents and building indexes.
        
        Args:
            force_rebuild: If True, force rebuilding all indexes
        """
        # Scan all documents and build indexes
        logger.info("Initializing knowledge base...")
        
        # Initialize the retriever with all documents
        self.retriever.build_indexes(force_rebuild=force_rebuild)
        
        # Update metadata
        self.metadata["stats"]["document_count"] = self.retriever.get_document_count()
        self._save_metadata()
        
        logger.info(f"Knowledge base initialized with {self.metadata['stats']['document_count']} documents")
    
    def query(
        self, 
        query_text: str, 
        session_id: Optional[str] = None,
        top_k: int = 5,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query_text: The query text
            session_id: Optional session ID to associate the query with
            top_k: Number of top results to return per domain
            domains: Optional list of domains to restrict the search to

        Returns:
            Dict containing retrieved documents and metadata
        """
        # Record query in session if session_id is provided
        if session_id:
            self.session_manager.add_query(session_id, query_text)
        
        # Retrieve documents hierarchically
        results = self.retriever.retrieve(
            query=query_text,
            domains=domains,
            top_k=top_k
        )
        
        # Record results in session
        if session_id:
            self.session_manager.add_result(session_id, query_text, results)
        
        return results
    
    def update_from_transcript(
        self, 
        transcript_path: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the knowledge base with information from a transcript.
        
        Args:
            transcript_path: Path to the transcript file
            session_id: Optional session ID to associate with this transcript

        Returns:
            Dict containing update status and metadata
        """
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Process and store transcript in cache_history
        status = self.session_manager.store_transcript(
            transcript_path=transcript_path,
            session_id=session_id
        )
        
        # Update the retriever with the new transcript
        self.retriever.add_document(
            document_path=status["stored_path"],
            domain="cache_history"
        )
        
        # Update metadata
        self.metadata["stats"]["sessions"] += 1
        self.metadata["stats"]["document_count"] += 1
        self.metadata["domains"]["cache_history"]["document_count"] += 1
        self.metadata["domains"]["cache_history"]["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        return status
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve the history for a specific session.
        
        Args:
            session_id: The session ID to retrieve history for

        Returns:
            Dict containing session history
        """
        return self.session_manager.get_session_history(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in the knowledge base.

        Returns:
            List of session metadata
        """
        return self.session_manager.list_sessions()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dict containing knowledge base statistics
        """
        return self.metadata["stats"]
    
    def close(self) -> None:
        """
        Close the knowledge base and release resources.
        """
        self.cache.close()
        logger.info("Knowledge base closed") 