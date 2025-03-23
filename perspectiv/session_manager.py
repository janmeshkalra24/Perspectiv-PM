"""
Session Manager module for the Perspectiv knowledge base.

This module handles session persistence, caching, and history management
across different meeting sessions.
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import diskcache

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manager for handling session persistence, caching, and history.
    """
    
    def __init__(
        self, 
        cache: diskcache.Cache,
        history_dir: str
    ):
        """
        Initialize the session manager.
        
        Args:
            cache: Disk cache instance for session persistence
            history_dir: Directory for storing historical session data
        """
        self.cache = cache
        self.history_dir = os.path.abspath(history_dir)
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Session data is stored in the cache with keys like 'session:{session_id}'
        # Session index is stored at 'session:index'
        if 'session:index' not in self.cache:
            self.cache['session:index'] = {}
        
        logger.info(f"Session manager initialized with history directory: {self.history_dir}")
    
    def _get_session_key(self, session_id: str) -> str:
        """Get cache key for a session."""
        return f"session:{session_id}"
    
    def _get_session_dir(self, session_id: str) -> str:
        """Get directory path for a session's history files."""
        return os.path.join(self.history_dir, session_id)
    
    def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new session or reset an existing one.
        
        Args:
            session_id: Unique identifier for the session
            metadata: Optional metadata about the session

        Returns:
            Session information
        """
        session_key = self._get_session_key(session_id)
        session_dir = self._get_session_dir(session_id)
        
        # Create session directory if it doesn't exist
        os.makedirs(session_dir, exist_ok=True)
        
        # Default session structure
        session = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "queries": [],
            "transcripts": [],
            "metadata": metadata or {}
        }
        
        # Store in cache
        self.cache[session_key] = session
        
        # Update session index
        session_index = self.cache['session:index']
        session_index[session_id] = {
            "id": session_id,
            "created_at": session["created_at"],
            "last_updated": session["last_updated"]
        }
        self.cache['session:index'] = session_index
        
        return session
    
    def get_session(self, session_id: str, create_if_missing: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: The session ID
            create_if_missing: Create the session if it doesn't exist

        Returns:
            Session data or None if not found and not creating
        """
        session_key = self._get_session_key(session_id)
        
        if session_key in self.cache:
            return self.cache[session_key]
        elif create_if_missing:
            return self.create_session(session_id)
        else:
            return None
    
    def add_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Add a query to the session history.
        
        Args:
            session_id: The session ID
            query: The query text

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        session_key = self._get_session_key(session_id)
        
        # Create query entry
        query_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": None  # Will be updated when results are added
        }
        
        # Add to session
        session["queries"].append(query_entry)
        session["last_updated"] = datetime.now().isoformat()
        
        # Update in cache
        self.cache[session_key] = session
        
        # Update session index
        session_index = self.cache['session:index']
        session_index[session_id]["last_updated"] = session["last_updated"]
        self.cache['session:index'] = session_index
        
        return session
    
    def add_result(self, session_id: str, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add query results to the session history.
        
        Args:
            session_id: The session ID
            query: The query text that produced these results
            results: The query results

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        session_key = self._get_session_key(session_id)
        
        # Find the query entry
        query_found = False
        for q in reversed(session["queries"]):
            if q["query"] == query and q["results"] is None:
                q["results"] = results
                query_found = True
                break
        
        # If query not found, add a new entry
        if not query_found:
            query_entry = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            session["queries"].append(query_entry)
        
        session["last_updated"] = datetime.now().isoformat()
        
        # Update in cache
        self.cache[session_key] = session
        
        # Update session index
        session_index = self.cache['session:index']
        session_index[session_id]["last_updated"] = session["last_updated"]
        self.cache['session:index'] = session_index
        
        return session
    
    def store_transcript(self, transcript_path: str, session_id: str) -> Dict[str, Any]:
        """
        Store a transcript file in the session history.
        
        Args:
            transcript_path: Path to the transcript file
            session_id: The session ID

        Returns:
            Dict with status and stored path
        """
        session = self.get_session(session_id)
        session_key = self._get_session_key(session_id)
        session_dir = self._get_session_dir(session_id)
        
        # Get the transcript filename
        filename = os.path.basename(transcript_path)
        base, ext = os.path.splitext(filename)
        
        # Add timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_filename = f"{base}_{timestamp}{ext}"
        
        # Copy transcript to session directory
        dest_path = os.path.join(session_dir, new_filename)
        shutil.copy2(transcript_path, dest_path)
        
        # Add to session
        transcript_entry = {
            "original_path": transcript_path,
            "stored_path": dest_path,
            "timestamp": datetime.now().isoformat()
        }
        
        session["transcripts"].append(transcript_entry)
        session["last_updated"] = datetime.now().isoformat()
        
        # Update in cache
        self.cache[session_key] = session
        
        # Update session index
        session_index = self.cache['session:index']
        session_index[session_id]["last_updated"] = session["last_updated"]
        self.cache['session:index'] = session_index
        
        return transcript_entry
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get the complete history for a session.
        
        Args:
            session_id: The session ID

        Returns:
            Session history data
        """
        session = self.get_session(session_id, create_if_missing=False)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return {"error": f"Session not found: {session_id}"}
        
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session summaries
        """
        session_index = self.cache.get('session:index', {})
        return list(session_index.values())
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its data.
        
        Args:
            session_id: The session ID to delete

        Returns:
            True if successful, False otherwise
        """
        session_key = self._get_session_key(session_id)
        session_dir = self._get_session_dir(session_id)
        
        # Delete from cache
        if session_key in self.cache:
            del self.cache[session_key]
        
        # Delete from index
        session_index = self.cache['session:index']
        if session_id in session_index:
            del session_index[session_id]
            self.cache['session:index'] = session_index
        
        # Delete session directory
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                logger.error(f"Error deleting session directory {session_dir}: {e}")
                return False
        
        return True 