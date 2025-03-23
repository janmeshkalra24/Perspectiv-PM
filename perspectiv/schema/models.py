"""
Data models for the Perspectiv knowledge base.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for storing document metadata and content."""
    
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    domain: str = Field(..., description="Knowledge domain this document belongs to")
    file_path: str = Field(..., description="Path to the source file")
    file_name: str = Field(..., description="Filename of the source file")
    file_type: str = Field(..., description="File type (extension)")
    chunk_id: Optional[int] = Field(None, description="Chunk ID if this is part of a larger document")
    created_at: datetime = Field(default_factory=datetime.now, description="When this document was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When this document was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Query(BaseModel):
    """Model for storing query information."""
    
    query: str = Field(..., description="Query text")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this query was made")
    results: Optional[Dict[str, Any]] = Field(None, description="Query results if available")


class QueryResult(BaseModel):
    """Model for storing query result information."""
    
    content: str = Field(..., description="Result content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class Transcript(BaseModel):
    """Model for storing transcript information."""
    
    original_path: str = Field(..., description="Original path to the transcript file")
    stored_path: str = Field(..., description="Path where the transcript is stored in the knowledge base")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this transcript was added")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Session(BaseModel):
    """Model for storing session information."""
    
    id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="When this session was created")
    last_updated: datetime = Field(default_factory=datetime.now, description="When this session was last updated")
    queries: List[Query] = Field(default_factory=list, description="Queries made in this session")
    transcripts: List[Transcript] = Field(default_factory=list, description="Transcripts added in this session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DomainMetadata(BaseModel):
    """Model for storing domain metadata."""
    
    document_count: int = Field(0, description="Number of documents in this domain")
    last_updated: datetime = Field(default_factory=datetime.now, description="When this domain was last updated")


class KnowledgeBaseStats(BaseModel):
    """Model for storing knowledge base statistics."""
    
    document_count: int = Field(0, description="Total number of documents in the knowledge base")
    total_tokens: int = Field(0, description="Total number of tokens in the knowledge base")
    sessions: int = Field(0, description="Total number of sessions")


class KnowledgeBaseMetadata(BaseModel):
    """Model for storing knowledge base metadata."""
    
    created_at: datetime = Field(default_factory=datetime.now, description="When this knowledge base was created")
    last_updated: datetime = Field(default_factory=datetime.now, description="When this knowledge base was last updated")
    version: str = Field("0.1.0", description="Knowledge base version")
    stats: KnowledgeBaseStats = Field(default_factory=KnowledgeBaseStats, description="Knowledge base statistics")
    domains: Dict[str, DomainMetadata] = Field(default_factory=dict, description="Domain metadata") 