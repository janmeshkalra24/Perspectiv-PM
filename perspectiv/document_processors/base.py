"""
Base document processor for handling documents in the Perspectiv knowledge base.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class BaseDocumentProcessor(ABC):
    """
    Base class for document processors.
    
    All document processors should extend this class and implement the process method.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a document file and convert it to a list of Document objects.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata to include in the documents

        Returns:
            List of Document objects
        """
        pass
    
    @staticmethod
    def can_process(file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the document file

        Returns:
            True if this processor can handle the file, False otherwise
        """
        return False
    
    def _prepare_metadata(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare metadata for a document.
        
        Args:
            file_path: Path to the document file
            metadata: Optional additional metadata

        Returns:
            Metadata dictionary
        """
        meta = metadata or {}
        meta.update({
            "file_path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": os.path.splitext(file_path)[1][1:],
            "processor": self.__class__.__name__
        })
        return meta 