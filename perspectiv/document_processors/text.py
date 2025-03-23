"""
Text document processor for the Perspectiv knowledge base.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseDocumentProcessor

class TextProcessor(BaseDocumentProcessor):
    """
    Processor for text files (txt, md, etc.).
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @staticmethod
    def can_process(file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the document file

        Returns:
            True if this processor can handle the file, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.txt', '.md', '.text']
    
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a text file and convert it to a list of Document objects.
        
        Args:
            file_path: Path to the text file
            metadata: Optional metadata to include in the documents

        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Prepare metadata
        meta = self._prepare_metadata(file_path, metadata)
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split the text into chunks
        texts = self.text_splitter.split_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(texts):
            doc_meta = meta.copy()
            doc_meta["chunk"] = i
            documents.append(Document(page_content=chunk, metadata=doc_meta))
        
        return documents


class TranscriptProcessor(TextProcessor):
    """
    Specialized processor for transcript files.
    
    This is a specialized version of the TextProcessor that adds additional
    processing specific to call transcripts.
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Initialize the transcript processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents (larger for context)
            chunk_overlap: Overlap between text chunks (larger for context)
        """
        super().__init__(chunk_size, chunk_overlap)
    
    @staticmethod
    def can_process(file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the document file

        Returns:
            True if this processor can handle the file, False otherwise
        """
        # Check if it's a text file
        if not TextProcessor.can_process(file_path):
            return False
        
        # Check if it has transcript in the name
        filename = os.path.basename(file_path).lower()
        return 'transcript' in filename or 'meeting' in filename
    
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a transcript file with specialized handling.
        
        Args:
            file_path: Path to the transcript file
            metadata: Optional metadata to include in the documents

        Returns:
            List of Document objects
        """
        # Use the base text processor to process the file
        documents = super().process(file_path, metadata)
        
        # Add transcript-specific metadata
        for doc in documents:
            doc.metadata["content_type"] = "transcript"
            
            # Try to extract speaker information if available
            speakers = self._extract_speakers(doc.page_content)
            if speakers:
                doc.metadata["speakers"] = speakers
        
        return documents
    
    @staticmethod
    def _extract_speakers(text: str) -> List[str]:
        """
        Extract speakers from transcript text.
        
        Args:
            text: The transcript text

        Returns:
            List of speaker names found in the text
        """
        import re
        
        # Look for common transcript patterns like "Name: text" or "[Name]: text"
        speaker_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+):', # Name: text
            r'\[([A-Z][a-z]+ [A-Z][a-z]+)\]', # [Name] text
            r'([A-Z][a-z]+):', # FirstName: text
        ]
        
        speakers = set()
        for pattern in speaker_patterns:
            matches = re.findall(pattern, text)
            speakers.update(matches)
        
        return list(speakers) 