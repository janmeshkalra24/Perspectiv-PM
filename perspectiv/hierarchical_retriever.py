"""
Hierarchical Retriever module for the Perspectiv knowledge base.

This module implements a hierarchical retrieval system that can traverse from top-level 
directories to specific documents based on query relevance.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pickle

import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)

from .embeddings import PerspectivEmbeddings

logger = logging.getLogger(__name__)

class HierarchicalRetriever:
    """
    A hierarchical retriever that can navigate from top-level directories to specific
    documents based on query relevance.
    """
    
    def __init__(
        self, 
        data_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_cache_dir: Optional[str] = None
    ):
        """
        Initialize the hierarchical retriever.
        
        Args:
            data_dir: Directory containing the knowledge base data
            embedding_model: Name of the sentence transformer model to use
            index_cache_dir: Directory to cache vector indexes, defaults to data_dir/indexes
        """
        self.data_dir = os.path.abspath(data_dir)
        self.index_cache_dir = index_cache_dir or os.path.join(self.data_dir, "indexes")
        os.makedirs(self.index_cache_dir, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embeddings = PerspectivEmbeddings(embedding_model)
        
        # Domain paths
        self.domains = self._get_domain_paths()
        
        # Store domain descriptions for routing
        self.domain_descriptions = {
            "tech_decoding": "Technical documentation, diagrams, guides, textbooks, and resources to help simplify technical jargon.",
            "product_manager_faqs": "Common questions and information that product managers might need during engineering standups.",
            "tasks_blockers_deps": "Task decomposition, resource requirements, dependencies, blockers, and ETAs for engineering work.",
            "cache_history": "Historical meeting transcripts, shared files, and context from previous sessions."
        }
        
        # Document count cache
        self._document_count = 0
        
        # Domain indexes
        self.domain_indexes = {}
        
        # Domain metadata
        self.domain_metadata = {}
        
        logger.info(f"Hierarchical retriever initialized with model: {embedding_model}")
    
    def _get_domain_paths(self) -> Dict[str, str]:
        """Get paths for all domain directories."""
        domains = {}
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and not item == "indexes":
                domains[item] = item_path
        return domains
    
    def _get_file_loader(self, file_path: str) -> Optional[Any]:
        """Get the appropriate loader for a file based on its extension."""
        # Skip system files and hidden files
        filename = os.path.basename(file_path)
        if filename.startswith('.') or filename.endswith('.db') or \
           filename.endswith('.db-shm') or filename.endswith('.db-wal'):
            return None
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.txt' or ext == '.md':
            return TextLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        elif ext == '.pptx' or ext == '.ppt':
            return UnstructuredPowerPointLoader(file_path)
        else:
            logger.debug(f"Unsupported document type: {ext} for file {file_path}")
            return None
    
    def _load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load all documents from a directory."""
        documents = []
        
        for root, _, files in os.walk(directory):
            # Skip system directories
            if os.path.basename(root).startswith('.'):
                continue
                
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                loader = self._get_file_loader(file_path)
                
                if loader:
                    try:
                        docs = loader.load()
                        # Add domain metadata
                        domain = os.path.basename(directory)
                        for doc in docs:
                            doc.metadata["domain"] = domain
                            doc.metadata["file_path"] = file_path
                            doc.metadata["file_name"] = file
                            
                        documents.extend(docs)
                        logger.debug(f"Loaded {len(docs)} documents from {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _build_domain_index(self, domain: str, force_rebuild: bool = False) -> FAISS:
        """Build or load a vector index for a domain."""
        index_path = os.path.join(self.index_cache_dir, f"{domain}.faiss")
        index_metadata_path = os.path.join(self.index_cache_dir, f"{domain}.json")
        
        # Check if index exists and we're not forcing a rebuild
        if os.path.exists(index_path) and os.path.exists(index_metadata_path) and not force_rebuild:
            try:
                # Load the index with safe deserialization
                index = FAISS.load_local(
                    folder_path=self.index_cache_dir,
                    index_name=domain,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True  # We trust our own index files
                )
                
                # Load metadata
                with open(index_metadata_path, 'r') as f:
                    self.domain_metadata[domain] = json.load(f)
                
                logger.info(f"Loaded existing index for domain: {domain}")
                return index
            except Exception as e:
                logger.error(f"Error loading index for domain {domain}: {e}")
                logger.info(f"Rebuilding index for domain: {domain}")
        
        # Build new index
        domain_path = self.domains[domain]
        documents = self._load_documents_from_directory(domain_path)
        
        if not documents:
            logger.warning(f"No documents found in domain: {domain}")
            return None
        
        # Create FAISS index
        index = FAISS.from_documents(documents, self.embeddings)
        
        # Save index
        index.save_local(self.index_cache_dir, domain)
        
        # Save metadata
        metadata = {
            "document_count": len(documents),
            "embedding_model": self.embedding_model_name,
            "files": list(set(doc.metadata["file_path"] for doc in documents))
        }
        
        with open(index_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.domain_metadata[domain] = metadata
        logger.info(f"Built new index for domain: {domain} with {len(documents)} documents")
        
        return index
    
    def build_indexes(self, force_rebuild: bool = False) -> None:
        """
        Build or load vector indexes for all domains.
        
        Args:
            force_rebuild: If True, force rebuilding all indexes
        """
        self._document_count = 0
        self.domain_indexes = {}
        
        for domain in self.domains:
            index = self._build_domain_index(domain, force_rebuild=force_rebuild)
            if index:
                self.domain_indexes[domain] = index
                self._document_count += self.domain_metadata[domain]["document_count"]
        
        logger.info(f"Built indexes for {len(self.domain_indexes)} domains with a total of {self._document_count} documents")
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the knowledge base."""
        return self._document_count
    
    def _rank_domains(self, query: str) -> List[Tuple[str, float]]:
        """Rank domains by relevance to the query."""
        # Use embeddings to calculate similarity between query and domain descriptions
        query_embedding = self.embeddings.embed_query(query)
        
        domain_scores = []
        for domain, description in self.domain_descriptions.items():
            # Skip domains that don't have an index
            if domain not in self.domain_indexes:
                continue
                
            description_embedding = self.embeddings.embed_query(description)
            similarity = np.dot(query_embedding, description_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(description_embedding)
            )
            domain_scores.append((domain, float(similarity)))
        
        # Sort by similarity score in descending order
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores
    
    def retrieve(
        self, 
        query: str, 
        domains: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The query text
            domains: Optional list of domains to search within, if None searches all
            top_k: Number of top results to return per domain

        Returns:
            Dict containing retrieved documents and metadata
        """
        # Determine which domains to search
        if domains:
            domains_to_search = [d for d in domains if d in self.domain_indexes]
        else:
            domains_to_search = list(self.domain_indexes.keys())
        
        if not domains_to_search:
            logger.warning("No valid domains found to search")
            return {"results": {}, "domain_scores": []}
        
        # Rank domains by relevance to query
        domain_scores = self._rank_domains(query)
        ranked_domains = [d for d, _ in domain_scores if d in domains_to_search]
        
        # Search in each domain
        results = {}
        for domain in ranked_domains:
            domain_index = self.domain_indexes[domain]
            domain_results = domain_index.similarity_search_with_score(query, k=top_k)
            
            # Format results
            formatted_results = []
            for doc, score in domain_results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            results[domain] = formatted_results
        
        return {
            "results": results,
            "domain_scores": domain_scores
        }
    
    def add_document(self, document_path: str, domain: str) -> bool:
        """
        Add a new document to the knowledge base.
        
        Args:
            document_path: Path to the document to add
            domain: Domain to add the document to

        Returns:
            True if successful, False otherwise
        """
        if domain not in self.domains:
            logger.error(f"Domain not found: {domain}")
            return False
        
        if not os.path.exists(document_path):
            logger.error(f"Document not found: {document_path}")
            return False
        
        loader = self._get_file_loader(document_path)
        if not loader:
            logger.error(f"Unsupported file type: {document_path}")
            return False
        
        try:
            # Load the document
            documents = loader.load()
            
            # Add domain metadata
            for doc in documents:
                doc.metadata["domain"] = domain
                doc.metadata["file_path"] = document_path
                doc.metadata["file_name"] = os.path.basename(document_path)
            
            # Add to the index
            if domain in self.domain_indexes:
                self.domain_indexes[domain].add_documents(documents)
                
                # Update index on disk
                self.domain_indexes[domain].save_local(self.index_cache_dir, domain)
                
                # Update metadata
                self.domain_metadata[domain]["document_count"] += len(documents)
                if document_path not in self.domain_metadata[domain]["files"]:
                    self.domain_metadata[domain]["files"].append(document_path)
                
                # Update metadata file
                metadata_path = os.path.join(self.index_cache_dir, f"{domain}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.domain_metadata[domain], f, indent=2)
                
                # Update document count
                self._document_count += len(documents)
                
                logger.info(f"Added {len(documents)} documents from {document_path} to domain {domain}")
                return True
            else:
                logger.error(f"Index not found for domain: {domain}")
                return False
        except Exception as e:
            logger.error(f"Error adding document {document_path}: {e}")
            return False 