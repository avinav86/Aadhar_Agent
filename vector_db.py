"""
Vector Database Module for Aadhaar Chat Agent

This module implements a high-performance vector database using ChromaDB and 
BGE (BAAI General Embedding) models for semantic search capabilities. It provides
functionality for storing, indexing, and retrieving document chunks based on
semantic similarity.

Key Features:
- BGE-large-en-v1.5 embeddings (1024 dimensions)
- ChromaDB persistent storage
- Intelligent text chunking with overlap
- Semantic similarity search
- Fallback mechanisms for robustness

Technical Details:
- Embedding Model: BAAI/bge-large-en-v1.5
- Vector Dimensions: 1024
- Similarity Metric: Cosine similarity with normalized embeddings
- Storage: Persistent ChromaDB collection

Author: Avinav Mishra
Repository: https://github.com/avinav86/Aadhar_Agent
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid
import hashlib
import re
import numpy as np

class VectorDatabase:
    """
    Handles vector database operations using ChromaDB with BGE embeddings.
    
    This class provides a complete vector database solution for the Aadhaar Chat Agent,
    including document storage, embedding generation, and semantic search capabilities.
    It uses state-of-the-art BGE embeddings for superior semantic understanding.
    
    Key Components:
    - BGE embedding model for high-quality vector representations
    - ChromaDB for persistent vector storage
    - Intelligent text chunking for optimal retrieval
    - Normalized embeddings for better similarity matching
    
    Attributes:
        persist_directory (str): Directory for ChromaDB persistence
        embedding_model (SentenceTransformer): BGE model for embeddings
        client (chromadb.PersistentClient): ChromaDB client instance
        collection (chromadb.Collection): Document collection for storage
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database with BGE embeddings and ChromaDB storage.
        
        This constructor sets up the complete vector database infrastructure:
        1. Loads the BGE embedding model (with fallback options)
        2. Initializes ChromaDB client with persistent storage
        3. Creates or connects to the document collection
        
        Args:
            persist_directory (str): Directory path for ChromaDB persistence
        """
        self.persist_directory = persist_directory
        
        print("🔄 Loading BGE embedding model...")
        
        # Initialize BGE embedding model with fallback hierarchy
        try:
            # Try BGE models in order of preference (higher dimensions first)
            # BGE models are specifically designed for retrieval tasks
            bge_models = [
                'BAAI/bge-large-en-v1.5',      # 1024 dimensions - best quality
                'BAAI/bge-base-en-v1.5',       # 768 dimensions - good balance
                'BAAI/bge-small-en-v1.5',      # 384 dimensions - faster
                'sentence-transformers/all-mpnet-base-v2'  # 768 dimensions fallback
            ]
            
            self.embedding_model = None
            # Try each model until one loads successfully
            for model_name in bge_models:
                try:
                    print(f"🔄 Loading BGE model: {model_name}")
                    self.embedding_model = SentenceTransformer(model_name)
                    dimensions = self.embedding_model.get_sentence_embedding_dimension()
                    print(f"✅ Loaded BGE model: {model_name} ({dimensions} dimensions)")
                    break
                except Exception as model_error:
                    print(f"⚠️  Failed to load {model_name}: {model_error}")
                    continue
            
            # Ensure at least one model loaded successfully
            if not self.embedding_model:
                raise Exception("All BGE models failed to load")
                
        except Exception as e:
            print(f"❌ Error loading BGE models: {e}")
            print("🔄 Falling back to ChromaDB default embeddings...")
            self.embedding_model = None
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        if self.embedding_model:
            dimensions = self.embedding_model.get_sentence_embedding_dimension()
            self.collection = self.client.get_or_create_collection(
                name="aadhaar_documents",
                metadata={"description": f"Aadhaar documents with BGE {dimensions}D embeddings"}
            )
            print(f"✅ Vector database initialized with BGE {dimensions}-dimensional embeddings!")
        else:
            self.collection = self.client.get_or_create_collection(
                name="aadhaar_documents",
                metadata={"description": "Aadhaar documents with default embeddings"}
            )
            print("✅ Vector database initialized with default embeddings!")
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector database with BGE embeddings"""
        if self.embedding_model:
            dimensions = self.embedding_model.get_sentence_embedding_dimension()
            print(f"📄 Adding documents to vector database with BGE {dimensions}D embeddings...")
        else:
            print("📄 Adding documents to vector database with default embeddings...")
        
        all_texts = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        for doc in documents:
            # Chunk the document content
            chunks = self.chunk_text(doc["content"])
            
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append({
                    "filename": doc["filename"],
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(f"{doc['filename']}_chunk_{i}")
                
                # Generate BGE embeddings if model is available
                if self.embedding_model:
                    # Use BGE model for encoding
                    embedding = self.embedding_model.encode(chunk, normalize_embeddings=True)
                    all_embeddings.append(embedding.tolist())
        
        # Add to collection with BGE embeddings if available
        if self.embedding_model and all_embeddings:
            self.collection.add(
                documents=all_texts,
                metadatas=all_metadatas,
                ids=all_ids,
                embeddings=all_embeddings
            )
            dimensions = len(all_embeddings[0])
            print(f"✅ Added {len(all_texts)} document chunks with BGE {dimensions}D embeddings")
        else:
            # Fallback to default embeddings
            self.collection.add(
                documents=all_texts,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"✅ Added {len(all_texts)} document chunks with default embeddings")
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into intelligent overlapping chunks for better embeddings"""
        # Clean and normalize text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        words = text.split()
        chunks = []
        
        # Use smaller chunks for better semantic understanding
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            
            # Ensure chunk is not empty and has meaningful content
            if chunk.strip() and len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        
        return chunks
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents using BGE embeddings"""
        if self.embedding_model:
            # Use BGE model to encode the query
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
        else:
            # Fallback to text-based search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name
        }
