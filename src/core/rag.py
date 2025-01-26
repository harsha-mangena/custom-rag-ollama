# src/core/rag.py

import logging
import asyncio
import aiohttp
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import pandas as pd
from pathlib import Path

from config.settings import Settings
from core.document import Document
from core.embeddings import EmbeddingManager
from data.database import DatabaseManager
from utils.file_processing import FileProcessor
from utils.metrics import MetricsCollector
from utils.ollama_utils import generate_response
from utils.search_utils import WebSearchManager
from utils.document_mapper import DocumentMapper

class EnhancedRAG:
    """Enhanced RAG system with web search and document mapping capabilities."""
    
    def __init__(
        self,
        *,  # Force keyword arguments
        hf_token: Optional[str] = None,
        embedding_model: str = Settings.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = Settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.CHUNK_OVERLAP,
        db_path: str = Settings.DB_PATH
    ):
        """
        Initialize RAG system.
        
        Args:
            hf_token: Optional Hugging Face API token
            embedding_model: Name of embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            db_path: Path to database file
        """
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self._initialized = False
        
        # Initialize core components
        self.db = DatabaseManager(db_path)
        self.embedding_manager = EmbeddingManager(hf_token)
        self.file_processor = FileProcessor(chunk_size, chunk_overlap)
        self.metrics = MetricsCollector(db_path)
        
        # Initialize enhanced components
        self.search_manager = WebSearchManager()
        self.document_mapper = DocumentMapper()

    async def initialize(self):
        """Initialize system components."""
        if not self._initialized:
            try:
                await self.db.initialize()
                self._initialized = True
                self.logger.info("RAG system initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG system: {e}")
                raise

    async def process_document(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> bool:
        """
        Process a document for RAG.
        
        Args:
            content: Raw document content
            filename: Name of the file
            file_type: Type of file (pdf, txt, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if processing was successful
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Extract text
            text_content = await self.file_processor.extract_content(content, file_type)
            if not text_content:
                raise ValueError(f"No content extracted from {filename}")

            # Create chunks
            chunks = self.file_processor.create_chunks(text_content)
            if not chunks:
                raise ValueError(f"No chunks created from {filename}")

            # Process chunks
            total_chunks = len(chunks)
            successful_chunks = 0

            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding, embed_time = await self.embedding_manager.get_embedding(
                        chunk,
                        self.embedding_model
                    )
                    
                    # Create document
                    doc = Document(
                        content=chunk,
                        source=filename,
                        file_type=file_type,
                        chunk_id=i,
                        embedding=embedding,
                        metadata={
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk),
                            "embedding_time": embed_time,
                            "embedding_model": self.embedding_model,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    
                    # Store in database
                    if await self.db.store_document(doc):
                        successful_chunks += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback((i + 1) / total_chunks)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {i} of {filename}: {e}")
                    continue

            # Update mind map
            await self.update_document_mind_map()
            
            self.logger.info(
                f"Processed {successful_chunks}/{total_chunks} chunks "
                f"from {filename}"
            )
            
            return successful_chunks > 0

        except Exception as e:
            self.logger.error(f"Document processing failed for {filename}: {e}")
            return False

    async def optimized_query(
        self,
        query_text: str,
        n_results: int = 5,
        ollama_model: Optional[str] = None,
        has_documents: bool = False
    ) -> Tuple[str, Dict[str, float]]:
        """
        Optimized query processing with support for both documents and web search.
        """
        try:
            start_time = datetime.now()
            metrics = {}
            
            # Get local results first if documents exist
            local_results = []
            if has_documents:
                try:
                    embed_start = datetime.now()
                    embedding, _ = await self.embedding_manager.get_embedding(
                        query_text,
                        self.embedding_model
                    )
                    metrics["embedding_time"] = (datetime.now() - embed_start).total_seconds()
                    
                    search_start = datetime.now()
                    results = await self.db.search_similar_documents(
                        query_embedding=embedding,
                        embedding_model=self.embedding_model,
                        n_results=n_results
                    )
                    metrics["search_time"] = (datetime.now() - search_start).total_seconds()
                    
                    local_results = [
                        {
                            "content": doc.content,
                            "source": doc.source,
                            "similarity": score,
                            "metadata": doc.metadata
                        }
                        for doc, score in results
                    ]
                except Exception as e:
                    self.logger.error(f"Error getting local results: {e}")

            # Get web results with local context if available
            web_start = datetime.now()
            web_results = await self.search_manager.search_with_context(
                query=query_text,
                documents=local_results if local_results else None,
                max_results=n_results
            )
            metrics["web_time"] = (datetime.now() - web_start).total_seconds()

            # Prepare reference context
            refs = []
            context_parts = []

            # Add local document references
            if local_results:
                refs.extend([
                    f"[L{i+1}] {r['source']}" 
                    for i, r in enumerate(local_results)
                ])
                context_parts.extend([
                    f"[L{i+1}] (Similarity: {r['similarity']:.2f}): {r['content'][:200]}..."
                    for i, r in enumerate(local_results)
                ])

            # Add web references
            if web_results:
                refs.extend([
                    f"[W{i+1}] {r['title']} ({r['url']})" 
                    for i, r in enumerate(web_results)
                ])
                context_parts.extend([
                    f"[W{i+1}]: {r['snippet']}"
                    for i, r in enumerate(web_results)
                ])

            if not context_parts:
                return "No relevant information found from either documents or web search.", metrics

            # Create prompt with both local and web context
            prompt = f"""Question: {query_text}

    Context:
    {chr(10).join(context_parts)}

    Please provide a clear and concise answer based on the available information. 
    When citing sources, use:
    - [L#] for information from uploaded documents
    - [W#] for information from web sources

    If information comes from multiple sources, cite all relevant references."""

            # Generate response
            if ollama_model:
                llm_start = datetime.now()
                response = await generate_response(
                    model=ollama_model,
                    prompt=prompt,
                    system_prompt=self._get_optimized_system_prompt()
                )
                metrics["llm_time"] = (datetime.now() - llm_start).total_seconds()
            else:
                response = "No LLM model specified for response generation."

            # Add references footer
            if refs:
                response += "\n\nReferences:\n" + "\n".join(refs)

            metrics["total_time"] = (datetime.now() - start_time).total_seconds()
            return response, metrics

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    def _get_optimized_system_prompt(self) -> str:
        """Get optimized system prompt for response generation."""
        return """You are a helpful assistant that provides accurate answers based on the given context.

    Your task is to:
    1. Analyze both uploaded documents and web search results
    2. Synthesize information from all available sources
    3. Provide clear, concise answers with proper source citations
    4. Use [L#] for local documents and [W#] for web sources
    5. Compare and contrast information when sources differ
    6. Clearly state if information is incomplete or uncertain

    Focus on accuracy and maintain a professional tone."""

    async def update_document_mind_map(self):
        """Update document mind map with current documents."""
        try:
            # Get all documents
            all_docs = await self.db.get_all_documents()
            
            # Create mind map
            graph = self.document_mapper.create_mind_map(
                all_docs,
                similarity_threshold=Settings.SIMILARITY_THRESHOLD
            )
            
            # Generate visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Settings.MINDMAP_OUTPUT_DIR / f"mind_map_{timestamp}.png"
            
            self.document_mapper.visualize_mind_map(str(output_path))
            
            self.logger.info(f"Mind map updated and saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to update mind map: {e}")
            raise

    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        try:
            if not self._initialized:
                await self.initialize()
                
            stats = {
                "total_documents": await self.db.get_document_count(),
                "file_types": await self.db.get_file_type_distribution(),
                "avg_chunk_size": await self.db.get_average_chunk_size(),
                "total_chunks": await self.db.get_total_chunks(),
                "embedding_models": await self.db.get_embedding_model_distribution(),
                "last_update": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get document statistics: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.db.close()
            self._initialized = False
            self.logger.info("RAG system closed successfully")
        except Exception as e:
            self.logger.error(f"Failed to close RAG system: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()