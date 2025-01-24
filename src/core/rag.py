import logging
import asyncio
import aiohttp
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import pandas as pd

from config.settings import Settings
from core.document import Document
from core.embeddings import EmbeddingManager
from data.database import DatabaseManager
from utils.file_processing import FileProcessor
from utils.metrics import MetricsCollector
from utils.ollama_utils import generate_response

class EnhancedRAG:
    """Enhanced RAG (Retrieval Augmented Generation) system implementation."""
    
    def __init__(
        self,
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
        
        # Initialize components
        self.db = DatabaseManager(db_path)
        self.embedding_manager = EmbeddingManager(hf_token)
        self.file_processor = FileProcessor(chunk_size, chunk_overlap)
        self.metrics = MetricsCollector(db_path)

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

            # Log results
            self.logger.info(
                f"Processed {successful_chunks}/{total_chunks} chunks "
                f"from {filename}"
            )
            
            return successful_chunks > 0

        except Exception as e:
            self.logger.error(f"Document processing failed for {filename}: {e}")
            return False

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        similarity_threshold: float = 0.5,
        ollama_model: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[str]]:
        """
        Query the RAG system.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            ollama_model: Optional Ollama model for response generation
            
        Returns:
            Tuple containing:
            - List of results
            - Dictionary of metrics
            - Generated response (if model specified)
        """
        try:
            if not self._initialized:
                await self.initialize()

            metrics = {}
            start_time = datetime.now()

            # Generate query embedding
            query_embedding, embed_time = await self.embedding_manager.get_embedding(
                query_text,
                self.embedding_model
            )
            metrics["embedding_time"] = embed_time

            # Search documents
            search_start = datetime.now()
            results = await self.db.search_similar_documents(
                query_embedding,
                self.embedding_model,
                n_results,
                similarity_threshold
            )
            metrics["search_time"] = (datetime.now() - search_start).total_seconds()

            # Format results
            formatted_results = []
            for doc, similarity in results:
                excerpt = self._extract_relevant_excerpt(doc.content, query_text)
                formatted_results.append({
                    "content": doc.content,
                    "source": doc.source,
                    "similarity": similarity,
                    "metadata": doc.metadata,
                    "highlight": excerpt
                })

            # Generate response if model specified
            response = None
            if ollama_model and formatted_results:
                llm_start = datetime.now()
                
                # Prepare context
                context = self._format_context(formatted_results)
                
                # Generate response
                response = await generate_response(
                    model=ollama_model,
                    prompt=self._create_prompt(query_text, context),
                    system_prompt=self._get_system_prompt()
                )
                
                if response:
                    metrics["llm_time"] = (datetime.now() - llm_start).total_seconds()

            # Calculate total time
            metrics["total_time"] = (datetime.now() - start_time).total_seconds()

            # Store metrics
            await self.metrics.store_query_metrics(
                query_text=query_text,
                embedding_model=self.embedding_model,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
                metrics=metrics,
                result_count=len(formatted_results)
            )

            return formatted_results, metrics, response

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    def _extract_relevant_excerpt(
        self,
        text: str,
        query: str,
        context_words: int = 50
    ) -> str:
        """Extract relevant excerpt from text."""
        words = text.split()
        query_words = set(query.lower().split())
        
        if len(words) <= context_words:
            return text
            
        best_score = 0
        best_start = 0
        
        for i in range(len(words) - context_words):
            excerpt = words[i:i + context_words]
            score = sum(1 for word in excerpt if word.lower() in query_words)
            if score > best_score:
                best_score = score
                best_start = i
        
        excerpt_start = max(0, best_start - context_words//2)
        excerpt_end = min(len(words), best_start + context_words * 2)
        
        return " ".join(words[excerpt_start:excerpt_end])

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format results into context string."""
        return "\n\n".join(
            f"[Source: {r['source']}]\n{r['content']}"
            for r in results
        )

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM."""
        return f"""Based on the following context, please answer the question. 
        If the context doesn't contain enough information, please say so.

Context:
{context}

Question: {query}

Answer:"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a helpful AI assistant that provides accurate answers 
        based on the given context. If the context doesn't contain enough 
        information to answer the question, clearly state that. Always maintain
        a professional and informative tone."""

    async def run_benchmark(
        self,
        queries: List[str],
        llm_models: List[str],
        n_results: int = 5,
        similarity_threshold: float = 0.5,
        concurrent_requests: int = 3
    ) -> pd.DataFrame:
        """
        Run benchmarking tests.
        
        Args:
            queries: List of test queries
            llm_models: List of LLM models to test
            n_results: Number of results per query
            similarity_threshold: Minimum similarity score
            concurrent_requests: Number of concurrent requests
            
        Returns:
            DataFrame with benchmark results
        """
        try:
            if not self._initialized:
                await self.initialize()

            results = []
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def process_query(model: str, query: str):
                async with semaphore:
                    start_time = datetime.now()
                    
                    # Run query
                    query_results, metrics, response = await self.query(
                        query_text=query,
                        n_results=n_results,
                        similarity_threshold=similarity_threshold,
                        ollama_model=model
                    )
                    
                    if query_results:
                        # Calculate metrics
                        result = {
                            "model_name": model,
                            "query": query,
                            "query_time": metrics["search_time"],
                            "embedding_time": metrics["embedding_time"],
                            "response_time": metrics.get("llm_time", 0),
                            "total_time": (datetime.now() - start_time).total_seconds(),
                            "retrieval_count": len(query_results),
                            "avg_similarity": np.mean([r["similarity"] for r in query_results])
                        }
                        
                        results.append(result)
                        
                        # Store result
                        await self.metrics.store_benchmark_result(
                            model_name=model,
                            metrics=result
                        )

            # Create tasks
            tasks = [
                process_query(model, query)
                for model in llm_models
                for query in queries
            ]
            
            # Run concurrently
            await asyncio.gather(*tasks)
            
            return pd.DataFrame(results)

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
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