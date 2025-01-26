# src/data/database.py

import sqlite3
import aiosqlite
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime
import logging
from core.document import Document
from config.settings import Settings

class DatabaseManager:
    """Manages database operations for the RAG system."""
    
    def __init__(self, db_path: str = Settings.DB_PATH):
        """Initialize database connection and tables."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self):
        """Initialize the database asynchronously."""
        if not self._initialized:
            await self._setup_database()
            self._initialized = True

    async def _setup_database(self):
        """Set up database tables."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create documents table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding BLOB,
                        source TEXT,
                        file_type TEXT,
                        metadata TEXT,
                        embedding_model TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data TEXT
                    )
                """)
                
                await db.commit()
                self.logger.info("Database tables created successfully")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise

    async def store_document(self, document: Document) -> bool:
        """Store a document in the database."""
        try:
            # Ensure database is initialized
            if not self._initialized:
                await self.initialize()

            doc_id = f"{document.source}_{document.chunk_id}"
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO documents 
                    (id, content, embedding, source, file_type, metadata, embedding_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        document.content,
                        document.embedding.tobytes() if document.embedding is not None else None,
                        document.source,
                        document.file_type,
                        json.dumps(document.metadata),
                        document.metadata.get("embedding_model")
                    )
                )
                await db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store document: {e}")
            return False

    async def search_similar_documents(
        self,
        query_embedding: np.ndarray,
        embedding_model: str,
        n_results: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity."""
        try:
            # Ensure database is initialized
            if not self._initialized:
                await self.initialize()

            results = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT content, embedding, source, file_type, metadata
                    FROM documents
                    WHERE embedding_model = ?
                    """,
                    (embedding_model,)
                ) as cursor:
                    async for row in cursor:
                        content, embedding_bytes, source, file_type, metadata = row
                        if embedding_bytes is None:
                            continue
                            
                        doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        
                        if similarity >= similarity_threshold:
                            doc = Document(
                                content=content,
                                source=source,
                                file_type=file_type,
                                metadata=json.loads(metadata)
                            )
                            results.append((doc, similarity))
            
            # Sort by similarity and get top n
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def get_document_count(self) -> int:
        """Get total number of documents in the database."""
        try:
            # Ensure database is initialized
            if not self._initialized:
                await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT COUNT(*) FROM documents") as cursor:
                    result = await cursor.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            return 0

    async def close(self):
        """Close database connection."""
        self._initialized = False

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from the database."""
        try:
            if not self._initialized:
                await self.initialize()

            documents = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT content, source, file_type, metadata, embedding
                    FROM documents
                    """
                ) as cursor:
                    async for row in cursor:
                        content, source, file_type, metadata_str, embedding_bytes = row
                        
                        try:
                            metadata = json.loads(metadata_str) if metadata_str else {}
                        except json.JSONDecodeError:
                            metadata = {}
                        
                        doc = {
                            "content": content,
                            "source": source,
                            "file_type": file_type,
                            "metadata": metadata
                        }
                        
                        if embedding_bytes:
                            try:
                                doc["embedding"] = np.frombuffer(
                                    embedding_bytes,
                                    dtype=np.float32
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to convert embedding for {source}: {e}"
                                )
                        
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            raise

    async def get_file_type_distribution(self) -> Dict[str, int]:
        """Get distribution of file types in the database."""
        try:
            distribution = {}
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT file_type, COUNT(*) as count
                    FROM documents
                    GROUP BY file_type
                    """
                ) as cursor:
                    async for row in cursor:
                        file_type, count = row
                        distribution[file_type] = count
            return distribution
        except Exception as e:
            self.logger.error(f"Failed to get file type distribution: {e}")
            return {}

    async def get_average_chunk_size(self) -> float:
        """Get average chunk size of documents."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT AVG(LENGTH(content)) as avg_size
                    FROM documents
                    """
                ) as cursor:
                    row = await cursor.fetchone()
                    return float(row[0]) if row and row[0] else 0.0
        except Exception as e:
            self.logger.error(f"Failed to get average chunk size: {e}")
            return 0.0

    async def get_total_chunks(self) -> int:
        """Get total number of chunks in the database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT COUNT(*) FROM documents"
                ) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else 0
        except Exception as e:
            self.logger.error(f"Failed to get total chunks: {e}")
            return 0

    async def get_embedding_model_distribution(self) -> Dict[str, int]:
        """Get distribution of embedding models used."""
        try:
            distribution = {}
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT embedding_model, COUNT(*) as count
                    FROM documents
                    WHERE embedding_model IS NOT NULL
                    GROUP BY embedding_model
                    """
                ) as cursor:
                    async for row in cursor:
                        model, count = row
                        distribution[model] = count
            return distribution
        except Exception as e:
            self.logger.error(f"Failed to get embedding model distribution: {e}")
            return {}