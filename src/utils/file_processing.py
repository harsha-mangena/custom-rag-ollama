from typing import List, Dict, Any, Optional
import io
from pathlib import Path
import logging
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import markdown
import tiktoken
from config.settings import Settings

class FileProcessor:
    """Handles file processing and text chunking operations."""
    
    def __init__(
        self,
        chunk_size: int = Settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.CHUNK_OVERLAP
    ):
        """Initialize file processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def extract_content(self, content: bytes, file_type: str) -> str:
        """Extract text content from various file types."""
        try:
            if file_type == 'pdf':
                return await self._extract_pdf(content)
            elif file_type == 'docx':
                return await self._extract_docx(content)
            elif file_type == 'md':
                return markdown.markdown(content.decode('utf-8'))
            elif file_type == 'txt':
                return content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            self.logger.error(f"Content extraction failed: {e}")
            raise

    async def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                
            return self._clean_text(text)
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            raise

    async def _extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            docx_file = io.BytesIO(content)
            doc = DocxDocument(docx_file)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            return self._clean_text(text)
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove multiple newlines
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text

    def create_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """Create overlapping chunks of text."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap

        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            # Create chunks based on tokens
            i = 0
            while i < len(tokens):
                # Get chunk tokens
                chunk_tokens = tokens[i:i + chunk_size]
                
                # Decode chunk tokens back to text
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                # Move to next chunk with overlap
                i += (chunk_size - chunk_overlap)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            # Fallback to simple text splitting
            return self._simple_chunk_text(text, chunk_size, chunk_overlap)

    def _simple_chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Simple text chunking fallback method."""
        words = text.split()
        chunks = []
        i = 0
        
        while i < len(words):
            # Get chunk words
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
            
            # Move to next chunk with overlap
            i += (chunk_size - chunk_overlap)
            
        return chunks
