from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

@dataclass
class Document:
    """Represents a document with its content and metadata."""
    content: str
    source: str
    file_type: str
    chunk_id: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            "content": self.content,
            "source": self.source,
            "file_type": self.file_type,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }