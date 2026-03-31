from typing import List, Dict, Any

from llama_index.core import Document

from base import IDocumentChunker


class BaseChunker(IDocumentChunker):
    """Абстрактный базовый класс для всех чанкеров"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: Document) -> List[Dict[str, Any]]:

        return [{
            "text": None, 
            "metadata": {}
        }]      