from typing import Dict, Any

from base import IDocumentChunker


class BaseChunker(IDocumentChunker):
    """Абстрактный базовый класс для всех чанкеров"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Создает структуру чанка"""
        return {
            "text": text,
            "metadata": {
                **metadata,
                "chunk_index": index
            }
        }