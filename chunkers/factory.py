# chunker_factory.py
from typing import Dict, Type

from base import IDocumentChunker

class ChunkerFactory:
    """
    Фабрика для создания чанкеров на основе типа документа
    Реализует паттерн Factory Method
    """
    
    _chunkers: Dict[str, Type[IDocumentChunker]] = {}
    
    @classmethod
    def create_chunker(cls, doc_type: str, **kwargs) -> IDocumentChunker:
        """
        Возвращает чанкер для указанного типа документа
        
        Args:
            doc_type: тип документа ("legal_act" или "internal_policy")
            **kwargs: параметры для чанкера (chunk_size, chunk_overlap)
        
        Returns:
            DocumentChunker: экземпляр чанкера
        """
        chunker_class = cls._chunkers.get(doc_type, )

        return chunker_class(**kwargs)
    
    @classmethod
    def register_chunker(cls, doc_type: str):
        """Регистрирует новый чанкер для типа документа"""

        def decorator(chunker_class: Type[IDocumentChunker]):
            cls._chunkers[doc_type] = chunker_class
            return chunker_class
        
        return decorator