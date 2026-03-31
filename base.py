# interfaces.py
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from llama_index.core import Document
    
class DocumentType(Enum):

    LEGAL_ACT = "legal_act"
    INTERNAL_POLICY = "internal_policy"
    
class IDocumentLoader(ABC):
    """Интерфейс загрузчика документов"""
    
    @abstractmethod
    def load(self, input_dir: Path) -> List[Document]:
        """Загружает документы из папки источника"""
        pass


class IDocumentChunker(ABC):
    """Интерфейс чанкера документов"""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Dict[str, Any]]:
        """
        Разбивает документ на чанки
        
        Args:
            document: исходный документ 

        Returns:
            List[Dict]: список чанков с полями:
                - text: текст чанка
                - metadata: метаданные чанка
        """
        pass


class IVectorStore(ABC):
    """Интерфейс векторного хранилища"""
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]], collection: str) -> None:
        """Добавляет чанки в хранилище"""
        pass
    
    @abstractmethod
    def search(self, query: str, collection: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Поиск похожих чанков по текстовому запросу"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию"""
        pass