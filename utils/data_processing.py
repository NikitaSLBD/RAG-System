# data_processing.py
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from base import IDocumentLoader, IVectorStore
from utils.document_loader import DocumentLoader
from utils.text_cleaner import TextCleaner
from chunkers.factory import ChunkerFactory  
from .vector_store import ChromaVectorStore  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    """
    Основной пайплайн обработки документов
    Инжектирует зависимости через конструктор
    """
    
    def __init__(
        self,
        loaders: List[IDocumentLoader] = None,
        cleaner: TextCleaner = None,
        vector_store: IVectorStore = None,
        default_collection: str = "legal_documents",
        chunk_size: int = 1024,
        chunk_overlap: int = 200
    ):
        """
        Args:
            loaders: список загрузчиков документов
            cleaner: очиститель текста
            vector_store: векторное хранилище
            default_collection: имя коллекции по умолчанию
            chunk_size: размер чанка
            chunk_overlap: перекрытие чанков
        """
        self.loaders = loaders or [DocumentLoader()]
        self.cleaner = cleaner or TextCleaner()
        self.vector_store = vector_store or ChromaVectorStore()
        self.default_collection = default_collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Статистика обработки
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "errors": []
        }
    
    def process_file(
        self, 
        file_path: str, 
        collection: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Обрабатывает один файл: загрузка → очистка → чанкинг → индексация
        """
        collection_name = collection or self.default_collection
        
        if verbose:
            logger.info(f"📄 Начало обработки: {file_path}")
        
        # 1. Загрузка документа (используем DocumentLoader)
        try:
            # DocumentLoader.load ожидает Path и список расширений
            # Для одного файла создаем временный список
            file_path_obj = Path(file_path)
            docs = self.loaders[0].load(
                input_dir=Path(file_path_obj.parent)
            )
            # Фильтруем нужный файл
            documents = [doc for doc in docs if doc.metadata.get('file_name') == file_path_obj.name]
            
            if not documents:
                error_msg = f"Не удалось загрузить: {file_path}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
                return {"success": False, "error": error_msg}
            
            if verbose:
                logger.info(f"   ✅ Загружено {len(documents)} документов")
        except Exception as e:
            error_msg = f"Ошибка загрузки {file_path}: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return {"success": False, "error": error_msg}
        
        # 2. Очистка и чанкинг для каждого документа
        all_chunks = []
        
        for doc in documents:
            try:
                # Очистка
                cleaned_doc = self.cleaner.clean_document(doc)
                
                # Получение типа документа из метаданных
                doc_type = cleaned_doc.metadata.get('doc_type', 'unknown')
                
                # Получение чанкера через фабрику
                chunker = ChunkerFactory.create_chunker(
                    doc_type=doc_type,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Чанкинг
                chunks = chunker.chunk(cleaned_doc)
                all_chunks.extend(chunks)
                
                if verbose:
                    logger.info(f"   📝 {doc_type}: {len(chunks)} чанков")
            except Exception as e:
                error_msg = f"Ошибка обработки документа {doc.metadata.get('file_name', 'unknown')}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        # 3. Индексация в векторном хранилище
        if all_chunks:
            try:
                self.vector_store.add_documents(all_chunks, collection=collection_name)
                self.stats["documents_processed"] += len(documents)
                self.stats["total_chunks"] += len(all_chunks)
            except Exception as e:
                error_msg = f"Ошибка индексации {file_path}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
                return {"success": False, "error": error_msg}
        
        if verbose:
            logger.info(f"   ✅ Завершено: {len(all_chunks)} чанков добавлено в '{collection_name}'")
        
        return {
            "success": True,
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "collection": collection_name,
            "file": str(file_path)
        }
    
    def process_directory(
        self,
        directory_path: str,
        collection: str = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Обрабатывает все файлы в директории
        """

        path = Path(directory_path)
        
        if not path.exists():
            return {"success": False, "error": f"Директория не найдена: {directory_path}"}
        
        # Загружаем все документы через DocumentLoader
        try:
            documents = self.loaders[0].load(
                input_dir=path
            )
            
            if verbose:
                logger.info(f"📁 Загружено {len(documents)} документов из {directory_path}")
        except Exception as e:
            return {"success": False, "error": f"Ошибка загрузки документов: {e}"}
        
        # Обработка каждого документа
        all_chunks = []
        
        for doc in documents:
            try:
                # Очистка
                cleaned_doc = self.cleaner.clean_document(doc)
                
                # Получение типа документа из метаданных
                doc_type = cleaned_doc.metadata.get('doc_type', 'unknown')
                
                # Получение чанкера через фабрику
                chunker = ChunkerFactory.create_chunker(
                    doc_type=doc_type,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Чанкинг
                chunks = chunker.chunk(cleaned_doc)
                all_chunks.extend(chunks)
                
                if verbose:
                    logger.info(f"   📝 {doc.metadata.get('file_name')} ({doc_type}): {len(chunks)} чанков")
            except Exception as e:
                error_msg = f"Ошибка обработки {doc.metadata.get('file_name', 'unknown')}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        # Индексация
        collection_name = collection or self.default_collection
        if all_chunks:
            try:
                self.vector_store.add_documents(all_chunks, collection=collection_name)
                self.stats["documents_processed"] += len(documents)
                self.stats["total_chunks"] += len(all_chunks)
            except Exception as e:
                error_msg = f"Ошибка индексации: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
                return {"success": False, "error": error_msg}
        
        if verbose:
            logger.info(f"🎉 Обработка завершена: {len(documents)} документов, {len(all_chunks)} чанков")
        
        return {
            "success": True,
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "collection": collection_name,
            "directory": str(directory_path)
        }
    
    def search(
        self,
        query: str,
        collection: str = None,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        doc_type_filter: Optional[str] = None,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Поиск в индексе
        """
        collection_name = collection or self.default_collection
        
        # Добавляем фильтр по типу документа
        where_filter = filters.copy() if filters else {}
        if doc_type_filter:
            where_filter["doc_type"] = doc_type_filter
        
        results = self.vector_store.search(
            query=query,
            collection=collection_name,
            top_k=top_k,
            where=where_filter if where_filter else None,
            score_threshold=score_threshold
        )
        
        return results
    
    def search_with_context(
        self,
        query: str,
        collection: str = None,
        top_k: int = 5,
        include_neighbors: bool = True,
        neighbor_count: int = 1,
        **kwargs
    ) -> List[Dict]:
        """
        Поиск с контекстом соседних чанков (полезно для статей)
        """
        results = self.search(query, collection, top_k, **kwargs)
        
        if not include_neighbors:
            return results
        
        # Для каждого результата ищем соседние чанки
        for result in results:
            metadata = result.get("metadata", {})
            
            neighbors = []
            
            if "prev_article" in metadata and metadata["prev_article"]:
                prev_results = self.search(
                    query=metadata["prev_article"],
                    collection=collection,
                    top_k=neighbor_count,
                    doc_type_filter=metadata.get("doc_type")
                )
                if prev_results:
                    neighbors.append({"position": "prev", "chunks": prev_results[:neighbor_count]})
            
            if "next_article" in metadata and metadata["next_article"]:
                next_results = self.search(
                    query=metadata["next_article"],
                    collection=collection,
                    top_k=neighbor_count,
                    doc_type_filter=metadata.get("doc_type")
                )
                if next_results:
                    neighbors.append({"position": "next", "chunks": next_results[:neighbor_count]})
            
            if neighbors:
                result["neighbors"] = neighbors
        
        return results
    
    def get_stats(self) -> Dict:
        """Возвращает статистику обработки"""
        stats = self.vector_store.get_collection_stats(self.default_collection)
        stats.update({
            "documents_processed": self.stats["documents_processed"],
            "total_chunks": self.stats["total_chunks"],
            "errors": self.stats["errors"]
        })
        return stats
    
    def clear_collection(self, collection: str = None) -> None:
        """Очищает коллекцию"""
        collection_name = collection or self.default_collection
        self.vector_store.delete_collection(collection_name)
        self.stats = {"documents_processed": 0, "total_chunks": 0, "errors": []}
    
    def reset(self) -> None:
        """Полный сброс"""
        self.vector_store.reset()
        self.stats = {"documents_processed": 0, "total_chunks": 0, "errors": []}