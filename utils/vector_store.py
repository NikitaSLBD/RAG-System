# vector_store.py
import uuid
import logging
import chromadb

from typing import List, Dict, Any, Optional
from chromadb.config import Settings as ChromaSettings
from base import IVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore(IVectorStore):
    """
    Векторное хранилище на основе ChromaDB с поддержкой метаданных
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "legal_documents",
        embedding_model_name: str = "cointegrated/LaBSE-en-ru",
        embed_batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Args:
            persist_directory: путь для сохранения базы данных
            collection_name: имя коллекции по умолчанию
            embedding_model_name: имя модели HuggingFace
            embed_batch_size: размер батча для эмбеддингов
            device: устройство ('cpu' или 'cuda')
        """
        self.persist_directory = persist_directory
        self.default_collection = collection_name
        self.embedding_model_name = embedding_model_name
        self.embed_batch_size = embed_batch_size
        self.device = device
        
        # Инициализация HuggingFaceEmbedding
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            print(f"🔄 Загрузка модели: {embedding_model_name}")
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                embed_batch_size=embed_batch_size,
                device=device
            )
            self.use_embeddings = True
            # Проверка размерности
            test_embedding = self.embedding_model.get_text_embedding("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Размерность эмбеддингов: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"⚠️ Ошибка загрузки модели: {e}")
            self.use_embeddings = False
            
        
        # Настройка ChromaDB клиента
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Кэш коллекций
        self._collections = {}
    
    def _get_collection(self, collection_name: str):
        """Получает или создает коллекцию"""
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_name]
    
    def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Генерирует эмбеддинги для текстов через HuggingFaceEmbedding
        """
        if self.use_embeddings:
            embeddings = []
            for text in texts:
                embedding = self.embedding_model.get_text_embedding(text)
                embeddings.append(embedding)
            return embeddings
        return None
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Генерирует эмбеддинг для поискового запроса"""
        if self.use_embeddings:
            return self.embedding_model.get_query_embedding(query)
        return None
    
    def add_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        collection: str = None,
        show_progress: bool = False
    ) -> None:
        """
        Добавляет чанки в векторное хранилище
        """
        collection_name = collection or self.default_collection
        coll = self._get_collection(collection_name)
        
        # Подготовка данных
        ids = []
        documents = []
        metadatas = []
        texts_for_embeddings = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk["text"])
            # Добавляем uuid в метаданные
            metadata = chunk["metadata"].copy()
            metadata["chunk_uuid"] = chunk_id
            metadatas.append(metadata)
            texts_for_embeddings.append(chunk["text"])
        
        # Генерация эмбеддингов
        if show_progress:
            logger.info(f"🔄 Генерация эмбеддингов для {len(chunks)} чанков...")
        
        embeddings = self._get_embeddings(texts_for_embeddings)
        
        # Добавление в ChromaDB
        if embeddings:
            coll.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            coll.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        logger.info(f"✅ Добавлено {len(chunks)} чанков в коллекцию '{collection_name}'")
    
    def search(
        self, 
        query: str, 
        collection: str = None, 
        top_k: int = 5, 
        filters: Optional[Dict] = None,
        where: Optional[Dict] = None,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Поиск похожих чанков
        """
        collection_name = collection or self.default_collection
        coll = self._get_collection(collection_name)
        
        # Генерация эмбеддинга для запроса
        query_embedding = self._get_query_embedding(query)
        
        if query_embedding:
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Запрашиваем больше для фильтрации по порогу
                where=filters or where,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = coll.query(
                query_texts=[query],
                n_results=top_k * 2,
                where=filters or where,
                include=["documents", "metadatas", "distances"]
            )
        
        # Форматирование результатов
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                score = 1 - distance
                
                if score >= score_threshold:
                    formatted_results.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": score,
                        "distance": distance,
                        "id": results['ids'][0][i]
                    })
        
        # Сортируем по score и обрезаем до top_k
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        return formatted_results[:top_k]
    
    def search_batch(
        self,
        queries: List[str],
        collection: str = None,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        Пакетный поиск для нескольких запросов
        """
        collection_name = collection or self.default_collection
        coll = self._get_collection(collection_name)
        
        # Генерация эмбеддингов для всех запросов
        query_embeddings = []
        for query in queries:
            emb = self._get_query_embedding(query)
            if emb:
                query_embeddings.append(emb)
        
        if query_embeddings:
            results = coll.query(
                query_embeddings=query_embeddings,
                n_results=top_k * 2,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
        else:
            # Fallback на текстовый поиск
            batch_results = []
            for query in queries:
                res = self.search(query, collection, top_k, filters, score_threshold=score_threshold)
                batch_results.append(res)
            return batch_results
        
        # Форматирование результатов
        batch_results = []
        for batch_idx in range(len(queries)):
            query_results = []
            if results['ids'] and results['ids'][batch_idx]:
                for i in range(len(results['ids'][batch_idx])):
                    distance = results['distances'][batch_idx][i]
                    score = 1 - distance
                    
                    if score >= score_threshold:
                        query_results.append({
                            "text": results['documents'][batch_idx][i],
                            "metadata": results['metadatas'][batch_idx][i],
                            "score": score,
                            "distance": distance,
                            "id": results['ids'][batch_idx][i]
                        })
            
            query_results.sort(key=lambda x: x["score"], reverse=True)
            batch_results.append(query_results[:top_k])
        
        return batch_results
    
    def delete_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию"""
        if collection_name in self._collections:
            del self._collections[collection_name]
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"🗑️ Коллекция '{collection_name}' удалена")
        except Exception as e:
            logger.error(f"⚠️ Ошибка удаления коллекции: {e}")
    
    def list_collections(self) -> List[str]:
        """Возвращает список всех коллекций"""
        return [col.name for col in self.client.list_collections()]
    
    def get_collection_stats(self, collection: str = None) -> Dict:
        """Возвращает статистику по коллекции"""
        collection_name = collection or self.default_collection
        try:
            coll = self._get_collection(collection_name)
            count = coll.count()
            
            return {
                "collection_name": collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name if self.use_embeddings else "chroma_default",
                "embedding_dim": getattr(self, 'embedding_dim', 'unknown')
            }
        except Exception as e:
            return {
                "collection_name": collection_name,
                "error": str(e),
                "total_chunks": 0
            }
    
    def reset(self) -> None:
        """Полный сброс хранилища"""
        for collection_name in self.list_collections():
            self.delete_collection(collection_name)
        self._collections = {}
        logger.info("🔄 Хранилище полностью сброшено")