# main.py
from pathlib import Path

from chunkers.internalpolicy_chunker import InternalPolicyChunker
from chunkers.legalact_chunker import LegalActChunker

from utils.data_processing import DocumentProcessingPipeline
from utils.vector_store import ChromaVectorStore
from utils.text_cleaner import TextCleaner
from utils.document_loader import DocumentLoader

from infrastructure.config import settings

if __name__ == "__main__":
    
    print("=" * 60)
    print("🚀 ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ ДОКУМЕНТОВ")
    print("=" * 60)
    print(f"📁 Входная директория: {str(settings.DOCS_DIR)}")
    print(f"💾 Директория хранения: {str(settings.PERSIST_DIR)}")
    print(f"🤖 Модель эмбеддингов: {settings.EMBED_MODEL}")
    print(f"⚙️ Устройство: {settings.DEVICE}")
    print("=" * 60)
    
    # 1. Инициализация компонентов
    loader = DocumentLoader()
    cleaner = TextCleaner()
    vector_store = ChromaVectorStore(
        persist_directory=str(settings.PERSIST_DIR),
        collection_name=settings.COLLECTION_NAME,
        embedding_model_name=settings.EMBED_MODEL,
        embed_batch_size=settings.EMBED_BATCH_SIZE,
        device=settings.DEVICE
    )
    
    # 2. Создание пайплайна
    pipeline = DocumentProcessingPipeline(
        loaders=[loader],
        cleaner=cleaner,
        vector_store=vector_store,
        default_collection=settings.COLLECTION_NAME,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # 3. Обработка директории
    print("\n📚 НАЧАЛО ОБРАБОТКИ ДОКУМЕНТОВ")
    print("-" * 60)
    
    result = pipeline.process_directory(
        directory_path=str(settings.DOCS_DIR),
        collection=settings.COLLECTION_NAME,
        verbose=True
    )
    
    if not result["success"]:
        print(f"\n❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
    
    # 4. Вывод статистики
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ОБРАБОТКИ")
    print("=" * 60)
    stats = pipeline.get_stats()
    for key, value in stats.items():
        if key != 'errors' or (key == 'errors' and value):
            print(f"   {key}: {value}")
    
    # 5. Тестовый поиск
    print("\n" + "=" * 60)
    print("🔍 ТЕСТОВЫЙ ПОИСК")
    print("=" * 60)
    
    test_queries = [
        "Какие права имеют работники по Трудовому кодексу?",
        "обязанности нанимателя при приеме на работу",
        "защита персональных данных работников",
        "якія правы маюць работнікі?",  # белорусский
        "employer obligations when hiring",  # английский
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: {query}")
        print("-" * 40)
        
        results = pipeline.search(
            query=query,
            top_k=settings.TOP_K,
            score_threshold=settings.SIMILARITY_THRESHOLD
        )
        
        if results:
            for i, r in enumerate(results):
                print(f"\n   {i+1}. Score: {r['score']:.3f}")
                print(f"      Источник: {r['metadata'].get('file_name', 'unknown')}")
                print(f"      Тип: {r['metadata'].get('doc_type', 'unknown')}")
                if 'article_number' in r['metadata']:
                    print(f"      Статья: {r['metadata']['article_number']}")
                if 'chapter_title' in r['metadata']:
                    print(f"      Глава: {r['metadata']['chapter_title']}")
                print(f"      Текст: {r['text'][:200]}...")
        else:
            print("   Результатов не найдено")
    
    print("\n" + "=" * 60)
    print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    print("=" * 60)