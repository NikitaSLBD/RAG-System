from typing import List
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Конфигурация приложения из переменных окружения"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env.example",  # файл .env.example в той же папке
        env_file_encoding="utf-8",
        case_sensitive=False  # регистронезависимые имена
    )
    
    # ==== Эмбеддер ====
    EMBED_MODEL: str = "cointegrated/LaBSE-en-ru" 
    DEVICE: str = "cpu"
    EMBED_TIMEOUT: int = 120
    EMBED_BATCH_SIZE: int = 32
    
    # ==== Chroma ====
    COLLECTION_NAME: str = "legal_documents"
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.6   

    # ==== Чанкер ====
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100

    
    # ==== LLM ====
    LLM: str = "o1" # заглушка для OpenAI, модель не эта
    LLM_URL: str = "http://127.0.0.1:1234/v1" # нужно захостить в LM Studio
    LLM_TIMEOUT: int = 120
    SYSTEM_PROMPT: str = """
    Ты - эксперт по анализу документов. 
    Твоя задача - отвечать на вопросы, используя ТОЛЬКО информацию из предоставленных документов.

    ОСНОВНЫЕ ПРАВИЛА:
    1. Отвечай исключительно на основе контекста документов.
    2. Если ответа нет в документах, скажи: "В предоставленных документах нет информации по этому вопросу."
    3. Цитируй релевантные фрагменты документов.
    4. Обязательно указывай источники (название документа, и т. д.).
    5. Не добавляй информацию из внешних источников.
    6. Не обобщай и не делай предположений.
    7. Отвечай на том же языке, на котором задан вопрос.
    8. Не перефразируй юридические или технические термины.
    9. Старайся отвечать кратко, ограничь длину ответа до 100 слов.
    """

    SUP_EXTS: List[str] = ["pdf", "txt"]

    # ==== Пути ====
    DOCS_DIR: Path = Path("./data")
    PERSIST_DIR: Path = Path("./chroma_db")
        
settings = Settings()