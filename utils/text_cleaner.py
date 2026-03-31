import re
import logging

from typing import List


from llama_index.core import Document

class TextCleaner:
    """
    Класс для очистки текста от специальных символов и разрывов страниц
    """
    
    # Паттерны для разрывов страниц в разных форматах
    PAGE_BREAK_PATTERNS = [
        r'\f',                          # Form feed (ASCII 12)
        r'---+\s*',                     # Горизонтальные линии
        r'_{3,}\s*',                    # Подчеркивания
        r'={3,}\s*',                    # Знаки равенства
        r'\n\s*\n\s*\n+',               # Множественные переносы строк (3+)
        r'Page\s+\d+\s*',                # "Page 1"
        r'Страница\s+\d+\s*',            # "Страница 1"
        r'стр\.?\s*\d+\s*',              # "стр. 1"
        r'\[\d+\]',                      # Номера страниц в квадратных скобках [1]
        r'\d+\s*\n\s*\d+\s*\n',          # Номера страниц отдельно
        r'----+\s*'                       # Линии из дефисов
    ]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Основной метод очистки текста
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Очищенный текст
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # 1. Удаляем разрывы страниц
        for pattern in cls.PAGE_BREAK_PATTERNS:
            text = re.sub(pattern, '\n', text, flags=re.IGNORECASE)
        
        # 2. Нормализуем переносы строк
        text = cls._normalize_line_breaks(text)
        
        # 3. Удаляем лишние пробелы
        text = cls._remove_extra_spaces(text)
        
        # 4. Восстанавливаем форматирование
        text = cls._restore_formatting(text)
        
        cleaned_length = len(text)
        cls.logger.info(f"Очистка текста: {original_length} -> {cleaned_length} символов (удалено {original_length - cleaned_length})")
        
        return text
    
    @classmethod
    def _normalize_line_breaks(cls, text: str) -> str:
        """Нормализация переносов строк"""

        # Заменяем множественные переносы на двойные
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Убираем переносы в начале и конце
        text = text.strip()

        return text
    
    @classmethod
    def _remove_extra_spaces(cls, text: str) -> str:
        """Удаление лишних пробелов"""

        # Убираем множественные пробелы
        text = re.sub(r' {2,}', ' ', text)

        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)

        # Добавляем пробелы после знаков препинания где нужно
        text = re.sub(r'([.,;:!?)])([^\s])', r'\1 \2', text)

        return text
    
    @classmethod
    def _restore_formatting(cls, text: str) -> str:
        """Восстановление форматирования"""

        # Восстанавливаем переносы после точек, если это конец предложения
        text = re.sub(r'\.([А-ЯA-Z])', r'. \1', text)

        return text
    
    @classmethod
    def clean_document(cls, document: Document) -> Document:
        """Очистка документа"""
        cleaned_text = cls.clean_text(document.text)

        

        # Создаем новый документ с очищенным текстом
        return Document(
            text=cleaned_text,
            metadata=document.metadata,
            doc_id=document.doc_id
        )
    
    @classmethod
    def clean_documents(cls, documents: List[Document]) -> List[Document]:
        """Очистка списка документов"""

        cleaned = []

        for doc in documents:
            cleaned_doc = cls.clean_document(doc)
            cleaned.append(cleaned_doc)

        return cleaned