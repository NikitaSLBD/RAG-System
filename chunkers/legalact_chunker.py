import re

from typing import List, Dict, Any

from  llama_index.core import Document

from base import DocumentType

from .base_chunker import BaseChunker
from .factory import ChunkerFactory

@ChunkerFactory.register_chunker(DocumentType.LEGAL_ACT.value)
class LegalActChunker(BaseChunker):
    """
    Чанкер для нормативно-правовых актов (Конституция, Кодексы)
    Разбивает по статьям с сохранением иерархии
    """
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        
        # Паттерн для поиска статей
        self.article_pattern = re.compile(
            r'(Статья\s+(\d+|[IVX]+)(?:[А-Я])?[\.]?\s*\n?)(.*?)(?=\n\s*Статья\s+\d+|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        
        # Паттерны для иерархии
        self.section_pattern = re.compile(r'^РАЗДЕЛ\s+([IVX]+)\s+(.+)$', re.MULTILINE | re.IGNORECASE)
        self.chapter_pattern = re.compile(r'^ГЛАВА\s+(\d+)\s+(.+)$', re.MULTILINE | re.IGNORECASE)
    
    def chunk(self, document: Document) -> List[Dict[str, Any]]:
        """Разбивает документ на чанки по статьям"""
        chunks = []
        text = document.text
        
        # Извлекаем иерархию документа
        hierarchy = self._extract_hierarchy(text)
        
        # Находим все статьи
        articles = list(self.article_pattern.finditer(text))
        
        for idx, match in enumerate(articles):
            article_title = match.group(1).strip()
            article_num = match.group(2)
            article_content = match.group(3).strip()
            
            # Очищаем контент
            article_content = re.sub(r'\n\s*\n', '\n', article_content)
            
            # Формируем полный текст чанка
            chunk_text = f"{article_title}\n{article_content}"
            
            # Создаем метаданные
            metadata = {
                **document.metadata,
                "article_number": article_title,
                "article_index": idx,
                "total_articles": len(articles),
                "section": hierarchy.get("section", ""),
                "chapter": hierarchy.get("chapter", ""),
                "section_raw": hierarchy.get("section_raw", ""),
                "chapter_raw": hierarchy.get("chapter_raw", ""),
            }
            
            # Добавляем связи с соседними статьями
            if idx > 0:
                metadata["prev_article"] = articles[idx-1].group(1).strip()
            if idx < len(articles) - 1:
                metadata["next_article"] = articles[idx+1].group(1).strip()
            
            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
        
        return chunks
    
    def _extract_hierarchy(self, text: str) -> Dict[str, str]:
        """Извлекает разделы и главы из текста"""
        hierarchy = {
            "section": "",
            "section_raw": "",
            "chapter": "",
            "chapter_raw": ""
        }
        
        # Ищем разделы
        sections = list(self.section_pattern.finditer(text))
        if sections:
            last_section = sections[-1]
            hierarchy["section_raw"] = last_section.group(0).strip()
            hierarchy["section"] = last_section.group(2).strip()
        
        # Ищем главы
        chapters = list(self.chapter_pattern.finditer(text))
        if chapters:
            last_chapter = chapters[-1]
            hierarchy["chapter_raw"] = last_chapter.group(0).strip()
            hierarchy["chapter"] = last_chapter.group(2).strip()
        
        return hierarchy