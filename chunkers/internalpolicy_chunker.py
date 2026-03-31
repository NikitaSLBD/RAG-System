# chunkers/policy_chunker.py
import re

from typing import List, Dict, Any

from llama_index.core import Document

from base import DocumentType

from .base_chunker import BaseChunker
from .factory import ChunkerFactory

@ChunkerFactory.register_chunker(DocumentType.INTERNAL_POLICY.value)
class InternalPolicyChunker(BaseChunker):
    """
    Чанкер для внутренних политик и положений
    Использует семантическую разбивку с учетом глав и пунктов
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        
        # Паттерны для структуры политики
        self.chapter_pattern = re.compile(
            r'(##?\s*ГЛАВА\s+(\d+)[^\n]*\n)(.*?)(?=\n##?\s*ГЛАВА|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        self.section_pattern = re.compile(
            r'((?:\d+\.\d+|\d+\.)\s+[^\n]+\n)(.*?)(?=\n\d+\.\d+\s+|\n##?\s*ГЛАВА|\Z)',
            re.DOTALL
        )
    
    def chunk(self, document: Document) -> List[Dict[str, Any]]:
        """Разбивает документ на чанки"""
        
        # Сначала пробуем разбить по главам
        chapters = list(self.chapter_pattern.finditer(document.text))
        
        if chapters:
            return self._chunk_by_chapters(document, chapters)
        else:
            return self._chunk_semantic(document)
    
    def _chunk_by_chapters(self, document: Document, chapters: List) -> List[Dict[str, Any]]:
        """Разбивает документ по главам"""
        chunks = []
        
        for idx, match in enumerate(chapters):
            chapter_header = match.group(1).strip()
            chapter_num = match.group(2)
            chapter_content = match.group(3).strip()
            
            # Пробуем разбить главу на пункты
            sections = list(self.section_pattern.finditer(chapter_content))
            
            if sections and len(chapter_content) > self.chunk_size:
                # Разбиваем главу на пункты
                for sec_idx, sec_match in enumerate(sections):
                    sec_header = sec_match.group(1).strip()
                    sec_content = sec_match.group(2).strip()
                    
                    chunk_text = f"{chapter_header}\n{sec_header}\n{sec_content}"
                    
                    metadata = {
                        **document.metadata,
                        "chunk_type": "section",
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_header,
                        "section_title": sec_header,
                        "chapter_index": idx,
                        "section_index": sec_idx,
                    }
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata
                    })
            else:
                # Вся глава как один чанк
                chunk_text = f"{chapter_header}\n{chapter_content}"
                
                # Если глава слишком большая, дополнительно разбиваем
                if len(chunk_text) > self.chunk_size:
                    sub_chunks = self._split_long_text(chunk_text)
                    for sub_idx, sub_text in enumerate(sub_chunks):
                        metadata = {
                            **document.metadata,
                            "chapter_number": chapter_num,
                            "chapter_title": chapter_header,
                            "chapter_index": idx,
                            "subchunk_index": sub_idx,
                        }
                        chunks.append({
                            "text": sub_text,
                            "metadata": metadata
                        })
                else:
                    metadata = {
                        **document.metadata,
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_header,
                        "chapter_index": idx,
                    }
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata
                    })
        
        return chunks
    
    def _chunk_semantic(self, document: Document) -> List[Dict[str, Any]]:
        """Семантическая разбивка (если нет четкой структуры)"""
        chunks = []
        text = document.text
        
        # Разбиваем по абзацам
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len > self.chunk_size and current_chunk:
                # Сохраняем текущий чанк
                chunk_text = '\n\n'.join(current_chunk)
                metadata = {
                    **document.metadata,
                    "chunk_index": len(chunks),
                }
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })
                
                # Начинаем новый чанк с перекрытием
                overlap_size = self.chunk_overlap // 2
                if len(current_chunk) > 1:
                    current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                else:
                    current_chunk = []
                current_length = sum(len(p) for p in current_chunk)
            
            current_chunk.append(para)
            current_length += para_len
        
        # Последний чанк
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            metadata = {
                **document.metadata,
                "chunk_type": "semantic",
                "chunk_index": len(chunks),
            }
            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
        
        return chunks
    
    def _split_long_text(self, text: str) -> List[str]:
        """Разбивает длинный текст на части"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunks.append(' '.join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks