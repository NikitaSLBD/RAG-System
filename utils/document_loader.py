import re
import logging

from typing import List
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader


from infrastructure.config import settings
from base import DocumentType, IDocumentLoader

def detect_document_type(text: str, filename: str = "") -> str:
    """
    Определяет тип документа по содержанию и/или имени файла

    Args:
        text: 

    
    """
    # 1. По имени файла (самый простой и надежный способ)
    if "конституция" in filename.lower() or "кодекс" in filename.lower():
        return DocumentType.LEGAL_ACT.value
    if "политика" in filename.lower() or "попд" in filename.lower():
        return DocumentType.INTERNAL_POLICY.value
    
    # 2. По содержанию (первые несколько тысяч символов)
    header = text[:2000].lower()
    
    # Признаки НПА
    legal_indicators = [
        r"статья\s+\d+",           
        r"глава\s+\d+",             
        r"раздел\s+[ivx]+",         
        r"конституция\s+",
        r"настоящий\s+кодекс",
    ]
    
    # Признаки внутренней политики
    policy_indicators = [
        r"политика\s+в\s+отношении\s+обработки\s+персональных\s+данных",
        r"оператор\s+осуществляет",
        r"субъект\s+персональных\s+данных\s+имеет\s+право",
        r"утверждено\s+приказом",
    ]
    
    for pattern in legal_indicators:
        if re.search(pattern, header):
            return DocumentType.LEGAL_ACT.value
            
    for pattern in policy_indicators:
        if re.search(pattern, header):
            return DocumentType.INTERNAL_POLICY.value
    
    # Дефолтный вариант
    return DocumentType.LEGAL_ACT.value if "статья" in header else DocumentType.INTERNAL_POLICY.value

class DocumentLoader(IDocumentLoader):

    """
    Класс для загрузки документов в систему
    """

    @staticmethod
    def load(input_dir: Path) -> List[Document]:

        """
        Загрузка как единых документов (все страницы одного документа вместе)

        Args:
            input_dir: Путь к директории с документами
            sup_exts: Список поддерживаемых расширений документов (без точки)

        Returns:
            List[Document]: Список загруженных документов 
        """
        
        all_docs = []

        files = []
        for ext in settings.SUP_EXTS: files.extend(input_dir.glob(f"*.{ext}"))
        
        for file in files:
            
            # Загружаем постранично
            reader = SimpleDirectoryReader(input_files=[str(file)])
            pages = reader.load_data()
            
            # Объединяем все страницы
            full_text = ""
            for i, page in enumerate(pages):
                full_text += page.text + "\n"

            # Создаем единый документ
            combined_doc = Document(
                text=full_text,
                metadata={
                    'file_name': file.name,
                    'total_pages': len(pages),
                    'doc_type': detect_document_type(full_text, file.name)
                },
                doc_id=f"{file.stem}"
            )
            
            all_docs.append(combined_doc)

        return all_docs
        