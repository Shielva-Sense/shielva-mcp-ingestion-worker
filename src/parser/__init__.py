"""
Parser Module
Document parsing for various formats.
"""
from typing import Dict, Any, Union
import structlog
from abc import ABC, abstractmethod

from ..models import DocumentType

logger = structlog.get_logger(__name__)


class BaseParser(ABC):
    """Base document parser"""
    
    @abstractmethod
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        """Parse document content to text"""
        pass


class TextParser(BaseParser):
    """Plain text parser"""
    
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        if isinstance(content, bytes):
            return content.decode('utf-8', errors='ignore')
        return content


class HTMLParser(BaseParser):
    """HTML to text parser"""
    
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        import re
        
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # Remove scripts and styles
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert common elements
        content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'</p>', '\n\n', content, flags=re.IGNORECASE)
        content = re.sub(r'</div>', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'</li>', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'<h[1-6][^>]*>', '\n\n## ', content, flags=re.IGNORECASE)
        content = re.sub(r'</h[1-6]>', '\n', content, flags=re.IGNORECASE)
        
        # Remove remaining tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Decode entities
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        content = content.replace('&#39;', "'")
        
        # Clean whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        return content.strip()


class MarkdownParser(BaseParser):
    """Markdown parser (keeps as-is for now)"""
    
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        if isinstance(content, bytes):
            return content.decode('utf-8', errors='ignore')
        return content


class PDFParser(BaseParser):
    """PDF parser using pymupdf"""
    
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        try:
            import fitz  # pymupdf
            
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            doc = fitz.open(stream=content, filetype="pdf")
            text_parts = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("pymupdf not installed, PDF parsing unavailable")
            return "[PDF content - requires pymupdf]"
        except Exception as e:
            logger.error("PDF parsing failed", error=str(e))
            raise


class DocxParser(BaseParser):
    """DOCX parser using python-docx"""
    
    async def parse(self, content: Union[bytes, str], metadata: Dict[str, Any] = None) -> str:
        try:
            from docx import Document as DocxDocument
            import io
            
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            doc = DocxDocument(io.BytesIO(content))
            text_parts = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            # Also get text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    text_parts.append(" | ".join(row_text))
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("python-docx not installed")
            return "[DOCX content - requires python-docx]"
        except Exception as e:
            logger.error("DOCX parsing failed", error=str(e))
            raise


# Parser registry
PARSERS: Dict[DocumentType, BaseParser] = {
    DocumentType.TEXT: TextParser(),
    DocumentType.HTML: HTMLParser(),
    DocumentType.MARKDOWN: MarkdownParser(),
    DocumentType.PDF: PDFParser(),
    DocumentType.DOCX: DocxParser(),
}

__all__ = ["BaseParser", "PARSERS", "TextParser", "HTMLParser", "MarkdownParser", "PDFParser", "DocxParser"]
