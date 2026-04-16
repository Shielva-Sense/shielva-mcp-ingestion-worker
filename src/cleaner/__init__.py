"""
Cleaner Module
Text cleaning and normalization utilities.
"""
import re
from typing import Dict, Any


class TextCleaner:
    """
    Text cleaner for document preprocessing.
    
    Features:
    - Whitespace normalization
    - Special character handling
    - Unicode normalization
    - HTML entity decoding
    """
    
    @staticmethod
    def clean(text: str, config: Dict[str, Any] = None) -> str:
        """
        Clean text content.
        
        Args:
            text: Raw text
            config: Cleaning options
            
        Returns:
            Cleaned text
        """
        config = config or {}
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        
        # Decode HTML entities
        try:
            import html
            text = html.unescape(text)
        except:
            pass
        
        # Remove null bytes
        text = text.replace("\x00", "")
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Normalize whitespace
        if config.get("normalize_whitespace", True):
            text = re.sub(r"[ \t]+", " ", text)
        
        # Remove control characters (except newline and tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r"https?://\S+", "", text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r"\S+@\S+\.\S+", "", text)
    
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """Normalize numbers (optional)."""
        # Replace long numbers with placeholder
        return re.sub(r"\d{10,}", "[NUMBER]", text)
