"""
Fetcher Module
Document fetching from various sources.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FetchResult:
    """Result of document fetch."""
    content: bytes
    content_type: str
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class DocumentFetcher:
    """
    Fetches documents from various sources.
    
    Supports:
    - HTTP/HTTPS URLs
    - S3 buckets
    - Local files
    - Base64 encoded content
    """
    
    def __init__(self, http_client = None):
        """
        Initialize fetcher.
        
        Args:
            http_client: HTTP client for URL fetching
        """
        self.http_client = http_client
        
        logger.info("DocumentFetcher initialized")
    
    async def fetch(
        self,
        source: str,
        source_type: str = "auto"
    ) -> FetchResult:
        """
        Fetch document from source.
        
        Args:
            source: Source location (URL, path, etc.)
            source_type: Type of source (url, s3, file, base64)
            
        Returns:
            Fetch result with content
        """
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        try:
            if source_type == "url":
                return await self._fetch_url(source)
            elif source_type == "s3":
                return await self._fetch_s3(source)
            elif source_type == "file":
                return self._fetch_file(source)
            elif source_type == "base64":
                return self._decode_base64(source)
            else:
                return FetchResult(
                    content=b"",
                    content_type="unknown",
                    error=f"Unknown source type: {source_type}"
                )
        except Exception as e:
            logger.error("Fetch failed", source=source, error=str(e))
            return FetchResult(
                content=b"",
                content_type="unknown",
                error=str(e)
            )
    
    def _detect_source_type(self, source: str) -> str:
        """Detect source type from URL/path."""
        if source.startswith("http://") or source.startswith("https://"):
            return "url"
        elif source.startswith("s3://"):
            return "s3"
        elif source.startswith("data:"):
            return "base64"
        else:
            return "file"
    
    async def _fetch_url(self, url: str) -> FetchResult:
        """Fetch document from URL."""
        if self.http_client:
            response = await self.http_client.get(url, timeout=60)
            return FetchResult(
                content=response.content,
                content_type=response.headers.get("content-type", "text/html"),
                metadata={"status_code": response.status_code}
            )
        
        # Mock
        return FetchResult(
            content=b"Mock content from URL",
            content_type="text/html"
        )
    
    async def _fetch_s3(self, s3_uri: str) -> FetchResult:
        """Fetch document from S3."""
        # Parse s3://bucket/key
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        # TODO: Implement with boto3
        return FetchResult(
            content=f"Mock S3 content from {bucket}/{key}".encode(),
            content_type="application/octet-stream"
        )
    
    def _fetch_file(self, path: str) -> FetchResult:
        """Fetch document from local file."""
        try:
            with open(path, "rb") as f:
                content = f.read()
            
            # Detect content type
            content_type = self._detect_content_type(path)
            
            return FetchResult(
                content=content,
                content_type=content_type
            )
        except FileNotFoundError:
            return FetchResult(
                content=b"",
                content_type="unknown",
                error=f"File not found: {path}"
            )
    
    def _decode_base64(self, data_uri: str) -> FetchResult:
        """Decode base64 encoded content."""
        import base64
        
        # Parse data:mime;base64,content
        if "," in data_uri:
            header, content = data_uri.split(",", 1)
            content_type = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        else:
            content = data_uri
            content_type = "application/octet-stream"
        
        try:
            decoded = base64.b64decode(content)
            return FetchResult(content=decoded, content_type=content_type)
        except Exception as e:
            return FetchResult(content=b"", content_type="unknown", error=str(e))
    
    def _detect_content_type(self, path: str) -> str:
        """Detect content type from file extension."""
        ext_map = {
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".csv": "text/csv"
        }
        
        for ext, content_type in ext_map.items():
            if path.lower().endswith(ext):
                return content_type
        
        return "application/octet-stream"


__all__ = ["DocumentFetcher", "FetchResult"]
