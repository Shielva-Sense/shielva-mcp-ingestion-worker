"""
Embedder Module
Embedding generation using various providers.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EmbedderConfig:
    """Embedder configuration."""
    provider: str = "openai"  # openai, cohere, local
    model: str = "text-embedding-3-small"
    api_key: str = ""
    batch_size: int = 100
    dimension: int = 1536


class EmbeddingClient:
    """
    Embedding generation client.
    
    Supports:
    - OpenAI embeddings
    - Cohere embeddings
    - Local models (sentence-transformers)
    """
    
    def __init__(self, config: EmbedderConfig = None):
        """
        Initialize embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config or EmbedderConfig()
        self._client = None
        
        logger.info(
            "EmbeddingClient initialized",
            provider=self.config.provider,
            model=self.config.model
        )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            if self.config.provider == "openai":
                embeddings = await self._embed_openai(batch)
            elif self.config.provider == "cohere":
                embeddings = await self._embed_cohere(batch)
            elif self.config.provider == "gemini":
                embeddings = await self._embed_gemini(batch)
            elif self.config.provider == "local":
                embeddings = await self._embed_local(batch)
            else:
                embeddings = self._embed_mock(batch)
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        result = await self.embed([text])
        return result[0] if result else []
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.config.api_key)
            response = await client.embeddings.create(
                model=self.config.model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error("OpenAI embedding failed", error=str(e))
            return self._embed_mock(texts)
    
    async def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Gemini."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.config.api_key)
            
            # Gemini batch embedding
            # "models/text-embedding-004"
            result = genai.embed_content(
                model=self.config.model,
                content=texts,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error("Gemini embedding failed", error=str(e))
            return self._embed_mock(texts)

    async def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        try:
            import cohere
            
            client = cohere.AsyncClient(self.config.api_key)
            response = await client.embed(
                texts=texts,
                model=self.config.model,
                input_type="search_document"
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error("Cohere embedding failed", error=str(e))
            return self._embed_mock(texts)
    
    async def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            if not self._client:
                self._client = SentenceTransformer(self.config.model)
            
            embeddings = self._client.encode(texts)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error("Local embedding failed", error=str(e))
            return self._embed_mock(texts)
    
    def _embed_mock(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing."""
        import random
        
        return [
            [random.random() for _ in range(self.config.dimension)]
            for _ in texts
        ]


__all__ = ["EmbeddingClient", "EmbedderConfig"]
