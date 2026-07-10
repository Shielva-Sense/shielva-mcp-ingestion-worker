"""
Chunker Module
Document chunking strategies.
"""

from typing import List, Dict, Any
from ..models import Document, Chunk, ChunkingStrategy


class Chunker:
    """
    Document chunker with multiple strategies.

    Supports:
    - Fixed size chunking
    - Sentence-based chunking
    - Paragraph-based chunking
    - Recursive chunking with overlap
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 10,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a document"""
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentences(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraphs(document)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(document)
        else:
            return self._chunk_fixed(document)

    def _chunk_fixed(self, document: Document) -> List[Chunk]:
        """Fixed size chunking"""
        chunks = []
        content = document.content

        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_content = content[start:end]

            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{chunk_idx}",
                        document_id=document.id,
                        content=chunk_content,
                        chunk_index=chunk_idx,
                        start_char=start,
                        end_char=end,
                        metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
                    )
                )
                chunk_idx += 1

            start = end - self.chunk_overlap

        return chunks

    def _chunk_sentences(self, document: Document) -> List[Chunk]:
        """Sentence-based chunking"""
        import re

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", document.content)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        start_char = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{chunk_idx}",
                        document_id=document.id,
                        content=chunk_content,
                        chunk_index=chunk_idx,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
                    )
                )
                chunk_idx += 1
                start_char += len(chunk_content) + 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len + 1

        # Don't forget the last chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_content = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                    metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
                )
            )

        return chunks

    def _chunk_paragraphs(self, document: Document) -> List[Chunk]:
        """Paragraph-based chunking"""
        paragraphs = document.content.split("\n\n")

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        start_char = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if current_length + para_len > self.chunk_size and current_chunk:
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{chunk_idx}",
                        document_id=document.id,
                        content=chunk_content,
                        chunk_index=chunk_idx,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
                    )
                )
                chunk_idx += 1
                start_char += len(chunk_content) + 2
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_len + 2

        if current_chunk and current_length >= self.min_chunk_size:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                    metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
                )
            )

        return chunks

    def _chunk_recursive(self, document: Document) -> List[Chunk]:
        """
        Recursive chunking - splits by headers, then paragraphs, then sentences.
        Best for structured documents.

        Section heading extraction (P4): the first header line of each section
        is stored as `section_heading` and `section_level` in every chunk's
        metadata so retrievers can surface which section a chunk came from.
        Sub-chunks of a large section inherit the heading via `parent_section`.
        """
        import re

        content = document.content
        chunks = []
        chunk_idx = 0

        _HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

        # First, try to split by headers
        header_pattern = r"(?=^#{1,3}\s|\n#{1,3}\s)"
        sections = re.split(header_pattern, content, flags=re.MULTILINE)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # P4: Extract section heading from the first header line
            section_heading = None
            section_level = None
            heading_match = _HEADING_RE.match(section)
            if heading_match:
                section_level = len(heading_match.group(1))
                section_heading = heading_match.group(2).strip()

            section_meta = {
                "title": document.title,
                "source_url": document.source_url,
                **document.metadata,
            }
            if section_heading:
                section_meta["section_heading"] = section_heading
                section_meta["section_level"] = section_level

            if len(section) <= self.chunk_size:
                if len(section) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            id=f"{document.id}_chunk_{chunk_idx}",
                            document_id=document.id,
                            content=section,
                            chunk_index=chunk_idx,
                            metadata=section_meta,
                        )
                    )
                    chunk_idx += 1
            else:
                # Section too large, split by paragraphs.
                # Sub-chunks inherit heading as parent_section.
                sub_meta = dict(section_meta)
                if section_heading:
                    sub_meta["parent_section"] = section_heading

                paragraphs = section.split("\n\n")
                current_chunk = []
                current_length = 0

                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    if current_length + len(para) > self.chunk_size and current_chunk:
                        chunk_content = "\n\n".join(current_chunk)
                        chunks.append(
                            Chunk(
                                id=f"{document.id}_chunk_{chunk_idx}",
                                document_id=document.id,
                                content=chunk_content,
                                chunk_index=chunk_idx,
                                metadata=sub_meta,
                            )
                        )
                        chunk_idx += 1
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(para)
                    current_length += len(para) + 2

                if current_chunk and current_length >= self.min_chunk_size:
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(
                        Chunk(
                            id=f"{document.id}_chunk_{chunk_idx}",
                            document_id=document.id,
                            content=chunk_content,
                            chunk_index=chunk_idx,
                            metadata=sub_meta,
                        )
                    )
                    chunk_idx += 1

        return chunks


__all__ = ["Chunker"]
