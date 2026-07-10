"""Chunking strategies — fixed, sentence, paragraph, recursive."""

from __future__ import annotations

from src.chunker import Chunker
from src.models import ChunkingStrategy, Document


def _doc(content: str, **kw) -> Document:
    return Document(
        id="d1",
        tenant_id="t",
        kb_id="kb",
        content=content,
        title="Title",
        source_url="http://x",
        metadata={"lang": "en"},
        **kw,
    )


def test_fixed_size_chunking_respects_size_and_overlap():
    c = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=10, chunk_overlap=2, min_chunk_size=1)
    chunks = c.chunk(_doc("abcdefghijklmnopqrstuvwxyz"))
    assert len(chunks) >= 2
    assert all(len(ch.content) <= 10 for ch in chunks)
    # chunk ids + metadata propagated
    assert chunks[0].id == "d1_chunk_0"
    assert chunks[0].metadata["title"] == "Title"
    assert chunks[0].metadata["lang"] == "en"
    assert chunks[0].start_char == 0


def test_fixed_size_drops_tiny_tail_below_min():
    c = Chunker(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=10, chunk_overlap=0, min_chunk_size=5)
    chunks = c.chunk(_doc("abcdefghijkl"))  # 12 chars -> 10 + 2(too small)
    assert [ch.content for ch in chunks] == ["abcdefghij"]


def test_sentence_chunking_splits_on_punctuation():
    text = "One sentence here. Two sentence there! Three now? Four last."
    c = Chunker(strategy=ChunkingStrategy.SENTENCE, chunk_size=25, chunk_overlap=0, min_chunk_size=1)
    chunks = c.chunk(_doc(text))
    assert len(chunks) >= 2
    joined = " ".join(ch.content for ch in chunks)
    assert "One sentence here." in joined


def test_paragraph_chunking_groups_paragraphs():
    text = "Para one is long enough.\n\nPara two also here.\n\nPara three tail."
    c = Chunker(strategy=ChunkingStrategy.PARAGRAPH, chunk_size=30, chunk_overlap=0, min_chunk_size=1)
    chunks = c.chunk(_doc(text))
    assert chunks
    assert any("Para one" in ch.content for ch in chunks)


def test_recursive_extracts_section_heading_metadata():
    text = "# Intro\nHello there this is intro.\n\n## Details\nMore detail content here."
    c = Chunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=200, chunk_overlap=0, min_chunk_size=1)
    chunks = c.chunk(_doc(text))
    headings = {ch.metadata.get("section_heading") for ch in chunks}
    assert "Intro" in headings
    assert "Details" in headings
    assert any(ch.metadata.get("section_level") == 1 for ch in chunks)


def test_recursive_splits_large_section_into_subchunks_with_parent():
    big = "# Big\n" + "\n\n".join(f"Paragraph number {i} with text." for i in range(20))
    c = Chunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=60, chunk_overlap=0, min_chunk_size=1)
    chunks = c.chunk(_doc(big))
    assert len(chunks) >= 2
    assert any(ch.metadata.get("parent_section") == "Big" for ch in chunks)


def test_unknown_strategy_falls_back_to_fixed():
    c = Chunker(strategy=ChunkingStrategy.SEMANTIC, chunk_size=8, chunk_overlap=0, min_chunk_size=1)
    chunks = c.chunk(_doc("abcdefghijkl"))
    assert chunks  # semantic isn't implemented -> fixed fallback
    assert chunks[0].content == "abcdefgh"
