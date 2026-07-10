"""Document parsers — text/html/markdown/csv/json (pure) + office/pdf (real libs)."""

from __future__ import annotations

import io

import pytest

from src.models import DocumentType
from src.parser import (
    PARSERS,
    CSVParser,
    DocxParser,
    HTMLParser,
    JSONParser,
    MarkdownParser,
    PDFParser,
    PPTXParser,
    TextParser,
    XLSXParser,
)


async def test_text_parser_decodes_bytes_and_str():
    assert await TextParser().parse(b"hello") == "hello"
    assert await TextParser().parse("plain") == "plain"


async def test_html_parser_strips_tags_scripts_styles():
    html = (
        "<html><head><style>.x{}</style><script>evil()</script></head>"
        "<body><h1>Head</h1><p>Para&nbsp;one</p><br><li>item</li></body></html>"
    )
    out = await HTMLParser().parse(html.encode())
    assert "evil()" not in out
    assert ".x{}" not in out
    assert "<p>" not in out
    assert "Head" in out
    assert "Para one" in out


async def test_markdown_parser_passthrough():
    assert await MarkdownParser().parse("# Title") == "# Title"
    assert await MarkdownParser().parse(b"# Bytes") == "# Bytes"


async def test_csv_parser_flattens_rows_to_pairs():
    csv_data = "name,age\nAlice,30\nBob,25"
    out = await CSVParser().parse(csv_data)
    assert "Columns: name, age" in out
    assert "name: Alice | age: 30" in out
    assert "name: Bob | age: 25" in out


async def test_csv_parser_empty_returns_blank():
    assert await CSVParser().parse("") == ""


async def test_json_parser_flattens_nested_paths():
    out = await JSONParser().parse('{"a": {"b": 1}, "c": [10, 20]}')
    assert "a.b: 1" in out
    assert "c[0]: 10" in out
    assert "c[1]: 20" in out


async def test_json_parser_invalid_falls_back_to_text():
    out = await JSONParser().parse("not json {")
    assert out == "not json {"


async def test_pdf_parser_reads_real_pdf():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF world")
    data = doc.tobytes()
    doc.close()
    out = await PDFParser().parse(data)
    assert "Hello PDF world" in out
    assert "[Page 1]" in out


async def test_docx_parser_reads_real_docx():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("First paragraph")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "k"
    table.rows[0].cells[1].text = "v"
    buf = io.BytesIO()
    d.save(buf)
    out = await DocxParser().parse(buf.getvalue())
    assert "First paragraph" in out
    assert "k | v" in out


async def test_xlsx_parser_reads_real_workbook():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["col1", "col2"])
    ws.append(["x", "y"])
    buf = io.BytesIO()
    wb.save(buf)
    out = await XLSXParser().parse(buf.getvalue())
    assert "[Sheet:" in out
    assert "col1: x | col2: y" in out


async def test_pptx_parser_reads_real_deck():
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Slide Title"
    buf = io.BytesIO()
    prs.save(buf)
    out = await PPTXParser().parse(buf.getvalue())
    assert "[Slide 1]" in out
    assert "Slide Title" in out


def test_parser_registry_covers_all_types():
    for dt in [
        DocumentType.TEXT,
        DocumentType.HTML,
        DocumentType.CSV,
        DocumentType.JSON,
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.XLSX,
        DocumentType.PPTX,
        DocumentType.MARKDOWN,
    ]:
        assert dt in PARSERS
