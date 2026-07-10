"""Text cleaning + normalization behaviour."""

from __future__ import annotations

from src.cleaner import TextCleaner


def test_clean_normalizes_whitespace_and_newlines():
    raw = "a\t  b\r\nc\r\n\n\n\nd"
    out = TextCleaner.clean(raw)
    assert "\t" not in out
    assert "\r" not in out
    # 3+ newlines collapse to a double newline
    assert "\n\n\n" not in out
    assert out.startswith("a b")


def test_clean_strips_null_bytes_and_control_chars():
    raw = "hello\x00\x07world"
    out = TextCleaner.clean(raw)
    assert "\x00" not in out
    assert "\x07" not in out
    assert out == "helloworld"


def test_clean_decodes_html_entities_and_unicode():
    out = TextCleaner.clean("café &amp; tea")
    assert "&amp;" not in out
    assert "&" in out
    assert "café" in out


def test_clean_preserves_whitespace_when_disabled():
    out = TextCleaner.clean("a     b", config={"normalize_whitespace": False})
    assert "a     b" == out


def test_remove_urls_and_emails_and_numbers():
    assert TextCleaner.remove_urls("see https://x.com/y now").strip() == "see  now".strip()
    assert "@" not in TextCleaner.remove_emails("mail me a@b.com ok")
    assert "[NUMBER]" in TextCleaner.normalize_numbers("id 12345678901 end")
