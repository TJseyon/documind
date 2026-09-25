from pathlib import Path

import pytest
from pydantic import ValidationError

from app.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.ingestion.extractors import extract_text
from app.models import QueryRequest


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="  ")


def test_query_request_rejects_too_long_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="a" * 2001)


def test_query_request_accepts_valid_question():
    req = QueryRequest(question="What is the vacation policy?")
    assert req.question == "What is the vacation policy?"


def test_query_request_rejects_out_of_range_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(question="valid question here", top_k=100)


def test_extract_text_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "malware.exe"
    bad_file.write_bytes(b"binary junk")
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(bad_file, supported_extensions=[".pdf", ".docx", ".txt", ".md"])


def test_extract_text_rejects_empty_document(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        extract_text(empty_file, supported_extensions=[".txt"])


def test_extract_text_handles_plain_txt_happy_path(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("This is a perfectly normal document with real content.")
    result = extract_text(f, supported_extensions=[".txt"])
    assert "perfectly normal document" in result.text
