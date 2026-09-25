"""
Named exceptions for the specific edge cases this system is expected to
handle, rather than letting everything bubble up as a generic 500.
"""


class DocuMindError(Exception):
    """Base class for all application-raised errors."""

    error_code = "internal_error"
    status_code = 500


class UnsupportedFileTypeError(DocuMindError):
    error_code = "unsupported_file_type"
    status_code = 415


class FileTooLargeError(DocuMindError):
    error_code = "file_too_large"
    status_code = 413


class EmptyDocumentError(DocuMindError):
    """Raised when a file extracts to zero (or near-zero) usable text."""

    error_code = "empty_document"
    status_code = 422


class ExtractionError(DocuMindError):
    """Raised when a file appears malformed / cannot be parsed at all."""

    error_code = "extraction_failed"
    status_code = 422


class NoDocumentsIndexedError(DocuMindError):
    error_code = "no_documents_indexed"
    status_code = 409


class DownstreamLLMError(DocuMindError):
    error_code = "llm_call_failed"
    status_code = 502


class EmbeddingError(DocuMindError):
    """Raised when generating embeddings fails -- e.g. the embedding model
    couldn't be downloaded (no internet on first run), ran out of memory,
    or the local model cache is corrupted. Wrapping this separately from a
    generic crash means the person using the app sees a specific, readable
    reason instead of a blank 'something went wrong'."""

    error_code = "embedding_failed"
    status_code = 502


class DocumentNotFoundError(DocuMindError):
    error_code = "document_not_found"
    status_code = 404
