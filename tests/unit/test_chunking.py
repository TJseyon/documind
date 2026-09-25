from app.ingestion.chunking import chunk_text


def test_short_text_is_a_single_chunk():
    text = "This is a short document. It fits in one chunk easily."
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_text_is_split_into_multiple_chunks_within_size_limit():
    paragraph_template = "Sentence number {}. Sentence number {}. Sentence number {}. "
    text = "\n\n".join(paragraph_template.format(i, i, i) for i in range(40))
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow slight overshoot from overlap prefix, but not runaway growth
        assert len(c.text) <= 200 + 20


def test_overlap_is_present_between_consecutive_chunks():
    paragraph_template = "Unique marker {} appears here. Unique marker {} appears here. "
    text = "\n\n".join(paragraph_template.format(i, i) for i in range(20))
    chunks = chunk_text(text, chunk_size=150, overlap=40)
    assert len(chunks) > 1
    # the tail of chunk N should reappear at the head of chunk N+1
    tail_of_first = chunks[0].text[-40:]
    assert tail_of_first[:15] in chunks[1].text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=500, overlap=50) == []
    assert chunk_text("   \n\n  ", chunk_size=500, overlap=50) == []


def test_tiny_fragments_below_min_length_are_dropped():
    text = "A.\n\nB.\n\n" + ("This is a real sentence with real content. " * 4)
    chunks = chunk_text(text, chunk_size=1000, overlap=0, min_chunk_chars=20)
    for c in chunks:
        assert len(c.text) >= 20


def test_no_separator_falls_back_to_hard_cut():
    text = "x" * 500  # no spaces or punctuation anywhere
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) == 5
    assert all(len(c.text) == 100 for c in chunks)
