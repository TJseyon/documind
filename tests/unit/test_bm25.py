from app.retrieval.bm25_index import BM25Index


def test_exact_keyword_match_ranks_highest():
    index = BM25Index(persist_path=None)
    index.add_documents(
        ids=["a", "b", "c"],
        texts=[
            "The SR-40 drone has a maximum payload of 2.5 kilograms.",
            "Our return policy allows refunds within 30 days of purchase.",
            "Employees receive 20 vacation days per calendar year.",
        ],
    )
    results = index.search("payload kilograms drone", top_k=3)
    assert results[0].chunk_id == "a"


def test_search_on_empty_index_returns_empty_list():
    index = BM25Index(persist_path=None)
    assert index.search("anything", top_k=5) == []


def test_delete_removes_document_from_results():
    index = BM25Index(persist_path=None)
    index.add_documents(ids=["a", "b"], texts=["unique alpha term here", "unique beta term here"])
    index.delete_by_ids(["a"])
    results = index.search("alpha", top_k=5)
    assert all(r.chunk_id != "a" for r in results)


def test_persists_and_reloads_from_disk(tmp_path):
    # Three documents (not two) so the term's IDF is unambiguously positive --
    # with N=2 and the term in exactly 1 of them, BM25's IDF works out to
    # exactly log(1) = 0, which the search filters out as "not a match" even
    # though it technically appears. That's a real property of BM25 on tiny
    # corpora, not a bug, so the test corpus avoids that boundary case.
    path = tmp_path / "corpus.jsonl"
    index = BM25Index(persist_path=str(path))
    index.add_documents(
        ids=["a", "b", "c"],
        texts=[
            "persisted content about rockets",
            "unrelated content about gardening",
            "another unrelated document about cooking",
        ],
    )

    reloaded = BM25Index(persist_path=str(path))
    results = reloaded.search("rockets", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "a"
