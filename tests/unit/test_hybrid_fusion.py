from app.retrieval.bm25_index import KeywordMatch
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorMatch


def _vm(chunk_id: str, score: float) -> VectorMatch:
    return VectorMatch(chunk_id=chunk_id, document_id="doc1", text="text", metadata={}, score=score)


def test_document_in_both_lists_outranks_top_of_single_list():
    # "b" is #1 in vector search but absent from BM25.
    # "a" is #2 in vector and #1 in BM25 -- present in both.
    vector_results = [_vm("b", 0.99), _vm("a", 0.80), _vm("c", 0.50)]
    bm25_results = [KeywordMatch("a", 9.0), KeywordMatch("d", 3.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    fused_ids = [f.chunk_id for f in fused]

    assert fused_ids[0] == "a"  # appears in both lists -> wins over single-list top rank


def test_fusion_includes_documents_from_either_list():
    vector_results = [_vm("x", 0.9)]
    bm25_results = [KeywordMatch("y", 5.0)]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    ids = {f.chunk_id for f in fused}
    assert ids == {"x", "y"}


def test_empty_inputs_produce_empty_output():
    assert reciprocal_rank_fusion([], []) == []


def test_rank_metadata_is_tracked_correctly():
    vector_results = [_vm("a", 0.9), _vm("b", 0.8)]
    bm25_results = [KeywordMatch("b", 5.0)]
    fused = {f.chunk_id: f for f in reciprocal_rank_fusion(vector_results, bm25_results)}
    assert fused["a"].vector_rank == 1
    assert fused["a"].bm25_rank is None
    assert fused["b"].vector_rank == 2
    assert fused["b"].bm25_rank == 1
