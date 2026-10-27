import pytest
from langchain_core.documents import Document
from html_rag.retrieval import BM25Index, RetrievalConfig, HybridRetriever, diversify, fuse, tokenize
from html_rag.index import open_index
from test_rag import TestEmbeddings, html_file


def doc(identifier, text):
    return Document(page_content=text, metadata={"chunk_id": identifier})


def test_bm25_preserves_identifiers_and_ignores_zero_overlap():
    corpus = [doc("a", "Resolve ERR-401 with credential renewal"),
              doc("b", "Resolve ERR-403 with permission changes"),
              doc("c", "Annual leave policy")]
    index = BM25Index(corpus)
    assert index.search("ERR-401", 3)[0][0].metadata["chunk_id"] == "a"
    assert index.search("nonexistent", 3) == []
    assert {"err-401", "err", "401"}.issubset(tokenize("ERR-401"))


def test_rrf_rewards_agreement_and_deduplicates_ids():
    a, b, c = doc("a", "a"), doc("b", "b"), doc("c", "c")
    result = fuse([[a, b, b], [c, b]], [0.5, 0.5])
    assert result[0][0].metadata["chunk_id"] == "b"
    assert len(result) == 3
    assert result[0][1] == pytest.approx(1 / 62)


def test_mmr_can_replace_a_duplicate_with_complementary_evidence():
    ranked = [(doc("a", "Policy"), 1.0), (doc("b", "Same policy"), .99),
              (doc("c", "Different evidence"), .9)]
    vectors = {"a": [1., 0.], "b": [1., 0.], "c": [0., 1.]}
    diverse = diversify(ranked, vectors, 2, .7)
    assert [d.metadata["chunk_id"] for d, _ in diverse] == ["a", "c"]
    pure_relevance = diversify(ranked, vectors, 2, 1)
    assert [d.metadata["chunk_id"] for d, _ in pure_relevance] == ["a", "b"]


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid", "hybrid-mmr"])
def test_retrieval_modes_with_real_chroma_and_local_embeddings(tmp_path, mode):
    path = html_file(tmp_path, "<h1>Codes</h1><h2>Authentication</h2><p>ERR-401 requires credential renewal.</p><h2>Authorization</h2><p>ERR-403 requires permission changes.</p>")
    store, chunks = open_index(path, TestEmbeddings(), tmp_path / "db", "test-hash")
    retriever = HybridRetriever(store, chunks, RetrievalConfig(mode=mode, k=1, candidates=4))
    result = retriever.as_runnable().invoke("ERR-401 credential")
    assert len(result) == 1
    assert "ERR-401" in result[0].page_content
    assert result[0].metadata["retrieval_mode"] == mode
    assert "retrieval_mode" not in chunks[0].metadata


def test_bm25_mode_never_calls_vector_search():
    class NoVectorCalls:
        def similarity_search(self, *args, **kwargs):
            raise AssertionError("Vector branch should not run")
    retriever = HybridRetriever(NoVectorCalls(), [doc("a", "SpecificIdentifier")],
                                RetrievalConfig(mode="bm25", k=1, candidates=1))
    assert retriever.search("SpecificIdentifier")
    assert retriever.search("missing") == []


@pytest.mark.parametrize("options", [{"k": 0}, {"k": 10, "candidates": 2},
                                     {"mmr_lambda": 1.1}, {"bm25_weight": -1}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        RetrievalConfig(**options)
