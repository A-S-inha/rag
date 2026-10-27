import hashlib
import re
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from html_rag.processing import process_html
from html_rag.index import open_index
from html_rag.conversation import build_conversation
from html_rag.evaluation import score
from html_rag.vision import make_image_describer


class TestEmbeddings(Embeddings):
    """Deterministic local hashing for wiring tests, not retrieval quality evals."""
    __test__ = False

    def embed_query(self, text):
        values = [0.0] * 256
        for word in re.findall(r"\w+", text.lower()):
            values[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 256] += 1
        return values

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def html_file(tmp_path, body):
    path = tmp_path / "test.html"
    path.write_text(f"<html><body><main>{body}</main></body></html>", encoding="utf-8")
    return path


def test_cleaning_and_prose_chunks(tmp_path):
    path = html_file(tmp_path, "<h1>Guide</h1><h2>Policy</h2><script>secret_code</script><nav>menu</nav><p>" + "Important policy. " * 100 + "</p>")
    docs = process_html(path, chunk_size=150, chunk_overlap=20)
    assert len(docs) > 2
    assert all(len(d.page_content) <= 150 for d in docs)
    assert all(d.metadata["h2"] == "Policy" for d in docs)
    assert all("secret_code" not in d.page_content and "menu" not in d.page_content for d in docs)


def test_table_rows_preserve_headers_and_merged_cells(tmp_path):
    path = html_file(tmp_path, '<h2>Plans</h2><table><tr><th>Plan</th><th>Price</th></tr><tr><td rowspan="2">Pro</td><td>$30</td></tr><tr><td>$40</td></tr></table>')
    docs = [d for d in process_html(path, 70, 10) if d.metadata["kind"] == "table"]
    assert len(docs) == 2
    assert all("Plan = Pro" in d.page_content for d in docs)
    assert all("Headers: Plan | Price" in d.page_content for d in docs)
    assert "Price = $40" in docs[1].page_content
    assert docs[1].metadata["row_start"] == 2


def test_image_alt_caption_and_optional_description(tmp_path):
    path = html_file(tmp_path, '<h2>Architecture</h2><figure><img src="diagram.png" alt="RAG flow"><figcaption>Retrieval precedes answering</figcaption></figure>')
    docs = process_html(path, image_describer=lambda path, src: "Two boxes: search and answer")
    image = next(d for d in docs if d.metadata["kind"] == "image")
    assert "RAG flow" in image.page_content
    assert "Retrieval precedes answering" in image.page_content
    assert "Two boxes" in image.page_content
    assert image.metadata["vision_processed"]
    assert image.metadata["image_src"] == "diagram.png"


def test_vision_rejects_remote_and_outside_paths(tmp_path):
    describe = make_image_describer(None)
    for src in ("https://example.com/image.png", "../outside.png"):
        with pytest.raises(ValueError):
            describe(tmp_path / "page.html", src)


def test_empty_and_invalid_chunking(tmp_path):
    path = html_file(tmp_path, "<script>no</script>")
    with pytest.raises(ValueError):
        process_html(path)
    with pytest.raises(ValueError):
        process_html(path, 10, 10)


def test_chroma_persistence_idempotence_and_changed_source(tmp_path):
    path = html_file(tmp_path, "<h1>Guide</h1><p>Bananas are yellow.</p>")
    embeddings = TestEmbeddings()
    store, docs = open_index(path, embeddings, tmp_path / "db", "test-hash")
    assert "Bananas" in store.similarity_search("bananas", k=1)[0].page_content
    reopened, _ = open_index(path, embeddings, tmp_path / "db", "test-hash")
    assert len(reopened.get()["ids"]) == len(docs)
    path.write_text("<p>Apples are red.</p>", encoding="utf-8")
    changed, _ = open_index(path, embeddings, tmp_path / "db", "test-hash")
    assert all("Bananas" not in text for text in changed.get()["documents"])


def test_history_rewrites_before_retrieval_and_isolates_sessions():
    queries, prompts = [], []

    def model(prompt):
        messages = prompt.to_messages()
        prompts.append(messages)
        return "annual leave carryover" if "Rewrite" in messages[0].content else "You can carry 5 days [1]."

    def retrieve(query):
        queries.append(query)
        return [Document(page_content="Carry 5 days", metadata={"source": "guide"})]

    chain, histories = build_conversation(RunnableLambda(retrieve), RunnableLambda(model), max_turns=1)
    config = {"configurable": {"session_id": "a"}}
    chain.invoke({"question": "Annual leave?"}, config=config)
    result = chain.invoke({"question": "Can I carry it over?"}, config=config)
    assert queries == ["Annual leave?", "annual leave carryover"]
    assert result["search_query"] == "annual leave carryover"
    assert any(m.content == "Annual leave?" for m in prompts[1])
    chain.invoke({"question": "Equipment?"}, config={"configurable": {"session_id": "b"}})
    assert queries[-1] == "Equipment?"
    assert len(histories["b"].messages) == 2


def test_no_context_abstains_without_answer_call():
    def model(_):
        raise AssertionError("No model call expected")
    chain, _ = build_conversation(RunnableLambda(lambda _: []), RunnableLambda(model))
    result = chain.invoke({"question": "Anything?"}, config={"configurable": {"session_id": "a"}})
    assert "do not know" in result["answer"]


def test_evaluation_does_not_confuse_valid_citations_with_correct_facts():
    metrics = score({"evidence": ["24 days"], "answer_contains": ["24"]},
                    {"sources": [Document(page_content="24 days")], "answer": "99 days [8]"})
    assert metrics["evidence_coverage_at_k"] == 1
    assert not metrics["required_fact_match"]
    assert not metrics["citation_ids_valid"]
