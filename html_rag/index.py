import hashlib
import json
from pathlib import Path
from langchain_chroma import Chroma
from .processing import process_html, PIPELINE_VERSION

def collection_name(path, embedding_model, chunk_size=900, chunk_overlap=150):
    """Content/config addressing prevents stale results or mixing embedding spaces."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(f"{path}|{embedding_model}|{chunk_size}|{chunk_overlap}|{PIPELINE_VERSION}".encode())
    return "html_" + digest.hexdigest()[:24]


def open_index(path, embeddings, persist_directory, embedding_model,
               chunk_size=900, chunk_overlap=150, chunks=None):
    chunks = process_html(path, chunk_size, chunk_overlap) if chunks is None else chunks
    for index, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = f"chunk-{index}"
    content_hash = hashlib.sha256(json.dumps([
        {"text": d.page_content, "metadata": d.metadata} for d in chunks
    ], sort_keys=True).encode()).hexdigest()[:12]
    store = Chroma(
        collection_name=collection_name(path, embedding_model, chunk_size, chunk_overlap) + "_" + content_hash,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    present = set(store.get(ids=ids)["ids"])
    missing = [(key, doc) for key, doc in zip(ids, chunks) if key not in present]
    if missing:
        store.add_documents([doc for _, doc in missing], ids=[key for key, _ in missing])
    return store, chunks


