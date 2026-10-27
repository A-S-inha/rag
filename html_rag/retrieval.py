"""BM25 + Chroma semantic search, rank fusion, and optional fused-result MMR.

All branches use the same corpus. This small-corpus BM25 implementation uses
the standard positive Robertson IDF variant; no external search service is needed.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def tokenize(text):
    # Preserve complete identifiers AND their parts: ERR-401, foo_bar, v1.2.
    tokens = []
    for token in re.findall(r"\w+(?:[-.]\w+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-._]", token):
            tokens.extend(part for part in re.split(r"[-._]+", token) if part)
    return tokens


def key(doc):
    return str(doc.metadata["chunk_id"])


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(" ".join([
            str(doc.metadata.get("title", "")),
            *(str(doc.metadata.get(h, "")) for h in ("h1", "h2", "h3")),
            doc.page_content,
        ]))) for doc in self.documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / max(len(self.lengths), 1) or 1
        self.df = Counter(term for count in self.counts for term in count)

    def search(self, query, k):
        terms = set(tokenize(query))
        scores = []
        for doc, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.df[term]
                idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((doc, score))
        return sorted(scores, key=lambda item: (-item[1], key(item[0])))[:k]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid-mmr"
    k: int = 4
    candidates: int = 20
    bm25_weight: float = 0.5
    rrf_constant: int = 60
    mmr_lambda: float = 0.8

    def __post_init__(self):
        if self.mode not in {"semantic", "bm25", "hybrid", "hybrid-mmr"}:
            raise ValueError("Unknown retrieval mode")
        if self.k < 1 or self.candidates < self.k or self.rrf_constant < 1:
            raise ValueError("Require candidates >= k >= 1 and positive RRF constant")
        if not 0 <= self.bm25_weight <= 1 or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("Weights must be between zero and one")


def fuse(rankings, weights, constant=60):
    scores, docs = {}, {}
    for ranking, weight in zip(rankings, weights):
        if weight <= 0:
            continue
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            identifier = key(doc)
            if identifier in seen:
                continue
            seen.add(identifier)
            docs[identifier] = doc
            scores[identifier] = scores.get(identifier, 0) + weight / (constant + rank)
    return [(docs[identifier], scores[identifier]) for identifier in
            sorted(scores, key=lambda identifier: (-scores[identifier], identifier))]


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ")
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


def diversify(ranked, vectors, k, relevance_weight):
    """MMR variant: normalized fusion relevance minus cosine redundancy.

    Unlike vanilla query-vector MMR, this preserves BM25's relevance contribution.
    Select the best fused hit first, then trade relevance against redundancy.
    """
    if not ranked:
        return []
    if relevance_weight == 1:
        return ranked[:k]
    highest = ranked[0][1]
    remaining = list(ranked)
    selected = [remaining.pop(0)]
    while remaining and len(selected) < k:
        def objective(item):
            doc, relevance = item
            redundancy = max(max(0.0, cosine(vectors[key(doc)], vectors[key(other)]))
                             for other, _ in selected)
            return relevance_weight * relevance / highest - (1 - relevance_weight) * redundancy
        best = max(range(len(remaining)), key=lambda i: objective(remaining[i]))
        selected.append(remaining.pop(best))
    return selected


class HybridRetriever:
    def __init__(self, store, documents, config=None):
        self.store = store
        self.config = config or RetrievalConfig()
        self.documents = list(documents)
        ids = [key(doc) for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Corpus chunk IDs must be unique")
        self.lexical = BM25Index(self.documents)

    def search(self, query):
        if not query.strip():
            raise ValueError("Search query must not be blank")
        cfg = self.config
        lexical_pairs = self.lexical.search(query, cfg.candidates) if cfg.mode != "semantic" else []
        dense = self.store.similarity_search(query, k=cfg.candidates) if cfg.mode != "bm25" else []
        if cfg.mode == "bm25":
            ranked = lexical_pairs
        elif cfg.mode == "semantic":
            ranked = [(doc, 1 / (cfg.rrf_constant + rank)) for rank, doc in enumerate(dense, 1)]
        else:
            ranked = fuse([[doc for doc, _ in lexical_pairs], dense],
                          [cfg.bm25_weight, 1 - cfg.bm25_weight], cfg.rrf_constant)
        if cfg.mode == "hybrid-mmr" and ranked and cfg.mmr_lambda < 1:
            records = self.store.get(ids=[key(doc) for doc, _ in ranked], include=["embeddings"])
            vectors = dict(zip(records["ids"], records["embeddings"]))
            missing = [key(doc) for doc, _ in ranked if key(doc) not in vectors]
            if missing:
                raise ValueError("Lexical and vector corpora are out of sync")
            selected = diversify(ranked, vectors, cfg.k, cfg.mmr_lambda)
        else:
            selected = ranked[:cfg.k]
        lexical_ranks = {key(doc): i for i, (doc, _) in enumerate(lexical_pairs, 1)}
        dense_ranks = {key(doc): i for i, doc in enumerate(dense, 1)}
        results = []
        for doc, relevance in selected:
            # Copy rather than mutate the indexed corpus on each query.
            metadata = {**doc.metadata, "retrieval_mode": cfg.mode, "retrieval_score": relevance}
            if key(doc) in lexical_ranks:
                metadata["bm25_rank"] = lexical_ranks[key(doc)]
            if key(doc) in dense_ranks:
                metadata["semantic_rank"] = dense_ranks[key(doc)]
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results

    def as_runnable(self):
        return RunnableLambda(self.search)


def add_retrieval_arguments(parser):
    parser.add_argument("--retrieval", choices=["semantic", "bm25", "hybrid", "hybrid-mmr"], default="hybrid-mmr")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--mmr-lambda", type=float, default=0.8)


def config_from_args(args):
    return RetrievalConfig(mode=args.retrieval, k=args.k, candidates=args.candidates,
                           bm25_weight=args.bm25_weight, mmr_lambda=args.mmr_lambda)
