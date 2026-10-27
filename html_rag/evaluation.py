"""Live, local-report RAG evaluations. No judge or tracing service is required."""
import argparse
import json
import os
import re
import time
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from .cli import ROOT
from .index import open_index
from .conversation import build_conversation
from .retrieval import HybridRetriever, add_retrieval_arguments, config_from_args
from dataclasses import asdict


def score(case, result):
    context = "\n".join(d.page_content for d in result["sources"]).casefold()
    evidence = case.get("evidence", [])
    answer = result["answer"].casefold()
    required = case.get("answer_contains", [])
    citations = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    # This is a deterministic string-check baseline, not a semantic quality judge.
    return {
        "evidence_coverage_at_k": sum(x.casefold() in context for x in evidence) / len(evidence) if evidence else None,
        "required_fact_match": all(x.casefold() in answer for x in required) if required else None,
        "citation_ids_valid": all(1 <= x <= len(result["sources"]) for x in citations),
        "has_citation": bool(citations),
        "abstention_phrase_detected": any(x in answer for x in ("do not know", "don't know", "not provided", "does not", "doesn't", "not specified")) if case.get("unanswerable") else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "data/sample.html")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/results.json")
    parser.add_argument("--k", type=int, default=4)
    add_retrieval_arguments(parser)
    args = parser.parse_args()
    retrieval_config = config_from_args(args)
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Live eval requires OPENAI_API_KEY; use pytest for offline checks")
    if args.k < 1:
        parser.error("--k must be positive")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    store, chunks = open_index(args.html, OpenAIEmbeddings(model=embedding_model, request_timeout=60, max_retries=2),
                          ROOT / "chroma_db", embedding_model)
    llm = ChatOpenAI(model=model, timeout=60, max_retries=2)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        parser.error("No evaluation cases found")
    rows = []
    for case in cases:
        # Fresh chain and memory for every independent test case.
        retriever = HybridRetriever(store, chunks, retrieval_config).as_runnable()
        chain, histories = build_conversation(retriever, llm)
        from langchain_core.chat_history import InMemoryChatMessageHistory
        histories["eval"] = InMemoryChatMessageHistory(messages=[
            message for question, answer in case.get("history", [])
            for message in (HumanMessage(content=question), AIMessage(content=answer))
        ])
        start = time.perf_counter()
        result = chain.invoke({"question": case["question"]},
                              config={"configurable": {"session_id": "eval"}})
        rows.append({"id": case["id"], "slice": case.get("slice"),
                     "seconds": time.perf_counter() - start,
                     "answer": result["answer"], "search_query": result["search_query"],
                     "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in result["sources"]],
                     "metrics": score(case, result)})
    report = {"model": model, "embedding_model": embedding_model, "k": args.k,
              "retrieval": asdict(retrieval_config),
              "collection": store._collection.name,
              "notes": "String heuristics only. Citation validity is not citation support or faithfulness.",
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
