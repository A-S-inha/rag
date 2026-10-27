import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .processing import process_html
from .index import open_index
from .conversation import build_conversation
from .vision import make_image_describer
from .retrieval import HybridRetriever, add_retrieval_arguments, config_from_args

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="?", type=Path, default=ROOT / "data/sample.html")
    parser.add_argument("--inspect", action="store_true", help="Print processed chunks without API calls")
    parser.add_argument("--vision", action="store_true", help="Send local images to the chat model for text enrichment")
    parser.add_argument("--db", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--k", type=int, default=4)
    add_retrieval_arguments(parser)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    try:
        retrieval_config = config_from_args(args)
        if args.inspect and args.vision:
            parser.error("--inspect is offline; use --vision only for live ingestion")
        describer = None
        if args.vision:
            if not os.getenv("OPENAI_API_KEY"):
                parser.error("--vision requires OPENAI_API_KEY")
            describer = make_image_describer(ChatOpenAI(model=args.model, timeout=60, max_retries=2))
        chunks = process_html(args.html, args.chunk_size, args.overlap, describer)
        if args.inspect:
            for doc in chunks:
                print(doc.metadata, "\n", doc.page_content, "\n")
            print(f"{len(chunks)} chunks")
            return
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY in .env or the environment; see .env.example")
        embeddings = OpenAIEmbeddings(model=args.embedding_model, request_timeout=60, max_retries=2)
        store, chunks = open_index(args.html, embeddings, args.db, args.embedding_model,
                                   args.chunk_size, args.overlap, chunks=chunks)
        llm = ChatOpenAI(model=args.model, timeout=60, max_retries=2)
        retriever = HybridRetriever(store, chunks, retrieval_config).as_runnable()
        chain, histories = build_conversation(retriever, llm)
        print(f"Ready: {len(chunks)} chunks. /exit to quit; /clear to reset chat memory.")
        while True:
            question = input("\nYou: ").strip()
            if question == "/exit":
                break
            if question == "/clear":
                histories.clear()
                print("Chat memory cleared.")
                continue
            if not question:
                continue
            result = chain.invoke({"question": question},
                                  config={"configurable": {"session_id": "local"}})
            print("\nAnswer:", result["answer"])
            print("Search query:", result["search_query"])
            for index, doc in enumerate(result["sources"], 1):
                print(f"[{index}] {doc.metadata['source']} | "
                      f"{doc.metadata.get('h2', doc.metadata.get('h1', ''))} | "
                      f"chunk {doc.metadata['chunk_index']}")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
    except Exception as exc:
        # Avoid logging exception bodies that may contain request/document contents.
        parser.exit(1, f"Error ({type(exc).__name__}). Check HTML, configuration, API access, and database path.\n")


if __name__ == "__main__":
    main()
