# Modular HTML RAG

Python 3.11, LangChain components, OpenAI embeddings/chat, and persistent Chroma.
The virtual environment is already installed on this machine.

## Run

```powershell
cd D:\Coding\html-rag
# Offline HTML processing:
.\.venv\Scripts\python.exe rag.py --inspect
# Offline behavior and integration tests (local test embeddings, no OpenAI calls):
.\.venv\Scripts\python.exe -m pytest -q
# Copy .env.example to .env and set OPENAI_API_KEY locally, then:
.\.venv\Scripts\python.exe rag.py data/sample.html
```

For a fresh installation: create a Python virtual environment and run
`python -m pip install -r requirements.txt`. `.env` is ignored by Git.
The application uses the OpenAI API; configure an API key and model access.
Live embedding and chat calls may incur usage charges.

## Modules and current status

- `processing.py`: cleanup, heading-aware LangChain splitting, recursive splitting,
  table extraction, image alt/caption extraction.
- `vision.py`: optional local image descriptions via the chat model.
- `index.py`: content/config-specific Chroma collections; repeated indexing skips
  existing chunk IDs. Changed input creates a new collection; old collections
  are retained and require eventual lifecycle management.
- `retrieval.py`: in-memory BM25, dense Chroma retrieval, weighted reciprocal rank
  fusion, and an optional MMR variant over the fused candidates.
- `conversation.py`: session-scoped conversational query rewriting and grounded
  answering with source labels. Recent six exchanges are held in process memory.
- `evaluation.py`: live evaluation cases, per-case evidence/fact/citation heuristics,
  source traces, latency, model and retrieval configuration.
- `cli.py`: command-line orchestration.

SQLite ingestion state, a BFS crawler, durable chat memory, token-budget packing,
reranking models, and automatic semantic faithfulness grading are NOT implemented
yet. SQLite is intended for page ingestion tracking; Chroma remains the vector DB.
The conversation code currently uses history APIs deprecated by the installed
LangChain version; migrate to a supported persistence approach before extending
durable memory. The current tests pass but report these deprecation warnings.

## Hybrid retrieval

The history-aware question rewrite runs BEFORE either retrieval branch.

```
Standalone question
  +-- BM25 over chunk text, title, and headings ----+
  +-- Chroma semantic similarity ------------------+
                                                   |
                                 Weighted rank fusion (RRF)
                                                   |
                                 Optional diversity selection
                                                   |
                                      Top k chunks -> answer
```

BM25 uses a positive Robertson IDF formulation, term frequency saturation, and
document length normalization. Identifiers such as ERR-401 are preserved alongside
their component tokens. Chunks with no lexical matches are excluded. This is a
small-corpus implementation, not a distributed lexical search service.

RRF merges by chunk ID and adds `weight / (60 + rank)` for each branch. Raw BM25
and vector-distance scores are not added together because their scales differ.
All candidates must belong to the same active corpus. The current single-file
application enforces that by using its content-specific Chroma collection and
the exact same chunks for BM25. Multi-page ingestion will need an active manifest.

The MMR variant uses normalized RRF relevance and stored embedding cosine
redundancy. It runs AFTER fusion so exact lexical hits can participate. This is
not vanilla Chroma query-vector MMR. The first hit is the best fused result;
subsequent choices balance fused relevance against similarity to selected chunks.
`mmr_lambda=1` preserves fusion order; lower values favor more diversity.
Embedding vectors are read from Chroma, not regenerated or stored in SQLite.

Defaults are experimental starting points, not measured optimal settings:

```powershell
.\.venv\Scripts\python.exe rag.py --retrieval hybrid-mmr --candidates 20 --k 4 --bm25-weight 0.5 --mmr-lambda 0.8
```

Modes: `semantic`, `bm25`, `hybrid`, `hybrid-mmr`. Each candidate count is per branch.
BM25 is rebuilt from the loaded chunks at startup. Returned document metadata
includes branch ranks, retrieval mode, and ranking score. Scores are not calibrated
probabilities and do not establish that an answer exists in the corpus.

Do not automatically enable every advanced technique. Compare hybrid without MMR
against hybrid with MMR, then consider a cross-encoder reranker, parent-section
expansion, or multi-query retrieval if evaluated failures justify them. Diversity
can remove useful adjacent table rows; parent expansion can overflow context;
query expansion can lose identifiers; rerankers add latency and dependencies.

## Tables and images

Table headers repeat with each row group, and rows remain intact. Rowspan/colspan
cells are expanded. A single wide row may exceed the prose chunk-size limit.
Nested tables are rejected; multi-row hierarchical headers need a richer parser.
Whole-table aggregation is not guaranteed by retrieving only a few row groups.

Images index alt text/captions by default. `--vision` adds image descriptions and
transcribed visible text using a vision-capable configured model. Only local images
inside the HTML directory are supported. Remote image URLs are rejected in vision
mode. Vision enrichment runs on ingestion each startup; it is not cached separately.
Descriptions are text embeddings, not native image embeddings. Exact chart values
and illegible text require verification. Do not use `--vision` with offline `--inspect`.

## Evaluation

After setting the API key, run the same cases for each retrieval mode:

```powershell
.\.venv\Scripts\python.exe -m html_rag.evaluation --retrieval semantic --output evals/semantic.json
.\.venv\Scripts\python.exe -m html_rag.evaluation --retrieval bm25 --output evals/bm25.json
.\.venv\Scripts\python.exe -m html_rag.evaluation --retrieval hybrid --output evals/hybrid.json
.\.venv\Scripts\python.exe -m html_rag.evaluation --retrieval hybrid-mmr --output evals/hybrid-mmr.json
```

The bundled five-case JSONL dataset covers prose, tables, follow-ups, and unknown
answers. It is a starter set, not a benchmark. Add exact identifiers, paraphrases,
cross-page questions, image questions, injected instructions, and duplicate chunks.
Use fixed corpus/model settings and repeat live runs to account for model variance.

Reports measure expected evidence substring coverage at k, required answer strings,
citation-number validity/presence, abstention phrases, and end-to-end latency.
These heuristics do not measure full semantic correctness, citation support, or
faithfulness. Human-reviewed relevance labels are needed for proper Recall@k,
MRR, and nDCG; independent claim review or a calibrated judge is needed for
grounding quality. Token usage and cost instrumentation are future work.

## References

- https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
- https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- https://developers.openai.com/api/docs/guides/images-vision
