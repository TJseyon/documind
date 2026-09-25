# DocuMind — Hybrid-Search RAG with a Real Evaluation Harness

A RAG system that answers questions over your own documents (PDF/DOCX/TXT),
using hybrid search (BM25 + vector) and cross-encoder reranking, with
citation-enforced answers and a hallucination check — plus an evaluation
harness that actually measures whether retrieval works, instead of just
demoing that it runs.

## Why this exists

Most RAG portfolio projects stop at "it answers questions." The question
that actually separates a hobby project from engineering is: **how do you
know it's any good?** This project's evaluation harness — a hand-built
Q&A test set, automatic + manual scoring, and a regression test — is the
part that answers that.

## Architecture

```
                    ┌─────────────┐
   PDF/DOCX/TXT ──▶ │  Extractor  │  (edge cases: empty docs, malformed
                    └──────┬──────┘   PDFs, non-English text, scanned images)
                           ▼
                    ┌─────────────┐
                    │   Chunker   │  (recursive char split, paragraph-aware,
                    └──────┬──────┘   with overlap)
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌───────────────┐         ┌──────────────┐
      │ Vector Store   │         │  BM25 Index  │
      │ (Chroma)       │         │ (keyword)    │
      └───────┬────────┘         └──────┬───────┘
              │                          │
              └──────────┬───────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Reciprocal Rank  │  (fuses two ranked lists by
                 │ Fusion (hybrid)  │   position, not raw score)
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │  Cross-Encoder   │  (re-scores top ~10 candidates
                 │  Reranker        │   by reading query+chunk together)
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │  LLM (Claude)    │  (citation-enforcing prompt)
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │ Groundedness /   │  (citation check + unsupported-
                 │ Hallucination    │   number/date check)
                 │ Check            │
                 └────────┬─────────┘
                          ▼
                  Answer + citations
                  + grounded: true/false
```

Every box above is an interface (`VectorStore`, `KeywordIndex`,
`EmbeddingModel`, `Reranker`, `LLMClient`), which is what makes both the
production code path (Chroma, sentence-transformers, Claude) and the test
code path (in-memory store, hash-based fake embedder, canned LLM) share the
same pipeline logic. See `tests/conftest.py`.

## Quickstart

```bash
git clone <this repo>
cd documind
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-ant-...

docker compose up --build
```

First startup downloads the embedding model (~90MB) and reranker model
(~90MB) from Hugging Face — this needs outbound internet access once; after
that they're cached in the `model_cache` volume.

Then, in another terminal:

```bash
# Ingest the sample documents (or drop your own into sample_docs/ first)
python scripts/ingest_sample_docs.py

# Ask a question
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?"}'
```

Or just open `http://localhost:8000/docs` for the interactive Swagger UI.

### Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/ingest` | POST (multipart file) | Extract, chunk, embed, and index a document |
| `/query` | POST `{"question": "...", "top_k": 4}` | Hybrid retrieve → rerank → answer with citations |
| `/documents` | GET | List indexed documents |
| `/documents/{filename}` | DELETE | Remove a document and its chunks |
| `/health` | GET | Doc/chunk counts, for liveness checks |

## Design decisions (and their trade-offs)

**Chunking: recursive character split, not fixed-size slicing.**
Splitting mid-sentence destroys retrieval quality — half a sentence embeds
to something that doesn't resemble the original meaning. The chunker tries
paragraph breaks first, then sentence-ending punctuation, then whitespace,
only hard-cutting as a last resort. See `app/ingestion/chunking.py` for the
full rationale, including why character count (not tokens) is used as the
size unit.

**Hybrid search, not vector-only.** Vector search misses exact terms it
hasn't learned to associate — SKUs, error codes, proper nouns. BM25 misses
paraphrases. Reciprocal Rank Fusion (RRF) combines their two ranked lists
using rank position rather than raw score, because BM25 scores and cosine
similarity live on incomparable scales. **When they disagree:** a chunk
that's moderately-ranked in both lists usually beats a chunk that's #1 in
only one, because RRF sums contributions across lists rather than rewarding
a single strong signal in isolation.

**Reranking after fusion, not instead of it.** RRF is a cheap, rank-only
signal. A cross-encoder reads the query and each candidate's actual text
together and scores true relevance — much more accurate, but too slow to
run over an entire corpus, so it only runs on the ~10 candidates fusion
already narrowed down.

**Chroma (embedded) instead of a Qdrant/pgvector server.** For a solo
project, running a second database process isn't worth the operational
cost. The `VectorStore` interface means swapping to a real server in
production is a new class, not a rewrite.

**BM25Okapi rebuilds its index on every write.** This is a real, named
limitation, not an oversight: `rank_bm25` has no incremental-add API. At
personal-portfolio scale (hundreds to low thousands of chunks) a full
rebuild is milliseconds. At real scale you'd swap in OpenSearch/
Elasticsearch, which do support incremental indexing, behind the same
`KeywordIndex` interface.

**Re-indexing without a full rebuild (the vector side).** Document IDs are
derived deterministically from filename. Re-ingesting a file with the same
name triggers a targeted `delete(where={"document_id": ...})` against Chroma
before the new chunks are added — not a full collection rebuild. The BM25
side still rebuilds (see above); that asymmetry is worth mentioning
proactively in an interview.

## Hallucination handling

Two independent, deterministic checks — neither relies on trusting the
model's own confidence:

1. **Citation requirement.** The system prompt requires every claim to be
   tagged `[chunk_N]`. If the model's answer contains no citation tag at
   all, the whole answer is treated as unsupported, regardless of how
   fluent it sounds.
2. **Numeric/date grounding check.** Every number, percentage, currency
   amount, or date in the answer is checked against the retrieved context.
   Anything not found there is flagged as an unsupported claim. This
   specifically targets the most common and most dangerous hallucination
   pattern — a fabricated statistic that reads as authoritative — but it
   will not catch a fabricated *claim* with no numbers in it. That's why
   it's paired with the citation check, and why the eval harness still has
   a manual "is this actually accurate" review column: no automatic check
   here claims to catch everything.

See `app/generation/grounding.py`.

## Evaluation harness

```bash
python -m eval.run_eval
```

This runs every question in `eval/qa_testset.json` through the live
pipeline, and writes:

- `eval/results/run_<timestamp>.json` — full input/output log (every
  question, retrieved documents, generated answer, groundedness result)
- `eval/results/run_<timestamp>.csv` — the same data as a spreadsheet, with
  a blank `manual_score` column for the judgment calls automation can't
  make ("is this summary actually accurate?")

**`eval/qa_testset.json` ships with only 6 example questions**, built from
the sample docs so the harness runs out of the box. Before this goes on a
resume, replace it with 30–50 questions you write by hand against your own
real documents — that's what makes the retrieval-precision and
groundedness numbers mean anything. If the same pipeline (or an LLM) wrote
the test questions, it's testing whether the system agrees with itself, not
whether it's correct.

Metrics computed automatically (`eval/metrics.py`):

- **retrieval_hit_rate** — did the expected source document actually appear
  among the retrieved chunks for that question? (retrieval precision — the
  thing most RAG demos never measure)
- **contains_answer_rate** — does the generated answer contain the expected
  substring? (a crude but honest automatic correctness proxy)
- **groundedness_rate** — did the hallucination check pass?

## Testing strategy: unit tests vs. eval tests

This is a genuinely useful interview talking point, not just a folder
structure:

- **`tests/unit/`** — deterministic code: chunking boundaries, input
  validation, BM25 ranking, RRF math, the groundedness regex logic, and
  full API routes exercised through fakes (`InMemoryVectorStore`, a
  hash-based fake embedder, a canned `FakeLLMClient`). No network, no API
  key, no downloaded model, runs in well under a second. A failure here
  means "you broke something."
  ```bash
  pytest              # excludes eval tests by default (see pytest.ini)
  ```

- **`tests/eval/`** — one regression test that runs the real pipeline
  (real embeddings, real reranker, real Claude calls) against the Q&A test
  set and asserts the aggregate scores haven't dropped more than a
  tolerance below a checked-in baseline (`eval/baseline_score.json`). This
  costs real API calls and needs `ANTHROPIC_API_KEY` set, so it's excluded
  from the default test run and skips itself automatically without
  credentials.
  ```bash
  pytest -m eval
  ```

## Known limitations (say these out loud, don't wait to be asked)

- No OCR — a scanned, image-only PDF extracts to no text and is rejected
  with a clear `EmptyDocumentError`, not silently ingested as garbage.
- BM25 index rebuild is O(corpus size) per write (see above).
- Single-process, single-node. No auth, no multi-tenancy, no queueing for
  large batch ingestion — all reasonable v2 scope, deliberately out of
  scope for a v1 meant to demonstrate the retrieval/eval/grounding story
  clearly.
- The default embedding model is English-optimized; non-English documents
  are ingested (with a warning), not rejected, but retrieval quality on
  them is unverified.

## Interview prep: likely questions

**"How did you pick your chunk size?"** 1200 characters (~250-300 tokens)
with 200 characters of overlap, chosen empirically by checking that the
example Q&A pairs in the eval set retrieved their source chunk in the top
few results — not a number picked in the abstract. This is exactly the
kind of thing the eval harness gives you an honest answer for, versus
guessing.

**"What happens when hybrid search and vector search disagree?"** See
"Hybrid search" above — RRF favors chunks that show up in both lists, even
moderately, over a chunk that's #1 in only one.

**"How do you know your RAG system is actually good?"** The eval harness:
retrieval-hit rate and groundedness rate measured against a hand-built,
held-out test set, tracked over time via the regression test — not "I ran
it a few times and it looked right."

**"Why didn't you use Elasticsearch/Qdrant/a real vector DB server?"**
Operational cost versus project scope — see "Design decisions" above. The
interfaces are there specifically so that answer doesn't sound like an
excuse; it's a documented, deliberate trade-off with a clear upgrade path.

## Resume bullet

> Built a hybrid-search (BM25 + vector) RAG system with cross-encoder
> reranking, citation-enforced generation, and an automated hallucination
> check; validated with a hand-built 30+ question evaluation harness
> measuring retrieval precision and answer groundedness, with a regression
> test to catch quality drops.

Fill in your own measured numbers once you've run `eval/run_eval.py`
against your real 30-50 question set — don't invent them.

## Project structure

```
app/
  config.py            # settings layer (env-configurable, nothing hardcoded)
  models.py            # request/response schemas + validation
  exceptions.py         # named errors for each handled edge case
  ingestion/
    extractors.py       # PDF/DOCX/TXT extraction + edge-case handling
    chunking.py          # recursive chunking strategy
    pipeline.py          # orchestrates ingest + incremental re-index
  retrieval/
    embeddings.py         # EmbeddingModel interface + sentence-transformers impl
    vector_store.py        # VectorStore interface + Chroma / in-memory impls
    bm25_index.py           # KeywordIndex interface + BM25 impl
    hybrid.py                # reciprocal rank fusion
    reranker.py               # Reranker interface + cross-encoder / no-op impls
  generation/
    prompts.py           # citation-enforcing prompt templates
    llm_client.py          # LLMClient interface + Anthropic / fake impls
    grounding.py             # hallucination / groundedness checks
  db/metadata_store.py    # SQLite: document registry + full query/answer log
  api/routes.py            # FastAPI endpoints
  query_pipeline.py         # retrieve -> rerank -> generate -> ground -> log
eval/
  qa_testset.json          # hand-built Q&A pairs (starter set -- expand this)
  metrics.py                # automatic scoring functions
  run_eval.py                 # runs test set through the live pipeline
  baseline_score.json          # regression floor for tests/eval/
tests/
  unit/                       # fast, deterministic, no network
  eval/                        # live regression test (needs API key)
  conftest.py                    # fake embedder/store/LLM fixtures
sample_docs/                     # generated demo files (see scripts/generate_sample_docs.py)
scripts/
  ingest_sample_docs.py           # POSTs sample_docs/ to a running instance
  generate_sample_docs.py          # regenerates the sample .docx/.pdf files
```
