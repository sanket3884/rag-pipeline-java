# rag-pipeline-java

A from-scratch, hands-on RAG (Retrieval-Augmented Generation) / LLM pipeline
built in plain Java — no LangChain-style framework in the way. The point is
to see and touch every stage of a RAG system yourself: chunking, embeddings,
vector indexing, retrieval, and prompt assembly for generation.

Every stage is an interface with at least one dependency-free implementation,
so you can run the whole thing today with **zero API keys and zero external
services**, then swap in real components one at a time and see what changes.

## Architecture

```
Document
   │  TextChunker (word-based sliding window, with overlap)
   ▼
Chunk[]
   │  EmbeddingClient.embed(text) -> float[]
   ▼
(Chunk, vector) pairs
   │  VectorStore.add(chunk, vector)
   ▼
Index  ──────────────────────────────────────────────┐
                                                       │
Question ─ EmbeddingClient.embed(question) ─ VectorStore.search(vector, topK)
                                                       │
                                          top-K (Chunk, score) pairs
                                                       │
                            assemble prompt: question + retrieved context
                                                       │
                                        LlmClient.generate(system, prompt)
                                                       ▼
                                                    Answer
```

`RagPipeline` (`pipeline/RagPipeline.java`) wires these four interfaces
together and exposes just `ingest(documents)` and `query(question, topK)`.

## Component choices and tradeoffs

This is the part worth understanding, not just running.

### Vector store (`vectorstore/`)

| | InMemoryVectorStore | LuceneVectorStore | (not built) external DB — Qdrant/Milvus/pgvector |
|---|---|---|---|
| Setup | None | None (embedded) | Separate service (often Docker) |
| Search | Brute-force cosine, O(n) | HNSW approximate nearest-neighbor | HNSW/IVF, usually approximate |
| Persistence | None — lost on exit | Disk-backed | Disk-backed, replicated |
| Scale | Fine to ~thousands of chunks | Scales much further, single-node | Multi-node, production scale |
| Extra features | None | None | Metadata filtering, hybrid search, multi-tenancy |

Start with `InMemoryVectorStore` to see cosine similarity with your own eyes
(read `InMemoryVectorStoreTest` — it's a good place to start reading the
codebase). Move to `LuceneVectorStore` to see what a *real* ANN index looks
like — this is the same HNSW mechanism Elasticsearch/OpenSearch use
internally for vector search, just without the cluster around it.

### Embeddings (`embedding/`)

| | HashingEmbeddingClient | OpenAiEmbeddingClient |
|---|---|---|
| Setup | None | `OPENAI_API_KEY` |
| Quality | Weak — bag-of-words, hashed into a fixed vector, no notion of meaning or synonyms | Real semantic embeddings |
| Cost | Free, offline | API cost per call |

The hashing embedder exists so you can run the *entire* pipeline today and
watch it retrieve chunks based on literal word overlap. Then flip
`EMBEDDING_MODE=openai` and ask the same questions — you should see
noticeably better retrieval on questions that paraphrase the source text
instead of quoting it.

### LLM (`llm/`)

| | MockLlmClient | AnthropicLlmClient |
|---|---|---|
| Setup | None | `ANTHROPIC_API_KEY` |
| What it does | Echoes the assembled prompt back, unmodified | Real generation via the Messages API |

Use the mock first to confirm retrieval + prompt assembly are correct before
spending any API budget on generation.

### Chunking (`chunking/TextChunker.java`)

Currently a fixed-size, word-count sliding window with overlap. It knows
nothing about sentences, paragraphs, or semantic boundaries — it will happily
cut a sentence in half. This is deliberately the simplest thing that works;
see "Learning path" below for how to improve it.

## Setup

- JDK 17+
- Maven 3.8+

```bash
mvn compile
```

## Running

Zero-setup mode (hashing embeddings, in-memory store, mock LLM):

```bash
mvn compile exec:java
```

It ingests the three sample docs in `data/sample_docs/`, then drops you into
a prompt where you can ask questions like:

```
> what is HNSW?
> how does java manage memory?
> what are the four stages of a RAG pipeline?
```

Type `exit` to quit.

### Using real components

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

EMBEDDING_MODE=openai VECTOR_STORE_MODE=lucene LLM_MODE=anthropic mvn compile exec:java
```

Mix and match — e.g. keep hashing embeddings but switch to Lucene, or keep
in-memory but switch to a real LLM. See `Main.java` for the wiring.

### Building a standalone jar

```bash
mvn package
java -jar target/rag-pipeline.jar
```

### Running tests

```bash
mvn test
```

## Project layout

```
src/main/java/com/learning/rag/
  model/        Document, Chunk, ScoredChunk — plain data classes
  chunking/     TextChunker
  embedding/    EmbeddingClient interface + Hashing / OpenAI implementations
  vectorstore/  VectorStore interface + InMemory / Lucene implementations
  llm/          LlmClient interface + Mock / Anthropic implementations
  pipeline/     RagPipeline — wires the above together
  Main.java     CLI demo
src/test/java/...  unit tests for the chunker and in-memory vector store
data/sample_docs/  three short .txt files used as the demo corpus
```

## Learning path / exercises

Roughly in order of difficulty:

1. **Read `InMemoryVectorStoreTest`** and hand-verify one cosine similarity
   calculation on paper. Make sure the intuition ("same direction = high
   score, regardless of magnitude") actually clicks.
2. **Break the chunker on purpose.** Set `CHUNK_SIZE_WORDS` very small (e.g.
   15) in `Main.java` and see retrieval quality degrade as chunks lose
   context. Then try it very large and see how noisy the retrieved context
   gets.
3. **Compare embedders on the same question.** Run once with
   `EMBEDDING_MODE=hashing` and once with `EMBEDDING_MODE=openai`, same
   question, and diff the retrieved chunks.
4. **Add a metadata filter.** Extend `VectorStore.search` to accept a source
   filter (e.g. only search `rag_concepts.txt`), and implement it in both
   `InMemoryVectorStore` and `LuceneVectorStore`. Notice how much more
   natural this is in Lucene (a `BooleanQuery` combining the KNN query with a
   term filter) vs. in-memory (just an `if` check) — this is the kind of
   thing that gets much harder again once you're multi-node.
5. **Add a re-ranker.** After retrieving top-20 with a cheap method, re-score
   the top-20 with a more expensive/accurate method and re-sort before
   taking the final top-3. This is a very common real-world RAG pattern.
6. **Add evaluation.** Write a small set of (question, expected source)
   pairs and compute retrieval precision@k — did the right document actually
   show up in the top-k?
7. **Replace the chunker.** Try sentence-boundary-aware chunking (don't split
   mid-sentence) or semantic chunking (split where the topic shifts), and
   see if it improves the eval numbers from step 6.
8. **Graduate to a real vector DB.** Stand up Qdrant or Milvus in Docker,
   write a `VectorStore` implementation that talks to it over HTTP/gRPC, and
   compare operational complexity against `LuceneVectorStore`.

## Pushing to GitHub

This repo is initialized locally with an initial commit. To push it:

```bash
git remote add origin <your-empty-github-repo-url>
git branch -M main
git push -u origin main
```
