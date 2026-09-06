package com.learning.rag.vectorstore;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.ScoredChunk;

import java.io.IOException;
import java.util.List;

/**
 * Anything that can index (chunk, vector) pairs and retrieve the top-K most
 * similar chunks for a query vector.
 *
 * Implementations trade off simplicity, persistence, and scalability:
 *  - InMemoryVectorStore: brute-force cosine similarity, no persistence, O(n) search.
 *  - LuceneVectorStore: embedded HNSW approximate-nearest-neighbor index, persistent to disk.
 *  - (not implemented here, but the natural next step) an external vector DB
 *    like Qdrant/Milvus/pgvector for multi-node scale and rich metadata filtering.
 */
public interface VectorStore {

    void add(Chunk chunk, float[] vector) throws IOException;

    List<ScoredChunk> search(float[] queryVector, int topK) throws IOException;

    default void close() throws IOException {}
}
