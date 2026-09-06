package com.learning.rag.vectorstore;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.ScoredChunk;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Simplest possible vector store: keep everything in a list, compute cosine
 * similarity against every entry on every query, sort, take top-K.
 *
 * Tradeoffs:
 *  + Zero setup, zero dependencies, trivial to understand and debug.
 *  + Fine for learning / small corpora (thousands of chunks).
 *  - O(n) per query — does not scale to large corpora.
 *  - No persistence: everything is lost when the JVM exits.
 */
public class InMemoryVectorStore implements VectorStore {

    private static class Entry {
        final Chunk chunk;
        final float[] vector;
        Entry(Chunk chunk, float[] vector) { this.chunk = chunk; this.vector = vector; }
    }

    private final List<Entry> entries = new ArrayList<>();

    @Override
    public void add(Chunk chunk, float[] vector) {
        entries.add(new Entry(chunk, vector));
    }

    @Override
    public List<ScoredChunk> search(float[] queryVector, int topK) {
        List<ScoredChunk> scored = new ArrayList<>();
        for (Entry entry : entries) {
            float score = cosineSimilarity(queryVector, entry.vector);
            scored.add(new ScoredChunk(entry.chunk, score));
        }
        scored.sort(Comparator.comparingDouble(ScoredChunk::getScore).reversed());
        return scored.subList(0, Math.min(topK, scored.size()));
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) return 0f;
        return (float) (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    public int size() {
        return entries.size();
    }
}
