package com.learning.rag.vectorstore;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.ScoredChunk;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class InMemoryVectorStoreTest {

    @Test
    void identicalVectorsScorePerfectSimilarity() {
        float[] v = {1f, 0f, 0f};
        assertEquals(1.0f, InMemoryVectorStore.cosineSimilarity(v, v), 1e-6);
    }

    @Test
    void orthogonalVectorsScoreZero() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(0.0f, InMemoryVectorStore.cosineSimilarity(a, b), 1e-6);
    }

    @Test
    void searchReturnsMostSimilarFirst() {
        InMemoryVectorStore store = new InMemoryVectorStore();
        Chunk close = new Chunk("c1", "d1", "s1", "close match", 0);
        Chunk far = new Chunk("c2", "d1", "s1", "far match", 1);

        store.add(far, new float[]{0f, 1f});
        store.add(close, new float[]{1f, 0.01f});

        List<ScoredChunk> results = store.search(new float[]{1f, 0f}, 2);

        assertEquals("c1", results.get(0).getChunk().getId());
        assertEquals("c2", results.get(1).getChunk().getId());
        assertTrue(results.get(0).getScore() > results.get(1).getScore());
    }

    @Test
    void topKLimitsResultCount() {
        InMemoryVectorStore store = new InMemoryVectorStore();
        for (int i = 0; i < 5; i++) {
            store.add(new Chunk("c" + i, "d1", "s1", "text " + i, i), new float[]{(float) i, 1f});
        }
        List<ScoredChunk> results = store.search(new float[]{2f, 1f}, 2);
        assertEquals(2, results.size());
    }
}
