package com.learning.rag.embedding;

import java.util.Locale;

/**
 * A dependency-free, API-key-free "embedding" using the classic feature
 * hashing trick (bag-of-words hashed into a fixed-size vector, TF-weighted,
 * L2-normalized).
 *
 * This is NOT a real semantic embedding — two sentences with the same words
 * in different orders look identical, and there's no notion of synonyms.
 */
public class HashingEmbeddingClient implements EmbeddingClient {

    private final int dimension;

    public HashingEmbeddingClient(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public float[] embed(String text) {
        float[] vector = new float[dimension];
        String[] tokens = text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+");

        for (String token : tokens) {
            if (token.isEmpty()) continue;
            int bucket = Math.floorMod(token.hashCode(), dimension);
            vector[bucket] += 1.0f; // term frequency
        }

        normalize(vector);
        return vector;
    }

    private void normalize(float[] v) {
        double sumSquares = 0.0;
        for (float x : v) sumSquares += x * x;
        double norm = Math.sqrt(sumSquares);
        if (norm == 0.0) return;
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (v[i] / norm);
        }
    }

    @Override
    public int dimension() {
        return dimension;
    }
}
