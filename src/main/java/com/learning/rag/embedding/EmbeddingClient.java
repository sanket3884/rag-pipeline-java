package com.learning.rag.embedding;

/** Anything that can turn text into a fixed-length numeric vector. */
public interface EmbeddingClient {

    float[] embed(String text);

    int dimension();
}
