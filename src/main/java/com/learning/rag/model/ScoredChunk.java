package com.learning.rag.model;

/** A retrieved chunk together with its similarity score. */
public class ScoredChunk {
    private final Chunk chunk;
    private final float score;

    public ScoredChunk(Chunk chunk, float score) {
        this.chunk = chunk;
        this.score = score;
    }

    public Chunk getChunk() { return chunk; }
    public float getScore() { return score; }

    @Override
    public String toString() {
        return String.format("[score=%.4f] %s", score, chunk.getText().replace("\n", " "));
    }
}
