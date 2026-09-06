package com.learning.rag.chunking;

import java.util.ArrayList;
import java.util.List;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.Document;

/**
 * Splits documents into overlapping word-based chunks.
 *
 * This is the simplest possible chunking strategy: fixed word count with
 * overlap between consecutive chunks so that context isn't lost at chunk
 * boundaries. It ignores sentence/paragraph structure entirely.
 *
 */
public class TextChunker {

    private final int chunkSizeWords;
    private final int overlapWords;

    public TextChunker(int chunkSizeWords, int overlapWords) {
        if (overlapWords >= chunkSizeWords) {
            throw new IllegalArgumentException("overlapWords must be smaller than chunkSizeWords");
        }
        this.chunkSizeWords = chunkSizeWords;
        this.overlapWords = overlapWords;
    }

    public List<Chunk> chunk(Document document) {
        List<Chunk> chunks = new ArrayList<>();
        String[] words = document.getText().trim().split("\\s+");
        if (words.length == 0 || (words.length == 1 && words[0].isEmpty())) {
            return chunks;
        }

        int step = chunkSizeWords - overlapWords;
        int position = 0;
        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(start + chunkSizeWords, words.length);
            String text = String.join(" ", java.util.Arrays.copyOfRange(words, start, end));
            String chunkId = document.getId() + "-chunk-" + position;
            chunks.add(new Chunk(chunkId, document.getId(), document.getSource(), text, position));
            position++;
            if (end == words.length) break;
        }
        return chunks;
    }
}
