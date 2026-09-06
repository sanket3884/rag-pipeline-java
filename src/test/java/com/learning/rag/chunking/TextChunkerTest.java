package com.learning.rag.chunking;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.Document;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TextChunkerTest {

    @Test
    void producesOverlappingChunks() {
        // 25 words, chunk size 10, overlap 3 -> step 7
        String text = String.join(" ", java.util.Collections.nCopies(25, "word"));
        Document doc = new Document("doc1", "test-source", text);

        TextChunker chunker = new TextChunker(10, 3);
        List<Chunk> chunks = chunker.chunk(doc);

        assertFalse(chunks.isEmpty());
        // every chunk except the last should have exactly chunkSize words
        for (int i = 0; i < chunks.size() - 1; i++) {
            String[] words = chunks.get(i).getText().split("\\s+");
            assertEquals(10, words.length);
        }
        // positions should be sequential starting at 0
        for (int i = 0; i < chunks.size(); i++) {
            assertEquals(i, chunks.get(i).getPosition());
        }
    }

    @Test
    void singleChunkWhenTextShorterThanChunkSize() {
        Document doc = new Document("doc2", "test-source", "just a few words here");
        TextChunker chunker = new TextChunker(50, 10);
        List<Chunk> chunks = chunker.chunk(doc);

        assertEquals(1, chunks.size());
        assertEquals("just a few words here", chunks.get(0).getText());
    }

    @Test
    void rejectsOverlapLargerThanChunkSize() {
        assertThrows(IllegalArgumentException.class, () -> new TextChunker(10, 10));
    }
}
