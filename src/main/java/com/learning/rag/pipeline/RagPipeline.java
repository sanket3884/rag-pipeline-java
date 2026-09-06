package com.learning.rag.pipeline;

import com.learning.rag.chunking.TextChunker;
import com.learning.rag.embedding.EmbeddingClient;
import com.learning.rag.llm.LlmClient;
import com.learning.rag.model.Chunk;
import com.learning.rag.model.Document;
import com.learning.rag.model.ScoredChunk;
import com.learning.rag.vectorstore.VectorStore;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Orchestrates the full RAG flow:
 *
 *   ingest:  Document -> chunk -> embed -> store in VectorStore
 *   query:   question -> embed -> retrieve top-K from VectorStore
 *                      -> assemble prompt (question + retrieved context)
 *                      -> LlmClient.generate(...)
 *
 * Every stage is an injected interface, so you can swap chunker, embedder,
 * vector store, or LLM independently to see how each choice affects the
 * final answer.
 */
public class RagPipeline {

    private final TextChunker chunker;
    private final EmbeddingClient embeddingClient;
    private final VectorStore vectorStore;
    private final LlmClient llmClient;

    public RagPipeline(TextChunker chunker, EmbeddingClient embeddingClient,
                        VectorStore vectorStore, LlmClient llmClient) {
        this.chunker = chunker;
        this.embeddingClient = embeddingClient;
        this.vectorStore = vectorStore;
        this.llmClient = llmClient;
    }

    public int ingest(List<Document> documents) throws IOException {
        int chunkCount = 0;
        for (Document document : documents) {
            List<Chunk> chunks = chunker.chunk(document);
            for (Chunk chunk : chunks) {
                float[] vector = embeddingClient.embed(chunk.getText());
                vectorStore.add(chunk, vector);
                chunkCount++;
            }
        }
        return chunkCount;
    }

    public RagResult query(String question, int topK) throws IOException {
        float[] queryVector = embeddingClient.embed(question);
        List<ScoredChunk> retrieved = vectorStore.search(queryVector, topK);

        String context = retrieved.stream()
                .map(sc -> "Source: " + sc.getChunk().getSource() + "\n" + sc.getChunk().getText())
                .collect(Collectors.joining("\n\n---\n\n"));

        String systemPrompt = "You are a helpful assistant. Answer the user's question using ONLY the "
                + "provided context. If the context does not contain the answer, say you don't know. "
                + "Cite which source(s) you used.";

        String userPrompt = "Context:\n" + context + "\n\nQuestion: " + question;

        String answer = llmClient.generate(systemPrompt, userPrompt);
        return new RagResult(question, retrieved, answer);
    }

    /** Result of a query: the retrieved chunks plus the generated answer. */
    public record RagResult(String question, List<ScoredChunk> retrievedChunks, String answer) {}
}
