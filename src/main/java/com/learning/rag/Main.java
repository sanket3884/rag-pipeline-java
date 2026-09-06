package com.learning.rag;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

import com.learning.rag.chunking.TextChunker;
import com.learning.rag.embedding.EmbeddingClient;
import com.learning.rag.embedding.HashingEmbeddingClient;
import com.learning.rag.embedding.OpenAiEmbeddingClient;
import com.learning.rag.llm.AnthropicLlmClient;
import com.learning.rag.llm.LlmClient;
import com.learning.rag.llm.MockLlmClient;
import com.learning.rag.model.Document;
import com.learning.rag.pipeline.RagPipeline;
import com.learning.rag.vectorstore.InMemoryVectorStore;
import com.learning.rag.vectorstore.LuceneVectorStore;
import com.learning.rag.vectorstore.VectorStore;

/**
 * Runnable demo of the whole pipeline.
 *
 * Zero-setup mode (default): HashingEmbeddingClient + InMemoryVectorStore + MockLlmClient.
 *
 */
public class Main {

    // ---- Flip these to swap components -------------------------------------------------
    private static final String EMBEDDING_MODE = System.getenv().getOrDefault("EMBEDDING_MODE", "hashing"); // hashing | openai
    private static final String VECTOR_STORE_MODE = System.getenv().getOrDefault("VECTOR_STORE_MODE", "memory"); // memory | lucene
    private static final String LLM_MODE = System.getenv().getOrDefault("LLM_MODE", "mock"); // mock | anthropic
    // --------------------------------------------------------------------------------------

    private static final int CHUNK_SIZE_WORDS = 120;
    private static final int CHUNK_OVERLAP_WORDS = 20;
    private static final int TOP_K = 3;

    public static void main(String[] args) throws IOException {
        System.out.println("=== RAG Pipeline (Java) — learning demo ===");
        System.out.println("embedding=" + EMBEDDING_MODE + " vectorStore=" + VECTOR_STORE_MODE + " llm=" + LLM_MODE);
        System.out.println();

        EmbeddingClient embeddingClient = buildEmbeddingClient();
        VectorStore vectorStore = buildVectorStore(embeddingClient.dimension());
        LlmClient llmClient = buildLlmClient();
        TextChunker chunker = new TextChunker(CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS);

        RagPipeline pipeline = new RagPipeline(chunker, embeddingClient, vectorStore, llmClient);

        List<Document> documents = loadSampleDocuments();
        int chunkCount = pipeline.ingest(documents);
        System.out.printf("Ingested %d documents into %d chunks.%n%n", documents.size(), chunkCount);

        System.out.println("Type a question (or 'exit' to quit):");
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n> ");
                if (!scanner.hasNextLine()) break;
                String question = scanner.nextLine().trim();
                if (question.isEmpty()) continue;
                if (question.equalsIgnoreCase("exit")) break;

                RagPipeline.RagResult result = pipeline.query(question, TOP_K);

                System.out.println("\n--- Retrieved chunks ---");
                result.retrievedChunks().forEach(System.out::println);

                System.out.println("\n--- Answer ---");
                System.out.println(result.answer());
            }
        } finally {
            vectorStore.close();
        }
    }

    private static EmbeddingClient buildEmbeddingClient() {
        if ("openai".equalsIgnoreCase(EMBEDDING_MODE)) {
            String apiKey = requireEnv("OPENAI_API_KEY");
            return new OpenAiEmbeddingClient(apiKey, "text-embedding-3-small", 1536);
        }
        return new HashingEmbeddingClient(256);
    }

    private static VectorStore buildVectorStore(int dimension) throws IOException {
        if ("lucene".equalsIgnoreCase(VECTOR_STORE_MODE)) {
            Path indexPath = Path.of("lucene-index");
            Files.createDirectories(indexPath);
            return new LuceneVectorStore(indexPath, dimension);
        }
        return new InMemoryVectorStore();
    }

    private static LlmClient buildLlmClient() {
        if ("anthropic".equalsIgnoreCase(LLM_MODE)) {
            String apiKey = requireEnv("ANTHROPIC_API_KEY");
            // See the product's current model list in Anthropic's docs; adjust as needed.
            return new AnthropicLlmClient(apiKey, "claude-sonnet-5", 1024);
        }
        return new MockLlmClient();
    }

    private static String requireEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Missing required environment variable: " + name);
        }
        return value;
    }

    private static List<Document> loadSampleDocuments() throws IOException {
        Path dir = Path.of("data/sample_docs");
        try (Stream<Path> files = Files.list(dir)) {
            return files
                    .filter(p -> p.toString().endsWith(".txt"))
                    .sorted()
                    .map(Main::readDocument)
                    .toList();
        }
    }

    private static Document readDocument(Path path) {
        try {
            String text = Files.readString(path);
            String id = path.getFileName().toString();
            return new Document(id, id, text);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read " + path, e);
        }
    }
}
