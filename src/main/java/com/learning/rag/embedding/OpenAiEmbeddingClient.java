package com.learning.rag.embedding;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Real semantic embeddings via OpenAI's embeddings API.
 * Requires the OPENAI_API_KEY environment variable.
 *
 */
public class OpenAiEmbeddingClient implements EmbeddingClient {

    private static final String ENDPOINT = "https://api.openai.com/v1/embeddings";
    private final String apiKey;
    private final String model;
    private final int dimension;
    private final HttpClient http = HttpClient.newHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();

    public OpenAiEmbeddingClient(String apiKey, String model, int dimension) {
        this.apiKey = apiKey;
        this.model = model;
        this.dimension = dimension;
    }

    @Override
    public float[] embed(String text) {
        try {
            String body = mapper.writeValueAsString(new EmbeddingRequest(model, text));
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(ENDPOINT))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Embedding API error " + response.statusCode() + ": " + response.body());
            }

            JsonNode root = mapper.readTree(response.body());
            JsonNode embeddingNode = root.get("data").get(0).get("embedding");
            float[] vector = new float[embeddingNode.size()];
            for (int i = 0; i < embeddingNode.size(); i++) {
                vector[i] = (float) embeddingNode.get(i).asDouble();
            }
            return vector;
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Failed to call embedding API", e);
        }
    }

    @Override
    public int dimension() {
        return dimension;
    }

    // Minimal request DTO for Jackson serialization
    private record EmbeddingRequest(String model, String input) {}
}
