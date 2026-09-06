package com.learning.rag.llm;

/**
 * Zero-setup LLM stand-in: no API key, no network call. It just echoes back
 * the retrieved context so you can verify the retrieval half of the pipeline
 * (chunking -> embedding -> vector search -> prompt assembly) works before
 * spending any API budget on the generation half.
 *
 */
public class MockLlmClient implements LlmClient {

    @Override
    public String generate(String systemPrompt, String userPrompt) {
        return "[MOCK LLM — no real generation happened]\n"
                + "This client just proves the prompt was assembled correctly.\n"
                + "Swap in AnthropicLlmClient (set ANTHROPIC_API_KEY) for a real answer.\n\n"
                + "---- Prompt that would have been sent ----\n"
                + "SYSTEM:\n" + systemPrompt + "\n\n"
                + "USER:\n" + userPrompt;
    }
}
