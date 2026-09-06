package com.learning.rag.llm;

/** Anything that can turn (system prompt, user prompt) into a generated answer. */
public interface LlmClient {

    String generate(String systemPrompt, String userPrompt);
}
