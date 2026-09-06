package com.learning.rag.model;

/** A raw source document before chunking. */
public class Document {
    private final String id;
    private final String source;
    private final String text;

    public Document(String id, String source, String text) {
        this.id = id;
        this.source = source;
        this.text = text;
    }

    public String getId() { return id; }
    public String getSource() { return source; }
    public String getText() { return text; }
}
