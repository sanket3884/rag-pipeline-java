package com.learning.rag.model;

/** A chunk of a document — the unit we embed and index. */
public class Chunk {
    private final String id;
    private final String docId;
    private final String source;
    private final String text;
    private final int position;

    public Chunk(String id, String docId, String source, String text, int position) {
        this.id = id;
        this.docId = docId;
        this.source = source;
        this.text = text;
        this.position = position;
    }

    public String getId() { return id; }
    public String getDocId() { return docId; }
    public String getSource() { return source; }
    public String getText() { return text; }
    public int getPosition() { return position; }
}
