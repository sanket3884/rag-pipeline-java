package com.learning.rag.vectorstore;

import com.learning.rag.model.Chunk;
import com.learning.rag.model.ScoredChunk;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A real, persistent vector store backed by Lucene's HNSW approximate
 * nearest-neighbor index (org.apache.lucene.document.KnnFloatVectorField).
 *
 * This is the same underlying mechanism used inside Elasticsearch/OpenSearch
 * for vector search. Compared to InMemoryVectorStore:
 *  + Persists to disk — survives restarts.
 *  + HNSW gives sub-linear approximate search — scales to much larger corpora.
 *  + Runs embedded in the same JVM, no extra service to operate.
 *  - Approximate (not exact) nearest-neighbor — a tiny amount of recall is traded for speed.
 *  - Still single-node; a dedicated vector DB (Qdrant/Milvus/pgvector) is the
 *    next step if you need multi-node scale, hybrid search, or rich metadata filters.
 */
public class LuceneVectorStore implements VectorStore {

    private static final String FIELD_VECTOR = "vector";
    private static final String FIELD_TEXT = "text";
    private static final String FIELD_SOURCE = "source";
    private static final String FIELD_CHUNK_ID = "chunkId";
    private static final String FIELD_DOC_ID = "docId";
    private static final String FIELD_POSITION = "position";

    private final Directory directory;
    private final IndexWriter writer;
    private final int dimension;

    public LuceneVectorStore(Path indexPath, int dimension) throws IOException {
        this.directory = FSDirectory.open(indexPath);
        this.dimension = dimension;
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        this.writer = new IndexWriter(directory, config);
    }

    @Override
    public void add(Chunk chunk, float[] vector) throws IOException {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: expected " + dimension + " but got " + vector.length);
        }
        Document doc = new Document();
        doc.add(new KnnFloatVectorField(FIELD_VECTOR, vector, VectorSimilarityFunction.COSINE));
        doc.add(new StoredField(FIELD_TEXT, chunk.getText()));
        doc.add(new StoredField(FIELD_SOURCE, chunk.getSource()));
        doc.add(new StoredField(FIELD_CHUNK_ID, chunk.getId()));
        doc.add(new StoredField(FIELD_DOC_ID, chunk.getDocId()));
        doc.add(new StoredField(FIELD_POSITION, chunk.getPosition()));
        writer.addDocument(doc);
    }

    @Override
    public List<ScoredChunk> search(float[] queryVector, int topK) throws IOException {
        try (DirectoryReader reader = DirectoryReader.open(writer)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_VECTOR, queryVector, topK);
            TopDocs topDocs = searcher.search(query, topK);

            StoredFields storedFields = reader.storedFields();
            List<ScoredChunk> results = new ArrayList<>();
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = storedFields.document(scoreDoc.doc);
                Chunk chunk = new Chunk(
                        doc.get(FIELD_CHUNK_ID),
                        doc.get(FIELD_DOC_ID),
                        doc.get(FIELD_SOURCE),
                        doc.get(FIELD_TEXT),
                        Integer.parseInt(doc.get(FIELD_POSITION))
                );
                results.add(new ScoredChunk(chunk, scoreDoc.score));
            }
            return results;
        }
    }

    public void commit() throws IOException {
        writer.commit();
    }

    @Override
    public void close() throws IOException {
        writer.commit();
        writer.close();
        directory.close();
    }
}
