package rag.search

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import java.nio.file.{Files, Paths}
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.{IndexWriter, IndexWriterConfig, DirectoryReader}
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document._
import org.apache.lucene.index.VectorSimilarityFunction

/**
 * Integration tests for vector search functionality.
 */
class VectorSearchEngineTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private val tempIndexDir: java.nio.file.Path = Files.createTempDirectory("test-vector-index")

  override def beforeAll(): Unit = {
    val analyzer = new StandardAnalyzer()
    val config = new IndexWriterConfig(analyzer)
    val writer = new IndexWriter(FSDirectory.open(tempIndexDir), config)

    try {
      val testDocuments = Seq(
        ("doc1.pdf", 0, "Machine learning is a subset of artificial intelligence", Array.fill(1024)(0.1f)),
        ("doc1.pdf", 1, "Deep learning uses neural networks", Array.fill(1024)(0.2f)),
        ("doc2.pdf", 0, "Natural language processing analyzes text", Array.fill(1024)(0.3f))
      )

      testDocuments.foreach { case (docKey, segIdx, content, vec) =>
        val doc = new Document()
        doc.add(new StringField("document_key", docKey, Field.Store.YES))
        doc.add(new StringField("segment_index", segIdx.toString, Field.Store.YES))
        doc.add(new TextField("content", content, Field.Store.YES))
        doc.add(new KnnFloatVectorField("vector", vec, VectorSimilarityFunction.COSINE))
        writer.addDocument(doc)
      }

      writer.commit()
    } finally {
      writer.close()
    }
  }

  override def afterAll(): Unit = {
    Files.walk(tempIndexDir)
      .sorted(java.util.Comparator.reverseOrder())
      .forEach(Files.delete)
  }

  "VectorSearchEngine.searchPartition" should "return query results" in {
    val queryVector = Array.fill(1024)(0.15f)
    val results = VectorSearchEngine.searchPartition(tempIndexDir, queryVector, limit = 3)

    results should not be empty
    results.size should be <= 3
    results.foreach { r =>
      r.documentKey should not be empty
      r.content should not be empty
    }
  }

  it should "sort results by relevance score" in {
    val queryVector = Array.fill(1024)(0.15f)
    val results = VectorSearchEngine.searchPartition(tempIndexDir, queryVector, limit = 3)

    if (results.size > 1) {
      results.sliding(2).foreach { case Seq(a, b) =>
        a.relevanceScore should be >= b.relevanceScore
      }
    }
  }

  it should "respect the limit parameter" in {
    val queryVector = Array.fill(1024)(0.15f)
    val results1 = VectorSearchEngine.searchPartition(tempIndexDir, queryVector, limit = 1)
    val results2 = VectorSearchEngine.searchPartition(tempIndexDir, queryVector, limit = 2)

    results1.size shouldBe 1
    results2.size shouldBe 2
  }

  it should "return stored fields correctly" in {
    val queryVector = Array.fill(1024)(0.15f)
    val results = VectorSearchEngine.searchPartition(tempIndexDir, queryVector, limit = 1)

    results.head.documentKey should include("doc")
    results.head.segmentIndex should be >= 0
    results.head.content should include regex "(?i)(machine|deep|natural)"
  }
}
