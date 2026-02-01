package rag

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
 * Integration tests for RAG query functionality.
 * Creates a temporary Lucene index for testing.
 */
class RagQueryTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private val tempIndexDir: java.nio.file.Path = Files.createTempDirectory("test-lucene-index")

  override def beforeAll(): Unit = {
    // Create a small test index
    val analyzer = new StandardAnalyzer()
    val config = new IndexWriterConfig(analyzer)
    val writer = new IndexWriter(FSDirectory.open(tempIndexDir), config)

    try {
      // Add test documents
      val testDocs = Seq(
        ("doc1.pdf", 0, "Machine learning is a subset of artificial intelligence", Array.fill(1024)(0.1f)),
        ("doc1.pdf", 1, "Deep learning uses neural networks", Array.fill(1024)(0.2f)),
        ("doc2.pdf", 0, "Natural language processing analyzes text", Array.fill(1024)(0.3f))
      )

      testDocs.foreach { case (docId, chunkId, text, vec) =>
        val doc = new Document()
        doc.add(new StringField("doc_id", docId, Field.Store.YES))
        doc.add(new StringField("chunk_id", chunkId.toString, Field.Store.YES))
        doc.add(new TextField("text", text, Field.Store.YES))
        doc.add(new KnnFloatVectorField("vec", vec, VectorSimilarityFunction.COSINE))
        writer.addDocument(doc)
      }

      writer.commit()
    } finally {
      writer.close()
    }
  }

  override def afterAll(): Unit = {
    // Clean up temporary directory
    Files.walk(tempIndexDir)
      .sorted(java.util.Comparator.reverseOrder())
      .forEach(Files.delete)
  }

  "RagQuery.searchShard" should "return search results" in {
    val queryVec = Array.fill(1024)(0.15f)  // Query vector
    val results = RagQuery.searchShard(tempIndexDir, queryVec, k = 3)

    results should not be empty
    results.size should be <= 3
    results.foreach { r =>
      r.docId should not be empty
      r.text should not be empty
    }
  }

  it should "return results sorted by score" in {
    val queryVec = Array.fill(1024)(0.15f)
    val results = RagQuery.searchShard(tempIndexDir, queryVec, k = 3)

    if (results.size > 1) {
      // Scores should be in descending order
      results.sliding(2).foreach { case Seq(a, b) =>
        a.score should be >= b.score
      }
    }
  }

  it should "respect the k parameter" in {
    val queryVec = Array.fill(1024)(0.15f)
    val results1 = RagQuery.searchShard(tempIndexDir, queryVec, k = 1)
    val results2 = RagQuery.searchShard(tempIndexDir, queryVec, k = 2)

    results1.size shouldBe 1
    results2.size shouldBe 2
  }

  it should "return stored fields correctly" in {
    val queryVec = Array.fill(1024)(0.15f)
    val results = RagQuery.searchShard(tempIndexDir, queryVec, k = 1)

    results.head.docId should include("doc")
    results.head.chunkId should be >= 0
    results.head.text should include regex "(?i)(machine|deep|natural)"
  }
}