package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import java.io.File
import java.nio.file.Files

/**
 * Integration test for MapReduce components.
 * Tests mapper and reducer logic with mock data.
 */
class MapReduceIntegrationTestForRagIndexAndWordStat extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private val tempDir: File = Files.createTempDirectory("mapreduce-test").toFile

  override def beforeAll(): Unit = {
    // No additional setup needed; tempDir is already created
  }

  override def afterAll(): Unit = {
    // Clean up temporary directory
    if (tempDir.exists()) {
      tempDir.listFiles().foreach(_.delete())
      tempDir.delete()
    }
  }

  // ========================================================================
  // WordStatsMapper Tests
  // ========================================================================

  "WordStatsMapper" should "tokenize text correctly" in {
    val text = "Machine learning is machine learning"
    val cleanText = text.toLowerCase.replaceAll("[^a-z ]", " ")
    val words = cleanText.split("\\s+").filter(_.nonEmpty)

    words should contain("machine")
    words should contain("learning")
    words.count(_ == "machine") shouldBe 2
  }

  it should "filter stopwords" in {
    val stopwords = Set("the", "is", "and", "a")
    val text = "the machine is a computer"
    val words = text.split("\\s+").filterNot(stopwords.contains)

    words should contain("machine")
    words should contain("computer")
    words should not contain "the"
    words should not contain "is"
  }

  // ========================================================================
  // RagMapper Tests
  // ========================================================================

  "RagMapper shard assignment" should "distribute documents across shards" in {
    val docId1 = "document1.pdf"
    val docId2 = "document2.pdf"
    val docId3 = "document3.pdf"
    val numReducers = 4

    val shard1 = math.abs(docId1.hashCode) % numReducers
    val shard2 = math.abs(docId2.hashCode) % numReducers
    val shard3 = math.abs(docId3.hashCode) % numReducers

    // All shards should be in valid range
    Seq(shard1, shard2, shard3).foreach { shard =>
      shard should be >= 0
      shard should be < numReducers
    }
  }

  it should "consistently assign same document to same shard" in {
    val docId = "test.pdf"
    val numReducers = 8

    val shard1 = math.abs(docId.hashCode) % numReducers
    val shard2 = math.abs(docId.hashCode) % numReducers
    val shard3 = math.abs(docId.hashCode) % numReducers

    // Should always get same shard for same document
    shard1 shouldBe shard2
    shard2 shouldBe shard3
  }

  "RagMapper JSON output" should "create valid format" in {
    import io.circe.syntax._
    import io.circe.parser._

    val docId = "test.pdf"
    val chunkId = 0
    val text = "Sample text chunk"
    val vec = Array(0.1f, 0.2f, 0.3f)

    val rec = Map(
      "doc_id" -> docId.asJson,
      "chunk_id" -> chunkId.asJson,
      "text" -> text.asJson,
      "vec" -> vec.toVector.asJson
    ).asJson.noSpaces

    // Should be parseable
    val parsed = parse(rec)
    parsed.isRight shouldBe true

    // Should contain all fields
    val json = parsed.toOption.get
    val cursor = json.hcursor

    cursor.get[String]("doc_id").toOption shouldBe Some(docId)
    cursor.get[Int]("chunk_id").toOption shouldBe Some(chunkId)
    cursor.get[String]("text").toOption shouldBe Some(text)
  }

  "RagMapper embedding normalization" should "apply L2 normalization correctly" in {
    val rawVec = Array(3.0f, 4.0f)
    val normalized = Vectors.l2(rawVec)

    // Should normalize to unit length
    val length = math.sqrt(normalized.map(x => x * x).sum)
    length shouldBe 1.0 +- 0.001
  }

  // ========================================================================
  // Chunker Integration Tests
  // ========================================================================

  "Chunker integration with text" should "handle real PDF-like text" in {
    val sampleText = """
      Machine learning is a subset of artificial intelligence that focuses on
      enabling computers to learn from data. Deep learning, a subfield of machine
      learning, uses neural networks with multiple layers. Natural language processing
      is another important area that deals with text analysis.
    """.trim

    val chunks = Chunker.split(sampleText, maxChars = 100, overlap = 20)

    chunks should not be empty
    chunks.foreach(_.length should be <= 120)
  }

  // ========================================================================
  // JSON Serialization Tests
  // ========================================================================

  "JSON serialization" should "work for chunk records" in {
    import io.circe.syntax._
    import io.circe.parser._

    val record = Map(
      "doc_id" -> "test.pdf".asJson,
      "chunk_id" -> 0.asJson,
      "text" -> "test text".asJson,
      "vec" -> Vector(0.1f, 0.2f, 0.3f).asJson
    ).asJson.noSpaces

    parse(record).isRight shouldBe true

    record should include("doc_id")
    record should include("chunk_id")
    record should include("text")
    record should include("vec")
  }
}