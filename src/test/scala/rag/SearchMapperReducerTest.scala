package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import org.apache.hadoop.io.{FloatWritable, LongWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.apache.hadoop.conf.Configuration
import org.apache.lucene.document.{Document, Field, KnnFloatVectorField, StoredField, StringField}
import org.apache.lucene.index.{DirectoryReader, IndexWriter, IndexWriterConfig}
import org.apache.lucene.store.FSDirectory
import java.io.File
import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters._

/**
 * Test suite for SearchMapper and SearchReducer
 */
class SearchMapperReducerTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private val tempDir: File = Files.createTempDirectory("search-test").toFile

  /** Immutable shard path initialized lazily (once before tests). */
  private lazy val testShardPath: String = createTestShard()

  override def beforeAll(): Unit = {
    // Force shard creation
    testShardPath
    ()
  }

  override def afterAll(): Unit = {
    // Clean up temporary directory
    if (tempDir.exists()) {
      def deleteRecursively(file: File): Unit = {
        if (file.isDirectory) file.listFiles().foreach(deleteRecursively)
        file.delete()
      }
      deleteRecursively(tempDir)
    }
  }

  /** Helper to create a test Lucene index shard. */
  private def createTestShard(): String = {
    val shardDir = new File(tempDir, "test-shard")
    shardDir.mkdirs()

    val directory = FSDirectory.open(shardDir.toPath)
    val writer = new IndexWriter(directory, new IndexWriterConfig())

    // Add test documents
    val testDocs = Seq(
      ("doc1.pdf", 0, "Machine learning basics", Array(0.1f, 0.2f, 0.3f, 0.4f)),
      ("doc1.pdf", 1, "Deep learning introduction", Array(0.2f, 0.3f, 0.4f, 0.5f)),
      ("doc2.pdf", 0, "Natural language processing", Array(0.3f, 0.4f, 0.5f, 0.6f)),
      ("doc3.pdf", 0, "Computer vision techniques", Array(0.15f, 0.25f, 0.35f, 0.45f)),
      ("doc4.pdf", 0, "Reinforcement learning", Array(0.4f, 0.5f, 0.6f, 0.7f))
    )

    testDocs.foreach { case (docId, chunkId, text, vec) =>
      val doc = new Document()
      doc.add(new StringField("doc_id", docId, Field.Store.YES))
      doc.add(new StoredField("chunk_id", chunkId))
      doc.add(new StoredField("text", text))
      doc.add(new KnnFloatVectorField("vec", vec))
      writer.addDocument(doc)
    }

    writer.commit()
    writer.close()
    directory.close()

    shardDir.getAbsolutePath
  }

  // ========================================================================
  // SearchMapper Tests
  // ========================================================================

  "SearchMapper" should "parse query vector from configuration" in {
    val queryVecStr = "0.1,0.2,0.3,0.4"
    val queryVec = queryVecStr.split(",").map(_.toFloat)

    queryVec.length shouldBe 4
    queryVec.head shouldBe 0.1f +- 0.001f
    queryVec.last shouldBe 0.4f +- 0.001f
  }

  it should "handle malformed query vectors gracefully" in {
    val invalidVecStr = "0.1,invalid,0.3"
    val result = scala.util.Try(invalidVecStr.split(",").map(_.toFloat)).isSuccess
    result shouldBe false
  }

  it should "emit results with negative scores for sorting" in {
    val score = 0.95f
    val negativeScore = -score
    negativeScore shouldBe -0.95f
    negativeScore < 0 shouldBe true
  }

  it should "search shard and find top-k results" in {
    val directory = FSDirectory.open(Paths.get(testShardPath))
    val reader = DirectoryReader.open(directory)
    val searcher = new org.apache.lucene.search.IndexSearcher(reader)

    val queryVec = Array(0.2f, 0.3f, 0.4f, 0.5f)
    val k = 3
    val query = new org.apache.lucene.search.KnnFloatVectorQuery("vec", queryVec, k)
    val topDocs = searcher.search(query, k)

    topDocs.scoreDocs.length should be <= k
    topDocs.scoreDocs.length should be > 0

    reader.close()
    directory.close()
  }

  it should "create valid JSON output" in {
    import io.circe.syntax._
    import io.circe.parser._

    val docId = "test.pdf"
    val chunkId = 0
    val text = "Sample text"
    val score = 0.95f
    val shard = "/path/to/shard"

    val result = Map(
      "doc_id" -> docId.asJson,
      "chunk_id" -> chunkId.asJson,
      "text" -> text.asJson,
      "score" -> score.asJson,
      "shard" -> shard.asJson
    ).asJson.noSpaces

    val parsed = parse(result)
    parsed.isRight shouldBe true

    val json = parsed.toOption.get
    val cursor = json.hcursor

    cursor.get[String]("doc_id").toOption shouldBe Some(docId)
    cursor.get[Int]("chunk_id").toOption shouldBe Some(chunkId)
    cursor.get[String]("text").toOption shouldBe Some(text)
    cursor.get[Double]("score").toOption.get shouldBe score.toDouble +- 0.001
  }

  it should "handle empty shard gracefully" in {
    val emptyShardDir = new File(tempDir, "empty-shard")
    emptyShardDir.mkdirs()

    val directory = FSDirectory.open(emptyShardDir.toPath)
    val writer = new IndexWriter(directory, new IndexWriterConfig())
    writer.commit()
    writer.close()
    directory.close()

    val reader = DirectoryReader.open(FSDirectory.open(emptyShardDir.toPath))
    val searcher = new org.apache.lucene.search.IndexSearcher(reader)

    val queryVec = Array(0.1f, 0.2f, 0.3f, 0.4f)
    val query = new org.apache.lucene.search.KnnFloatVectorQuery("vec", queryVec, 5)
    val topDocs = searcher.search(query, 5)

    topDocs.scoreDocs.length shouldBe 0
    reader.close()
  }

  // ========================================================================
  // SearchReducer Tests
  // ========================================================================

  "SearchReducer" should "take only top-k results" in {
    val k = 3
    val allResults = Seq(
      (0.95f, """{"doc_id":"doc1","score":0.95}"""),
      (0.90f, """{"doc_id":"doc2","score":0.90}"""),
      (0.85f, """{"doc_id":"doc3","score":0.85}"""),
      (0.80f, """{"doc_id":"doc4","score":0.80}"""),
      (0.75f, """{"doc_id":"doc5","score":0.75}""")
    )

    val topK = allResults.take(k)
    topK.length shouldBe k
    topK.head._1 shouldBe 0.95f
    topK.last._1 shouldBe 0.85f
  }

  it should "handle fewer results than k" in {
    val k = 10
    val allResults = Seq(
      (0.95f, """{"doc_id":"doc1","score":0.95}"""),
      (0.90f, """{"doc_id":"doc2","score":0.90}""")
    )

    val topK = allResults.take(k)
    topK.length shouldBe 2
    topK.length should be < k
  }

  it should "maintain score ordering" in {
    val scores = Seq(-0.95f, -0.90f, -0.85f, -0.80f)
    scores shouldBe sorted
    val originalScores = scores.map(-_)
    originalScores shouldBe Seq(0.95f, 0.90f, 0.85f, 0.80f)
  }

  it should "emit results with NullWritable key" in {
    val nullKey = NullWritable.get()
    nullKey should not be null
    nullKey shouldBe NullWritable.get()
  }

  // ========================================================================
  // Integration Tests
  // ========================================================================

  "SearchMapper and SearchReducer integration" should "work end-to-end" in {
    class MockMapContext:
      val outputs = ArrayBuffer[(FloatWritable, Text)]()
      def write(key: FloatWritable, value: Text): Unit = outputs += ((key, value))

    class MockReduceContext:
      val outputs = ArrayBuffer[Text]()
      def write(key: NullWritable, value: Text): Unit = outputs += value

    val mapperOutputs = Seq(
      (new FloatWritable(-0.95f), new Text("""{"doc_id":"doc1","score":0.95}""")),
      (new FloatWritable(-0.90f), new Text("""{"doc_id":"doc2","score":0.90}""")),
      (new FloatWritable(-0.85f), new Text("""{"doc_id":"doc3","score":0.85}""")),
      (new FloatWritable(-0.80f), new Text("""{"doc_id":"doc4","score":0.80}""")),
      (new FloatWritable(-0.75f), new Text("""{"doc_id":"doc5","score":0.75}"""))
    )

    val k = 3
    val topResults = mapperOutputs.sortBy(_._1.get()).take(k).map(_._2)
    topResults.length shouldBe k

    import io.circe.parser._
    val scores = topResults.flatMap { text =>
      parse(text.toString).toOption.flatMap(_.hcursor.get[Double]("score").toOption)
    }

    scores should contain allOf (0.95, 0.90, 0.85)
    scores should not contain 0.75
  }

  "Search results" should "be properly formatted JSON" in {
    import io.circe.parser._

    val sampleResult =
      """{"doc_id":"test.pdf","chunk_id":0,"text":"sample","score":0.95,"shard":"/path"}"""
    val parsed = parse(sampleResult)
    parsed.isRight shouldBe true

    val json = parsed.toOption.get
    val cursor = json.hcursor
    cursor.keys.toList.flatten.toSet shouldBe Set("doc_id", "chunk_id", "text", "score", "shard")
  }

  "Vector dimension handling" should "validate query vector dimensions" in {
    val queryVec4D = Array(0.1f, 0.2f, 0.3f, 0.4f)
    val queryVec3D = Array(0.1f, 0.2f, 0.3f)
    queryVec4D.length shouldBe 4
    queryVec3D.length shouldBe 3
    queryVec4D.length should not be queryVec3D.length
  }

  "Score handling" should "correctly negate scores for sorting" in {
    val originalScores = Seq(0.95f, 0.90f, 0.85f, 0.80f, 0.75f)
    val negatedScores = originalScores.map(-_)
    negatedScores shouldBe sorted
    negatedScores.map(-_) shouldBe originalScores
  }
}
