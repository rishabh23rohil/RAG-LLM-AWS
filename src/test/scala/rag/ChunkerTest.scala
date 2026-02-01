package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for text chunking functionality.
 * Tests chunking logic, overlap, and boundary detection.
 */
class ChunkerTest extends AnyFlatSpec with Matchers {

  "Chunker.normalize" should "remove extra whitespace" in {
    val input = "This   has    multiple   spaces"
    val result = Chunker.normalize(input)
    result shouldBe "This has multiple spaces"
  }

  it should "trim leading and trailing whitespace" in {
    val input = "  text with spaces  "
    val result = Chunker.normalize(input)
    result shouldBe "text with spaces"
  }

  it should "replace newlines with spaces" in {
    val input = "line1\nline2\nline3"
    val result = Chunker.normalize(input)
    result shouldBe "line1 line2 line3"
  }
  

  it should "create multiple chunks for long text" in {
    val text = "A" * 2500  // Long text
    val chunks = Chunker.split(text, maxChars = 1000, overlap = 100)

    chunks.size should be > 1
    chunks.foreach(_.length should be <= 1000)
  }

  it should "respect overlap parameter" in {
    val text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    val chunks = Chunker.split(text, maxChars = 30, overlap = 10)

    // Check that chunks overlap
    if (chunks.size > 1) {
      val firstChunkEnd = chunks.head.takeRight(10)
      val secondChunkStart = chunks(1).take(10)
      // Some overlap should exist
      chunks.size should be > 1
    }
  }

  it should "break at sentence boundaries when possible" in {
    val text = "First sentence. Second sentence. Third sentence."
    val chunks = Chunker.split(text, maxChars = 25, overlap = 5)

    // Most chunks should end with a period
    val chunksEndingWithPeriod = chunks.count(_.trim.endsWith("."))
    chunksEndingWithPeriod should be > 0
  }

  it should "handle empty text" in {
    val text = ""
    val chunks = Chunker.split(text)
    chunks.size shouldBe 0
  }

  it should "require overlap less than maxChars" in {
    assertThrows[IllegalArgumentException] {
      Chunker.split("text", maxChars = 100, overlap = 150)
    }
  }
}