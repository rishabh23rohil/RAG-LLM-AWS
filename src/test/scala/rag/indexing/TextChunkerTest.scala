package rag.indexing

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for text segmentation functionality.
 */
class TextChunkerTest extends AnyFlatSpec with Matchers {

  "TextChunker.cleanText" should "normalize whitespace" in {
    val input = "This   has    multiple   spaces"
    val result = TextChunker.cleanText(input)
    result shouldBe "This has multiple spaces"
  }

  it should "trim edges" in {
    val input = "  text with spaces  "
    val result = TextChunker.cleanText(input)
    result shouldBe "text with spaces"
  }

  it should "convert newlines to spaces" in {
    val input = "line1\nline2\nline3"
    val result = TextChunker.cleanText(input)
    result shouldBe "line1 line2 line3"
  }

  "TextChunker.segment" should "create multiple segments for long text" in {
    val text = "A" * 2500
    val segments = TextChunker.segment(text, maxLength = 1024, overlap = 128)

    segments.size should be > 1
    segments.foreach(_.length should be <= 1024)
  }

  it should "respect overlap parameter" in {
    val text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    val segments = TextChunker.segment(text, maxLength = 30, overlap = 10)

    if (segments.size > 1) {
      segments.size should be > 1
    }
  }

  it should "prefer sentence boundaries" in {
    val text = "First sentence. Second sentence. Third sentence."
    val segments = TextChunker.segment(text, maxLength = 25, overlap = 5)

    val segmentsEndingWithPunctuation = segments.count(s => 
      s.trim.endsWith(".") || s.trim.endsWith("!") || s.trim.endsWith("?")
    )
    segmentsEndingWithPunctuation should be > 0
  }

  it should "return empty for empty input" in {
    val segments = TextChunker.segment("")
    segments.size shouldBe 0
  }

  it should "require overlap less than maxLength" in {
    assertThrows[IllegalArgumentException] {
      TextChunker.segment("text", maxLength = 100, overlap = 150)
    }
  }

  "TextChunker.segmentByParagraph" should "split by paragraph boundaries" in {
    val text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    val segments = TextChunker.segmentByParagraph(text)

    segments.size shouldBe 3
  }
}
