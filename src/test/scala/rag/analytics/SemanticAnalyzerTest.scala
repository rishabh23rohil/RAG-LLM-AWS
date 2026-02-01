package rag.analytics

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for semantic analysis utilities.
 */
class SemanticAnalyzerTest extends AnyFlatSpec with Matchers {

  "SemanticAnalyzer.computeMagnitude" should "calculate L2 norm correctly" in {
    val vec = Array(3.0f, 4.0f)
    val magnitude = SemanticAnalyzer.computeMagnitude(vec)
    magnitude shouldBe 5.0f +- 0.001f
  }

  it should "handle zero vector" in {
    val vec = Array(0.0f, 0.0f)
    val magnitude = SemanticAnalyzer.computeMagnitude(vec)
    magnitude shouldBe 0.0f
  }

  it should "handle single element" in {
    val vec = Array(5.0f)
    val magnitude = SemanticAnalyzer.computeMagnitude(vec)
    magnitude shouldBe 5.0f +- 0.001f
  }

  "SemanticAnalyzer.similarity" should "return 1 for identical vectors" in {
    val entry = SemanticAnalyzer.VocabularyEntry(
      term = "test",
      vector = Array(1.0f, 2.0f, 3.0f),
      magnitude = SemanticAnalyzer.computeMagnitude(Array(1.0f, 2.0f, 3.0f)),
      occurrences = 1
    )

    val sim = SemanticAnalyzer.similarity(entry, entry)
    sim shouldBe 1.0 +- 0.001
  }

  it should "return 0 for orthogonal vectors" in {
    val entry1 = SemanticAnalyzer.VocabularyEntry("a", Array(1.0f, 0.0f), 1.0f, 1)
    val entry2 = SemanticAnalyzer.VocabularyEntry("b", Array(0.0f, 1.0f), 1.0f, 1)

    val sim = SemanticAnalyzer.similarity(entry1, entry2)
    sim shouldBe 0.0 +- 0.001
  }

  it should "handle zero magnitude" in {
    val entry1 = SemanticAnalyzer.VocabularyEntry("a", Array(1.0f, 2.0f), 2.236f, 1)
    val entry2 = SemanticAnalyzer.VocabularyEntry("b", Array(0.0f, 0.0f), 0.0f, 1)

    val sim = SemanticAnalyzer.similarity(entry1, entry2)
    sim shouldBe 0.0
  }

  "SemanticAnalyzer.parseVocabularyLine" should "parse valid lines" in {
    val line = "hello\t5,[0.1,0.2,0.3]"
    val result = SemanticAnalyzer.parseVocabularyLine(line)

    result shouldBe defined
    result.get._1 shouldBe "hello"
    result.get._2.length shouldBe 3
    result.get._3 shouldBe 5
  }

  it should "return None for invalid lines" in {
    val line = "invalid line without brackets"
    val result = SemanticAnalyzer.parseVocabularyLine(line)
    result shouldBe None
  }

  it should "handle empty lines" in {
    val result = SemanticAnalyzer.parseVocabularyLine("")
    result shouldBe None
  }
}
