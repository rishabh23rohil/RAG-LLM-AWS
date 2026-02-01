package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for vector operations.
 * Tests L2 normalization and vector utilities.
 */
class VectorsTest extends AnyFlatSpec with Matchers {

  "Vectors.l2" should "normalize a vector to unit length" in {
    val input = Array(3.0f, 4.0f)  // Length 5
    val normalized = Vectors.l2(input)

    normalized(0) shouldBe 0.6f +- 0.001f
    normalized(1) shouldBe 0.8f +- 0.001f

    // Check unit length
    val length = math.sqrt(normalized.map(x => x * x).sum)
    length shouldBe 1.0 +- 0.001
  }

  it should "handle zero vector" in {
    val input = Array(0.0f, 0.0f, 0.0f)
    val normalized = Vectors.l2(input)

    // Zero vector should remain zero
    normalized.forall(_ == 0.0f) shouldBe true
  }

  it should "preserve direction" in {
    val input = Array(1.0f, 2.0f, 3.0f)
    val normalized = Vectors.l2(input)

    // Ratios should be preserved
    val ratio1 = normalized(1) / normalized(0)
    val ratio2 = input(1) / input(0)
    ratio1 shouldBe ratio2 +- 0.001f
  }

  it should "work with negative values" in {
    val input = Array(-3.0f, 4.0f)
    val normalized = Vectors.l2(input)

    val length = math.sqrt(normalized.map(x => x * x).sum)
    length shouldBe 1.0 +- 0.001
  }

  it should "not modify already normalized vector significantly" in {
    val input = Array(0.6f, 0.8f)  // Already normalized
    val normalized = Vectors.l2(input)

    normalized(0) shouldBe input(0) +- 0.001f
    normalized(1) shouldBe input(1) +- 0.001f
  }
}