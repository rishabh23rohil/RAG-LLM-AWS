package rag.embedding

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for vector mathematics operations.
 */
class VectorOpsTest extends AnyFlatSpec with Matchers {

  "VectorOps.normalize" should "produce unit length vector" in {
    val input = Array(3.0f, 4.0f)
    val normalized = VectorOps.normalize(input)

    normalized(0) shouldBe 0.6f +- 0.001f
    normalized(1) shouldBe 0.8f +- 0.001f

    val magnitude = math.sqrt(normalized.map(x => x * x).sum)
    magnitude shouldBe 1.0 +- 0.001
  }

  it should "handle zero vector" in {
    val input = Array(0.0f, 0.0f, 0.0f)
    val normalized = VectorOps.normalize(input)

    normalized.forall(_ == 0.0f) shouldBe true
  }

  it should "preserve direction" in {
    val input = Array(1.0f, 2.0f, 3.0f)
    val normalized = VectorOps.normalize(input)

    val ratio1 = normalized(1) / normalized(0)
    val ratio2 = input(1) / input(0)
    ratio1 shouldBe ratio2 +- 0.001f
  }

  it should "work with negative values" in {
    val input = Array(-3.0f, 4.0f)
    val normalized = VectorOps.normalize(input)

    val magnitude = math.sqrt(normalized.map(x => x * x).sum)
    magnitude shouldBe 1.0 +- 0.001
  }

  "VectorOps.cosineSimilarity" should "return 1.0 for identical vectors" in {
    val v = Array(1.0f, 0.0f, 0.0f)
    VectorOps.cosineSimilarity(v, v) shouldBe 1.0f +- 0.001f
  }

  it should "return 0.0 for orthogonal vectors" in {
    val v1 = Array(1.0f, 0.0f)
    val v2 = Array(0.0f, 1.0f)
    VectorOps.cosineSimilarity(v1, v2) shouldBe 0.0f +- 0.001f
  }

  "VectorOps.euclideanDistance" should "compute correct distance" in {
    val v1 = Array(0.0f, 0.0f)
    val v2 = Array(3.0f, 4.0f)
    VectorOps.euclideanDistance(v1, v2) shouldBe 5.0f +- 0.001f
  }

  "VectorOps.add" should "perform element-wise addition" in {
    val v1 = Array(1.0f, 2.0f)
    val v2 = Array(3.0f, 4.0f)
    val result = VectorOps.add(v1, v2)
    
    result(0) shouldBe 4.0f
    result(1) shouldBe 6.0f
  }

  "VectorOps.subtract" should "perform element-wise subtraction" in {
    val v1 = Array(5.0f, 8.0f)
    val v2 = Array(2.0f, 3.0f)
    val result = VectorOps.subtract(v1, v2)
    
    result(0) shouldBe 3.0f
    result(1) shouldBe 5.0f
  }
}
