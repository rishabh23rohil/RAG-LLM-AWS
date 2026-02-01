package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Unit tests for word embedding post-processing utilities.
 * Tests similarity calculations and vector arithmetic.
 */
class PostProcessEmbeddingsTest extends AnyFlatSpec with Matchers {

  "PostProcessEmbeddings.computeNorm" should "calculate L2 norm correctly" in {
    val vec = Array(3.0f, 4.0f)
    val norm = PostProcessEmbeddings.computeNorm(vec)
    norm shouldBe 5.0f +- 0.001f
  }

  it should "handle zero vector" in {
    val vec = Array(0.0f, 0.0f)
    val norm = PostProcessEmbeddings.computeNorm(vec)
    norm shouldBe 0.0f
  }

  it should "handle single element" in {
    val vec = Array(5.0f)
    val norm = PostProcessEmbeddings.computeNorm(vec)
    norm shouldBe 5.0f +- 0.001f
  }

  "PostProcessEmbeddings.cosineSimilarity" should "return 1 for identical vectors" in {
    val v1 = Array(1.0f, 2.0f, 3.0f)
    val norm1 = PostProcessEmbeddings.computeNorm(v1)

    val sim = PostProcessEmbeddings.cosineSimilarity(v1, norm1, v1, norm1)
    sim shouldBe 1.0 +- 0.001
  }

  it should "return 0 for orthogonal vectors" in {
    val v1 = Array(1.0f, 0.0f)
    val v2 = Array(0.0f, 1.0f)
    val norm1 = PostProcessEmbeddings.computeNorm(v1)
    val norm2 = PostProcessEmbeddings.computeNorm(v2)

    val sim = PostProcessEmbeddings.cosineSimilarity(v1, norm1, v2, norm2)
    sim shouldBe 0.0 +- 0.001
  }

  it should "handle zero vectors" in {
    val v1 = Array(1.0f, 2.0f)
    val v2 = Array(0.0f, 0.0f)
    val norm1 = PostProcessEmbeddings.computeNorm(v1)
    val norm2 = PostProcessEmbeddings.computeNorm(v2)

    val sim = PostProcessEmbeddings.cosineSimilarity(v1, norm1, v2, norm2)
    sim shouldBe 0.0
  }

  it should "return negative for opposite vectors" in {
    val v1 = Array(1.0f, 1.0f)
    val v2 = Array(-1.0f, -1.0f)
    val norm1 = PostProcessEmbeddings.computeNorm(v1)
    val norm2 = PostProcessEmbeddings.computeNorm(v2)

    val sim = PostProcessEmbeddings.cosineSimilarity(v1, norm1, v2, norm2)
    sim shouldBe -1.0 +- 0.001
  }

  "PostProcessEmbeddings.add" should "add vectors element-wise" in {
    val v1 = Array(1.0f, 2.0f, 3.0f)
    val v2 = Array(4.0f, 5.0f, 6.0f)
    val result = PostProcessEmbeddings.add(v1, v2)

    result shouldBe Array(5.0f, 7.0f, 9.0f)
  }

  "PostProcessEmbeddings.sub" should "subtract vectors element-wise" in {
    val v1 = Array(5.0f, 7.0f, 9.0f)
    val v2 = Array(1.0f, 2.0f, 3.0f)
    val result = PostProcessEmbeddings.sub(v1, v2)

    result shouldBe Array(4.0f, 5.0f, 6.0f)
  }

  it should "handle negative results" in {
    val v1 = Array(1.0f, 2.0f)
    val v2 = Array(3.0f, 4.0f)
    val result = PostProcessEmbeddings.sub(v1, v2)

    result shouldBe Array(-2.0f, -2.0f)
  }
}