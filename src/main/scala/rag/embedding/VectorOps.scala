package rag.embedding

/**
 * Vector mathematics operations for embedding processing.
 */
object VectorOps {

  /**
   * Normalizes a vector to unit length (L2 normalization).
   * Essential for cosine similarity computations.
   */
  def normalize(vector: Array[Float]): Array[Float] = {
    val magnitude = math.sqrt(vector.map(x => x.toDouble * x).sum).toFloat
    if (magnitude == 0f) vector else vector.map(_ / magnitude)
  }

  /**
   * Computes cosine similarity between two vectors.
   */
  def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, "Vectors must have equal dimensions")
    
    val dotProduct = v1.zip(v2).map { case (a, b) => a * b }.sum
    val norm1 = math.sqrt(v1.map(x => x * x).sum).toFloat
    val norm2 = math.sqrt(v2.map(x => x * x).sum).toFloat
    
    if (norm1 == 0f || norm2 == 0f) 0f else dotProduct / (norm1 * norm2)
  }

  /**
   * Computes Euclidean distance between two vectors.
   */
  def euclideanDistance(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, "Vectors must have equal dimensions")
    math.sqrt(v1.zip(v2).map { case (a, b) => math.pow(a - b, 2) }.sum).toFloat
  }

  /**
   * Vector addition.
   */
  def add(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    require(v1.length == v2.length, "Vectors must have equal dimensions")
    v1.zip(v2).map { case (a, b) => a + b }
  }

  /**
   * Vector subtraction.
   */
  def subtract(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    require(v1.length == v2.length, "Vectors must have equal dimensions")
    v1.zip(v2).map { case (a, b) => a - b }
  }
}
