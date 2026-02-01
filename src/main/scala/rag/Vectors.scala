package rag

/**
 * Vector mathematics utilities for embedding operations.
 */
object Vectors {

  /**
   * Applies L2 (Euclidean) normalization to a vector.
   * Required for cosine similarity and inner product comparisons.
   */
  def l2(v: Array[Float]): Array[Float] = {
    val norm = math.sqrt(v.foldLeft(0.0)((sum, x) => sum + x * x)).toFloat
    if (norm == 0f) v else v.map(_ / norm)
  }
}
