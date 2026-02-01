package rag.core

/**
 * Centralized configuration for RAG pipeline components.
 */
object Config {

  // Embedding configuration
  object Embedding {
    val defaultModel: String = "mxbai-embed-large"
    val dimensionality: Int = 1024
    val batchSize: Int = 32
  }

  // Text segmentation configuration
  object Segmentation {
    val maxSegmentLength: Int = 1024
    val overlapSize: Int = 128
    val minSegmentLength: Int = 50
  }

  // Index configuration
  object Index {
    val defaultSimilarity: String = "COSINE"
    val defaultPartitions: Int = 8
  }

  // Search configuration
  object Search {
    val defaultResultLimit: Int = 5
    val minScoreThreshold: Float = 0.0f
  }

  // Analytics configuration
  object Analytics {
    val minTokenFrequency: Int = 1
    val nearestNeighborCount: Int = 5
  }

  // API configuration
  object Api {
    val defaultPort: Int = 8080
    val requestTimeoutMs: Long = 60000
  }

  /**
   * Gets embedding model from environment or config.
   */
  def embeddingModel: String = 
    sys.env.getOrElse("RAG_EMBED_MODEL", Embedding.defaultModel)

  /**
   * Gets similarity function from environment or config.
   */
  def similarityFunction: String =
    sys.env.getOrElse("RAG_SIMILARITY", Index.defaultSimilarity)
}
