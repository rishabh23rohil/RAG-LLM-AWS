package rag

import scala.util.hashing.MurmurHash3

/** TEMP stub for embeddings so MR compiles/runs without Ollama.
  * Returns a small, deterministic 16-dim vector per input text.
  */
class OllamaClient() {
  private val dim = 16
  def embed(texts: Vector[String], model: String): Vector[Array[Float]] =
    texts.map { t =>
      val seed = MurmurHash3.stringHash(t)
      val rng  = new scala.util.Random(seed)
      Array.fill(dim)(rng.nextFloat())
    }
}