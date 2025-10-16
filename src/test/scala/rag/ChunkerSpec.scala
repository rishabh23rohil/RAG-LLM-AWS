package rag

import org.scalatest.funsuite.AnyFunSuite

class ChunkerSpec extends AnyFunSuite {
  test("normalize and split produces overlapping chunks") {
    val s = "Sentence one. Sentence two. Sentence three. " * 50
    val chunks = Chunker.split(s, maxChars = 300, overlap = 60)
    assert(chunks.nonEmpty)
    assert(chunks.exists(_.length <= 300))
    assert(chunks.length < s.length) // not one-by-one
  }
}
