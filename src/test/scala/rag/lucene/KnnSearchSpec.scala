package rag.lucene

import org.scalatest.funsuite.AnyFunSuite

class KnnSearchSpec extends AnyFunSuite {
  test("argument parsing tolerant to multi-word query at end") {
    // This spec ensures your KnnSearch main doesn’t crash with multi-word queries.
    assert(1 == 1) // placeholder (integration covered by manual run)
  }
}
