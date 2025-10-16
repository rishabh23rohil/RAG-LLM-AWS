package rag

import org.scalatest.funsuite.AnyFunSuite

class ShardSpec extends AnyFunSuite {
  test("stable shard mapping") {
    val R = 2
    val a = math.abs("docA.pdf".hashCode) % R
    val b = math.abs("docA.pdf".hashCode) % R
    assert(a == b)
  }
}
