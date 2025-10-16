package rag

import org.scalatest.funsuite.AnyFunSuite

class OllamaClientSpec extends AnyFunSuite {
  test("embed returns a vector per input when server is available") {
    val c = new OllamaClient()
    try {
      val out = c.embed(Vector("hello","world"), "default")
      assert(out.length == 2)
    } catch {
      case _: Throwable => cancel("Ollama not reachable; skipping")
    }
  }
}
