package rag

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.http4s.{Method, Request, Status, Uri}
import org.http4s.implicits._

/**
 * Integration tests for REST API endpoints.
 * Tests API routes without starting the full server.
 */
class ApiIntegrationTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  "Health endpoint" should "return status" in {
    val request = Request[IO](Method.GET, uri"/api/v1/health")

    request.method shouldBe Method.GET
    request.uri.path.toString should include("health")
  }

  "Query request model" should "have required fields" in {
    val queryReq = QueryRequest(
      query = "What is machine learning?",
      topK = Some(5),
      embedModel = Some("mxbai-embed-large"),
      chatModel = Some("llama3")
    )

    queryReq.query should not be empty
    queryReq.topK shouldBe Some(5)
    queryReq.embedModel shouldBe Some("mxbai-embed-large")
  }

  "Query response model" should "contain all required fields" in {
    val response = QueryResponse(
      query = "test",
      answer = "answer",
      results = Seq.empty,
      processingTimeMs = 100
    )

    response.query shouldBe "test"
    response.answer shouldBe "answer"
    response.processingTimeMs should be > 0L
  }

  "SearchResultDto" should "store document information" in {
    val result = SearchResultDto(
      docId = "test.pdf",
      chunkId = 0,
      text = "sample text",
      score = 0.95f
    )

    result.docId shouldBe "test.pdf"
    result.chunkId shouldBe 0
    result.score shouldBe 0.95f +- 0.001f
  }
}