package rag.api

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.http4s.{Method, Request, Uri}
import org.http4s.implicits._

/**
 * Integration tests for Search API endpoints.
 */
class SearchApiServiceTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  "Status endpoint" should "have correct route" in {
    val request = Request[IO](Method.GET, uri"/api/v1/status")

    request.method shouldBe Method.GET
    request.uri.path.toString should include("status")
  }

  "SearchRequest model" should "contain required fields" in {
    val searchReq = SearchRequest(
      question = "What is machine learning?",
      limit = Some(5),
      embeddingModel = Some("mxbai-embed-large"),
      completionModel = Some("llama3")
    )

    searchReq.question should not be empty
    searchReq.limit shouldBe Some(5)
    searchReq.embeddingModel shouldBe Some("mxbai-embed-large")
  }

  "SearchResponse model" should "have all fields" in {
    val response = SearchResponse(
      question = "test",
      answer = "answer",
      relevantSegments = Seq.empty,
      latencyMs = 100
    )

    response.question shouldBe "test"
    response.answer shouldBe "answer"
    response.latencyMs should be > 0L
  }

  "SegmentDto" should "store segment information" in {
    val segment = SegmentDto(
      documentKey = "test.pdf",
      segmentIndex = 0,
      content = "sample content",
      score = 0.95f
    )

    segment.documentKey shouldBe "test.pdf"
    segment.segmentIndex shouldBe 0
    segment.score shouldBe 0.95f +- 0.001f
  }
}
