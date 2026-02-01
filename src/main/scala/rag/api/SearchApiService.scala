package rag.api

import cats.effect._
import com.comcast.ip4s._
import io.circe.generic.auto._
import io.circe.syntax._
import org.http4s._
import org.http4s.circe._
import org.http4s.dsl.io._
import org.http4s.ember.server.EmberServerBuilder
import org.http4s.implicits._
import org.http4s.server.Router
import org.slf4j.LoggerFactory
import rag.search.{VectorSearchEngine, QueryResult}
import rag.embedding.{OllamaClient, ConversationMessage}

import java.nio.file.{Files, Paths}

// Request/Response DTOs
case class SearchRequest(
  question: String,
  limit: Option[Int] = Some(5),
  embeddingModel: Option[String] = Some("mxbai-embed-large"),
  completionModel: Option[String] = Some("llama3")
)

case class SearchResponse(
  question: String,
  answer: String,
  relevantSegments: Seq[SegmentDto],
  latencyMs: Long
)

case class SegmentDto(documentKey: String, segmentIndex: Int, content: String, score: Float)

case class StatusResponse(status: String, indexLocation: String, partitionCount: Int)

case class ApiError(errorType: String, description: String)

/**
 * REST API service for vector search and RAG queries.
 */
object SearchApiService extends IOApp {

  private val logger = LoggerFactory.getLogger(classOf[SearchApiService.type])

  private val indexLocation = sys.env.getOrElse("RAG_INDEX_PATH", s"${System.getProperty("user.home")}/lucene-index")
  private val serverPort = sys.env.getOrElse("RAG_API_PORT", "8080").toInt

  // Codecs
  given EntityDecoder[IO, SearchRequest] = jsonOf[IO, SearchRequest]
  given EntityEncoder[IO, SearchResponse] = jsonEncoderOf[IO, SearchResponse]
  given EntityEncoder[IO, StatusResponse] = jsonEncoderOf[IO, StatusResponse]
  given EntityEncoder[IO, ApiError] = jsonEncoderOf[IO, ApiError]
  given EntityEncoder[IO, Seq[SegmentDto]] = jsonEncoderOf[IO, Seq[SegmentDto]]

  private val apiRoutes = HttpRoutes.of[IO] {

    // POST /ask - Full RAG pipeline
    case req @ POST -> Root / "ask" =>
      (for {
        searchReq <- req.as[SearchRequest]
        requestStart = System.currentTimeMillis()
        
        _ = logger.info(s"[API] Query: ${searchReq.question.take(50)}...")

        answer <- IO {
          VectorSearchEngine.answerQuery(
            question = searchReq.question,
            indexPath = indexLocation,
            embeddingModel = searchReq.embeddingModel.getOrElse("mxbai-embed-large"),
            resultLimit = searchReq.limit.getOrElse(5),
            generationModel = searchReq.completionModel.getOrElse("llama3")
          )
        }

        segments <- IO {
          val client = new OllamaClient()
          try {
            val queryVector = client.generateEmbeddings(
              Vector(searchReq.question),
              searchReq.embeddingModel.getOrElse("mxbai-embed-large")
            ).head

            VectorSearchEngine.searchAllPartitions(
              Paths.get(indexLocation),
              queryVector,
              searchReq.limit.getOrElse(5)
            ).map(r => SegmentDto(r.documentKey, r.segmentIndex, r.content, r.relevanceScore))
          } finally {
            client.shutdown()
          }
        }

        latency = System.currentTimeMillis() - requestStart

        response = SearchResponse(
          question = searchReq.question,
          answer = answer,
          relevantSegments = segments,
          latencyMs = latency
        )

        result <- Ok(response.asJson)
      } yield result).handleErrorWith { ex =>
        logger.error(s"[API] Query failed: ${ex.getMessage}", ex)
        InternalServerError(ApiError("query_error", ex.getMessage).asJson)
      }

    // GET /search - Vector search only
    case GET -> Root / "search" :? QuestionParam(q) +& LimitParam(limit) +& ModelParam(model) =>
      (for {
        client <- IO(new OllamaClient())

        queryVector <- IO {
          client.generateEmbeddings(Vector(q), model.getOrElse("mxbai-embed-large")).head
        }

        results <- IO {
          VectorSearchEngine.searchAllPartitions(
            Paths.get(indexLocation),
            queryVector,
            limit.getOrElse(5)
          ).map(r => SegmentDto(r.documentKey, r.segmentIndex, r.content, r.relevanceScore))
        }

        _ <- IO(client.shutdown())

        response <- Ok(results.asJson)
      } yield response).handleErrorWith { ex =>
        logger.error(s"[API] Search failed: ${ex.getMessage}", ex)
        InternalServerError(ApiError("search_error", ex.getMessage).asJson)
      }

    // GET /status - Health check
    case GET -> Root / "status" =>
      IO {
        val indexDir = Paths.get(indexLocation)
        val partitions = if (Files.exists(indexDir)) {
          Files.list(indexDir)
            .filter(p => p.getFileName.toString.startsWith("partition_"))
            .count()
            .toInt
        } else 0

        StatusResponse(
          status = if (partitions > 0) "operational" else "no_index",
          indexLocation = indexLocation,
          partitionCount = partitions
        )
      }.flatMap(status => Ok(status.asJson))
  }

  // Query parameter extractors
  object QuestionParam extends QueryParamDecoderMatcher[String]("q")
  object LimitParam extends OptionalQueryParamDecoderMatcher[Int]("limit")
  object ModelParam extends OptionalQueryParamDecoderMatcher[String]("model")

  private val httpApp = Router("/api/v1" -> apiRoutes).orNotFound

  def run(args: List[String]): IO[ExitCode] = {
    logger.info(s"[API] Starting Search API Service")
    logger.info(s"[API] Port: $serverPort")
    logger.info(s"[API] Index: $indexLocation")

    EmberServerBuilder
      .default[IO]
      .withHost(ipv4"0.0.0.0")
      .withPort(Port.fromInt(serverPort).get)
      .withHttpApp(httpApp)
      .build
      .use(_ => IO.never)
      .as(ExitCode.Success)
  }
}
