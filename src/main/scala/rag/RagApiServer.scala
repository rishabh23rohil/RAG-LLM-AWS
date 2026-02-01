package rag

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

import java.nio.file.{Files, Paths}

// API Models
case class QueryRequest(
                         query: String,
                         topK: Option[Int] = Some(5),
                         embedModel: Option[String] = Some("mxbai-embed-large"),
                         chatModel: Option[String] = Some("llama3")
                       )

case class QueryResponse(
                          query: String,
                          answer: String,
                          results: Seq[SearchResultDto],
                          processingTimeMs: Long
                        )

case class SearchResultDto(docId: String, chunkId: Int, text: String, score: Float)

case class HealthResponse(status: String, indexPath: String, shardsAvailable: Int)

case class ErrorResponse(error: String, message: String)

object RagApiServer extends IOApp {

  private val log = LoggerFactory.getLogger(getClass)

  // Configuration
  private val indexPath = sys.env.getOrElse("RAG_INDEX_PATH", s"${System.getProperty("user.home")}/lucene-index")
  private val port = sys.env.getOrElse("RAG_API_PORT", "8080").toInt

  // JSON codecs
  implicit val queryRequestDecoder: EntityDecoder[IO, QueryRequest] = jsonOf[IO, QueryRequest]
  implicit val queryResponseEncoder: EntityEncoder[IO, QueryResponse] = jsonEncoderOf[IO, QueryResponse]
  implicit val healthResponseEncoder: EntityEncoder[IO, HealthResponse] = jsonEncoderOf[IO, HealthResponse]
  implicit val errorResponseEncoder: EntityEncoder[IO, ErrorResponse] = jsonEncoderOf[IO, ErrorResponse]
  implicit val searchResultsEncoder: EntityEncoder[IO, Seq[SearchResultDto]] = jsonEncoderOf[IO, Seq[SearchResultDto]]

  // Routes
  val ragRoutes = HttpRoutes.of[IO] {

    // POST /query - Full RAG: search + answer generation
    case req @ POST -> Root / "query" =>
      (for {
        queryReq <- req.as[QueryRequest]
        _ = log.info(s"Query received: ${queryReq.query}")

        startTime = System.currentTimeMillis()

        // Execute RAG pipeline
        answerText <- IO {
          RagQuery.answer(
            query = queryReq.query,
            indexDir = indexPath,
            model = queryReq.embedModel.getOrElse("mxbai-embed-large"),
            topK = queryReq.topK.getOrElse(5),
            chatModel = queryReq.chatModel.getOrElse("llama3")
          )
        }

        // Get search results for response
        results <- IO {
          val client = new Ollama()
          try {
            val queryVec = client.embed(
              Vector(queryReq.query),
              queryReq.embedModel.getOrElse("mxbai-embed-large")
            ).head

            RagQuery.searchAllShards(
              Paths.get(indexPath),
              queryVec,
              queryReq.topK.getOrElse(5)
            ).map(r => SearchResultDto(r.docId, r.chunkId, r.text, r.score))
          } finally {
            client.close()
          }
        }

        endTime = System.currentTimeMillis()

        response = QueryResponse(
          query = queryReq.query,
          answer = answerText,
          results = results,
          processingTimeMs = endTime - startTime
        )

        resp <- Ok(response.asJson)
      } yield resp).handleErrorWith { e =>
        log.error(s"Query failed: ${e.getMessage}", e)
        InternalServerError(ErrorResponse("query_failed", e.getMessage).asJson)
      }

    // GET /search - Search only (no answer generation)
    case GET -> Root / "search" :? QueryParam(query) +& TopKParam(topK) +& ModelParam(model) =>
      (for {
        client <- IO(new Ollama())

        queryVec <- IO {
          client.embed(Vector(query), model.getOrElse("mxbai-embed-large")).head
        }

        results <- IO {
          RagQuery.searchAllShards(
            Paths.get(indexPath),
            queryVec,
            topK.getOrElse(5)
          ).map(r => SearchResultDto(r.docId, r.chunkId, r.text, r.score))
        }

        _ <- IO(client.close())

        resp <- Ok(results.asJson)
      } yield resp).handleErrorWith { e =>
        log.error(s"Search failed: ${e.getMessage}", e)
        InternalServerError(ErrorResponse("search_failed", e.getMessage).asJson)
      }

    // GET /health - Health check
    case GET -> Root / "health" =>
      IO {
        val indexDir = Paths.get(indexPath)
        val shardCount = if (Files.exists(indexDir)) {
          Files.list(indexDir)
            .filter(p => p.getFileName.toString.startsWith("index_shard_"))
            .count()
            .toInt
        } else 0

        HealthResponse(
          status = if (shardCount > 0) "healthy" else "no_index",
          indexPath = indexPath,
          shardsAvailable = shardCount
        )
      }.flatMap(health => Ok(health.asJson))
  }

  // Query parameter extractors
  object QueryParam extends QueryParamDecoderMatcher[String]("q")
  object TopKParam extends OptionalQueryParamDecoderMatcher[Int]("topK")
  object ModelParam extends OptionalQueryParamDecoderMatcher[String]("model")

  // HTTP app with routes
  private val httpApp = Router("/api/v1" -> ragRoutes).orNotFound

  // Server entry point
  def run(args: List[String]): IO[ExitCode] = {
    log.info(s"Initializing RAG REST API")
    log.info(s"  Port: $port")
    log.info(s"  Index: $indexPath")

    EmberServerBuilder
      .default[IO]
      .withHost(ipv4"0.0.0.0")
      .withPort(Port.fromInt(port).get)
      .withHttpApp(httpApp)
      .build
      .use(_ => IO.never)
      .as(ExitCode.Success)
  }
}