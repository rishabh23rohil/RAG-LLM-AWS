package rag.embedding

import io.circe.*
import io.circe.generic.semiauto.*
import sttp.client3.*
import sttp.client3.circe.*
import org.slf4j.LoggerFactory

/**
 * Client for Ollama LLM API integration.
 *
 * Provides:
 * - Dense vector generation from text
 * - Conversational completions for answer generation
 *
 * Configuration via OLLAMA_HOST environment variable.
 */

// API DTOs
private[embedding] final case class EmbeddingRequest(model: String, prompt: String)
private[embedding] object EmbeddingRequest {
  given Encoder[EmbeddingRequest] = deriveEncoder
}

private[embedding] final case class EmbeddingResponse(embeddings: Vector[Vector[Float]])
private[embedding] object EmbeddingResponse {
  given Decoder[EmbeddingResponse] = Decoder.instance { cursor =>
    cursor.downField("embeddings").as[Vector[Vector[Float]]].map(EmbeddingResponse.apply)
      .orElse(cursor.downField("embedding").as[Vector[Float]].map(v => EmbeddingResponse(Vector(v))))
  }
}

final case class ConversationMessage(role: String, content: String)
object ConversationMessage {
  given Encoder[ConversationMessage] = deriveEncoder
}

private[embedding] final case class CompletionRequest(model: String, messages: Vector[ConversationMessage], stream: Boolean = false)
private[embedding] object CompletionRequest {
  given Encoder[CompletionRequest] = deriveEncoder
}

private[embedding] final case class ResponseMessage(role: String, content: String)
private[embedding] final case class CompletionResponse(message: ResponseMessage)
private[embedding] object CompletionResponse {
  given Decoder[ResponseMessage] = deriveDecoder
  given Decoder[CompletionResponse] = deriveDecoder
}

/**
 * Synchronous client for Ollama embedding and completion APIs.
 */
class OllamaClient(
  baseUrl: String = sys.env.getOrElse("OLLAMA_HOST", "http://127.0.0.1:11434")
) {
  private val logger = LoggerFactory.getLogger(classOf[OllamaClient])
  private val httpBackend = HttpClientSyncBackend()
  
  private val embeddingEndpoint = uri"$baseUrl/api/embeddings"
  private val completionEndpoint = uri"$baseUrl/api/chat"

  logger.debug(s"[OllamaClient] Initialized with endpoint: $baseUrl")

  /**
   * Generates vector embeddings for input texts.
   */
  def generateEmbeddings(texts: Vector[String], modelName: String): Vector[Array[Float]] = {
    val startTime = System.currentTimeMillis()
    logger.debug(s"[OllamaClient] Embedding ${texts.size} texts with $modelName")

    val vectors = texts.zipWithIndex.map { case (text, index) =>
      try {
        val request = basicRequest
          .post(embeddingEndpoint)
          .body(EmbeddingRequest(modelName, text))
          .response(asJson[EmbeddingResponse])

        request.send(httpBackend).body match {
          case Right(resp) if resp.embeddings.nonEmpty =>
            resp.embeddings.head.toArray
          case Right(_) =>
            logger.warn(s"[OllamaClient] Empty response for text $index")
            Array.empty[Float]
          case Left(err) =>
            logger.error(s"[OllamaClient] API error for text $index: $err")
            Array.empty[Float]
        }
      } catch {
        case ex: Exception =>
          logger.error(s"[OllamaClient] Exception embedding text $index: ${ex.getMessage}")
          Array.empty[Float]
      }
    }

    val duration = System.currentTimeMillis() - startTime
    val successRate = vectors.count(_.nonEmpty).toDouble / texts.size * 100
    logger.debug(f"[OllamaClient] Embedded ${texts.size} texts in ${duration}ms ($successRate%.1f%% success)")

    vectors
  }

  /**
   * Generates completion response from conversation history.
   */
  def generateCompletion(messages: Vector[ConversationMessage], modelName: String): String = {
    logger.debug(s"[OllamaClient] Generating completion with $modelName (${messages.size} messages)")

    try {
      val request = basicRequest
        .post(completionEndpoint)
        .body(CompletionRequest(modelName, messages))
        .response(asJson[CompletionResponse])

      request.send(httpBackend).body match {
        case Right(resp) =>
          logger.debug(s"[OllamaClient] Completion generated: ${resp.message.content.length} chars")
          resp.message.content
        case Left(err) =>
          throw new RuntimeException(s"Completion API error: $err")
      }
    } catch {
      case ex: Exception =>
        logger.error(s"[OllamaClient] Completion failed: ${ex.getMessage}", ex)
        throw ex
    }
  }

  def shutdown(): Unit = {
    logger.debug("[OllamaClient] Shutting down HTTP backend")
    httpBackend.close()
  }
}
