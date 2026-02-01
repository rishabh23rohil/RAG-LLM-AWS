package rag

import io.circe.*
import io.circe.generic.semiauto.*
import sttp.client3.*
import sttp.client3.circe.*
import org.slf4j.LoggerFactory

/**
 * HTTP client for interacting with Ollama LLM API.
 *
 * Capabilities:
 * - Vector embedding generation via local models
 * - Chat completion for answer synthesis in RAG pipeline
 *
 * Endpoint: http://127.0.0.1:11434 (configurable via OLLAMA_HOST env)
 */

// ============================================================================
// Request/Response Models for Ollama API
// ============================================================================

/**
 * Single text embedding request (Ollama's preferred format).
 *
 * @param model The embedding model name (e.g., "mxbai-embed-large")
 * @param prompt The text to embed
 */
final case class EmbedSingleReq(model: String, prompt: String)
object EmbedSingleReq {
  given Encoder[EmbedSingleReq] = deriveEncoder
}

/**
 * Batch embedding request (fallback - not supported by all Ollama versions).
 *
 * @param model The embedding model name
 * @param input Vector of texts to embed
 */
final case class EmbedBatchReq(model: String, input: Vector[String])
object EmbedBatchReq {
  given Encoder[EmbedBatchReq] = deriveEncoder
}

/**
 * Embedding response from Ollama API.
 *
 * @param embeddings Vector of embedding vectors (each vector is Float array)
 */
final case class EmbedResp(embeddings: Vector[Vector[Float]])
object EmbedResp {
  // Custom decoder to handle both single and batch response formats
  given Decoder[EmbedResp] =
  Decoder.instance { c =>
    // Try "embeddings" field first (batch response format)
    c.downField("embeddings").as[Vector[Vector[Float]]].map(EmbedResp.apply)
      // Fall back to "embedding" field (single response format)
      .orElse(c.downField("embedding").as[Vector[Float]].map(v => EmbedResp(Vector(v))))
  }
}

/**
 * Chat message for conversation context.
 *
 * @param role Message role ("system", "user", or "assistant")
 * @param content Message text content
 */
final case class ChatMessage(role: String, content: String)
object ChatMessage {
  given Encoder[ChatMessage] = deriveEncoder
}

/**
 * Chat completion request.
 *
 * @param model The chat model name (e.g., "llama3.2")
 * @param messages Conversation history
 * @param stream Whether to stream response (not used in sync client)
 */
final case class ChatReq(model: String, messages: Vector[ChatMessage], stream: Boolean = false)
object ChatReq {
  given Encoder[ChatReq] = deriveEncoder
}

/**
 * Chat message in response.
 */
final case class ChatMsg(role: String, content: String)

/**
 * Chat completion response.
 *
 * @param message The generated message
 */
final case class ChatResp(message: ChatMsg)
object ChatResp {
  given Decoder[ChatMsg]  = deriveDecoder
  given Decoder[ChatResp] = deriveDecoder
}

// ============================================================================
// Ollama Client
// ============================================================================

/**
 * Synchronous Ollama API client.
 *
 * @param base Base URL for Ollama API (default: http://127.0.0.1:11434)
 */
class Ollama(base: String = sys.env.getOrElse("OLLAMA_HOST", "http://127.0.0.1:11434")) {

  private val log = LoggerFactory.getLogger(getClass)

  // HTTP client backend for synchronous requests
  private val be = HttpClientSyncBackend()

  // API endpoints
  private val eurl = uri"$base/api/embeddings"
  private val curl = uri"$base/api/chat"

  log.info(s"Initialized Ollama client with base URL: $base")

  /**
   * Generate embeddings for a vector of texts.
   *
   * Currently uses individual API calls for each text since batch embedding
   * may not be supported by all Ollama versions.
   *
   * @param texts Vector of texts to embed
   * @param model Embedding model name (e.g., "mxbai-embed-large")
   * @return Vector of embedding arrays (one per input text)
   */
  def embed(texts: Vector[String], model: String): Vector[Array[Float]] = {
    log.info(s"Generating embeddings for ${texts.size} texts using model: $model")

    // Process each text individually
    val results = texts.zipWithIndex.map { case (text, idx) =>
      log.debug(s"Embedding text ${idx + 1}/${texts.size}: ${text.take(50)}...")

      try {
        // Create embedding request with single text
        val req = basicRequest
          .post(eurl)
          .body(EmbedSingleReq(model, text))
          .response(asJson[EmbedResp])

        // Send request and get response
        val response = req.send(be)

        response.body match {
          case Right(embedResp) =>
            // Check if we got a valid embedding
            if (embedResp.embeddings.nonEmpty && embedResp.embeddings.head.nonEmpty) {
              val embedding = embedResp.embeddings.head.toArray
              log.debug(s"Successfully generated embedding for text ${idx + 1}: ${embedding.length} dimensions")
              embedding
            } else {
              // Empty embedding received
              log.warn(s"Empty embedding response for text ${idx + 1}: ${text.take(50)}...")
              Array.empty[Float]
            }

          case Left(error) =>
            // Failed to parse response
            log.error(s"Failed to parse embedding response for text ${idx + 1}: $error")
            log.error(s"Raw response body: ${response.body}")
            Array.empty[Float]
        }

      } catch {
        case e: Exception =>
          // Exception during API call
          log.error(s"Exception during embedding for text ${idx + 1} '${text.take(50)}...': ${e.getMessage}", e)
          Array.empty[Float]
      }
    }

    // Count successful embeddings
    val successCount = results.count(_.nonEmpty)
    log.info(s"Completed embedding: $successCount/${texts.size} successful")

    results
  }

  /**
   * Generate chat completion using conversation history.
   *
   * @param messages Conversation messages (system, user, assistant)
   * @param model Chat model name (e.g., "llama3.2")
   * @return Generated response text
   * @throws Exception if API call fails
   */
  def chat(messages: Vector[ChatMessage], model: String): String = {
    log.info(s"Generating chat completion using model: $model")
    log.debug(s"Message count: ${messages.size}")

    // Log conversation context (first and last message only for brevity)
    if (messages.nonEmpty) {
      log.debug(s"First message: [${messages.head.role}] ${messages.head.content.take(100)}...")
      if (messages.size > 1) {
        log.debug(s"Last message: [${messages.last.role}] ${messages.last.content.take(100)}...")
      }
    }

    try {
      // Create chat request
      val req = basicRequest
        .post(curl)
        .body(ChatReq(model, messages))
        .response(asJson[ChatResp])

      // Send request and get response
      val response = req.send(be).body.fold(
        error => {
          log.error(s"Chat API call failed: $error")
          throw new RuntimeException(s"Chat API error: $error")
        },
        success => {
          log.info("Chat completion generated successfully")
          log.debug(s"Response length: ${success.message.content.length} chars")
          success.message.content
        }
      )

      response

    } catch {
      case e: Exception =>
        log.error(s"Exception during chat completion: ${e.getMessage}", e)
        throw e
    }
  }

  /**
   * Close the HTTP client and release resources.
   * Should be called when done using the client.
   */
  def close(): Unit = {
    log.debug("Closing Ollama client HTTP backend")
    be.close()
    log.info("Ollama client closed")
  }
}