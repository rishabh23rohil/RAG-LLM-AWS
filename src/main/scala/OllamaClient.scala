package rag

import sttp.client3.*
import sttp.client3.circe.*
import io.circe.*
import io.circe.generic.semiauto.*

/** Request payload for /api/embeddings (Ollama expects "prompt") */
final case class EmbedReq(model: String, prompt: String)
object EmbedReq:
  given Encoder[EmbedReq] = deriveEncoder

/** Response can be either "embedding" (single) or "embeddings" (batch) */
final case class EmbedResp(embedding: Option[Vector[Float]], embeddings: Option[Vector[Vector[Float]]])
object EmbedResp:
  given Decoder[EmbedResp] =
    Decoder.instance { c =>
      val single = c.downField("embedding").as[Vector[Float]].toOption
      val multi  = c.downField("embeddings").as[Vector[Vector[Float]]].toOption
      Right(EmbedResp(single, multi))
    }

class Ollama(base: String = sys.env.getOrElse("OLLAMA_HOST", "http://127.0.0.1:11434")):
  private val be   = HttpClientSyncBackend()
  private val eurl = uri"$base/api/embeddings"

  /** Returns the embedding as a Float array (first vector if batch) */
  def embed(text: String, model: String = "mxbai-embed-large"): Array[Float] =
    val req = basicRequest
      .post(eurl)
      .body(EmbedReq(model, text)) // requires Encoder[EmbedReq]
      .response(asJson[EmbedResp])

    val resp = req.send(be).body.fold(throw _, identity)
    resp.embedding
      .orElse(resp.embeddings.flatMap(_.headOption))
      .map(_.toArray)
      .getOrElse(Array.emptyFloatArray)
