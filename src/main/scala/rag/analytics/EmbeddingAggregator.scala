package rag.analytics

import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.slf4j.LoggerFactory
import rag.embedding.OllamaClient

import scala.jdk.CollectionConverters.*

/**
 * Aggregates token frequencies and generates vector embeddings.
 *
 * For each unique token:
 * - Sums occurrence counts across all mappers
 * - Generates dense vector representation via LLM
 * - Outputs token with frequency and embedding
 */
class EmbeddingAggregator extends Reducer[Text, IntWritable, Text, Text] {

  private val logger = LoggerFactory.getLogger(classOf[EmbeddingAggregator])
  
  private lazy val embeddingClient = new OllamaClient()

  override def reduce(
    token: Text,
    counts: java.lang.Iterable[IntWritable],
    context: Reducer[Text, IntWritable, Text, Text]#Context
  ): Unit = {
    val tokenStr = token.toString
    
    // Aggregate frequency
    val totalFrequency = counts.asScala.map(_.get).sum
    
    // Apply minimum frequency filter
    val minFrequency = context.getConfiguration.getInt("analytics.min.frequency", 1)
    if (totalFrequency < minFrequency) {
      context.getCounter("aggregator", "filtered_tokens").increment(1)
      return
    }

    // Generate embedding
    val embeddingModel = context.getConfiguration.get("rag.embed.model", "mxbai-embed-large")
    
    val (vectorStr, success) = try {
      val vectors = embeddingClient.generateEmbeddings(Vector(tokenStr), embeddingModel)
      val vector = vectors.head
      
      if (vector.isEmpty) {
        logger.warn(s"[Aggregator] Empty embedding for: $tokenStr")
        context.getCounter("aggregator", "empty_embeddings").increment(1)
        ("", false)
      } else {
        (vector.mkString(","), true)
      }
    } catch {
      case ex: Exception =>
        logger.error(s"[Aggregator] Embedding failed for $tokenStr: ${ex.getMessage}")
        context.getCounter("aggregator", "embedding_errors").increment(1)
        ("", false)
    }

    // Output: token \t frequency,[vector]
    val outputValue = s"$totalFrequency,[$vectorStr]"
    context.write(token, new Text(outputValue))

    context.getCounter("aggregator", "tokens_processed").increment(1)
    if (!success) {
      context.getCounter("aggregator", "tokens_without_embedding").increment(1)
    }

    // Progress logging
    val processed = context.getCounter("aggregator", "tokens_processed").getValue
    if (processed % 500 == 0) {
      logger.info(s"[Aggregator] Progress: $processed tokens processed")
    }
  }

  override def cleanup(context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
    val processed = context.getCounter("aggregator", "tokens_processed").getValue
    val errors = context.getCounter("aggregator", "embedding_errors").getValue
    
    logger.info(s"[Aggregator] Finalized: $processed tokens, $errors errors")
    embeddingClient.shutdown()
  }
}
