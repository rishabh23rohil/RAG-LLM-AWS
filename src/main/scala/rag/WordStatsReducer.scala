package rag

import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.slf4j.LoggerFactory

import scala.jdk.CollectionConverters.*

/**
 * Reducer for Word Statistics with embeddings.
 */
class WordStatsReducer extends Reducer[Text, IntWritable, Text, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  // Ollama client for embedding generation (shared across all reduce calls)
  private val client = new Ollama()

  /**
   * Reduce function: Aggregate counts and generate embedding for each word.
   *
   * @param word The word to process
   * @param counts Iterator of counts from all mappers
   * @param ctx Reducer context for output and configuration
   */
  override def reduce(word: Text, counts: java.lang.Iterable[IntWritable], ctx: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
    val wordStr = word.toString

    // Step 1: Aggregate word frequency from all mappers
    log.debug(s"Processing word: $wordStr")

    // Sum all counts for this word across mappers
    val frequency = counts.asScala.map(_.get).sum
    log.debug(s"  Frequency: $frequency occurrences")

    // Skip words with very low frequency (optional filter)
    val minFreq = ctx.getConfiguration.getInt("vocab.min.frequency", 1)
    if (frequency < minFreq) {
      log.trace(s"  Skipping '$wordStr' (frequency $frequency < $minFreq)")
      ctx.getCounter("wordstats", "filtered_low_frequency").increment(1)
      return
    }

    // Step 2: Generate embedding for the word
    log.debug(s"  Generating embedding for: $wordStr")

    val (embedding, isFailure) = try {
      // Call Ollama to embed the word
      val embeddingModel = ctx.getConfiguration.get("rag.embed.model", "mxbai-embed-large")
      val vec = client.embed(Vector(wordStr), embeddingModel).head

      if (vec.isEmpty) {
        log.warn(s"  Empty embedding returned for word: $wordStr")
        ctx.getCounter("wordstats", "empty_embeddings").increment(1)
        ("" , true) // Empty string as fallback
      } else {
        log.trace(s"  Generated ${vec.length}-dimensional embedding")
        (vec.mkString(","), false)
      }
    } catch {
      case e: Exception =>
        log.error(s"  Failed to generate embedding for '$wordStr': ${e.getMessage}")
        ctx.getCounter("wordstats", "embedding_errors").increment(1)
        ("" , true) // Empty string as fallback
    }

    // Step 3: Format and emit output
    // CSV format: word,frequency,[embedding_vector]
    val csvRow = s"$wordStr,$frequency,[$embedding]"
    ctx.write(word, new Text(csvRow))

    // Update counters for processed words and failures
    ctx.getCounter("wordstats", "processed_words").increment(1)
    if (isFailure) {
      ctx.getCounter("wordstats", "embedding_failures").increment(1)
    }

    // Log progress every 100 words
    val processedWords = ctx.getCounter("wordstats", "processed_words").getValue
    if (processedWords % 100 == 0) {
      val embeddingFailures = ctx.getCounter("wordstats", "embedding_failures").getValue
      log.info(s"Progress: $processedWords words processed, $embeddingFailures failures")
    }

    log.trace(s"  Emitted CSV row for: $wordStr")
  }

  /**
   * Cleanup method called once after all reduce tasks complete.
   * Closes Ollama client and logs final statistics.
   *
   * @param ctx Reducer context
   */
  override def cleanup(ctx: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
    log.info("=" * 80)
    log.info("WordStatsReducer cleanup")
    log.info("=" * 80)

    val processedWords = ctx.getCounter("wordstats", "processed_words").getValue
    val embeddingFailures = ctx.getCounter("wordstats", "embedding_failures").getValue

    log.info(s"Total words processed: $processedWords")
    log.info(s"Embedding failures: $embeddingFailures")

    if (embeddingFailures > 0) {
      val failureRate = (embeddingFailures.toDouble / processedWords * 100)
      log.warn(f"Failure rate: $failureRate%.2f%%")
    }

    log.info("Closing Ollama client...")
    client.close()
    log.info("WordStatsReducer cleanup complete")
    log.info("=" * 80)
  }
}