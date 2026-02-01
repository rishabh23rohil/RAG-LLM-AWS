package rag.analytics

import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.LoggerFactory
import java.nio.file.Paths
import rag.util.PdfExtractor

/**
 * Mapper for extracting and counting token occurrences from documents.
 *
 * Pipeline:
 * - Loads document from distributed or local storage
 * - Extracts text via PDF parser
 * - Tokenizes and filters content
 * - Emits (token, 1) pairs for aggregation
 */
class TokenFrequencyMapper extends Mapper[LongWritable, Text, Text, IntWritable] {

  private val logger = LoggerFactory.getLogger(classOf[TokenFrequencyMapper])

  private val commonStopwords = Set(
    "the", "is", "and", "a", "an", "in", "on", "of", "for", "to",
    "this", "that", "it", "with", "as", "by", "at", "from", "or", "be",
    "are", "was", "were", "has", "had", "have", "not", "but", "can", "could",
    "do", "does", "did", "will", "would", "should", "may", "might", "must"
  )

  private val unitCount = new IntWritable(1)

  override def map(
    key: LongWritable,
    value: Text,
    context: Mapper[LongWritable, Text, Text, IntWritable]#Context
  ): Unit = {
    val documentPath = value.toString.trim
    
    if (documentPath.isEmpty) {
      context.getCounter("token_mapper", "empty_inputs").increment(1)
      return
    }

    val startTime = System.currentTimeMillis()
    logger.info(s"[TokenMapper] Processing: $documentPath")

    val config = context.getConfiguration
    val hdfsPath = new Path(documentPath)
    val documentId = hdfsPath.getName

    // Load document content
    val contentResult: Either[String, String] = try {
      if (documentPath.startsWith("s3://") || documentPath.startsWith("s3a://") || documentPath.startsWith("hdfs://")) {
        val fs = FileSystem.get(hdfsPath.toUri, config)
        if (!fs.exists(hdfsPath)) {
          Left(s"Document not found: $documentPath")
        } else {
          logger.debug(s"[TokenMapper] Loading from ${fs.getScheme}")
          val stream = fs.open(hdfsPath)
          try Right(PdfExtractor.extractText(stream))
          finally stream.close()
        }
      } else {
        logger.debug("[TokenMapper] Loading from local filesystem")
        Right(PdfExtractor.extractText(Paths.get(documentPath)))
      }
    } catch {
      case ex: Exception =>
        logger.error(s"[TokenMapper] Load failed: ${ex.getMessage}", ex)
        Left(ex.getMessage)
    }

    val textContent = contentResult match {
      case Right(text) if text.trim.nonEmpty => text
      case Right(_) =>
        logger.warn(s"[TokenMapper] Empty content: $documentId")
        context.getCounter("token_mapper", "empty_documents").increment(1)
        return
      case Left(error) =>
        logger.error(s"[TokenMapper] Error for $documentId: $error")
        context.getCounter("token_mapper", "load_errors").increment(1)
        return
    }

    // Tokenize and normalize
    val normalizedText = textContent.toLowerCase.replaceAll("[^a-z\\s]", " ")
    
    val tokens = normalizedText
      .split("\\s+")
      .filter(token => token.length >= 2 && !commonStopwords.contains(token))

    logger.debug(s"[TokenMapper] Extracted ${tokens.length} tokens from $documentId")

    // Emit token counts
    val tokenText = new Text()
    tokens.foreach { token =>
      tokenText.set(token)
      context.write(tokenText, unitCount)
    }

    context.getCounter("token_mapper", "documents_processed").increment(1)
    context.getCounter("token_mapper", "tokens_emitted").increment(tokens.length)

    val duration = System.currentTimeMillis() - startTime
    logger.info(s"[TokenMapper] Completed $documentId: ${tokens.length} tokens in ${duration}ms")
  }
}
