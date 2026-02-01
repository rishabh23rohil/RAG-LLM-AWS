package rag.indexing

import io.circe.syntax.*
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.slf4j.LoggerFactory
import java.nio.file.Paths
import rag.embedding.{OllamaClient, VectorOps}
import rag.util.PdfExtractor

/**
 * Hadoop mapper that transforms PDF documents into vector embeddings.
 *
 * Workflow:
 * - Reads PDF content from distributed or local storage
 * - Extracts and segments text into overlapping windows
 * - Generates dense vector representations via LLM
 * - Outputs partition-keyed records for index construction
 */
class DocumentVectorizer extends Mapper[LongWritable, Text, IntWritable, Text] {

  private val logger = LoggerFactory.getLogger(classOf[DocumentVectorizer])
  private lazy val embeddingClient = new OllamaClient()

  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    val inputPath = value.toString.trim
    val startTime = System.currentTimeMillis()
    
    logger.info(s"[Vectorizer] Processing: $inputPath")

    val config = context.getConfiguration
    val fileSystem = FileSystem.get(config)
    val hdfsPath = new Path(inputPath)
    val documentKey = hdfsPath.getName

    logger.debug(s"[Vectorizer] Document key: $documentKey, FS: ${fileSystem.getScheme}")

    // Phase 1: Document loading
    val content = try {
      if (inputPath.startsWith("s3://") || inputPath.startsWith("s3a://") || inputPath.startsWith("hdfs://")) {
        logger.debug(s"[Vectorizer] Loading from distributed storage")
        val stream = fileSystem.open(hdfsPath)
        try PdfExtractor.extractText(stream)
        finally stream.close()
      } else {
        logger.debug(s"[Vectorizer] Loading from local filesystem")
        PdfExtractor.extractText(Paths.get(inputPath))
      }
    } catch {
      case ex: Exception =>
        logger.error(s"[Vectorizer] Document load failed: ${ex.getMessage}", ex)
        context.getCounter("vectorizer", "load_failures").increment(1)
        return
    }

    if (content.trim.isEmpty) {
      logger.warn(s"[Vectorizer] Empty content: $documentKey")
      context.getCounter("vectorizer", "empty_documents").increment(1)
      return
    }

    // Phase 2: Text segmentation
    val segments = TextChunker.segment(content, maxLength = 1024, overlap = 128).toVector
    if (segments.isEmpty) {
      logger.warn(s"[Vectorizer] No segments produced: $documentKey")
      context.getCounter("vectorizer", "segmentation_failures").increment(1)
      return
    }
    logger.debug(s"[Vectorizer] Segmented into ${segments.size} parts")

    // Phase 3: Embedding generation
    val modelName = config.get("rag.embed.model", "mxbai-embed-large")
    
    val embeddings = try embeddingClient.generateEmbeddings(segments, modelName)
    catch {
      case ex: Exception =>
        logger.error(s"[Vectorizer] Embedding failed: ${ex.getMessage}", ex)
        context.getCounter("vectorizer", "embedding_failures").increment(1)
        return
    }
    
    // Phase 4: Vector normalization
    val normalizeL2 = config.getBoolean("rag.normalize.l2", true)
    val normalizedVectors = if (normalizeL2) embeddings.map(VectorOps.normalize) else embeddings

    // Phase 5: Partition assignment and output
    val numPartitions = context.getNumReduceTasks
    val partitionId = math.abs(documentKey.hashCode) % numPartitions

    var outputCount = 0
    segments.zip(normalizedVectors).zipWithIndex.foreach {
      case ((segment, vector), segmentIndex) =>
        if (segment.trim.nonEmpty && vector.nonEmpty) {
          val record = Map(
            "document_key" -> documentKey.asJson,
            "segment_index" -> segmentIndex.asJson,
            "content" -> segment.asJson,
            "vector" -> vector.toVector.asJson
          ).asJson.noSpaces

          context.write(new IntWritable(partitionId), new Text(record))
          outputCount += 1
        } else {
          context.getCounter("vectorizer", "skipped_segments").increment(1)
        }
    }

    val duration = System.currentTimeMillis() - startTime
    logger.info(s"[Vectorizer] Completed $documentKey: $outputCount segments -> partition $partitionId (${duration}ms)")
  }

  override def cleanup(context: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    logger.debug("[Vectorizer] Releasing embedding client resources")
    embeddingClient.shutdown()
  }
}
