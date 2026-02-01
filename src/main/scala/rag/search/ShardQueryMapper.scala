package rag.search

import org.apache.hadoop.io.{FloatWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.search.{IndexSearcher, KnnFloatVectorQuery}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory
import io.circe.syntax._
import io.circe.generic.auto._
import java.nio.file.Paths

/**
 * Mapper for distributed partition search.
 *
 * Each mapper:
 * - Opens one index partition
 * - Executes KNN query
 * - Emits results keyed by negative score (for descending sort)
 */
class ShardQueryMapper extends Mapper[LongWritable, Text, FloatWritable, Text] {

  private val logger = LoggerFactory.getLogger(classOf[ShardQueryMapper])

  override def map(
    key: LongWritable,
    value: Text,
    context: Mapper[LongWritable, Text, FloatWritable, Text]#Context
  ): Unit = {
    val partitionPath = value.toString.trim
    logger.debug(s"[QueryMapper] Searching partition: $partitionPath")

    val vectorStr = context.getConfiguration.get("query.vector")
    val limit = context.getConfiguration.getInt("query.limit", 5)

    if (vectorStr == null || vectorStr.isEmpty) {
      logger.error("[QueryMapper] No query vector in configuration")
      context.getCounter("query_mapper", "missing_vector").increment(1)
      return
    }

    val queryVector = try {
      vectorStr.split(",").map(_.toFloat)
    } catch {
      case ex: Exception =>
        logger.error(s"[QueryMapper] Vector parse error: ${ex.getMessage}")
        context.getCounter("query_mapper", "vector_parse_error").increment(1)
        return
    }

    try {
      val directory = FSDirectory.open(Paths.get(partitionPath))
      val reader = DirectoryReader.open(directory)
      val searcher = new IndexSearcher(reader)

      val knnQuery = new KnnFloatVectorQuery("vector", queryVector, limit)
      val hits = searcher.search(knnQuery, limit)
      val fieldReader = searcher.storedFields()

      hits.scoreDocs.foreach { hit =>
        val doc = fieldReader.document(hit.doc)
        val result = QueryResult(
          documentKey = doc.get("document_key"),
          segmentIndex = doc.get("segment_index").toInt,
          content = doc.get("content"),
          relevanceScore = hit.score
        )

        // Use negative score as key for descending sort
        context.write(new FloatWritable(-hit.score), new Text(result.asJson.noSpaces))
      }

      context.getCounter("query_mapper", "partitions_searched").increment(1)
      context.getCounter("query_mapper", "results_emitted").increment(hits.scoreDocs.length)

      reader.close()
      directory.close()

      logger.debug(s"[QueryMapper] Partition $partitionPath: ${hits.scoreDocs.length} hits")

    } catch {
      case ex: Exception =>
        logger.error(s"[QueryMapper] Search error for $partitionPath: ${ex.getMessage}", ex)
        context.getCounter("query_mapper", "search_errors").increment(1)
    }
  }
}
