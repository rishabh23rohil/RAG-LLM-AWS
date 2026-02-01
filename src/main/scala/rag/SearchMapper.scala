package rag

import org.apache.hadoop.io.{FloatWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.search.{IndexSearcher, KnnFloatVectorQuery}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory
import io.circe.syntax._
import java.nio.file.Paths

/**
 * Mapper for parallel shard search at query time.
 *
 * Each mapper searches one shard and emits top-k results.
 *
 * Input: LongWritable (shard number), Text (shard path)
 * Output: FloatWritable (negative score for sorting), Text (JSON result)
 */
class SearchMapper extends Mapper[LongWritable, Text, FloatWritable, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  override def map(
                    key: LongWritable,
                    value: Text,
                    ctx: Mapper[LongWritable, Text, FloatWritable, Text]#Context
                  ): Unit = {

    val shardPath = value.toString.trim
    log.info(s"=" * 80)
    log.info(s"SearchMapper processing shard: $shardPath")

    // Get query vector and k from configuration
    val queryVecStr = ctx.getConfiguration.get("search.query.vector")
    val k = ctx.getConfiguration.getInt("search.top.k", 5)

    if (queryVecStr == null || queryVecStr.isEmpty) {
      log.error("No query vector found in configuration")
      ctx.getCounter("search", "missing_query_vector").increment(1)
      return
    }

    // Parse query vector
    val queryVec = try {
      queryVecStr.split(",").map(_.toFloat)
    } catch {
      case e: Exception =>
        log.error(s"Failed to parse query vector: ${e.getMessage}")
        ctx.getCounter("search", "parse_vector_error").increment(1)
        return
    }

    log.info(s"Searching shard for top-$k results")
    log.debug(s"Query vector dimension: ${queryVec.length}")

    // Search the shard
    try {
      val path = Paths.get(shardPath)
      val directory = FSDirectory.open(path)
      val reader = DirectoryReader.open(directory)
      val searcher = new IndexSearcher(reader)

      // Create KNN query
      val query = new KnnFloatVectorQuery("vec", queryVec, k)
      val topDocs = searcher.search(query, k)
      val storedFields = searcher.storedFields()

      log.info(s"Found ${topDocs.scoreDocs.length} results in this shard")

      // Emit each result
      topDocs.scoreDocs.foreach { scoreDoc =>
        val doc = storedFields.document(scoreDoc.doc)
        val docId = doc.get("doc_id")
        val chunkId = doc.get("chunk_id").toInt
        val text = doc.get("text")
        val score = scoreDoc.score

        // Create JSON result
        val result = Map(
          "doc_id" -> docId.asJson,
          "chunk_id" -> chunkId.asJson,
          "text" -> text.asJson,
          "score" -> score.asJson,
          "shard" -> shardPath.asJson
        ).asJson.noSpaces

        // Emit with negative score as key (for sorting in descending order)
        ctx.write(new FloatWritable(-score), new Text(result))
        log.trace(s"Emitted: $docId chunk $chunkId (score: $score)")
      }

      reader.close()
      directory.close()

      ctx.getCounter("search", "shards_searched").increment(1)
      log.info(s"Completed searching shard: $shardPath")

    } catch {
      case e: Exception =>
        log.error(s"Failed to search shard $shardPath: ${e.getMessage}", e)
        ctx.getCounter("search", "shard_search_failures").increment(1)
    }

    log.info("=" * 80)
  }
}