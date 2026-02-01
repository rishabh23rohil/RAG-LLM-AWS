package rag.search

import org.apache.hadoop.io.{FloatWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.slf4j.LoggerFactory
import scala.jdk.CollectionConverters._

/**
 * Reduces search results into global top-k.
 *
 * Receives results pre-sorted by score (negative) from mapper output.
 * Emits only the top-k globally ranked results.
 */
class ResultMerger extends Reducer[FloatWritable, Text, NullWritable, Text] {

  private val logger = LoggerFactory.getLogger(classOf[ResultMerger])

  override def reduce(
    scoreKey: FloatWritable,
    results: java.lang.Iterable[Text],
    context: Reducer[FloatWritable, Text, NullWritable, Text]#Context
  ): Unit = {
    val limit = context.getConfiguration.getInt("query.limit", 5)
    val actualScore = -scoreKey.get()

    var emitCount = 0
    results.asScala.take(limit).foreach { result =>
      context.write(NullWritable.get(), result)
      emitCount += 1
    }

    logger.trace(s"[Merger] Score $actualScore: emitted $emitCount results")
  }
}
