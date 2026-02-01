package rag

import org.apache.hadoop.io.{FloatWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.slf4j.LoggerFactory
import scala.jdk.CollectionConverters._

/**
 * Reducer for merging search results into global top-k.
 *
 * Receives results sorted by score (negative) and takes top-k.
 *
 * Input: FloatWritable (negative score), Text (JSON result)
 * Output: NullWritable, Text (JSON result)
 */
class SearchReducer extends Reducer[FloatWritable, Text, NullWritable, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  override def reduce(
                       key: FloatWritable,
                       values: java.lang.Iterable[Text],
                       ctx: Reducer[FloatWritable, Text, NullWritable, Text]#Context
                     ): Unit = {

    val k = ctx.getConfiguration.getInt("search.top.k", 5)

    // Emit top-k results (already sorted by mapper output key)
    val emittedCount = values.asScala.take(k).foldLeft(0) { (count, value) =>
      ctx.write(NullWritable.get(), value)
      count + 1
    }

    log.debug(s"Emitted $emittedCount results for score ${-key.get()}")
  }
}