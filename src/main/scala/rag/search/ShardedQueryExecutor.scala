package rag.search

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{FloatWritable, LongWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.{NLineInputFormat, FileInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.slf4j.LoggerFactory
import java.io.{BufferedWriter, OutputStreamWriter}
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._
import io.circe.parser.decode
import io.circe.generic.auto._

/**
 * Distributed query execution across index partitions via MapReduce.
 *
 * Launches a MapReduce job where:
 * - Each mapper searches one partition
 * - Reducer merges results into global top-k
 */
object ShardedQueryExecutor {

  private val logger = LoggerFactory.getLogger(classOf[ShardedQueryExecutor.type])

  /**
   * Executes distributed search across all partitions.
   */
  def executeQuery(indexPath: String, queryVector: Array[Float], resultLimit: Int = 5): Seq[QueryResult] = {
    logger.info(s"[Executor] Starting distributed query")
    logger.info(s"[Executor] Index: $indexPath, limit: $resultLimit, dimensions: ${queryVector.length}")

    val indexDir = Paths.get(indexPath)
    if (!Files.exists(indexDir)) {
      logger.error(s"[Executor] Index directory not found: $indexPath")
      return Seq.empty
    }

    val partitionPaths = Files.list(indexDir)
      .iterator()
      .asScala
      .filter(p => p.getFileName.toString.startsWith("partition_"))
      .toSeq

    if (partitionPaths.isEmpty) {
      logger.warn(s"[Executor] No partitions found in $indexPath")
      return Seq.empty
    }

    logger.info(s"[Executor] Found ${partitionPaths.size} partitions")

    // Create temporary input file listing partitions
    val hadoopConf = new Configuration()
    val tempInputPath = new Path(s"/tmp/query_input_${System.currentTimeMillis()}")
    val tempOutputPath = new Path(s"/tmp/query_output_${System.currentTimeMillis()}")

    val fs = tempInputPath.getFileSystem(hadoopConf)

    try {
      // Write partition paths to temp file
      val writer = new BufferedWriter(new OutputStreamWriter(fs.create(tempInputPath)))
      partitionPaths.foreach(p => writer.write(p.toAbsolutePath.toString + "\n"))
      writer.close()

      // Configure and run job
      hadoopConf.set("query.vector", queryVector.mkString(","))
      hadoopConf.setInt("query.limit", resultLimit)

      val job = Job.getInstance(hadoopConf, "Distributed-Query")
      job.setJarByClass(this.getClass)

      job.setMapperClass(classOf[ShardQueryMapper])
      job.setReducerClass(classOf[ResultMerger])

      job.setMapOutputKeyClass(classOf[FloatWritable])
      job.setMapOutputValueClass(classOf[Text])
      job.setOutputKeyClass(classOf[NullWritable])
      job.setOutputValueClass(classOf[Text])

      job.setInputFormatClass(classOf[NLineInputFormat])
      job.setOutputFormatClass(classOf[TextOutputFormat[NullWritable, Text]])
      job.setNumReduceTasks(1)

      FileInputFormat.addInputPath(job, tempInputPath)
      FileOutputFormat.setOutputPath(job, tempOutputPath)

      val success = job.waitForCompletion(true)

      if (!success) {
        logger.error("[Executor] Query job failed")
        return Seq.empty
      }

      // Parse results
      val resultFiles = fs.listStatus(tempOutputPath)
        .filter(_.getPath.getName.startsWith("part-"))

      val results = resultFiles.flatMap { status =>
        val reader = new java.io.BufferedReader(
          new java.io.InputStreamReader(fs.open(status.getPath))
        )
        try {
          Iterator.continually(reader.readLine())
            .takeWhile(_ != null)
            .flatMap(line => decode[QueryResult](line).toOption)
            .toSeq
        } finally {
          reader.close()
        }
      }

      logger.info(s"[Executor] Retrieved ${results.length} results")
      results.sortBy(-_.relevanceScore).take(resultLimit).toSeq

    } finally {
      // Cleanup
      fs.delete(tempInputPath, false)
      fs.delete(tempOutputPath, true)
    }
  }
}
