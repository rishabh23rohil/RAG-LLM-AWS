package rag

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, NLineInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.slf4j.LoggerFactory

/**
 * Entry point for the distributed RAG pipeline.
 *
 * Execution modes:
 * - rag: Constructs sharded Lucene vector indexes from PDF documents
 * - wordstats: Computes vocabulary frequencies and generates word embeddings
 * - postprocess: Performs semantic analysis on embeddings (similarity, analogies)
 */
object Driver {

  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    // Validate minimum required arguments
    if (args.length < 4) {
      System.err.println(
        """Usage: Driver <className> <mode:rag|wordstats|postprocess> <input> <output> [model] [similarity] [linesPerMap] [timeoutMs] [numReducers]
          |
          |Defaults:
          |  model        = mxbai-embed-large
          |  similarity   = COSINE
          |  linesPerMap  = 50
          |  timeoutMs    = 3600000
          |  numReducers  = 8
          |""".stripMargin)
      System.exit(1)
    }

    // Parse command-line arguments
    val mode         = args(1).toLowerCase
    val inputPath    = args(2)
    val outputPath   = args(3)
    val model        = if (args.length >= 5) args(4) else "mxbai-embed-large"
    val similarity   = if (args.length >= 6) args(5) else "COSINE"
    val linesPerMap  = if (args.length >= 7) args(6).toInt else 50
    val timeoutMs    = if (args.length >= 8) args(7).toLong else 3600000L
    val numReducers  = if (args.length >= 9) args(8).toInt else 8

    log.info(s"Launching pipeline in $mode mode")
    log.info(s"Input path: $inputPath")
    log.info(s"Output path: $outputPath")

    mode match {
      case "rag" | "wordstats" =>
        // Create Hadoop configuration
        val conf = new Configuration()

        // Set embedding model configuration
        conf.set("rag.embed.model", model)
        log.info(s"Embedding model: $model")

        // Set similarity function configuration
        conf.set("rag.similarity", similarity)
        log.info(s"Similarity function: $similarity")

        // Enable L2 normalization for COSINE and DOT_PRODUCT
        conf.setBoolean("rag.normalize.l2", similarity.equalsIgnoreCase("COSINE") || similarity.equalsIgnoreCase("DOT_PRODUCT"))

        // Configure MapReduce performance parameters
        conf.setInt("mapreduce.input.lineinputformat.linespermap", linesPerMap)
        log.info(s"Lines per mapper: $linesPerMap")

        conf.setLong("mapreduce.task.timeout", timeoutMs)
        log.info(s"Task timeout: ${timeoutMs}ms")

        log.info(s"Starting MapReduce job [mode=$mode, linesPerMap=$linesPerMap, timeoutMs=$timeoutMs, numReducers=$numReducers]")

        // Initialize MapReduce job
        val job = Job.getInstance(conf, s"RAG-Pipeline-$mode")
        job.setJarByClass(this.getClass)

        // Configure mapper and reducer classes based on mode
        if (mode == "rag") {
          log.info("Configuring RAG mode: RagMapper + ShardReducer")
          job.setMapperClass(classOf[RagMapper])
          job.setReducerClass(classOf[RagShardReducer])
          job.setMapOutputKeyClass(classOf[IntWritable])
          job.setMapOutputValueClass(classOf[Text])
          job.setInputFormatClass(classOf[NLineInputFormat])
        } else {
          log.info("Configuring wordstats mode: WordStatsMapper + WordStatsReducer")
          job.setMapperClass(classOf[WordStatsMapper])
          job.setReducerClass(classOf[WordStatsReducer])
          job.setMapOutputKeyClass(classOf[Text])
          job.setMapOutputValueClass(classOf[IntWritable])
          job.setInputFormatClass(classOf[NLineInputFormat])
        }

        // Set output key/value types
        job.setOutputKeyClass(classOf[Text])
        job.setOutputValueClass(classOf[Text])
        job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(inputPath))
        FileOutputFormat.setOutputPath(job, new Path(outputPath))

        // Set number of reducer tasks
        job.setNumReduceTasks(numReducers)
        log.info(s"Number of reducers: $numReducers")

        // Submit job and wait for completion
        log.info("Submitting job to cluster...")
        val success = job.waitForCompletion(true)

        // Log job completion status
        if (success) log.info(s"$mode job completed successfully.")
        else log.error(s"$mode job failed.")

        System.exit(if (success) 0 else 1)

      case "postprocess" =>
        // Post-processing mode: analyze word embeddings
        log.info("Running post-embedding analysis...")
        PostProcessEmbeddings.run(inputPath)
        log.info("Post-processing completed successfully.")
        System.exit(0)

      case _ =>
        // Invalid mode specified
        log.error(s"Invalid mode: $mode")
        System.err.println("Invalid mode. Must be 'rag', 'wordstats', or 'postprocess'")
        System.exit(1)
    }
  }
}