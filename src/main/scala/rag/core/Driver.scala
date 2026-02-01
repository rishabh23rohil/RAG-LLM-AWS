package rag.core

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, NLineInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.slf4j.LoggerFactory
import rag.indexing.{DocumentVectorizer, IndexPartitionBuilder}
import rag.analytics.{TokenFrequencyMapper, EmbeddingAggregator, SemanticAnalyzer}

/**
 * Main entry point for RAG pipeline execution.
 *
 * Supported modes:
 * - index: Build vector index from document corpus
 * - vocabulary: Extract and embed vocabulary terms
 * - analyze: Run semantic analysis on embeddings
 */
object Driver {

  private val logger = LoggerFactory.getLogger(classOf[Driver.type])

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      printUsage()
      System.exit(1)
    }

    val executionMode = args(1).toLowerCase
    val inputPath = args(2)
    val outputPath = args(3)
    val embeddingModel = if (args.length >= 5) args(4) else Config.embeddingModel
    val similarityFn = if (args.length >= 6) args(5) else Config.similarityFunction
    val docsPerMapper = if (args.length >= 7) args(6).toInt else 50
    val taskTimeout = if (args.length >= 8) args(7).toLong else 3600000L
    val partitionCount = if (args.length >= 9) args(8).toInt else Config.Index.defaultPartitions

    logger.info(s"[Driver] Mode: $executionMode")
    logger.info(s"[Driver] Input: $inputPath")
    logger.info(s"[Driver] Output: $outputPath")

    executionMode match {
      case "index" | "vocabulary" =>
        runMapReduceJob(executionMode, inputPath, outputPath, embeddingModel, similarityFn, docsPerMapper, taskTimeout, partitionCount)
      
      case "analyze" =>
        logger.info("[Driver] Running semantic analysis...")
        SemanticAnalyzer.analyze(inputPath)
        logger.info("[Driver] Analysis complete")
        System.exit(0)
      
      case _ =>
        logger.error(s"[Driver] Unknown mode: $executionMode")
        printUsage()
        System.exit(1)
    }
  }

  private def runMapReduceJob(
    mode: String,
    inputPath: String,
    outputPath: String,
    embeddingModel: String,
    similarity: String,
    docsPerMapper: Int,
    timeout: Long,
    partitions: Int
  ): Unit = {
    val hadoopConf = new Configuration()

    // Configure embedding settings
    hadoopConf.set("rag.embed.model", embeddingModel)
    hadoopConf.set("rag.similarity", similarity)
    hadoopConf.setBoolean("rag.normalize.l2", 
      similarity.equalsIgnoreCase("COSINE") || similarity.equalsIgnoreCase("DOT_PRODUCT"))

    // Configure MapReduce settings
    hadoopConf.setInt("mapreduce.input.lineinputformat.linespermap", docsPerMapper)
    hadoopConf.setLong("mapreduce.task.timeout", timeout)

    logger.info(s"[Driver] Embedding model: $embeddingModel")
    logger.info(s"[Driver] Similarity: $similarity")
    logger.info(s"[Driver] Documents per mapper: $docsPerMapper")
    logger.info(s"[Driver] Partitions: $partitions")

    val job = Job.getInstance(hadoopConf, s"RAG-$mode")
    job.setJarByClass(this.getClass)

    mode match {
      case "index" =>
        logger.info("[Driver] Configuring index pipeline")
        job.setMapperClass(classOf[DocumentVectorizer])
        job.setReducerClass(classOf[IndexPartitionBuilder])
        job.setMapOutputKeyClass(classOf[IntWritable])
        job.setMapOutputValueClass(classOf[Text])

      case "vocabulary" =>
        logger.info("[Driver] Configuring vocabulary pipeline")
        job.setMapperClass(classOf[TokenFrequencyMapper])
        job.setReducerClass(classOf[EmbeddingAggregator])
        job.setMapOutputKeyClass(classOf[Text])
        job.setMapOutputValueClass(classOf[IntWritable])
    }

    job.setInputFormatClass(classOf[NLineInputFormat])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])

    FileInputFormat.addInputPath(job, new Path(inputPath))
    FileOutputFormat.setOutputPath(job, new Path(outputPath))
    job.setNumReduceTasks(partitions)

    logger.info("[Driver] Submitting job...")
    val success = job.waitForCompletion(true)

    if (success) {
      logger.info(s"[Driver] Job completed successfully")
    } else {
      logger.error(s"[Driver] Job failed")
    }

    System.exit(if (success) 0 else 1)
  }

  private def printUsage(): Unit = {
    System.err.println(
      """
        |Usage: Driver <class> <mode> <input> <output> [options]
        |
        |Modes:
        |  index      - Build vector index from documents
        |  vocabulary - Extract vocabulary with embeddings
        |  analyze    - Run semantic analysis on embeddings
        |
        |Options:
        |  model        - Embedding model (default: mxbai-embed-large)
        |  similarity   - Similarity function (default: COSINE)
        |  docsPerMap   - Documents per mapper (default: 50)
        |  timeout      - Task timeout in ms (default: 3600000)
        |  partitions   - Number of partitions (default: 8)
        |""".stripMargin)
  }
}
