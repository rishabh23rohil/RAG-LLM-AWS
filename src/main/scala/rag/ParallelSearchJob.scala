package rag

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

/**
 * Launches MapReduce job for parallel shard search.
 */
object ParallelSearchJob {

  private val log = LoggerFactory.getLogger(getClass)

  /**
   * Execute parallel search across all shards using MapReduce.
   *
   * @param indexDir Directory containing index shards
   * @param queryVec Query embedding vector
   * @param topK Number of top results to return
   * @return Sequence of search results
   */
  def search(indexDir: String, queryVec: Array[Float], topK: Int = 5): Seq[SearchResults] = {

    log.info("=" * 80)
    log.info("Launching Parallel MapReduce Search")
    log.info("=" * 80)
    log.info(s"Index directory: $indexDir")
    log.info(s"Top-K: $topK")
    log.info(s"Query vector dimension: ${queryVec.length}")

    // ========================================================================
    // Step 1: Find all shard directories
    // ========================================================================
    val indexPath = Paths.get(indexDir)
    if (!Files.exists(indexPath)) {
      log.error(s"Index directory does not exist: $indexDir")
      return Seq.empty
    }

    val shardPaths = Files.list(indexPath)
      .iterator()
      .asScala
      .filter(p => p.getFileName.toString.startsWith("index_shard_"))
      .map(_.toString)
      .toSeq

    if (shardPaths.isEmpty) {
      log.warn("No index shards found")
      return Seq.empty
    }

    log.info(s"Found ${shardPaths.size} shards to search")

    // ========================================================================
    // Step 2: Create temporary input file with shard paths
    // ========================================================================
    val tempInputDir = s"/tmp/search-input-${System.currentTimeMillis()}"
    val tempOutputDir = s"/tmp/search-output-${System.currentTimeMillis()}"

    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    // Write shard paths to input file
    val inputPath = new Path(s"$tempInputDir/shards.txt")
    val writer = new BufferedWriter(new OutputStreamWriter(fs.create(inputPath, true)))
    try {
      shardPaths.foreach { path =>
        writer.write(path)
        writer.newLine()
      }
    } finally {
      writer.close()
    }

    log.info(s"Created input file: $inputPath")

    // ========================================================================
    // Step 3: Configure and launch MapReduce job
    // ========================================================================

    // Set query parameters in configuration
    conf.set("search.query.vector", queryVec.mkString(","))
    conf.setInt("search.top.k", topK)
    conf.setInt("mapreduce.input.lineinputformat.linespermap", 1) // One mapper per shard

    val job = Job.getInstance(conf, "Parallel Shard Search")
    job.setJarByClass(this.getClass)

    // Set mapper and reducer
    job.setMapperClass(classOf[SearchMapper])
    job.setReducerClass(classOf[SearchReducer])

    // Set output types
    job.setMapOutputKeyClass(classOf[FloatWritable])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[NullWritable])
    job.setOutputValueClass(classOf[Text])

    // Set input/output format
    job.setInputFormatClass(classOf[NLineInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[NullWritable, Text]])

    // Set input/output paths
    FileInputFormat.addInputPath(job, new Path(tempInputDir))
    FileOutputFormat.setOutputPath(job, new Path(tempOutputDir))

    // Use single reducer for global top-k merge
    job.setNumReduceTasks(1)

    log.info("Submitting MapReduce job...")
    val startTime = System.currentTimeMillis()

    // ========================================================================
    // Step 4: Wait for job completion
    // ========================================================================
    val success = job.waitForCompletion(true)
    val duration = (System.currentTimeMillis() - startTime) / 1000

    if (!success) {
      log.error(s"MapReduce search job failed after ${duration}s")
      cleanupTempDirs(fs, tempInputDir, tempOutputDir)
      return Seq.empty
    }

    log.info(s"MapReduce job completed successfully in ${duration}s")

    // ========================================================================
    // Step 5: Read results from output
    // ========================================================================
    val results = try {
      parseResults(fs, tempOutputDir)
    } catch {
      case e: Exception =>
        log.error(s"Failed to parse results: ${e.getMessage}", e)
        Seq.empty
    }

    // ========================================================================
    // Step 6: Cleanup temporary directories
    // ========================================================================
    cleanupTempDirs(fs, tempInputDir, tempOutputDir)

    log.info("=" * 80)
    log.info(s"Parallel search complete: ${results.size} results")
    log.info("=" * 80)

    results
  }

  /**
   * Parse search results from MapReduce output.
   */
  private def parseResults(fs: FileSystem, outputDir: String): Seq[SearchResults] = {
    import io.circe.parser._

    val outputPath = new Path(outputDir)
    val files = fs.listStatus(outputPath)
      .filter(_.getPath.getName.startsWith("part-"))

    files.flatMap { fileStatus =>
      val reader = new java.io.BufferedReader(
        new java.io.InputStreamReader(fs.open(fileStatus.getPath))
      )

      try {
        Iterator.continually(reader.readLine())
          .takeWhile(_ != null)
          .flatMap { line =>
            parse(line).toOption.flatMap { json =>
              val cursor = json.hcursor
              for {
                docId <- cursor.get[String]("doc_id").toOption
                chunkId <- cursor.get[Int]("chunk_id").toOption
                text <- cursor.get[String]("text").toOption
                score <- cursor.get[Float]("score").toOption
              } yield SearchResults(docId, chunkId, text, score)
            }
          }.toSeq
      } finally {
        reader.close()
      }
    }.toSeq
  }

  /**
   * Cleanup temporary directories.
   */
  private def cleanupTempDirs(fs: FileSystem, dirs: String*): Unit = {
    dirs.foreach { dir =>
      try {
        fs.delete(new Path(dir), true)
        log.debug(s"Deleted temp directory: $dir")
      } catch {
        case e: Exception =>
          log.warn(s"Failed to delete temp directory $dir: ${e.getMessage}")
      }
    }
  }
}